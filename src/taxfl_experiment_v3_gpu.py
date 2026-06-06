"""
TaxFL: Federated Learning and Federated Graph Intelligence for Cross-Border Tax Compliance
Full Experiment Suite — Version 3.0 (GPU-accelerated)

Includes:
  (A) Scale experiment: 3, 5, 10 jurisdictions          (Appendix B.7.1)
  (B) Class imbalance / loss function comparison         (Appendix B.7.2)
  (C) Realistic scenario: only beneficiary labeled       (Appendix B.7.3)
  (D) Tier 2 fix                                         (Appendix B.7.4)
  (E) Large-scale vertical+horizontal (10 jur, 150k)    (Appendix B.7.5)
  (F) INTRA-COUNTRY 5-silo experiment (Section 6)
  (G) *** COMPLETE GraphSAGE vs LogReg ablation          NEW ***
      — all 6 fraud typologies, scheme-level recall

Run modes:
  python taxfl_experiment_v3_gpu.py scale
  python taxfl_experiment_v3_gpu.py imbalance
  python taxfl_experiment_v3_gpu.py realistic
  python taxfl_experiment_v3_gpu.py intra
  python taxfl_experiment_v3_gpu.py ablation   ← NEW: complete typology ablation
  python taxfl_experiment_v3_gpu.py all

Frantz, P.A. (2026). TaxFL v9.1. DOI: 10.5281/zenodo.18602470
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import copy, random, sys, warnings
warnings.filterwarnings('ignore')

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from taxfl_experiment import (
    set_seed, to_pyg, GraphSAGE,
    local_train, fedavg, evaluate, class_weight_tensor
)
from taxfl_experiment_v2 import fedprox_train, trimmed_mean, krum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# SECTION A: SCALE EXPERIMENT DATA GENERATOR
# ============================================================

def make_scale_data(n_jur=3, n_wallets_per=400, n_entities_per=800,
                    n_evasion=60, seed=42):
    """Generate data for scale experiment (3/5/10 jurisdictions)."""
    rng = np.random.RandomState(seed)
    datasets = []

    for jid in range(n_jur):
        ne, nw = n_entities_per, n_wallets_per
        n = ne + nw
        feats = np.zeros((n, 10), dtype=np.float32)

        feats[:ne, 0] = rng.lognormal(10, 1.5, ne)
        feats[:ne, 1] = rng.uniform(0.1, 0.35, ne)
        feats[:ne, 2] = rng.poisson(12, ne).astype(float)
        feats[:ne, 3] = rng.exponential(0.3, ne)
        feats[ne:, 4] = rng.lognormal(8, 2, nw)
        feats[ne:, 5] = rng.poisson(50, nw).astype(float)
        feats[ne:, 6] = rng.lognormal(6, 1, nw)
        feats[ne:, 7] = rng.uniform(0, 1, nw)
        feats[ne:, 8] = rng.exponential(0.1, nw)
        feats[ne:, 9] = rng.poisson(2, nw).astype(float)

        feats /= (feats.std(axis=0) + 1e-8)
        labels = np.zeros(n, dtype=np.int64)

        src, dst = [], []
        for _ in range(2000):
            e = rng.randint(0, ne)
            w = ne + rng.randint(0, nw)
            src += [e, w]; dst += [w, e]

        # Inject evasion nodes
        chains_this_jur = n_evasion // n_jur
        for _ in range(chains_this_jur):
            idx_e = rng.randint(0, ne)
            idx_w = ne + rng.randint(0, nw)
            feats[idx_e, 0] *= 0.2
            feats[idx_w, 4] *= 8
            feats[idx_w, 8] += 0.7
            labels[idx_e] = 1
            labels[idx_w] = 1
            src += [idx_e, idx_w]; dst += [idx_w, idx_e]

        datasets.append(to_pyg(feats, labels, src, dst, seed + jid, DEVICE))

    return datasets


def fed_run_scale(datasets, n_rounds=40, local_epochs=10,
                  method='fedavg', n_layers=2, hidden=64, seed=42,
                  tier2_indices=None):
    """Run federation over list of datasets. Returns (fed_auc, fed_f1, fed_auprc) for last jur."""
    set_seed(seed)
    in_ch = datasets[0].x.shape[1]
    n = len(datasets)

    models = [GraphSAGE(in_ch, hidden, 2, n_layers).to(DEVICE) for _ in range(n)]
    init_w = models[0].get_weights()
    for m in models: m.set_weights(init_w)
    global_w = init_w

    tier2 = set(tier2_indices or [])

    for _ in range(n_rounds):
        local_ws, sizes = [], []
        for i, (m, data) in enumerate(zip(models, datasets)):
            m.set_weights(global_w)
            if i not in tier2:
                if method == 'fedprox':
                    w = fedprox_train(m, data, global_w, epochs=local_epochs)
                else:
                    w = local_train(m, data, local_epochs)
                local_ws.append(w)
                sizes.append(len(data.x))

        if local_ws:
            global_w = fedavg(local_ws, sizes)

    models[-1].set_weights(global_w)
    return evaluate(models[-1], datasets[-1])


# ============================================================
# SECTION F: INTRA-COUNTRY 5-SILO SYNTHETIC DATA
# ============================================================

# ---- Fraud typology definitions ----
# Each typology: list of (node_type, silo, feature_mutations, edges_within_silo)

SILO_NAMES   = ['RFB', 'Bancos', 'COAF', 'MP', 'PF']
TYPOLOGY_NAMES = ['T1-Smurfing', 'T2-NF_Fria', 'T3-Cripto',
                  'T4-CNAE_Falso', 'T5-Money_Mule', 'T6-Caixa_Dois']

# Target silo sizes (from paper)
SILO_SIZES = {'RFB': 19470, 'Bancos': 19031, 'COAF': 10, 'MP': 22, 'PF': 37}

# Minimum viable silo for injection
SILO_MIN   = {'RFB': 500, 'Bancos': 500, 'COAF': 10, 'MP': 22, 'PF': 37}


def make_intra_country_data(seed=42, n_rfb=500, n_bancos=500,
                             n_coaf=10, n_mp=22, n_pf=37,
                             target_fraud_rate=0.06):
    """
    Build 5-silo intra-country graph with 6 fraud typologies.

    Returns:
        silo_data: dict silo_name -> PyG Data
        scheme_nodes: dict typology_name -> list of fraud node indices per silo
        silo_overlap: shared node indices (entities appearing in multiple silos)
    """
    rng = np.random.RandomState(seed)
    sizes = {'RFB': n_rfb, 'Bancos': n_bancos, 'COAF': n_coaf,
             'MP': n_mp, 'PF': n_pf}

    # We build a shared entity pool — PJ nodes visible to both RFB and Bancos
    n_pj = min(n_rfb, n_bancos)   # shared companies
    n_pf_nodes = 200               # individual persons (shared)

    # ---- Build per-silo node features ----
    silo_feats, silo_labels, silo_src, silo_dst = {}, {}, {}, {}
    silo_node_counts = {}

    # RFB: tax features — NF-e, e-Financeira, DIRPF, ownership
    #   Features: declared_revenue, nfe_volume, cfop_code, cnae_declared,
    #             efin_income, ownership_degree, dirpf_assets, dirpf_income,
    #             filing_consistency, crypto_declared
    n = sizes['RFB']
    f = np.zeros((n, 10), dtype=np.float32)
    f[:, 0] = rng.lognormal(11, 1.5, n)   # declared_revenue
    f[:, 1] = rng.poisson(100, n).astype(float)  # nfe_volume
    f[:, 2] = rng.randint(0, 20, n).astype(float) # cfop_code (sector proxy)
    f[:, 3] = rng.randint(0, 10, n).astype(float) # cnae_declared
    f[:, 4] = rng.lognormal(10, 1.2, n)   # e-Financeira income
    f[:, 5] = rng.poisson(3, n).astype(float)     # ownership_degree
    f[:, 6] = rng.lognormal(12, 2, n)     # DIRPF assets
    f[:, 7] = rng.lognormal(10, 1, n)     # DIRPF income
    f[:, 8] = rng.uniform(0, 1, n)        # filing_consistency
    f[:, 9] = rng.exponential(0.1, n)     # crypto_declared
    f /= (f.std(0) + 1e-8)
    silo_feats['RFB'] = f
    silo_labels['RFB'] = np.zeros(n, dtype=np.int64)
    silo_src['RFB'], silo_dst['RFB'] = [], []
    silo_node_counts['RFB'] = n

    # Add legitimate RFB ownership edges (PJ→PJ)
    for _ in range(n * 2):
        a, b = rng.randint(0, n, 2)
        if a != b:
            silo_src['RFB'] += [a, b]; silo_dst['RFB'] += [b, a]

    # Bancos: transaction features — PIX/TED flows, KYC, account behavior
    #   Features: avg_pix_value, pix_frequency, pix_std, ted_volume,
    #             n_counterparts, avg_inflow, avg_outflow, kyc_score,
    #             avg_tx_size_cv, monthly_balance_var
    n = sizes['Bancos']
    f = np.zeros((n, 10), dtype=np.float32)
    f[:, 0] = rng.lognormal(7, 1.5, n)    # avg_pix_value
    f[:, 1] = rng.poisson(40, n).astype(float)    # pix_frequency
    f[:, 2] = rng.exponential(0.5, n)     # pix_std
    f[:, 3] = rng.lognormal(9, 2, n)      # ted_volume
    f[:, 4] = rng.poisson(15, n).astype(float)    # n_counterparts
    f[:, 5] = rng.lognormal(8, 1.5, n)    # avg_inflow
    f[:, 6] = rng.lognormal(8, 1.5, n)    # avg_outflow
    f[:, 7] = rng.uniform(0.5, 1, n)      # kyc_score (high for legit)
    f[:, 8] = rng.exponential(0.3, n)     # avg_tx_size_cv
    f[:, 9] = rng.exponential(0.2, n)     # monthly_balance_var
    f /= (f.std(0) + 1e-8)
    silo_feats['Bancos'] = f
    silo_labels['Bancos'] = np.zeros(n, dtype=np.int64)
    silo_src['Bancos'], silo_dst['Bancos'] = [], []
    silo_node_counts['Bancos'] = n

    # Legitimate Bancos edges (PIX/TED transactions)
    for _ in range(n * 3):
        a, b = rng.randint(0, n, 2)
        if a != b:
            silo_src['Bancos'] += [a, b]; silo_dst['Bancos'] += [b, a]

    # COAF: SAR nodes — ALL fraud by definition
    n = sizes['COAF']
    f = rng.rand(n, 10).astype(np.float32)
    f /= (f.std(0) + 1e-8)
    silo_feats['COAF'] = f
    silo_labels['COAF'] = np.ones(n, dtype=np.int64)  # all fraud
    silo_src['COAF'], silo_dst['COAF'] = [], []
    silo_node_counts['COAF'] = n

    # MP: investigation nodes — all fraud
    n = sizes['MP']
    f = rng.rand(n, 10).astype(np.float32)
    f /= (f.std(0) + 1e-8)
    silo_feats['MP'] = f
    silo_labels['MP'] = np.ones(n, dtype=np.int64)
    silo_src['MP'], silo_dst['MP'] = [], []
    silo_node_counts['MP'] = n

    # PF: criminal records
    n = sizes['PF']
    f = rng.rand(n, 10).astype(np.float32)
    f /= (f.std(0) + 1e-8)
    silo_feats['PF'] = f
    silo_labels['PF'] = np.ones(n, dtype=np.int64)
    silo_src['PF'], silo_dst['PF'] = [], []
    silo_node_counts['PF'] = n

    # ============================================================
    # INJECT FRAUD TYPOLOGIES
    # ============================================================
    scheme_nodes = {t: {s: [] for s in SILO_NAMES} for t in TYPOLOGY_NAMES}

    n_rfb = silo_node_counts['RFB']
    n_ban = silo_node_counts['Bancos']

    # ---- T1: Structuring / Smurfing (10 instances) ----
    # Central PJ → 5–12 PF intermediaries → recipient
    # RFB: sees income inconsistency on central PJ
    # Bancos: sees fragmented PIX convergence on recipient
    for _ in range(10):
        central_rfb = rng.randint(0, n_rfb)
        recipient_ban = rng.randint(0, n_ban)
        n_pf_int = rng.randint(5, 13)
        pf_nodes_ban = rng.choice(n_ban, n_pf_int, replace=False).tolist()

        # Mutate features: under-reported income, many small PIX
        silo_feats['RFB'][central_rfb, 0] *= 0.2
        silo_feats['RFB'][central_rfb, 4] *= 3.0   # high e-Financeira vs declared
        silo_feats['Bancos'][recipient_ban, 1] += 30  # many PIX
        silo_feats['Bancos'][recipient_ban, 4] += 20  # many counterparts
        for pf_n in pf_nodes_ban:
            silo_feats['Bancos'][pf_n, 0] = silo_feats['Bancos'][pf_n, 0] * 0.3  # small PIX
            silo_src['Bancos'] += [pf_n, recipient_ban]
            silo_dst['Bancos'] += [recipient_ban, pf_n]

        silo_labels['RFB'][central_rfb] = 1
        for pf_n in pf_nodes_ban:
            silo_labels['Bancos'][pf_n] = 1
        silo_labels['Bancos'][recipient_ban] = 1

        scheme_nodes['T1-Smurfing']['RFB'].append(central_rfb)
        scheme_nodes['T1-Smurfing']['Bancos'].extend(pf_nodes_ban + [recipient_ban])

    # ---- T2: Phantom / NF Fria (8 instances) ----
    # Shell PJ issues fictitious NF-e, dormant bank account, criminal record
    # RFB: sees NF-e volume with dormant fiscal history
    # Bancos: sees dormant account
    # PF: sees criminal record
    for i in range(8):
        shell_rfb = rng.randint(0, n_rfb)
        shell_ban = rng.randint(0, n_ban)
        pf_idx = min(i, silo_node_counts['PF'] - 1)

        silo_feats['RFB'][shell_rfb, 1] *= 5.0    # high NF-e volume
        silo_feats['RFB'][shell_rfb, 4] *= 0.05   # near-zero e-Financeira
        silo_feats['Bancos'][shell_ban, 1] = 0.1  # near-dormant
        silo_feats['Bancos'][shell_ban, 5] = 0.0  # zero inflow
        silo_feats['Bancos'][shell_ban, 6] = 0.0  # zero outflow

        silo_labels['RFB'][shell_rfb] = 1
        silo_labels['Bancos'][shell_ban] = 1
        silo_labels['PF'][pf_idx] = 1

        scheme_nodes['T2-NF_Fria']['RFB'].append(shell_rfb)
        scheme_nodes['T2-NF_Fria']['Bancos'].append(shell_ban)
        scheme_nodes['T2-NF_Fria']['PF'].append(pf_idx)

    # ---- T3: Crypto Concealment (5 instances) — ALL 5 SILOS ----
    # PF declares low income → transfers to exchange → SAR → PJ → real estate
    for i in range(5):
        pf_rfb = rng.randint(0, n_rfb)         # low DIRPF in RFB
        pf_ban = rng.randint(0, n_ban)          # crypto transfer in Bancos
        coaf_idx = min(i, silo_node_counts['COAF'] - 1)   # SAR in COAF
        mp_idx   = min(i, silo_node_counts['MP']  - 1)    # investigation in MP
        pf_idx   = min(i, silo_node_counts['PF']  - 1)    # criminal record in PF

        silo_feats['RFB'][pf_rfb, 7] *= 0.15   # very low DIRPF income
        silo_feats['RFB'][pf_rfb, 9] += 2.0    # high crypto declared
        silo_feats['Bancos'][pf_ban, 8] += 1.5  # very high tx_size_cv (crypto)
        silo_feats['Bancos'][pf_ban, 9] += 2.0  # high monthly variance

        silo_labels['RFB'][pf_rfb] = 1
        silo_labels['Bancos'][pf_ban] = 1
        # COAF, MP, PF already all-fraud

        scheme_nodes['T3-Cripto']['RFB'].append(pf_rfb)
        scheme_nodes['T3-Cripto']['Bancos'].append(pf_ban)
        scheme_nodes['T3-Cripto']['COAF'].append(coaf_idx)
        scheme_nodes['T3-Cripto']['MP'].append(mp_idx)
        scheme_nodes['T3-Cripto']['PF'].append(pf_idx)

    # ---- T4: False CNAE / Corporate Identity Fraud (8 instances) ----
    # PJ declares low-risk CNAE but real transactions match high-risk sector
    # RFB: sees CFOP inconsistency
    # Bancos: sees counterpart sector mismatch
    for _ in range(8):
        pj_rfb = rng.randint(0, n_rfb)
        pj_ban = rng.randint(0, n_ban)

        # Declared CNAE = 0 (tech, low risk), real CNAE = 5 (construction, high risk)
        silo_feats['RFB'][pj_rfb, 3] = 0.0   # declared CNAE: tech
        silo_feats['RFB'][pj_rfb, 2] = 8.0   # CFOP code pointing to construction
        silo_feats['Bancos'][pj_ban, 0] *= 4.0  # large irregular TEDs (construction)
        silo_feats['Bancos'][pj_ban, 2] += 2.0  # high PIX std
        silo_feats['Bancos'][pj_ban, 4] += 10   # many counterparts in real sector

        # Add edge to a construction-sector counterpart
        counterpart = rng.randint(0, n_ban)
        silo_src['Bancos'] += [pj_ban, counterpart]
        silo_dst['Bancos'] += [counterpart, pj_ban]

        silo_labels['RFB'][pj_rfb] = 1
        silo_labels['Bancos'][pj_ban] = 1

        scheme_nodes['T4-CNAE_Falso']['RFB'].append(pj_rfb)
        scheme_nodes['T4-CNAE_Falso']['Bancos'].append(pj_ban)

    # ---- T5: Sophisticated Money Mule (10 instances) ----
    # PF receives PIX from 15–45 distinct PJs
    # RFB: NF-e irregularities on originating PJs
    # Bancos: multi-source PIX convergence on mule account
    for _ in range(10):
        mule_ban = rng.randint(0, n_ban)
        n_sources = rng.randint(15, 46)
        source_pjs_rfb = rng.choice(n_rfb, min(n_sources, n_rfb//2), replace=False).tolist()
        source_pjs_ban = rng.choice(n_ban, min(n_sources, n_ban//2), replace=False).tolist()

        # Mule account: many inflows, many counterparts
        silo_feats['Bancos'][mule_ban, 4] += n_sources  # many counterparts
        silo_feats['Bancos'][mule_ban, 1] += n_sources  # many PIX
        silo_feats['Bancos'][mule_ban, 5] *= 5.0        # high inflow

        # Add PIX edges to mule
        for sp in source_pjs_ban[:10]:
            silo_src['Bancos'] += [sp, mule_ban]
            silo_dst['Bancos'] += [mule_ban, sp]

        # Mark RFB source PJs as suspicious
        for pj in source_pjs_rfb[:5]:
            silo_feats['RFB'][pj, 1] *= 0.1  # low NF-e volume (irregularity)
            silo_labels['RFB'][pj] = 1
            scheme_nodes['T5-Money_Mule']['RFB'].append(pj)

        silo_labels['Bancos'][mule_ban] = 1
        scheme_nodes['T5-Money_Mule']['Bancos'].append(mule_ban)

    # ---- T6: Caixa Dois via Interposed PJ (6 instances) ----
    # Controller PF owns both PJ-A and PJ-B; A→B→Controller circular flow
    # RFB: ownership graph (socio)
    # Bancos: circular A→B→PF cash flow
    # MP: investigation
    for i in range(6):
        controller_rfb = rng.randint(0, n_rfb)
        pj_a_rfb       = rng.randint(0, n_rfb)
        pj_b_rfb       = rng.randint(0, n_rfb)

        controller_ban = rng.randint(0, n_ban)
        pj_a_ban       = rng.randint(0, n_ban)
        pj_b_ban       = rng.randint(0, n_ban)
        mp_idx         = min(i, silo_node_counts['MP'] - 1)

        # RFB: ownership edges Controller→PJ-A and Controller→PJ-B
        silo_src['RFB'] += [controller_rfb, controller_rfb, pj_a_rfb]
        silo_dst['RFB'] += [pj_a_rfb, pj_b_rfb, pj_b_rfb]
        silo_feats['RFB'][controller_rfb, 5] += 2   # high ownership_degree

        # Bancos: A→B→Controller circular flow
        silo_src['Bancos'] += [pj_a_ban, pj_b_ban, controller_ban]
        silo_dst['Bancos'] += [pj_b_ban, controller_ban, pj_a_ban]
        silo_feats['Bancos'][pj_a_ban, 6] *= 3.0   # high outflow (payments to B)
        silo_feats['Bancos'][pj_b_ban, 5] *= 3.0   # high inflow (from A)
        silo_feats['Bancos'][controller_ban, 5] *= 2.0  # dividends inflow

        # Labels
        for n_idx in [controller_rfb, pj_a_rfb, pj_b_rfb]:
            silo_labels['RFB'][n_idx] = 1
            scheme_nodes['T6-Caixa_Dois']['RFB'].append(n_idx)
        for n_idx in [controller_ban, pj_a_ban, pj_b_ban]:
            silo_labels['Bancos'][n_idx] = 1
            scheme_nodes['T6-Caixa_Dois']['Bancos'].append(n_idx)
        scheme_nodes['T6-Caixa_Dois']['MP'].append(mp_idx)

    # ---- Convert to PyG ----
    silo_data = {}
    for sname in SILO_NAMES:
        silo_data[sname] = to_pyg(silo_feats[sname], silo_labels[sname],
                                   silo_src[sname], silo_dst[sname],
                                   seed=seed, device=DEVICE)

    return silo_data, scheme_nodes


# ============================================================
# SCHEME-LEVEL RECALL
# ============================================================

def scheme_recall(model_or_probs, data_or_none, scheme_node_list, threshold=0.5):
    """
    Compute scheme-level recall:
    fraction of fraud schemes where at least one node is detected.

    scheme_node_list: list of lists, each inner list = nodes belonging to one scheme instance.
    """
    if isinstance(model_or_probs, np.ndarray):
        probs = model_or_probs
    else:
        model_or_probs.eval()
        with torch.no_grad():
            out = model_or_probs(data_or_none.x, data_or_none.edge_index)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    if not scheme_node_list:
        return 0.0

    detected = 0
    for nodes in scheme_node_list:
        if any(probs[n] >= threshold for n in nodes if n < len(probs)):
            detected += 1
    return detected / len(scheme_node_list)


def build_scheme_lists(scheme_nodes, typology, silo):
    """
    Build list of scheme instance node lists for a given typology+silo.
    scheme_nodes[typology][silo] is a flat list; group into instances of fixed size.
    """
    flat = scheme_nodes[typology][silo]
    if not flat:
        return []
    # Group by instance (depends on typology)
    sizes_per_instance = {
        'T1-Smurfing': None,  # variable — treat each node as its own
        'T2-NF_Fria': 1,
        'T3-Cripto': 1,
        'T4-CNAE_Falso': 1,
        'T5-Money_Mule': 1,
        'T6-Caixa_Dois': 3,
    }
    s = sizes_per_instance.get(typology, 1)
    if s is None or s == 1:
        return [[n] for n in flat]
    # Group into chunks of s
    return [flat[i:i+s] for i in range(0, len(flat), s)]


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

def train_logreg(silo_data, silo_name):
    """Train LogReg on a single silo, return probability function."""
    data = silo_data[silo_name]
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    tr = data.train_mask.cpu().numpy()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])

    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    try:
        clf.fit(X_tr, y[tr])
    except Exception:
        return np.zeros(len(y))

    X_all = scaler.transform(X)
    try:
        probs = clf.predict_proba(X_all)[:, 1]
    except Exception:
        probs = np.zeros(len(y))
    return probs


def eval_logreg_probs(probs, data):
    te = data.test_mask.cpu().numpy()
    yt = data.y.cpu().numpy()
    if len(np.unique(yt[te])) < 2:
        return 0.5, 0.0
    auc = roc_auc_score(yt[te], probs[te])
    f1  = f1_score(yt[te], (probs[te] >= 0.5).astype(int), zero_division=0)
    return auc, f1


# ============================================================
# SECTION F: INTRA-COUNTRY EXPERIMENT (Section 6)
# ============================================================

def run_intra_country(seed=42, n_rounds=40, local_epochs=10, verbose=True):
    """
    Reproduce Section 6 intra-country experiment.
    Incremental silo ablation with LogReg proxy.
    Returns results dict.
    """
    print("\n" + "=" * 65)
    print(f"INTRA-COUNTRY 5-SILO EXPERIMENT (seed={seed})")
    print("=" * 65)

    silo_data, scheme_nodes = make_intra_country_data(seed=seed)

    # Isolated silo evaluation (LogReg)
    print("\n--- Isolated Silo Performance ---")
    print(f"{'Silo':<10} {'AUC':>6} {'F1':>6} {'Notes'}")
    print("-" * 50)
    isolated_probs = {}
    for sname in SILO_NAMES:
        probs = train_logreg(silo_data, sname)
        auc, f1 = eval_logreg_probs(probs, silo_data[sname])
        isolated_probs[sname] = probs
        note = "all fraud — no negative pop" if sname in ('COAF', 'MP', 'PF') else ""
        print(f"  {sname:<8} {auc:>6.3f} {f1:>6.3f}  {note}")

    # Incremental federation
    print("\n--- Incremental Federation (LogReg proxy) ---")
    print(f"{'Configuration':<25} {'Silos':>5} {'Nodes':>8} {'AUC':>6} {'F1':>6}")
    print("-" * 55)

    fed_orders = [
        ['RFB'],
        ['RFB', 'Bancos'],
        ['RFB', 'Bancos', 'COAF', 'MP'],
        ['RFB', 'Bancos', 'COAF', 'MP', 'PF'],
    ]

    intra_results = {}
    for silo_subset in fed_orders:
        models_sub = {s: GraphSAGE(10, 64, 2, 2).to(DEVICE) for s in silo_subset}
        init_w = list(models_sub.values())[0].get_weights()
        for m in models_sub.values(): m.set_weights(init_w)
        global_w = init_w

        for _ in range(n_rounds):
            ws, sizes = [], []
            for sname, m in models_sub.items():
                m.set_weights(global_w)
                w = local_train(m, silo_data[sname], local_epochs)
                ws.append(w)
                sizes.append(len(silo_data[sname].x))
            global_w = fedavg(ws, sizes)

        # Evaluate on RFB+Bancos combined (main fraud population)
        ref_silo = 'Bancos' if 'Bancos' in silo_subset else 'RFB'
        models_sub[ref_silo].set_weights(global_w)
        auc, f1, _ = evaluate(models_sub[ref_silo], silo_data[ref_silo])

        label = '+'.join(silo_subset)
        total_nodes = sum(len(silo_data[s].x) for s in silo_subset)
        print(f"  {label:<23} {len(silo_subset):>5} {total_nodes:>8} {auc:>6.3f} {f1:>6.3f}")
        intra_results[label] = {'auc': auc, 'f1': f1, 'silos': silo_subset}

    return silo_data, scheme_nodes, intra_results


# ============================================================
# SECTION G: COMPLETE GraphSAGE vs LogReg ABLATION — ALL 6 TYPOLOGIES
# ============================================================

def run_complete_typology_ablation(seed=42, n_rounds=40, local_epochs=10):
    """
    *** NEW CONTRIBUTION ***
    Complete ablation: LogReg vs GraphSAGE for all 6 fraud typologies.
    Computes scheme-level recall per typology for RFB+Bancos bilateral federation.
    This completes the gap identified in Section 12.1 of TaxFL v9.1.
    """
    print("\n" + "=" * 70)
    print("COMPLETE TYPOLOGY ABLATION: LogReg vs GraphSAGE — All 6 Typologies")
    print("Bilateral RFB+Bancos Federation")
    print("=" * 70)

    silo_data, scheme_nodes = make_intra_country_data(seed=seed)

    results = {}

    print(f"\n{'Typology':<20} {'LogReg AUC':>10} {'LogReg Rec':>10} "
          f"{'SAGE AUC':>10} {'SAGE Rec':>10} {'ΔAUC':>8} {'ΔRecall':>9}")
    print("-" * 80)

    typologies_silos = {
        'T1-Smurfing':   ['RFB', 'Bancos'],
        'T2-NF_Fria':    ['RFB', 'Bancos'],
        'T3-Cripto':     ['RFB', 'Bancos'],   # uses all 5 internally
        'T4-CNAE_Falso': ['RFB', 'Bancos'],
        'T5-Money_Mule': ['RFB', 'Bancos'],
        'T6-Caixa_Dois': ['RFB', 'Bancos'],
    }

    for typology in TYPOLOGY_NAMES:
        active_silos = typologies_silos[typology]

        # ---- LogReg on Bancos (primary detection silo) ----
        lr_probs_ban = train_logreg(silo_data, 'Bancos')
        lr_auc, lr_f1 = eval_logreg_probs(lr_probs_ban, silo_data['Bancos'])

        # Scheme recall for LogReg on Bancos
        ban_scheme_list = build_scheme_lists(scheme_nodes, typology, 'Bancos')
        lr_recall = scheme_recall(lr_probs_ban, None, ban_scheme_list)

        # ---- GraphSAGE federation (RFB + Bancos) ----
        models_fed = {s: GraphSAGE(10, 64, 2, 2).to(DEVICE) for s in active_silos}
        init_w = list(models_fed.values())[0].get_weights()
        for m in models_fed.values(): m.set_weights(init_w)
        global_w = init_w

        for _ in range(n_rounds):
            ws, sizes = [], []
            for sname, m in models_fed.items():
                m.set_weights(global_w)
                w = local_train(m, silo_data[sname], local_epochs)
                ws.append(w)
                sizes.append(len(silo_data[sname].x))
            global_w = fedavg(ws, sizes)

        # Evaluate GraphSAGE on Bancos
        models_fed['Bancos'].set_weights(global_w)
        sage_auc, sage_f1, _ = evaluate(models_fed['Bancos'], silo_data['Bancos'])

        # Scheme recall for GraphSAGE
        sage_recall = scheme_recall(models_fed['Bancos'], silo_data['Bancos'],
                                     ban_scheme_list)

        delta_auc    = sage_auc    - lr_auc
        delta_recall = sage_recall - lr_recall

        print(f"  {typology:<18} {lr_auc:>10.3f} {lr_recall:>10.2%} "
              f"{sage_auc:>10.3f} {sage_recall:>10.2%} "
              f"{delta_auc:>+8.3f} {delta_recall:>+8.2%}")

        results[typology] = {
            'lr_auc': lr_auc, 'lr_recall': lr_recall,
            'sage_auc': sage_auc, 'sage_recall': sage_recall,
            'delta_auc': delta_auc, 'delta_recall': delta_recall,
        }

    print()
    # Summary
    mean_dauc = np.mean([v['delta_auc'] for v in results.values()])
    mean_drec = np.mean([v['delta_recall'] for v in results.values()])
    print(f"  {'Mean improvement':<18} {'':>10} {'':>10} "
          f"{'':>10} {'':>10} {mean_dauc:>+8.3f} {mean_drec:>+8.2%}")

    return results


# ============================================================
# SCALE EXPERIMENTS  (Appendix B.7.1)
# ============================================================

def run_scale_experiment(seeds=(42, 123, 456)):
    print("\n" + "=" * 65)
    print("SCALE EXPERIMENT: 3 / 5 / 10 Jurisdictions")
    print("=" * 65)
    print(f"{'Config':<10} {'Jur':>4} {'Method':<10} {'F1 mean±std':>14} "
          f"{'AUC mean±std':>14} {'AUPRC mean±std':>16}")
    print("-" * 65)

    for n_jur, ne, nw, n_ev in [(3, 400, 400, 60),
                                  (5, 400, 800, 100),
                                  (10, 400, 2000, 150)]:
        label = {3: 'Small', 5: 'Medium', 10: 'Large'}[n_jur]
        for method in ['local', 'fedavg', 'fedprox']:
            f1s, aucs, auprcs = [], [], []
            for s in seeds:
                set_seed(s)
                ds = make_scale_data(n_jur, nw, ne, n_ev, s)
                if method == 'local':
                    m = GraphSAGE(10, 64, 2, 2).to(DEVICE)
                    for _ in range(40): local_train(m, ds[-1], 10)
                    auc, f1, auprc = evaluate(m, ds[-1])
                else:
                    auc, f1, auprc = fed_run_scale(ds, method=method, seed=s)
                f1s.append(f1); aucs.append(auc); auprcs.append(auprc)

            print(f"  {label:<8} {n_jur:>4} {method:<10} "
                  f"{np.mean(f1s):>6.3f}±{np.std(f1s):.3f}  "
                  f"{np.mean(aucs):>6.3f}±{np.std(aucs):.3f}  "
                  f"{np.mean(auprcs):>6.3f}±{np.std(auprcs):.3f}")


# ============================================================
# CLASS IMBALANCE EXPERIMENT  (Appendix B.7.2)
# ============================================================

def local_train_focal(model, data, epochs=10, lr=0.01, gamma=2.0):
    """Local training with focal loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        pt = probs[data.train_mask].gather(1, data.y[data.train_mask].unsqueeze(1)).squeeze()
        fl = -((1 - pt) ** gamma) * torch.log(pt + 1e-8)
        fl.mean().backward()
        opt.step()
    return model.get_weights()


def run_imbalance_experiment(seeds=(42, 123, 456)):
    print("\n" + "=" * 65)
    print("CLASS IMBALANCE: Loss Function Comparison (3 Jurisdictions)")
    print("=" * 65)
    print(f"{'Loss':<15} {'Method':<10} {'F1 mean±std':>14} "
          f"{'AUC mean':>10} {'AUPRC mean±std':>16}")
    print("-" * 55)

    ds_all = {s: make_scale_data(3, 400, 400, 60, s) for s in seeds}

    for loss_type in ['standard', 'weighted', 'focal']:
        for method in ['local', 'fedavg']:
            f1s, aucs, auprcs = [], [], []
            for s in seeds:
                set_seed(s)
                ds = ds_all[s]
                if method == 'local':
                    m = GraphSAGE(10, 64, 2, 2).to(DEVICE)
                    for _ in range(40):
                        if loss_type == 'focal':
                            local_train_focal(m, ds[-1])
                        else:
                            local_train(m, ds[-1])
                    auc, f1, auprc = evaluate(m, ds[-1])
                else:
                    models = [GraphSAGE(10, 64, 2, 2).to(DEVICE) for _ in range(3)]
                    iw = models[0].get_weights()
                    for m in models: m.set_weights(iw)
                    gw = iw
                    for _ in range(40):
                        ws, sz = [], []
                        for m, d in zip(models, ds):
                            m.set_weights(gw)
                            if loss_type == 'focal':
                                w = local_train_focal(m, d)
                            else:
                                w = local_train(m, d)
                            ws.append(w); sz.append(len(d.x))
                        gw = fedavg(ws, sz)
                    models[-1].set_weights(gw)
                    auc, f1, auprc = evaluate(models[-1], ds[-1])

                f1s.append(f1); aucs.append(auc); auprcs.append(auprc)

            print(f"  {loss_type:<13} {method:<10} "
                  f"{np.mean(f1s):>6.3f}±{np.std(f1s):.3f}  "
                  f"{np.mean(aucs):>10.3f}  "
                  f"{np.mean(auprcs):>6.3f}±{np.std(auprcs):.3f}")


# ============================================================
# REALISTIC SCENARIO + TIER 2 FIX  (Appendix B.7.3 & B.7.4)
# ============================================================

def run_realistic_scenario(seeds=(42, 123, 456)):
    print("\n" + "=" * 65)
    print("REALISTIC SCENARIO: Only Beneficiary Labeled (~3% illicit)")
    print("=" * 65)
    print(f"{'Config':<8} {'Jur':>4} {'Method':<18} {'F1':>7} {'AUC':>7} {'ΔAUC vs local':>14}")
    print("-" * 60)

    for n_jur in [3, 5]:
        label = {3: 'Small', 5: 'Medium'}[n_jur]
        local_aucs = []

        for method in ['local', 'fedavg_all', 'fedprox_all',
                        'fedavg_tier2', 'fedprox_tier2']:
            f1s, aucs = [], []
            for s in seeds:
                set_seed(s)
                ds = make_scale_data(n_jur, 400, 400, 60, s)

                # Only beneficiary (last jurisdiction) has positive labels
                for i in range(len(ds) - 1):
                    ds[i].y = torch.zeros_like(ds[i].y)

                tier2 = list(range(n_jur - 1)) if 'tier2' in method else []

                if method == 'local':
                    m = GraphSAGE(10, 64, 2, 2).to(DEVICE)
                    for _ in range(40): local_train(m, ds[-1])
                    auc, f1, _ = evaluate(m, ds[-1])
                else:
                    m_method = 'fedprox' if 'fedprox' in method else 'fedavg'
                    auc, f1, _ = fed_run_scale(ds, method=m_method,
                                                seed=s, tier2_indices=tier2)

                f1s.append(f1); aucs.append(auc)
                if method == 'local': local_aucs.append(auc)

            loc_mean = np.mean(local_aucs) if local_aucs else 0
            delta = np.mean(aucs) - loc_mean
            print(f"  {label:<6} {n_jur:>4} {method:<18} "
                  f"{np.mean(f1s):>7.3f} {np.mean(aucs):>7.3f} {delta:>+14.3f}")
        print()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    seeds = [42, 123, 456]

    if mode in ('scale', 'all'):
        run_scale_experiment(seeds)

    if mode in ('imbalance', 'all'):
        run_imbalance_experiment(seeds)

    if mode in ('realistic', 'all'):
        run_realistic_scenario(seeds)

    if mode in ('intra', 'all'):
        run_intra_country(seed=42)

    if mode in ('ablation', 'all'):
        run_complete_typology_ablation(seed=42)

    print("\nDone.")


if __name__ == '__main__':
    main()
