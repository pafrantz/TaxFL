"""
TaxFL: Federated Learning and Federated Graph Intelligence for Cross-Border Tax Compliance
Baseline Experiment — Version 1.0

Cross-border 3-jurisdiction synthetic experiment.
Reproduces Appendix B baseline results from:
  Frantz, P.A. (2026). TaxFL v9.1. DOI: 10.5281/zenodo.18602470

Requirements: torch, torch_geometric, scikit-learn, numpy
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import copy
import random

# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SYNTHETIC DATA — 3-JURISDICTION CROSS-BORDER
# ============================================================

def generate_jurisdiction_data(n_entities=300, n_wallets=400,
                                n_legit_tx=1000, jid=0, seed=42):
    """
    Generate synthetic tax/crypto data for one jurisdiction.

    Node types:
      0..n_entities-1    : tax entities (fiscal features)
      n_entities..N-1    : crypto wallets (transactional features)

    Features (10-dim):
      [0] declared_revenue  [1] tax_rate        [2] filing_count
      [3] revenue_variance  [4] tx_volume       [5] tx_frequency
      [6] avg_tx_size       [7] kyc_score       [8] crypto_exposure
      [9] vasp_count
    """
    rng = np.random.RandomState(seed + jid * 137)
    n = n_entities + n_wallets
    feats = np.zeros((n, 10), dtype=np.float32)

    # Entity fiscal features
    feats[:n_entities, 0] = rng.lognormal(10, 1.5, n_entities)
    feats[:n_entities, 1] = rng.uniform(0.1, 0.35, n_entities)
    feats[:n_entities, 2] = rng.poisson(12, n_entities).astype(float)
    feats[:n_entities, 3] = rng.exponential(0.3, n_entities)

    # Wallet transactional/KYC features
    feats[n_entities:, 4] = rng.lognormal(8, 2, n_wallets)
    feats[n_entities:, 5] = rng.poisson(50, n_wallets).astype(float)
    feats[n_entities:, 6] = rng.lognormal(6, 1, n_wallets)
    feats[n_entities:, 7] = rng.uniform(0, 1, n_wallets)
    feats[n_entities:, 8] = rng.exponential(0.1, n_wallets)
    feats[n_entities:, 9] = rng.poisson(2, n_wallets).astype(float)

    # Normalise
    col_std = feats.std(axis=0) + 1e-8
    feats = feats / col_std

    labels = np.zeros(n, dtype=np.int64)

    # Legitimate transaction edges (entity → wallet, bidirectional)
    src, dst = [], []
    for _ in range(n_legit_tx):
        e = rng.randint(0, n_entities)
        w = n_entities + rng.randint(0, n_wallets)
        src += [e, w]; dst += [w, e]

    return feats, labels, src, dst


def inject_evasion_chains(jurs, n_evasion=60, seed=42):
    """
    Inject A→B→C evasion chains across three jurisdictions.
    Origin in A, intermediary wallet in B, beneficiary in C.
    All three chain nodes labeled fraud=1.
    Returns modified jurs dict and lists of fraud node indices per jurisdiction.
    """
    rng = np.random.RandomState(seed)
    ne = jurs[0]['n_entities']
    nw = jurs[0]['n_wallets']

    fraud_nodes = {0: [], 1: [], 2: []}

    for _ in range(n_evasion):
        # A: origin entity + outflow wallet
        oe = rng.randint(0, ne)
        ow = ne + rng.randint(0, nw)
        # B: intermediary wallet
        iw = ne + rng.randint(0, nw)
        # C: beneficiary entity + inflow wallet
        be = rng.randint(0, ne)
        bw = ne + rng.randint(0, nw)

        # Modify features to embed evasion signal
        jurs[0]['feats'][oe, 0] *= 0.25   # under-reported revenue
        jurs[0]['feats'][oe, 3] += 2.5
        jurs[0]['feats'][ow, 4] *= 6.0
        jurs[0]['feats'][ow, 8] += 0.6

        jurs[1]['feats'][iw, 4] *= 4.0
        jurs[1]['feats'][iw, 7]  = 0.05   # suspicious KYC

        jurs[2]['feats'][be, 0] *= 0.15
        jurs[2]['feats'][bw, 4] *= 9.0
        jurs[2]['feats'][bw, 8] += 0.8

        # Edges within each jurisdiction
        jurs[0]['src'] += [oe, ow]; jurs[0]['dst'] += [ow, oe]
        jurs[1]['src'] += [iw, iw]; jurs[1]['dst'] += [iw, iw]  # self-loop marker
        jurs[2]['src'] += [bw, be]; jurs[2]['dst'] += [be, bw]

        # Labels
        jurs[0]['labels'][oe] = 1; jurs[0]['labels'][ow] = 1
        jurs[1]['labels'][iw] = 1
        jurs[2]['labels'][be] = 1; jurs[2]['labels'][bw] = 1

        fraud_nodes[0] += [oe, ow]
        fraud_nodes[1] += [iw]
        fraud_nodes[2] += [be, bw]

    return jurs, fraud_nodes


def to_pyg(feats, labels, src, dst, seed=0, device='cpu'):
    """Convert numpy arrays to PyG Data with train/val/test masks."""
    set_seed(seed)
    n = len(labels)
    x = torch.tensor(feats, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)

    if len(src) > 0:
        ei = torch.tensor([src, dst], dtype=torch.long).to(device)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long).to(device)

    perm = torch.randperm(n)
    tr = torch.zeros(n, dtype=torch.bool)
    va = torch.zeros(n, dtype=torch.bool)
    te = torch.zeros(n, dtype=torch.bool)
    tr[perm[:int(0.6*n)]] = True
    va[perm[int(0.6*n):int(0.8*n)]] = True
    te[perm[int(0.8*n):]] = True

    return Data(x=x, edge_index=ei, y=y,
                train_mask=tr.to(device),
                val_mask=va.to(device),
                test_mask=te.to(device))


# ============================================================
# MODEL
# ============================================================

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, hidden=64, out_ch=2, n_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        dims = [in_ch] + [hidden] * (n_layers - 1) + [out_ch]
        for i in range(n_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_weights(self):
        return copy.deepcopy(self.state_dict())

    def set_weights(self, w):
        self.load_state_dict(copy.deepcopy(w))


# ============================================================
# FEDERATED LEARNING
# ============================================================

def class_weight_tensor(y_train, device):
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        return None
    return torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)


def local_train(model, data, epochs=10, lr=0.01, wd=5e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    w = class_weight_tensor(data.y[data.train_mask], data.x.device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=w)
        loss.backward()
        opt.step()
    return model.get_weights()


def fedavg(weights_list, sizes):
    total = sum(sizes)
    avg = {}
    for k in weights_list[0]:
        avg[k] = sum(w[k].float() * (s / total) for w, s in zip(weights_list, sizes))
    return avg


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        yt = data.y.cpu().numpy()
        tm = data.test_mask.cpu().numpy()

        yl, pl, pp = yt[tm], preds[tm], probs[tm]
        auc = roc_auc_score(yl, pp) if len(np.unique(yl)) > 1 else 0.5
        f1  = f1_score(yl, pl, zero_division=0)
        auprc = average_precision_score(yl, pp) if len(np.unique(yl)) > 1 else 0.0
    return auc, f1, auprc


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(seed=42, n_rounds=40, local_epochs=10,
                   n_entities=300, n_wallets=400, n_legit=1000,
                   n_evasion=60, method='fedavg',
                   n_layers=2, hidden=64, verbose=True):
    """Run cross-border 3-jurisdiction experiment. Returns result dict."""
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build raw data
    jurs = []
    for jid in range(3):
        f, l, s, d = generate_jurisdiction_data(n_entities, n_wallets, n_legit, jid, seed)
        jurs.append({'feats': f, 'labels': l, 'src': list(s), 'dst': list(d),
                     'n_entities': n_entities, 'n_wallets': n_wallets})

    jurs, _ = inject_evasion_chains(jurs, n_evasion, seed)

    # PyG datasets
    datasets = [to_pyg(j['feats'], j['labels'], j['src'], j['dst'],
                       seed, device) for j in jurs]

    in_ch = datasets[0].x.shape[1]

    # ---- Local-only baseline ----
    local_results = []
    for i, data in enumerate(datasets):
        m = GraphSAGE(in_ch, hidden, 2, n_layers).to(device)
        for _ in range(n_rounds):
            local_train(m, data, local_epochs)
        local_results.append(evaluate(m, data))

    # ---- Federated training ----
    models = [GraphSAGE(in_ch, hidden, 2, n_layers).to(device) for _ in range(3)]
    init_w = models[0].get_weights()
    for m in models:
        m.set_weights(init_w)
    global_w = init_w

    auc_history = []
    for rnd in range(n_rounds):
        local_ws, sizes = [], []
        for model, data in zip(models, datasets):
            model.set_weights(global_w)
            w = local_train(model, data, local_epochs)
            local_ws.append(w)
            sizes.append(len(data.x))

        global_w = fedavg(local_ws, sizes)

        models[2].set_weights(global_w)
        auc, f1, _ = evaluate(models[2], datasets[2])
        auc_history.append(auc)

        if verbose and (rnd + 1) % 10 == 0:
            print(f"  Round {rnd+1:3d} | AUC(C): {auc:.4f} | F1(C): {f1:.4f}")

    # Final fed results
    fed_results = []
    for m, data in zip(models, datasets):
        m.set_weights(global_w)
        fed_results.append(evaluate(m, data))

    return {
        'local': local_results,
        'federated': fed_results,
        'auc_history': auc_history,
        'delta_auc_c': fed_results[2][0] - local_results[2][0],
    }


def main():
    print("=" * 60)
    print("TaxFL — Baseline Cross-Border Experiment (3 Jurisdictions)")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 1024]

    for method in ['fedavg']:
        print(f"\nMethod: {method.upper()} | Jurisdiction C results")
        fed_aucs, loc_aucs, fed_f1s = [], [], []

        for s in seeds:
            print(f"\n  Seed {s}")
            r = run_experiment(seed=s, method=method, verbose=True)
            fed_aucs.append(r['federated'][2][0])
            loc_aucs.append(r['local'][2][0])
            fed_f1s.append(r['federated'][2][1])

        print(f"\n  Local-only  AUC: {np.mean(loc_aucs):.3f} ± {np.std(loc_aucs):.3f}")
        print(f"  TaxFL FedAvg AUC: {np.mean(fed_aucs):.3f} ± {np.std(fed_aucs):.3f}")
        print(f"  TaxFL FedAvg F1:  {np.mean(fed_f1s):.3f} ± {np.std(fed_f1s):.3f}")
        print(f"  ΔAUC (mean):      {np.mean(fed_aucs) - np.mean(loc_aucs):+.3f}")


if __name__ == '__main__':
    main()
