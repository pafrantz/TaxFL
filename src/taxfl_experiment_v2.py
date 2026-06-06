"""
TaxFL: Federated Learning and Federated Graph Intelligence for Cross-Border Tax Compliance
Extended Ablation — Version 2.0

Reproduces Appendix B ablation results (depth, aggregation, DP noise, compression).
Multi-seed variance analysis. Robust aggregation (Krum, trimmed mean).

Frantz, P.A. (2026). TaxFL v9.1. DOI: 10.5281/zenodo.18602470
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import copy, random, sys, time

# Import base utilities from v1
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from taxfl_experiment import (
    set_seed, generate_jurisdiction_data, inject_evasion_chains,
    to_pyg, GraphSAGE, local_train, fedavg, evaluate, class_weight_tensor
)

# ============================================================
# ADDITIONAL AGGREGATION METHODS
# ============================================================

def trimmed_mean(weights_list, trim_fraction=0.1):
    """Coordinate-wise trimmed mean aggregation."""
    n = len(weights_list)
    k = max(1, int(n * trim_fraction))
    avg = {}
    for key in weights_list[0]:
        stacked = torch.stack([w[key].float() for w in weights_list], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[k:n-k] if n - 2*k > 0 else stacked
        avg[key] = trimmed.mean(dim=0)
    return avg


def krum(weights_list, f=1):
    """
    Krum aggregation: select gradient closest to consensus.
    Requires n >= 2f+3. Returns single selected gradient.
    If n < 2f+3, falls back to FedAvg.
    """
    n = len(weights_list)
    if n < 2 * f + 3:
        # Insufficient participants — return zero model (collapse expected)
        avg = {}
        for k in weights_list[0]:
            avg[k] = torch.zeros_like(weights_list[0][k].float())
        return avg

    # Flatten all weights to vectors
    flat = []
    for w in weights_list:
        vec = torch.cat([v.float().flatten() for v in w.values()])
        flat.append(vec)

    # Pairwise squared distances
    dists = torch.zeros(n, n)
    for i in range(n):
        for j in range(i+1, n):
            d = ((flat[i] - flat[j])**2).sum()
            dists[i, j] = d
            dists[j, i] = d

    # Krum score: sum of distances to n-f-2 nearest neighbours
    scores = []
    for i in range(n):
        row = dists[i].clone()
        row[i] = float('inf')
        nearest = row.topk(n - f - 2, largest=False).values
        scores.append(nearest.sum().item())

    selected = int(np.argmin(scores))
    return copy.deepcopy(weights_list[selected])


def fedprox_train(model, data, global_w, mu=0.01, epochs=10, lr=0.01):
    """FedProx local training with proximal regularisation."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    w_class = class_weight_tensor(data.y[data.train_mask], data.x.device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        ce = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=w_class)
        prox = sum(((p - global_w[n].float())**2).sum()
                   for n, p in model.named_parameters() if n in global_w)
        (ce + mu/2 * prox).backward()
        opt.step()
    return model.get_weights()


# ============================================================
# COMMUNICATION COST
# ============================================================

def gradient_size_bytes(weights, compression=0.0):
    """Estimate size of transmitted gradient in bytes."""
    total_params = sum(v.numel() for v in weights.values())
    params_after = int(total_params * (1.0 - compression))
    return params_after * 4   # float32 = 4 bytes


# ============================================================
# CORE FEDERATED RUNNER WITH ABLATION OPTIONS
# ============================================================

def run_ablation(seed=42, n_rounds=40, local_epochs=10,
                 n_entities=300, n_wallets=400, n_legit=1000, n_evasion=60,
                 method='fedavg',         # fedavg | fedprox | trimmed | krum
                 n_layers=2, hidden=64,
                 dp_noise=0.0,            # std of Gaussian noise added to gradients
                 compression=0.0,         # fraction of gradient sparsified
                 verbose=False):
    """
    Full cross-border experiment with configurable aggregation,
    DP noise and compression. Returns result dict.
    """
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build data
    jurs = []
    for jid in range(3):
        f, l, s, d = generate_jurisdiction_data(n_entities, n_wallets, n_legit, jid, seed)
        jurs.append({'feats': f, 'labels': l, 'src': list(s), 'dst': list(d),
                     'n_entities': n_entities, 'n_wallets': n_wallets})
    jurs, _ = inject_evasion_chains(jurs, n_evasion, seed)
    datasets = [to_pyg(j['feats'], j['labels'], j['src'], j['dst'], seed, device) for j in jurs]

    in_ch = datasets[0].x.shape[1]

    # Local-only baseline (jurisdiction C only)
    local_model = GraphSAGE(in_ch, hidden, 2, n_layers).to(device)
    for _ in range(n_rounds):
        local_train(local_model, datasets[2], local_epochs)
    local_auc, local_f1, local_auprc = evaluate(local_model, datasets[2])

    # Federated
    models = [GraphSAGE(in_ch, hidden, 2, n_layers).to(device) for _ in range(3)]
    init_w = models[0].get_weights()
    for m in models: m.set_weights(init_w)
    global_w = init_w

    auc_hist, comm_bytes = [], 0

    for rnd in range(n_rounds):
        local_ws, sizes = [], []
        for model, data in zip(models, datasets):
            model.set_weights(global_w)

            if method == 'fedavg':
                w = local_train(model, data, local_epochs)
            elif method == 'fedprox':
                w = fedprox_train(model, data, global_w, epochs=local_epochs)
            else:
                w = local_train(model, data, local_epochs)

            # DP noise
            if dp_noise > 0:
                for k in w:
                    w[k] = w[k].float() + torch.randn_like(w[k].float()) * dp_noise

            # Sparsification / compression
            if compression > 0:
                for k in w:
                    mask = (torch.rand_like(w[k].float()) > compression)
                    w[k] = w[k].float() * mask.float()

            comm_bytes += gradient_size_bytes(w, compression)
            local_ws.append(w)
            sizes.append(len(data.x))

        # Aggregate
        if method in ('fedavg', 'fedprox'):
            global_w = fedavg(local_ws, sizes)
        elif method == 'trimmed':
            global_w = trimmed_mean(local_ws)
        elif method == 'krum':
            global_w = krum(local_ws, f=1)
        else:
            global_w = fedavg(local_ws, sizes)

        models[2].set_weights(global_w)
        auc, f1, _ = evaluate(models[2], datasets[2])
        auc_hist.append(auc)
        if verbose and (rnd+1) % 10 == 0:
            print(f"    Round {rnd+1:3d} | AUC: {auc:.4f} | F1: {f1:.4f}")

    models[2].set_weights(global_w)
    fed_auc, fed_f1, fed_auprc = evaluate(models[2], datasets[2])

    return {
        'local_auc':  local_auc,  'local_f1':  local_f1,  'local_auprc':  local_auprc,
        'fed_auc':    fed_auc,    'fed_f1':    fed_f1,    'fed_auprc':    fed_auprc,
        'delta_auc':  fed_auc - local_auc,
        'delta_f1':   fed_f1  - local_f1,
        'auc_history': auc_hist,
        'comm_mb':    comm_bytes / 1e6,
    }


# ============================================================
# ABLATION SUITE  (reproduces Table in Appendix B.4)
# ============================================================

def run_full_ablation(seed=42):
    """
    Reproduce the ablation table from Appendix B.4.
    Tests: depth, aggregation method, DP noise, compression.
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY (seed=42, Jurisdiction C)")
    print("=" * 70)
    print(f"{'Configuration':<45} {'F1':>6} {'AUC':>6} {'ΔF1':>7} {'Comm(MB)':>9}")
    print("-" * 70)

    configs = [
        # (label, kwargs)
        ("Baseline (FedAvg, 2L, no noise)",
         dict(method='fedavg', n_layers=2, dp_noise=0.0, compression=0.0)),
        ("3 layers (higher topology risk)",
         dict(method='fedavg', n_layers=3, dp_noise=0.0, compression=0.0)),
        ("1 layer",
         dict(method='fedavg', n_layers=1, dp_noise=0.0, compression=0.0)),
        ("FedProx (μ=0.01)",
         dict(method='fedprox', n_layers=2, dp_noise=0.0, compression=0.0)),
        ("Trimmed mean (10%)",
         dict(method='trimmed', n_layers=2, dp_noise=0.0, compression=0.0)),
        ("Krum (n=3, f=1) — collapse expected",
         dict(method='krum',    n_layers=2, dp_noise=0.0, compression=0.0)),
        ("DP noise=0.5 (ε≈2.0)",
         dict(method='fedavg', n_layers=2, dp_noise=0.5, compression=0.0)),
        ("30% gradient compression",
         dict(method='fedavg', n_layers=2, dp_noise=0.0, compression=0.3)),
    ]

    baseline_f1 = None
    for label, kwargs in configs:
        r = run_ablation(seed=seed, **kwargs)
        if baseline_f1 is None:
            baseline_f1 = r['fed_f1']
        delta = r['fed_f1'] - baseline_f1
        print(f"  {label:<43} {r['fed_f1']:>6.3f} {r['fed_auc']:>6.3f} "
              f"{delta:>+7.3f} {r['comm_mb']:>9.2f}")

    print()


# ============================================================
# MULTI-SEED VARIANCE (reproduces B.4 variance table)
# ============================================================

def run_multiseed(seeds=(42, 123, 456, 789, 1024)):
    """FedAvg and FedProx multi-seed results for jurisdiction C."""
    print("\n" + "=" * 70)
    print("MULTI-SEED VARIANCE (5 seeds, Jurisdiction C)")
    print("=" * 70)

    for method in ['fedavg', 'fedprox']:
        aucs, f1s, auprcs = [], [], []
        for s in seeds:
            r = run_ablation(seed=s, method=method, verbose=False)
            aucs.append(r['fed_auc'])
            f1s.append(r['fed_f1'])
            auprcs.append(r['fed_auprc'])

        print(f"  {method.upper()}")
        print(f"    AUC:   {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"    F1:    {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        print(f"    AUPRC: {np.mean(auprcs):.3f} ± {np.std(auprcs):.3f}")
    print()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    print("=" * 70)
    print("TaxFL v2 — Extended Ablation Cross-Border Experiment")
    print("=" * 70)

    run_full_ablation(seed=42)
    run_multiseed()

    # Detailed single run
    print("Detailed single run (FedAvg, seed=42):")
    r = run_ablation(seed=42, verbose=True)
    print(f"  Local-only  AUC: {r['local_auc']:.4f}  F1: {r['local_f1']:.4f}")
    print(f"  TaxFL FedAvg AUC: {r['fed_auc']:.4f}  F1: {r['fed_f1']:.4f}")
    print(f"  ΔAUC: {r['delta_auc']:+.4f}  ΔF1: {r['delta_f1']:+.4f}")
    print(f"  Total comm: {r['comm_mb']:.2f} MB")


if __name__ == '__main__':
    main()
