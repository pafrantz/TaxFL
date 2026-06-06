"""
TaxFL — federated harness for the synthetic tax/AML graph experiment (v11).
Framework-agnostic re-implementation of the evaluation protocol for Sections 9-10
and Appendix B, with the Fase 1 corrections baked in:

  (1) GLOBAL TEST SPLIT: every method is evaluated on the SAME held-out node set.
  (2) FedAvg / FedProx / SCAFFOLD as selectable optimizers.
  (3) Multi-seed runs with 95% confidence intervals.
  (4) Imbalance-aware metrics: AUPRC (primary), AUC, F1, precision, recall.

HOW TO USE WITH YOUR DATA
-------------------------
Replace `demo_build_data` with a function that wraps your oracle generator from
`taxfl_experiment_v3_gpu.py`. It must return, for a given seed, a dict with:

    x           : FloatTensor [N, F]   node features
    edge_index  : LongTensor  [2, E]   graph edges (use the UNION graph; each silo
                                        trains on its own node mask but message
                                        passing may use the local edges it owns)
    y           : LongTensor  [N]      binary labels (1 = fraud)
    client_nodes: list[LongTensor]     TRAINING node indices per silo (non-IID)
    test_nodes  : LongTensor           GLOBAL test node indices (shared by all)
    val_nodes   : LongTensor           (optional) global validation indices

The key invariant: client_nodes are disjoint from test_nodes, and test_nodes is
the SAME tensor for every method. That single change is what fixes the v10 bias.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- model -----------------------------
class GraphSAGE(torch.nn.Module):
    """2-layer GraphSAGE (mean aggregation), binary node classification."""
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.c2 = SAGEConv(hidden, 2, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.c2(x, edge_index)

# ----------------------------- helpers -----------------------------
def clone_state(m): return [p.detach().clone() for p in m.parameters()]
def load_params(m, ps):
    with torch.no_grad():
        for p, q in zip(m.parameters(), ps): p.copy_(q)
def zeros_like(m): return [torch.zeros_like(p) for p in m.parameters()]

def class_weights(y, nodes):
    """Inverse-frequency weights -> weighted cross-entropy (Appendix B.4 default)."""
    yt = y[nodes]
    n_pos = max(int((yt == 1).sum()), 1)
    n_neg = max(int((yt == 0).sum()), 1)
    w = torch.tensor([1.0 / n_neg, 1.0 / n_pos], device=y.device, dtype=torch.float)
    return w / w.sum() * 2.0

@torch.no_grad()
def evaluate(model, d):
    model.eval()
    logits = model(d["x"], d["edge_index"])
    prob = F.softmax(logits, dim=1)[:, 1]
    idx = d["test_nodes"]
    y_true = d["y"][idx].cpu().numpy()
    p = prob[idx].cpu().numpy()
    pred = (p >= 0.5).astype(int)
    out = {}
    out["AUPRC"] = average_precision_score(y_true, p) if y_true.sum() > 0 else float("nan")
    out["AUC"] = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
    out["F1"] = f1_score(y_true, pred, zero_division=0)
    out["precision"] = precision_score(y_true, pred, zero_division=0)
    out["recall"] = recall_score(y_true, pred, zero_division=0)
    return out

# ----------------------------- optimizers -----------------------------
def _local_steps_fedprox(model, d, nodes, gparams, epochs, lr, mu, wd, cw):
    model.train()
    for _ in range(epochs):
        for p in model.parameters(): p.grad = None
        out = model(d["x"], d["edge_index"])
        loss = F.cross_entropy(out[nodes], d["y"][nodes], weight=cw)
        if mu > 0:
            loss = loss + (mu / 2.0) * sum(((p - g) ** 2).sum()
                          for p, g in zip(model.parameters(), gparams))
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * (p.grad + wd * p)

def federated(d, make_model, method, rounds, local_epochs, lr, mu=0.01, wd=5e-4):
    g = make_model().to(DEVICE)
    sizes = np.array([len(c) for c in d["client_nodes"]], float)
    w = sizes / sizes.sum()
    cw_global = class_weights(d["y"], torch.cat(d["client_nodes"]))
    c_global = zeros_like(g)
    c_clients = [zeros_like(g) for _ in d["client_nodes"]]

    for _ in range(rounds):
        gparams = clone_state(g)
        agg = [torch.zeros_like(p) for p in gparams]
        agg_dc = [torch.zeros_like(p) for p in gparams]
        for k, nodes in enumerate(d["client_nodes"]):
            local = make_model().to(DEVICE); load_params(local, gparams)
            cw = class_weights(d["y"], nodes)
            if method in ("fedavg", "fedprox"):
                _local_steps_fedprox(local, d, nodes, gparams, local_epochs, lr,
                                     mu if method == "fedprox" else 0.0, wd, cw)
                with torch.no_grad():
                    for i, p in enumerate(local.parameters()):
                        agg[i] += w[k] * p.detach()
            elif method == "scaffold":
                local.train()
                for _ in range(local_epochs):
                    for p in local.parameters(): p.grad = None
                    out = local(d["x"], d["edge_index"])
                    loss = F.cross_entropy(out[nodes], d["y"][nodes], weight=cw)
                    loss.backward()
                    with torch.no_grad():
                        for i, p in enumerate(local.parameters()):
                            p -= lr * (p.grad + wd * p - c_clients[k][i] + c_global[i])
                with torch.no_grad():
                    yk = [p.detach().clone() for p in local.parameters()]
                    new_c = []
                    for i in range(len(gparams)):
                        nc = (c_clients[k][i] - c_global[i]
                              + (gparams[i] - yk[i]) / (local_epochs * lr))
                        new_c.append(nc)
                        agg[i] += w[k] * yk[i]
                        agg_dc[i] += (nc - c_clients[k][i]) / len(d["client_nodes"])
                    c_clients[k] = new_c
        load_params(g, agg)
        if method == "scaffold":
            with torch.no_grad():
                for i in range(len(c_global)): c_global[i] += agg_dc[i]
    return g

def local_only(d, make_model, epochs, lr, wd=5e-4):
    """Average over per-silo models, each evaluated on the GLOBAL test set."""
    metrics = []
    for nodes in d["client_nodes"]:
        m = make_model().to(DEVICE)
        cw = class_weights(d["y"], nodes)
        _local_steps_fedprox(m, d, nodes, clone_state(m), epochs, lr, 0.0, wd, cw)
        metrics.append(evaluate(m, d))
    return {k: float(np.mean([mm[k] for mm in metrics])) for k in metrics[0]}

# ----------------------------- runner -----------------------------
def ci95(vals):
    a = np.array(vals, float); a = a[~np.isnan(a)]
    if len(a) == 0: return float("nan"), float("nan")
    m = a.mean()
    if len(a) < 2: return m, 0.0
    return m, 1.96 * a.std(ddof=1) / np.sqrt(len(a))

def run(build_data, make_model, seeds=(42, 123, 456, 789, 1011),
        rounds=40, local_epochs=10, local_only_epochs=150, lr=0.05, mu=0.01):
    methods = ["local_only", "FedAvg", "FedProx", "SCAFFOLD"]
    metric_keys = ["AUPRC", "AUC", "F1", "precision", "recall"]
    store = {m: {k: [] for k in metric_keys} for m in methods}
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        d = build_data(s)
        for k in ("x", "edge_index", "y"):
            d[k] = d[k].to(DEVICE)
        d["client_nodes"] = [c.to(DEVICE) for c in d["client_nodes"]]
        d["test_nodes"] = d["test_nodes"].to(DEVICE)

        r = {}
        r["local_only"] = local_only(d, make_model, local_only_epochs, lr)
        r["FedAvg"] = evaluate(federated(d, make_model, "fedavg", rounds, local_epochs, lr), d)
        r["FedProx"] = evaluate(federated(d, make_model, "fedprox", rounds, local_epochs, lr, mu=mu), d)
        r["SCAFFOLD"] = evaluate(federated(d, make_model, "scaffold", rounds, local_epochs, lr), d)
        for m in methods:
            for k in metric_keys: store[m][k].append(r[m][k])
        print(f"seed {s}: " + " | ".join(
            f"{m} AUPRC={r[m]['AUPRC']:.3f}/AUC={r[m]['AUC']:.3f}" for m in methods), flush=True)

    print(f"\n=== mean +/- 95% CI over {len(seeds)} seeds (GLOBAL test split) ===")
    for k in metric_keys:
        print(f"\n[{k}]")
        for m in methods:
            mu_, h = ci95(store[m][k])
            print(f"  {m:12s} {mu_:.3f} +/- {h:.3f}")
    return store

# ----------------------------- demo data (REPLACE) -----------------------------
def demo_build_data(seed, n_per_silo=600, n_silos=3, frac_fraud=0.06):
    """
    Placeholder generator so the harness runs out-of-the-box. It builds a toy
    vertically-heterogeneous graph with planted 3-hop chains and a GLOBAL test
    split. REPLACE this with a wrapper around your oracle generator.
    """
    rng = np.random.default_rng(seed)
    N = n_per_silo * n_silos
    F_dim = 8
    x = torch.tensor(rng.normal(size=(N, F_dim)), dtype=torch.float)
    y = torch.zeros(N, dtype=torch.long)
    n_fraud = int(frac_fraud * N)
    fraud = rng.choice(N, size=n_fraud, replace=False)
    y[fraud] = 1
    x[fraud] += torch.tensor(rng.normal(0.8, 0.5, size=(n_fraud, F_dim)),
                             dtype=torch.float)  # weak signal
    # random sparse edges + intra-fraud chains
    src = rng.integers(0, N, size=4 * N)
    dst = rng.integers(0, N, size=4 * N)
    for i in range(0, len(fraud) - 2, 3):
        a, b, c = fraud[i], fraud[i + 1], fraud[i + 2]
        src = np.concatenate([src, [a, b]]); dst = np.concatenate([dst, [b, c]])
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    # global stratified split
    idx_all = np.arange(N)
    test_nodes, train_pool = [], []
    for cls in (0, 1):
        nodes = idx_all[(y.numpy() == cls)]; rng.shuffle(nodes)
        cut = int(0.3 * len(nodes))
        test_nodes += nodes[:cut].tolist(); train_pool += nodes[cut:].tolist()
    train_pool = np.array(train_pool)
    client_nodes = [torch.tensor(c, dtype=torch.long)
                    for c in np.array_split(rng.permutation(train_pool), n_silos)]
    return dict(x=x, edge_index=edge_index, y=y,
                client_nodes=client_nodes,
                test_nodes=torch.tensor(test_nodes, dtype=torch.long))

if __name__ == "__main__":
    make = lambda: GraphSAGE(in_dim=8)
    run(demo_build_data, make, seeds=(42, 123, 456), rounds=30,
        local_epochs=8, local_only_epochs=120, lr=0.05, mu=0.01)
