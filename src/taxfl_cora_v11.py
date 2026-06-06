"""
TaxFL — Appendix A (Cora) corrected experiment, v11.
Fixes the v10 evaluation artifact (local-only measured as TRAIN accuracy on each
client's own skewed mask) by introducing a SINGLE GLOBAL TEST SPLIT shared by all
methods. Adds FedProx and SCAFFOLD alongside FedAvg, and reports multi-seed means
with 95% confidence intervals.

Key methodological points (Fase 1.1 / 1.2 / 1.4):
  - Non-IID skew governs only which TRAINING nodes each client sees.
  - ALL models (local-only, centralized, FedAvg, FedProx, SCAFFOLD) are evaluated
    on the same global, class-balanced TEST set.
  - We also report the OLD (buggy) "train-acc-on-own-mask" number to quantify the
    inflation it produced in v10.

All FL methods use plain SGD locally so the comparison is apples-to-apples and
SCAFFOLD's control variates are well defined (Karimireddy et al., 2020, option II).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- model -----------------------------
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=7, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)

# ----------------------------- helpers -----------------------------
def clone_state(model):
    return [p.detach().clone() for p in model.parameters()]

def load_params(model, params):
    with torch.no_grad():
        for p, q in zip(model.parameters(), params):
            p.copy_(q)

def zeros_like_params(model):
    return [torch.zeros_like(p) for p in model.parameters()]

@torch.no_grad()
def test_acc(model, data, idx):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    return accuracy_score(data.y[idx].cpu().numpy(), pred[idx].cpu().numpy())

@torch.no_grad()
def train_acc_own_mask(model, data, idx):
    # The v10-style metric: accuracy on the client's own (training) nodes.
    return test_acc(model, data, idx)

def make_global_split(data, seed, train_frac=0.6, val_frac=0.2):
    """Single global stratified split over ALL nodes, shared by every method."""
    rng = np.random.default_rng(seed)
    y = data.y.cpu().numpy()
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        nodes = np.where(y == c)[0]
        rng.shuffle(nodes)
        n_tr = int(train_frac * len(nodes))
        n_va = int(val_frac * len(nodes))
        train_idx += nodes[:n_tr].tolist()
        val_idx += nodes[n_tr:n_tr + n_va].tolist()
        test_idx += nodes[n_tr + n_va:].tolist()
    t = lambda L: torch.tensor(sorted(L), dtype=torch.long, device=DEVICE)
    return t(train_idx), t(val_idx), t(test_idx)

def make_noniid_clients(data, train_idx, seed):
    """Label-skew partition of the TRAINING pool only (mirrors v10 intent)."""
    rng = np.random.default_rng(seed + 7)
    y = data.y.cpu().numpy()
    tr = train_idx.cpu().numpy()
    y_tr = y[tr]
    c1 = tr[np.isin(y_tr, [0, 1, 2])]
    c2 = tr[np.isin(y_tr, [3, 4])]
    base3 = np.isin(y_tr, [5, 6])
    extra3 = rng.random(len(tr)) < 0.3
    c3 = tr[base3 | extra3]
    return [torch.tensor(c, dtype=torch.long, device=DEVICE) for c in (c1, c2, c3)]

# ----------------------------- training routines -----------------------------
def supervised_train(model, data, train_nodes, epochs, lr, wd=5e-4):
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_nodes], data.y[train_nodes])
        loss.backward()
        opt.step()
    return model

def fedavg_fedprox(data, clients, in_dim, n_classes, rounds, local_epochs,
                   lr, mu=0.0, wd=5e-4):
    """mu=0 -> FedAvg; mu>0 -> FedProx. Weighted by client train size."""
    g = GCN(in_dim, output_dim=n_classes).to(DEVICE)
    sizes = np.array([len(c) for c in clients], dtype=float)
    w = sizes / sizes.sum()
    for _ in range(rounds):
        global_params = clone_state(g)
        agg = [torch.zeros_like(p) for p in global_params]
        for k, nodes in enumerate(clients):
            local = GCN(in_dim, output_dim=n_classes).to(DEVICE)
            load_params(local, global_params)
            local.train()
            for _ in range(local_epochs):
                for p in local.parameters():
                    p.grad = None
                out = local(data.x, data.edge_index)
                loss = F.cross_entropy(out[nodes], data.y[nodes])
                if mu > 0:
                    prox = sum(((p - gp) ** 2).sum()
                               for p, gp in zip(local.parameters(), global_params))
                    loss = loss + (mu / 2.0) * prox
                loss.backward()
                with torch.no_grad():
                    for p in local.parameters():
                        p -= lr * (p.grad + wd * p)
            with torch.no_grad():
                for i, p in enumerate(local.parameters()):
                    agg[i] += w[k] * p.detach()
        load_params(g, agg)
    return g

def scaffold(data, clients, in_dim, n_classes, rounds, local_epochs, lr, wd=5e-4):
    """SCAFFOLD with full participation, lr_g=1, option-II control-variate update."""
    g = GCN(in_dim, output_dim=n_classes).to(DEVICE)
    sizes = np.array([len(c) for c in clients], dtype=float)
    w = sizes / sizes.sum()
    c_global = zeros_like_params(g)
    c_clients = [zeros_like_params(g) for _ in clients]
    for _ in range(rounds):
        global_params = clone_state(g)
        agg_dy = [torch.zeros_like(p) for p in global_params]
        agg_dc = [torch.zeros_like(p) for p in global_params]
        for k, nodes in enumerate(clients):
            local = GCN(in_dim, output_dim=n_classes).to(DEVICE)
            load_params(local, global_params)
            local.train()
            for _ in range(local_epochs):
                for p in local.parameters():
                    p.grad = None
                out = local(data.x, data.edge_index)
                loss = F.cross_entropy(out[nodes], data.y[nodes])
                loss.backward()
                with torch.no_grad():
                    for i, p in enumerate(local.parameters()):
                        corrected = p.grad + wd * p - c_clients[k][i] + c_global[i]
                        p -= lr * corrected
            with torch.no_grad():
                y_params = [p.detach().clone() for p in local.parameters()]
                new_c = []
                for i in range(len(global_params)):
                    nc = (c_clients[k][i] - c_global[i]
                          + (global_params[i] - y_params[i]) / (local_epochs * lr))
                    new_c.append(nc)
                    agg_dy[i] += w[k] * (y_params[i] - global_params[i])
                    agg_dc[i] += (new_c[i] - c_clients[k][i]) / len(clients)
                c_clients[k] = new_c
        with torch.no_grad():
            for i, p in enumerate(g.parameters()):
                p += agg_dy[i]                 # lr_g = 1
            for i in range(len(c_global)):
                c_global[i] += agg_dc[i]
    return g

# ----------------------------- experiment -----------------------------
def run_seed(seed, ds, data, cfg):
    torch.manual_seed(seed); np.random.seed(seed)
    in_dim, n_classes = ds.num_features, ds.num_classes
    train_idx, val_idx, test_idx = make_global_split(data, seed)
    clients = make_noniid_clients(data, train_idx, seed)

    res = {}

    # --- v10-style buggy metric (train acc on own mask), for contrast ---
    buggy = []
    for nodes in clients:
        m = GCN(in_dim, output_dim=n_classes).to(DEVICE)
        supervised_train(m, data, nodes, cfg["local_only_epochs"], cfg["lr"])
        buggy.append(train_acc_own_mask(m, data, nodes))
    res["local_only_OWNMASK_v10style"] = float(np.mean(buggy))

    # --- local-only, evaluated on GLOBAL TEST (the fix) ---
    local_test = []
    for nodes in clients:
        m = GCN(in_dim, output_dim=n_classes).to(DEVICE)
        supervised_train(m, data, nodes, cfg["local_only_epochs"], cfg["lr"])
        local_test.append(test_acc(m, data, test_idx))
    res["local_only_GLOBALTEST"] = float(np.mean(local_test))

    # --- centralized upper bound (all training nodes) ---
    m = GCN(in_dim, output_dim=n_classes).to(DEVICE)
    supervised_train(m, data, train_idx, cfg["local_only_epochs"], cfg["lr"])
    res["centralized"] = float(test_acc(m, data, test_idx))

    # --- federated methods, evaluated on GLOBAL TEST ---
    g = fedavg_fedprox(data, clients, in_dim, n_classes, cfg["rounds"],
                       cfg["local_epochs"], cfg["lr"], mu=0.0)
    res["FedAvg"] = float(test_acc(g, data, test_idx))

    g = fedavg_fedprox(data, clients, in_dim, n_classes, cfg["rounds"],
                       cfg["local_epochs"], cfg["lr"], mu=cfg["mu"])
    res["FedProx"] = float(test_acc(g, data, test_idx))

    g = scaffold(data, clients, in_dim, n_classes, cfg["rounds"],
                 cfg["local_epochs"], cfg["lr"])
    res["SCAFFOLD"] = float(test_acc(g, data, test_idx))
    return res

def ci95(vals):
    a = np.array(vals, dtype=float)
    m = a.mean()
    if len(a) < 2:
        return m, 0.0
    se = a.std(ddof=1) / np.sqrt(len(a))
    return m, 1.96 * se

def main():
    ds = Planetoid(root="/tmp/Cora", name="Cora")
    data = ds[0].to(DEVICE)
    cfg = dict(seeds=[42, 123, 456, 789, 1011],
               rounds=50, local_epochs=10, local_only_epochs=150,
               lr=0.15, mu=0.05)
    keys = ["local_only_OWNMASK_v10style", "local_only_GLOBALTEST",
            "centralized", "FedAvg", "FedProx", "SCAFFOLD"]
    acc = {k: [] for k in keys}
    for s in cfg["seeds"]:
        r = run_seed(s, ds, data, cfg)
        for k in keys:
            acc[k].append(r[k])
        print(f"seed {s}: " + " | ".join(f"{k}={r[k]:.3f}" for k in keys))
    print("\n=== Cora — global-test accuracy (mean ± 95% CI over "
          f"{len(cfg['seeds'])} seeds) ===")
    for k in keys:
        m, h = ci95(acc[k])
        print(f"{k:32s} {m:.3f} ± {h:.3f}")

if __name__ == "__main__":
    main()
