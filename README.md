# TaxFL — Reproducibility Package (v11)

Federated learning and federated graph intelligence for cross-border tax and AML compliance.
This package accompanies the working paper *TaxFL* (v11) and contains the complete code to
reproduce every experiment, together with an honest record of the corrections made between
v10 and v11 (see `CHANGELOG.md`).

> **Note on the v10 reproducibility gap.** Earlier releases referenced experiment scripts
> (`taxfl_experiment_v2.py`, `taxfl_experiment_v3_gpu.py`) that were not present in the public
> repository, and the public repo contained only the Cora script. This package restores all
> scripts and adds the v11 corrections, resolving that gap.

## Layout

```
src/
  taxfl.py                    Cora GCN federated demo (ORIGINAL; superseded by taxfl_cora_v11.py)
  taxfl_experiment.py         v1: 3-jurisdiction cross-border generator, model, FedAvg, eval
  taxfl_experiment_v2.py      v2: FedProx, trimmed mean, Krum, ablation suite, multi-seed
  taxfl_experiment_v3_gpu.py  v3: scale (3/5/10), imbalance, realistic, intra-country 5-silo,
                              typology ablation (GraphSAGE vs LogReg)
  taxfl_cora_v11.py           Cora CORRECTED: global test split + FedAvg/FedProx/SCAFFOLD + CIs
  fl_harness_v11.py           Reusable, framework-agnostic FL harness (global split, all methods,
                              AUPRC/AUC/F1 with CIs); plug in your own build_data()
  taxfl_scaffold.py           SCAFFOLD integrated into the real pipeline (was missing)
  intra_bilateral.py          Bilateral RFB+Bancos: local vs FedAvg/FedProx/SCAFFOLD, per-silo
  scale_recheck.py            Independent corrected re-run of the 3-jur scale experiment
  scale_finalize.py           Regenerate 5/10-jur scale results from current code
  run_compare.py              Local vs federated delta on the current pipeline
results/
  scale_results.json          Measured 5/10-jurisdiction scale results
  RESULTS.md                  Headline measured numbers (mean ± 95% CI)
CONFIG.md                     The single canonical configuration for reproducible numbers
CHANGELOG.md                  v10 → v11 corrections (bugs, methodology, new results)
requirements.txt              Pinned dependencies
```

## Quick start

```bash
pip install -r requirements.txt

# Cora benchmark (corrected: global split, SCAFFOLD, CIs)
python src/taxfl_cora_v11.py

# 3-jurisdiction cross-border (original pipeline)
python src/taxfl_experiment.py

# Ablation, multi-seed, robust aggregation (original)
python src/taxfl_experiment_v2.py

# Scale / imbalance / realistic / intra-country / typology ablation (original)
python src/taxfl_experiment_v3_gpu.py all

# SCAFFOLD vs FedAvg/FedProx/local on the real pipeline (v11 addition)
python src/taxfl_scaffold.py

# Where federation helps: bilateral RFB+Bancos (v11 addition)
python src/intra_bilateral.py

# Finalize 5/10-jurisdiction scale results (v11)
python src/scale_finalize.py 5
python src/scale_finalize.py 10
```

`taxfl_experiment_v2.py` and `taxfl_experiment_v3_gpu.py` import from `taxfl_experiment.py`
via `sys.path.insert(0, '/home/claude')`. **Change that path** to your `src/` directory (or run
from inside `src/`) before executing.

## Reproducibility

- All experiments use fixed seeds (default `[42, 123, 456, 789, 1024]`).
- The canonical configuration is documented in `CONFIG.md`. Numbers in the paper's v11 tables
  are produced under that single configuration; deviations (e.g. the legacy 19,470-node intra
  configuration) are flagged in `CHANGELOG.md`.
- Headline measured numbers are in `results/RESULTS.md`.

## Known issues addressed in v11

See `CHANGELOG.md`. In brief: (1) the Cora script measured training accuracy on each client's
own mask (no held-out split) — fixed; (2) all-fraud silos (COAF/MP/PF) produce NaN updates and
must be Tier-2 consumers, not training contributors; (3) the intra-country routine switched the
evaluation silo between configurations — fixed to a single evaluation target; (4) SCAFFOLD was
recommended but never implemented for the domain — now provided.
