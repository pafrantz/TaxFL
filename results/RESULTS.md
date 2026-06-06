# RESULTS — measured headline numbers (v11)

All numbers produced by the code in this repository. Mean ± 95% CI unless noted.

## Cora benchmark (corrected, global split, 5 seeds) — `taxfl_cora_v11.py`

| Method | Global-test accuracy |
|---|---|
| Local-only — own-mask (v10-style, biased) | 0.921 ± 0.005 |
| Local-only — global test (corrected) | 0.508 ± 0.006 |
| Centralized (upper bound) | 0.796 ± 0.013 |
| FedAvg | 0.800 ± 0.018 |
| FedProx | 0.801 ± 0.011 |
| SCAFFOLD | **0.819 ± 0.013** |

## Cross-border 3-jur, jurisdiction C, SGD, 5 seeds — `taxfl_scaffold.py`

| Method | AUC | AUPRC | F1 |
|---|---|---|---|
| Local-only | 0.815 ± 0.031 | 0.471 ± 0.111 | 0.478 ± 0.074 |
| FedAvg | 0.794 ± 0.029 | 0.419 ± 0.084 | 0.375 ± 0.052 |
| FedProx | 0.793 ± 0.028 | 0.419 ± 0.085 | 0.375 ± 0.052 |
| SCAFFOLD | 0.812 ± 0.028 | 0.445 ± 0.088 | 0.461 ± 0.064 |

## Scale (current code, 3 seeds for 5/10) — `scale_finalize.py`

| Jur | Method | AUC | AUPRC | F1 |
|---|---|---|---|---|
| 5 | Local | 0.716 ± 0.061 | 0.095 ± 0.007 | 0.159 ± 0.031 |
| 5 | FedAvg | 0.784 ± 0.105 | 0.243 ± 0.173 | 0.120 ± 0.034 |
| 5 | FedProx | 0.786 ± 0.100 | 0.245 ± 0.169 | 0.152 ± 0.090 |
| 10 | Local | 0.749 ± 0.048 | 0.125 ± 0.093 | 0.165 ± 0.076 |
| 10 | FedAvg | 0.786 ± 0.041 | 0.176 ± 0.086 | 0.107 ± 0.053 |
| 10 | FedProx | 0.789 ± 0.042 | 0.150 ± 0.063 | 0.119 ± 0.044 |

Federation exceeds local on AUC and AUPRC at 5 and 10 jurisdictions.

## Typology ablation: GraphSAGE (RFB+Bancos) vs LogReg, scheme recall (seed 42) — `taxfl_experiment_v3_gpu.py ablation`

| Typology | LogReg recall | GraphSAGE recall | ΔAUC | ΔRecall |
|---|---|---|---|---|
| T1 | 53.8% | 99.1% | +0.250 | +45.3 pp |
| T2 | 50.0% | 100% | +0.298 | +50.0 pp |
| T3 | 40.0% | 80.0% | +0.293 | +40.0 pp |
| T4 | 100% | 100% | +0.245 | 0 |
| T5 | 100% | 100% | +0.267 | 0 |
| T6 | 83.3% | 100% | +0.281 | +16.7 pp |
| **Mean** | | | **+0.272** | **+25.3 pp** |

## Isolated LogReg per silo (5 seeds) — Table 1 (corrected, default config)

| Silo | AUC | F1 |
|---|---|---|
| RFB | 0.589 ± 0.073 | 0.334 ± 0.059 |
| Bancos | 0.611 ± 0.072 | 0.396 ± 0.038 |
| COAF / MP / PF | 0.500 ± 0.000 | 0.000 ± 0.000 (all-fraud artifact → Tier-2) |

## Loss comparison (3-jur, 3 seeds)

| Loss | Method | AUC | AUPRC | F1 |
|---|---|---|---|---|
| standard/weighted | Local | 0.656 | 0.190 ± 0.089 | 0.180 ± 0.085 |
| standard/weighted | FedAvg | 0.744 | 0.223 ± 0.070 | 0.179 ± 0.066 |
| focal | FedAvg | 0.698 | 0.129 | 0.000 |
