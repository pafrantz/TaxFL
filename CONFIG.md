# CONFIG — Canonical configuration for reproducible v11 numbers

To make numbers comparable across tables, fix the following configuration. Deviations from it
(e.g. the legacy 19,470-node intra-country setup) must be stated explicitly in any table caption.

## Seeds
`[42, 123, 456, 789, 1024]` (5 seeds). Report mean ± 95% CI (t-based). For the heavier 5/10-jur
scale runs, 3 seeds `[42, 123, 456]` are acceptable but must be labelled as such.

## Data generators
- **Cross-border (Appendix B.4 main)**: `generate_jurisdiction_data` + `inject_evasion_chains`,
  3 jurisdictions, `n_entities=300`, `n_wallets=400`, `n_legit_tx=1000`, `n_evasion=60`
  (~4.8% illicit). Evaluation target: jurisdiction C, held-out `test_mask`.
- **Scale (Appendix B.3)**: `make_scale_data`, `n_entities_per=400`, `n_wallets_per ∈ {400, 800, 2000}`
  for {3, 5, 10} jurisdictions, `n_evasion ∈ {60, 100, 150}`.
- **Intra-country (Section 9 / Table 1)**: `make_intra_country_data`, default
  `n_rfb=500, n_bancos=500, n_coaf=10, n_mp=22, n_pf=37`, `target_fraud_rate=0.06`.
  **Do not** use the legacy 19,470/19,031 sizes unless explicitly reported as a separate
  large-scale configuration.

## Split
60/20/20 train/val/test, **stratified by label** (recommended for the low base rate). The test
split is held out for evaluation across all methods. Never evaluate on training nodes.

## Model
GraphSAGE, 2 layers, hidden 64, mean aggregation, dropout 0.3, input dim 10, output dim 2.

## Federation
- Rounds: 40. Local epochs: 10. Aggregation weighted by client train size.
- Optimizers: FedAvg, FedProx (μ=0.01), **SCAFFOLD** (control variates, option II).
- Local optimizer: Adam(lr=0.01, wd=5e-4) for FedAvg/FedProx baselines; SGD(lr=0.1, wd=5e-4)
  for the apples-to-apples SCAFFOLD comparison (report which is used).
- Loss: cross-entropy with inverse-frequency class weights (the default). Note: this makes a
  separate "weighted" condition redundant.

## Tiered participation (required)
All-fraud silos (COAF, MP, PF) are **Tier-2 consumers**: excluded from training, included only
for inference/consumption. They cannot contribute training gradients (no negative population).

## Metrics
Primary: **AUPRC** (≈6% positive rate). Secondary: AUC, F1 (note F1@0.5 is threshold-sensitive
under imbalance). Report scheme-level recall for the typology ablation.

## Environment
See `requirements.txt`. Numbers were produced on CPU; GPU acceleration changes timing only.
