# Offline results

- **`v2_cv_trials_table.csv`** — mirror of the **Trials** table in `wellbore_report V2.ipynb` (trials **1-17** as logged there; no rolling Milestone 1 features — tabular baseline only).

- **`v3_cv_trials_table.csv`** — mirror of the **Trials** table in `wellbore_report V3.ipynb`: row **1** is the V3 daily CV run (fold-safe geology, RMSE on manual TVT); row **2** is the archived **Trial 2** markdown row (5-fold, `learning_rate=0.07`, no early stopping in that logged run).

- **`v3_feature_corr_heatmap.png`** (optional) — Pearson correlation heatmap for **one** tidied lateral + Milestone 1–2 features; generate with `python wellbore_v3_feature_corr_heatmap.py --output results/v3_feature_corr_heatmap.png` when `data/train_tidy/` is present.
