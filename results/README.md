# Offline results

- **`v2_cv_trials_table.csv`** — mirror of the **Trials** table in `wellbore_report V2.ipynb` (trials **1-17** as logged there; no rolling Milestone 1 features — tabular baseline only).

- **`v3_cv_trials_table.csv`** — offline CV trial log (manual TVT RMSE unless noted). **Trials 1–3** mirror the **Trials** table in **`wellbore_report V3.ipynb`**: row **1** — 5-fold, `learning_rate=0.015`, early stopping; row **2** — 5-fold, `learning_rate=0.07`, no `eval_set` in that logged run; row **3** — 5-fold, 3200 estimators, mean RMSE **16.9725**. **Trials 4–5** are from **`wellbore_report V3_pruned.ipynb`**: row **4** — `USE_TABULAR_PRUNED_TRAIN=True`, pruned merged CSVs under `data/tabular_pruned/train/`, **13** LightGBM features, 5-fold mean RMSE **18.5493**; row **5** — `USE_TABULAR_PRUNED_TRAIN=False`, full `train_tidy` with roll and lag, **fold 0 only** (`FOLD_INDICES_TO_SCORE=[0]`), **86** features, fold-0 RMSE **15.1845**.

- **`v3_feature_corr_heatmap.png`** (optional) — Pearson correlation heatmap for **one** tidied lateral + Milestone 1–2 features; generate with `python wellbore_v3_feature_corr_heatmap.py --output results/v3_feature_corr_heatmap.png` when `data/train_tidy/` is present.
