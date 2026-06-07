# CV manifests

`kfold4_well_folds.csv` lists each training **lateral well id** and its fold index `0`–`3` for grouped 4-fold validation. Regenerate after changing the training tree:

```bash
python wellbore_cv.py
```

Defaults to `data/train_tidy/` when present, otherwise `data/train/`. Override with `--data-root` and `--out` if needed.

The CSV is small and safe to commit so everyone shares the same fold assignment for a given `random_state` (default `42`).
