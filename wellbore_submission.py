"""
Milestone 4 — test inference and **sample_submission**-aligned CSV export.

Builds the same wide tabular features as the Version 3 LightGBM cell (``train_tidy``
path: join + optional roll/lag + Milestone 3 boundary column on the merged frame),
fits a full-data **LightGBM** regressor on training rows with non-null manual **TVT**,
then writes ``id,tvt`` predictions in **exact** row count and order as
``data/sample_submission.csv``.

Raw ``data/train`` / ``data/test`` are never modified; reads default to
``wellbore_cv.default_train_root()`` and ``wellbore_cv.default_test_root()``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import wellbore_cv as wcv
import wellbore_feature_pruning as wbprune
import wellbore_features as wbfeat
import wellbore_geology as wbgeo
import wellbore_geology_boundary as wbgb

REPO_ROOT = Path(__file__).resolve().parent


def parse_submission_id(row_id: str) -> tuple[str, int]:
    """Split ``{well_id}_{foot_index}`` on the last underscore (well ids are hex-like)."""
    if "_" not in row_id:
        raise ValueError(f"submission id has no underscore: {row_id!r}")
    well, _, foot_s = row_id.rpartition("_")
    if not well:
        raise ValueError(f"empty well id in {row_id!r}")
    return well, int(foot_s)


def load_sample_submission(path: Path) -> pd.DataFrame:
    """Read Kaggle sample submission; add ``well_id`` and ``foot`` columns."""
    df = pd.read_csv(path)
    if list(df.columns) != ["id", "tvt"]:
        raise ValueError(f"expected columns id,tvt; got {list(df.columns)}")
    wells: list[str] = []
    feet: list[int] = []
    for rid in df["id"].astype(str):
        w, f = parse_submission_id(rid)
        wells.append(w)
        feet.append(f)
    out = df.copy()
    out["well_id"] = wells
    out["foot"] = feet
    return out


def _validate_sample_foot_sequences(sample: pd.DataFrame) -> None:
    """Each well's ``foot`` values must be contiguous integers in file order (local sanity)."""
    for well_id, g in sample.groupby("well_id", sort=False):
        feet = g["foot"].to_numpy(dtype=np.int64)
        if feet.size == 0:
            raise ValueError(f"empty foot block for well {well_id!r}")
        if not np.all(np.diff(feet) == 1):
            raise ValueError(f"non-contiguous foot indices for well {well_id!r}")
        span = int(feet.max() - feet.min() + 1)
        if span != len(feet):
            raise ValueError(f"foot span mismatch for well {well_id!r}")


def candidate_model_feature_names(
    *,
    use_roll: bool,
    use_lag: bool,
) -> list[str]:
    """Feature names matching ``wellbore_report V3_pruned`` LightGBM cell (non-pruned path)."""
    names = [
        "MD",
        "X",
        "Y",
        "Z",
        "GR",
        "GR_was_null",
        "TVT_input",
        "TVT_input_ffill",
        "tw_GR_interp",
        "tw_interp_missing",
        "tw_gr_extrapolation_zone",
        "lat_tvt_below_tw_min",
        "lat_tvt_above_tw_max",
        "tw_geology_code",
    ]
    if use_roll:
        names = names + list(wbfeat.roll_feature_column_names())
    if use_lag:
        names = names + list(wbfeat.lag_feature_column_names())
    return names


def collect_train_geology_labels(train_root: Path) -> pd.Series:
    """
    Pool every ``Geology`` label from **typewell** CSVs under ``train_root``.

    This matches the vocabulary used by ``tw_Geology`` on merged laterals without
    building the full join for every well twice.
    """
    parts: list[pd.Series] = []
    for lateral_path in wcv.discover_lateral_csvs(train_root):
        tw_path = lateral_path.with_name(
            lateral_path.name.replace("__horizontal_well.csv", "__typewell.csv")
        )
        if not tw_path.is_file():
            continue
        try:
            tw = pd.read_csv(tw_path, usecols=["Geology"])
        except ValueError:
            continue
        parts.append(tw["Geology"])
    if not parts:
        return pd.Series(dtype="object")
    return pd.concat(parts, ignore_index=True)


def stack_train_encoded(
    train_root: Path,
    geology_code_map: dict[str, int],
) -> pd.DataFrame:
    """All training wells: merged + roll/lag/M3 + ``tw_geology_code``."""
    chunks: list[pd.DataFrame] = []
    for lateral_path in wcv.discover_lateral_csvs(train_root):
        m = wbprune.load_merged_with_alonghole_features(
            lateral_path,
            use_roll=True,
            use_lag=True,
            allow_tvt_input_fallback=False,
            use_milestone3_geology_boundary=True,
        )
        if m is None:
            continue
        m = wbgeo.encode_merged_tw_geology(m, geology_code_map)
        chunks.append(m)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_test_merged_sorted(
    lateral_path: Path,
    geology_code_map: dict[str, int],
) -> pd.DataFrame | None:
    """Single test well: same feature pipeline as training, with TVT join fallback for tidy test."""
    m = wbprune.load_merged_with_alonghole_features(
        lateral_path,
        use_roll=True,
        use_lag=True,
        allow_tvt_input_fallback=True,
        use_milestone3_geology_boundary=True,
    )
    if m is None:
        return None
    m = wbgeo.encode_merged_tw_geology(m, geology_code_map)
    return m


def merged_rows_for_submission_feet(
    merged: pd.DataFrame,
    foot_min: int,
    foot_max: int,
    n_expected: int,
) -> pd.DataFrame:
    """
    Kaggle ``id`` foot integers are **1-based indices** along the **MD-sorted** lateral
    (first MD row → foot ``1``). The sample file usually spans a contiguous sub-range
    ``foot_min .. foot_max`` (inclusive), which is fewer rows than the full CSV when the
    sponsor omits a heel / build section from scoring.
    """
    if foot_min < 1 or foot_max < foot_min:
        raise ValueError(f"invalid foot range: {foot_min}..{foot_max}")
    if foot_max > len(merged):
        raise ValueError(
            f"foot_max={foot_max} exceeds merged lateral length {len(merged)} "
            f"(well {merged['well_id'].iloc[0]!r})"
        )
    # iloc end is exclusive → ``foot_max`` selects 0-based row ``foot_max - 1``.
    sub = merged.iloc[foot_min - 1 : foot_max].copy()
    if len(sub) != n_expected:
        raise ValueError(
            f"slice len {len(sub)} != sample block {n_expected} for feet {foot_min}..{foot_max}"
        )
    return sub


def assert_submission_frame_ok(sample: pd.DataFrame, out: pd.DataFrame) -> None:
    if len(out) != len(sample):
        raise AssertionError(
            f"submission row count {len(out)} != sample_submission {len(sample)}"
        )
    if not (out["id"].to_numpy() == sample["id"].to_numpy()).all():
        raise AssertionError("submission id column does not match sample_submission order")


def build_submission_dataframe(
    sample: pd.DataFrame,
    pred_by_id: dict[str, float],
    *,
    tvt_decimals: int | None = 6,
) -> pd.DataFrame:
    """Map predictions by ``id``; optional float rounding for picky uploaders."""
    tvt = sample["id"].map(pred_by_id).astype(float)
    if tvt.isna().any():
        missing = sample.loc[tvt.isna(), "id"].head(10).tolist()
        raise KeyError(f"missing predictions for ids (showing up to 10): {missing}")
    out = pd.DataFrame({"id": sample["id"], "tvt": tvt})
    if tvt_decimals is not None:
        out["tvt"] = out["tvt"].round(tvt_decimals)
    assert_submission_frame_ok(sample, out)
    return out


def run_submission_pipeline(
    *,
    train_root: Path,
    test_root: Path,
    sample_submission_path: Path,
    output_csv: Path,
    model_in_path: Path | None = None,
    model_out_path: Path | None = None,
    tvt_decimals: int | None = 6,
    verbose: bool = True,
) -> Path:
    """
    End-to-end: geology map from all train labels → fit LGBM (or load pickle) →
    predict test wells → write ``output_csv`` aligned to sample submission.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:  # pragma: no cover
        raise ImportError("wellbore_submission requires lightgbm") from e

    sample = load_sample_submission(sample_submission_path)
    _validate_sample_foot_sequences(sample)

    if model_in_path is not None:
        with open(model_in_path, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        geology_code_map = bundle["geology_code_map"]
        feat_cols: list[str] = bundle["feat_cols"]
        if verbose:
            print(f"Loaded model bundle from {model_in_path}")
    else:
        geo_series = collect_train_geology_labels(train_root)
        geology_code_map = wbgeo.build_geology_code_map(geo_series)
        train_df = stack_train_encoded(train_root, geology_code_map)
        train_df = train_df.loc[train_df["TVT"].notna()].copy()
        cand = candidate_model_feature_names(use_roll=True, use_lag=True)
        feat_cols = [c for c in cand if c in train_df.columns]
        if not feat_cols:
            raise RuntimeError("no usable feature columns after intersection with train frame")

        X_train = train_df[feat_cols].copy()
        y_train = train_df["TVT"].to_numpy(dtype=float)
        if verbose:
            print(
                f"Fitting LightGBM on {len(train_df):,} train rows, {len(feat_cols)} features, "
                f"{len(geology_code_map)} geology codes"
            )
        model = lgb.LGBMRegressor(
            n_estimators=3200,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        if model_out_path is not None:
            model_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_out_path, "wb") as f:
                pickle.dump(
                    {"model": model, "geology_code_map": geology_code_map, "feat_cols": feat_cols},
                    f,
                )
            if verbose:
                print(f"Wrote model bundle to {model_out_path}")

    pred_by_id: dict[str, float] = {}
    lateral_paths = {wcv.lateral_well_stem(p): p for p in wcv.discover_lateral_csvs(test_root)}
    for well_id, g in sample.groupby("well_id", sort=False):
        if well_id not in lateral_paths:
            raise FileNotFoundError(
                f"well {well_id!r} from sample_submission has no "
                f"*__horizontal_well.csv under {test_root}"
            )
        merged = load_test_merged_sorted(lateral_paths[well_id], geology_code_map)
        if merged is None:
            raise RuntimeError(f"failed to build merged frame for test well {well_id!r}")
        missing_cols = [c for c in feat_cols if c not in merged.columns]
        if missing_cols:
            raise KeyError(f"test well {well_id} missing feature columns: {missing_cols[:15]}")
        foot_min = int(g["foot"].min())
        foot_max = int(g["foot"].max())
        sub = merged_rows_for_submission_feet(merged, foot_min, foot_max, len(g))
        preds = model.predict(sub[feat_cols].copy()).astype(float)
        for rid, yhat in zip(g["id"].astype(str), preds):
            pred_by_id[rid] = float(yhat)

    out_df = build_submission_dataframe(sample, pred_by_id, tvt_decimals=tvt_decimals)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    if verbose:
        print(f"Wrote {output_csv} ({len(out_df):,} rows)")
    return output_csv


def _main() -> None:
    p = argparse.ArgumentParser(
        description="Milestone 4: test inference + sample_submission-aligned CSV."
    )
    p.add_argument(
        "--train-root",
        type=Path,
        default=None,
        help="Tidied training folder (default: wellbore_cv.default_train_root()).",
    )
    p.add_argument(
        "--test-root",
        type=Path,
        default=None,
        help="Tidied or raw test folder (default: wellbore_cv.default_test_root()).",
    )
    p.add_argument(
        "--sample-submission",
        type=Path,
        default=REPO_ROOT / "data" / "sample_submission.csv",
        help="Kaggle sample_submission.csv (defines row order and ids).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "submission.csv",
        help="Output CSV path (id,tvt).",
    )
    p.add_argument(
        "--model-in",
        type=Path,
        default=None,
        help="Optional pickle from a prior --model-out (skips full-train refit).",
    )
    p.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="If set (without --model-in), save fitted model + geology map + feat list.",
    )
    p.add_argument(
        "--tvt-decimals",
        type=int,
        default=6,
        help="Round tvt to this many decimals (set -1 to disable).",
    )
    args = p.parse_args()
    train_root = args.train_root or wcv.default_train_root()
    test_root = args.test_root or wcv.default_test_root()
    dec = None if args.tvt_decimals < 0 else int(args.tvt_decimals)
    run_submission_pipeline(
        train_root=train_root,
        test_root=test_root,
        sample_submission_path=args.sample_submission,
        output_csv=args.output,
        model_in_path=args.model_in,
        model_out_path=args.model_out,
        tvt_decimals=dec,
        verbose=True,
    )


if __name__ == "__main__":
    _main()
