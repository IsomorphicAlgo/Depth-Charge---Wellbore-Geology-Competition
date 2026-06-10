"""
Build a Pearson correlation heatmap for **one tidied lateral CSV** plus **V3 plan
features shipped so far** (Milestone 1 rolling + Milestone 2 lags), after the
same lateral–typewell join used in training.

**Usage** (from repo root, with ``data/train_tidy/`` populated)::

    python wellbore_v3_feature_corr_heatmap.py --output results/v3_feature_corr_heatmap.png

**Pick a specific well file**::

    python wellbore_v3_feature_corr_heatmap.py \\
        --horizontal-csv data/train_tidy/SomeWell__horizontal_well.csv \\
        --output results/my_well_corr.png

The script resolves the paired typewell by replacing ``__horizontal_well.csv``
with ``__typewell.csv``. It writes a single PNG (no notebook required).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wellbore_cv as wcv
import wellbore_features as wbfeat
from wellbore_join import attach_typewell_by_tvt

REPO_ROOT = Path(__file__).resolve().parent

# Numeric columns from the tidied lateral + join contract (subset; extend as needed).
_DEFAULT_BASE_FOR_CORR: tuple[str, ...] = (
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "TVT_input",
    "TVT_input_ffill",
    "TVT",
    "tw_GR_interp",
    "tw_interp_missing",
    "tw_gr_extrapolation_zone",
    "lat_tvt_below_tw_min",
    "lat_tvt_above_tw_max",
    "GR_was_null",
)


def _first_horizontal_csv(data_root: Path) -> Path:
    paths = sorted(data_root.glob("__horizontal_well.csv"))
    if not paths:
        raise FileNotFoundError(f"No *__horizontal_well.csv under {data_root}")
    return paths[0]


def _typewell_path(lateral_path: Path) -> Path:
    name = lateral_path.name
    if not name.endswith("__horizontal_well.csv"):
        raise ValueError(
            f"Lateral filename must end with '__horizontal_well.csv', got {name!r}"
        )
    return lateral_path.with_name(name.replace("__horizontal_well.csv", "__typewell.csv"))


def _well_id_from_lateral_path(lateral_path: Path) -> str:
    stem = lateral_path.name.replace("__horizontal_well.csv", "")
    return stem or "unknown_well"


def _coerce_training_numerics(lateral: pd.DataFrame) -> pd.DataFrame:
    out = lateral.copy()
    for col in ("MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "GR_was_null" in out.columns:
        out["GR_was_null"] = (
            pd.to_numeric(out["GR_was_null"], errors="coerce").fillna(0).astype(np.int8)
        )
    return out


def _coerce_typewell_numerics(typewell: pd.DataFrame) -> pd.DataFrame:
    out = typewell.copy()
    for col in ("TVT", "GR", "MD"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_merged_with_v3_features(lateral_path: Path) -> pd.DataFrame:
    """Load lateral + typewell, join on TVT, add ffill channel + roll + lag features."""
    tw_path = _typewell_path(lateral_path)
    if not tw_path.is_file():
        raise FileNotFoundError(f"Missing typewell CSV: {tw_path}")

    lateral_df = _coerce_training_numerics(pd.read_csv(lateral_path))
    typewell_df = _coerce_typewell_numerics(pd.read_csv(tw_path))
    merged = attach_typewell_by_tvt(lateral_df, typewell_df)
    merged["well_id"] = _well_id_from_lateral_path(lateral_path)

    merged = merged.sort_values("MD", kind="mergesort").reset_index(drop=True)
    if "TVT_input" in merged.columns:
        merged["TVT_input_ffill"] = merged["TVT_input"].ffill().bfill()
    else:
        merged["TVT_input_ffill"] = np.nan

    merged = wbfeat.add_alonghole_roll_features(merged)
    merged = wbfeat.add_alonghole_lag_features(merged)
    return merged


def _corr_column_order(merged: pd.DataFrame, *, include_target: bool) -> list[str]:
    base = [c for c in _DEFAULT_BASE_FOR_CORR if c in merged.columns]
    if not include_target and "TVT" in base:
        base.remove("TVT")

    roll = wbfeat.roll_feature_column_names(available_cols=merged.columns)
    lag = wbfeat.lag_feature_column_names(available_cols=merged.columns)
    ordered = base + roll + lag

    numeric: list[str] = []
    for c in ordered:
        if c not in merged.columns:
            continue
        if not pd.api.types.is_numeric_dtype(merged[c]):
            continue
        s = merged[c]
        if np.isfinite(s).sum() < 2:
            continue
        if float(np.nanstd(s.to_numpy(dtype=float))) < 1e-12:
            continue
        numeric.append(c)
    return numeric


def _write_heatmap(
    merged: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    include_target: bool,
    mask_upper_triangle: bool,
) -> int:
    cols = _corr_column_order(merged, include_target=include_target)
    if len(cols) < 2:
        print("Error: fewer than two usable numeric columns after filtering.", file=sys.stderr)
        return 1

    corr = merged[cols].corr(method="pearson", min_periods=10)
    n = len(cols)
    corr_arr = corr.to_numpy(dtype=float)
    non_finite = ~np.isfinite(corr_arr)
    if mask_upper_triangle:
        tri = np.triu(np.ones((n, n), dtype=bool), k=1)
        corr_plot = np.ma.array(corr_arr, mask=tri | non_finite)
    else:
        corr_plot = np.ma.array(corr_arr, mask=non_finite)

    fig_w = max(10.0, 0.22 * n)
    fig_h = max(8.0, 0.20 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(corr_plot, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(cols, rotation=90, fontsize=max(4, min(8, 120 // n)))
    ax.set_yticklabels(cols, fontsize=max(4, min(8, 120 // n)))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")
    plt.tight_layout()

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path} ({n}x{n} correlations)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Correlation heatmap: one train_tidy lateral + V3 roll/lag features."
    )
    parser.add_argument(
        "--horizontal-csv",
        type=Path,
        default=None,
        help="Path to *__horizontal_well.csv (default: first file under --data-root).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder with tidied laterals (default: wellbore_cv.default_train_root()).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "v3_feature_corr_heatmap.png",
        help="PNG path to write.",
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Exclude manual TVT from the correlation matrix (features + join only).",
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Show the full symmetric matrix instead of masking the upper triangle.",
    )
    args = parser.parse_args(argv)

    data_root = args.data_root or wcv.default_train_root()
    try:
        lateral_path = args.horizontal_csv or _first_horizontal_csv(data_root)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    lateral_path = lateral_path.resolve()
    if not lateral_path.is_file():
        print(f"Error: lateral CSV not found: {lateral_path}", file=sys.stderr)
        return 1

    try:
        merged = build_merged_with_v3_features(lateral_path)
    except Exception as exc:
        print(f"Error building merged frame: {exc}", file=sys.stderr)
        return 1

    include_target = not args.no_target
    n_preview = len(_corr_column_order(merged, include_target=include_target))
    title = (
        f"Pearson correlation — {_well_id_from_lateral_path(lateral_path)}\n"
        f"base + Milestone 1 roll + Milestone 2 lag ({n_preview} cols)"
    )

    return _write_heatmap(
        merged,
        args.output,
        title=title,
        include_target=include_target,
        mask_upper_triangle=not args.full_matrix,
    )


if __name__ == "__main__":
    raise SystemExit(main())
