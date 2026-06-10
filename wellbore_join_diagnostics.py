"""
Per-well **join QA** for lateral vs typewell on **TVT** (Iteration 2 style).

Moved from **`wellbore_report V3.ipynb`** so batch summaries and notebooks share
the same diagnostics without duplicating long cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import wellbore_cv as wcv


def count_strict_decreases(values: np.ndarray) -> int:
    """Count pairs where value[i+1] < value[i] (along row order as stored in CSV)."""
    deltas = np.diff(np.asarray(values, dtype=float))
    return int(np.sum(np.isfinite(deltas) & (deltas < 0)))


def count_strict_plateaus(values: np.ndarray) -> int:
    """Count pairs where value[i+1] == value[i] (duplicate depth/TVT steps)."""
    deltas = np.diff(np.asarray(values, dtype=float))
    return int(np.sum(np.isfinite(deltas) & (deltas == 0)))


def lateral_typewell_join_diagnostics(
    lateral_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    *,
    well_id: str,
) -> dict[str, object]:
    """
    Return one dict of scalar diagnostics for a single (lateral, typewell) pair.

    Includes: duplicate TVT on typewell, monotonicity along CSV row order,
    lateral TVT vs typewell [min, max] span, and fraction of finite interp GR.
    """
    row: dict[str, object] = {"well_id": well_id}

    lateral_tvt = lateral_df["TVT"].to_numpy(dtype=float)
    lateral_tvt_is_finite = np.isfinite(lateral_tvt)
    row["lat_rows"] = int(len(lateral_tvt))

    if "MD" in lateral_df.columns:
        row["lat_md_decreases"] = count_strict_decreases(lateral_df["MD"].to_numpy(dtype=float))
    else:
        row["lat_md_decreases"] = np.nan

    if "MD" in typewell_df.columns:
        row["tw_md_decreases"] = count_strict_decreases(typewell_df["MD"].to_numpy(dtype=float))
    else:
        row["tw_md_decreases"] = np.nan

    row["lat_tvt_roworder_decreases"] = count_strict_decreases(lateral_tvt)
    row["tw_tvt_roworder_decreases"] = count_strict_decreases(
        typewell_df["TVT"].to_numpy(dtype=float)
    )

    typewell_sorted_by_tvt = typewell_df.sort_values("TVT", kind="mergesort", ignore_index=True)
    duplicate_tvt_row_mask = typewell_sorted_by_tvt["TVT"].duplicated(keep=False)
    row["tw_dup_tvt_rowcount"] = int(duplicate_tvt_row_mask.sum())

    typewell_one_row_per_tvt = typewell_sorted_by_tvt.drop_duplicates(
        subset=["TVT"], keep="last", ignore_index=True
    )
    row["tw_rows_raw"] = int(len(typewell_sorted_by_tvt))
    row["tw_unique_tvt"] = int(typewell_one_row_per_tvt["TVT"].nunique(dropna=False))
    row["tw_dedup_dropped"] = int(len(typewell_sorted_by_tvt) - len(typewell_one_row_per_tvt))

    tvt_knots = typewell_one_row_per_tvt["TVT"].to_numpy(dtype=float)
    gr_at_knots = typewell_one_row_per_tvt["GR"].to_numpy(dtype=float)

    if tvt_knots.size == 0 or not np.all(np.isfinite(tvt_knots)):
        row["tw_tvt_min"] = np.nan
        row["tw_tvt_max"] = np.nan
        row["lat_tvt_below_tw_min"] = np.nan
        row["lat_tvt_above_tw_max"] = np.nan
        row["lat_tvt_overlap_rows"] = np.nan
        row["tw_interp_finite_frac"] = np.nan
        row["tw_tvt_sorted_decreases"] = np.nan
        row["tw_tvt_sorted_plateaus"] = np.nan
        return row

    typewell_tvt_min = float(np.min(tvt_knots))
    typewell_tvt_max = float(np.max(tvt_knots))
    row["tw_tvt_min"] = typewell_tvt_min
    row["tw_tvt_max"] = typewell_tvt_max

    row["lat_tvt_below_tw_min"] = int((lateral_tvt_is_finite & (lateral_tvt < typewell_tvt_min)).sum())
    row["lat_tvt_above_tw_max"] = int((lateral_tvt_is_finite & (lateral_tvt > typewell_tvt_max)).sum())
    row["lat_tvt_overlap_rows"] = int(
        (
            lateral_tvt_is_finite
            & (lateral_tvt >= typewell_tvt_min)
            & (lateral_tvt <= typewell_tvt_max)
        ).sum()
    )

    tw_gr_interp = np.full_like(lateral_tvt, np.nan, dtype=float)
    tw_gr_interp[lateral_tvt_is_finite] = np.interp(
        lateral_tvt[lateral_tvt_is_finite],
        tvt_knots,
        gr_at_knots,
        left=np.nan,
        right=np.nan,
    )
    row["tw_interp_finite_frac"] = float(np.isfinite(tw_gr_interp).mean())

    row["tw_tvt_sorted_decreases"] = count_strict_decreases(tvt_knots)
    row["tw_tvt_sorted_plateaus"] = count_strict_plateaus(tvt_knots)

    return row


def load_lateral_typewell_or_none(
    well_id: str,
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (lateral_df, typewell_df) if both CSVs exist under ``data_root``, else None."""
    lateral_path = data_root / f"{well_id}__horizontal_well.csv"
    typewell_path = data_root / f"{well_id}__typewell.csv"
    if not (lateral_path.is_file() and typewell_path.is_file()):
        return None
    return pd.read_csv(lateral_path), pd.read_csv(typewell_path)


def diagnostics_dataframe_for_well_ids(well_ids: list[str], data_root: Path) -> pd.DataFrame:
    """Build a long table with one diagnostics row per well_id (skips missing pairs)."""
    rows: list[dict[str, object]] = []
    for wid in well_ids:
        pair = load_lateral_typewell_or_none(wid, data_root)
        if pair is None:
            continue
        lateral_df, typewell_df = pair
        required = {"TVT", "GR"}
        if not required.issubset(typewell_df.columns) or "TVT" not in lateral_df.columns:
            continue
        rows.append(lateral_typewell_join_diagnostics(lateral_df, typewell_df, well_id=wid))
    return pd.DataFrame(rows)


def discover_sorted_well_ids(data_root: Path | None = None) -> list[str]:
    """All lateral well ids under ``data_root`` (default: ``wellbore_cv.default_train_root()``)."""
    root = data_root if data_root is not None else wcv.default_train_root()
    stems: list[str] = []
    for p in wcv.discover_lateral_csvs(root):
        stem = wcv.lateral_well_stem(p)
        if stem is not None:
            stems.append(stem)
    return sorted(set(stems))
