"""
Lateral–typewell join on **TVT** (canonical wide merge for tabular modeling).

Moved from **`wellbore_report V3.ipynb`** (Iteration 3–4) so training, CV, and
submission code import a single implementation instead of duplicating notebook cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import wellbore_cv as wcv

REPO_ROOT = Path(__file__).resolve().parent
JOINED_TRAIN_DIR = REPO_ROOT / "data" / "joined_train"

# Typewell-side columns appended after all lateral columns (stable contract).
TYPEWELL_FEATURE_COLS: tuple[str, ...] = (
    "tw_GR",
    "tw_GR_interp",
    "tw_Geology",
    "tw_TVT",
    "lat_tvt_below_tw_min",
    "lat_tvt_above_tw_max",
    "tw_gr_extrapolation_zone",
    "tw_interp_missing",
)

# Backward-compatible name used in older notebook text.
ITER3_TYPEWELL_FEATURE_COLS = TYPEWELL_FEATURE_COLS


def merge_lateral_typewell_schema_tvt(
    lateral_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return lateral_df plus typewell-derived columns joined on TVT.

    - tw_GR / tw_Geology / tw_TVT: left merge on exact TVT equality (sparse for tw_GR).
    - tw_GR_interp: piecewise-linear interp of typewell GR to each lateral TVT;
      NaN outside the typewell TVT knot span (left/right=np.nan).
    - lat_tvt_below_tw_min / lat_tvt_above_tw_max: per-well counts (same values on
      every row) for finite lateral TVT below / above the deduped typewell TVT span.
    - tw_gr_extrapolation_zone: int8 1 where finite lateral TVT is outside that span.
    - tw_interp_missing: int8 1 where tw_GR_interp is NaN (e.g. out-of-span or non-finite lateral TVT).
    """
    if "TVT" not in lateral_df.columns:
        raise KeyError("lateral_df must contain column 'TVT'")
    if "TVT" not in typewell_df.columns or "GR" not in typewell_df.columns:
        raise KeyError("typewell_df must contain 'TVT' and 'GR'")

    lateral_column_order = list(lateral_df.columns)
    reserved = set(TYPEWELL_FEATURE_COLS)
    collisions = set(lateral_column_order) & reserved
    if collisions:
        raise ValueError(
            "Lateral columns collide with reserved typewell feature names; "
            "rename on the lateral side or adjust TYPEWELL_FEATURE_COLS. "
            f"Overlapping: {sorted(collisions)}"
        )

    typewell_sorted_by_tvt = typewell_df.sort_values("TVT", kind="mergesort", ignore_index=True)
    typewell_one_row_per_tvt = typewell_sorted_by_tvt.drop_duplicates(
        subset=["TVT"], keep="last", ignore_index=True
    )
    tvt_knots = typewell_one_row_per_tvt["TVT"].to_numpy(dtype=float)
    gr_at_knots = typewell_one_row_per_tvt["GR"].to_numpy(dtype=float)

    typewell_for_exact_merge = typewell_one_row_per_tvt[["TVT", "GR"]].copy()
    typewell_for_exact_merge = typewell_for_exact_merge.rename(columns={"GR": "tw_GR"})
    typewell_for_exact_merge["tw_TVT"] = typewell_for_exact_merge["TVT"].to_numpy(dtype=float)

    if "Geology" in typewell_one_row_per_tvt.columns:
        typewell_for_exact_merge["tw_Geology"] = typewell_one_row_per_tvt["Geology"].to_numpy()

    merge_columns = ["TVT", "tw_GR", "tw_TVT"] + (
        ["tw_Geology"] if "tw_Geology" in typewell_for_exact_merge.columns else []
    )
    merged = lateral_df.merge(typewell_for_exact_merge[merge_columns], on="TVT", how="left", sort=False)

    lateral_tvt = lateral_df["TVT"].to_numpy(dtype=float)
    lateral_tvt_finite = np.isfinite(lateral_tvt)
    tw_gr_interp = np.full_like(lateral_tvt, np.nan, dtype=float)
    if tvt_knots.size and np.all(np.isfinite(tvt_knots)):
        tw_gr_interp[lateral_tvt_finite] = np.interp(
            lateral_tvt[lateral_tvt_finite],
            tvt_knots,
            gr_at_knots,
            left=np.nan,
            right=np.nan,
        )
    merged["tw_GR_interp"] = tw_gr_interp

    if tvt_knots.size == 0 or not np.all(np.isfinite(tvt_knots)):
        merged["lat_tvt_below_tw_min"] = np.nan
        merged["lat_tvt_above_tw_max"] = np.nan
        tw_gr_extrapolation_zone = np.zeros(len(merged), dtype=np.int8)
    else:
        typewell_tvt_min = float(np.min(tvt_knots))
        typewell_tvt_max = float(np.max(tvt_knots))
        merged["lat_tvt_below_tw_min"] = int(
            (lateral_tvt_finite & (lateral_tvt < typewell_tvt_min)).sum()
        )
        merged["lat_tvt_above_tw_max"] = int(
            (lateral_tvt_finite & (lateral_tvt > typewell_tvt_max)).sum()
        )
        tw_gr_extrapolation_zone = (
            lateral_tvt_finite
            & ((lateral_tvt < typewell_tvt_min) | (lateral_tvt > typewell_tvt_max))
        ).astype(np.int8)
    merged["tw_gr_extrapolation_zone"] = tw_gr_extrapolation_zone
    merged["tw_interp_missing"] = merged["tw_GR_interp"].isna().astype(np.int8)

    tail_columns = [c for c in TYPEWELL_FEATURE_COLS if c in merged.columns]
    ordered_columns = lateral_column_order + [c for c in tail_columns if c not in lateral_column_order]
    return merged.reindex(columns=ordered_columns)


def load_well_pair(
    well_id: str,
    data_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load paired lateral and typewell CSVs for ``well_id``."""
    root = data_root if data_root is not None else wcv.default_train_root()
    lateral_path = root / f"{well_id}__horizontal_well.csv"
    typewell_path = root / f"{well_id}__typewell.csv"
    if not lateral_path.is_file():
        raise FileNotFoundError(f"Missing lateral CSV: {lateral_path}")
    if not typewell_path.is_file():
        raise FileNotFoundError(f"Missing typewell CSV: {typewell_path}")
    return pd.read_csv(lateral_path), pd.read_csv(typewell_path)


def attach_typewell_by_tvt(
    lateral_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    tvt_col: str = "TVT",
) -> pd.DataFrame:
    """
    Merge typewell GR (exact + interp), optional geology, debug ``tw_TVT`` onto lateral rows.

    Delegates to :func:`merge_lateral_typewell_schema_tvt`, which expects the join column
    to be named ``TVT``. If ``tvt_col != "TVT"``, both frames must use that same column
    name as the join key, and neither may also carry a separate ``TVT`` column (ambiguous).
    The output restores your original join column name on the lateral side.
    """
    if tvt_col not in lateral_df.columns:
        raise KeyError(f"lateral_df missing join column {tvt_col!r}")
    if tvt_col not in typewell_df.columns:
        raise KeyError(f"typewell_df missing join column {tvt_col!r}")
    if "GR" not in typewell_df.columns:
        raise KeyError("typewell_df must contain 'GR'")

    if tvt_col == "TVT":
        return merge_lateral_typewell_schema_tvt(lateral_df, typewell_df)

    if "TVT" in lateral_df.columns:
        raise ValueError(
            "lateral_df already has column 'TVT' while tvt_col is not 'TVT'; ambiguous. "
            "Drop/rename one column, or use tvt_col='TVT'."
        )
    if "TVT" in typewell_df.columns:
        raise ValueError(
            "typewell_df already has column 'TVT' while tvt_col is not 'TVT'; ambiguous. "
            "Drop/rename one column, or use tvt_col='TVT'."
        )

    lateral_renamed = lateral_df.rename(columns={tvt_col: "TVT"})
    typewell_renamed = typewell_df.rename(columns={tvt_col: "TVT"})
    merged = merge_lateral_typewell_schema_tvt(lateral_renamed, typewell_renamed)
    return merged.rename(columns={"TVT": tvt_col})
