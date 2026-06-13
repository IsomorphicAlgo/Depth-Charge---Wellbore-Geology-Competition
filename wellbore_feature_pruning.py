"""
Greedy Pearson-correlation pruning for wide tabular feature frames.

Used after lateral–typewell merge and optional along-hole roll/lag features.
Does **not** modify raw ``data/train`` or ``data/test`` — export helpers write
under ``data/tabular_pruned/`` (or another caller-provided root).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import wellbore_cv as wcv
import wellbore_features as wbfeat
from wellbore_join import attach_typewell_by_tvt

# Columns to keep out of the |r| > threshold game (identifiers, targets, masks).
DEFAULT_PROTECT_COLS: tuple[str, ...] = (
    "well_id",
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "TVT",
    "TVT_input",
    "TVT_input_ffill",
    "tw_geology_code",
    "tw_interp_missing",
    "tw_gr_extrapolation_zone",
    "lat_tvt_below_tw_min",
    "lat_tvt_above_tw_max",
    "GR_was_null",
)


@dataclass(frozen=True)
class PruneResult:
    """Outcome of :func:`prune_correlated_features`."""

    kept_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    protected_columns: tuple[str, ...]
    n_feature_candidates_initial: int
    n_feature_candidates_final: int


def prune_correlated_features(
    df: pd.DataFrame,
    target_col: str,
    *,
    threshold: float = 0.95,
    protect_cols: Sequence[str] | None = None,
    feature_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, PruneResult]:
    """
    Drop numeric feature columns that are highly Pearson-correlated with each other.

    Greedy loop: among pairs with ``|r| > threshold``, drop the column whose
    absolute correlation with ``target_col`` is weaker (tie-break). Only numeric
    ``feature_cols`` (or inferred candidates) participate; protected, non-numeric,
    and unknown dtypes are left unchanged.

    Rows with NaN in ``target_col`` are ignored when computing correlations.
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col!r} not in dataframe columns")

    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be strictly between 0 and 1")

    prot_in = tuple(DEFAULT_PROTECT_COLS) if protect_cols is None else tuple(protect_cols)
    protected = {c for c in prot_in if c in df.columns} | {target_col}

    non_numeric = set(df.select_dtypes(exclude=[np.number]).columns)
    numeric = set(df.select_dtypes(include=[np.number]).columns)

    if feature_cols is None:
        candidates = sorted(numeric - protected)
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"feature_cols not in df: {missing[:10]!r}{'...' if len(missing) > 10 else ''}")
        candidates = [c for c in feature_cols if c in numeric and c not in protected]

    if not candidates:
        kept = tuple(df.columns)
        return df.copy(), PruneResult(
            kept_columns=kept,
            dropped_columns=(),
            protected_columns=tuple(sorted(protected)),
            n_feature_candidates_initial=0,
            n_feature_candidates_final=0,
        )

    base = df.loc[df[target_col].notna(), list({*candidates, target_col})].copy()
    active = list(candidates)
    n0 = len(active)
    dropped: list[str] = []

    target_series = base[target_col]

    while len(active) > 1:
        sub = base[active]
        if sub.shape[1] < 2:
            break
        C = sub.corr(numeric_only=True)
        C_abs = C.abs().fillna(0.0)
        np.fill_diagonal(C_abs.values, 0.0)
        vmax = float(C_abs.max().max())
        if vmax <= threshold or not np.isfinite(vmax):
            break
        ridx, cidx = np.unravel_index(int(np.argmax(C_abs.values)), C_abs.shape)
        ci, cj = C_abs.index[int(ridx)], C_abs.columns[int(cidx)]

        t_abs: dict[str, float] = {}
        for col in (ci, cj):
            pair = base[[col, target_col]].dropna()
            if len(pair) < 2:
                t_abs[col] = 0.0
            else:
                t_abs[col] = abs(float(pair[col].corr(pair[target_col])))

        drop = cj if t_abs.get(ci, 0.0) >= t_abs.get(cj, 0.0) else ci
        if drop not in active:
            break
        active.remove(drop)
        dropped.append(drop)
        base.drop(columns=[drop], inplace=True, errors="ignore")

    removed_set = set(dropped)
    kept_columns = tuple(c for c in df.columns if c not in removed_set)

    out = df.loc[:, list(kept_columns)].copy()
    return out, PruneResult(
        kept_columns=kept_columns,
        dropped_columns=tuple(dropped),
        protected_columns=tuple(sorted(protected)),
        n_feature_candidates_initial=n0,
        n_feature_candidates_final=len(active),
    )


def _read_numeric_horizon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "GR_was_null" in out.columns:
        out["GR_was_null"] = (
            pd.to_numeric(out["GR_was_null"], errors="coerce").fillna(0).astype(np.int8)
        )
    return out


def _read_numeric_typewell(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("TVT", "GR", "MD"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _ensure_lateral_tvt_for_join(
    lateral_df: pd.DataFrame,
    *,
    allow_tvt_input_fallback: bool,
) -> pd.DataFrame:
    """
    Guarantee a numeric ``TVT`` column for :func:`attach_typewell_by_tvt`.

    Training laterals always carry ``TVT``. Tidied **test** horizontals may omit it
    and only ship ``TVT_input``; when ``allow_tvt_input_fallback`` is True, copy
    ``TVT_input`` into ``TVT``, and fill NaN ``TVT`` from ``TVT_input`` where both exist.
    """
    out = lateral_df.copy()
    if "TVT" not in out.columns:
        if allow_tvt_input_fallback and "TVT_input" in out.columns:
            out["TVT"] = out["TVT_input"]
        else:
            raise KeyError(
                "lateral horizon CSV has no 'TVT' column. Tidied test horizontals may only "
                "have 'TVT_input'; export with allow_tvt_input_fallback=True for that split."
            )
    elif allow_tvt_input_fallback and "TVT_input" in out.columns:
        out["TVT"] = out["TVT"].fillna(out["TVT_input"])
    return out


def load_premerged_lateral_csv(
    lateral_path: Path,
    *,
    well_id: str | None = None,
) -> pd.DataFrame | None:
    """
    Load a CSV that is already the wide merged lateral+typewell table (e.g. from
    :func:`export_pruned_merged_tabular`). Skips join and along-hole feature steps.
    """
    if not lateral_path.is_file():
        return None
    out = pd.read_csv(lateral_path)
    if well_id is not None:
        if "well_id" not in out.columns:
            out = out.copy()
            out["well_id"] = well_id
        else:
            out = out.copy()
            out["well_id"] = well_id
    return out


def load_merged_with_alonghole_features(
    lateral_path: Path,
    *,
    well_id: str | None = None,
    use_roll: bool = True,
    use_lag: bool = True,
    allow_tvt_input_fallback: bool = False,
) -> pd.DataFrame | None:
    """
    Load tidied lateral + paired typewell, join on TVT, add roll/lag like the V3 notebook.

    ``well_id`` defaults to the stem parsed from ``*__horizontal_well.csv``.

    ``allow_tvt_input_fallback``: set True for **test** tidied horizontals that omit ``TVT``
    (see :func:`_ensure_lateral_tvt_for_join`). Leave False for training so ``TVT`` is never
    silently borrowed from ``TVT_input``.
    """
    if not lateral_path.is_file():
        return None
    wid = well_id if well_id is not None else wcv.lateral_well_stem(lateral_path)
    if wid is None:
        return None
    typewell_path = lateral_path.with_name(
        lateral_path.name.replace("__horizontal_well.csv", "__typewell.csv")
    )
    if not typewell_path.is_file():
        return None

    lateral_df = _read_numeric_horizon(pd.read_csv(lateral_path))
    lateral_df = _ensure_lateral_tvt_for_join(
        lateral_df, allow_tvt_input_fallback=allow_tvt_input_fallback
    )
    typewell_df = _read_numeric_typewell(pd.read_csv(typewell_path))
    merged = attach_typewell_by_tvt(lateral_df, typewell_df)
    merged["well_id"] = wid
    merged = merged.sort_values("MD", kind="mergesort").reset_index(drop=True)
    if "TVT_input" in merged.columns:
        merged["TVT_input_ffill"] = merged["TVT_input"].ffill().bfill()
    else:
        merged["TVT_input_ffill"] = np.nan
    if use_roll:
        merged = wbfeat.add_alonghole_roll_features(merged)
    if use_lag:
        merged = wbfeat.add_alonghole_lag_features(merged)
    return merged


def stack_merged_frames_for_pruning(
    data_root: Path,
    *,
    use_roll: bool = True,
    use_lag: bool = True,
) -> pd.DataFrame:
    """Concatenate every tidied lateral (with merge + along-hole features) under ``data_root``."""
    paths = wcv.discover_lateral_csvs(data_root)
    chunks: list[pd.DataFrame] = []
    for lateral_path in paths:
        m = load_merged_with_alonghole_features(
            lateral_path,
            use_roll=use_roll,
            use_lag=use_lag,
        )
        if m is not None:
            chunks.append(m)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def export_pruned_merged_tabular(
    data_root: Path,
    dest_root: Path,
    keep_columns: Sequence[str],
    *,
    use_roll: bool = True,
    use_lag: bool = True,
    allow_tvt_input_fallback: bool = False,
) -> int:
    """
    Write one merged CSV per lateral under ``dest_root``, mirroring relative paths from ``data_root``.

    Only columns present in both the merged frame and ``keep_columns`` are written
    (stable order follows ``keep_columns``). Returns the number of files written.

    For **tidytest** (or any split where lateral CSVs lack ``TVT``), pass
    ``allow_tvt_input_fallback=True`` so ``TVT_input`` backs the typewell join.
    """
    keep_list = list(keep_columns)
    dest_root.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for lateral_path in wcv.discover_lateral_csvs(data_root):
        merged = load_merged_with_alonghole_features(
            lateral_path,
            use_roll=use_roll,
            use_lag=use_lag,
            allow_tvt_input_fallback=allow_tvt_input_fallback,
        )
        if merged is None:
            continue
        rel = lateral_path.relative_to(data_root)
        out_path = dest_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in keep_list if c in merged.columns]
        merged.loc[:, cols].to_csv(out_path, index=False)
        n_written += 1
    return n_written


def save_prune_metadata(
    path: Path,
    *,
    result: PruneResult,
    threshold: float,
    use_roll: bool,
    use_lag: bool,
    data_root: str,
) -> None:
    """Write a small JSON sidecar (paths as strings) for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": threshold,
        "use_roll": use_roll,
        "use_lag": use_lag,
        "data_root": data_root,
        "n_dropped": len(result.dropped_columns),
        "dropped_columns": list(result.dropped_columns),
        "kept_columns": list(result.kept_columns),
        "protected_columns": list(result.protected_columns),
        "n_feature_candidates_initial": result.n_feature_candidates_initial,
        "n_feature_candidates_final": result.n_feature_candidates_final,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
