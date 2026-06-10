"""
Along-hole tabular features for lateral feet (Version 3+).

Rolling statistics are computed in **CSV row order after sorting by MD** within
each ``well_id`` group. With ~1 ft MD steps, ``windows_ft`` corresponds to
**approximately** that many feet of hole; see ``V3_plan.md``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_ROLL_WINDOWS_FT: tuple[int, ...] = (5, 10, 20)
DEFAULT_ROLL_VALUE_COLS: tuple[str, ...] = ("MD", "Z", "tw_GR_interp", "GR")
DEFAULT_ROLL_STATS: tuple[str, ...] = ("mean", "median", "grad")


def roll_feature_column_names(
    *,
    windows_ft: tuple[int, ...] = DEFAULT_ROLL_WINDOWS_FT,
    value_cols: tuple[str, ...] = DEFAULT_ROLL_VALUE_COLS,
    stats: tuple[str, ...] = DEFAULT_ROLL_STATS,
    available_cols: Iterable[str] | None = None,
) -> list[str]:
    """
    Return LightGBM-ready column names for rolling features that would be added
    for the given parameters, restricted to ``value_cols`` intersected with
    ``available_cols`` when the latter is provided.
    """
    cols = list(value_cols)
    if available_cols is not None:
        have = set(available_cols)
        cols = [c for c in cols if c in have]
    names: list[str] = []
    for col in cols:
        for w in windows_ft:
            for stat in stats:
                if stat == "grad" and w < 2:
                    continue
                names.append(f"{col}_roll{w}ft_{stat}")
    return names


def add_alonghole_roll_features(
    df: pd.DataFrame,
    *,
    group_col: str = "well_id",
    md_col: str = "MD",
    windows_ft: tuple[int, ...] = DEFAULT_ROLL_WINDOWS_FT,
    value_cols: tuple[str, ...] = DEFAULT_ROLL_VALUE_COLS,
    stats: tuple[str, ...] = DEFAULT_ROLL_STATS,
) -> pd.DataFrame:
    """
    Append backward-looking rolling **mean**, **median**, and **gradient** (``dy/dMD``)
    over the last ``w`` feet (rows) ending at each foot, separately per ``group_col``.

    Gradient for window size ``w`` uses the endpoints of the inclusive span
    ``[i - w + 1, i]``: ``(y[i] - y[i-w+1]) / (MD[i] - MD[i-w+1])``, with NaN when
    the MD span is ~0 or any endpoint ``y`` is non-finite.

    Rows must belong to a single lateral; ``group_col`` is required so stacked
    multi-well frames do not leak across well boundaries.
    """
    if group_col not in df.columns:
        raise KeyError(f"df missing group column {group_col!r} (required for per-well rolling)")
    if md_col not in df.columns:
        raise KeyError(f"df missing MD column {md_col!r}")

    out = df.sort_values([group_col, md_col], kind="mergesort").reset_index(drop=True)

    for col in value_cols:
        if col not in out.columns:
            continue
        y_roll = pd.to_numeric(out[col], errors="coerce")
        md_roll = pd.to_numeric(out[md_col], errors="coerce")
        g_y = y_roll.groupby(out[group_col], sort=False)
        g_md = md_roll.groupby(out[group_col], sort=False)

        for w in windows_ft:
            if w < 1:
                continue

            if "mean" in stats:
                rm = g_y.rolling(window=w, min_periods=1).mean()
                out[f"{col}_roll{w}ft_mean"] = rm.to_numpy(dtype=float)

            if "median" in stats:
                rmed = g_y.rolling(window=w, min_periods=1).median()
                out[f"{col}_roll{w}ft_median"] = rmed.to_numpy(dtype=float)

            if "grad" in stats and w >= 2:
                y_start = g_y.transform(lambda s: s.shift(w - 1))
                md_start = g_md.transform(lambda s: s.shift(w - 1))
                dy = y_roll.to_numpy(dtype=float) - y_start.to_numpy(dtype=float)
                dmd = md_roll.to_numpy(dtype=float) - md_start.to_numpy(dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    grad = dy / dmd
                bad = (~np.isfinite(dy)) | (~np.isfinite(dmd)) | (np.abs(dmd) < 1e-9)
                grad = np.where(bad, np.nan, grad)
                out[f"{col}_roll{w}ft_grad"] = grad

    return out
