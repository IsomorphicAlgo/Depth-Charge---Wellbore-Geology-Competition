"""
Along-hole tabular features for lateral feet (Version 3+).

Rolling statistics and lag features are computed in **CSV row order after sorting
by MD** within each ``well_id`` group. With ~1 ft MD steps, ``windows_ft`` and
``lag_feet`` correspond to **approximately** that many feet of hole; see
``V3_plan.md``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_ROLL_WINDOWS_FT: tuple[int, ...] = (5, 10, 20)
DEFAULT_ROLL_VALUE_COLS: tuple[str, ...] = ("MD", "Z", "tw_GR_interp", "GR")
DEFAULT_ROLL_STATS: tuple[str, ...] = ("mean", "median", "grad")

# Milestone 2 — explicit lag distances (feet ≈ rows); ``Gate 2`` kept full set.
DEFAULT_LAG_FEET: tuple[int, ...] = (3, 5, 10)
DEFAULT_LAG_VALUE_COLS: tuple[str, ...] = ("MD", "Z", "tw_GR_interp", "GR")
DEFAULT_LAG_KINDS: tuple[str, ...] = ("lag", "diff", "ratio")


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


def lag_feature_column_names(
    *,
    lag_feet: tuple[int, ...] = DEFAULT_LAG_FEET,
    value_cols: tuple[str, ...] = DEFAULT_LAG_VALUE_COLS,
    kinds: tuple[str, ...] = DEFAULT_LAG_KINDS,
    available_cols: Iterable[str] | None = None,
) -> list[str]:
    """
    Column names for lag features from :func:`add_alonghole_lag_features`, optionally
    restricted to ``value_cols`` present in ``available_cols``.
    """
    cols = list(value_cols)
    if available_cols is not None:
        have = set(available_cols)
        cols = [c for c in cols if c in have]
    names: list[str] = []
    for col in cols:
        for ft in lag_feet:
            if ft < 1:
                continue
            for kind in kinds:
                if kind == "lag":
                    names.append(f"{col}_lag{ft}ft")
                elif kind == "diff":
                    names.append(f"{col}_diff_lag{ft}ft")
                elif kind == "ratio":
                    names.append(f"{col}_ratio_lag{ft}ft")
    return names


def add_alonghole_lag_features(
    df: pd.DataFrame,
    *,
    group_col: str = "well_id",
    md_col: str = "MD",
    lag_feet: tuple[int, ...] = DEFAULT_LAG_FEET,
    value_cols: tuple[str, ...] = DEFAULT_LAG_VALUE_COLS,
    kinds: tuple[str, ...] = DEFAULT_LAG_KINDS,
    ratio_abs_floor: float = 1e-6,
) -> pd.DataFrame:
    """
    Append **per-well** lagged values along hole after sorting by ``md_col``.

    For each integer lag ``k`` in ``lag_feet``, row ``i`` compares the current foot
    to the foot ``k`` rows **earlier** in MD order within the same ``group_col``
    (``~k`` ft when spacing is ~1 ft/row).

    **Kinds**

    - ``lag``: shifted series :math:`y_{t-k}` (pandas ``shift(k)`` within the well).
    - ``diff``: :math:`y_t - y_{t-k}`.
    - ``ratio``: :math:`y_t / y_{t-k}` with NaN when the lagged value is non-finite
      or ``|y_{t-k}| < ratio_abs_floor`` (avoids divide-by-zero blowups on GR).

    **Edge behavior:** the first ``k`` rows of each well have no predecessor; **lag,
    diff, and ratio are NaN** there (no zero-fill). Rows are **not** padded across
    well boundaries.
    """
    if group_col not in df.columns:
        raise KeyError(f"df missing group column {group_col!r} (required for per-well lags)")
    if md_col not in df.columns:
        raise KeyError(f"df missing MD column {md_col!r}")

    allowed = {"lag", "diff", "ratio"}
    bad_kinds = set(kinds) - allowed
    if bad_kinds:
        raise ValueError(f"kinds must be subset of {allowed!r}, got {bad_kinds!r}")

    out = df.sort_values([group_col, md_col], kind="mergesort").reset_index(drop=True)

    for col in value_cols:
        if col not in out.columns:
            continue
        y = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float, copy=False)
        g_series = pd.Series(y, index=out.index)
        gy = g_series.groupby(out[group_col], sort=False)

        for ft in lag_feet:
            k = int(ft)
            if k < 1:
                continue
            y_lag = gy.transform(lambda s, kk=k: s.shift(kk)).to_numpy(dtype=float, copy=False)

            if "lag" in kinds:
                out[f"{col}_lag{k}ft"] = y_lag

            if "diff" in kinds:
                out[f"{col}_diff_lag{k}ft"] = y - y_lag

            if "ratio" in kinds:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rat = y / y_lag
                bad = (~np.isfinite(y)) | (~np.isfinite(y_lag)) | (np.abs(y_lag) < ratio_abs_floor)
                out[f"{col}_ratio_lag{k}ft"] = np.where(bad, np.nan, rat)

    return out
