"""
Milestone 3 — distance along **typewell TVT** to the **next** geology label change.

Uses the same typewell **TVT** ordering and **one-row-per-TVT** rule as
:func:`wellbore_join.merge_lateral_typewell_schema_tvt` (``keep="last"`` on
duplicates). For each lateral foot whose ``TVT`` matches a typewell knot within
``tvt_match_atol``, ``tw_geology_next_change_ft`` is the forward distance in **ft**
(in TVT units) to the first knot **strictly ahead** in increasing TVT where the
normalized ``Geology`` string differs. Last segment / no forward change → **NaN**.

Rows without an exact knot match (or missing typewell ``Geology``) get **NaN**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TW_GEOLOGY_NEXT_CHANGE_FT_COL = "tw_geology_next_change_ft"


def milestone3_feature_column_names() -> tuple[str, ...]:
    """Column names added by :func:`add_tw_geology_next_change_ft`."""
    return (TW_GEOLOGY_NEXT_CHANGE_FT_COL,)


def _norm_geology_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text


def _typewell_tvt_knot_table(typewell_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique TVT knots (``keep="last"``) and aligned geology labels."""
    if "TVT" not in typewell_df.columns:
        raise KeyError("typewell_df must contain 'TVT'")
    tw = typewell_df.sort_values("TVT", kind="mergesort", ignore_index=True)
    tw = tw.drop_duplicates(subset=["TVT"], keep="last", ignore_index=True)
    tvt = tw["TVT"].to_numpy(dtype=float)
    if "Geology" not in tw.columns:
        labels = np.array([""] * len(tw), dtype=object)
    else:
        labels = np.array([_norm_geology_label(x) for x in tw["Geology"].to_numpy()], dtype=object)
    return tvt, labels


def _forward_next_change_ft_per_knot(tvt: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    For each knot index ``i``, return ``T[j] - T[i]`` where ``j`` is the smallest
    index ``> i`` with a different normalized label; else NaN.
    """
    n = len(tvt)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    run_end = np.arange(n, dtype=np.int64)
    for i in range(n - 2, -1, -1):
        if labels[i] == labels[i + 1]:
            run_end[i] = run_end[i + 1]
        else:
            run_end[i] = i

    for i in range(n):
        j = int(run_end[i]) + 1
        if j < n:
            out[i] = float(tvt[j] - tvt[i])
    return out


def _knot_index_for_lateral_tvt(
    tvt_knots: np.ndarray,
    lateral_tvt: np.ndarray,
    *,
    atol: float,
) -> np.ndarray:
    """Map each lateral ``TVT`` to the knot index in ``tvt_knots``, or ``-1``."""
    m = lateral_tvt.shape[0]
    out = np.full(m, -1, dtype=np.int64)
    for j in range(m):
        t = lateral_tvt[j]
        if not np.isfinite(t):
            continue
        il = int(np.searchsorted(tvt_knots, t, side="left"))
        for cand in (il, il - 1):
            if 0 <= cand < len(tvt_knots) and abs(float(tvt_knots[cand]) - float(t)) <= atol:
                out[j] = cand
                break
    return out


def add_tw_geology_next_change_ft(
    merged_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    *,
    lateral_tvt_col: str = "TVT",
    tvt_match_atol: float = 1e-4,
) -> pd.DataFrame:
    """
    Append ``tw_geology_next_change_ft`` to a **single-well** merged lateral+typewell frame.

    Expects ``merged_df`` to already carry ``tw_Geology`` from
    :func:`wellbore_join.attach_typewell_by_tvt`. The typewell frame must be the
    same one used for that join (same ``Geology`` / ``TVT`` source).

    If the typewell has no ``Geology`` column, writes **NaN** for every row.
    """
    out = merged_df.copy()
    if lateral_tvt_col not in out.columns:
        raise KeyError(f"merged_df missing {lateral_tvt_col!r}")

    tvt_knots, labels = _typewell_tvt_knot_table(typewell_df)
    if len(tvt_knots) == 0:
        out[TW_GEOLOGY_NEXT_CHANGE_FT_COL] = np.nan
        return out

    per_knot = _forward_next_change_ft_per_knot(tvt_knots, labels)
    lateral_tvt = pd.to_numeric(out[lateral_tvt_col], errors="coerce").to_numpy(dtype=float, copy=False)
    knot_idx = _knot_index_for_lateral_tvt(tvt_knots, lateral_tvt, atol=tvt_match_atol)
    dist = np.where(knot_idx >= 0, per_knot[knot_idx], np.nan)
    out[TW_GEOLOGY_NEXT_CHANGE_FT_COL] = dist
    return out
