"""
Fold-safe **typewell geology** string → integer codes for tree models.

Moved from **`wellbore_report V3.ipynb`**. CV code should rebuild
:func:`build_geology_code_map` on **training-fold** labels only, then encode
both train and validation with that map.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Optional stratigraphic priority — names listed first receive smaller codes if present.
KNOWN_FORMATION_ORDER: tuple[str, ...] = (
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
)


def _sorted_unique_strings(values: Iterable[object]) -> list[str]:
    """Collect unique, non-empty string labels (skips NaN / blank)."""
    seen: set[str] = set()
    for raw in values:
        if pd.isna(raw):
            continue
        text = str(raw).strip()
        if text:
            seen.add(text)
    return sorted(seen)


def build_geology_code_map(
    observed_labels: Iterable[object],
    *,
    priority_order: tuple[str, ...] = KNOWN_FORMATION_ORDER,
) -> dict[str, int]:
    """Return label → int code (starts at 1). Code ``0`` is reserved for unknown."""
    present = set(_sorted_unique_strings(observed_labels))
    ordered: list[str] = []
    for name in priority_order:
        if name in present:
            ordered.append(name)
            present.remove(name)
    ordered.extend(sorted(present))
    return {label: idx for idx, label in enumerate(ordered, start=1)}


def encode_geology_series(
    series: pd.Series,
    code_map: dict[str, int],
    *,
    unknown_code: int = 0,
) -> pd.Series:
    """Map ``series`` through ``code_map``; missing / unseen labels → ``unknown_code``."""
    as_str = series.astype("string")
    coded = as_str.map(code_map).fillna(unknown_code).astype(np.int32)
    blank = as_str.str.len().fillna(0) == 0
    coded = coded.where(~blank, unknown_code)
    return coded


def encode_merged_tw_geology(
    merged: pd.DataFrame,
    code_map: dict[str, int],
    *,
    source_col: str = "tw_Geology",
    target_col: str = "tw_geology_code",
) -> pd.DataFrame:
    """Return a copy of ``merged`` with ``target_col`` added from ``source_col``."""
    if source_col not in merged.columns:
        raise KeyError(
            f"merged frame missing {source_col!r} — typewell may lack a Geology column."
        )
    out = merged.copy()
    out[target_col] = encode_geology_series(out[source_col], code_map)
    return out
