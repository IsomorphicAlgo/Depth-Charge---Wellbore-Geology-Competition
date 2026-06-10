"""
Batch **join diagnostics** over many wells (Iteration 5 style).

Provides :func:`summarize_join_for_training_wells` for notebooks and a small CLI
for offline CSV export. Uses :mod:`wellbore_join` for the canonical merge and
:mod:`wellbore_join_diagnostics` for per-well scalar QA.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import wellbore_cv as wcv
from wellbore_join import attach_typewell_by_tvt, load_well_pair
from wellbore_join_diagnostics import lateral_typewell_join_diagnostics


def resolve_training_well_ids(
    *,
    manifest_path: Path | None = None,
    data_root: Path | None = None,
) -> tuple[list[str], str]:
    """
    Return sorted unique lateral well ids and a short provenance string.

    If ``manifest_path`` exists, well ids come from its ``well_id`` column;
    otherwise ids are discovered from ``data_root`` (default train tidy / train).
    """
    root = data_root if data_root is not None else wcv.default_train_root()
    if manifest_path is not None and manifest_path.is_file():
        manifest_df = pd.read_csv(manifest_path)
        ids = sorted(manifest_df["well_id"].astype(str).unique().tolist())
        note = f"manifest:{manifest_path.name}"
        return ids, note
    stems: set[str] = set()
    for csv_path in wcv.discover_lateral_csvs(root):
        stem = wcv.lateral_well_stem(csv_path)
        if stem is not None:
            stems.add(stem)
    ids = sorted(stems)
    note = "discover_lateral_csvs(no manifest)"
    return ids, note


def summarize_join_for_training_wells(
    data_root: Path,
    well_ids: Sequence[str],
    *,
    max_wells: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """
    For each well: load pair, run :func:`lateral_typewell_join_diagnostics`,
    :func:`attach_typewell_by_tvt`, and append ``tw_GR_na_frac`` / ``lat_tvt_overlap_frac``.

    Returns ``(summary_df, error_rows)`` where ``error_rows`` is a list of
    ``{"well_id", "error"}`` dicts for load/join failures.
    """
    to_run = list(well_ids) if max_wells is None else list(well_ids)[: int(max_wells)]
    diagnostic_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, str]] = []

    for well_id in to_run:
        try:
            lateral_df, typewell_df = load_well_pair(well_id, data_root=data_root)
        except FileNotFoundError as exc:
            error_rows.append({"well_id": well_id, "error": str(exc)})
            continue
        try:
            diagnostics = lateral_typewell_join_diagnostics(
                lateral_df, typewell_df, well_id=well_id
            )
            merged = attach_typewell_by_tvt(lateral_df, typewell_df, tvt_col="TVT")
            diagnostics["tw_GR_na_frac"] = float(merged["tw_GR"].isna().mean())
            lateral_row_count = int(diagnostics.get("lat_rows") or 0) or 1
            overlap_row_count = int(diagnostics.get("lat_tvt_overlap_rows") or 0)
            diagnostics["lat_tvt_overlap_frac"] = float(overlap_row_count / lateral_row_count)
            diagnostic_rows.append(diagnostics)
        except Exception as exc:  # noqa: BLE001 — batch QC: record and continue
            error_rows.append({"well_id": well_id, "error": repr(exc)})

    summary_df = pd.DataFrame(diagnostic_rows)
    if len(summary_df):
        summary_df = summary_df.sort_values("well_id").reset_index(drop=True)
    return summary_df, error_rows


def join_risk_heuristic_mask(summary_df: pd.DataFrame) -> pd.Series:
    """Boolean Series: True where Iteration 5-style join-risk heuristic fires."""
    if len(summary_df) == 0:
        return pd.Series(dtype=bool)
    return (
        (summary_df["tw_interp_finite_frac"] < 0.99)
        | (summary_df["lat_tvt_overlap_frac"] < 0.98)
        | (summary_df["tw_tvt_sorted_decreases"].fillna(0) > 0)
        | (summary_df["tw_dedup_dropped"].fillna(0) > 0)
    )


def plot_iter5_histograms(summary_df: pd.DataFrame) -> None:
    """Three histograms used in the V3 notebook (requires ``matplotlib``)."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].hist(summary_df["tw_interp_finite_frac"].dropna(), bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("tw_interp_finite_frac")
    axes[1].hist(summary_df["lat_tvt_overlap_frac"].dropna(), bins=30, color="seagreen", edgecolor="white")
    axes[1].set_title("lat_tvt_overlap_frac")
    axes[2].hist(summary_df["tw_GR_na_frac"].dropna(), bins=30, color="coral", edgecolor="white")
    axes[2].set_title("tw_GR_na_frac (exact merge)")
    plt.tight_layout()
    plt.show()


def _main() -> None:
    p = argparse.ArgumentParser(description="Batch lateral–typewell join diagnostics (read-only).")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current working directory).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder with tidied train CSVs (default: train_tidy if present else train under repo).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV (default: <repo>/cv_manifests/kfold5_well_folds.csv if present).",
    )
    p.add_argument("--max-wells", type=int, default=None, help="Cap number of wells for a quick run.")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write summary table to this path.",
    )
    args = p.parse_args()
    repo = args.repo_root.resolve()
    data_root = args.data_root
    if data_root is None:
        data_root = wcv.default_train_root()
    else:
        data_root = Path(data_root).resolve()

    manifest = args.manifest
    if manifest is None:
        candidate = repo / "cv_manifests" / "kfold5_well_folds.csv"
        manifest = candidate if candidate.is_file() else None
    elif not Path(manifest).is_file():
        manifest = None

    well_ids, note = resolve_training_well_ids(
        manifest_path=manifest,
        data_root=data_root,
    )
    print("well list:", note, "| n =", len(well_ids))
    print("data_root:", data_root)

    summary_df, errors = summarize_join_for_training_wells(
        data_root,
        well_ids,
        max_wells=args.max_wells,
    )
    print("wells summarized:", len(summary_df), "| errors:", len(errors))
    if args.output_csv is not None:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out, index=False)
        print("wrote:", out)


if __name__ == "__main__":
    _main()
