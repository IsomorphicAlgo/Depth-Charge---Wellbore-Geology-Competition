"""
K-fold split **by lateral well id** for local validation.

Each competition lateral is one `*__horizontal_well.csv`; the well id is the
filename stem before `__horizontal_well`. All MD rows for that file stay in the
same fold so metrics are not leaked across train/val.

Folds are built by sorting well ids (stable), shuffling with a fixed RNG, then
round-robin assignment to folds `0 .. n_splits-1` so fold sizes differ by at
most one well.

Default data root is `data/train_tidy/` when present, else `data/train/`.
Manifests are written under `cv_manifests/` at the repo root (small CSVs, safe
to commit for reproducibility).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
# Matches Kaggle-style lateral filenames: `<well_id>__horizontal_well.csv`.
RE_LATERAL = re.compile(r"(.+?)__horizontal_well\.csv$", re.IGNORECASE)

# Default output for `python wellbore_cv.py` with no `--out` flag.
DEFAULT_MANIFEST = REPO_ROOT / "cv_manifests" / "kfold4_well_folds.csv"


def lateral_well_stem(csv_path: Path) -> str | None:
    """Return the well id string from a lateral CSV filename, or None if the name does not match."""
    m = RE_LATERAL.match(csv_path.name)
    return m.group(1) if m else None


def discover_lateral_csvs(data_root: Path) -> list[Path]:
    """All `*__horizontal_well.csv` paths under ``data_root`` (deduped, sorted)."""
    if not data_root.is_dir():
        return []
    # Resolve paths so the same file reached via different spellings counts once.
    return sorted(
        {p.resolve(): p for p in data_root.rglob("*__horizontal_well.csv")}.values()
    )


def default_train_root() -> Path:
    """Prefer cleaned training data if that folder exists; otherwise use raw `data/train`."""
    tidy = REPO_ROOT / "data" / "train_tidy"
    raw = REPO_ROOT / "data" / "train"
    if tidy.is_dir():
        return tidy
    return raw


def default_test_root() -> Path:
    """Prefer ``data/tidytest`` when it actually contains laterals; else ``data/test``."""
    tidy = REPO_ROOT / "data" / "tidytest"
    raw = REPO_ROOT / "data" / "test"
    if tidy.is_dir() and discover_lateral_csvs(tidy):
        return tidy
    return raw


def build_well_fold_manifest(
    data_root: Path,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    One row per lateral well: ``well_id``, ``fold`` (0 .. n_splits-1),
    ``horizontal_relative`` (path relative to ``data_root``).

    Wells are shuffled with ``random_state``, then folds are assigned in a
    repeating 0..K-1 pattern down the list so each fold holds roughly the same
    number of wells (sizes differ by at most one).
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    lateral_paths = discover_lateral_csvs(data_root)
    if not lateral_paths:
        raise FileNotFoundError(f"No lateral CSVs under: {data_root}")

    # Build one manifest row per lateral file, then collapse duplicate well ids.
    rows: list[dict[str, str]] = []
    for p in lateral_paths:
        stem = lateral_well_stem(p)
        if stem is None:
            continue
        rows.append(
            {
                "well_id": stem,
                # Always use forward slashes so manifests look the same on Windows and Linux.
                "horizontal_relative": str(p.relative_to(data_root)).replace(
                    "\\", "/"
                ),
            }
        )

    df = pd.DataFrame(rows)
    # One row per well id (first path if duplicates ever appear).
    df = df.drop_duplicates(subset=["well_id"], keep="first")

    n = len(df)
    # Shuffle row order with a fixed seed so the same inputs always yield the same folds.
    order = np.random.default_rng(random_state).permutation(n)
    # Round-robin fold ids 0..n_splits-1 along the shuffled list → nearly balanced fold sizes.
    fold_ids = np.arange(n) % n_splits
    df = df.iloc[order].reset_index(drop=True)
    df["fold"] = fold_ids.astype(np.int8)
    # Sort again by well_id for a stable, human-readable CSV; fold ids stay attached to each well.
    df = df.sort_values("well_id").reset_index(drop=True)
    df.insert(1, "fold", df.pop("fold"))
    return df


def load_manifest(path: Path | str) -> pd.DataFrame:
    """Load a manifest CSV produced by :func:`write_manifest` (columns: well_id, fold, horizontal_relative)."""
    return pd.read_csv(path)


def train_val_well_ids(manifest: pd.DataFrame, val_fold: int) -> tuple[set[str], set[str]]:
    """Return ``(train_well_ids, val_well_ids)`` for a single fold held out for validation."""
    if val_fold not in set(manifest["fold"].unique()):
        raise ValueError(f"val_fold {val_fold} not in manifest folds")
    # All wells in val_fold form the validation set; every other well stays in training.
    val = set(manifest.loc[manifest["fold"] == val_fold, "well_id"])
    train = set(manifest["well_id"]) - val
    return train, val


def write_manifest(
    out_path: Path,
    data_root: Path | None = None,
    *,
    n_splits: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build the manifest and write it to ``out_path``; returns the same DataFrame."""
    root = data_root or default_train_root()
    df = build_well_fold_manifest(root, n_splits=n_splits, random_state=random_state)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def _main() -> None:
    p = argparse.ArgumentParser(description="Build grouped-by-well k-fold manifest.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder containing training laterals (default: train_tidy if present else train).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Output CSV path.",
    )
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    # CLI entry point: writes the CSV and prints a short fold-size summary.
    root = args.data_root or default_train_root()
    df = write_manifest(
        args.out,
        root,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    print(f"Wrote {args.out} with {len(df)} wells from {root.resolve()}")
    print(df.groupby("fold")["well_id"].count().rename("n_wells").to_string())


if __name__ == "__main__":
    _main()
