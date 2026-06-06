"""
Exploratory pass over local competition files.

This script is meant to answer a practical question before modeling: is the data
already in a clean, consistent shape, or are there gaps, odd steps, and joins
that need tidying first?

Besides paired **horizontal** and **typewell** CSVs, the competition bundle can
include **PNG** assets in the same splits. Those are inventoried here. Pixel
data is not read unless Pillow is available, and then only for a tiny sample
(file size and image dimensions).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("This script needs pandas. Install with:  python -m pip install pandas")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"


def _print_intro() -> None:
    print(
        """
================================================================================
What this script does (plain language)
================================================================================
The competition gives you CSV files: one file per horizontal well, and a
matched typewell file for each. Train and test folders may also include PNG
sidecars (maps, log strips, etc.). The modeling goal is still to predict TVT
along the lateral in one-foot steps. Before choosing models, it helps to see
whether the files line up (same wells, steady depth steps, missing values where
you expect), whether PNG counts track with well IDs, and whether anything obvious
needs cleaning or reshaping.
================================================================================
""".strip()
    )


# Below is a quick inventory plus light checks: file counts (CSV + PNG), column
# types, missingness, depth step patterns, and a few sanity checks on TVT fields
# where labels exist.

def _candidate_roots() -> list[Path]:
    
    roots: list[Path] = []
    for p in (
        DATA_DIR,
        REPO_ROOT,
        DATA_DIR / "train",
        DATA_DIR / "test",
    ):
        if p.is_dir() and p not in roots:
            roots.append(p)
    return roots


def _glob_csvs(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        out.extend(sorted(root.rglob("*.csv")))
    # De-duplicate same file resolved differently
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        try:
            r = p.resolve()
        except OSError:
            r = p
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return uniq


def _glob_pngs(roots: list[Path]) -> list[Path]:
    """All PNG paths under candidate roots (case-sensitive FS)."""
    out: list[Path] = []
    for root in roots:
        out.extend(root.rglob("*.png"))
        out.extend(root.rglob("*.PNG"))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        try:
            r = p.resolve()
        except OSError:
            r = p
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return sorted(uniq, key=lambda x: str(x).lower())


def _png_lateral_stem(path: Path) -> str | None:
    """Lateral id for pairing: full basename without extension, or prefix before '__'."""
    if path.suffix.lower() != ".png":
        return None
    s = path.stem.strip()
    if not s:
        return None
    if "__" in s:
        return s.split("__", 1)[0]
    return s


def _inventory_pngs(paths: list[Path]) -> None:
    print("\n--- PNG inventory ---\n")
    if not paths:
        print("No PNG files found under searched folders.")
        return
    by_parent: dict[str, int] = defaultdict(int)
    for p in paths:
        try:
            rel_parent = str(p.relative_to(REPO_ROOT).parent)
        except ValueError:
            rel_parent = str(p.parent)
        by_parent[rel_parent] += 1
    print(f"Total PNGs: {len(paths)}")
    for parent in sorted(by_parent):
        print(f"  {parent}/  {by_parent[parent]}")
    print("\nSample paths (up to 12):")
    for p in paths[:12]:
        print(f"  {p.relative_to(REPO_ROOT)}")


def _summarize_pngs(
    paths: list[Path],
    horizontal_paths: list[Path],
    max_dim_samples: int = 3,
) -> None:
    print("\n--- PNGs vs lateral CSV stems ---\n")
    if not paths:
        return
    h_stems = {s for s in (_well_stem_horizontal(p) for p in horizontal_paths) if s}
    p_stems = {s for s in (_png_lateral_stem(p) for p in paths) if s}
    matched = h_stems & p_stems
    png_only = p_stems - h_stems
    h_only = h_stems - p_stems
    print(f"Unique lateral stems from horizontal CSVs: {len(h_stems)}")
    print(f"Unique keys from PNG basenames (stem, or text before '__'): {len(p_stems)}")
    print(f"Stems appearing in both: {len(matched)}")
    if png_only and len(png_only) <= 15:
        print(f"PNG stems with no horizontal CSV stem match (sample): {sorted(png_only)}")
    elif png_only:
        print(
            f"PNG stems with no horizontal CSV stem match: {len(png_only)} "
            f"(examples: {sorted(list(png_only))[:10]})"
        )
    if h_only:
        print(
            f"Lateral CSV stems with no PNG basename match: {len(h_only)} "
            "(images may use a different naming pattern or live elsewhere.)"
        )

    pillow = None
    try:
        from PIL import Image  # type: ignore[import-untyped]

        pillow = Image
    except ImportError:
        print(
            "\nOptional: install Pillow (`python -m pip install pillow`) to print "
            "width/height for a few PNG samples."
        )
        return

    print(f"\nPNG shape/size sample (first {max_dim_samples} files):")
    for p in paths[:max_dim_samples]:
        try:
            with pillow.open(p) as im:
                w, h = im.size
            nbytes = p.stat().st_size
            print(f"  {p.name}: {w}x{h} px, {nbytes:,} bytes")
        except OSError as e:
            print(f"  {p.name}: could not read ({e})")


def _classify_csv(path: Path) -> str:
    name = path.name.lower()
    if "horizontal_well" in name or "horizontal" in name and "well" in name:
        return "horizontal"
    if "typewell" in name:
        return "typewell"
    return "other"


def _well_stem_horizontal(path: Path) -> str | None:
    # Typical: Something__horizontal_well.csv
    m = re.match(r"(.+?)__horizontal_well\.csv$", path.name, re.IGNORECASE)
    return m.group(1) if m else None


def _well_stem_typewell(path: Path) -> str | None:
    # e.g. 000d7d20__typewell.csv  (also accepts longer ...__typewell__... names)
    m = re.match(r"(.+?)__typewell", path.name, re.IGNORECASE)
    return m.group(1) if m else None


def _missing_fraction(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().sort_values(ascending=False)


def _md_step_summary(series: pd.Series) -> dict[str, float | int | str]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "note": "no numeric MD"}
    diffs = s.diff().dropna()
    if diffs.empty:
        return {"n": int(s.shape[0]), "note": "single MD value"}
    uniq = diffs.round(6).value_counts()
    top = uniq.head(3)
    return {
        "n_rows": int(s.shape[0]),
        "md_min": float(s.min()),
        "md_max": float(s.max()),
        "diff_min": float(diffs.min()),
        "diff_max": float(diffs.max()),
        "diff_median": float(diffs.median()),
        "most_common_diffs": {float(k): int(v) for k, v in top.items()},
        "non_positive_steps": int((diffs <= 0).sum()),
    }


def _summarize_horizontal(paths: list[Path], max_files_sample: int = 5) -> None:
    print("\n--- Horizontal well CSVs ---\n")
    if not paths:
        print("No horizontal well CSVs found.")
        return

    stems = [_well_stem_horizontal(p) for p in paths]
    print(f"Count: {len(paths)}  (unique name stems: {len({s for s in stems if s})})")

    # only warn about multiple files within the same parent folder.
    by_stem_parent: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p, stem in zip(paths, stems):
        if stem:
            by_stem_parent[(stem, str(p.parent.name))].append(p.name)

    multi_same_folder = {k: v for k, v in by_stem_parent.items() if len(v) > 1}
    if multi_same_folder:
        print(
            f"Note: {len(multi_same_folder)} well id(s) have multiple horizontal "
            f"CSVs in the same folder (unexpected)."
        )

    for p in paths[:max_files_sample]:
        print(f"\nSample file: {p.relative_to(REPO_ROOT)}")
        try:
            df = pd.read_csv(p, nrows=5000)
        except Exception as e:  # pragma: no cover
            print(f"  Could not read: {e}")
            continue
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"  dtypes:\n{df.dtypes.to_string()}")
        miss = _missing_fraction(df)
        top_miss = miss[miss > 0].head(8)
        if top_miss.empty:
            print("  Missing values: none in sample head")
        else:
            print("  Missing fraction (top columns in sample head):")
            print(top_miss.to_string())

        if "MD" in df.columns:
            md_info = _md_step_summary(df["MD"])
            print(f"  MD step summary (sample rows): {md_info}")

        for col in ("TVT", "TVT_input"):
            if col in df.columns:
                t = pd.to_numeric(df[col], errors="coerce")
                print(
                    f"  {col}: min={t.min(skipna=True)} max={t.max(skipna=True)} "
                    f"non_null={t.notna().sum()}/{len(t)}"
                )

        if "TVT" in df.columns and "TVT_input" in df.columns:
            t_full = pd.to_numeric(df["TVT"], errors="coerce")
            t_in = pd.to_numeric(df["TVT_input"], errors="coerce")
            both = t_full.notna() & t_in.notna()
            if both.any():
                agree = (t_full[both] - t_in[both]).abs() < 1e-6
                print(
                    f"  TVT vs TVT_input (where both numeric): "
                    f"{int(agree.sum())}/{int(both.sum())} rows match within 1e-6"
                )


def _summarize_typewell(paths: list[Path], max_files_sample: int = 5) -> None:
    print("\n--- Typewell CSVs ---\n")
    if not paths:
        print("No typewell CSVs found.")
        return

    stems = [_well_stem_typewell(p) for p in paths]
    print(f"Count: {len(paths)}  (unique lateral stems from filenames: {len({s for s in stems if s})})")

    for p in paths[:max_files_sample]:
        print(f"\nSample file: {p.relative_to(REPO_ROOT)}")
        try:
            df = pd.read_csv(p, nrows=5000)
        except Exception as e:  # pragma: no cover
            print(f"  Could not read: {e}")
            continue
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        miss = _missing_fraction(df)
        top_miss = miss[miss > 0].head(8)
        if top_miss.empty:
            print("  Missing values: none in sample head")
        else:
            print("  Missing fraction (top columns in sample head):")
            print(top_miss.to_string())

        if "Geology" in df.columns:
            nunique = df["Geology"].nunique(dropna=True)
            print(f"  Geology: {nunique} distinct values in sample head")

        if "MD" in df.columns:
            print(f"  MD step summary (sample rows): {_md_step_summary(df['MD'])}")



def _pairing_report(horizontal: list[Path], typewell: list[Path]) -> None:
    print("\n--- Lateral / typewell pairing (from filenames) ---\n")
    h_stems = {s: p for s, p in zip([_well_stem_horizontal(x) for x in horizontal], horizontal) if s}
    t_stems = defaultdict(list)
    for p in typewell:
        s = _well_stem_typewell(p)
        if s:
            t_stems[s].append(p)

    if not h_stems and not t_stems:
        print("Not enough files to compare stems.")
        return

    h_only = sorted(set(h_stems) - set(t_stems))
    t_only = sorted(set(t_stems) - set(h_stems))
    paired = sorted(set(h_stems) & set(t_stems))

    print(f"Paired stems (appear in both): {len(paired)}")
    print(f"Lateral stem without typewell file: {len(h_only)}")
    if h_only[:10]:
        print(f"  examples: {h_only[:10]}")
    print(f"Typewell stem without lateral file: {len(t_only)}")
    if t_only[:10]:
        print(f"  examples: {t_only[:10]}")


def _inventory(all_csvs: list[Path], *, n_pngs: int = 0) -> None:
    print("\n--- File inventory ---\n")
    if not all_csvs:
        print("No CSV files found under searched folders.")
        if n_pngs:
            print(f"({n_pngs} PNG file(s) found; see PNG inventory below.)")
        print(f"Expected locations include: {DATA_DIR}, {DATA_DIR / 'train'}, {DATA_DIR / 'test'}")
        return
    by_root: dict[str, list[str]] = defaultdict(list)
    for p in all_csvs:
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            rel = p
        parent = str(rel.parent)
        by_root[parent].append(rel.name)
    for parent in sorted(by_root):
        names = sorted(by_root[parent])
        print(f"{parent}/  ({len(names)} csv)")
        for n in names[:20]:
            print(f"  {n}")
        if len(names) > 20:
            print(f"  ... and {len(names) - 20} more")


def main() -> None:
    _print_intro()
    roots = _candidate_roots()
    all_csvs = _glob_csvs(roots)
    all_pngs = _glob_pngs(roots)

    if not all_csvs and not all_pngs:
        print(
            "\nNo CSV or PNG files found. Unzip the Kaggle bundle under `data/train` "
            "and `data/test` (or under `train/` and `test/` at the repo root). "
            "See ReadMe.md.\n"
        )
        return

    _inventory(all_csvs, n_pngs=len(all_pngs))
    _inventory_pngs(all_pngs)

    horizontal = [p for p in all_csvs if _classify_csv(p) == "horizontal"]
    typewell = [p for p in all_csvs if _classify_csv(p) == "typewell"]
    other = [p for p in all_csvs if _classify_csv(p) == "other"]

    _summarize_horizontal(horizontal)
    _summarize_typewell(typewell)
    _pairing_report(horizontal, typewell)
    _summarize_pngs(all_pngs, horizontal)

    if other:
        print("\n--- Other CSVs (not classified as lateral or typewell) ---\n")
        for p in sorted(other)[:30]:
            print(f"  {p.relative_to(REPO_ROOT)}")
        if len(other) > 30:
            print(f"  ... and {len(other) - 30} more")

    print(
        "\n--- Interpretation hint ---\n"
        "If MD steps are not ~1 ft, or TVT/TVT_input disagree where both exist, "
        "or pairing counts look off, plan a tidying pass before heavy modeling. "
        "If PNG basenames and lateral CSV stems disagree, the images may use a different "
        "naming scheme than the basename / `hash__...` heuristic used here. "
        "If CSV counts are zero, unzip Kaggle data into `data/train` and `data/test` "
        "(see ReadMe.md).\n"
    )


if __name__ == "__main__":
    main()
