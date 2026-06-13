import json
from pathlib import Path

p = Path(__file__).resolve().parent / "wellbore_report V3_pruned.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
cell = nb["cells"][58]
src = "".join(cell["source"])
old = """    def load_merged_from_train_tidy(well_id: str, lateral_rel: str) -> pd.DataFrame | None:
        \"\"\"Load tidied pair, attach typewell, add ``well_id`` + ``TVT_input_ffill``.\"\"\"
        lateral_path = COMPETITION_TRAIN_TIDY / lateral_rel
        if not lateral_path.is_file():
            print(\"missing lateral:\", lateral_path.relative_to(REPO_ROOT))
            return None
        typewell_path = lateral_path.with_name(
            lateral_path.name.replace(\"__horizontal_well.csv\", \"__typewell.csv\")
        )
        if not typewell_path.is_file():
            print(\"missing typewell:\", typewell_path.relative_to(REPO_ROOT))
            return None

        lateral_df = _read_numeric_horizon(pd.read_csv(lateral_path))
        typewell_df = _read_numeric_typewell(pd.read_csv(typewell_path))
        merged = attach_typewell_by_tvt(lateral_df, typewell_df)
        merged[\"well_id\"] = well_id

        merged = merged.sort_values(\"MD\", kind=\"mergesort\").reset_index(drop=True)
        if \"TVT_input\" in merged.columns:
            merged[\"TVT_input_ffill\"] = merged[\"TVT_input\"].ffill().bfill()
        else:
            merged[\"TVT_input_ffill\"] = np.nan

        if USE_V3_ROLL_FEATURES:
            merged = wbfeat.add_alonghole_roll_features(merged)
        if USE_V3_LAG_FEATURES:
            merged = wbfeat.add_alonghole_lag_features(merged)

        return merged
"""
new = """    def load_merged_from_train_tidy(well_id: str, lateral_rel: str) -> pd.DataFrame | None:
        \"\"\"Load tidied pair, attach typewell, add ``well_id`` + ``TVT_input_ffill`` — or read a pre-merged export.\"\"\"
        lateral_path = COMPETITION_TRAIN_TIDY / lateral_rel
        if USE_TABULAR_PRUNED_TRAIN:
            import wellbore_feature_pruning as _wbprune

            return _wbprune.load_premerged_lateral_csv(lateral_path, well_id=well_id)
        if not lateral_path.is_file():
            print(\"missing lateral:\", lateral_path.relative_to(REPO_ROOT))
            return None
        typewell_path = lateral_path.with_name(
            lateral_path.name.replace(\"__horizontal_well.csv\", \"__typewell.csv\")
        )
        if not typewell_path.is_file():
            print(\"missing typewell:\", typewell_path.relative_to(REPO_ROOT))
            return None

        lateral_df = _read_numeric_horizon(pd.read_csv(lateral_path))
        typewell_df = _read_numeric_typewell(pd.read_csv(typewell_path))
        merged = attach_typewell_by_tvt(lateral_df, typewell_df)
        merged[\"well_id\"] = well_id

        merged = merged.sort_values(\"MD\", kind=\"mergesort\").reset_index(drop=True)
        if \"TVT_input\" in merged.columns:
            merged[\"TVT_input_ffill\"] = merged[\"TVT_input\"].ffill().bfill()
        else:
            merged[\"TVT_input_ffill\"] = np.nan

        if USE_V3_ROLL_FEATURES:
            merged = wbfeat.add_alonghole_roll_features(merged)
        if USE_V3_LAG_FEATURES:
            merged = wbfeat.add_alonghole_lag_features(merged)

        return merged
"""
if old not in src:
    raise SystemExit("OLD BLOCK NOT FOUND")
src2 = src.replace(old, new, 1)
needle = "    import wellbore_features as wbfeat\n\n    # Milestone 1 (V3):"
ins = """    import wellbore_features as wbfeat

    # Pre-merged ``tabular_pruned/train`` exports already include roll/lag columns.
    USE_V3_ROLL_FEATURES = (not USE_TABULAR_PRUNED_TRAIN) and True
    USE_V3_LAG_FEATURES = (not USE_TABULAR_PRUNED_TRAIN) and True

    # Milestone 1 (V3):"""
if needle not in src2:
    raise SystemExit("NEEDLE NOT FOUND for roll/lag")
src2 = src2.replace(needle, ins, 1)
lines = src2.splitlines(keepends=True)
if not lines[-1].endswith("\n"):
    lines[-1] += "\n"
cell["source"] = lines
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("patched", len(lines), "lines")
