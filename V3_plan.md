# Version 3 plan — feature engineering and Kaggle submission

This document breaks **Version 3** into small milestones. Each milestone ends with an **approval gate**: the project owner confirms whether to proceed, adjust scope, or pause before the next chunk of work lands in **`wellbore_report V3.ipynb`** (and supporting modules, without altering raw **`data/train/`** or **`data/test/`**).

**Submission contract:** Kaggle expects a CSV with header **`id,tvt`**, matching **`data/sample_submission.csv`**. Each **`id`** is **`{well_id}_{foot_index}`** (see sample rows); **`tvt`** is the predicted value for that foot.

---

## Milestone 0 — Design lock (approval gate 0)

**Goal:** Agree on definitions so later features stay consistent with the 1 ft lateral index and fold-safe rules.

**Deliverables**

- List the **exact columns** used for rolling/lag math (**MD**, **Z**, **`tw_GR_interp`**, lateral **GR**, etc.) and whether operations are **per well** only (they should be).
- Confirm **window semantics**: e.g. backward-looking only past PS vs symmetric windows; whether **5 / 10 / 20 ft** means **5 / 10 / 20 rows** at 1 ft spacing (default assumption).
- For **distance to boundary**, define “change” as **first foot where `tw_Geology` differs from current** along **typewell TVT order** (or an alternative the owner prefers), and how **NaN** geology is handled.

**Gate 0 question for the owner:** Are the window direction, width in feet (= rows), and boundary definition above acceptable? (Yes)

---

## Milestone 0.5 — Notebook slimming: join API in Python (approval gate 0.5) — **expanded**

**Goal:** Move **stable, reusable** lateral–typewell join, **join QA**, **batch summaries**, and **geology encoding** out of **`wellbore_report V3.ipynb`** into importable modules.

**Shipped**

- **`wellbore_join.py`** — canonical merge + `load_well_pair` / `attach_typewell_by_tvt` (earlier step).
- **`wellbore_join_diagnostics.py`** — Iteration 2 diagnostics (`lateral_typewell_join_diagnostics`, `diagnostics_dataframe_for_well_ids`, `discover_sorted_well_ids`).
- **`wellbore_join_batch.py`** — Iteration 5 batch driver (`summarize_join_for_training_wells`, `resolve_training_well_ids`, `join_risk_heuristic_mask`, `plot_iter5_histograms`) plus CLI `python wellbore_join_batch.py`.
- **`wellbore_geology.py`** — `KNOWN_FORMATION_ORDER`, `build_geology_code_map`, `encode_geology_series`, `encode_merged_tw_geology` (used by geology demo cell and the LightGBM CV cell).

**Still reasonable in the notebook**

- Early **TVT / interp teaching** cells (Iteration 1, etc.) that build intuition; they can be trimmed later to call **`wellbore_join.merge_lateral_typewell_schema_tvt`** only.

**Gate 0.5 question for the owner:** Pause notebook extractions here, or continue by **trimming early teaching cells** to call **`wellbore_join`** / **`wellbore_join_diagnostics`** only (no duplicate interp math)?
**Gate 0.5 Answer**: Good here!

---

## Milestone 1 — Rolling statistics (approval gate 1) — **implemented**

**Goal:** Add **tabular** rolling **mean / median / gradient** (or slope vs MD) over **5, 10, 20 ft** on selected curves, computed **within each lateral well** on the merged foot grid.

**Shipped:** `wellbore_features.add_alonghole_roll_features` plus integration in **`wellbore_report V3.ipynb`** (LightGBM cell: `USE_V3_ROLL_FEATURES`). Toggle off to reproduce the pre-V3 feature set for ablations.

**Tuning / ablations**

- Defaults live in **`wellbore_features.py`** (`DEFAULT_ROLL_WINDOWS_FT`, `DEFAULT_ROLL_VALUE_COLS`, `DEFAULT_ROLL_STATS`); narrow there or set **`USE_V3_ROLL_FEATURES = False`** in the notebook to reproduce the pre-rolling baseline.

**Validation**

- Re-run CV with **feature list toggles**; compare **offline RMSE** on the existing **`VAL_METRIC_ROWS`** slice vs a **no-new-features** baseline.

**Tradeoffs (short)**

- **More windows × more stats:** higher effort and overfit risk; often the first mid-size window captures most lift.
- **Gradients:** useful for TVT-like curvature; slightly noisier than means unless smoothed.

**Gate 1 question:** After reviewing CV deltas and column list, proceed to lags as specified, or trim the rolling set first?

---

## Milestone 2 — Lag features (approval gate 2)

**Goal:** Compare each foot to **history along hole** (e.g. lateral **GR** at **current MD** vs **3 ft behind** along MD / foot index): differences, ratios, or z-scores **within well**.

**Deliverables**

- Explicit **lag in feet** (3 ft → 3 rows) with clear naming (e.g. **`gr_lag3`**, **`gr_diff_lag3`**).
- Document edge behavior at the **start of the lateral** (NaN vs fill policy).

**Gate 2 question:** Keep the proposed lag set, or cap at one lag distance until RMSE stabilizes?

---

## Milestone 3 — Distance to typewell geology boundary (approval gate 3)

**Goal:** For each lateral foot, engineer **distance (in typewell TVT or in feet along typewell index)** to the **next** geology label change on the paired typewell, using **`tw_Geology`** (and the same TVT ordering already used for **`attach_typewell_by_tvt`**).

**Risks**

- Label noise or single-foot flips can create spurious boundaries; optional **minimum run length** or merge of repeats may be needed.

**Gate 3 question:** Ship v1 as raw “feet to next label change,” or add a smoothing rule first?

---

## Milestone 4 — Test inference and submission CSV (approval gate 4)

**Goal:** **Inference-only** path on **`data/test/`** (or **`data/tidytest/`** if tidying stays the source) that outputs **`submission.csv`** (or similar) with:

- **Same row count and order** as **`data/sample_submission.csv`** (or a documented join to it),
- Columns exactly **`id`** and **`tvt`**, dtypes and formatting acceptable to Kaggle.

**Deliverables**

- A function or script section: **load test features → apply trained model(s) → map predictions to submission ids**.
- A **checksum** step: assert lengths match sample submission; optionally round **`tvt`** consistently if the platform is picky.

**Gate 4 question:** Is local **`sample_submission.csv`** verified row-for-row against a fresh Kaggle download before upload?

---

## Milestone 5 — Notebook and docs sync (approval gate 5)

**Goal:** **`wellbore_report V3.ipynb`** narrates what shipped, with **references** updated in **`ReadMe.md`** / **`Methodology.md`** as needed.

**Gate 5 question:** Close V3 at this point, or open a **V3.1** mini-plan (e.g. tuning after features)?

---

## How approvals are expected to work

After each milestone, the owner replies in chat with **pass / revise** and any edits. Implementation proceeds only for **passed** milestones (or for an explicitly revised subset).
