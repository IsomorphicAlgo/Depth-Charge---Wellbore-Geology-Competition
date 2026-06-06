# Depth Charge — Wellbore Geology Prediction

This repository supports a learning-focused entry in the [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) competition on Kaggle. The aim is to grow practical data science and machine learning skills while contributing a serious, rule-compliant submission.

## What you will find here

- **`ReadMe.md`** — orientation and links (this file).
- **`Methodology.md`** — why the project is organized the way it is, and how modeling choices will be recorded as they land.
- **`rules.md`** — a local copy of the competition rules text for quick reference. The authoritative version remains on the [competition website](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).
- **`Wellbore Prompt.md`** — the project owner’s brief and constraints for collaborators and tooling.
- **`data/AI_wellbore_geology_prediction_task_en.pptx`** — sponsor task deck (maps, 3D context, GR–TVT intuition, and the scoring definition). Handy when someone wants the pictures, not just the CSV column list.
- **`explore_data.py`** — lightweight EDA over local files: CSV inventory (horizontal, typewell, and any other CSVs such as `sample_submission.csv`), **PNG** counts under the same trees, a crude stem match between PNGs and lateral CSVs, plus dtypes, missingness, MD steps, and a few TVT checks. Optional Pillow for image dimensions on a few samples. Searches `data/`, **`data/train`** and **`data/test`**, and the repo root.
- **`wellbore_report.ipynb`** — working report and progress log (introduction, data analysis, tidy, model choice, trials, conclusion, references).

The repository is intentionally lightweight at the start: documentation and governance first, then notebooks, code, and reproducible pipelines once modeling work begins.

## Competition snapshot

- **Sponsor:** ROGII  
- **Theme:** Build models that help interpret geology along horizontal wellbores from operational and log-derived signals, supporting safer and more efficient drilling decisions.  

## What the model is actually predicting

In one sentence: along each **horizontal** well, the job is to recover **TVT** (the geology position on a true-vertical-thickness axis) at **one-foot** increments, including the lateral **after** a fixed **prediction start (PS)** where only partial TVT is given as input.

The competition ships each well as **two CSVs** — the lateral (`*_horizontal_well.csv`) and a paired vertical **typewell** (`*_typewell__*.csv`). Inputs lean on **MD**, **XYZ**, **GR** on the lateral, plus **GR** and **TVT** (and named geology) on the typewell. Training rows also carry the full manual TVT and formation tops so people can learn the mapping; scoring cares about how well TVT is continued past PS.

**Evaluation** (when labels exist) is RMSE on **dTVT = manualTVT − predictedTVT** across those foot-by-foot points. The sponsor deck spells out the same story with diagrams; `Methodology.md` goes a bit deeper on how they expect people to think about GR versus TVT.

## Data files and columns (Kaggle summary)

Below is a shortened version of the competition **Data** description. **WELLNAME** stands for the per-well id in filenames (in practice these look like opaque hashes). If anything drifts, the [Data tab](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data) on Kaggle stays authoritative.

### `{WELLNAME}__horizontal_well.csv` (lateral)

Trajectory, predicted formation surfaces (where provided), and logs along the horizontal. **MD** is measured depth (ft) along the borehole; **X** / **Y** are easting and northing (ft); **Z** is true vertical depth (ft) below sea level. **GR** is gamma ray (API). **TVT_input** is the lateral TVT trace exposed as an input feature; values are **NaN in the evaluation zone** (the stretch to be predicted). **TVT** is the manually interpreted geological position each foot along the lateral—the **target**—and is **training only**. Several columns (**ANCC**, **ASTNU**, **ASTNL**, **EGFDU**, **EGFDL**, **BUDA**) give predicted depths of named formations (**training only**).

### `{WELLNAME}__typewell.csv` (vertical reference)

Vertical log used to correlate with the lateral: **TVT** here is the vertical depth index (ft) aligned with the horizontal well’s geological TVT frame; **GR** is the vertical gamma ray used for correlation; **Geology** is the formation label (categorical, e.g. EGFDL, BUDA).

### `{WELLNAME}.png` (optional visual)

Static visualization of the well path and geological cross-section for that well id (often present under **train**; layout may vary by split).

### `test/` split (evaluation)

Roughly **200** evaluation wells. Each has the same two CSV types: **horizontal_well** (trajectory and logs, with **TVT** masked as **NaN** in the evaluation zone) and **typewell** (vertical reference). There is no labeled TVT to train on in that zone—models predict it for submission.

## Getting started

1. Read the competition **Overview**, **Data**, and **Evaluation** tabs on Kaggle.  
2. Accept the competition rules on Kaggle and download the data into this project. A layout that matches the exploratory script is **`data/train/`** and **`data/test/`** (CSV pairs per well under each). Some competitors keep **`train/`** and **`test/`** at the repository root instead; `explore_data.py` still picks those up when it walks from the repo root. Whatever layout you choose, remember competition data is sponsor-controlled: do not push it to a public remote unless you are comfortable with that exposure and with repo size.  
3. Skim `Methodology.md` for how decisions will be documented as the project evolves.

## Background reading

- [Geosteering & LWD overview — ScienceDirect Topics](https://www.sciencedirect.com/topics/engineering/logging-while-drilling)
- [SPE paper: Geosteering using Gamma Ray (OnePetro)](https://onepetro.org/SPEATCE/proceedings/25ATCE/25ATCE/D021S015R008/792004)

## Data and compliance

Competition data is for **competition use on Kaggle** only, per the sponsor rules. Do not redistribute raw competition files outside approved channels. For a full legal read, use Kaggle’s rules UI and `rules.md` here as a convenience copy.

This checkout is configured so **`train/`** and **`test/`** splits are **not** hidden by `.gitignore`; whether those directories live under `data/` or at the root is up to the owner. If the tree is tracked in git, keep an eye on binary weight, accidental secrets, and the sponsor’s terms.

## Contributing (including AI-assisted work)

Work proceeds in **small, approval-gated steps**: implement features, refactors, or experiments only when the project owner has asked for them explicitly. That keeps the learning curve legible and the repo history easy to audit.

---

*Last updated: June 2026*
