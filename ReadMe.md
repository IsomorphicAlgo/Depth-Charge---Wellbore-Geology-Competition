# Depth Charge — Wellbore Geology Prediction

This repository supports a learning-focused entry in the [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) competition on Kaggle. The aim is to grow practical data science and machine learning skills while contributing a serious, rule-compliant submission.

## What you will find here

- **`ReadMe.md`** — orientation and links (this file).
- **`Methodology.md`** — why the project is organized the way it is, and how modeling choices will be recorded as they land.
- **`rules.md`** — a local copy of the competition rules text for quick reference. The authoritative version remains on the [competition website](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).
- **`Wellbore Prompt.md`** — the project owner’s brief and constraints for collaborators and tooling.
- **`data/AI_wellbore_geology_prediction_task_en.pptx`** — sponsor task deck (maps, 3D context, GR–TVT intuition, and the scoring definition). Handy when someone wants the pictures, not just the CSV column list.

The repository is intentionally lightweight at the start: documentation and governance first, then notebooks, code, and reproducible pipelines once modeling work begins.

## Competition snapshot

- **Sponsor:** ROGII  
- **Theme:** Build models that help interpret geology along horizontal wellbores from operational and log-derived signals, supporting safer and more efficient drilling decisions.  

## What the model is actually predicting

In one sentence: along each **horizontal** well, the job is to recover **TVT** (the geology position on a true-vertical-thickness axis) at **one-foot** increments, including the lateral **after** a fixed **prediction start (PS)** where only partial TVT is given as input.

The competition ships each well as **two CSVs** — the lateral (`*_horizontal_well.csv`) and a paired vertical **typewell** (`*_typewell__*.csv`). Inputs lean on **MD**, **XYZ**, **GR** on the lateral, plus **GR** and **TVT** (and named geology) on the typewell. Training rows also carry the full manual TVT and formation tops so people can learn the mapping; scoring cares about how well TVT is continued past PS.

**Evaluation** (when labels exist) is RMSE on **dTVT = manualTVT − predictedTVT** across those foot-by-foot points. The sponsor deck spells out the same story with diagrams; `Methodology.md` goes a bit deeper on how they expect people to think about GR versus TVT.

## Getting started

1. Read the competition **Overview**, **Data**, and **Evaluation** tabs on Kaggle.  
2. Accept the competition rules on Kaggle and download the data into a **local** path that suits your machine. A typical layout is to unzip the competition bundle so **`train/`** and **`test/`** sit at the **repository root**; those two folder names are listed in `.gitignore` so raw splits are not committed by accident. If you keep data somewhere else, adjust paths or `.gitignore` accordingly.  
3. Skim `Methodology.md` for how decisions will be documented as the project evolves.

## Data and compliance

Competition data is for **competition use on Kaggle** only, per the sponsor rules. Do not redistribute raw competition files outside approved channels. For a full legal read, use Kaggle’s rules UI and `rules.md` here as a convenience copy.

Root-level **`train/`** and **`test/`** directories are ignored by git (see `.gitignore`). That keeps large, sponsor-controlled datasets out of history while still letting notebooks and scripts assume a predictable local layout.

## Contributing (including AI-assisted work)

Work proceeds in **small, approval-gated steps**: implement features, refactors, or experiments only when the project owner has asked for them explicitly. That keeps the learning curve legible and the repo history easy to audit.

---

*Last updated: June 2026*
