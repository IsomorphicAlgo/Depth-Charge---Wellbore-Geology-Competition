# Methodology — process and documentation

This note explains why the repository is set up the way it is before there is much training code. As modeling picks up speed, new sections will be appended for validation design, model families, and ensembling — each tied to a clear hypothesis and to leaderboard or offline metrics.

## Task definition (from sponsor deck + CSV layout)

The English slide deck under `data/AI_wellbore_geology_prediction_task_en.pptx` is the friendly version of the problem statement: map views, 3D views, and a long walk through why **gamma ray (GR)** and **TVT** behave the way they do on a horizontal versus a vertical trace. The bullets below are the engineering distillation; if anything disagrees with Kaggle’s live **Data / Evaluation** pages, treat Kaggle as authoritative.

### Target and grid

Each horizontal well is evaluated on a **one-foot** TVT series along the lateral. Think of it as a dense depth index in measured depth (MD) space where every step needs a TVT answer, not a sparse pick list.

### Prediction start (PS) and `TVT_input`

The lateral file exposes **`TVT_input`**: TVT is revealed **only up to the prediction start (PS)**. Beyond PS, the model has to **continue** the TVT curve using whatever signal still exists — principally lateral **GR** and geometry (**MD**, **XYZ**), plus whatever can be borrowed from the paired typewell. Full **`TVT`** (manual reference) exists in **training** data for learning and offline checks; it is not something to “peek at” when simulating test inference.

### Files per well

- **Horizontal CSV** (`Well…__horizontal_well.csv`): MD, XYZ, GR (NaNs allowed), **`TVT_input`**, and in training the full **`TVT`** plus formation top depths for context.
- **Typewell CSV** (`Well…__typewell__Typewell….csv`): one assigned vertical analog per lateral, with **TVT**, **GR**, and **Geology** (layer names) on the vertical sampling.

Units in the materials are **feet**; keep that in mind when mixing in external literature that insists on meters.

### Metric

Where manual TVT is available, define **dTVT = manualTVT − predictedTVT** at each predicted foot. The leaderboard story is **RMSE over all those dTVT values** — a plain L2 on TVT error, no fancy weighting mentioned in the deck.

### How the sponsor suggests people think about it

The geology can be **flat or dipping**, and the **horizontal azimuth** relative to dip matters for what the lateral cuts through. Neighboring wells are waved in as qualitative support: dips tend to behave similarly in a neighborhood, so offset thinking is on the table if someone wants to engineer spatial features later.

On the signal side, the deck pushes a geosteering-flavored narrative: match the lateral **GR shape** (including whether TVT should be **increasing, decreasing, or flat** over a segment) to the **GR–TVT** relationship seen in the typewell. They also call out that **horizontal GR** can carry **finer resolution** along the path than a straight read of the typewell GR column, and they hint that **GR before PS** on the lateral may correlate more tightly with a high-resolution GR view than with the raw typewell trace alone — i.e., do not assume the vertical GR is the only fingerprint worth modeling.

None of that text replaces a real validation plan, but it does explain why a naive “concatenate both CSVs and feed a tabular model” might leave money on the table compared to something that respects **directionality** and **PS truncation**.

## Why documentation leads the repo

The project owner’s brief treats this work as a **learning exercise** first and a competition entry second. Leading with `ReadMe.md` and this file keeps goals, constraints, and compliance visible. It also matches how many strong Kaggle write-ups are structured: the story of the solution is as important as the code for reproducibility and for prize verification.

## Why some local files stay out of git

Certain working notes are meant to stay private to the author’s machine and chat sessions. Excluding them from version control keeps the public tree focused on shareable artifacts, avoids accidental leakage of half-baked strategy, and reduces noise for anyone browsing history later. The tracked docs deliberately do **not** point readers at those paths.

## Why `rules.md` lives in the tree

Having the competition rules text available offline speeds up double-checking submission limits, team rules, external-data policy, and winner obligations. Kaggle remains the source of truth if anything diverges.

## How modeling methodology will be recorded (once work starts)

When code and notebooks appear, this document will grow in a predictable pattern:

1. **Task framing** — exact prediction target, units, and any class imbalance or sequence structure.  
2. **Data hygiene** — what is never leaked from validation into training, and how folds or wells are grouped.  
3. **Baselines** — the simplest defensible model and its score; everything else is compared to that.  
4. **Experiments** — one paragraph per meaningful try: idea, change, metric, keep or discard.  
5. **Final submission** — what was selected, why, and what would be tried next with more time.

No step in that list will be implemented ahead of explicit owner approval; the brief treats that as a feature, not a delay.

---

*Last updated: June 2026*
