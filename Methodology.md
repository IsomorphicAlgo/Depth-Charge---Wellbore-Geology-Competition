# Methodology — process and documentation

This note explains why the repository is set up the way it is before there is any training code. As modeling begins, new sections will be appended for data handling, validation design, model families, and ensembling — each tied to a clear hypothesis and to leaderboard or offline metrics.

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
