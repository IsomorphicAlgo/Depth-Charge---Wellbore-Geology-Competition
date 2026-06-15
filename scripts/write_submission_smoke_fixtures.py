"""
Write minimal train/test tidy CSV pairs + a tiny sample_submission for
``wellbore_submission.run_submission_pipeline`` smoke testing (no competition data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO / "results" / "submission_smoke"


def _write_train_pair(dest: Path, wid: str, n: int) -> None:
    md = 10000.0 + np.arange(n, dtype=float)
    tvt = 8000.0 + np.arange(n, dtype=float)
    lat = pd.DataFrame(
        {
            "MD": md,
            "X": md * 0.01,
            "Y": md * 0.02,
            "Z": md * 0.03,
            "GR": 40.0 + (np.arange(n) % 11),
            "TVT_input": tvt + 0.1 * np.sin(np.arange(n)),
            "TVT": tvt + 0.05 * np.cos(np.arange(n)),
            "GR_was_null": np.zeros(n, dtype=np.int8),
        }
    )
    tw_tvt = np.concatenate([[tvt[0] - 20.0], tvt, [tvt[-1] + 20.0]])
    m = len(tw_tvt)
    half = m // 2
    geo = np.where(np.arange(m) < half, "FOOA", "FOOB")
    tw = pd.DataFrame(
        {
            "TVT": tw_tvt,
            "GR": 50.0 + np.arange(m, dtype=float) * 0.05,
            "Geology": geo,
        }
    )
    lat.to_csv(dest / f"{wid}__horizontal_well.csv", index=False)
    tw.to_csv(dest / f"{wid}__typewell.csv", index=False)


def _write_test_pair_no_geology(dest: Path, wid: str, n: int) -> None:
    md = 20000.0 + np.arange(n, dtype=float)
    tvt = 9000.0 + np.arange(n, dtype=float)
    lat = pd.DataFrame(
        {
            "MD": md,
            "X": md * 0.01,
            "Y": md * 0.02,
            "Z": md * 0.03,
            "GR": 35.0 + np.arange(n) % 8,
            "TVT_input": tvt + 0.2,
            "TVT": tvt,
            "GR_was_null": np.zeros(n, dtype=np.int8),
        }
    )
    tw_tvt = np.concatenate([[tvt[0] - 10.0], tvt, [tvt[-1] + 10.0]])
    tw = pd.DataFrame(
        {
            "TVT": tw_tvt,
            "GR": np.linspace(12.0, 88.0, len(tw_tvt)),
        }
    )
    lat.to_csv(dest / f"{wid}__horizontal_well.csv", index=False)
    tw.to_csv(dest / f"{wid}__typewell.csv", index=False)


def main() -> Path:
    train = SMOKE_ROOT / "train_tidy"
    test = SMOKE_ROOT / "test"
    train.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)

    _write_train_pair(train, "trainwlla", 160)
    _write_train_pair(train, "trainwllb", 140)
    _write_test_pair_no_geology(test, "testwllxx", 100)

    feet = list(range(8, 29))  # 21 contiguous feet; need foot_max <= n
    sample = pd.DataFrame(
        {"id": [f"testwllxx_{f}" for f in feet], "tvt": [0.0] * len(feet)}
    )
    sample_path = SMOKE_ROOT / "sample_smoke.csv"
    sample.to_csv(sample_path, index=False)
    print(f"Wrote smoke fixtures under {SMOKE_ROOT}")
    return sample_path


if __name__ == "__main__":
    main()
