"""Diagnose why per-leg LightGBM failed.

Hypothesis: per-leg fwd_ret is dominated by the (anti-correlated) factor between
legs. The structural alpha is the BASKET reversion, not per-leg direction.

Tests:
  1. IC of cv_dev predicting fwd_sum_CV (CHOC+VAN total move at H=20) — should be high.
  2. IC of cv_dev predicting fwd_diff_CV (CHOC-VAN move) — likely much lower.
  3. IC of psr_dev predicting fwd_sum_PSR.
  4. Cross-day stability: distribution of cv_dev across days 2/3/4.
  5. Feature importances from the failed per-leg model.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CACHE = Path(__file__).resolve().parent / "cache"
MODELS = Path(__file__).resolve().parents[1] / "models"

SNACKPACKS = [
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
]


def ic(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 100:
        return np.nan, np.nan
    pearson = float(np.corrcoef(p[m], t[m])[0, 1])
    rank, _ = spearmanr(p[m], t[m])
    return pearson, float(rank)


def main():
    df = pd.read_pickle(CACHE / "snackpack_features.pkl")

    # Pivot to wide so we can compute basket-level forward returns
    fwd = df.pivot_table(index=["day", "timestamp"], columns="product", values="fwd_ret_20")

    # Basket forward sums
    fwd_sum_cv = fwd["SNACKPACK_CHOCOLATE"] + fwd["SNACKPACK_VANILLA"]
    fwd_diff_cv = fwd["SNACKPACK_CHOCOLATE"] - fwd["SNACKPACK_VANILLA"]
    fwd_sum_psr = fwd["SNACKPACK_PISTACHIO"] + fwd["SNACKPACK_STRAWBERRY"] - 2 * fwd["SNACKPACK_RASPBERRY"]

    # Cross-leg features (same per timestamp, so take from any leg)
    cross = df[df["product"] == "SNACKPACK_CHOCOLATE"].set_index(["day", "timestamp"])[
        ["cv_dev", "cv_dev_z", "psr_dev", "psr_dev_z"]
    ]

    # Align
    cv_dev = cross["cv_dev"].reindex(fwd_sum_cv.index)
    psr_dev = cross["psr_dev"].reindex(fwd_sum_psr.index)

    print("=== Test 1: cv_dev predicting fwd_sum_CV (CHOC+VAN move at H=20) ===")
    print("    Hypothesis: basket sum is mean-reverting, so cv_dev should NEGATIVELY predict fwd_sum.")
    for d in (2, 3, 4):
        cv_d = cv_dev.xs(d, level="day")
        fwd_d = fwd_sum_cv.xs(d, level="day")
        p, r = ic(-cv_d.values, fwd_d.values)
        print(f"    day {d}:  IC(-cv_dev, fwd_sum_CV) = {p:+.3f} (rank {r:+.3f})  N={len(cv_d):,}")

    print("\n=== Test 2: cv_dev predicting fwd_diff_CV (CHOC-VAN move) ===")
    print("    Hypothesis: basket structure is symmetric, so cv_dev should NOT predict the spread.")
    for d in (2, 3, 4):
        cv_d = cv_dev.xs(d, level="day")
        fwd_d = fwd_diff_cv.xs(d, level="day")
        p, r = ic(cv_d.values, fwd_d.values)
        print(f"    day {d}:  IC(cv_dev, fwd_diff_CV)  = {p:+.3f} (rank {r:+.3f})")

    print("\n=== Test 3: psr_dev predicting fwd_sum_PSR (PIS+STR-2*RASP move at H=20) ===")
    for d in (2, 3, 4):
        psr_d = psr_dev.xs(d, level="day")
        fwd_d = fwd_sum_psr.xs(d, level="day")
        p, r = ic(-psr_d.values, fwd_d.values)
        print(f"    day {d}:  IC(-psr_dev, fwd_sum_PSR) = {p:+.3f} (rank {r:+.3f})")

    print("\n=== Test 4: cv_dev cross-day distribution stability ===")
    for d in (2, 3, 4):
        s = cv_dev.xs(d, level="day").dropna()
        print(f"    day {d}: mean={s.mean():+.3f}  std={s.std():.3f}  min={s.min():+.1f}  max={s.max():+.1f}")

    print("\n=== Test 5: feature importances from per-leg models ===")
    for leg in SNACKPACKS:
        path = MODELS / f"snackpack_phase1_{leg}_h20.txt"
        if not path.exists():
            continue
        booster = lgb.Booster(model_file=str(path))
        imp = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain")))
        top = sorted(imp.items(), key=lambda x: -x[1])[:5]
        print(f"    {leg:25s}  top features: {[(k, int(v)) for k, v in top]}")


if __name__ == "__main__":
    main()
