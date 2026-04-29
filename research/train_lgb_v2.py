"""LightGBM v2: pooled train (day 2+3) → test day 4. Try H=5, 20, 100.

If pooling and shorter horizon don't materially lift test IC, SNACKPACK alpha is
not exploitable with the available data and we pivot category.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache"
MODELS = ROOT / "models"
REPORTS = Path(__file__).resolve().parent / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

SNACKPACKS = [
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
]

FEATURE_COLS = [
    "spread", "micro_minus_mid",
    "queue_imb_l1", "queue_imb_l3",
    "ofi_5", "ofi_20", "ofi_100",
    "rv_50", "rv_200",
    "trade_flow_20",
    "cv_dev", "cv_dev_z",
    "psr_dev", "psr_dev_z",
]


def ic(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    if m.sum() < 100: return np.nan, np.nan
    pearson = float(np.corrcoef(p[m], t[m])[0, 1])
    rank, _ = spearmanr(p[m], t[m])
    return pearson, float(rank)


def prep(df):
    df = df.copy()
    df["micro_minus_mid"] = df["micro"] - df["mid"]
    return df


def train_one(df_all, leg, horizon):
    target = f"fwd_ret_{horizon}"
    df = df_all[df_all["product"] == leg].copy()

    train = df[df["day"].isin([2, 3])].dropna(subset=FEATURE_COLS + [target])
    test = df[df["day"] == 4].dropna(subset=FEATURE_COLS + [target])

    # Within-train CV via KFold (5 folds), use last fold as val for early stopping
    X_tr, y_tr = train[FEATURE_COLS].values, train[target].values
    X_te, y_te = test[FEATURE_COLS].values, test[target].values

    n = len(X_tr)
    val_start = int(0.85 * n)
    X_va, y_va = X_tr[val_start:], y_tr[val_start:]
    X_tr, y_tr = X_tr[:val_start], y_tr[:val_start]

    params = dict(
        objective="regression_l1",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
    )
    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
    val_set = lgb.Dataset(X_va, label=y_va, feature_name=FEATURE_COLS, reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=400,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(0)],
    )

    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    p_te = booster.predict(X_te, num_iteration=booster.best_iteration)

    return dict(
        leg=leg, horizon=horizon, best_iter=int(booster.best_iteration or 0),
        ic_val_pearson=ic(p_va, y_va)[0], ic_val_rank=ic(p_va, y_va)[1],
        ic_test_pearson=ic(p_te, y_te)[0], ic_test_rank=ic(p_te, y_te)[1],
        n_train=len(y_tr), n_test=len(y_te),
    )


def main():
    df = prep(pd.read_pickle(CACHE / "snackpack_features.pkl"))
    rows = []
    for H in (5, 20, 100):
        print(f"\n=== Horizon H={H} ===")
        for leg in SNACKPACKS:
            r = train_one(df, leg, H)
            rows.append(r)
            print(f"  {leg:25s}  iter={r['best_iter']:3d}  "
                  f"val IC={r['ic_val_pearson']:+.3f}  test IC={r['ic_test_pearson']:+.3f} "
                  f"(rank {r['ic_test_rank']:+.3f})")

    rep = pd.DataFrame(rows)
    rep.to_csv(REPORTS / "snackpack_phase1_v2_ic.csv", index=False)
    print("\n=== Test-set IC summary by horizon ===")
    for H in (5, 20, 100):
        sub = rep[rep["horizon"] == H]
        n_pass = int((sub["ic_test_pearson"].abs() >= 0.03).sum())
        print(f"  H={H:3d}:  {n_pass}/5 legs |test IC| >= 0.03  "
              f"(median |IC|={sub['ic_test_pearson'].abs().median():.3f}, "
              f"max={sub['ic_test_pearson'].abs().max():.3f})")


if __name__ == "__main__":
    main()
