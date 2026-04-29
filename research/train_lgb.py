"""Phase 1 LightGBM trainer for SNACKPACK forward-return prediction.

Train one model per (leg, horizon=20). Day 2 = train, Day 3 = val (early stop),
Day 4 = held-out test (used only for the final IC report and the backtest A/B).

Output:
  models/snackpack_phase1_<LEG>_h20.txt  (LightGBM Booster)
  research/reports/snackpack_phase1_ic.csv

Run: python research/train_lgb.py
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache"
MODELS = ROOT / "models"
REPORTS = Path(__file__).resolve().parent / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

SNACKPACKS = [
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_RASPBERRY",
]

# Features used at inference time. mid/mid_raw deliberately excluded (level, non-stationary).
FEATURE_COLS = [
    "spread",
    "micro_minus_mid",
    "queue_imb_l1",
    "queue_imb_l3",
    "ofi_5",
    "ofi_20",
    "ofi_100",
    "rv_50",
    "rv_200",
    "trade_flow_20",
    "cv_dev",
    "cv_dev_z",
    "psr_dev",
    "psr_dev_z",
]

HORIZON = 20
TARGET = f"fwd_ret_{HORIZON}"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["micro_minus_mid"] = df["micro"] - df["mid"]
    return df


def ic(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Pearson and Spearman IC between prediction and target."""
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 100:
        return np.nan, np.nan
    p = pred[mask]
    t = true[mask]
    pearson = float(np.corrcoef(p, t)[0, 1])
    rank, _ = spearmanr(p, t)
    return pearson, float(rank)


def train_one_leg(df_all: pd.DataFrame, leg: str) -> dict:
    df = df_all[df_all["product"] == leg].copy()

    train = df[df["day"] == 2].dropna(subset=FEATURE_COLS + [TARGET])
    val = df[df["day"] == 3].dropna(subset=FEATURE_COLS + [TARGET])
    test = df[df["day"] == 4].dropna(subset=FEATURE_COLS + [TARGET])

    X_tr, y_tr = train[FEATURE_COLS].values, train[TARGET].values
    X_va, y_va = val[FEATURE_COLS].values, val[TARGET].values
    X_te, y_te = test[FEATURE_COLS].values, test[TARGET].values

    params = dict(
        objective="regression_l1",  # Huber-like; robust to fat-tailed return outliers
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
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(0)],
    )

    p_tr = booster.predict(X_tr, num_iteration=booster.best_iteration)
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    p_te = booster.predict(X_te, num_iteration=booster.best_iteration)

    ic_tr = ic(p_tr, y_tr)
    ic_va = ic(p_va, y_va)
    ic_te = ic(p_te, y_te)

    booster.save_model(str(MODELS / f"snackpack_phase1_{leg}_h{HORIZON}.txt"), num_iteration=booster.best_iteration)

    return dict(
        leg=leg,
        best_iter=int(booster.best_iteration or 0),
        ic_train_pearson=ic_tr[0], ic_train_rank=ic_tr[1],
        ic_val_pearson=ic_va[0], ic_val_rank=ic_va[1],
        ic_test_pearson=ic_te[0], ic_test_rank=ic_te[1],
        n_train=len(y_tr), n_val=len(y_va), n_test=len(y_te),
    )


def main():
    print("Loading features ...")
    df = pd.read_pickle(CACHE / "snackpack_features.pkl")
    df = prep(df)
    print(f"  total rows: {len(df):,}")

    rows = []
    for leg in SNACKPACKS:
        print(f"\n--- Training {leg} (h={HORIZON}) ---")
        r = train_one_leg(df, leg)
        rows.append(r)
        print(f"  best_iter={r['best_iter']}  "
              f"IC train={r['ic_train_pearson']:+.3f} (rank {r['ic_train_rank']:+.3f})  "
              f"val={r['ic_val_pearson']:+.3f} (rank {r['ic_val_rank']:+.3f})  "
              f"test={r['ic_test_pearson']:+.3f} (rank {r['ic_test_rank']:+.3f})")

    report = pd.DataFrame(rows)
    out = REPORTS / "snackpack_phase1_ic.csv"
    report.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("\n=== IC Summary ===")
    print(report[["leg", "ic_val_pearson", "ic_val_rank", "ic_test_pearson", "ic_test_rank"]].to_string(index=False))

    # Kill criterion: 3 of 5 legs with val IC >= 0.03 at H=20
    val_ics = report["ic_val_pearson"].abs()
    n_pass = int((val_ics >= 0.03).sum())
    print(f"\nKill-criterion check: {n_pass}/5 legs have |val IC| >= 0.03 at H={HORIZON}")
    if n_pass >= 3:
        print("  PASS — proceed to backtest A/B")
    else:
        print("  FAIL — stop, write up negative finding")


if __name__ == "__main__":
    main()
