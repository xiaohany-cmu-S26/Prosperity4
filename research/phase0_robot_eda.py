"""Phase 0 EDA for ROBOT category (5 variants).

Key questions:
  1. Lag-1 (and lag-K) autocorrelation of mid-price RETURNS per leg.
     Bounce strategy works when AC(1) is strongly negative. The round5.py algo
     uses a fixed coef=0.20 on last_ret for ROBOT_DISHES; if other legs have
     similar AC, the fixed coef may be mis-calibrated rather than the strategy
     being wrong.
  2. Cross-leg correlation matrix of returns. Are ROBOTs basket-structured like
     SNACKPACKs (factor + residual), or independent?
  3. Cross-day stability of the lag-1 AC: if DISHES has stable AC across days
     but others don't, that's why DISHES is the earner.

Run: python research/phase0_robot_eda.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from loader import DAYS, ROBOTS, load_prices, pivot_mid


def main():
    print(f"Loading ROBOT prices for days {list(DAYS)} ...")
    prices = load_prices(DAYS, ROBOTS)
    mid = pivot_mid(prices, ROBOTS)
    print(f"  pivot shape: {mid.shape}")

    print("\n=== Pooled return correlation (5x5) ===")
    print(mid.pct_change().corr().round(3).to_string())

    print("\n=== Lag-K return autocorrelation per leg (pooled across days) ===")
    print(f"  Negative AC = bounce-style (round5.py uses coef=0.20 for DISHES)")
    rets = mid.diff()  # use raw diffs (ticks), not pct returns
    rows = []
    for leg in ROBOTS:
        r = rets[leg].dropna()
        rows.append({
            "leg": leg.replace("ROBOT_", ""),
            "AC1_pooled": float(r.autocorr(1)),
            "AC5": float(r.autocorr(5)),
            "AC20": float(r.autocorr(20)),
            "std": float(r.std()),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== Per-day lag-1 AC (regime stability) ===")
    rows = []
    for leg in ROBOTS:
        row = {"leg": leg.replace("ROBOT_", "")}
        for d in DAYS:
            r = rets[leg].xs(d, level="day").dropna()
            row[f"AC1_d{d}"] = float(r.autocorr(1))
            row[f"std_d{d}"] = float(r.std())
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== Forward 5-tick return predictability from last-1-tick return ===")
    print("    IC of (-last_diff) predicting fwd_5_diff. Bounce = positive IC.")
    print("    The round5.py strategy is FV -= coef * last_ret, equivalent to predicting")
    print("    fwd_ret to be -coef * last_ret. Per-day IC tells us whether the relationship")
    print("    is stable across regimes.")
    rows = []
    for leg in ROBOTS:
        row = {"leg": leg.replace("ROBOT_", "")}
        for d in DAYS:
            sub = mid[leg].xs(d, level="day")
            last = sub.diff()
            fwd5 = sub.shift(-5) - sub
            mask = last.notna() & fwd5.notna()
            ic = float(np.corrcoef(-last[mask].values, fwd5[mask].values)[0, 1])
            row[f"IC_d{d}_h5"] = ic
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== Same test for H=20 (matches round5.py horizon better) ===")
    rows = []
    for leg in ROBOTS:
        row = {"leg": leg.replace("ROBOT_", "")}
        for d in DAYS:
            sub = mid[leg].xs(d, level="day")
            last = sub.diff()
            fwd20 = sub.shift(-20) - sub
            mask = last.notna() & fwd20.notna()
            ic = float(np.corrcoef(-last[mask].values, fwd20[mask].values)[0, 1])
            row[f"IC_d{d}_h20"] = ic
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
