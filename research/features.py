"""SNACKPACK feature pipeline. Mirrors round5.py's `_book` so offline features
match the live-algo view at every snapshot.

Per-leg features (computed for each of the 5 SNACKPACKs):
  mid, micro, spread, queue_imb_l1, queue_imb_l3,
  ofi_5/20/100, signed_trade_vol_20, rv_50/200

Cross-leg features (computed once per timestamp):
  pair_sum_cv = CHOCOLATE_mid + VANILLA_mid
  pair_dev_cv = pair_sum_cv - EMA(0.01)
  pair_dev_cv_z = (pair_dev_cv - rolling_mean) / rolling_std

  basket_psr = PISTACHIO_mid + STRAWBERRY_mid - RASPBERRY_mid * 2  (centered)
  basket_psr_dev = basket_psr - EMA(0.01)

Forward returns:
  fwd_ret_H for H in {5, 20, 100}: mid[t+H] - mid[t]   (in ticks, NOT %)

Output: parquet/feather of the joined frame, one row per (day, timestamp, product).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from loader import DAYS, ROUND5_DIR, SNACKPACKS, load_prices, load_trades

OUT_DIR = Path(__file__).resolve().parent / "cache"
OUT_DIR.mkdir(exist_ok=True)


# ─────────────── Per-leg book features ──────────────────────────────────────

def _book_features(g: pd.DataFrame) -> pd.DataFrame:
    """g is a single (day, product) group, sorted by timestamp.

    Mirrors round5.py:_book for mid, micro, spread, but adds rolling/diff features
    that the live algo doesn't compute.
    """
    out = pd.DataFrame(index=g.index)
    bid1 = g["bid_price_1"].astype(float)
    ask1 = g["ask_price_1"].astype(float)
    bvol1 = g["bid_volume_1"].astype(float)
    avol1 = g["ask_volume_1"].astype(float)

    out["mid"] = (bid1 + ask1) / 2.0
    out["spread"] = ask1 - bid1
    # micro = (ask*bid_vol + bid*ask_vol) / (bid_vol + ask_vol)
    denom = (bvol1 + avol1).replace(0, np.nan)
    out["micro"] = (ask1 * bvol1 + bid1 * avol1) / denom
    out["queue_imb_l1"] = (bvol1 - avol1) / denom

    # L3 queue imbalance (sum across all 3 levels). Levels 2/3 may be NaN.
    bid_total = bvol1.add(g["bid_volume_2"].fillna(0), fill_value=0).add(g["bid_volume_3"].fillna(0), fill_value=0)
    ask_total = avol1.add(g["ask_volume_2"].fillna(0), fill_value=0).add(g["ask_volume_3"].fillna(0), fill_value=0)
    denom3 = (bid_total + ask_total).replace(0, np.nan)
    out["queue_imb_l3"] = (bid_total - ask_total) / denom3

    # Order-flow imbalance (Cont): ΔBidVol_1 if bid_price_1 unchanged, else
    # use the standard sign-flip. Approximate with simple diff at L1.
    bid_diff = bid1.diff()
    ask_diff = ask1.diff()
    bvol_diff = bvol1.diff()
    avol_diff = avol1.diff()
    # OFI: positive bid pressure adds, positive ask pressure subtracts
    e_b = pd.Series(0.0, index=g.index)
    e_b[bid_diff > 0] = bvol1[bid_diff > 0]
    e_b[bid_diff == 0] = bvol_diff[bid_diff == 0]
    e_b[bid_diff < 0] = -bvol1.shift(1)[bid_diff < 0]

    e_a = pd.Series(0.0, index=g.index)
    e_a[ask_diff < 0] = avol1[ask_diff < 0]
    e_a[ask_diff == 0] = avol_diff[ask_diff == 0]
    e_a[ask_diff > 0] = -avol1.shift(1)[ask_diff > 0]

    ofi = (e_b - e_a).fillna(0.0)
    out["ofi_5"] = ofi.rolling(5, min_periods=1).sum()
    out["ofi_20"] = ofi.rolling(20, min_periods=1).sum()
    out["ofi_100"] = ofi.rolling(100, min_periods=1).sum()

    # Realized vol over last K ticks (sum of |mid_diff|)
    mid_diff_abs = out["mid"].diff().abs()
    out["rv_50"] = mid_diff_abs.rolling(50, min_periods=1).sum()
    out["rv_200"] = mid_diff_abs.rolling(200, min_periods=1).sum()

    return out


# ─────────────── Trade-flow signed volume bucketed to snapshot grid ──────────

def _trade_flow(prices_pivot_mid: pd.Series, trades: pd.DataFrame) -> pd.Series:
    """Lee-Ready: sign of trade = +1 if price > mid_at_trade else -1.
    Assigns each trade to the snapshot timestamp <= trade.timestamp.
    Returns rolling sum over last 20 snapshots, indexed like prices_pivot_mid."""
    if trades.empty:
        return pd.Series(0.0, index=prices_pivot_mid.index)
    ts = prices_pivot_mid.index.get_level_values("timestamp").values
    # Per-day searchsorted
    out = np.zeros(len(prices_pivot_mid))
    days = prices_pivot_mid.index.get_level_values("day").values
    for d in np.unique(days):
        day_mask = days == d
        ts_d = ts[day_mask]
        offset = day_mask.argmax()  # index of first row in this day
        td = trades[trades["day"] == d]
        if td.empty:
            continue
        td_ts = td["timestamp"].values
        td_q = td["quantity"].astype(float).values
        td_p = td["price"].astype(float).values
        # Snapshot index for each trade (within day)
        idx = np.searchsorted(ts_d, td_ts, side="right") - 1
        valid = idx >= 0
        # Mid at snapshot
        mid_d = prices_pivot_mid.values[day_mask]
        sign = np.where(td_p[valid] > mid_d[idx[valid]], 1.0, -1.0)
        # Accumulate signed volume into snapshot bucket
        np.add.at(out, offset + idx[valid], sign * td_q[valid])
    s = pd.Series(out, index=prices_pivot_mid.index)
    return s.rolling(20, min_periods=1).sum()


# ─────────────── Cross-leg features ─────────────────────────────────────────

def _cross_leg_features(mid_wide: pd.DataFrame) -> pd.DataFrame:
    """mid_wide: index=(day,timestamp), columns=SNACKPACK_*."""
    out = pd.DataFrame(index=mid_wide.index)
    cv_sum = mid_wide["SNACKPACK_CHOCOLATE"] + mid_wide["SNACKPACK_VANILLA"]
    out["cv_sum"] = cv_sum
    cv_ema = cv_sum.ewm(alpha=0.01, adjust=False).mean()
    out["cv_dev"] = cv_sum - cv_ema
    out["cv_dev_z"] = out["cv_dev"] / out["cv_dev"].rolling(2000, min_periods=200).std()

    # PIS+STR vs 2*RASP basket residual
    psr = mid_wide["SNACKPACK_PISTACHIO"] + mid_wide["SNACKPACK_STRAWBERRY"] - 2 * mid_wide["SNACKPACK_RASPBERRY"]
    out["psr"] = psr
    psr_ema = psr.ewm(alpha=0.01, adjust=False).mean()
    out["psr_dev"] = psr - psr_ema
    out["psr_dev_z"] = out["psr_dev"] / out["psr_dev"].rolling(2000, min_periods=200).std()
    return out


# ─────────────── Master pipeline ────────────────────────────────────────────

def build_features(horizons=(5, 20, 100)) -> pd.DataFrame:
    print("Loading SNACKPACK prices/trades ...")
    prices = load_prices(DAYS, SNACKPACKS)
    trades = load_trades(DAYS, SNACKPACKS)

    # Per-leg book features
    print("Computing per-leg book features ...")
    pieces = []
    for (day, prod), g in prices.groupby(["day", "product"], sort=False):
        g_sorted = g.sort_values("timestamp")
        feats = _book_features(g_sorted)
        feats["day"] = day
        feats["product"] = prod
        feats["timestamp"] = g_sorted["timestamp"].values
        feats["mid_raw"] = g_sorted["mid_price"].astype(float).values
        pieces.append(feats)
    long = pd.concat(pieces, ignore_index=True)

    # Wide pivot for cross-leg
    print("Computing cross-leg features ...")
    mid_wide = long.set_index(["day", "timestamp", "product"])["mid"].unstack("product")
    cross = _cross_leg_features(mid_wide)

    # Trade-flow signed volume per leg
    print("Computing trade-flow features ...")
    flow = {}
    for prod in SNACKPACKS:
        if prod in mid_wide.columns:
            flow[prod] = _trade_flow(mid_wide[prod].dropna(), trades[trades["symbol"] == prod])

    # Forward returns per leg (per day; never cross day boundaries)
    print("Computing forward returns ...")
    fwd_returns = {}
    for prod in SNACKPACKS:
        if prod not in mid_wide.columns:
            continue
        per_day = []
        for d in DAYS:
            sub = mid_wide[prod].xs(d, level="day")
            for H in horizons:
                col = f"fwd_ret_{H}"
                fr = sub.shift(-H) - sub
                per_day.append(pd.DataFrame({col: fr}).assign(day=d, product=prod, timestamp=sub.index))
        # Merge horizons
        merged = None
        for H in horizons:
            df_H = pd.concat(
                [pd.DataFrame({f"fwd_ret_{H}": (mid_wide[prod].xs(d, level="day").shift(-H)
                                                - mid_wide[prod].xs(d, level="day"))})
                 .assign(day=d, product=prod, timestamp=mid_wide[prod].xs(d, level="day").index)
                 for d in DAYS], ignore_index=True)
            merged = df_H if merged is None else merged.merge(df_H, on=["day", "product", "timestamp"], how="outer")
        fwd_returns[prod] = merged

    fwd_long = pd.concat(fwd_returns.values(), ignore_index=True)

    # Join everything
    print("Joining ...")
    long = long.merge(fwd_long, on=["day", "product", "timestamp"], how="left")
    cross_long = cross.reset_index()
    long = long.merge(cross_long, on=["day", "timestamp"], how="left")

    # Add per-leg signed trade flow
    flow_pieces = []
    for prod, s in flow.items():
        df = s.reset_index()
        df.columns = ["day", "timestamp", "trade_flow_20"]
        df["product"] = prod
        flow_pieces.append(df)
    if flow_pieces:
        flow_df = pd.concat(flow_pieces, ignore_index=True)
        long = long.merge(flow_df, on=["day", "product", "timestamp"], how="left")
    else:
        long["trade_flow_20"] = 0.0

    long["trade_flow_20"] = long["trade_flow_20"].fillna(0.0)

    print(f"Final shape: {long.shape}")
    print("Columns:", list(long.columns))
    return long


def main():
    df = build_features()
    out_path = OUT_DIR / "snackpack_features.pkl"
    df.reset_index(drop=True).to_pickle(out_path)
    print(f"Wrote {out_path}")
    # Quick sanity print
    print("\nSample (first 3 rows of CHOCOLATE):")
    print(df[df["product"] == "SNACKPACK_CHOCOLATE"].head(3).to_string())
    print("\nNaN counts (first 12 cols):")
    print(df.isna().sum().head(12).to_string())


if __name__ == "__main__":
    main()
