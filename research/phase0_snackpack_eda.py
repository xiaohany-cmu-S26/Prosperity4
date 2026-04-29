"""Phase 0 sanity check: confirm SNACKPACK is the right pilot category.

Outputs:
  1. 5x5 cross-correlation matrix of SNACKPACK mid-price returns (per day + pooled).
  2. Pair-sum (CHOC + VAN) deviation autocorrelation (the algo already exploits this).
  3. Take-side adverse selection: conditional on a market trade lifting an ask
     (or hitting a bid), measure the distribution of mid-price moves over the
     next H ticks. If the post-trade move is systematically against the taker,
     that's the failure mode the model needs to fix.

Run: python research/phase0_snackpack_eda.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from loader import DAYS, SNACKPACKS, load_prices, load_trades, pivot_mid


def returns_corr(mid: pd.DataFrame) -> pd.DataFrame:
    return mid.pct_change().corr()


def per_day_corr(mid: pd.DataFrame) -> dict:
    out = {}
    for d in mid.index.get_level_values("day").unique():
        sub = mid.xs(d, level="day")
        out[int(d)] = sub.pct_change().corr()
    return out


def pair_sum_dev_autocorr(mid: pd.DataFrame, lag: int = 1) -> float:
    """Autocorrelation of (CHOC + VAN) - EMA. round5.py exploits this with SP_PAIR_ALPHA=0.01."""
    s = mid["SNACKPACK_CHOCOLATE"] + mid["SNACKPACK_VANILLA"]
    ema = s.ewm(alpha=0.01, adjust=False).mean()
    dev = s - ema
    return float(dev.autocorr(lag=lag))


def take_side_adverse_selection(prices: pd.DataFrame, trades: pd.DataFrame, horizons=(5, 20, 100)) -> pd.DataFrame:
    """Per-product, conditional on a trade @ price near best ask (lift) vs best bid (hit),
    compute mean and median of (mid[t+H] - mid[t]) for each H.

    A negative mean for "lift_ask" + positive mean for "hit_bid" = adverse selection
    (you take and the market moves against you).
    """
    rows = []
    for prod in SNACKPACKS:
        p = prices[prices["product"] == prod].set_index(["day", "timestamp"]).sort_index()
        if p.empty:
            continue
        t = trades[trades["symbol"] == prod].copy()
        if t.empty:
            continue
        # Align trades to nearest snapshot timestamp on same day
        for d in DAYS:
            tp = p.xs(d, level="day", drop_level=False)
            tt = t[t["day"] == d]
            if tp.empty or tt.empty:
                continue
            mid = tp["mid_price"].droplevel("day")
            best_bid = tp["bid_price_1"].droplevel("day")
            best_ask = tp["ask_price_1"].droplevel("day")
            # For each trade, find nearest snapshot ts <= trade ts
            tt_sorted = tt.sort_values("timestamp")
            snap_ts = mid.index.values
            idx = np.searchsorted(snap_ts, tt_sorted["timestamp"].values, side="right") - 1
            valid = idx >= 0
            for H in horizons:
                idx_fwd = np.minimum(idx + H, len(snap_ts) - 1)
                m_now = mid.values[idx[valid]]
                m_fwd = mid.values[idx_fwd[valid]]
                tp_prices = tt_sorted["price"].values[valid]
                bid_now = best_bid.values[idx[valid]]
                ask_now = best_ask.values[idx[valid]]
                # Classify trade direction by Lee-Ready (price vs mid)
                lift_ask = tp_prices >= ask_now  # taker bought aggressively
                hit_bid = tp_prices <= bid_now   # taker sold aggressively
                move = m_fwd - m_now
                rows.append(dict(
                    product=prod, day=d, H=H,
                    n_lift=int(lift_ask.sum()),
                    mean_move_after_lift=float(move[lift_ask].mean()) if lift_ask.any() else np.nan,
                    n_hit=int(hit_bid.sum()),
                    mean_move_after_hit=float(move[hit_bid].mean()) if hit_bid.any() else np.nan,
                ))
    return pd.DataFrame(rows)


def main():
    print("Loading SNACKPACK prices/trades for days", list(DAYS), "...")
    prices = load_prices(DAYS, SNACKPACKS)
    trades = load_trades(DAYS, SNACKPACKS)
    print(f"  prices: {len(prices):,} rows  trades: {len(trades):,} rows")

    mid = pivot_mid(prices, SNACKPACKS)
    print(f"  mid shape (after pivot): {mid.shape}")

    print("\n=== Pooled return correlation (5x5) ===")
    print(returns_corr(mid).round(3).to_string())

    print("\n=== Per-day return correlation (off-diagonal medians) ===")
    for d, c in per_day_corr(mid).items():
        offdiag = c.values[np.triu_indices_from(c.values, k=1)]
        print(f"  day {d}: median={np.median(offdiag):+.3f}  min={offdiag.min():+.3f}  max={offdiag.max():+.3f}")

    print("\n=== CHOC+VAN pair-sum deviation autocorr (round5.py exploits this) ===")
    pooled = mid.dropna()
    print(f"  lag-1 autocorr of (sum - EMA(alpha=0.01)): {pair_sum_dev_autocorr(pooled, 1):+.3f}")
    print(f"  lag-5 autocorr:                            {pair_sum_dev_autocorr(pooled, 5):+.3f}")
    print(f"  lag-20 autocorr:                           {pair_sum_dev_autocorr(pooled, 20):+.3f}")

    print("\n=== Take-side adverse selection per leg ===")
    print("  Negative mean_move_after_lift = market falls after takers lift = adverse selection on buys")
    print("  Positive mean_move_after_hit  = market rises after takers hit  = adverse selection on sells")
    adv = take_side_adverse_selection(prices, trades)
    if not adv.empty:
        agg = adv.groupby(["product", "H"]).agg(
            mean_after_lift=("mean_move_after_lift", "mean"),
            mean_after_hit=("mean_move_after_hit", "mean"),
            n_lift=("n_lift", "sum"),
            n_hit=("n_hit", "sum"),
        ).round(3)
        print(agg.to_string())


if __name__ == "__main__":
    main()
