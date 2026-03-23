"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-factor direction model for TOMATOES.

Fixes in this version
─────────────────────
1) F1 OBI and F5 DEPTH_ASYM are differentiated:
   - F1 = L1 imbalance (best level only)
   - F5 = full-depth asymmetry (all 3 levels)

2) F3 TFI uses a true tick-based rolling window:
   - recent 20 ticks, not "timestamp minus 20"

3) Trade side inference is improved:
   - infer buy/sell aggressor from trade price vs current best bid / ask / mid

Output (per tick)
─────────────────
  direction_score  ∈ [-1, +1]
    +1  = strong bullish
    -1  = strong bearish
     0  = neutral
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from fileinput import filename
import os

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import deque


# ── Tunable Parameters ────────────────────────────────────────────────────────

FACTOR_WEIGHTS = {
    "obi":         0.30,   # F1 — best-level imbalance
    "vwmid_mom":   0.20,   # F2 — weighted-mid trend
    "tfi":         0.25,   # F3 — trade flow imbalance
    "spread_z":   -0.05,   # F4 — wide spread = uncertainty / noise
    "depth_asym":  0.10,   # F5 — full-depth imbalance
    "price_mom":   0.10,   # F6 — mid-price EMA crossover
}

WINDOWS = {
    "obi_depth":       3,   # kept for flexibility; F5 uses all available depth
    "mom_fast":       10,
    "mom_slow":       30,
    "tfi_window":     20,   # TRUE tick window
    "spread_window":  50,
}


# ── Data Containers ───────────────────────────────────────────────────────────

@dataclass
class OBLevel:
    price: float
    volume: float


@dataclass
class OBSnapshot:
    timestamp: int
    bids: list[OBLevel]   # sorted best first
    asks: list[OBLevel]   # sorted best first
    mid: float


@dataclass
class Trade:
    timestamp: int
    price: float
    quantity: float
    side: str             # "buy" | "sell" | "unknown"


# ── Factor Engine ─────────────────────────────────────────────────────────────

class FactorEngine:
    """
    Stateful engine: call update(ob, trades) every tick.
    Reads .direction_score after each update.
    """

    def __init__(self):
        self._wmid_history: deque = deque(maxlen=WINDOWS["mom_slow"] + 5)
        self._mid_history: deque = deque(maxlen=WINDOWS["mom_slow"] + 5)
        self._spread_history: deque = deque(maxlen=WINDOWS["spread_window"] + 5)

        # FIX #1: true tick-based TFI window
        # store one signed volume per tick
        self._tfi_tick_window: deque = deque(maxlen=WINDOWS["tfi_window"])

        self.factors: dict = {}
        self.direction_score: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, ob: OBSnapshot, trades: list[Trade]) -> float:
        """
        Process one tick. Returns direction_score in [-1, +1].
        trades: all trades whose timestamp == ob.timestamp
        """
        wmid = self._weighted_mid(ob)
        self._wmid_history.append(wmid)
        self._mid_history.append(ob.mid)

        spread = self._spread(ob)
        self._spread_history.append(spread)

        # aggregate current tick trade flow into one signed volume
        self._add_trades_one_tick(trades)

        f1 = self._f_obi(ob)
        f2 = self._f_vwmid_mom()
        f3 = self._f_tfi()
        f4 = self._f_spread_z()
        f5 = self._f_depth_asym(ob)
        f6 = self._f_price_mom()

        self.factors = {
            "obi":        f1,
            "vwmid_mom":  f2,
            "tfi":        f3,
            "spread_z":   f4,
            "depth_asym": f5,
            "price_mom":  f6,
        }

        score = sum(FACTOR_WEIGHTS[k] * v for k, v in self.factors.items())
        self.direction_score = float(np.clip(score, -1.0, 1.0))
        return self.direction_score

    # ── Factor Definitions ────────────────────────────────────────────────────

    def _f_obi(self, ob: OBSnapshot) -> float:
        """
        F1 — L1 Order Book Imbalance
        Use best bid / best ask only.

        OBI = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1)
        Range: [-1, +1]
        """
        if not ob.bids or not ob.asks:
            return 0.0

        bid_vol = ob.bids[0].volume
        ask_vol = ob.asks[0].volume
        total = bid_vol + ask_vol

        if total == 0:
            return 0.0

        return float((bid_vol - ask_vol) / total)

    def _f_vwmid_mom(self) -> float:
        """
        F2 — Volume-Weighted Mid Momentum
        Fast vs slow mean difference on weighted mid, normalised by recent std.
        Range: tanh-squashed to (-1, +1)
        """
        if len(self._wmid_history) < WINDOWS["mom_fast"] + 1:
            return 0.0

        arr = np.array(self._wmid_history, dtype=float)
        fast = arr[-WINDOWS["mom_fast"]:].mean()
        slow = arr[-WINDOWS["mom_slow"]:].mean() if len(arr) >= WINDOWS["mom_slow"] else arr.mean()
        std = arr.std() + 1e-9

        return float(np.tanh((fast - slow) / std))

    def _f_tfi(self) -> float:
        """
        F3 — Trade Flow Imbalance
        TRUE recent-20-tick net signed volume / total signed volume.

        Range: [-1, +1]
        """
        if not self._tfi_tick_window:
            return 0.0

        net_vol = sum(self._tfi_tick_window)
        total_vol = sum(abs(v) for v in self._tfi_tick_window)

        if total_vol == 0:
            return 0.0

        return float(np.clip(net_vol / total_vol, -1.0, 1.0))

    def _f_spread_z(self) -> float:
        """
        F4 — Spread Z-Score
        High spread = uncertainty / noise.
        Range: tanh-squashed to (-1, +1)
        """
        if len(self._spread_history) < 5:
            return 0.0

        arr = np.array(self._spread_history, dtype=float)
        mean = arr.mean()
        std = arr.std() + 1e-9
        z = (arr[-1] - mean) / std

        return float(np.tanh(z))

    def _f_depth_asym(self, ob: OBSnapshot) -> float:
        """
        F5 — Total Depth Asymmetry
        Use all available depth (up to 3 levels).

        Range: [-1, +1]
        """
        bid_vol = sum(lvl.volume for lvl in ob.bids)
        ask_vol = sum(lvl.volume for lvl in ob.asks)
        total = bid_vol + ask_vol

        if total == 0:
            return 0.0

        return float((bid_vol - ask_vol) / total)

    def _f_price_mom(self) -> float:
        """
        F6 — Mid-Price EMA Crossover
        Fast EMA vs slow EMA on plain mid.
        Range: tanh-squashed to (-1, +1)
        """
        if len(self._mid_history) < WINDOWS["mom_fast"] + 1:
            return 0.0

        arr = np.array(self._mid_history, dtype=float)
        fast = self._ema(arr, WINDOWS["mom_fast"])
        slow = self._ema(arr, min(WINDOWS["mom_slow"], len(arr)))
        std = arr.std() + 1e-9

        return float(np.tanh((fast - slow) / std))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _weighted_mid(self, ob: OBSnapshot) -> float:
        """
        Micro-price using best-level volume.
        """
        if not ob.bids or not ob.asks:
            return ob.mid

        bv = ob.bids[0].volume
        av = ob.asks[0].volume
        bp = ob.bids[0].price
        ap = ob.asks[0].price
        total = bv + av

        if total == 0:
            return (bp + ap) / 2.0

        return (ap * bv + bp * av) / total

    def _spread(self, ob: OBSnapshot) -> float:
        if ob.bids and ob.asks:
            return ob.asks[0].price - ob.bids[0].price
        return 0.0

    def _ema(self, arr: np.ndarray, span: int) -> float:
        alpha = 2.0 / (span + 1)
        result = arr[0]
        for v in arr[1:]:
            result = alpha * v + (1 - alpha) * result
        return float(result)

    def _add_trades_one_tick(self, trades: list[Trade]):
        """
        Aggregate all trades in this tick into one signed volume,
        then append to a TRUE tick-count rolling window.
        """
        signed_vol = 0.0

        for t in trades:
            if t.side == "buy":
                signed_vol += t.quantity
            elif t.side == "sell":
                signed_vol -= t.quantity

        self._tfi_tick_window.append(signed_vol)


# ── Data Loaders ──────────────────────────────────────────────────────────────

def load_order_book(path: str, product: str = "TOMATOES") -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df[df["product"] == product].copy()

    # Robust sort: some files may or may not have a 'day' column
    sort_cols = ["timestamp"]
    if "day" in df.columns:
        sort_cols = ["day", "timestamp"]

    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def load_trades(path: str, product: str = "TOMATOES") -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df[df["symbol"] == product].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def parse_ob_row(row) -> OBSnapshot:
    bids, asks = [], []

    for i in range(1, 4):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and float(bv) > 0:
            bids.append(OBLevel(price=float(bp), volume=float(bv)))

        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(ap) and pd.notna(av) and float(av) > 0:
            asks.append(OBLevel(price=float(ap), volume=float(av)))

    return OBSnapshot(
        timestamp=int(row["timestamp"]),
        bids=bids,
        asks=asks,
        mid=float(row["mid_price"]),
    )


def infer_trade_side(price: float, ob: OBSnapshot) -> str:
    """
    Infer aggressor side from current order book snapshot.

    Rules:
      - if trade price >= best ask  -> buy aggressive
      - if trade price <= best bid  -> sell aggressive
      - else compare to mid:
          price > mid -> buy
          price < mid -> sell
          price == mid -> unknown
    """
    if not ob.bids or not ob.asks:
        return "unknown"

    best_bid = ob.bids[0].price
    best_ask = ob.asks[0].price
    mid = ob.mid

    if price >= best_ask:
        return "buy"
    elif price <= best_bid:
        return "sell"
    elif price >= mid:
        return "buy"
    else:
        return "sell"


def parse_trades_at(trades_df: pd.DataFrame, timestamp: int, ob: OBSnapshot) -> list[Trade]:
    """
    Return all trades at a given timestamp with side inferred from price vs book.
    """
    rows = trades_df[trades_df["timestamp"] == timestamp]
    result = []

    for _, r in rows.iterrows():
        price = float(r["price"])
        qty = float(r["quantity"])
        side = infer_trade_side(price, ob)

        result.append(Trade(
            timestamp=int(r["timestamp"]),
            price=price,
            quantity=qty,
            side=side,
        ))

    return result


# ── Runner ────────────────────────────────────────────────────────────────────

def run(ob_path: str, trades_path: str, product: str = "TOMATOES") -> pd.DataFrame:
    """
    Run the model over historical data. Returns a DataFrame with one row per
    tick, containing all factor values and the final direction_score.

    Important:
    factor columns are shifted by 1 tick to avoid lookahead / alignment leakage.
    """
    ob_df = load_order_book(ob_path, product)
    trades_df = load_trades(trades_path, product)

    engine = FactorEngine()
    records = []

    for _, row in ob_df.iterrows():
        ts = int(row["timestamp"])
        ob = parse_ob_row(row)
        trades = parse_trades_at(trades_df, ts, ob)

        score = engine.update(ob, trades)

        record = {
            "timestamp": ts,
            "mid_price": ob.mid,
            "F1_obi": engine.factors.get("obi", 0.0),
            "F2_vwmid_mom": engine.factors.get("vwmid_mom", 0.0),
            "F3_tfi": engine.factors.get("tfi", 0.0),
            "F4_spread_z": engine.factors.get("spread_z", 0.0),
            "F5_depth_asym": engine.factors.get("depth_asym", 0.0),
            "F6_price_mom": engine.factors.get("price_mom", 0.0),
            "direction_score": score,
        }

        if "day" in row.index:
            record["day"] = row["day"]

        records.append(record)

    out = pd.DataFrame(records)

    # sort again explicitly for safety
    sort_cols = ["timestamp"]
    if "day" in out.columns:
        sort_cols = ["day", "timestamp"]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    # factor columns to be used for prediction / skew
    factor_cols = [
        "F1_obi",
        "F2_vwmid_mom",
        "F3_tfi",
        "F4_spread_z",
        "F5_depth_asym",
        "F6_price_mom",
        "direction_score",
    ]

    # shift by 1 tick to remove lookahead
    out[factor_cols] = out[factor_cols].shift(1)

    # prefer day first if present
    cols = list(out.columns)
    if "day" in cols:
        ordered = [
            "day", "timestamp", "mid_price",
            "F1_obi", "F2_vwmid_mom", "F3_tfi",
            "F4_spread_z", "F5_depth_asym", "F6_price_mom",
            "direction_score",
        ]
        out = out[ordered]

    return out


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python multi_factor.py <ob_csv> <trades_csv>")
        print("")
        print("Example:")
        print("  python multi_factor.py "
              "TUTORIAL_ROUND_1/prices_round_0_day_-1_tomatoes.csv "
              "TUTORIAL_ROUND_1/trades_round_0_day_-1_tomatoes.csv")
        sys.exit(0)

    ob_path = sys.argv[1]
    trades_path = sys.argv[2]

    results = run(ob_path, trades_path)
    print(results.to_string(index=False))

    os.makedirs("factors", exist_ok=True)

    filename = os.path.basename(ob_path)
    out = os.path.join(
        "factors",
        filename.replace(".csv", "_factors.csv")
    )

    results.to_csv(out, index=False)

    print("Saved to:", out)