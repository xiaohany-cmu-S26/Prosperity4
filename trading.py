from datamodel import OrderDepth, UserId, TradingState, Order
from datamodel import Trade as DMTrade
from typing import List, Optional
import math
import json
from collections import deque
from dataclasses import dataclass, field
import numpy as np

EMERALD_FAIR_VALUE = 10000
POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80
}
DEFAULT_LIMIT = 80

# ── Factor Engine Config ──────────────────────────────────────────────────────
# Calibrated weights from IC analysis (horizon = 2)
FACTOR_WEIGHTS = {
    "obi":        0.013,
    "vwmid_mom":  0.322,
    "tfi":        0.012,
    "spread_z":   0.188,
    "depth_asym": -0.070,
    "price_mom":  -0.395,
}

WINDOWS = {
    "mom_fast":      10,
    "mom_slow":      30,
    "tfi_window":    80,
    "spread_window": 50,
}

# direction_score in [-1, 1] -> FV skew in price ticks
FACTOR_FV_SKEW = 3.0


# ── Factor Engine Dataclasses ─────────────────────────────────────────────────

@dataclass
class OBLevel:
    price: float
    volume: float


@dataclass
class OBSnapshot:
    timestamp: int
    bids: list
    asks: list
    mid: float


@dataclass
class FactorTrade:
    price: float
    quantity: float
    side: str  # "buy" | "sell" | "unknown"


# ── Factor Engine ─────────────────────────────────────────────────────────────

class FactorEngine:
    def __init__(self):
        self._wmid_history: deque = deque(maxlen=WINDOWS["mom_slow"] + 5)
        self._mid_history: deque = deque(maxlen=WINDOWS["mom_slow"] + 5)
        self._spread_history: deque = deque(maxlen=WINDOWS["spread_window"] + 5)
        self._obi_history: deque = deque(maxlen=50)
        self._tfi_tick_window: deque = deque(maxlen=WINDOWS["tfi_window"])
        self.factors: dict = {}

    def update(self, ob: OBSnapshot, trades: list) -> dict:
        wmid = self._weighted_mid(ob)
        self._wmid_history.append(wmid)
        self._mid_history.append(ob.mid)
        spread = self._spread(ob)
        self._spread_history.append(spread)
        self._add_trades_one_tick(trades)

        self.factors = {
            "obi":        self._f_obi(ob),
            "vwmid_mom":  self._f_vwmid_mom(),
            "tfi":        self._f_tfi(),
            "spread_z":   self._f_spread_z(),
            "depth_asym": self._f_depth_asym(ob),
            "price_mom":  self._f_price_mom(),
        }
        return self.factors

    def compute_direction_score(self) -> float:
        score = sum(FACTOR_WEIGHTS[k] * v for k, v in self.factors.items())
        return float(np.clip(score, -1.0, 1.0))

    def _f_obi(self, ob: OBSnapshot) -> float:
        OBI_LEVELS = 2
        OBI_SMOOTH = 5
        if not ob.bids or not ob.asks:
            return 0.0
        bid_vol = sum(lvl.volume for lvl in ob.bids[:OBI_LEVELS])
        ask_vol = sum(lvl.volume for lvl in ob.asks[:OBI_LEVELS])
        total = bid_vol + ask_vol
        raw_obi = ((bid_vol - ask_vol) / total) if total > 0 else 0.0
        self._obi_history.append(raw_obi)
        if OBI_SMOOTH <= 1:
            return float(raw_obi)
        recent = list(self._obi_history)[-OBI_SMOOTH:]
        return float(sum(recent) / len(recent))

    def _f_vwmid_mom(self) -> float:
        if len(self._wmid_history) < WINDOWS["mom_fast"] + 1:
            return 0.0
        arr = np.array(self._wmid_history, dtype=float)
        fast = arr[-WINDOWS["mom_fast"]:].mean()
        slow = arr[-WINDOWS["mom_slow"]:].mean() if len(arr) >= WINDOWS["mom_slow"] else arr.mean()
        std = arr[:-1].std() + 1e-9
        return float(np.tanh((fast - slow) / std))

    def _f_tfi(self) -> float:
        if not self._tfi_tick_window:
            return 0.0
        net_vol = sum(self._tfi_tick_window)
        total_vol = sum(abs(v) for v in self._tfi_tick_window)
        if total_vol == 0:
            return 0.0
        return float(np.clip(net_vol / total_vol, -1.0, 1.0))

    def _f_spread_z(self) -> float:
        if len(self._spread_history) < 5:
            return 0.0
        arr = np.array(self._spread_history, dtype=float)
        history = arr[:-1]
        if len(history) < 2:
            return 0.0
        mean = history.mean()
        std = history.std() + 1e-9
        z = (arr[-1] - mean) / std
        return float(np.tanh(z))

    def _f_depth_asym(self, ob: OBSnapshot) -> float:
        bid_vol = sum(lvl.volume for lvl in ob.bids)
        ask_vol = sum(lvl.volume for lvl in ob.asks)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def _f_price_mom(self) -> float:
        if len(self._mid_history) < WINDOWS["mom_fast"] + 1:
            return 0.0
        arr = np.array(self._mid_history, dtype=float)
        fast = self._ema(arr, WINDOWS["mom_fast"])
        slow = self._ema(arr, min(WINDOWS["mom_slow"], len(arr)))
        std = arr[:-1].std() + 1e-9
        return float(np.tanh((fast - slow) / std))

    def _weighted_mid(self, ob: OBSnapshot) -> float:
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

    def _ema(self, arr, span: int) -> float:
        alpha = 2.0 / (span + 1)
        result = arr[0]
        for v in arr[1:]:
            result = alpha * v + (1 - alpha) * result
        return float(result)

    def _add_trades_one_tick(self, trades: list):
        signed_vol = 0.0
        for t in trades:
            if t.side == "buy":
                signed_vol += t.quantity
            elif t.side == "sell":
                signed_vol -= t.quantity
        self._tfi_tick_window.append(signed_vol)

    def to_dict(self) -> dict:
        return {
            "wmid_history": list(self._wmid_history),
            "mid_history": list(self._mid_history),
            "spread_history": list(self._spread_history),
            "obi_history": list(self._obi_history),
            "tfi_tick_window": list(self._tfi_tick_window),
            "factors": self.factors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FactorEngine":
        engine = cls()
        engine._wmid_history = deque(d.get("wmid_history", []), maxlen=WINDOWS["mom_slow"] + 5)
        engine._mid_history = deque(d.get("mid_history", []), maxlen=WINDOWS["mom_slow"] + 5)
        engine._spread_history = deque(d.get("spread_history", []), maxlen=WINDOWS["spread_window"] + 5)
        engine._obi_history = deque(d.get("obi_history", []), maxlen=50)
        engine._tfi_tick_window = deque(d.get("tfi_tick_window", []), maxlen=WINDOWS["tfi_window"])
        engine.factors = d.get("factors", {})
        return engine


# ── State I/O ─────────────────────────────────────────────────────────────────

def _load_state(trader_data: str) -> dict:
    if not trader_data:
        return {}
    try:
        return json.loads(trader_data)
    except Exception:
        return {}


def _dump_state(state: dict) -> str:
    return json.dumps(state)


# ── OB / Trade Helpers ────────────────────────────────────────────────────────

def _make_ob_snapshot(ts: int, order_depth: OrderDepth) -> OBSnapshot:
    bids = [
        OBLevel(price=float(p), volume=float(abs(v)))
        for p, v in sorted(order_depth.buy_orders.items(), reverse=True)
        if v != 0
    ]
    asks = [
        OBLevel(price=float(p), volume=float(abs(v)))
        for p, v in sorted(order_depth.sell_orders.items())
        if v != 0
    ]
    if bids and asks:
        mid = (bids[0].price + asks[0].price) / 2.0
    elif bids:
        mid = bids[0].price
    elif asks:
        mid = asks[0].price
    else:
        mid = 0.0
    return OBSnapshot(timestamp=ts, bids=bids, asks=asks, mid=mid)


def _infer_side(price: float, ob: OBSnapshot) -> str:
    if not ob.bids or not ob.asks:
        return "unknown"
    best_bid = ob.bids[0].price
    best_ask = ob.asks[0].price
    if price >= best_ask:
        return "buy"
    elif price <= best_bid:
        return "sell"
    elif price >= ob.mid:
        return "buy"
    else:
        return "sell"


def _convert_trades(dm_trades: list, ob: OBSnapshot) -> list:
    result = []
    for t in dm_trades:
        side = _infer_side(float(t.price), ob)
        result.append(FactorTrade(price=float(t.price), quantity=float(t.quantity), side=side))
    return result


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def micro_dev_l2(self, order_depth: OrderDepth) -> Optional[float]:
        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        asks = sorted(order_depth.sell_orders.keys())
        if len(bids) < 2 or len(asks) < 2:
            return None
        bid_price_2 = bids[1]
        ask_price_2 = asks[1]
        bid_volume_2 = abs(order_depth.buy_orders[bid_price_2])
        ask_volume_2 = abs(order_depth.sell_orders[ask_price_2])
        total_vol = bid_volume_2 + ask_volume_2
        if total_vol == 0:
            return None
        return (bid_price_2 * ask_volume_2 + ask_price_2 * bid_volume_2) / total_vol

    def get_spread(self, order_depth: OrderDepth) -> Optional[int]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys()) - max(order_depth.buy_orders.keys())

    def get_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

    def get_fair_value(self, product: str, order_depth: OrderDepth, position: int, direction_score: float = 0.0) -> Optional[float]:
        if product == "EMERALDS":
            fv = EMERALD_FAIR_VALUE
        else:
            mid = self.get_mid(order_depth)
            if mid is None:
                return None

            spread = self.get_spread(order_depth)

            # L2 microprice only during wide spread (>= 10).
            # During tight spread it blows up (std 6.15 vs 0.35).
            if spread is not None and spread >= 10:
                mp_l2 = self.micro_dev_l2(order_depth)
                fv = mp_l2 if mp_l2 is not None else mid
            else:
                fv = mid

            # Factor-based FV skew for TOMATOES
            fv += direction_score * FACTOR_FV_SKEW

        if fv is None:
            return None

        # Inventory skew on FV for EMERALDS only.
        # TOMATOES skew is handled via asymmetric quoting in run().
        if product == "EMERALDS":
            if abs(position) < 40:
                skew = 0.0
            elif abs(position) < 60:
                skew = 0.05 * position
            else:
                skew = 0.2 * position
            fv -= skew

        return fv

    def run(self, state: TradingState):
        persistent = _load_state(state.traderData)
        result = {}

        # ── Factor Engine: update TOMATOES engine ─────────────────────────────
        tomato_engine_dict = persistent.get("tomato_engine", None)
        if tomato_engine_dict is not None:
            tomato_engine = FactorEngine.from_dict(tomato_engine_dict)
        else:
            tomato_engine = FactorEngine()

        tomato_direction_score = 0.0
        if "TOMATOES" in state.order_depths:
            ob = _make_ob_snapshot(state.timestamp, state.order_depths["TOMATOES"])
            raw_trades = state.market_trades.get("TOMATOES", [])
            factor_trades = _convert_trades(raw_trades, ob)
            tomato_engine.update(ob, factor_trades)
            tomato_direction_score = tomato_engine.compute_direction_score()

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, DEFAULT_LIMIT)

            direction_score = tomato_direction_score if product == "TOMATOES" else 0.0
            fv = self.get_fair_value(product, order_depth, position, direction_score)
            if fv is None:
                result[product] = orders
                continue

            buy_capacity = limit - position
            sell_capacity = limit + position

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            is_wide = spread >= 10

            # ── Step 1: Taking (Aggressive) ───────────────────────────
            #
            # Wide spread: take freely when price crosses FV.
            # Tight spread: take ONLY to unwind inventory.
            #   If we're short, we can lift asks below FV (buy to cover).
            #   If we're long, we can hit bids above FV (sell to flatten).
            #   Never take in the direction that worsens inventory.

            if product == "EMERALDS" or is_wide:
                # Full taking in both directions
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price > fv or buy_capacity <= 0:
                        break
                    vol = abs(order_depth.sell_orders[ask_price])
                    qty = min(buy_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty

                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price < fv or sell_capacity <= 0:
                        break
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty

            elif product == "TOMATOES":
                # Tight spread: only take to unwind
                if position < 0:
                    # Short -> buy to cover
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if ask_price > fv or buy_capacity <= 0:
                            break
                        vol = abs(order_depth.sell_orders[ask_price])
                        qty = min(buy_capacity, vol, abs(position))
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            buy_capacity -= qty

                elif position > 0:
                    # Long -> sell to flatten
                    for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                        if bid_price < fv or sell_capacity <= 0:
                            break
                        vol = order_depth.buy_orders[bid_price]
                        qty = min(sell_capacity, vol, position)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            sell_capacity -= qty

            # ── Step 2: Making (Passive) ──────────────────────────────
            #
            # Default: conservative bid+1 / ask-1 (proven profitable).
            #
            # Asymmetric inventory management for TOMATOES:
            #   The UNWINDING side quotes more aggressively based on
            #   inventory level, while the accumulating side stays at
            #   bid+1 / ask-1.
            #
            #   |pos| < 40:  both sides at bid+1 / ask-1 (pure MM)
            #   |pos| 40-63: unwinding side moves to FV ± 3
            #   |pos| > 63:  unwinding side quotes AT fair value
            #
            # Tight spread cap always applies (min/max with bid+1/ask-1)
            # to prevent microprice blowup from crossing the book.

            conservative_bid = best_bid + 1
            conservative_ask = best_ask - 1

            if product == "TOMATOES":
                abs_pos = abs(position)

                if abs_pos <= 63:
                    # Pure MM: symmetric conservative
                    our_bid = conservative_bid
                    our_ask = conservative_ask

                elif abs_pos <= 39:
                    # Moderate inventory: unwinding side at FV ± 3
                    if position > 0:
                        # Long: ask aggressively, bid conservatively
                        our_bid = conservative_bid
                        our_ask = int(fv) + 3
                    else:
                        # Short: bid aggressively, ask conservatively
                        our_bid = int(fv) - 3
                        our_ask = conservative_ask

                else:
                    # Critical inventory (>63): unwinding side as close
                    # to fair value as possible while staying live.
                    # ceil(fv) for asks guarantees ask > fv when fv is
                    # not an integer; floor(fv) + 1 when it is.
                    if position > 0:
                        our_bid = conservative_bid
                        our_ask = math.ceil(fv) if fv != int(fv) else int(fv) + 1
                    else:
                        our_bid = math.floor(fv) if fv != int(fv) else int(fv) - 1
                        our_ask = conservative_ask

                # Tight spread cap: never more aggressive than bid+1/ask-1
                if not is_wide:
                    our_bid = min(our_bid, conservative_bid)
                    our_ask = max(our_ask, conservative_ask)

                # Floor: never quote behind the bots
                our_bid = max(our_bid, best_bid)
                our_ask = min(our_ask, best_ask)

            else:
                # EMERALDS: conservative with FV cap
                fv_bid = int(fv) - 1
                fv_ask = int(fv) + 1
                our_bid = min(conservative_bid, fv_bid)
                our_ask = max(conservative_ask, fv_ask)

            # Safety: don't cross the book
            if our_bid >= best_ask:
                our_bid = best_bid
            if our_ask <= best_bid:
                our_ask = best_ask

            # Place orders if on correct side of FV
            if buy_capacity > 0 and our_bid < fv:
                orders.append(Order(product, our_bid, buy_capacity))

            if sell_capacity > 0 and our_ask > fv:
                orders.append(Order(product, our_ask, -sell_capacity))

            result[product] = orders

        persistent["tomato_engine"] = tomato_engine.to_dict()
        traderData = _dump_state(persistent)
        conversions = 0
        return result, conversions, traderData
