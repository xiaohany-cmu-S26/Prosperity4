from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import json
from collections import deque
from dataclasses import dataclass
import numpy as np


# ── Factor Engine Config ──────────────────────────────────────────────────────
# Calibrated from trading.py IC analysis (horizon = 2).
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


# ── Per-product configuration ─────────────────────────────────────────────────
# fair_value:          fixed float | "linear_trend" | "midwall"
# trend_slope:         slope per timestamp unit (linear_trend only)
# use_l2_microprice:   enable L2 microprice FV when spread >= wide_threshold
# use_factors:         run FactorEngine and apply direction_score * factor_fv_skew
# skew_tiers:          tiered FV inventory skew — [(|pos|_threshold, factor), ...]
# asymmetric_quoting:  inventory-driven asymmetric passive quotes (TOMATOES-style)
# fv_spread:           FV-anchored passive quotes at base_fv ± fv_spread
# quote_offset:        ticks from best bid/ask (negative = inside the book)
# max_passive_qty:     per-order cap on passive size (None = full capacity)
# spread_aware_taking: tight spread → only take to unwind inventory
PRODUCT_CONFIG = {
    "INTARIAN_PEPPER_ROOT": {
        "limit": 80,
        # Deterministic linear trend anchored at the first mid each day.
        # Empirically verified: slope 0.001/ts, R² ≈ 1.0 on rounds -2/-1/0.
        "fair_value": "linear_trend",
        "trend_slope": 0.001,

        # Spread ~13 ticks → L2 microprice reduces top-of-book noise.
        "use_l2_microprice": True,
        "wide_spread_threshold": 10,

        # Linear trend gives R²≈1 — factor residual alpha is ~0.
        # Infrastructure kept; flip use_factors to True to enable.
        "use_factors": False,
        "factor_fv_skew": 3.0,

        # Inventory handled via asymmetric quoting (not FV skew).
        "skew_tiers": [(80, 0.0)],
        "asymmetric_quoting": True,
        "asym_moderate_pos": 40,   # < moderate → pure MM
        "asym_critical_pos": 63,   # moderate–critical → unwind at FV ± asym_offset
        "asym_offset": 3,          # > critical → unwind AT fair value

        # Post 1 tick inside the best quotes (queue priority inside MMs).
        "quote_offset": -1,
        "max_passive_qty": None,

        # Wide spread book → take freely. Tight spread → only unwind.
        "spread_aware_taking": True,
    },
    "ASH_COATED_OSMIUM": {
        "limit": 80,
        # Strong mean reversion, anchor at 10000 (lag-1 autocorr ≈ -0.49).
        "fair_value": 10000,

        # Compressed spread → L2 microprice blows up, keep disabled.
        "use_l2_microprice": False,
        "wide_spread_threshold": 10,

        "use_factors": False,
        "factor_fv_skew": 0.0,

        # Tiered FV skew (trading.py EMERALDS-style).
        "skew_tiers": [
            (40, 0.0),    # |pos| < 40  → pure MM
            (60, 0.05),   # |pos| < 60  → light skew
            (80, 0.2),    # |pos| ≥ 60  → aggressive clear
        ],
        "asymmetric_quoting": False,

        # FV-anchored passive quotes at 9994 / 10006
        # (1 tick inside structural MMs resting at ±7).
        "fv_spread": 6,
        "max_passive_qty": 12,

        # Mean reversion makes any cross-FV take +EV.
        "spread_aware_taking": False,
    },
}

DEFAULT_LIMIT = 80


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


# ── Factor Engine (ported from trading.py) ────────────────────────────────────

class FactorEngine:
    def __init__(self):
        self._wmid_history: deque = deque(maxlen=WINDOWS["mom_slow"] + 5)
        self._mid_history: deque = deque(maxlen=WINDOWS["mom_slow"] + 5)
        self._spread_history: deque = deque(maxlen=WINDOWS["spread_window"] + 5)
        self._obi_history: deque = deque(maxlen=50)
        self._tfi_tick_window: deque = deque(maxlen=WINDOWS["tfi_window"])
        self.factors: dict = {}

    def update(self, ob: OBSnapshot, trades: list) -> dict:
        self._wmid_history.append(self._weighted_mid(ob))
        self._mid_history.append(ob.mid)
        self._spread_history.append(self._spread(ob))
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
        return float(np.tanh((arr[-1] - mean) / std))

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
        bv, av = ob.bids[0].volume, ob.asks[0].volume
        bp, ap = ob.bids[0].price, ob.asks[0].price
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
            "wmid_history":    list(self._wmid_history),
            "mid_history":     list(self._mid_history),
            "spread_history":  list(self._spread_history),
            "obi_history":     list(self._obi_history),
            "tfi_tick_window": list(self._tfi_tick_window),
            "factors":         self.factors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FactorEngine":
        e = cls()
        e._wmid_history    = deque(d.get("wmid_history", []),    maxlen=WINDOWS["mom_slow"] + 5)
        e._mid_history     = deque(d.get("mid_history", []),     maxlen=WINDOWS["mom_slow"] + 5)
        e._spread_history  = deque(d.get("spread_history", []),  maxlen=WINDOWS["spread_window"] + 5)
        e._obi_history     = deque(d.get("obi_history", []),     maxlen=50)
        e._tfi_tick_window = deque(d.get("tfi_tick_window", []), maxlen=WINDOWS["tfi_window"])
        e.factors          = d.get("factors", {})
        return e


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
    if price <= best_bid:
        return "sell"
    return "buy" if price >= ob.mid else "sell"


def _convert_trades(dm_trades: list, ob: OBSnapshot) -> list:
    return [
        FactorTrade(
            price=float(t.price),
            quantity=float(t.quantity),
            side=_infer_side(float(t.price), ob),
        )
        for t in dm_trades
    ]


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    # ── Book helpers ─────────────────────────────────────────────────────────

    def raw_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return None
        if not order_depth.buy_orders:
            return float(min(order_depth.sell_orders.keys()))
        if not order_depth.sell_orders:
            return float(max(order_depth.buy_orders.keys()))
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

    def micro_dev_l2(self, order_depth: OrderDepth) -> Optional[float]:
        """L2 microprice: VWAP of 2nd-best bid and 2nd-best ask."""
        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        asks = sorted(order_depth.sell_orders.keys())
        if len(bids) < 2 or len(asks) < 2:
            return None
        bid_p2, ask_p2 = bids[1], asks[1]
        bid_v2 = abs(order_depth.buy_orders[bid_p2])
        ask_v2 = abs(order_depth.sell_orders[ask_p2])
        total = bid_v2 + ask_v2
        if total == 0:
            return None
        return (bid_p2 * ask_v2 + ask_p2 * bid_v2) / total

    def get_spread(self, order_depth: OrderDepth) -> Optional[int]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys()) - max(order_depth.buy_orders.keys())

    def skew_factor(self, position: int, tiers: list) -> float:
        for threshold, factor in tiers:
            if abs(position) < threshold:
                return factor
        return tiers[-1][1]

    # ── Fair value ───────────────────────────────────────────────────────────

    def get_base_fv(
        self,
        product: str,
        order_depth: OrderDepth,
        persistent: Dict,
        timestamp: int,
    ) -> Optional[float]:
        """
        Returns the base (pre-skew) fair value.
        Fixed float         -> constant.
        'linear_trend'      -> init_fv + slope * (ts - init_ts), anchored once.
        'midwall'           -> EMA of top-of-book mid.
        When use_l2_microprice is set, overrides with L2 microprice during
        wide spreads.
        """
        cfg    = PRODUCT_CONFIG.get(product, {})
        fv_cfg = cfg.get("fair_value", "midwall")
        ema_state = persistent.setdefault("ema", {})

        if isinstance(fv_cfg, (int, float)):
            base = float(fv_cfg)

        elif fv_cfg == "linear_trend":
            slope       = cfg.get("trend_slope", 0.0)
            init_fv_key = product + "_init_fv"
            init_ts_key = product + "_init_ts"
            mid = self.raw_mid(order_depth)
            if mid is None:
                return None
            if init_fv_key not in ema_state:
                ema_state[init_fv_key] = mid
                ema_state[init_ts_key] = float(timestamp)
            base = ema_state[init_fv_key] + slope * (timestamp - ema_state[init_ts_key])

        else:
            mid = self.raw_mid(order_depth)
            if mid is None:
                return None
            alpha    = cfg.get("ema_alpha", 1.0)
            prev_ema = ema_state.get(product, mid)
            base     = alpha * mid + (1 - alpha) * prev_ema
            ema_state[product] = base

        if cfg.get("use_l2_microprice", False):
            spread = self.get_spread(order_depth)
            if spread is not None and spread >= cfg.get("wide_spread_threshold", 10):
                mp_l2 = self.micro_dev_l2(order_depth)
                if mp_l2 is not None:
                    base = mp_l2

        return base

    # ── Taking (aggressive) ──────────────────────────────────────────────────

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fv: float,
        buy_capacity: int,
        sell_capacity: int,
        position: int,
        spread_aware: bool,
        is_wide: bool,
    ) -> Tuple[List[Order], int, int]:
        """
        Take asks below FV and bids above FV.
        spread_aware + tight spread → only take in the direction that unwinds
        inventory. Wide spread or spread-unaware → full bidirectional taking.
        """
        orders: List[Order] = []

        if spread_aware and not is_wide:
            if position < 0:
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price > fv or buy_capacity <= 0:
                        break
                    vol = abs(order_depth.sell_orders[ask_price])
                    qty = min(buy_capacity, vol, abs(position))
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty
            elif position > 0:
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price < fv or sell_capacity <= 0:
                        break
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, vol, position)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty
            return orders, buy_capacity, sell_capacity

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

        return orders, buy_capacity, sell_capacity

    # ── Making (passive) ─────────────────────────────────────────────────────

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        base_fv: float,
        skewed_fv: float,
        buy_capacity: int,
        sell_capacity: int,
        position: int,
        cfg: Dict,
        is_wide: bool,
    ) -> List[Order]:
        orders: List[Order] = []
        fv_spread    = cfg.get("fv_spread", None)
        quote_offset = cfg.get("quote_offset", 0)
        max_qty      = cfg.get("max_passive_qty")

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if fv_spread is not None:
            fv_int = int(round(base_fv))
            our_bid = fv_int - fv_spread
            our_ask = fv_int + fv_spread
        else:
            if best_bid is None or best_ask is None:
                return orders
            conservative_bid = best_bid - quote_offset
            conservative_ask = best_ask + quote_offset

            if cfg.get("asymmetric_quoting", False):
                abs_pos  = abs(position)
                moderate = cfg.get("asym_moderate_pos", 40)
                critical = cfg.get("asym_critical_pos", 63)
                asym_off = cfg.get("asym_offset", 3)

                if abs_pos < moderate:
                    # Pure market making — symmetric conservative quotes.
                    our_bid = conservative_bid
                    our_ask = conservative_ask

                elif abs_pos <= critical:
                    # Moderate inventory — unwinding side at FV ± asym_off.
                    if position > 0:
                        our_bid = conservative_bid
                        our_ask = int(base_fv) + asym_off
                    else:
                        our_bid = int(base_fv) - asym_off
                        our_ask = conservative_ask

                else:
                    # Critical inventory — unwinding side as close to FV
                    # as possible while still being a valid quote.
                    if position > 0:
                        our_bid = conservative_bid
                        our_ask = (
                            math.ceil(base_fv)
                            if base_fv != int(base_fv)
                            else int(base_fv) + 1
                        )
                    else:
                        our_bid = (
                            math.floor(base_fv)
                            if base_fv != int(base_fv)
                            else int(base_fv) - 1
                        )
                        our_ask = conservative_ask

                # Tight spread: never more aggressive than conservative.
                if not is_wide:
                    our_bid = min(our_bid, conservative_bid)
                    our_ask = max(our_ask, conservative_ask)

                # Never quote behind the best bid/ask.
                our_bid = max(our_bid, best_bid)
                our_ask = min(our_ask, best_ask)
            else:
                our_bid = conservative_bid
                our_ask = conservative_ask

        # Safety: never cross the book.
        if best_bid is not None and best_ask is not None:
            if our_bid >= best_ask:
                our_bid = best_bid
            if our_ask <= best_bid:
                our_ask = best_ask

        # Post only on the profitable side of the skewed FV.
        if buy_capacity > 0 and our_bid < skewed_fv:
            qty = buy_capacity if max_qty is None else min(buy_capacity, max_qty)
            orders.append(Order(product, our_bid, qty))

        if sell_capacity > 0 and our_ask > skewed_fv:
            qty = sell_capacity if max_qty is None else min(sell_capacity, max_qty)
            orders.append(Order(product, our_ask, -qty))

        return orders

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self, state: TradingState):
        try:
            persistent: Dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            persistent = {}

        factor_engines_state = persistent.setdefault("factor_engines", {})
        direction_scores: Dict[str, float] = {}

        # Update factor engines for products with use_factors=True.
        for product, cfg in PRODUCT_CONFIG.items():
            if not cfg.get("use_factors", False):
                continue
            if product not in state.order_depths:
                continue
            engine_dict = factor_engines_state.get(product)
            engine = FactorEngine.from_dict(engine_dict) if engine_dict else FactorEngine()
            ob = _make_ob_snapshot(state.timestamp, state.order_depths[product])
            raw_trades = state.market_trades.get(product, [])
            engine.update(ob, _convert_trades(raw_trades, ob))
            direction_scores[product] = engine.compute_direction_score()
            factor_engines_state[product] = engine.to_dict()

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            cfg = PRODUCT_CONFIG.get(product, {})

            limit    = cfg.get("limit", DEFAULT_LIMIT)
            tiers    = cfg.get("skew_tiers", [(DEFAULT_LIMIT, 0.0)])
            position = state.position.get(product, 0)

            base_fv = self.get_base_fv(product, order_depth, persistent, state.timestamp)
            if base_fv is None:
                result[product] = orders
                continue

            # Factor-based FV shift (taking and sign check).
            factor_skew = cfg.get("factor_fv_skew", 0.0)
            base_fv += direction_scores.get(product, 0.0) * factor_skew

            # Tiered inventory skew — applied on top of factor-adjusted FV.
            sf        = self.skew_factor(position, tiers)
            skewed_fv = base_fv - sf * position

            buy_capacity  = limit - position
            sell_capacity = limit + position

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread   = best_ask - best_bid
            is_wide  = spread >= cfg.get("wide_spread_threshold", 10)

            # Step 1 — aggressive taking (uses skewed FV)
            taken, buy_capacity, sell_capacity = self.take_orders(
                product, order_depth, skewed_fv,
                buy_capacity, sell_capacity, position,
                cfg.get("spread_aware_taking", False), is_wide,
            )
            orders.extend(taken)

            # Step 2 — passive making
            orders.extend(self.make_orders(
                product, order_depth, base_fv, skewed_fv,
                buy_capacity, sell_capacity, position, cfg, is_wide,
            ))

            result[product] = orders

        return result, 0, json.dumps(persistent)