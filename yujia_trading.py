from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import json
from collections import deque
from dataclasses import dataclass
import numpy as np


# ── Per-product configuration ─────────────────────────────────────────────────
# fair_value:          fixed float | "linear_trend" | "midwall"
# trend_slope:         slope per timestamp unit (linear_trend only)
# take_fv_skew:        directional shift used for aggressive taking
# quote_fv_skew:       directional shift used for passive quoting
# factor_params:       Round 1 calibrated product-specific factor recipe
# min_long_position:   minimum inventory to carry for structurally bullish products
# core_build_window:   timestamp window used to ramp from 0 to min_long_position
# core_build_per_tick: max shares added per tick while building the core long
# soft_min_long_position: temporary lower floor allowed during strong bearish signals
# soft_floor_release_threshold: turn on soft floor when direction_score falls below this
# soft_floor_rebuild_threshold: return to the full floor when direction_score recovers above this
# skew_tiers:          tiered FV inventory skew — [(|pos|_threshold, factor), ...]
# asymmetric_quoting:  inventory-driven asymmetric passive quotes
# fv_spread:           FV-anchored passive quotes at quote_fv ± fv_spread
# quote_offset:        ticks from best bid/ask (negative = inside the book)
# max_passive_qty:     per-order cap on passive size (None = full capacity)
# passive_signal_suppress: shrink passive size on the side fighting the signal
# max_take_qty:        per-tick cap on aggressive taking size
# take_top_level_only: only cross the best level when taking
# take_side_weights:   optional asymmetric active-FV weights for buy/sell
PRODUCT_CONFIG = {
    "INTARIAN_PEPPER_ROOT": {
        "limit": 80,
        # Round 1 shows a near-perfect +0.001/ts trend with ~2 tick residuals.
        "fair_value": "linear_trend",
        "trend_slope": 0.001,
        "wide_spread_threshold": 10,
        "min_long_position": 70,
        "core_build_window": 1100,
        "core_build_per_tick": 60,
        "soft_min_long_position": 60,
        "soft_floor_release_threshold": -0.70,
        "soft_floor_rebuild_threshold": -0.10,
        "use_factors": True,
        "take_fv_skew": 2.5,
        "quote_fv_skew": 0.0,
        "take_signal_threshold": 0.85,
        "factor_params": {
            "stretch_window": 20,
            "obi_smooth": 3,
            "anchor_scale": 2.4,
            "stretch_scale": 2.8,
            "micro_scale": 2.0,
            "weights": {
                "anchor_gap": 0.42,
                "stretch": 0.22,
                "obi": 0.21,
                "depth_pressure": 0.10,
                "micro_gap": 0.05,
            },
        },
        "skew_tiers": [(80, 0.0)],
        "asymmetric_quoting": True,
        "asym_moderate_pos": 40,
        "asym_critical_pos": 63,
        "asym_offset": 3,
        "quote_offset": -1,
        "max_passive_qty": None,
        "passive_signal_suppress": 0.0,
        "spread_aware_taking": True,
    },
    "ASH_COATED_OSMIUM": {
        "limit": 80,
        # Stationary around 10000 with much stronger book-pressure alpha.
        "fair_value": 10000,
        "wide_spread_threshold": 10,
        "use_factors": True,
        "take_fv_skew": 3.0,
        "take_fv_skew_buy": 3.5,
        "take_fv_skew_sell": 2.5,
        "quote_fv_skew": 0.0,
        # Faster mean-reversion factors work better for active OSMIUM taking,
        # but only when execution stays shallow and size-capped.
        "take_signal_threshold": 0.9,
        "take_signal_threshold_buy": 0.75,
        "take_signal_threshold_sell": 0.80,
        "take_side_weights": {
            "buy": {
                "stretch": 0.85,
                "obi": 0.15,
            },
            "sell": {
                "stretch": 0.85,
                "anchor_gap": 0.15,
            },
        },
        "factor_params": {
            "stretch_window": 10,
            "obi_smooth": 1,
            "anchor_scale": 4.0,
            "stretch_scale": 3.5,
            "micro_scale": 2.5,
            "weights": {
                "anchor_gap": 0.20,
                "stretch": 0.45,
                "obi": 0.20,
                "depth_pressure": 0.00,
                "micro_gap": 0.15,
            },
        },
        "skew_tiers": [
            (40, 0.0),
            (60, 0.05),
            (80, 0.2),
        ],
        "asymmetric_quoting": False,
        "fv_spread": 6,
        "max_take_qty": 10,
        "take_top_level_only": True,
        "max_passive_qty": 12,
        "passive_signal_suppress": 0.0,
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


# ── Factor Engine ──────────────────────────────────────────────────────────────

class FactorEngine:
    def __init__(self, product: str, cfg: Dict):
        self.product = product
        self.cfg = cfg
        maxlen = max(cfg["stretch_window"], 8) + 5
        self._mid_history: deque = deque(maxlen=maxlen)
        self._obi_history: deque = deque(maxlen=cfg["obi_smooth"] + 5)
        self.factors: dict = {}

    def update(self, ob: OBSnapshot, reference_price: float) -> dict:
        self._mid_history.append(ob.mid)
        self._obi_history.append(self._raw_obi(ob))

        self.factors = {
            "anchor_gap": self._f_anchor_gap(ob.mid, reference_price),
            "stretch": self._f_stretch(ob.mid),
            "obi": self._f_obi(),
            "depth_pressure": self._f_depth_pressure(ob),
            "micro_gap": self._f_micro_gap(ob),
        }
        return self.factors

    def compute_direction_score(self) -> float:
        weights = self.cfg["weights"]
        score = sum(weights.get(k, 0.0) * self.factors.get(k, 0.0) for k in weights)
        return float(np.clip(score, -1.0, 1.0))

    def _squash(self, value: float, scale: float) -> float:
        return float(np.tanh(value / max(scale, 1e-9)))

    def _weighted_mid(self, ob: OBSnapshot) -> float:
        if not ob.bids or not ob.asks:
            return ob.mid
        bid_vol = ob.bids[0].volume
        ask_vol = ob.asks[0].volume
        total = bid_vol + ask_vol
        if total == 0:
            return ob.mid
        return (ob.asks[0].price * bid_vol + ob.bids[0].price * ask_vol) / total

    def _raw_obi(self, ob: OBSnapshot) -> float:
        if not ob.bids or not ob.asks:
            return 0.0
        bid_vol = ob.bids[0].volume
        ask_vol = ob.asks[0].volume
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def _f_anchor_gap(self, mid: float, reference_price: float) -> float:
        return self._squash(reference_price - mid, self.cfg["anchor_scale"])

    def _f_stretch(self, mid: float) -> float:
        if len(self._mid_history) < 2:
            return 0.0
        arr = np.array(self._mid_history, dtype=float)
        window = min(self.cfg["stretch_window"], len(arr) - 1)
        if window <= 0:
            return 0.0
        trailing_mean = arr[-window-1:-1].mean()
        return self._squash(trailing_mean - mid, self.cfg["stretch_scale"])

    def _f_obi(self) -> float:
        if not self._obi_history:
            return 0.0
        smooth = self.cfg["obi_smooth"]
        recent = list(self._obi_history)[-smooth:]
        return float(sum(recent) / len(recent))

    def _f_depth_pressure(self, ob: OBSnapshot) -> float:
        bid_vol = sum(level.volume for level in ob.bids)
        ask_vol = sum(level.volume for level in ob.asks)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def _f_micro_gap(self, ob: OBSnapshot) -> float:
        return self._squash(self._weighted_mid(ob) - ob.mid, self.cfg["micro_scale"])

    def positive_side_score(self, weights: Dict[str, float], sign: float) -> float:
        score = 0.0
        for name, weight in weights.items():
            score += weight * max(sign * self.factors.get(name, 0.0), 0.0)
        return float(np.clip(score, 0.0, 1.0))

    def to_dict(self) -> dict:
        return {
            "mid_history": list(self._mid_history),
            "obi_history": list(self._obi_history),
            "factors": self.factors,
        }

    @classmethod
    def from_dict(cls, product: str, cfg: Dict, data: dict) -> "FactorEngine":
        engine = cls(product, cfg)
        maxlen = max(cfg["stretch_window"], 8) + 5
        engine._mid_history = deque(data.get("mid_history", []), maxlen=maxlen)
        engine._obi_history = deque(data.get("obi_history", []), maxlen=cfg["obi_smooth"] + 5)
        engine.factors = data.get("factors", {})
        return engine


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

    def core_position_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        buy_capacity: int,
        target_long_position: Optional[int],
        core_build_per_tick: Optional[int],
    ) -> Tuple[List[Order], int]:
        """
        Build the structural long inventory gradually instead of all at once.
        """
        orders: List[Order] = []
        if target_long_position is None or position >= target_long_position or not order_depth.sell_orders:
            return orders, buy_capacity

        need = min(target_long_position - position, buy_capacity)
        if core_build_per_tick is not None:
            need = min(need, core_build_per_tick)
        if need <= 0:
            return orders, buy_capacity

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if need <= 0:
                break
            vol = abs(order_depth.sell_orders[ask_price])
            qty = min(need, vol)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                need -= qty

        return orders, buy_capacity

    def target_core_position(
        self,
        product: str,
        cfg: Dict,
        persistent: Dict,
        timestamp: int,
    ) -> Optional[int]:
        min_long_position = cfg.get("min_long_position")
        if min_long_position is None:
            return None

        build_window = cfg.get("core_build_window", 0)
        if build_window <= 0:
            return int(min_long_position)

        core_state = persistent.setdefault("core_build", {})
        start_key = product + "_start_ts"
        if start_key not in core_state:
            core_state[start_key] = timestamp

        elapsed = max(0, timestamp - core_state[start_key])
        progress = min(1.0, elapsed / build_window)
        target = int(math.floor(min_long_position * progress))
        if progress >= 1.0:
            target = int(min_long_position)
        return max(0, min(int(min_long_position), target))

    def effective_core_floor(
        self,
        product: str,
        cfg: Dict,
        persistent: Dict,
        position: int,
        target_long_position: Optional[int],
        direction_score: float,
    ) -> Optional[int]:
        min_long_position = cfg.get("min_long_position")
        if min_long_position is None:
            return None

        core_state = persistent.setdefault("core_build", {})
        lock_key = product + "_locked_min_long"
        soft_key = product + "_soft_floor_active"
        if position >= min_long_position:
            core_state[lock_key] = True

        if not core_state.get(lock_key, False):
            return target_long_position

        soft_min_long_position = cfg.get("soft_min_long_position", min_long_position)
        release_threshold = cfg.get("soft_floor_release_threshold")
        rebuild_threshold = cfg.get("soft_floor_rebuild_threshold", release_threshold)

        if release_threshold is not None and direction_score <= release_threshold:
            core_state[soft_key] = True
        elif core_state.get(soft_key, False) and rebuild_threshold is not None and direction_score >= rebuild_threshold:
            core_state[soft_key] = False

        if core_state.get(soft_key, False):
            return int(min(soft_min_long_position, min_long_position))
        return int(min_long_position)

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

        return base

    # ── Taking (aggressive) ──────────────────────────────────────────────────

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        buy_fv: float,
        sell_fv: Optional[float],
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
        cfg = PRODUCT_CONFIG.get(product, {})
        orders: List[Order] = []
        if sell_fv is None:
            sell_fv = buy_fv
        max_take_qty = cfg.get("max_take_qty")
        take_buy_capacity = buy_capacity if max_take_qty is None else min(buy_capacity, max_take_qty)
        take_sell_capacity = sell_capacity if max_take_qty is None else min(sell_capacity, max_take_qty)
        take_top_level_only = cfg.get("take_top_level_only", False)

        if spread_aware and not is_wide:
            if position < 0:
                for level_idx, ask_price in enumerate(sorted(order_depth.sell_orders.keys())):
                    if take_top_level_only and level_idx > 0:
                        break
                    if ask_price > buy_fv or take_buy_capacity <= 0:
                        break
                    vol = abs(order_depth.sell_orders[ask_price])
                    qty = min(take_buy_capacity, vol, abs(position))
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty
                        take_buy_capacity -= qty
            elif position > 0:
                for level_idx, bid_price in enumerate(sorted(order_depth.buy_orders.keys(), reverse=True)):
                    if take_top_level_only and level_idx > 0:
                        break
                    if bid_price < sell_fv or take_sell_capacity <= 0:
                        break
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(take_sell_capacity, vol, position)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty
                        take_sell_capacity -= qty
            return orders, buy_capacity, sell_capacity

        for level_idx, ask_price in enumerate(sorted(order_depth.sell_orders.keys())):
            if take_top_level_only and level_idx > 0:
                break
            if ask_price > buy_fv or take_buy_capacity <= 0:
                break
            vol = abs(order_depth.sell_orders[ask_price])
            qty = min(take_buy_capacity, vol)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                take_buy_capacity -= qty

        for level_idx, bid_price in enumerate(sorted(order_depth.buy_orders.keys(), reverse=True)):
            if take_top_level_only and level_idx > 0:
                break
            if bid_price < sell_fv or take_sell_capacity <= 0:
                break
            vol = order_depth.buy_orders[bid_price]
            qty = min(take_sell_capacity, vol)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                take_sell_capacity -= qty

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
        direction_score: float,
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

        buy_scale = 1.0
        sell_scale = 1.0
        suppress = cfg.get("passive_signal_suppress", 0.0)
        if direction_score > 0:
            sell_scale = max(0.0, 1.0 - suppress * direction_score)
        elif direction_score < 0:
            buy_scale = max(0.0, 1.0 + suppress * direction_score)

        # Post only on the profitable side of the skewed FV.
        if buy_capacity > 0 and our_bid < skewed_fv:
            qty = buy_capacity if max_qty is None else min(buy_capacity, max_qty)
            qty = int(math.floor(qty * buy_scale))
            if qty > 0:
                orders.append(Order(product, our_bid, qty))

        if sell_capacity > 0 and our_ask > skewed_fv:
            qty = sell_capacity if max_qty is None else min(sell_capacity, max_qty)
            qty = int(math.floor(qty * sell_scale))
            if qty > 0:
                orders.append(Order(product, our_ask, -qty))

        return orders

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self, state: TradingState):
        try:
            persistent: Dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            persistent = {}

        last_timestamp = persistent.get("last_timestamp")
        if last_timestamp is not None and state.timestamp < last_timestamp:
            persistent = {}

        factor_engines_state = persistent.setdefault("factor_engines", {})
        direction_scores: Dict[str, float] = {}
        factor_values: Dict[str, Dict[str, float]] = {}

        # Update factor engines for products with use_factors=True.
        for product, cfg in PRODUCT_CONFIG.items():
            if not cfg.get("use_factors", False):
                continue
            if product not in state.order_depths:
                continue
            engine_dict = factor_engines_state.get(product)
            factor_cfg = cfg.get("factor_params", {})
            engine = (
                FactorEngine.from_dict(product, factor_cfg, engine_dict)
                if engine_dict else
                FactorEngine(product, factor_cfg)
            )
            ob = _make_ob_snapshot(state.timestamp, state.order_depths[product])
            if ob.mid <= 0:
                continue
            reference_price = self.get_base_fv(product, state.order_depths[product], persistent, state.timestamp)
            if reference_price is None:
                continue
            engine.update(ob, reference_price)
            direction_scores[product] = engine.compute_direction_score()
            factor_values[product] = dict(engine.factors)
            factor_engines_state[product] = engine.to_dict()

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            cfg = PRODUCT_CONFIG.get(product, {})

            limit    = cfg.get("limit", DEFAULT_LIMIT)
            tiers    = cfg.get("skew_tiers", [(DEFAULT_LIMIT, 0.0)])
            position = state.position.get(product, 0)
            direction_score = direction_scores.get(product, 0.0)
            target_long_position = self.target_core_position(product, cfg, persistent, state.timestamp)
            core_floor_position = self.effective_core_floor(
                product, cfg, persistent, position, target_long_position, direction_score
            )

            structural_fv = self.get_base_fv(product, order_depth, persistent, state.timestamp)
            if structural_fv is None:
                result[product] = orders
                continue

            take_buy_fv = structural_fv
            take_sell_fv = structural_fv
            side_weights = cfg.get("take_side_weights")
            if side_weights and product in factor_values:
                factors = factor_values[product]
                buy_score = 0.0
                sell_score = 0.0
                if side_weights.get("buy"):
                    buy_score = sum(
                        weight * max(factors.get(name, 0.0), 0.0)
                        for name, weight in side_weights["buy"].items()
                    )
                if side_weights.get("sell"):
                    sell_score = sum(
                        weight * max(-factors.get(name, 0.0), 0.0)
                        for name, weight in side_weights["sell"].items()
                    )
                buy_score = float(np.clip(buy_score, 0.0, 1.0))
                sell_score = float(np.clip(sell_score, 0.0, 1.0))
                if buy_score >= cfg.get("take_signal_threshold_buy", cfg.get("take_signal_threshold", 0.0)):
                    take_buy_fv = structural_fv + buy_score * cfg.get("take_fv_skew_buy", cfg.get("take_fv_skew", 0.0))
                if sell_score >= cfg.get("take_signal_threshold_sell", cfg.get("take_signal_threshold", 0.0)):
                    take_sell_fv = structural_fv - sell_score * cfg.get("take_fv_skew_sell", cfg.get("take_fv_skew", 0.0))
            else:
                take_signal = direction_score
                if abs(take_signal) < cfg.get("take_signal_threshold", 0.0):
                    take_signal = 0.0
                take_buy_fv = structural_fv + take_signal * cfg.get("take_fv_skew", 0.0)
                take_sell_fv = take_buy_fv
            quote_fv = structural_fv + direction_score * cfg.get("quote_fv_skew", 0.0)
            sf        = self.skew_factor(position, tiers)
            skewed_fv = quote_fv - sf * position

            buy_capacity  = limit - position
            if core_floor_position is None:
                sell_capacity = limit + position
            else:
                sell_capacity = max(position - core_floor_position, 0)

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            spread   = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
            is_wide  = True if spread is None else spread >= cfg.get("wide_spread_threshold", 10)

            # Step 0 — build the structural core long gradually first.
            core_orders, buy_capacity = self.core_position_orders(
                product,
                order_depth,
                position,
                buy_capacity,
                core_floor_position,
                cfg.get("core_build_per_tick"),
            )
            orders.extend(core_orders)

            # Step 1 — aggressive taking is alpha-driven, not inventory-driven.
            taken, buy_capacity, sell_capacity = self.take_orders(
                product, order_depth, take_buy_fv, take_sell_fv,
                buy_capacity, sell_capacity, position,
                cfg.get("spread_aware_taking", False), is_wide,
            )
            orders.extend(taken)

            # Step 2 — passive making
            orders.extend(self.make_orders(
                product, order_depth, quote_fv, skewed_fv,
                buy_capacity, sell_capacity, position, cfg, is_wide,
                direction_score,
            ))

            result[product] = orders

        persistent["last_timestamp"] = state.timestamp
        return result, 0, json.dumps(persistent)
