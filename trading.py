from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import json
import os
from collections import deque
from dataclasses import dataclass
import numpy as np


# ── Option constants ──────────────────────────────────────────────────────────

OPTION_STRIKES: Dict[str, int] = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500,
}
OPTION_LIMIT = 300
OPTION_SIGMA = 0.20           # annual implied vol (calibrated: σ=0.20, T=days/252)
TIMESTAMPS_PER_DAY = 1_000_000
ROUND3_TTE_START = 5          # TTE in days at start of Round 3 sub-day 0
TRADING_DAYS_PER_YEAR = 252

# Per-option config: absolute take_threshold, max sizes, do_make flag.
OPTION_CFG: Dict[str, Dict] = {
    "VEV_4000": {"take_threshold": 5.0,  "max_take_qty": 15, "do_make": True,  "max_make_qty": 15},
    "VEV_4500": {"take_threshold": 4.0,  "max_take_qty": 15, "do_make": True,  "max_make_qty": 15},
    "VEV_5000": {"take_threshold": 2.0,  "max_take_qty": 20, "do_make": True,  "max_make_qty": 20},
    "VEV_5100": {"take_threshold": 2.0,  "max_take_qty": 20, "do_make": True,  "max_make_qty": 20},
    "VEV_5200": {"take_threshold": 3.0,  "max_take_qty": 25, "do_make": False, "max_make_qty": 0},
    "VEV_5300": {"take_threshold": 4.0,  "max_take_qty": 20, "do_make": False, "max_make_qty": 0},
    "VEV_5400": {"take_threshold": 2.0,  "max_take_qty": 20, "do_make": False, "max_make_qty": 0},
    "VEV_5500": {"take_threshold": 1.0,  "max_take_qty": 15, "do_make": False, "max_make_qty": 0},
    "VEV_6000": {"take_threshold": 0.5,  "max_take_qty": 10, "do_make": False, "max_make_qty": 0},
    "VEV_6500": {"take_threshold": 0.5,  "max_take_qty": 5,  "do_make": False, "max_make_qty": 0},
}


# ── Per-product configuration for delta-1 products ───────────────────────────
# fair_value:          fixed float | "linear_trend" | "midwall" | "ash_hybrid"
# skew_tiers:          tiered FV inventory skew — [(|pos|_threshold, factor), ...]
# quote_offset:        ticks from best bid/ask (negative = inside the book)
# min_passive_spread:  minimum market spread before posting passives
# size_imbalance_threshold / size_heavy_qty / size_light_qty:
#                      inventory-aware passive size overrides
PRODUCT_CONFIG = {
    "HYDROGEL_PACK": {
        "limit": 200,
        # Mean-reverts around 10000, spread ≈ 16 ticks — pure passive market maker.
        "fair_value": "ash_hybrid",
        "anchor_fv": 10000,
        "ema_alpha": 0.05,
        "ema_anchor_weight": 0.30,
        "mean_reversion_weight": 0.15,
        "mean_reversion_cap": 10.0,
        "last_move_reversion_weight": 0.20,
        "last_move_cap": 10.0,
        "microprice_weight": 0.15,
        "microprice_cap": 6.0,
        "wide_spread_threshold": 20,
        "use_factors": False,
        "take_fv_skew": 0.0,
        "quote_fv_skew": 0.0,
        "take_signal_threshold": 1.0,
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
        "skew_tiers": [(100, 0.0), (160, 0.015), (200, 0.03)],
        "asymmetric_quoting": False,
        "quote_offset": -1,
        "max_passive_qty": None,
        "min_passive_spread": 12,
        "size_imbalance_threshold": 160,
        "size_heavy_qty": 20,
        "size_light_qty": 8,
        "active_sell_can_open": True,
        "passive_signal_suppress": 0.0,
        "spread_aware_taking": False,
    },
    "VELVETFRUIT_EXTRACT": {
        "limit": 200,
        # Underlying for VEV options. Spread ≈ 5 ticks. Pure passive maker.
        # alpha=1.0 means FV = raw mid every tick, so EMA never lags and
        # take_fv = mid, preventing any aggressive taking (ask > mid always).
        "fair_value": "midwall",
        "ema_alpha": 1.0,
        "wide_spread_threshold": 8,
        "min_passive_spread": 3,
        "use_factors": False,
        "take_fv_skew": 0.0,
        "quote_fv_skew": 0.0,
        "take_signal_threshold": 1.0,
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
        "skew_tiers": [(200, 0.0)],
        "asymmetric_quoting": False,
        "quote_offset": -1,
        "max_passive_qty": None,
        "active_sell_can_open": True,
        "passive_signal_suppress": 0.0,
        "spread_aware_taking": False,
    },
}

DEFAULT_LIMIT = 200


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

    # ── Black-Scholes option pricing ──────────────────────────────────────────

    def _norm_cdf(self, x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def bs_call(self, S: float, K: float, T_years: float, sigma: float = OPTION_SIGMA) -> float:
        """European call (Black-Scholes, r=0). Calibrated: σ=0.20, T=TTE_days/252."""
        if T_years <= 1e-9 or S <= 0 or K <= 0:
            return max(S - K, 0.0)
        sqrtT = math.sqrt(T_years)
        d1 = (math.log(S / K) + sigma ** 2 / 2.0 * T_years) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

    def run_option(
        self,
        product: str,
        order_depth: OrderDepth,
        underlying_mid: Optional[float],
        position: int,
        tte_days: float,
    ) -> List[Order]:
        """Trade a VEV option using Black-Scholes fair value."""
        orders: List[Order] = []
        if underlying_mid is None or underlying_mid <= 0:
            return orders

        K = OPTION_STRIKES[product]
        cfg = OPTION_CFG.get(product, {})
        T_years = max(tte_days, 1e-6) / TRADING_DAYS_PER_YEAR
        bs_fv = self.bs_call(underlying_mid, K, T_years)

        take_threshold = cfg.get("take_threshold", 1.0)
        max_take_qty   = cfg.get("max_take_qty", 10)
        do_make        = cfg.get("do_make", False)
        max_make_qty   = cfg.get("max_make_qty", 0)

        buy_cap  = OPTION_LIMIT - position
        sell_cap = OPTION_LIMIT + position

        best_bid = max(order_depth.buy_orders)  if order_depth.buy_orders  else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        # Taking: cross when market is outside our BS threshold
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price >= bs_fv - take_threshold or buy_cap <= 0:
                    break
                vol = abs(order_depth.sell_orders[ask_price])
                qty = min(buy_cap, vol, max_take_qty)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_cap -= qty

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price <= bs_fv + take_threshold or sell_cap <= 0:
                    break
                vol = order_depth.buy_orders[bid_price]
                qty = min(sell_cap, vol, max_take_qty)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_cap -= qty

        # Making: quote one tick inside the market, anchored to BS FV
        if do_make and max_make_qty > 0 and best_bid is not None and best_ask is not None:
            if best_ask - best_bid >= 3:
                our_bid = min(best_bid + 1, int(bs_fv))
                our_ask = max(best_ask - 1, math.ceil(bs_fv))
                if our_bid < our_ask:
                    if buy_cap > 0 and our_bid < bs_fv:
                        qty = min(buy_cap, max_make_qty)
                        orders.append(Order(product, int(our_bid), qty))
                        buy_cap -= qty
                    if sell_cap > 0 and float(our_ask) > bs_fv:
                        qty = min(sell_cap, max_make_qty)
                        orders.append(Order(product, int(our_ask), -qty))

        return orders

    # ── Book helpers ──────────────────────────────────────────────────────────

    def clamp(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def raw_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return None
        if not order_depth.buy_orders:
            return float(min(order_depth.sell_orders.keys()))
        if not order_depth.sell_orders:
            return float(max(order_depth.buy_orders.keys()))
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

    def micro_dev_l1(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        bid = max(order_depth.buy_orders.keys())
        ask = min(order_depth.sell_orders.keys())
        bid_vol = abs(order_depth.buy_orders[bid])
        ask_vol = abs(order_depth.sell_orders[ask])
        total = bid_vol + ask_vol
        if total == 0:
            return None
        mid = (bid + ask) / 2
        micro = (ask * bid_vol + bid * ask_vol) / total
        return micro - mid

    def micro_dev_l2(self, order_depth: OrderDepth) -> Optional[float]:
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
        structural_fv: float,
        position: int,
        buy_capacity: int,
        target_long_position: Optional[int],
        core_build_per_tick: Optional[int],
        timestamp: int,
        trend_slope: float,
        buy_to_ask_max_offset: Optional[float],
        total_ticks: int = 100_000,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        if target_long_position is None or position >= target_long_position or not order_depth.sell_orders:
            return orders, buy_capacity

        need = min(target_long_position - position, buy_capacity)
        if core_build_per_tick is not None:
            need = min(need, core_build_per_tick)
        if need <= 0:
            return orders, buy_capacity

        buy_threshold = None
        if buy_to_ask_max_offset is not None:
            remaining_time = max(0, total_ticks - timestamp)
            dynamic_max = min(trend_slope * remaining_time, buy_to_ask_max_offset)
            buy_threshold = structural_fv + dynamic_max

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if buy_threshold is not None and ask_price > buy_threshold:
                break
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
        extreme_release_active: bool = False,
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
            if cfg.get("hold_min_long_during_build", False):
                return int(min_long_position)
            return target_long_position

        soft_min_long_position = cfg.get("soft_min_long_position", min_long_position)
        release_threshold = cfg.get("soft_floor_release_threshold")
        rebuild_threshold = cfg.get("soft_floor_rebuild_threshold", release_threshold)

        direction_soft_active = core_state.get(soft_key, False)
        if release_threshold is not None and direction_score <= release_threshold:
            direction_soft_active = True
        elif direction_soft_active and rebuild_threshold is not None and direction_score >= rebuild_threshold:
            direction_soft_active = False

        core_state[soft_key] = direction_soft_active or extreme_release_active

        if core_state.get(soft_key, False):
            return int(min(soft_min_long_position, min_long_position))
        return int(min_long_position)

    def update_extreme_release_state(
        self,
        product: str,
        cfg: Dict,
        persistent: Dict,
        raw_mid: Optional[float],
        structural_fv: float,
    ) -> bool:
        release_gap = cfg.get("extreme_release_gap")
        if release_gap is None or raw_mid is None:
            return False

        core_state = persistent.setdefault("core_build", {})
        active_key = product + "_extreme_soft_floor_active"
        release_count_key = product + "_extreme_release_count"
        rebuild_count_key = product + "_extreme_rebuild_count"

        rebuild_gap = cfg.get("extreme_rebuild_gap", release_gap / 2.0)
        release_window = cfg.get("extreme_release_window", 1)
        rebuild_window = cfg.get("extreme_rebuild_window", 1)
        gap = structural_fv - raw_mid
        active = core_state.get(active_key, False)

        if not active:
            if gap >= release_gap:
                core_state[release_count_key] = core_state.get(release_count_key, 0) + 1
            else:
                core_state[release_count_key] = 0
            core_state[rebuild_count_key] = 0

            if core_state[release_count_key] >= release_window:
                core_state[active_key] = True
                active = True
        else:
            if gap <= rebuild_gap:
                core_state[rebuild_count_key] = core_state.get(rebuild_count_key, 0) + 1
            else:
                core_state[rebuild_count_key] = 0

            if core_state[rebuild_count_key] >= rebuild_window:
                core_state[active_key] = False
                core_state[release_count_key] = 0
                core_state[rebuild_count_key] = 0
                active = False

        return active

    def check_trend_abort(
        self,
        product: str,
        cfg: Dict,
        persistent: Dict,
        mid: Optional[float],
    ) -> bool:
        fast_alpha = cfg.get("trend_abort_fast_alpha")
        slow_alpha = cfg.get("trend_abort_slow_alpha")
        abort_gap  = cfg.get("trend_abort_gap")
        if fast_alpha is None or slow_alpha is None or abort_gap is None or mid is None:
            return False

        abort_state = persistent.setdefault("trend_abort", {})
        fast_key    = product + "_abort_fast_ema"
        slow_key    = product + "_abort_slow_ema"
        active_key  = product + "_abort_active"

        fast_ema = fast_alpha * mid + (1 - fast_alpha) * abort_state.get(fast_key, mid)
        slow_ema = slow_alpha * mid + (1 - slow_alpha) * abort_state.get(slow_key, mid)
        abort_state[fast_key] = fast_ema
        abort_state[slow_key] = slow_ema

        recover_gap = cfg.get("trend_abort_recover_gap", abort_gap / 2.0)
        currently_aborted = abort_state.get(active_key, False)

        if not currently_aborted:
            if fast_ema < slow_ema - abort_gap:
                abort_state[active_key] = True
                return True
        else:
            if fast_ema >= slow_ema - recover_gap:
                abort_state[active_key] = False
                return False
            return True

        return False

    # ── Fair value ────────────────────────────────────────────────────────────

    def get_base_fv(
        self,
        product: str,
        order_depth: OrderDepth,
        persistent: Dict,
        timestamp: int,
    ) -> Optional[float]:
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

        elif fv_cfg == "ash_hybrid":
            mid = self.raw_mid(order_depth)
            if mid is None:
                return None

            alpha = cfg.get("ema_alpha", 0.05)
            prev_ema = ema_state.get(product, mid)
            ema = alpha * mid + (1 - alpha) * prev_ema
            ema_state[product] = ema

            anchor = cfg.get("anchor_fv", 10000.5)
            base = anchor + cfg.get("ema_anchor_weight", 0.25) * (ema - anchor)

            mean_reversion = self.clamp(anchor - mid, cfg.get("mean_reversion_cap", 4.0))
            base += cfg.get("mean_reversion_weight", 0.0) * mean_reversion

            last_mid_key = product + "_last_mid"
            last_mid = ema_state.get(last_mid_key)
            if last_mid is not None:
                last_move = self.clamp(mid - last_mid, cfg.get("last_move_cap", 6.0))
                base -= cfg.get("last_move_reversion_weight", 0.0) * last_move
            ema_state[last_mid_key] = mid

            micro_dev = self.micro_dev_l1(order_depth)
            if micro_dev is not None:
                base += cfg.get("microprice_weight", 0.0) * self.clamp(
                    micro_dev, cfg.get("microprice_cap", 4.0)
                )

        else:
            mid = self.raw_mid(order_depth)
            if mid is None:
                return None
            alpha    = cfg.get("ema_alpha", 1.0)
            prev_ema = ema_state.get(product, mid)
            base     = alpha * mid + (1 - alpha) * prev_ema
            ema_state[product] = base

        return base

    # ── Taking (aggressive) ───────────────────────────────────────────────────

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
        cfg = PRODUCT_CONFIG.get(product, {})
        orders: List[Order] = []
        if sell_fv is None:
            sell_fv = buy_fv
        max_take_qty = cfg.get("max_take_qty")
        buy_take_threshold = buy_fv
        sell_take_threshold = sell_fv
        can_open_short = cfg.get("active_sell_can_open", False)
        active_buy_ceiling = cfg.get("active_buy_ceiling")
        active_sell_floor = cfg.get("active_sell_floor")
        if active_buy_ceiling is not None:
            buy_take_threshold = min(buy_take_threshold, active_buy_ceiling)
        if active_sell_floor is not None:
            sell_take_threshold = max(sell_take_threshold, active_sell_floor)
        take_buy_capacity = buy_capacity if max_take_qty is None else min(buy_capacity, max_take_qty)
        take_sell_capacity = sell_capacity if max_take_qty is None else min(sell_capacity, max_take_qty)
        take_top_level_only = cfg.get("take_top_level_only", False)

        if spread_aware and not is_wide:
            if position < 0:
                for level_idx, ask_price in enumerate(sorted(order_depth.sell_orders.keys())):
                    if take_top_level_only and level_idx > 0:
                        break
                    if ask_price > buy_take_threshold or take_buy_capacity <= 0:
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
                    if bid_price < sell_take_threshold or take_sell_capacity <= 0:
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
            if ask_price > buy_take_threshold or take_buy_capacity <= 0:
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
            max_sell = take_sell_capacity if can_open_short else min(take_sell_capacity, max(position, 0))
            if bid_price < sell_take_threshold or take_sell_capacity <= 0 or max_sell <= 0:
                break
            vol = order_depth.buy_orders[bid_price]
            qty = min(max_sell, vol)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                take_sell_capacity -= qty

        return orders, buy_capacity, sell_capacity

    # ── Making (passive) ──────────────────────────────────────────────────────

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
        buy_qty_override: Optional[int] = None,
        sell_qty_override: Optional[int] = None,
    ) -> List[Order]:
        orders: List[Order] = []
        fv_spread    = cfg.get("fv_spread", None)
        quote_offset = cfg.get("quote_offset", 0)
        max_qty      = cfg.get("max_passive_qty")
        min_passive_spread = cfg.get("min_passive_spread")

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if (
            min_passive_spread is not None
            and best_bid is not None
            and best_ask is not None
            and (best_ask - best_bid) < min_passive_spread
        ):
            return orders

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
                    our_bid = conservative_bid
                    our_ask = conservative_ask

                elif abs_pos <= critical:
                    if position > 0:
                        our_bid = conservative_bid
                        our_ask = int(base_fv) + asym_off
                    else:
                        our_bid = int(base_fv) - asym_off
                        our_ask = conservative_ask

                else:
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

                if not is_wide:
                    our_bid = min(our_bid, conservative_bid)
                    our_ask = max(our_ask, conservative_ask)

                our_bid = max(our_bid, best_bid)
                our_ask = min(our_ask, best_ask)
            else:
                our_bid = conservative_bid
                our_ask = conservative_ask

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

        if buy_capacity > 0 and our_bid < skewed_fv:
            qty = buy_capacity if max_qty is None else min(buy_capacity, max_qty)
            if buy_qty_override is not None:
                qty = min(qty, buy_qty_override)
            qty = int(math.floor(qty * buy_scale))
            if qty > 0:
                orders.append(Order(product, our_bid, qty))

        if sell_capacity > 0 and our_ask > skewed_fv:
            qty = sell_capacity if max_qty is None else min(sell_capacity, max_qty)
            if sell_qty_override is not None:
                qty = min(qty, sell_qty_override)
            qty = int(math.floor(qty * sell_scale))
            if qty > 0:
                orders.append(Order(product, our_ask, -qty))

        return orders

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, state: TradingState):
        try:
            persistent: Dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            persistent = {}

        # Detect a new simulation day (timestamp resets to 0) and increment day_idx.
        # day_idx is preserved across the reset; all other state is cleared.
        last_ts = persistent.get("last_ts", -1)
        if last_ts > 0 and state.timestamp < last_ts:
            day_idx = persistent.get("day_idx", 0) + 1
            persistent = {"day_idx": day_idx}
        else:
            day_idx = persistent.get("day_idx", 0)

        # TTE in days, decreasing continuously as the simulation progresses.
        # PROSPERITY4BT_TTE_START is injected by the backtester for historical
        # data (day_0=TTE8, day_1=TTE7, day_2=TTE6); absent in live submission.
        _env_tte = os.environ.get("PROSPERITY4BT_TTE_START")
        tte_start = int(_env_tte) if _env_tte is not None else ROUND3_TTE_START
        tte_days = max(
            (tte_start - day_idx) - state.timestamp / TIMESTAMPS_PER_DAY,
            1e-6,
        )

        # Raw mid of the underlying for Black-Scholes pricing.
        vfe_mid: Optional[float] = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vfe_mid = self.raw_mid(state.order_depths["VELVETFRUIT_EXTRACT"])

        factor_engines_state = persistent.setdefault("factor_engines", {})
        direction_scores: Dict[str, float] = {}
        factor_values: Dict[str, Dict[str, float]] = {}

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

            # VEV options: Black-Scholes pricing, separate handler.
            if product in OPTION_STRIKES:
                position = state.position.get(product, 0)
                result[product] = self.run_option(
                    product, order_depth, vfe_mid, position, tte_days
                )
                continue

            orders: List[Order] = []
            cfg = PRODUCT_CONFIG.get(product, {})

            limit    = cfg.get("limit", DEFAULT_LIMIT)
            tiers    = cfg.get("skew_tiers", [(DEFAULT_LIMIT, 0.0)])
            position = state.position.get(product, 0)
            direction_score = direction_scores.get(product, 0.0)
            target_long_position = self.target_core_position(product, cfg, persistent, state.timestamp)

            structural_fv = self.get_base_fv(product, order_depth, persistent, state.timestamp)
            if structural_fv is None:
                result[product] = orders
                continue
            raw_mid = self.raw_mid(order_depth)
            extreme_release_active = self.update_extreme_release_state(
                product, cfg, persistent, raw_mid, structural_fv
            )
            core_floor_position = self.effective_core_floor(
                product,
                cfg,
                persistent,
                position,
                target_long_position,
                direction_score,
                extreme_release_active,
            )
            trend_aborted = self.check_trend_abort(product, cfg, persistent, raw_mid)
            if trend_aborted and core_floor_position is not None:
                core_floor_position = cfg.get("trend_abort_floor", 0)

            take_buy_fv = structural_fv
            take_sell_fv = structural_fv
            directional_take_active = False
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
                    directional_take_active = True
                if sell_score >= cfg.get("take_signal_threshold_sell", cfg.get("take_signal_threshold", 0.0)):
                    take_sell_fv = structural_fv - sell_score * cfg.get("take_fv_skew_sell", cfg.get("take_fv_skew", 0.0))
                    directional_take_active = True
            else:
                take_signal = direction_score
                if abs(take_signal) < cfg.get("take_signal_threshold", 0.0):
                    take_signal = 0.0
                take_buy_fv = structural_fv + take_signal * cfg.get("take_fv_skew", 0.0)
                take_sell_fv = take_buy_fv
                directional_take_active = take_signal != 0.0
            quote_fv = structural_fv + direction_score * cfg.get("quote_fv_skew", 0.0)
            sf        = self.skew_factor(position, tiers)
            skewed_fv = quote_fv - sf * position
            if not directional_take_active:
                take_buy_fv = skewed_fv
                take_sell_fv = skewed_fv

            buy_capacity  = limit - position
            if core_floor_position is None:
                sell_capacity = limit + position
            else:
                sell_capacity = max(position - core_floor_position, 0)

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            spread   = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
            is_wide  = True if spread is None else spread >= cfg.get("wide_spread_threshold", 10)

            core_orders, buy_capacity = self.core_position_orders(
                product,
                order_depth,
                structural_fv,
                position,
                buy_capacity,
                target_long_position,
                cfg.get("core_build_per_tick"),
                state.timestamp,
                cfg.get("trend_slope", 0.0),
                cfg.get("buy_to_ask_max_offset"),
            )
            orders.extend(core_orders)

            taken, buy_capacity, sell_capacity = self.take_orders(
                product, order_depth, take_buy_fv, take_sell_fv,
                buy_capacity, sell_capacity, position,
                cfg.get("spread_aware_taking", False), is_wide,
            )
            orders.extend(taken)

            buy_qty_override = None
            sell_qty_override = None
            imbalance_threshold = cfg.get("size_imbalance_threshold")
            if imbalance_threshold is not None and abs(position) >= imbalance_threshold:
                heavy = cfg.get("size_heavy_qty", 12)
                light = cfg.get("size_light_qty", 4)
                if position < 0:
                    buy_qty_override = heavy
                    sell_qty_override = light
                else:
                    buy_qty_override = light
                    sell_qty_override = heavy
            orders.extend(self.make_orders(
                product, order_depth, quote_fv, skewed_fv,
                buy_capacity, sell_capacity, position, cfg, is_wide,
                direction_score,
                buy_qty_override,
                sell_qty_override,
            ))

            result[product] = orders

        persistent["last_ts"] = state.timestamp
        return result, 0, json.dumps(persistent)
