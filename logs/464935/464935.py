from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import json
from collections import deque
from dataclasses import dataclass
import numpy as np


# ── Round 3 constants ────────────────────────────────────────────────────────

ROUND3_INITIAL_TTE = 5
# Timestamps go 0..999900 in steps of 100; one day spans 1_000_000 units.
TICKS_PER_DAY = 1_000_000
# Calibrated from historical ATM options across days 0-2: IV ≈ 0.015-0.018.
VEV_SIGMA = 0.0175
VEV_SIGMA_FALLBACK = 0.0175
VEV_IV_REFERENCE_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
VEV_STATIC_TIME_VALUE: Dict[int, float] = {
    5000: 4.5,
    5100: 16.5,
    5200: 45.5,
    5300: 47.0,
    5400: 16.5,
    5500: 6.5,
}
VEV_TIME_VALUE_OPTIONS: Tuple[int, ...] = (5300, 5400, 5500)
VEV_TIME_VALUE_MEAN: Dict[int, float] = {
    5300: 46.76,
    5400: 15.95,
    5500: 6.64,
}
VEV_TIME_VALUE_STD: Dict[int, float] = {
    5300: 6.23,
    5400: 3.43,
    5500: 1.74,
}
VEV_TIME_VALUE_ENTRY_Z: Dict[int, float] = {
    5300: 0.80,
    5400: 0.80,
    5500: 0.80,
}
VEV_TIME_VALUE_CAP: Dict[int, int] = {
    5300: 300,
    5400: 300,
    5500: 300,
}
VEV_TIME_VALUE_SHORT_Z: Dict[int, float] = {
    5300: 0.70,
    5400: 0.50,
    5500: 0.80,
}
VEV_TIME_VALUE_SHORT_CAP: Dict[int, int] = {
    5300: 300,
    5400: 300,
    5500: 300,
}
VEV_TIME_VALUE_QUOTE_EDGE: Dict[int, float] = {
    5300: 0.75,
    5400: 0.55,
    5500: 0.40,
}
VEV_TIME_VALUE_QUOTE_SIZE: Dict[int, int] = {
    5300: 18,
    5400: 18,
    5500: 20,
}
VEV_TIME_VALUE_INVENTORY_SKEW: Dict[int, float] = {
    5300: 2.4,
    5400: 1.4,
    5500: 0.8,
}
VEV_TIME_VALUE_MIN_OBS = 10
VEV_TIME_VALUE_STD_FLOOR_RATIO = 0.6
VEV_TV_EMA_ALPHA = 0.02
VEV_TV_PRIOR_WEIGHT = 0.50
VEV_OPTION_LIMIT = 300
VEV_OPTION_STRIKES: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}
DISABLE_VEV_OPTIONS = False

# HYDROGEL swing (mean-reversion) constants — from 463575
HYDRO_SWING_CAP = 200
HYDRO_SWING_SIZE = 40
HYDRO_LONG_ENTRY = 9950
HYDRO_LONG_EXIT = 9985
HYDRO_SHORT_ENTRY = 9980
HYDRO_SHORT_EXIT = 9960


# ── Product config ───────────────────────────────────────────────────────────

PRODUCT_CONFIG = {
    "HYDROGEL_PACK": {
        "limit": 200,
        "internal_limit": 90,
        "long_cap": 60,
        "short_cap": 24,
        "weak_long_cap": 24,
        "weak_short_cap": 24,
        "weak_mid_vs_anchor": 14.0,
        "weak_mid_vs_ema": 3.0,
        "weak_micro_threshold": -0.25,
        "weak_last_move": -1.2,
        "fair_value": "ash_hybrid",
        "anchor_fv": 9990.0,  # market mean is ~9990, not 10000
        "ema_alpha": 0.05,
        "ema_anchor_weight": 0.20,
        "mean_reversion_weight": 0.25,
        "mean_reversion_cap": 4.0,
        "last_move_reversion_weight": 0.25,
        "last_move_cap": 5.0,
        "microprice_weight": 0.30,
        "microprice_cap": 3.0,
        "wide_spread_threshold": 10,
        "use_factors": True,
        "factor_signal_weight": 0.80,
        "factor_params": {
            "stretch_window": 8,
            "obi_smooth": 1,
            "anchor_scale": 3.0,
            "stretch_scale": 2.5,
            "micro_scale": 2.0,
            "weights": {
                "anchor_gap": 0.05,
                "stretch": 0.25,
                "obi": 0.50,
                "depth_pressure": 0.00,
                "micro_gap": 0.20,
            },
        },
        "take_fv_skew": 2.0,
        "quote_fv_skew": 1.5,
        "take_signal_threshold": 0.15,
        "min_take_edge_ticks": 10.0,
        "take_edge_spread_ratio": 0.75,
        "skew_tiers": [(35, 0.0), (60, 0.08), (85, 0.20), (110, 0.36)],
        "max_take_qty": 6,
        "quote_offset": -1,
        "max_passive_qty": 8,
        "min_passive_spread": 10,
        "passive_signal_suppress": 0.35,
        "spread_aware_taking": False,
        "trend_guard": True,
        "trend_fast_alpha": 0.12,
        "trend_slow_alpha": 0.03,
        "trend_gap_threshold": 7.0,
        "trend_slope_threshold": 0.08,
        "trend_persist_ticks": 6,
        "trend_unwind_fv_boost": 8.0,
        "fill_cooldown_qty": 10,
        "fill_cooldown_ticks": 3000,
        "fill_decay_ticks": 2500,
        "de_risk_abs_pos": 50,
        "hard_de_risk_abs_pos": 70,
        "de_risk_fv_boost": 8.0,
        "peak_drawdown_threshold": 18.0,
        "peak_memory_ticks": 12000,
        "passive_buy_inventory_cap": 18,
        "passive_sell_inventory_cap": 28,
        "breakdown_lock_ticks": 28,
        "breakdown_start_pos": 18,
        "breakdown_mid_vs_ema": 4.0,
        "breakdown_micro_threshold": -0.8,
        "breakdown_last_move": -1.5,
        "breakdown_release_vs_ema": 1.0,
    },
    "VELVETFRUIT_EXTRACT": {
        "limit": 200,
        "internal_limit": 150,
        "disable_trading": False,
        "fair_value": "ash_hybrid",
        "anchor_fv": 5250.71,
        # Strong EMA tracking: FV ≈ EMA ≈ current market price.
        # Prevents FV from lagging below market and triggering spurious shorts.
        "ema_alpha": 0.15,
        "ema_anchor_weight": 0.85,
        "mean_reversion_weight": 0.0,
        "mean_reversion_cap": 15.0,
        "last_move_reversion_weight": 0.0,
        "last_move_cap": 6.0,
        "microprice_weight": 0.25,
        "microprice_cap": 2.0,
        "use_factors": False,
        "take_fv_skew": 0.0,
        "quote_fv_skew": 0.0,
        "take_signal_threshold": 1.0,   # effectively disables signal-based takes
        "min_take_edge_ticks": 4.5,
        "take_edge_spread_ratio": 0.90,
        "one_sided_take": False,
        "one_sided_quote": False,
        "active_sell_can_open": False,
        "passive_sell_can_open": True,
        "fv_spread": 2,
        "skew_tiers": [(50, 0.0), (90, 0.05), (120, 0.12), (150, 0.24)],
        "max_take_qty": 6,              # small takes only to reduce existing inventory
        "take_top_level_only": True,
        "max_passive_qty": 15,
        "passive_signal_suppress": 0.0,
        "trend_guard": True,
        "trend_fast_alpha": 0.05,
        "trend_slow_alpha": 0.015,
        "trend_gap_threshold": 6.0,
        "trend_slope_threshold": 0.04,
        "trend_persist_ticks": 15,
        "trend_unwind_fv_boost": 4.0,
        "spread_aware_taking": False,
    },
}

DEFAULT_LIMIT = 200
POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{symbol: VEV_OPTION_LIMIT for symbol in VEV_OPTION_STRIKES},
}


# ── Factor engine kept as utility ────────────────────────────────────────────

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
        trailing_mean = arr[-window - 1:-1].mean()
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
        engine._obi_history = deque(
            data.get("obi_history", []), maxlen=cfg["obi_smooth"] + 5
        )
        engine.factors = data.get("factors", {})
        return engine


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


class Trader:

    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def bs_call_price(S: float, K: int, T: float, sigma: float) -> float:
        if T <= 0:
            return max(0.0, S - K)
        sq = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
        d2 = d1 - sq
        return S * Trader.norm_cdf(d1) - K * Trader.norm_cdf(d2)

    @staticmethod
    def bs_implied_vol(price: float, S: float, K: int, T: float) -> Optional[float]:
        intrinsic = max(0.0, S - K)
        if T <= 0 or price <= intrinsic + 1e-9:
            return None

        lo = 1e-4
        hi = 0.20
        f_lo = Trader.bs_call_price(S, K, T, lo) - price
        f_hi = Trader.bs_call_price(S, K, T, hi) - price
        if f_lo * f_hi > 0:
            return None

        for _ in range(50):
            mid = 0.5 * (lo + hi)
            f_mid = Trader.bs_call_price(S, K, T, mid) - price
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo = mid
                f_lo = f_mid
        return 0.5 * (lo + hi)

    @staticmethod
    def get_tte(timestamp: int, day_count: int) -> float:
        return max(0.0, ROUND3_INITIAL_TTE - day_count - timestamp / TICKS_PER_DAY)

    def estimate_vev_sigma(
        self,
        state: TradingState,
        vev_mid: float,
        tte: float,
        persistent: Dict,
    ) -> float:
        ivs: List[float] = []
        for strike in VEV_IV_REFERENCE_STRIKES:
            symbol = f"VEV_{strike}"
            od = state.order_depths.get(symbol)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue
            mid_price = 0.5 * (
                max(od.buy_orders.keys()) + min(od.sell_orders.keys())
            )
            iv = self.bs_implied_vol(mid_price, vev_mid, strike, tte)
            if iv is not None:
                ivs.append(iv)

        sigma_state = persistent.setdefault("vev_sigma", {})
        prev_sigma = sigma_state.get("value", VEV_SIGMA_FALLBACK)
        if ivs:
            raw_sigma = float(np.median(np.array(ivs, dtype=float)))
            sigma = 0.35 * raw_sigma + 0.65 * prev_sigma
        else:
            sigma = prev_sigma

        sigma = float(np.clip(sigma, 0.013, 0.021))
        sigma_state["value"] = sigma
        sigma_state["sample_count"] = len(ivs)
        return sigma

    def option_mid_from_depth(self, order_depth: Optional[OrderDepth]) -> Optional[float]:
        if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return 0.5 * (
            max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())
        )

    def update_vev_time_value_memory(
        self,
        strike: int,
        option_mid: float,
        vev_mid: float,
        persistent: Dict,
    ) -> None:
        intrinsic = max(0.0, vev_mid - strike)
        time_value = option_mid - intrinsic
        tv_state = persistent.setdefault("vev_time_value", {}).setdefault(str(strike), {})

        prior_mean = VEV_TIME_VALUE_MEAN[strike]
        prior_var = VEV_TIME_VALUE_STD[strike] ** 2
        prev_mean = float(tv_state.get("mean", prior_mean))
        prev_var = float(tv_state.get("var", prior_var))
        diff = time_value - prev_mean

        new_mean = (1.0 - VEV_TV_EMA_ALPHA) * prev_mean + VEV_TV_EMA_ALPHA * time_value
        new_var = (1.0 - VEV_TV_EMA_ALPHA) * prev_var + VEV_TV_EMA_ALPHA * diff * diff

        tv_state["mean"] = new_mean
        tv_state["var"] = max(new_var, 1e-6)
        tv_state["count"] = int(tv_state.get("count", 0)) + 1

    def get_vev_adaptive_tv_stats(
        self,
        strike: int,
        persistent: Dict,
    ) -> Tuple[float, float, int]:
        prior_mean = VEV_TIME_VALUE_MEAN[strike]
        prior_std = VEV_TIME_VALUE_STD[strike]
        tv_state = persistent.get("vev_time_value", {}).get(str(strike), {})
        if not tv_state:
            return prior_mean, prior_std, 0

        live_mean = float(tv_state.get("mean", prior_mean))
        live_std = math.sqrt(max(1e-6, float(tv_state.get("var", prior_std * prior_std))))
        mean = VEV_TV_PRIOR_WEIGHT * prior_mean + (1.0 - VEV_TV_PRIOR_WEIGHT) * live_mean
        std_floor = prior_std * VEV_TIME_VALUE_STD_FLOOR_RATIO
        std = max(std_floor, live_std)
        count = int(tv_state.get("count", 0))
        return mean, max(std, 0.5), count

    def estimate_vev_curve_fair(
        self,
        strike: int,
        state: TradingState,
    ) -> Optional[float]:
        if strike not in VEV_IV_REFERENCE_STRIKES:
            return None

        mids: Dict[int, float] = {}
        for ref_strike in VEV_IV_REFERENCE_STRIKES:
            symbol = f"VEV_{ref_strike}"
            mid = self.option_mid_from_depth(state.order_depths.get(symbol))
            if mid is not None:
                mids[ref_strike] = mid

        if strike not in mids or len(mids) < 2:
            return None

        lower = sorted(k for k in mids if k < strike)
        upper = sorted(k for k in mids if k > strike)

        if lower and upper:
            left = lower[-1]
            right = upper[0]
            weight = (strike - left) / (right - left)
            return mids[left] + weight * (mids[right] - mids[left])

        if len(upper) >= 2:
            left = upper[0]
            right = upper[1]
            slope = (mids[right] - mids[left]) / (right - left)
            return mids[left] - slope * (left - strike)

        if len(lower) >= 2:
            left = lower[-2]
            right = lower[-1]
            slope = (mids[right] - mids[left]) / (right - left)
            return mids[right] + slope * (strike - right)

        return None

    def handle_vev_4000_parity(
        self,
        symbol: str,
        state: TradingState,
        vev_mid: float,
    ) -> List[Order]:
        od = state.order_depths.get(symbol)
        if od is None or not od.buy_orders or not od.sell_orders:
            return []

        position = state.position.get(symbol, 0)
        cap = 4
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        spread = best_ask - best_bid
        fair = max(0.0, vev_mid - 4000)
        fair_adj = fair - 10.0 * position / max(cap, 1)
        edge = max(6.5, 0.38 * spread)

        orders: List[Order] = []
        buy_cap = min(VEV_OPTION_LIMIT - position, max(0, cap - position))
        sell_cap = min(VEV_OPTION_LIMIT + position, max(0, cap + position))

        if best_ask <= fair_adj - (edge + 5.0) and buy_cap > 0:
            qty = min(abs(od.sell_orders[best_ask]), buy_cap, 1)
            if qty > 0:
                orders.append(Order(symbol, best_ask, qty))
                buy_cap -= qty

        if best_bid >= fair_adj + (edge + 5.0) and sell_cap > 0:
            qty = min(od.buy_orders[best_bid], sell_cap, 1)
            if qty > 0:
                orders.append(Order(symbol, best_bid, -qty))
                sell_cap -= qty

        if spread >= 12:
            bid_price = min(best_bid + 1, int(math.floor(fair_adj - edge)))
            ask_price = max(best_ask - 1, int(math.ceil(fair_adj + edge)))
            if bid_price < best_ask and fair_adj - bid_price >= 0.75 * edge and buy_cap > 0:
                orders.append(Order(symbol, bid_price, min(buy_cap, 1)))
            if ask_price > best_bid and ask_price - fair_adj >= 0.75 * edge and sell_cap > 0:
                orders.append(Order(symbol, ask_price, -min(sell_cap, 1)))

        return orders

    def handle_vev_time_value_option(
        self,
        symbol: str,
        strike: int,
        state: TradingState,
        vev_mid: float,
        persistent: Dict,
    ) -> List[Order]:
        od = state.order_depths.get(symbol)
        if od is None or not od.buy_orders or not od.sell_orders:
            return []

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])
        intrinsic = max(0.0, vev_mid - strike)
        ask_tv = best_ask - intrinsic
        bid_tv = best_bid - intrinsic
        mean, std, obs_count = self.get_vev_adaptive_tv_stats(strike, persistent)
        position = state.position.get(symbol, 0)

        orders: List[Order] = []
        long_cap = VEV_TIME_VALUE_CAP[strike]
        short_cap = VEV_TIME_VALUE_SHORT_CAP[strike]
        used_buy = 0
        used_sell = 0

        long_exit_tv = mean + 0.4 * std
        short_exit_tv = mean - 0.4 * std

        if position > 0 and bid_tv >= long_exit_tv:
            qty = min(position, bid_vol)
            if qty > 0:
                orders.append(Order(symbol, best_bid, -qty))
                used_sell += qty

        if position < 0 and ask_tv <= short_exit_tv:
            qty = min(-position, ask_vol)
            if qty > 0:
                orders.append(Order(symbol, best_ask, qty))
                used_buy += qty

        if obs_count < VEV_TIME_VALUE_MIN_OBS:
            return orders

        entry_z = VEV_TIME_VALUE_ENTRY_Z[strike]
        short_entry_z = VEV_TIME_VALUE_SHORT_Z[strike]
        long_entry_tv = mean - entry_z * std
        short_entry_tv = mean + short_entry_z * std

        buy_room = max(0, long_cap - position - used_buy)
        if ask_tv <= long_entry_tv and buy_room > 0:
            qty = min(ask_vol, buy_room)
            if qty > 0:
                orders.append(Order(symbol, best_ask, qty))
                used_buy += qty

        sell_room = max(0, short_cap + position - used_sell)
        if short_cap > 0 and bid_tv >= short_entry_tv and sell_room > 0:
            qty = min(bid_vol, sell_room)
            if qty > 0:
                orders.append(Order(symbol, best_bid, -qty))
                used_sell += qty

        fair_option = intrinsic + mean
        skew = VEV_TIME_VALUE_INVENTORY_SKEW[strike]
        cap_for_skew = max(1, max(long_cap, short_cap))
        fair_adj = fair_option - skew * position / cap_for_skew
        quote_edge = VEV_TIME_VALUE_QUOTE_EDGE[strike]
        quote_size = VEV_TIME_VALUE_QUOTE_SIZE[strike]
        spread = best_ask - best_bid

        bid_candidate = int(math.floor(fair_adj - quote_edge))
        ask_candidate = int(math.ceil(fair_adj + quote_edge))
        if spread > 1:
            quote_bid = min(best_bid + 1, bid_candidate)
            quote_ask = max(best_ask - 1, ask_candidate)
        else:
            quote_bid = min(best_bid, bid_candidate)
            quote_ask = max(best_ask, ask_candidate)

        passive_buy_room = max(0, long_cap - position - used_buy)
        if passive_buy_room > 0 and quote_bid >= 0 and quote_bid < best_ask and fair_adj - quote_bid >= 0.75 * quote_edge:
            qty = min(quote_size, passive_buy_room)
            if qty > 0:
                orders.append(Order(symbol, quote_bid, qty))

        passive_sell_room = max(0, short_cap + position - used_sell)
        if passive_sell_room > 0 and quote_ask > best_bid and quote_ask - fair_adj >= 0.75 * quote_edge:
            qty = min(quote_size, passive_sell_room)
            if qty > 0:
                orders.append(Order(symbol, quote_ask, -qty))

        return orders

    def handle_vev_option(
        self,
        symbol: str,
        strike: int,
        state: TradingState,
        vev_mid: float,
        tte: float,
        sigma: float,
        persistent: Dict,
    ) -> List[Order]:
        if symbol not in state.order_depths or tte <= 0:
            return []

        od = state.order_depths[symbol]
        if not od.buy_orders or not od.sell_orders:
            return []

        if strike == 4000:
            return self.handle_vev_4000_parity(symbol, state, vev_mid)
        if strike in VEV_TIME_VALUE_OPTIONS:
            return self.handle_vev_time_value_option(symbol, strike, state, vev_mid, persistent)

        position = state.position.get(symbol, 0)
        buy_cap = VEV_OPTION_LIMIT - position
        sell_cap = VEV_OPTION_LIMIT + position
        orders: List[Order] = []

        bs_theo = self.bs_call_price(vev_mid, strike, tte, sigma)

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        spread = best_ask - best_bid

        intrinsic = max(0.0, vev_mid - strike)
        curve_fair = self.estimate_vev_curve_fair(strike, state)
        static_time_value = VEV_STATIC_TIME_VALUE.get(strike)
        static_fair = None
        if static_time_value is not None:
            tv_scale = max(0.45, min(1.0, tte / ROUND3_INITIAL_TTE))
            static_fair = intrinsic + static_time_value * tv_scale

        candidates: List[Tuple[float, float]] = [(bs_theo, 0.20)]
        if curve_fair is not None and curve_fair > intrinsic:
            curve_weight = 0.55 if strike in (5100, 5200, 5300) else 0.65
            candidates.append((curve_fair, curve_weight))
        if static_fair is not None and static_fair > intrinsic:
            static_weight = 0.25 if strike in (5100, 5200, 5300) else 0.15
            candidates.append((static_fair, static_weight))

        total_weight = sum(weight for _, weight in candidates)
        theo = sum(price * weight for price, weight in candidates) / max(total_weight, 1e-9)

        time_value = theo - intrinsic

        if time_value < 1.5 or best_bid == 0:
            return []
        if strike <= 4500:
            return []

        if strike >= 6000:
            return []
        elif strike in (5000, 5100):
            soft_limit = 30
        elif strike in (5200, 5300):
            soft_limit = 10
        else:
            soft_limit = 40

        buy_cap = min(buy_cap, max(0, soft_limit - position))
        sell_cap = min(sell_cap, max(0, soft_limit + position))

        inv_guard = max(10, soft_limit // 2)
        if position >= inv_guard:
            buy_cap = 0
        if position <= -inv_guard:
            sell_cap = 0

        take_thresh = max(2.5, 0.65 * spread, min(5.0, 0.07 * max(theo, 1.0)))
        otm_buy_disabled = strike >= 5400 or strike in (5200, 5300)
        if not otm_buy_disabled and best_ask < theo - take_thresh and buy_cap > 0:
            qty = min(buy_cap, abs(od.sell_orders[best_ask]), 3)
            if qty > 0:
                orders.append(Order(symbol, best_ask, qty))
                buy_cap -= qty
        sell_thresh = max(take_thresh, min(5.0, 0.06 * max(theo, 1.0)))
        if strike >= 5400:
            sell_thresh = max(1.5, 0.30 * spread, 0.035 * max(theo, 1.0))
        if best_bid > theo + sell_thresh and sell_cap > 0:
            qty = min(sell_cap, od.buy_orders[best_bid], 3)
            if qty > 0:
                orders.append(Order(symbol, best_bid, -qty))
                sell_cap -= qty

        if position < 0 and best_ask <= theo + 1.5 and buy_cap > 0:
            orders.append(Order(symbol, best_ask, min(buy_cap, abs(position), 6)))
        elif position > 0 and best_bid >= theo - 1.5 and sell_cap > 0:
            orders.append(Order(symbol, best_bid, -min(sell_cap, position, 6)))

        return orders

    def get_spot_book_453001(self, order_depth: OrderDepth) -> Optional[Dict[str, float]]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        bid = max(order_depth.buy_orders.keys())
        ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders[bid]
        ask_vol = abs(order_depth.sell_orders[ask])
        if bid_vol <= 0 or ask_vol <= 0 or bid >= ask:
            return None
        mid = 0.5 * (bid + ask)
        micro = (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol)
        return {
            "best_bid": bid,
            "best_ask": ask,
            "bid_vol": bid_vol,
            "ask_vol": ask_vol,
            "mid": mid,
            "spread": ask - bid,
            "micro": micro,
        }

    def update_spot_memory_453001(
        self,
        persistent: Dict,
        product: str,
        book: Dict[str, float],
    ) -> None:
        item = persistent.setdefault("spot_453001", {}).setdefault(product, {})
        mid = float(book["mid"])
        if "ema" not in item:
            item["ema"] = mid
            item["slow"] = mid
            item["prev"] = mid
            item["last"] = mid
            item["vol"] = max(1.0, float(book["spread"]) / 2.0)
            return
        last = float(item.get("last", mid))
        item["prev"] = last
        item["ema"] = 0.10 * mid + 0.90 * float(item.get("ema", mid))
        item["slow"] = 0.01 * mid + 0.99 * float(item.get("slow", mid))
        item["vol"] = 0.08 * abs(mid - last) + 0.92 * float(item.get("vol", 1.0))
        item["last"] = mid

    def get_spot_fair_453001(
        self,
        product: str,
        book: Dict[str, float],
        persistent: Dict,
    ) -> float:
        item = persistent.get("spot_453001", {}).get(product, {})
        ema = float(item.get("ema", book["mid"]))
        slow = float(item.get("slow", ema))
        micro_edge = float(book["micro"]) - float(book["mid"])
        if product == "HYDROGEL_PACK":
            return book["mid"] + 0.35 * micro_edge + 0.04 * (ema - slow) + 0.025 * (9994.0 - slow)
        return book["mid"] + 0.20 * micro_edge + 0.04 * (ema - slow) + 0.015 * (5250.0 - slow)

    def add_order_453001(
        self,
        product: str,
        price: int,
        qty: int,
        position: int,
        used: Dict[str, int],
        orders: List[Order],
        cap: int,
        long_cap: Optional[int] = None,
        short_cap: Optional[int] = None,
    ) -> None:
        if qty == 0:
            return
        hard = POSITION_LIMITS.get(product, DEFAULT_LIMIT)
        long_limit = min(hard, cap if long_cap is None else long_cap)
        short_limit = min(hard, cap if short_cap is None else short_cap)
        used_buy = used.get("buy", 0)
        used_sell = used.get("sell", 0)
        if qty > 0:
            room = min(hard - position - used_buy, long_limit - position - used_buy)
            qty = min(qty, max(0, room))
            if qty <= 0:
                return
            used["buy"] = used_buy + qty
        else:
            sell_qty = -qty
            room = min(hard + position - used_sell, short_limit + position - used_sell)
            sell_qty = min(sell_qty, max(0, room))
            if sell_qty <= 0:
                return
            qty = -sell_qty
            used["sell"] = used_sell + sell_qty
        orders.append(Order(product, int(price), int(qty)))

    def passive_quote_453001(
        self,
        product: str,
        book: Dict[str, float],
        fair: float,
        edge: float,
        base_size: int,
        position: int,
        used: Dict[str, int],
        orders: List[Order],
        cap: int,
        allow_buy: bool = True,
        allow_sell: bool = True,
        buy_base_size: Optional[int] = None,
        sell_base_size: Optional[int] = None,
        long_cap: Optional[int] = None,
        short_cap: Optional[int] = None,
    ) -> None:
        buy_base = base_size if buy_base_size is None else buy_base_size
        sell_base = base_size if sell_base_size is None else sell_base_size
        buy_size = max(1, int(buy_base * max(0.25, 1.0 - max(position, 0) / max(1, cap))))
        sell_size = max(1, int(sell_base * max(0.25, 1.0 + min(position, 0) / max(1, cap))))

        bid_candidate = int(math.floor(fair - edge))
        ask_candidate = int(math.ceil(fair + edge))
        if book["spread"] > 1:
            bid_price = min(int(book["best_bid"]) + 1, bid_candidate)
            ask_price = max(int(book["best_ask"]) - 1, ask_candidate)
        else:
            bid_price = min(int(book["best_bid"]), bid_candidate)
            ask_price = max(int(book["best_ask"]), ask_candidate)

        if allow_buy and bid_price >= 0 and bid_price < int(book["best_ask"]) and fair - bid_price >= 0.75 * edge:
            self.add_order_453001(product, bid_price, buy_size, position, used, orders, cap, long_cap, short_cap)
        if allow_sell and ask_price > int(book["best_bid"]) and ask_price - fair >= 0.75 * edge:
            self.add_order_453001(product, ask_price, -sell_size, position, used, orders, cap, long_cap, short_cap)

    def trade_velvet_453001(
        self,
        product: str,
        book: Dict[str, float],
        persistent: Dict,
        position: int,
    ) -> List[Order]:
        fair = self.get_spot_fair_453001(product, book, persistent)
        cap = 90
        size = 8
        take_edge = max(4.5, 0.90 * book["spread"])
        passive_edge = max(1.5, 0.35 * book["spread"])
        fair_adj = fair - 5.0 * position / cap

        orders: List[Order] = []
        used = {"buy": 0, "sell": 0}
        if book["best_ask"] <= fair_adj - take_edge:
            self.add_order_453001(product, int(book["best_ask"]), min(max(2, size // 2), int(book["ask_vol"])), position, used, orders, cap)
        if book["best_bid"] >= fair_adj + take_edge:
            self.add_order_453001(product, int(book["best_bid"]), -min(max(2, size // 2), int(book["bid_vol"])), position, used, orders, cap)
        self.passive_quote_453001(product, book, fair_adj, passive_edge, size, position, used, orders, cap)
        return orders

    def trade_hydro_453001(
        self,
        product: str,
        book: Dict[str, float],
        persistent: Dict,
        position: int,
    ) -> List[Order]:
        item = persistent.get("spot_453001", {}).get(product, {})
        fair = self.get_spot_fair_453001(product, book, persistent)
        ema = float(item.get("ema", book["mid"]))
        slow = float(item.get("slow", ema))
        prev = float(item.get("prev", book["mid"]))

        cap = 80
        long_cap = 60
        base_short_cap = 16
        passive_edge = max(4.0, 0.34 * book["spread"])
        take_edge = max(10.0, 0.75 * book["spread"])
        base_size = 7

        short = position < 0
        lower_quantile = book["mid"] < 9961.0
        very_low = book["mid"] < 9948.0
        below_fair = book["mid"] < fair - 1.0
        below_ema = book["mid"] < ema - 2.0
        upward_turn = book["mid"] > prev + 1.0 and book["micro"] >= book["mid"]
        upward_reversal = upward_turn and (lower_quantile or below_ema)
        upper_regime = book["mid"] > 9994.0 + 18.0 or book["mid"] > ema + 4.0

        dynamic_short_cap = base_short_cap
        if lower_quantile or upward_reversal:
            dynamic_short_cap = 8
        if short and (very_low or (lower_quantile and (below_ema or upward_turn))):
            dynamic_short_cap = 6
        if upper_regime and not lower_quantile and not very_low:
            dynamic_short_cap = base_short_cap

        rebound_start = short and (
            (book["mid"] >= 9960.0 and prev < 9961.0 and upward_turn)
            or (
                book["mid"] >= 9968.0
                and ema > slow + 2.0
                and book["mid"] > slow + 5.0
                and book["mid"] >= ema - 1.0
            )
        )
        rebound_lock = int(item.get("rebound_lock", 0))
        if rebound_start:
            rebound_lock = 80
        elif rebound_lock > 0:
            if book["mid"] < ema - 2.5 and not upward_turn:
                rebound_lock = 0
            else:
                rebound_lock -= 1
        item["rebound_lock"] = rebound_lock
        persistent.setdefault("spot_453001", {}).setdefault(product, {})["rebound_lock"] = rebound_lock

        rebound_escape = short and position <= -6 and (rebound_start or rebound_lock > 0)
        rebound_sell_lock = short and rebound_lock > 0 and book["mid"] < 9998.0
        risky_add_short = short and position <= -dynamic_short_cap
        crowded_low_short = short and lower_quantile and position <= -6
        low_tail_short = short and very_low
        block_new_sells = (
            risky_add_short and (lower_quantile or below_fair or (below_ema and upward_turn))
        ) or crowded_low_short or low_tail_short or rebound_sell_lock

        skew = 12.0 if short else 9.0
        fair_adj = fair - skew * position / cap
        orders: List[Order] = []
        used = {"buy": 0, "sell": 0}

        if rebound_escape:
            escape_qty = min(3, max(0, -position - 4))
            escape_edge = max(5.0, 0.55 * book["spread"])
            if escape_qty > 0 and book["best_ask"] <= fair_adj + escape_edge:
                self.add_order_453001(
                    product,
                    int(book["best_ask"]),
                    min(escape_qty, int(book["ask_vol"])),
                    position,
                    used,
                    orders,
                    cap,
                    long_cap=long_cap,
                    short_cap=dynamic_short_cap,
                )

        buyback_needed = short and (
            (position <= -5 and very_low and book["micro"] >= book["mid"] - 0.5)
            or (position <= -6 and lower_quantile and (upward_reversal or book["micro"] >= book["mid"]))
        )
        if buyback_needed:
            buyback_qty = min(3 if position <= -8 else 2, -position)
            buyback_edge = max(2.0, 0.22 * book["spread"])
            if book["best_ask"] <= fair_adj + buyback_edge:
                self.add_order_453001(
                    product,
                    int(book["best_ask"]),
                    min(buyback_qty, int(book["ask_vol"])),
                    position,
                    used,
                    orders,
                    cap,
                    long_cap=long_cap,
                    short_cap=dynamic_short_cap,
                )

        if book["best_ask"] <= fair_adj - take_edge:
            self.add_order_453001(
                product,
                int(book["best_ask"]),
                min(max(2, base_size // 2), int(book["ask_vol"])),
                position,
                used,
                orders,
                cap,
                long_cap=long_cap,
                short_cap=dynamic_short_cap,
            )

        if not block_new_sells and book["best_bid"] >= fair_adj + take_edge:
            self.add_order_453001(
                product,
                int(book["best_bid"]),
                -min(max(2, base_size // 2), int(book["bid_vol"])),
                position,
                used,
                orders,
                cap,
                long_cap=long_cap,
                short_cap=dynamic_short_cap,
            )

        sell_base = 2 if (lower_quantile or upward_reversal) and position >= 0 else (1 if (lower_quantile or upward_reversal) else base_size)
        buy_base = base_size + 1 if short else base_size
        self.passive_quote_453001(
            product,
            book,
            fair_adj,
            passive_edge,
            base_size,
            position,
            used,
            orders,
            cap,
            allow_buy=True,
            allow_sell=not block_new_sells,
            buy_base_size=buy_base,
            sell_base_size=sell_base,
            long_cap=long_cap,
            short_cap=dynamic_short_cap,
        )
        return orders

    def trade_hydrogel_swing(
        self,
        book: Dict[str, float],
        position: int,
    ) -> List[Order]:
        """Pure price-level mean-reversion swing for HYDROGEL (463575 strategy)."""
        orders: List[Order] = []
        cap = HYDRO_SWING_CAP
        size = HYDRO_SWING_SIZE
        best_bid = int(book["best_bid"])
        best_ask = int(book["best_ask"])
        bid_vol = int(book["bid_vol"])
        ask_vol = int(book["ask_vol"])

        buy_used = 0
        sell_used = 0

        # Exit longs
        if position > 0 and best_bid >= HYDRO_LONG_EXIT:
            qty = min(position, bid_vol, size)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", best_bid, -qty))
                sell_used += qty

        # Exit shorts
        if position < 0 and best_ask <= HYDRO_SHORT_EXIT:
            qty = min(-position, ask_vol, size)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", best_ask, qty))
                buy_used += qty

        projected = position + buy_used - sell_used

        # Long entry
        if best_ask <= HYDRO_LONG_ENTRY and projected < cap:
            room = cap - projected
            qty = min(room, ask_vol, size)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", best_ask, qty))
                buy_used += qty

        projected = position + buy_used - sell_used

        # Short entry
        if best_bid >= HYDRO_SHORT_ENTRY and projected > -cap:
            room = cap + projected
            qty = min(room, bid_vol, size)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", best_bid, -qty))

        return orders

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

    def skew_factor(self, position: int, tiers: list) -> float:
        for threshold, factor in tiers:
            if abs(position) < threshold:
                return factor
        return tiers[-1][1]

    def get_signal_score(
        self,
        product: str,
        order_depth: OrderDepth,
        structural_fv: Optional[float] = None,
    ) -> float:
        cfg = PRODUCT_CONFIG.get(product, {})
        if cfg.get("signal_model") != "fv_gap":
            return 0.0
        mid = self.raw_mid(order_depth)
        if mid is None or structural_fv is None:
            return 0.0
        gap = structural_fv - mid
        deadband = cfg.get("signal_deadband_ticks", 0.0)
        if abs(gap) <= deadband:
            return 0.0
        scaled = (abs(gap) - deadband) / max(cfg.get("signal_scale_ticks", 1.0), 1e-9)
        return float(math.copysign(min(scaled, 1.0), gap))

    def get_trend_regime(
        self,
        product: str,
        cfg: Dict,
        raw_mid: Optional[float],
        persistent: Dict,
    ) -> int:
        if not cfg.get("trend_guard", False) or raw_mid is None:
            return 0

        trend_state = persistent.setdefault("trend_guard", {})
        fast_key = product + "_fast"
        slow_key = product + "_slow"
        up_key = product + "_up_count"
        down_key = product + "_down_count"

        fast_alpha = cfg.get("trend_fast_alpha", 0.05)
        slow_alpha = cfg.get("trend_slow_alpha", 0.015)
        prev_fast = trend_state.get(fast_key, raw_mid)
        prev_slow = trend_state.get(slow_key, raw_mid)

        fast = fast_alpha * raw_mid + (1 - fast_alpha) * prev_fast
        slow = slow_alpha * raw_mid + (1 - slow_alpha) * prev_slow
        trend_state[fast_key] = fast
        trend_state[slow_key] = slow

        gap = fast - slow
        slope = slow - prev_slow
        up_cond = gap > cfg.get("trend_gap_threshold", 6.0) and slope > cfg.get("trend_slope_threshold", 0.04)
        down_cond = gap < -cfg.get("trend_gap_threshold", 6.0) and slope < -cfg.get("trend_slope_threshold", 0.04)

        trend_state[up_key] = trend_state.get(up_key, 0) + 1 if up_cond else 0
        trend_state[down_key] = trend_state.get(down_key, 0) + 1 if down_cond else 0

        persist_ticks = cfg.get("trend_persist_ticks", 15)
        if trend_state[up_key] >= persist_ticks:
            return 1
        if trend_state[down_key] >= persist_ticks:
            return -1
        return 0

    def get_dynamic_contrarian_limit(
        self,
        cfg: Dict,
        direction_score: float,
    ) -> Optional[int]:
        max_limit = cfg.get("max_contrarian_abs_pos")
        if max_limit is None:
            return None
        min_limit = cfg.get("min_contrarian_abs_pos", max_limit)
        strength = float(np.clip(abs(direction_score), 0.0, 1.0))
        power = cfg.get("contrarian_limit_signal_power", 1.0)
        scaled_strength = strength ** max(power, 1e-9)
        dynamic_limit = min_limit + (max_limit - min_limit) * scaled_strength
        return int(round(dynamic_limit))

    def get_recent_fill_pressure(
        self,
        product: str,
        cfg: Dict,
        state: TradingState,
        persistent: Dict,
    ) -> Dict[str, bool]:
        cooldown_qty = cfg.get("fill_cooldown_qty")
        if cooldown_qty is None:
            return {"block_buy": False, "block_sell": False}

        fill_state = persistent.setdefault("fill_pressure", {}).setdefault(product, {})
        seen_keys = set(fill_state.get("seen_keys", []))
        buy_run = int(fill_state.get("buy_run_qty", 0))
        sell_run = int(fill_state.get("sell_run_qty", 0))
        buy_until = int(fill_state.get("block_buy_until", -1))
        sell_until = int(fill_state.get("block_sell_until", -1))
        last_fill_ts = fill_state.get("last_fill_ts")
        decay_ticks = int(cfg.get("fill_decay_ticks", 0))

        if last_fill_ts is not None and decay_ticks > 0 and state.timestamp - int(last_fill_ts) > decay_ticks:
            buy_run = 0
            sell_run = 0

        new_seen: List[str] = list(fill_state.get("seen_keys", []))
        signed_fill = 0
        for trade in state.own_trades.get(product, []):
            key = f"{trade.timestamp}:{trade.price}:{trade.quantity}:{trade.buyer}:{trade.seller}"
            if key in seen_keys:
                continue
            new_seen.append(key)
            if trade.buyer == "SUBMISSION":
                signed_fill += int(trade.quantity)
            elif trade.seller == "SUBMISSION":
                signed_fill -= int(trade.quantity)

        if signed_fill > 0:
            buy_run += signed_fill
            sell_run = 0
            buy_until = max(buy_until, state.timestamp + int(cfg.get("fill_cooldown_ticks", 0))) if buy_run >= cooldown_qty else buy_until
            last_fill_ts = state.timestamp
        elif signed_fill < 0:
            sell_run += abs(signed_fill)
            buy_run = 0
            sell_until = max(sell_until, state.timestamp + int(cfg.get("fill_cooldown_ticks", 0))) if sell_run >= cooldown_qty else sell_until
            last_fill_ts = state.timestamp

        fill_state["buy_run_qty"] = buy_run
        fill_state["sell_run_qty"] = sell_run
        fill_state["block_buy_until"] = buy_until
        fill_state["block_sell_until"] = sell_until
        fill_state["last_fill_ts"] = last_fill_ts
        fill_state["seen_keys"] = new_seen[-24:]

        return {
            "block_buy": state.timestamp < buy_until,
            "block_sell": state.timestamp < sell_until,
        }

    def get_peak_drawdown_regime(
        self,
        product: str,
        cfg: Dict,
        raw_mid: Optional[float],
        position: int,
        state: TradingState,
        persistent: Dict,
    ) -> int:
        threshold = cfg.get("peak_drawdown_threshold")
        if threshold is None or raw_mid is None:
            return 0

        peak_state = persistent.setdefault("peak_guard", {}).setdefault(product, {})
        peak = float(peak_state.get("peak_mid", raw_mid))
        trough = float(peak_state.get("trough_mid", raw_mid))
        peak_ts = int(peak_state.get("peak_ts", state.timestamp))
        trough_ts = int(peak_state.get("trough_ts", state.timestamp))
        memory_ticks = int(cfg.get("peak_memory_ticks", 0))

        if memory_ticks > 0 and state.timestamp - peak_ts > memory_ticks:
            peak = raw_mid
            peak_ts = state.timestamp
        if memory_ticks > 0 and state.timestamp - trough_ts > memory_ticks:
            trough = raw_mid
            trough_ts = state.timestamp

        if raw_mid >= peak:
            peak = raw_mid
            peak_ts = state.timestamp
        if raw_mid <= trough:
            trough = raw_mid
            trough_ts = state.timestamp

        peak_state["peak_mid"] = peak
        peak_state["trough_mid"] = trough
        peak_state["peak_ts"] = peak_ts
        peak_state["trough_ts"] = trough_ts

        if position > 0 and raw_mid <= peak - threshold:
            return -1
        if position < 0 and raw_mid >= trough + threshold:
            return 1
        return 0

    def get_breakdown_lock(
        self,
        product: str,
        cfg: Dict,
        order_depth: OrderDepth,
        raw_mid: Optional[float],
        position: int,
        state: TradingState,
        persistent: Dict,
    ) -> bool:
        if product != "HYDROGEL_PACK" or raw_mid is None:
            return False

        lock_ticks = int(cfg.get("breakdown_lock_ticks", 0))
        if lock_ticks <= 0:
            return False

        ema_state = persistent.get("ema", {})
        ema = float(ema_state.get(product, raw_mid))
        last_mid = float(ema_state.get(product + "_last_mid", raw_mid))
        micro_dev = self.micro_dev_l1(order_depth) or 0.0

        lock_state = persistent.setdefault("hydro_breakdown_lock", {})
        until = int(lock_state.get("until", -1))

        start_lock = (
            position >= int(cfg.get("breakdown_start_pos", 0))
            and raw_mid <= ema - cfg.get("breakdown_mid_vs_ema", 4.0)
            and micro_dev <= cfg.get("breakdown_micro_threshold", -0.8)
            and (raw_mid - last_mid) <= cfg.get("breakdown_last_move", -1.5)
        )
        if start_lock:
            until = max(until, state.timestamp + lock_ticks * 100)

        release_cond = (
            position <= max(10, int(cfg.get("breakdown_start_pos", 18)) // 2)
            or (raw_mid >= ema - cfg.get("breakdown_release_vs_ema", 1.0) and micro_dev >= -0.15)
        )
        if release_cond:
            until = -1

        lock_state["until"] = until
        return state.timestamp < until

    def get_dynamic_side_caps(
        self,
        product: str,
        cfg: Dict,
        order_depth: OrderDepth,
        raw_mid: Optional[float],
        persistent: Dict,
        trend_regime: int,
        breakdown_lock: bool,
    ) -> Tuple[Optional[int], Optional[int], bool]:
        long_cap = cfg.get("long_cap")
        short_cap = cfg.get("short_cap")
        if product != "HYDROGEL_PACK" or raw_mid is None:
            return long_cap, short_cap, False

        ema_state = persistent.get("ema", {})
        ema = float(ema_state.get(product, raw_mid))
        last_mid = float(ema_state.get(product + "_last_mid", raw_mid))
        micro_dev = self.micro_dev_l1(order_depth) or 0.0
        anchor = float(cfg.get("anchor_fv", raw_mid))

        weak_regime = (
            raw_mid <= anchor - float(cfg.get("weak_mid_vs_anchor", 12.0))
            or (
                raw_mid <= ema - float(cfg.get("weak_mid_vs_ema", 2.5))
                and micro_dev <= float(cfg.get("weak_micro_threshold", -0.25))
                and (raw_mid - last_mid) <= float(cfg.get("weak_last_move", -0.8))
            )
        )

        if weak_regime:
            long_cap = min(
                int(cfg.get("weak_long_cap", long_cap if long_cap is not None else DEFAULT_LIMIT)),
                int(cfg.get("internal_limit", cfg.get("limit", DEFAULT_LIMIT))),
            )
            weak_short_cap = cfg.get("weak_short_cap")
            if weak_short_cap is not None:
                short_cap = min(
                    int(weak_short_cap),
                    int(cfg.get("internal_limit", cfg.get("limit", DEFAULT_LIMIT))),
                )

        if trend_regime < 0 or breakdown_lock:
            long_cap = min(
                int(long_cap if long_cap is not None else DEFAULT_LIMIT),
                int(cfg.get("weak_long_cap", long_cap if long_cap is not None else DEFAULT_LIMIT)),
            )

        return long_cap, short_cap, weak_regime

    def get_base_fv(
        self,
        product: str,
        order_depth: OrderDepth,
        persistent: Dict,
    ) -> Optional[float]:
        cfg = PRODUCT_CONFIG.get(product, {})
        fv_cfg = cfg.get("fair_value")
        ema_state = persistent.setdefault("ema", {})

        if isinstance(fv_cfg, (int, float)):
            return float(fv_cfg)
        if fv_cfg != "ash_hybrid":
            return self.raw_mid(order_depth)

        mid = self.raw_mid(order_depth)
        if mid is None:
            return None

        alpha = cfg.get("ema_alpha", 0.05)
        prev_ema = ema_state.get(product, mid)
        ema = alpha * mid + (1 - alpha) * prev_ema
        ema_state[product] = ema

        anchor = cfg.get("anchor_fv", mid)
        base = anchor + cfg.get("ema_anchor_weight", 0.25) * (ema - anchor)
        base += cfg.get("mean_reversion_weight", 0.0) * self.clamp(
            anchor - mid, cfg.get("mean_reversion_cap", 4.0)
        )

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

        return base

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        buy_fv: float,
        sell_fv: float,
        buy_capacity: int,
        sell_capacity: int,
        position: int,
    ) -> Tuple[List[Order], int, int]:
        cfg = PRODUCT_CONFIG.get(product, {})
        orders: List[Order] = []
        max_take_qty = cfg.get("max_take_qty")
        take_buy_capacity = buy_capacity if max_take_qty is None else min(buy_capacity, max_take_qty)
        take_sell_capacity = sell_capacity if max_take_qty is None else min(sell_capacity, max_take_qty)
        take_top_level_only = cfg.get("take_top_level_only", False)
        can_open_short = cfg.get("active_sell_can_open", False)

        for level_idx, ask_price in enumerate(sorted(order_depth.sell_orders.keys())):
            if take_top_level_only and level_idx > 0:
                break
            if ask_price > buy_fv or take_buy_capacity <= 0:
                break
            qty = min(take_buy_capacity, abs(order_depth.sell_orders[ask_price]))
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                take_buy_capacity -= qty

        for level_idx, bid_price in enumerate(sorted(order_depth.buy_orders.keys(), reverse=True)):
            if take_top_level_only and level_idx > 0:
                break
            max_sell = take_sell_capacity if can_open_short else min(take_sell_capacity, max(position, 0))
            if bid_price < sell_fv or take_sell_capacity <= 0 or max_sell <= 0:
                break
            qty = min(max_sell, order_depth.buy_orders[bid_price])
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                take_sell_capacity -= qty

        return orders, buy_capacity, sell_capacity

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        base_fv: float,
        skewed_fv: float,
        buy_capacity: int,
        sell_capacity: int,
        position: int,
        direction_score: float,
        allow_passive_buy: bool = True,
        allow_passive_sell: bool = True,
    ) -> List[Order]:
        cfg = PRODUCT_CONFIG.get(product, {})
        orders: List[Order] = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return orders

        min_passive_spread = cfg.get("min_passive_spread")
        if min_passive_spread is not None and (best_ask - best_bid) < min_passive_spread:
            return orders

        fv_spread = cfg.get("fv_spread")
        if fv_spread is not None:
            fv_int = int(round(base_fv))
            our_bid = fv_int - fv_spread
            our_ask = fv_int + fv_spread
        else:
            quote_offset = cfg.get("quote_offset", 0)
            our_bid = best_bid - quote_offset
            our_ask = best_ask + quote_offset

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

        threshold = cfg.get("one_sided_quote_threshold", 0.0)
        if cfg.get("one_sided_quote", False):
            if direction_score > threshold:
                sell_scale = 0.0
            elif direction_score < -threshold:
                buy_scale = 0.0

        max_qty = cfg.get("max_passive_qty")
        if allow_passive_buy and buy_capacity > 0 and our_bid < skewed_fv:
            qty = buy_capacity if max_qty is None else min(buy_capacity, max_qty)
            qty = int(math.floor(qty * buy_scale))
            if qty > 0:
                orders.append(Order(product, our_bid, qty))

        if allow_passive_sell and sell_capacity > 0 and our_ask > skewed_fv:
            passive_sell_can_open = cfg.get("passive_sell_can_open", True)
            passive_sell_capacity = sell_capacity if passive_sell_can_open else min(sell_capacity, max(position, 0))
            passive_short_cap = cfg.get("passive_short_cap")
            if passive_short_cap is not None:
                passive_sell_capacity = min(passive_sell_capacity, max(0, int(passive_short_cap) + position))
            qty = passive_sell_capacity if max_qty is None else min(passive_sell_capacity, max_qty)
            qty = int(math.floor(qty * sell_scale))
            if qty > 0:
                orders.append(Order(product, our_ask, -qty))

        return orders

    def run(self, state: TradingState):
        try:
            persistent: Dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            persistent = {}

        last_timestamp = persistent.get("last_timestamp")
        if last_timestamp is not None and state.timestamp < last_timestamp:
            day_count = persistent.get("day_count", 0) + 1
            persistent = {"day_count": day_count}

        day_count = persistent.get("day_count", 0)
        persistent["day_count"] = day_count
        tte = self.get_tte(state.timestamp, day_count)

        vev_mid: Optional[float] = None
        vev_sigma: Optional[float] = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vev_mid = self.raw_mid(state.order_depths["VELVETFRUIT_EXTRACT"])
        if vev_mid is not None and tte > 0:
            vev_sigma = self.estimate_vev_sigma(state, vev_mid, tte, persistent)
            for strike in VEV_TIME_VALUE_OPTIONS:
                symbol = f"VEV_{strike}"
                option_mid = self.option_mid_from_depth(state.order_depths.get(symbol))
                if option_mid is not None:
                    self.update_vev_time_value_memory(strike, option_mid, vev_mid, persistent)

        spot_books_453001: Dict[str, Dict[str, float]] = {}
        for product in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"):
            od = state.order_depths.get(product)
            if od is None:
                continue
            book = self.get_spot_book_453001(od)
            if book is None:
                continue
            self.update_spot_memory_453001(persistent, product, book)
            spot_books_453001[product] = book

        factor_scores: Dict[str, float] = {}
        factor_state = persistent.setdefault("factor_engines", {})
        for product, cfg in PRODUCT_CONFIG.items():
            if not cfg.get("use_factors", False):
                continue
            if product not in state.order_depths:
                continue
            factor_cfg = cfg.get("factor_params")
            if not factor_cfg:
                continue
            engine_data = factor_state.get(product)
            engine = (
                FactorEngine.from_dict(product, factor_cfg, engine_data)
                if engine_data is not None
                else FactorEngine(product, factor_cfg)
            )
            ob = _make_ob_snapshot(state.timestamp, state.order_depths[product])
            if ob.mid <= 0:
                continue
            reference_price = self.get_base_fv(product, state.order_depths[product], persistent)
            if reference_price is None:
                continue
            engine.update(ob, reference_price)
            factor_scores[product] = engine.compute_direction_score()
            factor_state[product] = engine.to_dict()

        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if product in VEV_OPTION_STRIKES:
                if DISABLE_VEV_OPTIONS:
                    result[product] = []
                elif vev_mid is not None and vev_sigma is not None:
                    result[product] = self.handle_vev_option(
                        product,
                        VEV_OPTION_STRIKES[product],
                        state,
                        vev_mid,
                        tte,
                        vev_sigma,
                        persistent,
                    )
                else:
                    result[product] = []
                continue

            if product == "HYDROGEL_PACK":
                book = spot_books_453001.get(product)
                result[product] = (
                    self.trade_hydrogel_swing(book, state.position.get(product, 0))
                    if book is not None
                    else []
                )
                continue

            cfg = PRODUCT_CONFIG.get(product)
            if cfg is None or cfg.get("disable_trading", False):
                result[product] = []
                continue

            structural_fv = self.get_base_fv(product, order_depth, persistent)
            if structural_fv is None:
                result[product] = []
                continue

            position = state.position.get(product, 0)
            limit = cfg.get("internal_limit", cfg.get("limit", DEFAULT_LIMIT))
            base_signal_score = self.get_signal_score(product, order_depth, structural_fv)
            factor_score = factor_scores.get(product, 0.0)
            factor_weight = cfg.get("factor_signal_weight", 0.0) if cfg.get("use_factors", False) else 0.0
            direction_score = float(np.clip(
                (1.0 - factor_weight) * base_signal_score + factor_weight * factor_score,
                -1.0,
                1.0,
            ))
            raw_mid = self.raw_mid(order_depth)
            trend_regime = self.get_trend_regime(product, cfg, raw_mid, persistent)
            fill_pressure = self.get_recent_fill_pressure(product, cfg, state, persistent)
            peak_drawdown_regime = self.get_peak_drawdown_regime(
                product, cfg, raw_mid, position, state, persistent
            )
            breakdown_lock = self.get_breakdown_lock(
                product, cfg, order_depth, raw_mid, position, state, persistent
            )
            long_cap, short_cap, weak_regime = self.get_dynamic_side_caps(
                product, cfg, order_depth, raw_mid, persistent, trend_regime, breakdown_lock
            )

            take_buy_fv = structural_fv
            take_sell_fv = structural_fv
            take_signal = direction_score
            if abs(take_signal) < cfg.get("take_signal_threshold", 0.0):
                take_signal = 0.0

            if cfg.get("one_sided_take", False) and take_signal != 0.0:
                take_skew = abs(take_signal) * cfg.get("take_fv_skew", 0.0)
                if take_signal > 0:
                    take_buy_fv = structural_fv + take_skew
                    take_sell_fv = 10**9
                else:
                    take_buy_fv = -10**9
                    take_sell_fv = structural_fv - take_skew
            else:
                take_buy_fv = structural_fv + take_signal * cfg.get("take_fv_skew", 0.0)
                take_sell_fv = take_buy_fv

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
            if spread is not None:
                take_edge = max(
                    cfg.get("min_take_edge_ticks", 0.0),
                    cfg.get("take_edge_spread_ratio", 0.0) * spread,
                )
                take_buy_fv -= take_edge
                take_sell_fv += take_edge

            quote_fv = structural_fv + direction_score * cfg.get("quote_fv_skew", 0.0)
            skewed_fv = quote_fv - self.skew_factor(position, cfg.get("skew_tiers", [(limit, 0.0)])) * position

            buy_capacity = limit - position
            sell_capacity = limit + position
            if long_cap is not None:
                buy_capacity = min(buy_capacity, max(0, int(long_cap) - position))
            if short_cap is not None:
                sell_capacity = min(sell_capacity, max(0, int(short_cap) + position))

            contrarian_limit = self.get_dynamic_contrarian_limit(cfg, direction_score)
            if contrarian_limit is not None:
                if direction_score > 0 and position >= contrarian_limit:
                    buy_capacity = 0
                elif direction_score < 0 and position <= -contrarian_limit:
                    sell_capacity = 0

            if fill_pressure["block_buy"]:
                buy_capacity = 0
                if position > 0:
                    take_sell_fv = min(
                        take_sell_fv,
                        structural_fv - cfg.get("de_risk_fv_boost", 0.0),
                    )
                direction_score = min(direction_score, 0.0)
            if fill_pressure["block_sell"]:
                sell_capacity = 0
                if position < 0:
                    take_buy_fv = max(
                        take_buy_fv,
                        structural_fv + cfg.get("de_risk_fv_boost", 0.0),
                    )
                direction_score = max(direction_score, 0.0)

            de_risk_abs_pos = cfg.get("de_risk_abs_pos")
            hard_de_risk_abs_pos = cfg.get("hard_de_risk_abs_pos")
            if de_risk_abs_pos is not None:
                if position >= de_risk_abs_pos:
                    buy_capacity = 0
                    take_sell_fv = min(
                        take_sell_fv,
                        structural_fv - cfg.get("de_risk_fv_boost", 0.0),
                    )
                    direction_score = min(direction_score, 0.0)
                elif position <= -de_risk_abs_pos:
                    sell_capacity = 0
                    take_buy_fv = max(
                        take_buy_fv,
                        structural_fv + cfg.get("de_risk_fv_boost", 0.0),
                    )
                    direction_score = max(direction_score, 0.0)
            if hard_de_risk_abs_pos is not None:
                if position >= hard_de_risk_abs_pos:
                    buy_capacity = 0
                    sell_capacity = min(sell_capacity, position)
                    direction_score = -1.0
                elif position <= -hard_de_risk_abs_pos:
                    buy_capacity = min(buy_capacity, -position)
                    sell_capacity = 0
                    direction_score = 1.0

            flatten_bias_pos = cfg.get("flatten_bias_pos")
            flatten_bias_signal = cfg.get("flatten_bias_signal", 0.0)
            if (
                flatten_bias_pos is not None
                and abs(position) >= int(flatten_bias_pos)
                and abs(direction_score) <= flatten_bias_signal
            ):
                if position > 0:
                    buy_capacity = 0
                    sell_capacity = min(sell_capacity, position)
                    direction_score = min(direction_score, 0.0)
                elif position < 0:
                    buy_capacity = min(buy_capacity, -position)
                    sell_capacity = 0
                    direction_score = max(direction_score, 0.0)

            if peak_drawdown_regime > 0:
                buy_capacity = min(buy_capacity, max(0, -position))
                sell_capacity = 0
                direction_score = max(direction_score, 0.0)
                take_buy_fv = max(
                    take_buy_fv,
                    structural_fv + cfg.get("trend_unwind_fv_boost", 4.0),
                )
            elif peak_drawdown_regime < 0:
                buy_capacity = 0
                sell_capacity = min(sell_capacity, max(position, 0))
                direction_score = min(direction_score, 0.0)
                take_sell_fv = min(
                    take_sell_fv,
                    structural_fv - cfg.get("trend_unwind_fv_boost", 4.0),
                )

            if breakdown_lock:
                buy_capacity = 0
                sell_capacity = min(sell_capacity, max(position, 0))
                direction_score = -1.0
                take_sell_fv = min(
                    take_sell_fv,
                    structural_fv - cfg.get("trend_unwind_fv_boost", 4.0),
                )

            if trend_regime > 0:
                if position < 0:
                    buy_capacity = min(buy_capacity, -position)
                    sell_capacity = 0
                    direction_score = 1.0
                    take_buy_fv = max(
                        take_buy_fv,
                        structural_fv + cfg.get("trend_unwind_fv_boost", 4.0),
                    )
                elif position == 0:
                    buy_capacity = 0
                    sell_capacity = 0
                    direction_score = 0.0
                else:
                    buy_capacity = 0
                    sell_capacity = min(sell_capacity, position)
                    direction_score = 0.0
            elif trend_regime < 0:
                if position > 0:
                    buy_capacity = 0
                    sell_capacity = min(sell_capacity, position)
                    direction_score = -1.0
                    take_sell_fv = min(
                        take_sell_fv,
                        structural_fv - cfg.get("trend_unwind_fv_boost", 4.0),
                    )
                elif position == 0:
                    buy_capacity = 0
                    sell_capacity = 0
                    direction_score = 0.0
                else:
                    buy_capacity = min(buy_capacity, -position)
                    sell_capacity = 0
                    direction_score = 0.0

            quote_fv = structural_fv + direction_score * cfg.get("quote_fv_skew", 0.0)
            skewed_fv = quote_fv - self.skew_factor(position, cfg.get("skew_tiers", [(limit, 0.0)])) * position
            allow_passive_buy = True
            allow_passive_sell = True

            passive_buy_inventory_cap = cfg.get("passive_buy_inventory_cap")
            if (
                passive_buy_inventory_cap is not None
                and position >= passive_buy_inventory_cap
                and (trend_regime < 0 or peak_drawdown_regime < 0 or fill_pressure["block_buy"] or breakdown_lock or weak_regime)
            ):
                allow_passive_buy = False
            passive_sell_inventory_cap = cfg.get("passive_sell_inventory_cap")
            if (
                passive_sell_inventory_cap is not None
                and position <= -passive_sell_inventory_cap
                and (trend_regime > 0 or peak_drawdown_regime > 0 or fill_pressure["block_sell"])
            ):
                allow_passive_sell = False
            if breakdown_lock:
                allow_passive_buy = False
            if weak_regime and position > 0:
                allow_passive_buy = False

            taken, buy_capacity, sell_capacity = self.take_orders(
                product,
                order_depth,
                take_buy_fv,
                take_sell_fv,
                buy_capacity,
                sell_capacity,
                position,
            )
            orders = list(taken)
            is_wide = spread is None or spread >= cfg.get("wide_spread_threshold", 10)
            orders.extend(
                self.make_orders(
                    product,
                    order_depth,
                    quote_fv,
                    skewed_fv,
                    buy_capacity,
                    sell_capacity,
                    position,
                    direction_score,
                    allow_passive_buy=allow_passive_buy,
                    allow_passive_sell=allow_passive_sell,
                )
            )

            result[product] = orders

        persistent["last_timestamp"] = state.timestamp
        return result, 0, json.dumps(persistent)