from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import json
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
TIMESTAMPS_PER_DAY = 1_000_000
ROUND3_TTE_START = 5
TRADING_DAYS_PER_YEAR = 252

# basis_arb=True: deep-ITM, fair = S-K, quote ±1, take ±2 (req 8)
# buy_cap_mult / sell_cap_mult: per-strike long/short bias (req 9)
# max_take_qty=0: skip entirely (req 10)
OPTION_CFG: Dict[str, Dict] = {
    "VEV_4000": {"basis_arb": True,  "basis_make_spread": 5, "basis_take_threshold": 7,
                 "max_make_qty": 15, "max_take_qty": 15,
                 "do_make": True,  "buy_cap_mult": 1.0, "sell_cap_mult": 1.0},
    "VEV_4500": {"basis_arb": True,  "basis_make_spread": 4, "basis_take_threshold": 6,
                 "max_make_qty": 15, "max_take_qty": 15,
                 "do_make": True,  "buy_cap_mult": 1.0, "sell_cap_mult": 1.0},
    "VEV_5000": {"basis_arb": False, "take_threshold": 2.0, "max_take_qty": 20,
                 "do_make": True,  "max_make_qty": 20, "buy_cap_mult": 1.0, "sell_cap_mult": 1.0},
    "VEV_5100": {"basis_arb": False, "take_threshold": 2.0, "max_take_qty": 20,
                 "do_make": True,  "max_make_qty": 20, "buy_cap_mult": 1.0, "sell_cap_mult": 1.0},
    "VEV_5200": {"basis_arb": False, "take_threshold": 3.0, "max_take_qty": 25,
                 "do_make": False, "max_make_qty": 0,  "buy_cap_mult": 1.0, "sell_cap_mult": 1.0},
    "VEV_5300": {"basis_arb": False, "take_threshold": 4.0, "max_take_qty": 20,
                 "do_make": False, "max_make_qty": 0,  "buy_cap_mult": 0.5, "sell_cap_mult": 1.5},
    "VEV_5400": {"basis_arb": False, "take_threshold": 2.0, "max_take_qty": 20,
                 "do_make": False, "max_make_qty": 0,  "buy_cap_mult": 1.5, "sell_cap_mult": 0.5},
    "VEV_5500": {"basis_arb": False, "take_threshold": 1.0, "max_take_qty": 15,
                 "do_make": False, "max_make_qty": 0,  "buy_cap_mult": 0.5, "sell_cap_mult": 1.5},
    "VEV_6000": {"basis_arb": False, "take_threshold": 1.0, "max_take_qty": 0,
                 "do_make": False, "max_make_qty": 0,  "buy_cap_mult": 1.0, "sell_cap_mult": 1.0},
    "VEV_6500": {"basis_arb": False, "take_threshold": 1.0, "max_take_qty": 0,
                 "do_make": False, "max_make_qty": 0,  "buy_cap_mult": 1.0, "sell_cap_mult": 1.0},
}

# ── Per-product configuration ─────────────────────────────────────────────────
# skew_mode="tick": tier value is an absolute tick offset, not a multiplier.
#   skewed_fv = quote_fv - sign(position) * tier_value
# take_no_buy_above_fv / take_no_sell_below_fv: FV-relative hard guards on taking.
# hedge_pressure: FV -= hedge_pressure * net_delta (VFE only, req 7)
PRODUCT_CONFIG = {
    "HYDROGEL_PACK": {
        "limit": 200,
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
            "stretch_window": 10, "obi_smooth": 1,
            "anchor_scale": 4.0, "stretch_scale": 3.5, "micro_scale": 2.5,
            "weights": {"anchor_gap": 0.20, "stretch": 0.45, "obi": 0.20,
                        "depth_pressure": 0.00, "micro_gap": 0.15},
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
        "take_no_buy_above_fv": 10.0,   # req 6
        "take_no_sell_below_fv": -10.0, # req 6
    },
    "VELVETFRUIT_EXTRACT": {
        "limit": 200,
        "fair_value": "midwall",
        "ema_alpha": 1.0,            # raw mid, prevents EMA-lag adverse taking
        "wide_spread_threshold": 8,
        "min_passive_spread": 3,
        "use_factors": False,
        "take_fv_skew": 0.0,
        "quote_fv_skew": 0.0,
        "take_signal_threshold": 1.0,
        "factor_params": {
            "stretch_window": 10, "obi_smooth": 1,
            "anchor_scale": 4.0, "stretch_scale": 3.5, "micro_scale": 2.5,
            "weights": {"anchor_gap": 0.20, "stretch": 0.45, "obi": 0.20,
                        "depth_pressure": 0.00, "micro_gap": 0.15},
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


# ── Factor Engine ─────────────────────────────────────────────────────────────

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


# ── Greeks ────────────────────────────────────────────────────────────────────

class Greeks:
    def _norm_cdf(self, x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def _norm_pdf(self, x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def _d1(self, S: float, K: float, T: float, sigma: float) -> float:
        return (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))

    def delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """T in years, sigma annual. Returns call delta in [0,1]."""
        if T <= 1e-9 or S <= 0 or K <= 0 or sigma <= 0:
            return 1.0 if S > K else (0.5 if abs(S - K) < 1e-9 else 0.0)
        return self._norm_cdf(self._d1(S, K, T, sigma))

    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """T in years, sigma annual. Returns price sensitivity per unit annual sigma."""
        if T <= 1e-9 or S <= 0 or K <= 0 or sigma <= 0:
            return 0.0
        return S * self._norm_pdf(self._d1(S, K, T, sigma)) * math.sqrt(T)

    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """T in years, sigma annual."""
        if T <= 1e-9 or S <= 0 or K <= 0 or sigma <= 0:
            return 0.0
        return self._norm_pdf(self._d1(S, K, T, sigma)) / (S * sigma * math.sqrt(T))

    def implied_vol(self, price: float, S: float, K: float, T_days: float) -> Optional[float]:
        """Bisect to find annual sigma. Returns per-day IV (annual / sqrt(252))."""
        if T_days <= 0 or S <= 0 or K <= 0:
            return None
        T_years = T_days / TRADING_DAYS_PER_YEAR
        intrinsic = max(S - K, 0.0)
        if price < intrinsic + 1e-9:
            return None

        def bs_price(sig: float) -> float:
            if sig <= 1e-9:
                return intrinsic
            sqrtT = math.sqrt(T_years)
            d1 = (math.log(S / K) + 0.5 * sig ** 2 * T_years) / (sig * sqrtT)
            d2 = d1 - sig * sqrtT
            return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

        lo, hi = 1e-6, 5.0
        if bs_price(hi) < price:
            return None
        if bs_price(lo) > price:
            return None
        for _ in range(60):
            mid = (lo + hi) / 2.0
            v = bs_price(mid)
            if abs(v - price) < 1e-5:
                break
            if v < price:
                lo = mid
            else:
                hi = mid
        annual_sigma = (lo + hi) / 2.0
        return annual_sigma / math.sqrt(TRADING_DAYS_PER_YEAR)


# ── Volatility Engine ─────────────────────────────────────────────────────────

_SMILE_KEYS = [5000, 5100, 5200, 5300, 5400, 5500]
_SMILE_PRIORS: Dict[int, float] = {
    5000: 0.00000, 5100: 0.00000, 5200: 0.00000,
    5300: +0.00020, 5400: -0.00060, 5500: +0.00060,
}
_MID_HIST_LEN = 2000


class VolEngine:
    def __init__(self) -> None:
        self._g = Greeks()
        self.iv_ema: Dict[int, float] = {K: 0.01260 for K in _SMILE_KEYS}
        self.chain_ema: float = 0.01260
        self.smile_bias: Dict[int, float] = dict(_SMILE_PRIORS)
        self.mid_history: deque = deque(maxlen=_MID_HIST_LEN)
        self.bootstrap_count: int = 0
        self.realized_vol_ema: float = 0.0130
        self._iv_history: Dict[int, deque] = {K: deque(maxlen=50) for K in _SMILE_KEYS}

    def update(
        self,
        vfe_mid: float,
        tte_days: float,
        option_mids: Dict[int, float],
        option_spreads: Optional[Dict[int, float]] = None,
    ) -> None:
        self.mid_history.append(vfe_mid)
        self.bootstrap_count += 1

        if len(self.mid_history) < 200:
            return

        T_years = max(tte_days, 1e-6) / TRADING_DAYS_PER_YEAR
        alpha_iv = 0.05

        for K in _SMILE_KEYS:
            price = option_mids.get(K)
            if price is None or price <= 0:
                continue
            iv = self._g.implied_vol(price, vfe_mid, K, tte_days)
            if iv is None:
                continue
            hist = self._iv_history[K]
            if len(hist) >= 10:
                arr = list(hist)
                mean_iv = sum(arr) / len(arr)
                std_iv = math.sqrt(sum((x - mean_iv) ** 2 for x in arr) / len(arr)) + 1e-9
                if abs(iv - self.iv_ema[K]) > 3.0 * std_iv:
                    continue
            hist.append(iv)
            self.iv_ema[K] = (1.0 - alpha_iv) * self.iv_ema[K] + alpha_iv * iv

        # Vega-weighted chain consensus
        weights: Dict[int, float] = {}
        for K in _SMILE_KEYS:
            if self.iv_ema[K] <= 0:
                continue
            annual_iv = self.iv_ema[K] * math.sqrt(TRADING_DAYS_PER_YEAR)
            v = self._g.vega(vfe_mid, K, T_years, max(annual_iv, 0.01))
            if v < 50:
                continue
            w = v
            if option_spreads and K in option_spreads and option_spreads[K] > 5:
                w *= 5.0 / option_spreads[K]
            weights[K] = w

        if weights:
            total_w = sum(weights.values())
            chain_raw = sum(weights[K] * self.iv_ema[K] for K in weights) / total_w
            alpha_chain = 0.20 if self.bootstrap_count < 500 else 0.02
            self.chain_ema = (1.0 - alpha_chain) * self.chain_ema + alpha_chain * chain_raw

        for K in _SMILE_KEYS:
            bias = self.iv_ema[K] - self.chain_ema
            self.smile_bias[K] = 0.98 * self.smile_bias[K] + 0.02 * bias

        # Two-scale realized vol (Zhang 2005 style, relative returns)
        h = np.array(list(self.mid_history))
        n = len(h)
        if n >= 200 and vfe_mid > 0:
            fast_ret = np.diff(h) / h[:-1]
            rv_fast = float(np.mean(fast_ret ** 2))
            stride = 100
            if n > stride:
                slow_ret = (h[stride:] - h[:-stride]) / h[:-stride]
                rv_slow = float(np.mean(slow_ret ** 2))
                daily_var = rv_slow * (10000.0 / stride) - rv_fast * 10000.0 * (stride / _MID_HIST_LEN)
                rv = math.sqrt(max(daily_var, 1e-8))
                self.realized_vol_ema = 0.90 * self.realized_vol_ema + 0.10 * rv

    def chain_iv(self) -> float:
        return self.chain_ema

    def realized_vol(self) -> float:
        return self.realized_vol_ema

    def fair_vol(self, K: int) -> float:
        """Returns per-day vol for strike K, clamped to [0.005, 0.05]."""
        raw = self.chain_ema + self.smile_bias.get(K, 0.0)
        return max(0.005, min(0.05, raw))

    def vol_premium(self) -> float:
        """chain_iv - realized_vol; positive = options rich vs realized."""
        return self.chain_ema - self.realized_vol_ema

    def to_dict(self) -> dict:
        return {
            "iv_ema": {str(k): v for k, v in self.iv_ema.items()},
            "chain_ema": self.chain_ema,
            "smile_bias": {str(k): v for k, v in self.smile_bias.items()},
            "mid_history": list(self.mid_history),
            "bootstrap_count": self.bootstrap_count,
            "realized_vol_ema": self.realized_vol_ema,
            "iv_history": {str(k): list(v) for k, v in self._iv_history.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VolEngine":
        ve = cls()
        if "iv_ema" in data:
            ve.iv_ema = {int(k): float(v) for k, v in data["iv_ema"].items()}
        if "chain_ema" in data:
            ve.chain_ema = float(data["chain_ema"])
        if "smile_bias" in data:
            ve.smile_bias = {int(k): float(v) for k, v in data["smile_bias"].items()}
        if "mid_history" in data:
            ve.mid_history = deque(data["mid_history"], maxlen=_MID_HIST_LEN)
        if "bootstrap_count" in data:
            ve.bootstrap_count = int(data["bootstrap_count"])
        if "realized_vol_ema" in data:
            ve.realized_vol_ema = float(data["realized_vol_ema"])
        if "iv_history" in data:
            ve._iv_history = {int(k): deque(v, maxlen=50) for k, v in data["iv_history"].items()}
        return ve


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def __init__(self) -> None:
        self._greeks = Greeks()
        self.vol_engine = VolEngine()

    # ── Black-Scholes ─────────────────────────────────────────────────────────

    def _norm_cdf(self, x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def bs_call(self, S: float, K: float, T_years: float, sigma: float) -> float:
        """European call (r=0). sigma is annual vol."""
        if T_years <= 1e-9 or S <= 0 or K <= 0:
            return max(S - K, 0.0)
        sqrtT = math.sqrt(T_years)
        d1 = (math.log(S / K) + sigma ** 2 / 2.0 * T_years) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

    # ── Option handlers ───────────────────────────────────────────────────────

    def _run_basis_arb_option(
        self,
        product: str,
        order_depth: OrderDepth,
        underlying_mid: float,
        K: int,
        position: int,
    ) -> List[Order]:
        """Deep-ITM: fair = S-K, quote bid@fair-1 / ask@fair+1, take ±2."""
        orders: List[Order] = []
        fair = max(float(underlying_mid - K), 0.0)
        cfg = OPTION_CFG[product]
        max_q = cfg.get("max_make_qty", 20)
        buy_cap = OPTION_LIMIT - position
        sell_cap = OPTION_LIMIT + position

        take_thr = cfg.get("basis_take_threshold", 2.0)
        make_spread = cfg.get("basis_make_spread", 1)

        for ask in sorted(order_depth.sell_orders):
            if ask >= fair - take_thr or buy_cap <= 0:
                break
            v = abs(order_depth.sell_orders[ask])
            q = min(buy_cap, v, max_q)
            if q > 0:
                orders.append(Order(product, ask, q))
                buy_cap -= q

        for bid in sorted(order_depth.buy_orders, reverse=True):
            if bid <= fair + take_thr or sell_cap <= 0:
                break
            v = order_depth.buy_orders[bid]
            q = min(sell_cap, v, max_q)
            if q > 0:
                orders.append(Order(product, bid, -q))
                sell_cap -= q

        our_bid = int(fair) - make_spread
        our_ask = int(fair) + make_spread
        if our_ask > our_bid and cfg.get("do_make", True):
            if buy_cap > 0:
                orders.append(Order(product, our_bid, min(buy_cap, max_q)))
            if sell_cap > 0:
                orders.append(Order(product, our_ask, -min(sell_cap, max_q)))

        return orders

    def run_option(
        self,
        product: str,
        order_depth: OrderDepth,
        underlying_mid: float,
        position: int,
        tte_days: float,
        net_delta: float,
    ) -> List[Order]:
        """BS-based option trading with dynamic vol, delta budget, vol-premium tilt."""
        orders: List[Order] = []
        if underlying_mid is None or underlying_mid <= 0:
            return orders

        K = OPTION_STRIKES[product]
        cfg = OPTION_CFG.get(product, {})

        if cfg.get("basis_arb", False):
            return self._run_basis_arb_option(product, order_depth, underlying_mid, K, position)

        if cfg.get("max_take_qty", 1) == 0 and not cfg.get("do_make", False):
            return orders

        T_years = max(tte_days, 1e-6) / TRADING_DAYS_PER_YEAR
        daily_vol = self.vol_engine.fair_vol(K)
        annual_sigma = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
        bs_fv = self.bs_call(underlying_mid, K, T_years, annual_sigma)

        base_thr = cfg.get("take_threshold", 2.0)
        take_threshold = base_thr

        max_take_qty = cfg.get("max_take_qty", 10)
        do_make = cfg.get("do_make", False)
        max_make_qty = cfg.get("max_make_qty", 0)

        delta_K = self._greeks.delta(underlying_mid, K, T_years, annual_sigma)

        # Per-strike bias multipliers (req 9)
        buy_cap_mult = float(cfg.get("buy_cap_mult", 1.0))
        sell_cap_mult = float(cfg.get("sell_cap_mult", 1.0))

        # Vol-premium tilt (req 2c)
        vp = self.vol_engine.vol_premium()
        if vp > 0.001:
            m = min(vp / 0.002, 1.0)
            sell_cap_mult *= (1.0 + m)
            buy_cap_mult *= max(1.0 - m, 0.5)
        elif vp < -0.001:
            m = min(-vp / 0.002, 1.0)
            buy_cap_mult *= (1.0 + m)
            sell_cap_mult *= max(1.0 - m, 0.5)

        buy_cap = int((OPTION_LIMIT - position) * buy_cap_mult)
        sell_cap = int((OPTION_LIMIT + position) * sell_cap_mult)
        running_delta = net_delta

        def delta_ok(extra: float) -> bool:
            return abs(running_delta + extra) < 180.0

        # Taking
        if order_depth.sell_orders and max_take_qty > 0:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price >= bs_fv - take_threshold or buy_cap <= 0:
                    break
                vol = abs(order_depth.sell_orders[ask_price])
                qty = min(buy_cap, vol, max_take_qty)
                while qty > 0 and not delta_ok(qty * delta_K):
                    qty -= 1
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_cap -= qty
                    running_delta += qty * delta_K

        if order_depth.buy_orders and max_take_qty > 0:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price <= bs_fv + take_threshold or sell_cap <= 0:
                    break
                vol = order_depth.buy_orders[bid_price]
                qty = min(sell_cap, vol, max_take_qty)
                while qty > 0 and not delta_ok(-qty * delta_K):
                    qty -= 1
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_cap -= qty
                    running_delta -= qty * delta_K

        # Making
        if do_make and max_make_qty > 0:
            best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
            if best_bid is not None and best_ask is not None and best_ask - best_bid >= 3:
                our_bid = min(best_bid + 1, int(bs_fv))
                our_ask = max(best_ask - 1, math.ceil(bs_fv))
                if our_bid < our_ask:
                    if buy_cap > 0 and our_bid < bs_fv:
                        q = min(buy_cap, max_make_qty)
                        if delta_ok(q * delta_K):
                            orders.append(Order(product, int(our_bid), q))
                    if sell_cap > 0 and float(our_ask) > bs_fv:
                        q = min(sell_cap, max_make_qty)
                        if delta_ok(-q * delta_K):
                            orders.append(Order(product, int(our_ask), -q))

        return orders

    def run_delta_hedge(
        self,
        order_depth: OrderDepth,
        vfe_position: int,
        net_delta: float,
    ) -> List[Order]:
        """Aggressively hedge net delta using VFE. Cross spread if |net_delta|>30."""
        orders: List[Order] = []
        abs_nd = abs(net_delta)
        if abs_nd <= 10.0:
            return orders

        limit = PRODUCT_CONFIG["VELVETFRUIT_EXTRACT"]["limit"]
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        if net_delta > 0:
            # Long delta → sell VFE
            sell_cap = limit + vfe_position
            qty = min(int(abs_nd / 3), sell_cap, 10)
            if qty > 0:
                if abs_nd > 30 and best_bid is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -qty))
                elif best_ask is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", best_ask - 1, -qty))
        else:
            # Short delta → buy VFE
            buy_cap = limit - vfe_position
            qty = min(int(abs_nd / 3), buy_cap, 10)
            if qty > 0:
                if abs_nd > 30 and best_ask is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, qty))
                elif best_bid is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", best_bid + 1, qty))

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
        for threshold, value in tiers:
            if abs(position) < threshold:
                return value
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

    def target_core_position(self, product, cfg, persistent, timestamp) -> Optional[int]:
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

    def effective_core_floor(self, product, cfg, persistent, position,
                             target_long_position, direction_score,
                             extreme_release_active=False) -> Optional[int]:
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
        soft_min = cfg.get("soft_min_long_position", min_long_position)
        release_thr = cfg.get("soft_floor_release_threshold")
        rebuild_thr = cfg.get("soft_floor_rebuild_threshold", release_thr)
        dir_soft = core_state.get(soft_key, False)
        if release_thr is not None and direction_score <= release_thr:
            dir_soft = True
        elif dir_soft and rebuild_thr is not None and direction_score >= rebuild_thr:
            dir_soft = False
        core_state[soft_key] = dir_soft or extreme_release_active
        if core_state.get(soft_key, False):
            return int(min(soft_min, min_long_position))
        return int(min_long_position)

    def update_extreme_release_state(self, product, cfg, persistent, raw_mid, structural_fv) -> bool:
        release_gap = cfg.get("extreme_release_gap")
        if release_gap is None or raw_mid is None:
            return False
        core_state = persistent.setdefault("core_build", {})
        active_key = product + "_extreme_soft_floor_active"
        rc_key = product + "_extreme_release_count"
        rbc_key = product + "_extreme_rebuild_count"
        rebuild_gap = cfg.get("extreme_rebuild_gap", release_gap / 2.0)
        release_window = cfg.get("extreme_release_window", 1)
        rebuild_window = cfg.get("extreme_rebuild_window", 1)
        gap = structural_fv - raw_mid
        active = core_state.get(active_key, False)
        if not active:
            if gap >= release_gap:
                core_state[rc_key] = core_state.get(rc_key, 0) + 1
            else:
                core_state[rc_key] = 0
            core_state[rbc_key] = 0
            if core_state[rc_key] >= release_window:
                core_state[active_key] = True
                active = True
        else:
            if gap <= rebuild_gap:
                core_state[rbc_key] = core_state.get(rbc_key, 0) + 1
            else:
                core_state[rbc_key] = 0
            if core_state[rbc_key] >= rebuild_window:
                core_state[active_key] = False
                core_state[rc_key] = 0
                core_state[rbc_key] = 0
                active = False
        return active

    def check_trend_abort(self, product, cfg, persistent, mid) -> bool:
        fast_alpha = cfg.get("trend_abort_fast_alpha")
        slow_alpha = cfg.get("trend_abort_slow_alpha")
        abort_gap = cfg.get("trend_abort_gap")
        if fast_alpha is None or slow_alpha is None or abort_gap is None or mid is None:
            return False
        abort_state = persistent.setdefault("trend_abort", {})
        fast_key = product + "_abort_fast_ema"
        slow_key = product + "_abort_slow_ema"
        active_key = product + "_abort_active"
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

    def get_base_fv(self, product, order_depth, persistent, timestamp) -> Optional[float]:
        cfg = PRODUCT_CONFIG.get(product, {})
        fv_cfg = cfg.get("fair_value", "midwall")
        ema_state = persistent.setdefault("ema", {})

        if isinstance(fv_cfg, (int, float)):
            base = float(fv_cfg)

        elif fv_cfg == "linear_trend":
            slope = cfg.get("trend_slope", 0.0)
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
            alpha = cfg.get("ema_alpha", 1.0)
            prev_ema = ema_state.get(product, mid)
            base = alpha * mid + (1 - alpha) * prev_ema
            ema_state[product] = base

        return base

    # ── Taking (aggressive) ───────────────────────────────────────────────────

    def take_orders(self, product, order_depth, buy_fv, sell_fv, buy_capacity,
                    sell_capacity, position, spread_aware, is_wide) -> Tuple[List[Order], int, int]:
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
                for idx, ask_price in enumerate(sorted(order_depth.sell_orders.keys())):
                    if take_top_level_only and idx > 0:
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
                for idx, bid_price in enumerate(sorted(order_depth.buy_orders.keys(), reverse=True)):
                    if take_top_level_only and idx > 0:
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

        for idx, ask_price in enumerate(sorted(order_depth.sell_orders.keys())):
            if take_top_level_only and idx > 0:
                break
            if ask_price > buy_take_threshold or take_buy_capacity <= 0:
                break
            vol = abs(order_depth.sell_orders[ask_price])
            qty = min(take_buy_capacity, vol)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                take_buy_capacity -= qty

        for idx, bid_price in enumerate(sorted(order_depth.buy_orders.keys(), reverse=True)):
            if take_top_level_only and idx > 0:
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

    def make_orders(self, product, order_depth, base_fv, skewed_fv, buy_capacity,
                    sell_capacity, position, cfg, is_wide, direction_score,
                    buy_qty_override=None, sell_qty_override=None) -> List[Order]:
        orders: List[Order] = []
        fv_spread = cfg.get("fv_spread", None)
        quote_offset = cfg.get("quote_offset", 0)
        max_qty = cfg.get("max_passive_qty")
        min_passive_spread = cfg.get("min_passive_spread")

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if (min_passive_spread is not None and best_bid is not None and best_ask is not None
                and (best_ask - best_bid) < min_passive_spread):
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
                abs_pos = abs(position)
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
                        our_ask = math.ceil(base_fv) if base_fv != int(base_fv) else int(base_fv) + 1
                    else:
                        our_bid = math.floor(base_fv) if base_fv != int(base_fv) else int(base_fv) - 1
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

        # Day tracking: detect timestamp reset → increment day_idx, preserve vol state
        last_ts = persistent.get("last_ts", -1)
        if last_ts > 0 and state.timestamp < last_ts:
            day_idx = persistent.get("day_idx", 0) + 1
            vol_state = persistent.get("vol_engine")
            persistent = {"day_idx": day_idx}
            if vol_state:
                persistent["vol_engine"] = vol_state
        else:
            day_idx = persistent.get("day_idx", 0)

        tte_start = ROUND3_TTE_START
        tte_days = max(
            (tte_start - day_idx) - state.timestamp / TIMESTAMPS_PER_DAY,
            1e-6,
        )
        T_years = tte_days / TRADING_DAYS_PER_YEAR

        # Restore VolEngine
        vol_state = persistent.get("vol_engine")
        self.vol_engine = VolEngine.from_dict(vol_state) if vol_state else VolEngine()

        # VFE mid
        vfe_mid: Optional[float] = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vfe_mid = self.raw_mid(state.order_depths["VELVETFRUIT_EXTRACT"])

        # Collect option mids and spreads for VolEngine
        option_mids: Dict[int, float] = {}
        option_spreads: Dict[int, float] = {}
        for prod, K in OPTION_STRIKES.items():
            if K not in _SMILE_KEYS:
                continue
            od = state.order_depths.get(prod)
            if od is None:
                continue
            mid = self.raw_mid(od)
            if mid and mid > 0:
                option_mids[K] = mid
            if od.buy_orders and od.sell_orders:
                option_spreads[K] = float(min(od.sell_orders) - max(od.buy_orders))

        if vfe_mid and vfe_mid > 0:
            self.vol_engine.update(vfe_mid, tte_days, option_mids, option_spreads)

        # Pre-compute net delta from current positions (for hedge_pressure + delta budget)
        net_delta: float = float(state.position.get("VELVETFRUIT_EXTRACT", 0))
        if vfe_mid and vfe_mid > 0:
            for prod, K in OPTION_STRIKES.items():
                pos = state.position.get(prod, 0)
                if pos == 0:
                    continue
                dv = self.vol_engine.fair_vol(K)
                annual_s = dv * math.sqrt(TRADING_DAYS_PER_YEAR)
                d = self._greeks.delta(vfe_mid, K, T_years, annual_s)
                net_delta += pos * d

        # Factor engines
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
            engine = (FactorEngine.from_dict(product, factor_cfg, engine_dict)
                      if engine_dict else FactorEngine(product, factor_cfg))
            ob = _make_ob_snapshot(state.timestamp, state.order_depths[product])
            if ob.mid <= 0:
                continue
            reference_price = self.get_base_fv(product, state.order_depths[product],
                                                persistent, state.timestamp)
            if reference_price is None:
                continue
            engine.update(ob, reference_price)
            direction_scores[product] = engine.compute_direction_score()
            factor_values[product] = dict(engine.factors)
            factor_engines_state[product] = engine.to_dict()

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            # Options: BS or basis-arb handler
            if product in OPTION_STRIKES:
                if vfe_mid is None or vfe_mid <= 0:
                    result[product] = []
                    continue
                position = state.position.get(product, 0)
                result[product] = self.run_option(
                    product, order_depth, vfe_mid, position, tte_days, net_delta
                )
                continue

            orders: List[Order] = []
            cfg = PRODUCT_CONFIG.get(product, {})
            if not cfg:
                result[product] = orders
                continue

            limit = cfg.get("limit", DEFAULT_LIMIT)
            tiers = cfg.get("skew_tiers", [(DEFAULT_LIMIT, 0.0)])
            position = state.position.get(product, 0)
            direction_score = direction_scores.get(product, 0.0)
            target_long_position = self.target_core_position(product, cfg, persistent, state.timestamp)

            structural_fv = self.get_base_fv(product, order_depth, persistent, state.timestamp)
            if structural_fv is None:
                result[product] = orders
                continue

            # VFE: apply hedge-pressure shift to FV (req 7)
            if product == "VELVETFRUIT_EXTRACT":
                hp = cfg.get("hedge_pressure", 0.0)
                structural_fv -= hp * net_delta

            raw_m = self.raw_mid(order_depth)
            extreme_release_active = self.update_extreme_release_state(
                product, cfg, persistent, raw_m, structural_fv)
            core_floor_position = self.effective_core_floor(
                product, cfg, persistent, position, target_long_position,
                direction_score, extreme_release_active)
            trend_aborted = self.check_trend_abort(product, cfg, persistent, raw_m)
            if trend_aborted and core_floor_position is not None:
                core_floor_position = cfg.get("trend_abort_floor", 0)

            # FV for taking and quoting
            take_buy_fv = structural_fv
            take_sell_fv = structural_fv
            directional_take_active = False
            side_weights = cfg.get("take_side_weights")
            if side_weights and product in factor_values:
                factors = factor_values[product]
                buy_score = float(np.clip(
                    sum(w * max(factors.get(n, 0.0), 0.0) for n, w in side_weights.get("buy", {}).items()),
                    0.0, 1.0))
                sell_score = float(np.clip(
                    sum(w * max(-factors.get(n, 0.0), 0.0) for n, w in side_weights.get("sell", {}).items()),
                    0.0, 1.0))
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

            # Tick-based skew (reqs 6,7) vs legacy factor skew
            sf = self.skew_factor(position, tiers)
            if cfg.get("skew_mode") == "tick":
                pos_sign = 1 if position > 0 else (-1 if position < 0 else 0)
                skewed_fv = quote_fv - pos_sign * sf
            else:
                skewed_fv = quote_fv - sf * position

            if not directional_take_active:
                take_buy_fv = skewed_fv
                take_sell_fv = skewed_fv

            # Hard taking guards (req 6, HYDROGEL)
            if "take_no_buy_above_fv" in cfg:
                take_buy_fv = min(take_buy_fv, structural_fv + cfg["take_no_buy_above_fv"])
            if "take_no_sell_below_fv" in cfg:
                take_sell_fv = max(take_sell_fv, structural_fv + cfg["take_no_sell_below_fv"])

            buy_capacity = limit - position
            if core_floor_position is None:
                sell_capacity = limit + position
            else:
                sell_capacity = max(position - core_floor_position, 0)

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
            is_wide = True if spread is None else spread >= cfg.get("wide_spread_threshold", 10)

            core_orders, buy_capacity = self.core_position_orders(
                product, order_depth, structural_fv, position, buy_capacity,
                target_long_position, cfg.get("core_build_per_tick"),
                state.timestamp, cfg.get("trend_slope", 0.0), cfg.get("buy_to_ask_max_offset"),
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
                direction_score, buy_qty_override, sell_qty_override,
            ))
            result[product] = orders

        # Delta hedge disabled for testing

        # Market trade log — buyer/seller on every tick that has trades
        for prod, trades in state.market_trades.items():
            for t in trades:
                print(
                    f"[MKT ts={state.timestamp}] {prod} "
                    f"buyer={t.buyer} seller={t.seller} "
                    f"price={t.price} qty={t.quantity}"
                )

        # Periodic vol/delta summary every 1000 ticks
        if state.timestamp % 1000 == 0:
            biases = {str(k): round(v, 6) for k, v in self.vol_engine.smile_bias.items()}
            print(
                f"ts={state.timestamp} day={day_idx} tte={tte_days:.3f} "
                f"chain_iv={self.vol_engine.chain_iv():.5f} "
                f"rv={self.vol_engine.realized_vol():.5f} "
                f"vp={self.vol_engine.vol_premium():.5f} "
                f"net_delta={net_delta:.1f} biases={biases}"
            )

        # Persist vol debug info (req 11)
        persistent["vol_debug"] = {
            "chain_iv": self.vol_engine.chain_iv(),
            "net_delta": net_delta,
            "smile_biases": {str(k): v for k, v in self.vol_engine.smile_bias.items()},
        }
        persistent["vol_engine"] = self.vol_engine.to_dict()
        persistent["last_ts"] = state.timestamp

        return result, 0, json.dumps(persistent)
