"""
IMC Prosperity 4 Round 3 — single-file submission.

v2: Adds per-product FlowTracker for HYDROGEL_PACK and VELVETFRUIT_EXTRACT.
Flow imbalance biases fair value, capturing flow direction faster than the
EMA can. Large-trade imbalance separately tracks institutional-looking
counterparties that we may want to fade or follow.

Architecture (in order of expected PnL contribution on this regime):

  1. HYDROGEL_PACK aggressive swing-MM. Independent stationary process around
     ~9979 with ~50-tick swings. EMA fair value, full ±200 cycling, no
     anchor bias. Flow imbalance biases FV up to ±5 ticks.

  2. VELVETFRUIT_EXTRACT swing-MM. Same architecture, smaller swings (~25
     ticks), tighter book (5-tick spread). Doubles as delta hedge instrument.
     Flow imbalance biases FV up to ±2.5 ticks.

  3. Time-value mean reversion on VEV_5300 / 5400 / 5500. Per-strike EWMA
     of (mid - intrinsic) with z-score thresholds.

  4. Deep-ITM basis arb on VEV_4000. Fair = spot - 4000.

  5. Delta hedger via VFE.

Counterparty data observed empty in current regime — aggression imbalance
used as flow proxy. FlowTracker also exposes large_imbalance() for
fingerprinting big-size flows separately from retail-sized noise.
"""

from datamodel import Order, OrderDepth, TradingState, Trade
from typing import Any, Dict, List, Optional, Tuple
import json
import math


# =============================================================================
# FlowTracker — per-product aggression imbalance and counterparty fingerprinting
# =============================================================================
class FlowTracker:
    """
    Tracks aggressive trades on a single product to expose:
      - imbalance() in [-1, +1]: signed-volume aggression imbalance
      - large_imbalance() in [-1, +1]: same but only across trades >= threshold
      - count_large_buys / count_large_sells: persistent fingerprint of big flow

    A trade is buy-aggressive if it printed at or above the prior best ask;
    sell-aggressive if at or below prior best bid. Trades inside the spread
    are ignored as they don't represent decisive directional pressure.

    State is JSON-round-trippable via to_dict / from_dict.
    """

    def __init__(self, window: int = 200, large_threshold: int = 10):
        self.window = window
        self.large_threshold = large_threshold
        # Each entry: {ts, sign (+1/-1), qty, mid}
        self.trades: List[Dict[str, Any]] = []
        self.prev_bid: Optional[int] = None
        self.prev_ask: Optional[int] = None

    def observe(self, ts: int, market_trades: List[Trade], book: Dict[str, Any]) -> None:
        for t in market_trades or []:
            sign = 0
            if self.prev_ask is not None and t.price >= self.prev_ask:
                sign = +1
            elif self.prev_bid is not None and t.price <= self.prev_bid:
                sign = -1
            if sign != 0:
                self.trades.append({
                    "ts": ts, "sign": sign, "qty": int(t.quantity),
                    "mid": book["mid"],
                })
        # Trim to rolling window (by trade count, not timestamp)
        if len(self.trades) > self.window:
            self.trades = self.trades[-self.window:]
        # Update prior touch for next observation cycle
        self.prev_bid = book["best_bid"]
        self.prev_ask = book["best_ask"]

    def imbalance(self) -> float:
        if not self.trades:
            return 0.0
        s = sum(t["sign"] * t["qty"] for t in self.trades)
        v = sum(t["qty"] for t in self.trades)
        return s / v if v > 0 else 0.0

    def large_imbalance(self) -> float:
        """Imbalance restricted to trades of size >= large_threshold.
        Good for detecting persistent institutional sellers/buyers that
        retail-sized noise drowns out in the unweighted imbalance."""
        large = [t for t in self.trades if t["qty"] >= self.large_threshold]
        if not large:
            return 0.0
        s = sum(t["sign"] * t["qty"] for t in large)
        v = sum(t["qty"] for t in large)
        return s / v if v > 0 else 0.0

    def count_large_buys(self) -> int:
        return sum(1 for t in self.trades if t["sign"] > 0 and t["qty"] >= self.large_threshold)

    def count_large_sells(self) -> int:
        return sum(1 for t in self.trades if t["sign"] < 0 and t["qty"] >= self.large_threshold)

    def total_volume(self) -> int:
        return sum(t["qty"] for t in self.trades)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trades": self.trades,
            "prev_bid": self.prev_bid,
            "prev_ask": self.prev_ask,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]], window: int = 200,
                  large_threshold: int = 10) -> "FlowTracker":
        f = cls(window, large_threshold)
        if d:
            f.trades = list(d.get("trades", []))
            f.prev_bid = d.get("prev_bid")
            f.prev_ask = d.get("prev_ask")
        return f


# =============================================================================
# Trader
# =============================================================================
class Trader:
    LIMITS = {
        "HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300, "VEV_5100": 300,
        "VEV_5200": 300, "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
        "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
        "VEV_6000": 6000, "VEV_6500": 6500,
    }

    UNDERLYING = "VELVETFRUIT_EXTRACT"

    # ---------------------------------------------------------------- HYDROGEL_PACK
    HP_EMA_ALPHA = 0.001
    HP_FV_INIT = 9979.0
    HP_QUOTE_HALF = 5
    HP_QUOTE_SIZE = 18
    HP_TAKE_THRESHOLD = 8
    HP_HARD_GUARD = 12
    HP_LIMIT = 200
    HP_SKEW_TIERS = [(40, 0.5), (80, 1.5), (140, 3.0), (200, 6.0)]

    # Flow tracking on HP. 500-tick window matches the ~1000-tick mean-reversion
    # horizon (we want flow signal that's contemporaneous with our FV's frequency).
    HP_FLOW_WINDOW = 500
    HP_FLOW_LARGE_THRESHOLD = 10
    HP_AGG_FV_TICKS = 3.0          # max FV bias from saturated unweighted imbalance
    HP_LARGE_FV_TICKS = 2.0        # additional bias from large-trade imbalance

    # ---------------------------------------------------------------- VFE
    VFE_EMA_ALPHA = 0.005
    VFE_FV_INIT = 5260.0
    VFE_QUOTE_HALF = 1
    VFE_QUOTE_SIZE = 30
    VFE_TAKE_THRESHOLD = 2.0
    VFE_HARD_GUARD = 4
    VFE_LIMIT = 200
    VFE_SKEW_TIERS = [(50, 0.3), (100, 0.8), (150, 1.5), (200, 3.0)]

    VFE_FLOW_WINDOW = 200
    VFE_FLOW_LARGE_THRESHOLD = 8
    VFE_AGG_FV_TICKS = 1.5
    VFE_LARGE_FV_TICKS = 1.0

    # ---------------------------------------------------------------- Vouchers (real options)
    TV_OPTIONS = ("VEV_5300", "VEV_5400", "VEV_5500")
    TV_PRIOR_MEAN = {"VEV_5300": 46.76, "VEV_5400": 15.95, "VEV_5500": 6.64}
    TV_PRIOR_STD  = {"VEV_5300": 6.23,  "VEV_5400": 3.43,  "VEV_5500": 1.74}
    TV_ENTRY_LONG_Z  = {"VEV_5300": 0.55, "VEV_5400": 0.65, "VEV_5500": 0.50}
    TV_ENTRY_SHORT_Z = {"VEV_5300": 0.70, "VEV_5400": 0.50, "VEV_5500": 0.80}
    TV_EXIT_Z = 0.30
    TV_LONG_CAP  = {"VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300}
    TV_SHORT_CAP = {"VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300}
    TV_QUOTE_EDGE = {"VEV_5300": 0.75, "VEV_5400": 0.55, "VEV_5500": 0.40}
    TV_QUOTE_SIZE = {"VEV_5300": 18,   "VEV_5400": 18,   "VEV_5500": 20}
    TV_INVENTORY_SKEW = {"VEV_5300": 2.4, "VEV_5400": 1.4, "VEV_5500": 0.8}
    TV_EWMA_ALPHA = 0.20
    TV_PRIOR_W = 0.60
    TV_STD_FLOOR_RATIO = 0.6
    TV_MIN_OBS = 20

    # Deep-ITM basis arb
    BASIS_STRIKES = ("VEV_4000",)
    BASIS_QUOTE_HALF = 1
    BASIS_TAKE_THRESHOLD = 2
    BASIS_QUOTE_SIZE = 20
    BASIS_LIMIT = 50

    # Delta hedge
    OPTION_DELTA = {
        "VEV_4000": 1.00, "VEV_5000": 0.95, "VEV_5100": 0.85,
        "VEV_5200": 0.56, "VEV_5300": 0.33, "VEV_5400": 0.13,
        "VEV_5500": 0.06, "VEV_6000": 0.01, "VEV_6500": 0.00,
    }
    HEDGE_DEAD_ZONE = 12
    HEDGE_PASSIVE_MAX = 35
    HEDGE_AGGRESSIVE_MAX = 100
    HEDGE_BUDGET = 180

    LOG_EVERY = 1000

    # ============================================================= run()
    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}
        used: Dict[str, Dict[str, int]] = {}

        # Build books
        books: Dict[str, Dict[str, Any]] = {}
        for product, depth in state.order_depths.items():
            book = self._book(depth)
            if book is not None:
                books[product] = book

        # Flow trackers — restored, observed, persisted.
        # IMPORTANT: observe() uses the PRIOR best_bid/best_ask to classify
        # market_trades as aggressive. So we restore-then-observe-then-persist.
        flow_hp: Optional[FlowTracker] = None
        flow_vfe: Optional[FlowTracker] = None

        if "HYDROGEL_PACK" in books:
            flow_hp = FlowTracker.from_dict(
                memory.get("flow_hp"),
                window=self.HP_FLOW_WINDOW,
                large_threshold=self.HP_FLOW_LARGE_THRESHOLD,
            )
            flow_hp.observe(
                state.timestamp,
                state.market_trades.get("HYDROGEL_PACK", []),
                books["HYDROGEL_PACK"],
            )
            memory["flow_hp"] = flow_hp.to_dict()

        if self.UNDERLYING in books:
            flow_vfe = FlowTracker.from_dict(
                memory.get("flow_vfe"),
                window=self.VFE_FLOW_WINDOW,
                large_threshold=self.VFE_FLOW_LARGE_THRESHOLD,
            )
            flow_vfe.observe(
                state.timestamp,
                state.market_trades.get(self.UNDERLYING, []),
                books[self.UNDERLYING],
            )
            memory["flow_vfe"] = flow_vfe.to_dict()

        # FV state updates (EMAs)
        if "HYDROGEL_PACK" in books:
            self._update_hydro_state(memory, books["HYDROGEL_PACK"])
        if self.UNDERLYING in books:
            self._update_vfe_state(memory, books[self.UNDERLYING])

        # Time-value memory updates for vouchers
        if self.UNDERLYING in books:
            spot_mid = books[self.UNDERLYING]["mid"]
            for product in self.TV_OPTIONS:
                if product in books:
                    self._update_tv_memory(memory, product, books[product]["mid"], spot_mid)

        position = state.position

        # Per-product strategies
        if "HYDROGEL_PACK" in books and flow_hp is not None:
            self._trade_hydrogel(books["HYDROGEL_PACK"], memory, flow_hp,
                                  position, result, used)

        if self.UNDERLYING in books and flow_vfe is not None:
            self._trade_vfe(books[self.UNDERLYING], memory, flow_vfe,
                             position, result, used)

        # Deep-ITM basis arb
        if self.UNDERLYING in books:
            spot_mid = books[self.UNDERLYING]["mid"]
            for K_sym in self.BASIS_STRIKES:
                if K_sym in books:
                    self._trade_basis(K_sym, books[K_sym], spot_mid,
                                       position, result, used)

        # Time-value voucher trading
        if self.UNDERLYING in books:
            spot_mid = books[self.UNDERLYING]["mid"]
            for product in self.TV_OPTIONS:
                if product in books:
                    self._trade_time_value(product, books[product], spot_mid, memory,
                                            position, result, used)

        # Compute net delta and hedge with VFE
        if self.UNDERLYING in books:
            net_delta = self._compute_net_delta(position, result)
            self._delta_hedge(books[self.UNDERLYING], net_delta,
                               position, result, used)

        # Periodic diagnostic log
        ts = state.timestamp
        if ts % self.LOG_EVERY == 0:
            hp_pos = position.get("HYDROGEL_PACK", 0)
            vfe_pos = position.get(self.UNDERLYING, 0)
            net_d = self._compute_net_delta(position, result) if self.UNDERLYING in books else 0
            hp_agg = flow_hp.imbalance() if flow_hp else 0.0
            hp_lg = flow_hp.large_imbalance() if flow_hp else 0.0
            hp_lb = flow_hp.count_large_buys() if flow_hp else 0
            hp_ls = flow_hp.count_large_sells() if flow_hp else 0
            vfe_agg = flow_vfe.imbalance() if flow_vfe else 0.0
            print(
                f"[ts={ts}] hp={hp_pos:+4d} vfe={vfe_pos:+4d} netΔ={net_d:+5.0f} "
                f"hp_flow={hp_agg:+.2f}/lg={hp_lg:+.2f}({hp_lb}b/{hp_ls}s) "
                f"vfe_flow={vfe_agg:+.2f} "
                f"5300={position.get('VEV_5300',0):+4d} "
                f"5400={position.get('VEV_5400',0):+4d} "
                f"5500={position.get('VEV_5500',0):+4d}"
            )

        trader_data = json.dumps(memory, separators=(",", ":"))
        return result, 0, trader_data

    # ============================================================= memory & books
    def _load_memory(self, trader_data: str) -> Dict[str, Any]:
        if trader_data:
            try:
                m = json.loads(trader_data)
                if isinstance(m, dict):
                    return m
            except Exception:
                pass
        return {}

    def _book(self, depth: OrderDepth) -> Optional[Dict[str, Any]]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid = max(depth.buy_orders)
        ask = min(depth.sell_orders)
        bid_vol = depth.buy_orders[bid]
        ask_vol = -depth.sell_orders[ask]
        if bid_vol <= 0 or ask_vol <= 0 or bid >= ask:
            return None
        mid = 0.5 * (bid + ask)
        micro = (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol)
        return {
            "best_bid": bid, "best_ask": ask, "bid_vol": bid_vol, "ask_vol": ask_vol,
            "mid": mid, "micro": micro, "spread": ask - bid,
        }

    # ============================================================= state updates
    def _update_hydro_state(self, memory: Dict, book: Dict) -> None:
        s = memory.setdefault("hp", {"fv_ema": self.HP_FV_INIT, "last_mid": book["mid"]})
        s["fv_ema"] = (1 - self.HP_EMA_ALPHA) * s["fv_ema"] + self.HP_EMA_ALPHA * book["mid"]
        s["last_mid"] = book["mid"]

    def _update_vfe_state(self, memory: Dict, book: Dict) -> None:
        s = memory.setdefault("vfe", {"fv_ema": self.VFE_FV_INIT, "last_mid": book["mid"]})
        s["fv_ema"] = (1 - self.VFE_EMA_ALPHA) * s["fv_ema"] + self.VFE_EMA_ALPHA * book["micro"]
        s["last_mid"] = book["mid"]

    def _update_tv_memory(self, memory: Dict, product: str, opt_mid: float, spot_mid: float) -> None:
        intrinsic = max(spot_mid - self.STRIKES[product], 0.0)
        tv = opt_mid - intrinsic
        s = memory.setdefault("tv", {}).setdefault(product, {
            "mean": self.TV_PRIOR_MEAN[product],
            "var": self.TV_PRIOR_STD[product] ** 2,
            "count": 0,
        })
        a = self.TV_EWMA_ALPHA
        s["mean"] = (1 - a) * s["mean"] + a * tv
        s["var"] = (1 - a) * s["var"] + a * (tv - s["mean"]) ** 2
        s["count"] = s.get("count", 0) + 1

    def _adaptive_tv(self, memory: Dict, product: str) -> Tuple[float, float]:
        prior_m = self.TV_PRIOR_MEAN[product]
        prior_s = self.TV_PRIOR_STD[product]
        s = memory.get("tv", {}).get(product)
        if s is None or s.get("count", 0) < self.TV_MIN_OBS:
            return prior_m, prior_s
        live_m = s["mean"]
        live_s = max(math.sqrt(max(s["var"], 1e-9)), self.TV_STD_FLOOR_RATIO * prior_s)
        w = self.TV_PRIOR_W
        return w * prior_m + (1 - w) * live_m, w * prior_s + (1 - w) * live_s

    # ============================================================= helpers
    def _skew_factor(self, position: int, tiers: List[Tuple[int, float]]) -> float:
        ap = abs(position)
        for thresh, factor in tiers:
            if ap <= thresh:
                return factor * (1.0 if position >= 0 else -1.0)
        return tiers[-1][1] * (1.0 if position >= 0 else -1.0)

    def _add_order(self, product: str, price: int, qty: int,
                   position: Dict, result: Dict, used: Dict,
                   long_cap: Optional[int] = None,
                   short_cap: Optional[int] = None) -> int:
        if qty == 0:
            return 0
        limit = self.LIMITS[product]
        long_cap = long_cap if long_cap is not None else limit
        short_cap = short_cap if short_cap is not None else limit
        pos = position.get(product, 0)
        u = used.setdefault(product, {"buy": 0, "sell": 0})

        if qty > 0:
            avail = max(0, long_cap - pos - u["buy"])
            qty = min(qty, avail)
            if qty <= 0:
                return 0
            u["buy"] += qty
        else:
            avail = max(0, short_cap + pos - u["sell"])
            qty = -min(-qty, avail)
            if qty >= 0:
                return 0
            u["sell"] += -qty

        if qty != 0:
            result.setdefault(product, []).append(Order(product, int(round(price)), int(qty)))
        return qty

    # ============================================================= HYDROGEL_PACK
    def _trade_hydrogel(self, book: Dict, memory: Dict, flow: FlowTracker,
                         position: Dict, result: Dict, used: Dict) -> None:
        product = "HYDROGEL_PACK"
        s = memory.get("hp", {})
        fv = float(s.get("fv_ema", self.HP_FV_INIT))
        pos = position.get(product, 0)

        # Flow-driven FV bias. Two components:
        #   - imbalance(): unweighted aggression direction. Captures retail noise too.
        #   - large_imbalance(): only big-size flows. Captures institutional pressure.
        # When they agree, signal is strong; when they disagree, they cancel.
        agg = flow.imbalance()
        agg_lg = flow.large_imbalance()
        flow_bias = agg * self.HP_AGG_FV_TICKS + agg_lg * self.HP_LARGE_FV_TICKS
        # Cap total bias so a saturated signal can't override the EMA outright
        max_bias = self.HP_AGG_FV_TICKS + self.HP_LARGE_FV_TICKS
        flow_bias = max(-max_bias, min(max_bias, flow_bias))
        fv_with_flow = fv + flow_bias

        # Inventory skew
        skew = self._skew_factor(pos, self.HP_SKEW_TIERS)
        fv_eff = fv_with_flow - skew

        # TAKE
        if book["best_ask"] < fv_eff - self.HP_TAKE_THRESHOLD:
            qty = min(self.HP_QUOTE_SIZE, book["ask_vol"])
            self._add_order(product, book["best_ask"], qty, position, result, used)
        if book["best_bid"] > fv_eff + self.HP_TAKE_THRESHOLD:
            qty = min(self.HP_QUOTE_SIZE, book["bid_vol"])
            self._add_order(product, book["best_bid"], -qty, position, result, used)

        # MAKE
        bid_px = int(round(fv_eff - self.HP_QUOTE_HALF))
        ask_px = int(round(fv_eff + self.HP_QUOTE_HALF))
        # Hard guards (against unbiased FV — guard the absolute level, not flow-shifted)
        if bid_px > fv + self.HP_HARD_GUARD:
            bid_px = int(round(fv + self.HP_HARD_GUARD))
        if ask_px < fv - self.HP_HARD_GUARD:
            ask_px = int(round(fv - self.HP_HARD_GUARD))
        if bid_px < book["best_ask"]:
            self._add_order(product, bid_px, self.HP_QUOTE_SIZE, position, result, used)
        if ask_px > book["best_bid"]:
            self._add_order(product, ask_px, -self.HP_QUOTE_SIZE, position, result, used)

    # ============================================================= VFE
    def _trade_vfe(self, book: Dict, memory: Dict, flow: FlowTracker,
                    position: Dict, result: Dict, used: Dict) -> None:
        product = self.UNDERLYING
        s = memory.get("vfe", {})
        fv = float(s.get("fv_ema", self.VFE_FV_INIT))
        pos = position.get(product, 0)

        # Flow bias on VFE
        agg = flow.imbalance()
        agg_lg = flow.large_imbalance()
        flow_bias = agg * self.VFE_AGG_FV_TICKS + agg_lg * self.VFE_LARGE_FV_TICKS
        max_bias = self.VFE_AGG_FV_TICKS + self.VFE_LARGE_FV_TICKS
        flow_bias = max(-max_bias, min(max_bias, flow_bias))
        fv_with_flow = fv + flow_bias

        # Inventory skew
        skew = self._skew_factor(pos, self.VFE_SKEW_TIERS)
        fv_eff = fv_with_flow - skew

        # TAKE
        if book["best_ask"] < fv_eff - self.VFE_TAKE_THRESHOLD:
            qty = min(self.VFE_QUOTE_SIZE, book["ask_vol"])
            self._add_order(product, book["best_ask"], qty, position, result, used)
        if book["best_bid"] > fv_eff + self.VFE_TAKE_THRESHOLD:
            qty = min(self.VFE_QUOTE_SIZE, book["bid_vol"])
            self._add_order(product, book["best_bid"], -qty, position, result, used)

        # MAKE
        bid_px = int(round(fv_eff - self.VFE_QUOTE_HALF))
        ask_px = int(round(fv_eff + self.VFE_QUOTE_HALF))
        bid_px = min(bid_px, book["best_ask"] - 1)
        ask_px = max(ask_px, book["best_bid"] + 1)
        if bid_px > fv + self.VFE_HARD_GUARD:
            bid_px = int(round(fv + self.VFE_HARD_GUARD))
        if ask_px < fv - self.VFE_HARD_GUARD:
            ask_px = int(round(fv - self.VFE_HARD_GUARD))
        if bid_px < book["best_ask"]:
            self._add_order(product, bid_px, self.VFE_QUOTE_SIZE, position, result, used)
        if ask_px > book["best_bid"]:
            self._add_order(product, ask_px, -self.VFE_QUOTE_SIZE, position, result, used)

    # ============================================================= Time-value vouchers
    def _trade_time_value(self, product: str, book: Dict, spot_mid: float,
                           memory: Dict, position: Dict, result: Dict, used: Dict) -> None:
        long_cap = self.TV_LONG_CAP[product]
        short_cap = self.TV_SHORT_CAP[product]
        pos = position.get(product, 0)
        intrinsic = max(spot_mid - self.STRIKES[product], 0.0)
        mean, std = self._adaptive_tv(memory, product)
        ent_l = self.TV_ENTRY_LONG_Z[product]
        ent_s = self.TV_ENTRY_SHORT_Z[product]
        ask_tv = book["best_ask"] - intrinsic
        bid_tv = book["best_bid"] - intrinsic

        long_entry  = mean - ent_l * std
        short_entry = mean + ent_s * std
        long_exit   = mean + self.TV_EXIT_Z * std
        short_exit  = mean - self.TV_EXIT_Z * std

        # Exit longs to bid past mean
        if pos > 0 and bid_tv >= long_exit:
            qty = min(pos, book["bid_vol"])
            if qty > 0:
                self._add_order(product, book["best_bid"], -qty, position, result, used,
                                long_cap=max(long_cap, short_cap), short_cap=max(long_cap, short_cap))
        # Exit shorts to ask past mean
        if pos < 0 and ask_tv <= short_exit:
            qty = min(-pos, book["ask_vol"])
            if qty > 0:
                self._add_order(product, book["best_ask"], qty, position, result, used,
                                long_cap=max(long_cap, short_cap), short_cap=max(long_cap, short_cap))

        s = memory.get("tv", {}).get(product, {})
        if s.get("count", 0) < self.TV_MIN_OBS:
            return

        # Long underpriced calls
        if long_cap > 0 and ask_tv <= long_entry and pos < long_cap:
            qty = min(book["ask_vol"], long_cap - pos)
            if qty > 0:
                self._add_order(product, book["best_ask"], qty, position, result, used,
                                long_cap=long_cap, short_cap=short_cap)
        # Short overpriced calls
        if short_cap > 0 and bid_tv >= short_entry and pos > -short_cap:
            qty = min(book["bid_vol"], short_cap + pos)
            if qty > 0:
                self._add_order(product, book["best_bid"], -qty, position, result, used,
                                long_cap=long_cap, short_cap=short_cap)

        # Passive quotes around adaptive fair, inventory-skewed
        fair_opt = intrinsic + mean
        sk = self.TV_INVENTORY_SKEW[product]
        cap = max(1, max(long_cap, short_cap))
        fair_adj = fair_opt - sk * pos / cap
        edge = self.TV_QUOTE_EDGE[product]
        size = self.TV_QUOTE_SIZE[product]
        bid_px = int(round(fair_adj - edge))
        ask_px = int(round(fair_adj + edge))
        if bid_px < book["best_ask"] and pos < long_cap:
            self._add_order(product, bid_px, size, position, result, used,
                            long_cap=long_cap, short_cap=short_cap)
        if ask_px > book["best_bid"] and pos > -short_cap:
            self._add_order(product, ask_px, -size, position, result, used,
                            long_cap=long_cap, short_cap=short_cap)

    # ============================================================= Deep-ITM basis arb
    def _trade_basis(self, product: str, book: Dict, spot_mid: float,
                      position: Dict, result: Dict, used: Dict) -> None:
        K = self.STRIKES[product]
        fair = spot_mid - K
        pos = position.get(product, 0)
        cap = self.BASIS_LIMIT

        if book["best_ask"] < fair - self.BASIS_TAKE_THRESHOLD:
            qty = min(self.BASIS_QUOTE_SIZE, book["ask_vol"])
            self._add_order(product, book["best_ask"], qty, position, result, used,
                            long_cap=cap, short_cap=cap)
        if book["best_bid"] > fair + self.BASIS_TAKE_THRESHOLD:
            qty = min(self.BASIS_QUOTE_SIZE, book["bid_vol"])
            self._add_order(product, book["best_bid"], -qty, position, result, used,
                            long_cap=cap, short_cap=cap)

        bid_px = int(round(fair - self.BASIS_QUOTE_HALF))
        ask_px = int(round(fair + self.BASIS_QUOTE_HALF))
        if bid_px < book["best_ask"] and pos < cap:
            self._add_order(product, bid_px, self.BASIS_QUOTE_SIZE, position, result, used,
                            long_cap=cap, short_cap=cap)
        if ask_px > book["best_bid"] and pos > -cap:
            self._add_order(product, ask_px, -self.BASIS_QUOTE_SIZE, position, result, used,
                            long_cap=cap, short_cap=cap)

    # ============================================================= Delta hedge
    def _compute_net_delta(self, position: Dict, result: Dict) -> float:
        nd = position.get(self.UNDERLYING, 0)
        for o in result.get(self.UNDERLYING, []):
            nd += o.quantity
        for sym, d in self.OPTION_DELTA.items():
            cur = position.get(sym, 0)
            for o in result.get(sym, []):
                cur += o.quantity
            nd += d * cur
        return nd

    def _delta_hedge(self, book: Dict, net_delta: float,
                      position: Dict, result: Dict, used: Dict) -> None:
        ad = abs(net_delta)
        if ad <= self.HEDGE_DEAD_ZONE:
            return
        sign = -1 if net_delta > 0 else 1

        if ad <= self.HEDGE_PASSIVE_MAX:
            qty = sign * int(ad - self.HEDGE_DEAD_ZONE)
            px = book["best_bid"] if sign > 0 else book["best_ask"]
            px = px + sign
            self._add_order(self.UNDERLYING, px, qty, position, result, used)
        else:
            qty = sign * min(int(ad - self.HEDGE_DEAD_ZONE), self.HEDGE_AGGRESSIVE_MAX)
            px = book["best_ask"] if sign > 0 else book["best_bid"]
            available_vol = book["ask_vol"] if sign > 0 else book["bid_vol"]
            qty = sign * min(abs(qty), available_vol)
            self._add_order(self.UNDERLYING, px, qty, position, result, used)


# ============================================================= self-test
if __name__ == "__main__":
    from datamodel import OrderDepth, TradingState

    # ---- Test 1: FlowTracker arithmetic
    book = {"best_bid": 100, "best_ask": 102, "mid": 101.0,
            "bid_vol": 10, "ask_vol": 10, "micro": 101.0, "spread": 2}
    f = FlowTracker(window=10, large_threshold=5)
    f.prev_bid = 100
    f.prev_ask = 102
    # Two aggressive sells (price <= prior_bid)
    f.observe(0, [Trade("X", 100, 8, "", "", 0), Trade("X", 99, 3, "", "", 0)], book)
    assert f.imbalance() < 0, f"expected negative imbalance, got {f.imbalance()}"
    # Large-size imbalance: only the qty=8 counts (>= threshold 5)
    assert f.large_imbalance() == -1.0, f"expected -1.0, got {f.large_imbalance()}"
    assert f.count_large_sells() == 1
    assert f.count_large_buys() == 0

    # Round-trip through dict
    d = f.to_dict()
    f2 = FlowTracker.from_dict(d, window=10, large_threshold=5)
    assert f2.imbalance() == f.imbalance()
    assert f2.large_imbalance() == f.large_imbalance()
    print("FLOW TRACKER OK — imbalance, large_imbalance, round-trip all work")

    # ---- Test 2: Full Trader run()
    def make_book(bid, ask, bv=20, av=20):
        d = OrderDepth()
        d.buy_orders = {bid: bv}
        d.sell_orders = {ask: -av}
        return d

    state = TradingState(
        traderData="",
        timestamp=0,
        listings={},
        order_depths={
            "HYDROGEL_PACK":       make_book(9985, 10001, 12, 12),
            "VELVETFRUIT_EXTRACT": make_book(5263, 5268, 38, 38),
            "VEV_4000":            make_book(1255, 1276, 11, 11),
            "VEV_5300":            make_book(51, 53),
            "VEV_5400":            make_book(16, 18),
            "VEV_5500":            make_book(6, 7),
        },
        own_trades={}, market_trades={},
        position={"HYDROGEL_PACK": 0, "VELVETFRUIT_EXTRACT": 0,
                  "VEV_4000": 0, "VEV_5300": 0, "VEV_5400": 0, "VEV_5500": 0},
        observations=None,
    )
    t = Trader()
    for tick in range(5):
        state.timestamp = tick * 100
        # Inject some flow on tick 1+ to verify FlowTracker observes it
        if tick == 1:
            state.market_trades = {
                "HYDROGEL_PACK": [Trade("HYDROGEL_PACK", 9985, 8, "", "", 100)],
                "VELVETFRUIT_EXTRACT": [Trade("VELVETFRUIT_EXTRACT", 5263, 5, "", "", 100)],
            }
        else:
            state.market_trades = {}
        result, conv, td = t.run(state)
        assert isinstance(result, dict)
        assert isinstance(conv, int)
        assert isinstance(td, str)
        decoded = json.loads(td)
        assert "flow_hp" in decoded, "flow_hp not persisted"
        assert "flow_vfe" in decoded, "flow_vfe not persisted"
        state.traderData = td
    final_mem = json.loads(state.traderData)
    hp_trades = final_mem["flow_hp"]["trades"]
    assert len(hp_trades) >= 1, f"expected >=1 HP trade observed, got {len(hp_trades)}"
    print("TRADER OK — 5-tick run completes, flow trackers persist across ticks")
    print(f"  HP flow observed {len(hp_trades)} trade(s)")
    print(f"  VFE flow observed {len(final_mem['flow_vfe']['trades'])} trade(s)")