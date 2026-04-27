from datamodel import Order, OrderDepth, Trade, TradingState
from typing import Any, Dict, List, Optional, Tuple
import json
import math


class Trader:
    LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    RISK_CAPS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 0,
        "VEV_6500": 0,
    }

    LONG_CAPS = {
        "HYDROGEL_PACK": 200,
    }

    SHORT_CAPS = {
        "HYDROGEL_PACK": 200,
    }

    STRIKES = {
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

    UNDERLYING = "VELVETFRUIT_EXTRACT"
    HYDRO_ANCHOR = 9994.0
    VELVET_ANCHOR = 5250.0

    HYDRO_SWING_ENABLED = True
    HYDRO_SWING_CAP = 200
    HYDRO_SWING_SIZE = 40
    # --- Adaptive: offsets from slow EMA instead of hardcoded levels ---
    HYDRO_LONG_ENTRY_OFFSET = -15
    HYDRO_LONG_EXIT_OFFSET = +5
    HYDRO_SHORT_ENTRY_OFFSET = +15
    HYDRO_SHORT_EXIT_OFFSET = -5
    HYDRO_SCALE_OUT_POS = 180
    HYDRO_SCALE_OUT_SIZE = 4
    HYDRO_SCALE_OUT_EDGE = 20
    HYDRO_PASSIVE_SIZE = 6
    HYDRO_PASSIVE_OFFSET_LOW = -4
    HYDRO_PASSIVE_OFFSET_HIGH = +4

    # --- Adaptive: offsets from product slow EMA ---
    # (le_offset, lx_offset, se_offset, sx_offset, cap, levels)
    PRICE_SWING_CONFIG = {
        "VELVETFRUIT_EXTRACT": (-12, +6, +12, -14, 200, 2),
        "VEV_4000":            (-9, +1, +7, -9, 300, 2),
        "VEV_4500":            (-10, +1, +6, -14, 300, 2),
        "VEV_5000":            (-14, +2, +10, -17, 300, 2),
        "VEV_5100":            (-12, +7, +10, -12, 300, 2),
        "VEV_5200":            (-5, +8, +8, -5, 300, 2),
    }
    PRICE_SWING_USE_EMA = True   # flag for adaptive vs absolute

    TIME_VALUE_OPTIONS: Tuple[str, ...] = ("VEV_5300", "VEV_5400", "VEV_5500")
    TIME_VALUE_MEAN = {
        "VEV_5200": 45.45,
        "VEV_5300": 46.76,
        "VEV_5400": 15.95,
        "VEV_5500": 6.64,
    }
    TIME_VALUE_STD = {
        "VEV_5200": 7.85,
        "VEV_5300": 6.23,
        "VEV_5400": 3.43,
        "VEV_5500": 1.74,
    }
    TIME_VALUE_ENTRY_Z = {
        "VEV_5200": 0.65,
        "VEV_5300": 0.55,
        "VEV_5400": 0.65,
        "VEV_5500": 0.50,
    }
    TIME_VALUE_SHORT_ENTRY_Z = {
        "VEV_5200": 0.65,
        "VEV_5300": 0.70,
        "VEV_5400": 0.50,
        "VEV_5500": 0.80,
    }
    TIME_VALUE_LONG_CAP = {
        "VEV_5200": 200,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }
    TIME_VALUE_SHORT_CAP = {
        "VEV_5200": 200,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
    }
    TIME_VALUE_QUOTE_EDGE = {
        "VEV_5200": 0.85,
        "VEV_5300": 0.75,
        "VEV_5400": 0.55,
        "VEV_5500": 0.40,
    }
    TIME_VALUE_QUOTE_SIZE = {
        "VEV_5200": 16,
        "VEV_5300": 18,
        "VEV_5400": 18,
        "VEV_5500": 20,
    }
    TIME_VALUE_INVENTORY_SKEW = {
        "VEV_5200": 2.6,
        "VEV_5300": 0.8,
        "VEV_5400": 0.5,
        "VEV_5500": 0.25,
    }
    TIME_VALUE_STD_FLOOR_RATIO = 0.6
    TIME_VALUE_MIN_OBS = 0

    OPTION_DELTAS = {
        "VEV_4000": 1.00,
        "VEV_5200": 0.56,
        "VEV_5300": 0.33,
        "VEV_5400": 0.13,
        "VEV_5500": 0.06,
    }
    VFE_OPTION_HEDGE_RATIO = 0.0
    VFE_OPTION_HEDGE_TRIGGER = 18
    VFE_OPTION_HEDGE_CLIP = 80
    VFE_OPTION_HEDGE_SLICE = 12

    DEEP_ITM_TAKE_SIZE = {
        "VEV_4000": 8,
    }
    DEEP_ITM_QUOTE_SIZE = {
        "VEV_4000": 6,
    }

    TV_EMA_ALPHA = 1.00
    TV_PRIOR_WEIGHT = 0.60
    TV_CYCLE_ALPHA = 0.20
    TV_MATURE_COUNT = 35
    TV_COMPRESSED_GAP_SIGMAS = 0.75
    TV_SURFACE_ALPHA = 0.18
    TV_SURFACE_MIN_COUNT = 25
    TV_SURFACE_MIN_HOLD = 12
    TV_SURFACE_STD_MULT = 0.65
    TV_SURFACE_EDGE_FLOOR = 0.8
    TV_SURFACE_CLIP = 24
    TV_SURFACE_CORE = {
        "VEV_5300": 180,
        "VEV_5400": 240,
    }
    OTM_EVENT_SCALE = 18.0
    OTM_EVENT_CLIP = 2.5

    # Conservative round-4 counterparty overlay.
    # Positive trader weights mean "their direction tends to continue".
    # Negative trader weights mean "fade their direction".
    MARK_WEIGHTS = {
        "HYDROGEL_PACK": {
            "Mark 14": 0.9,
            "Mark 38": -0.9,
        },
        "VELVETFRUIT_EXTRACT": {
            "Mark 01": 0.9,
            "Mark 14": 0.7,
            "Mark 67": 0.4,
            "Mark 55": -0.9,
            "Mark 49": -0.4,
        },
        "VEV_4000": {
            "Mark 14": 0.9,
            "Mark 38": -0.9,
        },
        "VEV_5200": {
            "Mark 14": 0.5,
            "Mark 01": 0.4,
            "Mark 22": -0.6,
        },
        "VEV_5300": {
            "Mark 01": -0.15,
            "Mark 14": -0.25,
            "Mark 22": 0.30,
        },
        "VEV_5400": {
            "Mark 01": -0.04,
            "Mark 14": -0.08,
            "Mark 22": 0.10,
        },
        "VEV_5500": {
            "Mark 01": -0.08,
            "Mark 14": -0.12,
            "Mark 22": 0.18,
        },
    }

    MARK_SCALE = {
        "HYDROGEL_PACK": 40.0,
        "VELVETFRUIT_EXTRACT": 35.0,
        "VEV_4000": 22.0,
        "VEV_5200": 15.0,
        "VEV_5300": 12.0,
        "VEV_5400": 12.0,
        "VEV_5500": 12.0,
    }

    MARK_CLIP = {
        "HYDROGEL_PACK": 2.5,
        "VELVETFRUIT_EXTRACT": 2.5,
        "VEV_4000": 2.5,
        "VEV_5200": 2.0,
        "VEV_5300": 1.4,
        "VEV_5400": 0.6,
        "VEV_5500": 0.8,
    }

    MARK_DECAY = 0.88

    PRICE_SATELLITE_CONFIG = {}
    TV_SATELLITE_CONFIG = {
        "VEV_5300": (180, 12, 1.8, 0.55, 0.35, 24),
        "VEV_5400": (240, 20, 0.8, 0.55, 0.40, 12),
    }

    # Hydro and VFE often move like opposite sides of the same spread. Keep the
    # cross signal small: day 1 is not cleanly inverse, while days 2/3 are.
    MIRROR_HYDRO_SCALE = 35.0
    MIRROR_VFE_SCALE = 30.0
    MIRROR_Z_CLIP = 2.5
    MIRROR_HYDRO_SHIFT_PER_VFE_Z = 3.0
    MIRROR_PRICE_SHIFT_PER_HYDRO_Z = 2.0

    def bid(self):
        return 15

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}
        used: Dict[str, Dict[str, int]] = {}

        books: Dict[str, Dict[str, Any]] = {}
        for product, depth in state.order_depths.items():
            book = self._book(depth)
            if book is not None:
                books[product] = book
                self._update_memory(memory, product, book)
        self._update_mirror_memory(memory, books)

        self._decay_mark_memory(memory)
        self._ingest_mark_trades(memory, state.market_trades)
        self._update_hold_memory(memory, state.position)

        fairs: Dict[str, float] = {}
        deltas: Dict[str, float] = {}

        if "HYDROGEL_PACK" in books:
            fairs["HYDROGEL_PACK"] = self._spot_fair(
                "HYDROGEL_PACK", books["HYDROGEL_PACK"], memory
            )

        if self.UNDERLYING in books:
            fairs[self.UNDERLYING] = self._spot_fair(
                self.UNDERLYING, books[self.UNDERLYING], memory
            )
            deltas[self.UNDERLYING] = 1.0

        if self.UNDERLYING in fairs:
            spot = fairs[self.UNDERLYING]
            if "VEV_4000" in books and self.RISK_CAPS["VEV_4000"] > 0:
                fairs["VEV_4000"] = max(0.0, spot - self.STRIKES["VEV_4000"])
                deltas["VEV_4000"] = 1.0

            spot_mid = books[self.UNDERLYING]["mid"]
            for product in self.TIME_VALUE_OPTIONS:
                if product in books:
                    self._update_time_value_memory(
                        memory, product, books[product]["mid"], spot_mid
                    )
            if "VEV_5300" in books and "VEV_5400" in books:
                self._update_tv_surface_memory(
                    memory, spot_mid, books["VEV_5300"]["mid"], books["VEV_5400"]["mid"]
                )

        if "HYDROGEL_PACK" in fairs:
            self._trade_hydro(
                "HYDROGEL_PACK", books["HYDROGEL_PACK"], fairs["HYDROGEL_PACK"],
                memory, state.position, result, used
            )

        if self.UNDERLYING in fairs and self.UNDERLYING not in self.PRICE_SWING_CONFIG:
            self._trade_spot(
                self.UNDERLYING, books[self.UNDERLYING], fairs[self.UNDERLYING],
                state.position, result, used
            )

        for product in self.PRICE_SWING_CONFIG:
            if product in books:
                self._trade_price_swing(
                    product, books[product], memory, state.position, result, used
                )

        for product in ("VEV_4000",):
            if product in fairs and product not in self.PRICE_SWING_CONFIG:
                self._trade_deep_itm_parity(
                    product, books[product], fairs[product], state.position, result, used
                )

        if self.UNDERLYING in books:
            spot_mid = books[self.UNDERLYING]["mid"]
            for product in self.TIME_VALUE_OPTIONS:
                if product in books:
                    self._trade_time_value_call(
                        product, books[product], spot_mid, memory,
                        state.position, result, used,
                    )
            self._trade_tv_surface_rotation(
                books, spot_mid, memory, state.position, result, used
            )
            self._hedge_option_delta(
                books[self.UNDERLYING], state.position, result, used
            )

        trader_data = json.dumps(memory, separators=(",", ":"))
        return result, 0, trader_data

    def _load_memory(self, trader_data: str) -> Dict[str, Any]:
        if trader_data:
            try:
                decoded = json.loads(trader_data)
                if isinstance(decoded, dict) and isinstance(decoded.get("products"), dict):
                    return decoded
            except Exception:
                pass
        return {"products": {}}

    def _book(self, depth: OrderDepth) -> Optional[Dict[str, Any]]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid_levels = sorted(
            [(price, volume) for price, volume in depth.buy_orders.items() if volume > 0],
            reverse=True,
        )
        ask_levels = sorted(
            [(price, -volume) for price, volume in depth.sell_orders.items() if volume < 0]
        )
        if not bid_levels or not ask_levels:
            return None
        bid, bid_vol = bid_levels[0]
        ask, ask_vol = ask_levels[0]
        if bid_vol <= 0 or ask_vol <= 0 or bid >= ask:
            return None
        mid = (bid + ask) / 2.0
        micro = (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol)
        return {
            "best_bid": bid,
            "best_ask": ask,
            "bid_vol": bid_vol,
            "ask_vol": ask_vol,
            "mid": mid,
            "spread": ask - bid,
            "micro": micro,
            "bid_levels": bid_levels,
            "ask_levels": ask_levels,
        }

    def _update_memory(self, memory: Dict[str, Any], product: str, book: Dict[str, Any]) -> None:
        item = memory.setdefault("products", {}).setdefault(product, {})
        mid = book["mid"]
        if "ema" not in item:
            item["ema"] = mid
            item["slow"] = mid
            item["prev"] = mid
            item["last"] = mid
            item["vol"] = max(1.0, book["spread"] / 2.0)
            item["mark_bias"] = 0.0
            return
        last = float(item.get("last", mid))
        item["prev"] = last
        item["ema"] = 0.10 * mid + 0.90 * float(item.get("ema", mid))
        item["slow"] = 0.01 * mid + 0.99 * float(item.get("slow", mid))
        item["vol"] = 0.08 * abs(mid - last) + 0.92 * float(item.get("vol", 1.0))
        item["last"] = mid

    def _decay_mark_memory(self, memory: Dict[str, Any]) -> None:
        for item in memory.setdefault("products", {}).values():
            item["mark_bias"] = float(item.get("mark_bias", 0.0)) * self.MARK_DECAY
            item["otm_sell_bias"] = float(item.get("otm_sell_bias", 0.0)) * self.MARK_DECAY

    def _ingest_mark_trades(
        self, memory: Dict[str, Any], market_trades: Dict[str, List[Trade]]
    ) -> None:
        for product, trades in market_trades.items():
            weights = self.MARK_WEIGHTS.get(product)
            if weights:
                item = memory.setdefault("products", {}).setdefault(product, {})
                bias = float(item.get("mark_bias", 0.0))
                for trade in trades:
                    buyer_w = weights.get(trade.buyer, 0.0)
                    seller_w = weights.get(trade.seller, 0.0)
                    signed = buyer_w - seller_w
                    if signed == 0.0:
                        continue
                    qty = max(1, int(trade.quantity))
                    bias += signed * qty / self.MARK_SCALE[product]
                clip = self.MARK_CLIP[product]
                item["mark_bias"] = max(-clip, min(clip, bias))

            for trade in trades:
                self._ingest_otm_flow(memory, product, trade)

    def _mark_bias(self, memory: Dict[str, Any], product: str) -> float:
        item = memory.get("products", {}).get(product, {})
        return float(item.get("mark_bias", 0.0))

    def _update_mirror_memory(
        self, memory: Dict[str, Any], books: Dict[str, Dict[str, Any]]
    ) -> None:
        hydro = books.get("HYDROGEL_PACK")
        vfe = books.get(self.UNDERLYING)
        if hydro is None or vfe is None:
            return

        products = memory.setdefault("products", {})
        hydro_item = products.get("HYDROGEL_PACK", {})
        vfe_item = products.get(self.UNDERLYING, {})
        hydro_slow = float(hydro_item.get("slow", hydro["mid"]))
        vfe_slow = float(vfe_item.get("slow", vfe["mid"]))

        hydro_z = (hydro["mid"] - hydro_slow) / self.MIRROR_HYDRO_SCALE
        vfe_z = (vfe["mid"] - vfe_slow) / self.MIRROR_VFE_SCALE
        hydro_z = max(-self.MIRROR_Z_CLIP, min(self.MIRROR_Z_CLIP, hydro_z))
        vfe_z = max(-self.MIRROR_Z_CLIP, min(self.MIRROR_Z_CLIP, vfe_z))

        mirror = memory.setdefault("mirror", {})
        mirror["hydro_z"] = hydro_z
        mirror["vfe_z"] = vfe_z

    def _mirror_hydro_shift(self, memory: Dict[str, Any]) -> float:
        vfe_z = float(memory.get("mirror", {}).get("vfe_z", 0.0))
        return -self.MIRROR_HYDRO_SHIFT_PER_VFE_Z * vfe_z

    def _mirror_price_shift(self, memory: Dict[str, Any], product: str) -> float:
        hydro_z = float(memory.get("mirror", {}).get("hydro_z", 0.0))
        return -self.MIRROR_PRICE_SHIFT_PER_HYDRO_Z * hydro_z * self.OPTION_DELTAS.get(product, 0.0)

    def _bump_otm_sell_bias(self, memory: Dict[str, Any], delta: float) -> None:
        if delta == 0.0:
            return
        for product in ("VEV_5300", "VEV_5400"):
            item = memory.setdefault("products", {}).setdefault(product, {})
            nxt = float(item.get("otm_sell_bias", 0.0)) + delta
            item["otm_sell_bias"] = max(-self.OTM_EVENT_CLIP, min(self.OTM_EVENT_CLIP, nxt))

    def _ingest_otm_flow(self, memory: Dict[str, Any], product: str, trade: Trade) -> None:
        qty = max(1, int(trade.quantity))
        delta = 0.0
        if product in {"VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}:
            if trade.seller == "Mark 22" and trade.buyer in {"Mark 01", "Mark 14"}:
                mult = 1.15 if trade.buyer == "Mark 14" else 0.90
                delta += mult * qty / self.OTM_EVENT_SCALE
            elif trade.buyer == "Mark 22" and trade.seller in {"Mark 01", "Mark 14"}:
                mult = 1.15 if trade.seller == "Mark 14" else 0.90
                delta -= mult * qty / self.OTM_EVENT_SCALE
        elif product == self.UNDERLYING:
            if trade.buyer == "Mark 55" and trade.seller in {"Mark 14", "Mark 01"}:
                delta -= 0.35 * qty / self.OTM_EVENT_SCALE
            elif trade.buyer == "Mark 67" and trade.seller in {"Mark 22", "Mark 49"}:
                delta -= 0.25 * qty / self.OTM_EVENT_SCALE
            elif trade.seller == "Mark 55" and trade.buyer in {"Mark 14", "Mark 01"}:
                delta += 0.20 * qty / self.OTM_EVENT_SCALE
        self._bump_otm_sell_bias(memory, delta)

    def _update_hold_memory(
        self, memory: Dict[str, Any], position: Dict[str, int]
    ) -> None:
        cfg: Dict[str, Tuple[int, ...]] = dict(self.TV_SATELLITE_CONFIG)
        for product, vals in cfg.items():
            item = memory.setdefault("products", {}).setdefault(product, {})
            core_pos = int(vals[0])
            pos = position.get(product, 0)
            hold_ticks = int(item.get("long_hold_ticks", 0))
            if pos >= core_pos:
                hold_ticks += 1
            else:
                hold_ticks = 0
            item["long_hold_ticks"] = hold_ticks

    def _update_time_value_memory(
        self, memory: Dict[str, Any], product: str, option_mid: float, spot_mid: float
    ) -> None:
        strike = self.STRIKES[product]
        intrinsic = max(spot_mid - strike, 0.0)
        tv = option_mid - intrinsic
        item = memory.setdefault("products", {}).setdefault(product, {})
        prior_mean = self.TIME_VALUE_MEAN[product]
        prior_var = self.TIME_VALUE_STD[product] ** 2
        tv_mean = float(item.get("tv_mean", prior_mean))
        tv_var = float(item.get("tv_var", prior_var))
        a = self.TV_EMA_ALPHA
        new_mean = (1.0 - a) * tv_mean + a * tv
        diff = tv - tv_mean
        new_var = (1.0 - a) * tv_var + a * diff * diff
        item["tv_mean"] = new_mean
        item["tv_var"] = max(new_var, 1e-6)
        item["tv_count"] = int(item.get("tv_count", 0)) + 1

        cycle_mean = float(item.get("tv_cycle_mean", prior_mean))
        cycle_var = float(item.get("tv_cycle_var", prior_var))
        ca = self.TV_CYCLE_ALPHA
        cycle_diff = tv - cycle_mean
        item["tv_cycle_mean"] = (1.0 - ca) * cycle_mean + ca * tv
        item["tv_cycle_var"] = max((1.0 - ca) * cycle_var + ca * cycle_diff * cycle_diff, 1e-6)

    def _update_tv_surface_memory(
        self, memory: Dict[str, Any], spot_mid: float, tv5300_mid: float, tv5400_mid: float
    ) -> None:
        left_tv = tv5300_mid - max(spot_mid - self.STRIKES["VEV_5300"], 0.0)
        right_tv = tv5400_mid - max(spot_mid - self.STRIKES["VEV_5400"], 0.0)
        spread = left_tv - right_tv
        surface = memory.setdefault("surface", {})
        mean = float(surface.get("vev_5300_5400_mean", spread))
        var = float(surface.get("vev_5300_5400_var", 16.0))
        a = self.TV_SURFACE_ALPHA
        diff = spread - mean
        surface["vev_5300_5400_mean"] = (1.0 - a) * mean + a * spread
        surface["vev_5300_5400_var"] = max((1.0 - a) * var + a * diff * diff, 1e-6)
        surface["vev_5300_5400_count"] = int(surface.get("vev_5300_5400_count", 0)) + 1

    def _adaptive_tv_stats(self, memory: Dict[str, Any], product: str) -> Tuple[float, float]:
        item = memory.get("products", {}).get(product, {})
        prior_mean = self.TIME_VALUE_MEAN[product]
        prior_std = self.TIME_VALUE_STD[product]
        if "tv_mean" not in item:
            return prior_mean, prior_std
        live_mean = float(item["tv_mean"])
        live_std = math.sqrt(max(1e-6, float(item.get("tv_var", prior_std * prior_std))))
        w = self.TV_PRIOR_WEIGHT
        mean = w * prior_mean + (1.0 - w) * live_mean
        floor = prior_std * self.TIME_VALUE_STD_FLOOR_RATIO
        std = max(floor, live_std)
        return mean, std

    def _spot_fair(self, product: str, book: Dict[str, Any], memory: Dict[str, Any]) -> float:
        item = memory["products"].get(product, {})
        ema = float(item.get("ema", book["mid"]))
        slow = float(item.get("slow", ema))
        micro_edge = book["micro"] - book["mid"]
        if product == "HYDROGEL_PACK":
            return book["mid"] + 0.35 * micro_edge + 0.04 * (ema - slow) + 0.025 * (self.HYDRO_ANCHOR - slow)
        return book["mid"] + 0.20 * micro_edge + 0.04 * (ema - slow) + 0.015 * (self.VELVET_ANCHOR - slow)

    def _trade_hydro(
        self,
        product: str,
        book: Dict[str, Any],
        fair: float,
        memory: Dict[str, Any],
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        if self.HYDRO_SWING_ENABLED:
            self._trade_hydro_swing(product, book, memory, position, result, used)
            return

        pos = position.get(product, 0)
        cap = self.RISK_CAPS[product]
        item = memory["products"].get(product, {})
        ema = float(item.get("ema", book["mid"]))
        slow = float(item.get("slow", ema))
        prev = float(item.get("prev", book["mid"]))

        passive_edge = max(3.5, 0.32 * book["spread"])
        take_edge = max(9.0, 0.70 * book["spread"])
        base_size = 8

        short = pos < 0
        lower_quantile = book["mid"] < 9961.0
        very_low = book["mid"] < 9948.0
        below_fair = book["mid"] < fair - 1.0
        below_ema = book["mid"] < ema - 2.0
        upward_turn = book["mid"] > prev + 1.0 and book["micro"] >= book["mid"]
        upward_reversal = upward_turn and (lower_quantile or below_ema)
        upper_regime = book["mid"] > self.HYDRO_ANCHOR + 18.0 or book["mid"] > ema + 4.0

        dynamic_short_cap = 16
        if lower_quantile or upward_reversal:
            dynamic_short_cap = 8
        if short and (very_low or (lower_quantile and (below_ema or upward_turn))):
            dynamic_short_cap = 6
        if upper_regime and not lower_quantile and not very_low:
            dynamic_short_cap = 16

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

        rebound_escape = short and pos <= -6 and (rebound_start or rebound_lock > 0)
        rebound_sell_lock = short and rebound_lock > 0 and book["mid"] < 9998.0
        risky_add_short = short and pos <= -dynamic_short_cap
        crowded_low_short = short and lower_quantile and pos <= -6
        low_tail_short = short and very_low
        block_new_sells = (risky_add_short and (
            lower_quantile or below_fair or (below_ema and upward_turn)
        )) or crowded_low_short or low_tail_short or rebound_sell_lock

        skew = 12.0 if short else 9.0
        fair_adj = fair - skew * pos / max(1, cap)

        if rebound_escape:
            escape_qty = min(3, max(0, -pos - 4))
            escape_edge = max(5.0, 0.55 * book["spread"])
            if escape_qty > 0 and book["best_ask"] <= fair_adj + escape_edge:
                self._add_order(product, book["best_ask"], min(escape_qty, book["ask_vol"]),
                                position, result, used,
                                long_cap_override=self.LONG_CAPS[product],
                                short_cap_override=dynamic_short_cap)

        buyback_needed = short and (
            (pos <= -5 and very_low and book["micro"] >= book["mid"] - 0.5)
            or (pos <= -6 and lower_quantile and (upward_reversal or book["micro"] >= book["mid"]))
        )
        if buyback_needed:
            buyback_qty = min(3 if pos <= -8 else 2, -pos)
            buyback_edge = max(2.0, 0.22 * book["spread"])
            if book["best_ask"] <= fair_adj + buyback_edge:
                self._add_order(product, book["best_ask"], min(buyback_qty, book["ask_vol"]),
                                position, result, used,
                                long_cap_override=self.LONG_CAPS[product],
                                short_cap_override=dynamic_short_cap)

        if book["best_ask"] <= fair_adj - take_edge:
            self._add_order(product, book["best_ask"], min(max(2, base_size // 2), book["ask_vol"]),
                            position, result, used,
                            long_cap_override=self.LONG_CAPS[product],
                            short_cap_override=dynamic_short_cap)

        if not block_new_sells and book["best_bid"] >= fair_adj + take_edge:
            sell_qty = min(max(2, base_size // 2), book["bid_vol"])
            self._add_order(product, book["best_bid"], -sell_qty, position, result, used,
                            long_cap_override=self.LONG_CAPS[product],
                            short_cap_override=dynamic_short_cap)

        sell_base = base_size
        if lower_quantile or upward_reversal:
            sell_base = 2 if pos >= 0 else 1
        buy_base = base_size + 1 if short else base_size
        self._passive_quote(
            product, book, fair_adj, passive_edge, base_size,
            position, result, used,
            allow_buy=True,
            allow_sell=not block_new_sells,
            buy_base_size=buy_base,
            sell_base_size=sell_base,
            long_cap_override=self.LONG_CAPS[product],
            short_cap_override=dynamic_short_cap,
        )

    def _trade_hydro_swing(
        self,
        product: str,
        book: Dict[str, Any],
        memory: Dict[str, Any],
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        pos = position.get(product, 0)
        cap = self.HYDRO_SWING_CAP
        size = self.HYDRO_SWING_SIZE
        bias = self._mark_bias(memory, product)
        mirror_shift = self._mirror_hydro_shift(memory)

        # Adaptive anchor: use the slow EMA which tracks regime shifts
        item = memory.get("products", {}).get(product, {})
        slow = float(item.get("slow", book["mid"]))

        long_entry = slow + self.HYDRO_LONG_ENTRY_OFFSET + 3.0 * bias + mirror_shift
        long_exit = slow + self.HYDRO_LONG_EXIT_OFFSET + 2.0 * bias + mirror_shift
        short_entry = slow + self.HYDRO_SHORT_ENTRY_OFFSET + 3.0 * bias + mirror_shift
        short_exit = slow + self.HYDRO_SHORT_EXIT_OFFSET + 2.0 * bias + mirror_shift
        if pos > 0 and book["best_bid"] >= long_exit:
            sell_qty = min(pos, book["bid_vol"], size)
            self._add_order(
                product, book["best_bid"], -sell_qty, position, result, used,
                long_cap_override=cap, short_cap_override=cap,
            )
        elif pos > self.HYDRO_SCALE_OUT_POS and book["best_bid"] >= long_entry + self.HYDRO_SCALE_OUT_EDGE:
            sell_qty = min(pos - self.HYDRO_SCALE_OUT_POS, book["bid_vol"], self.HYDRO_SCALE_OUT_SIZE)
            self._add_order(
                product, book["best_bid"], -sell_qty, position, result, used,
                long_cap_override=cap, short_cap_override=cap,
            )

        if pos < 0 and book["best_ask"] <= short_exit:
            buy_qty = min(-pos, book["ask_vol"], size)
            self._add_order(
                product, book["best_ask"], buy_qty, position, result, used,
                long_cap_override=cap, short_cap_override=cap,
            )
        elif pos < -self.HYDRO_SCALE_OUT_POS and book["best_ask"] <= short_entry - self.HYDRO_SCALE_OUT_EDGE:
            buy_qty = min(-self.HYDRO_SCALE_OUT_POS - pos, book["ask_vol"], self.HYDRO_SCALE_OUT_SIZE)
            self._add_order(
                product, book["best_ask"], buy_qty, position, result, used,
                long_cap_override=cap, short_cap_override=cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)

        if book["best_ask"] <= long_entry and projected < cap:
            buy_room = cap - projected
            buy_qty = min(buy_room, book["ask_vol"], size)
            self._add_order(
                product, book["best_ask"], buy_qty, position, result, used,
                long_cap_override=cap, short_cap_override=cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)

        if book["best_bid"] >= short_entry and projected > -cap:
            sell_room = cap + projected
            sell_qty = min(sell_room, book["bid_vol"], size)
            self._add_order(
                product, book["best_bid"], -sell_qty, position, result, used,
                long_cap_override=cap, short_cap_override=cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)
        if (
            projected < cap
            and book["best_bid"] < book["best_ask"] - 1
            and book["mid"] <= slow + self.HYDRO_PASSIVE_OFFSET_LOW + mirror_shift
        ):
            buy_qty = min(self.HYDRO_PASSIVE_SIZE, cap - projected)
            self._add_order(
                product, book["best_bid"] + 1, buy_qty, position, result, used,
                long_cap_override=cap, short_cap_override=cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)
        if (
            projected > -cap
            and book["best_bid"] < book["best_ask"] - 1
            and book["mid"] >= slow + self.HYDRO_PASSIVE_OFFSET_HIGH + mirror_shift
        ):
            sell_qty = min(self.HYDRO_PASSIVE_SIZE, cap + projected)
            self._add_order(
                product, book["best_ask"] - 1, -sell_qty, position, result, used,
                long_cap_override=cap, short_cap_override=cap,
            )

    def _trade_price_swing(
        self,
        product: str,
        book: Dict[str, Any],
        memory: Dict[str, Any],
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        le_off, lx_off, se_off, sx_off, cap, levels = self.PRICE_SWING_CONFIG[product]
        bias = self._mark_bias(memory, product)
        mirror_shift = self._mirror_price_shift(memory, product)

        # Adaptive anchor: use the slow EMA for this product
        item = memory.get("products", {}).get(product, {})
        slow = float(item.get("slow", 0))
        if slow == 0:
            return  # EMA not yet initialized

        long_entry_adj = slow + le_off + 1.5 * bias + mirror_shift
        long_exit_adj = slow + lx_off + 0.8 * bias + mirror_shift
        short_entry_adj = slow + se_off + 1.5 * bias + mirror_shift
        short_exit_adj = slow + sx_off + 0.8 * bias + mirror_shift

        pos = position.get(product, 0)

        if pos > 0:
            self._take_book_levels(
                product, book, "sell", min(pos, cap), long_exit_adj, levels,
                position, result, used, cap, cap,
            )
        if pos < 0:
            self._take_book_levels(
                product, book, "buy", min(-pos, cap), short_exit_adj, levels,
                position, result, used, cap, cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)
        if projected < cap:
            self._take_book_levels(
                product, book, "buy", cap - projected, long_entry_adj, levels,
                position, result, used, cap, cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)
        if projected > -cap:
            self._take_book_levels(
                product, book, "sell", cap + projected, short_entry_adj, levels,
                position, result, used, cap, cap,
            )

    def _trade_spot(
        self,
        product: str,
        book: Dict[str, Any],
        fair: float,
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        pos = position.get(product, 0)
        cap = self.RISK_CAPS[product]
        if product == "HYDROGEL_PACK":
            passive_edge = max(4.0, 0.34 * book["spread"])
            take_edge = max(10.0, 0.75 * book["spread"])
            size = 7
            skew = 9.0
        else:
            passive_edge = max(1.5, 0.35 * book["spread"])
            take_edge = max(4.5, 0.90 * book["spread"])
            size = 8
            skew = 5.0

        fair_adj = fair - skew * pos / max(1, cap)
        self._take_if_clear(product, book, fair_adj, take_edge, max(2, size // 2),
                            position, result, used)
        self._passive_quote(product, book, fair_adj, passive_edge, size,
                            position, result, used)

    def _trade_deep_itm_parity(
        self,
        product: str,
        book: Dict[str, Any],
        fair: float,
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        pos = position.get(product, 0)
        cap = self.RISK_CAPS[product]
        if cap <= 0:
            return

        fair_adj = fair - 10.0 * pos / cap
        edge = max(6.5, 0.38 * book["spread"])
        take_size = self.DEEP_ITM_TAKE_SIZE.get(product, 1)
        quote_size = self.DEEP_ITM_QUOTE_SIZE.get(product, 1)
        self._take_if_clear(product, book, fair_adj, edge + 5.0, take_size, position, result, used)

        if book["spread"] >= 12:
            self._passive_quote(product, book, fair_adj, edge, quote_size, position, result, used)

    def _trade_time_value_call(
        self,
        product: str,
        book: Dict[str, Any],
        spot_mid: float,
        memory: Dict[str, Any],
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        long_cap = self.TIME_VALUE_LONG_CAP.get(product, 0)
        short_cap = self.TIME_VALUE_SHORT_CAP.get(product, 0)
        if long_cap <= 0 and short_cap <= 0:
            return

        pos = position.get(product, 0)
        strike = self.STRIKES[product]
        intrinsic = max(spot_mid - strike, 0.0)
        mean, std = self._adaptive_tv_stats(memory, product)
        entry_z = self.TIME_VALUE_ENTRY_Z[product]
        short_entry_z = self.TIME_VALUE_SHORT_ENTRY_Z[product]
        item = memory.get("products", {}).get(product, {})
        tv_count = int(item.get("tv_count", 0))
        bias = self._mark_bias(memory, product)
        otm_sell_bias = float(item.get("otm_sell_bias", 0.0))

        # Shift the time-value center conservatively using counterparty flow.
        mean += 0.45 * bias
        if product == "VEV_5300":
            mean -= 0.35 * otm_sell_bias

        long_entry_tv = mean - entry_z * std
        short_entry_tv = mean + short_entry_z * std

        ask_tv = book["best_ask"] - intrinsic
        bid_tv = book["best_bid"] - intrinsic

        order_long_cap = max(long_cap, short_cap)
        order_short_cap = max(long_cap, short_cap)

        long_exit_tv = mean + 0.3 * std
        short_exit_tv = mean - 0.3 * std

        if pos > 0 and bid_tv >= long_exit_tv:
            already_selling = used.get(product, {}).get("sell", 0)
            sell_room = max(0, pos - already_selling)
            sell_qty = min(sell_room, book["bid_vol"])
            if sell_qty > 0:
                self._add_order(product, book["best_bid"], -sell_qty,
                                position, result, used,
                                long_cap_override=order_long_cap,
                                short_cap_override=order_short_cap)

        if pos < 0 and ask_tv <= short_exit_tv:
            already_buying = used.get(product, {}).get("buy", 0)
            buy_room = max(0, -pos - already_buying)
            buy_qty = min(buy_room, book["ask_vol"])
            if buy_qty > 0:
                self._add_order(product, book["best_ask"], buy_qty,
                                position, result, used,
                                long_cap_override=order_long_cap,
                                short_cap_override=order_short_cap)

        if tv_count < self.TIME_VALUE_MIN_OBS:
            return

        if long_cap > 0 and ask_tv <= long_entry_tv and pos < long_cap:
            buy_room = max(0, long_cap - pos - used.get(product, {}).get("buy", 0))
            buy_qty = min(book["ask_vol"], buy_room)
            if buy_qty > 0:
                self._add_order(product, book["best_ask"], buy_qty,
                                position, result, used,
                                long_cap_override=long_cap,
                                short_cap_override=order_short_cap)

        if short_cap > 0 and bid_tv >= short_entry_tv and pos > -short_cap:
            sell_room = max(0, short_cap + pos - used.get(product, {}).get("sell", 0))
            sell_qty = min(book["bid_vol"], sell_room)
            if sell_qty > 0:
                self._add_order(product, book["best_bid"], -sell_qty,
                                position, result, used,
                                long_cap_override=order_long_cap,
                                short_cap_override=short_cap)

        fair_option = intrinsic + mean
        skew = self.TIME_VALUE_INVENTORY_SKEW[product]
        cap_for_skew = max(1, max(long_cap, short_cap))
        fair_adj = fair_option - skew * pos / cap_for_skew
        quote_edge = self.TIME_VALUE_QUOTE_EDGE[product]
        quote_size = self.TIME_VALUE_QUOTE_SIZE[product]
        self._passive_quote(
            product, book, fair_adj, quote_edge, quote_size,
            position, result, used,
            allow_buy=long_cap > 0 and pos < long_cap,
            allow_sell=short_cap > 0 and pos > -short_cap,
            long_cap_override=long_cap,
            short_cap_override=short_cap,
        )

        self._trade_tv_satellite(
            product, book, intrinsic, memory, position, result, used, long_cap, short_cap
        )

    def _trade_tv_satellite(
        self,
        product: str,
        book: Dict[str, Any],
        intrinsic: float,
        memory: Dict[str, Any],
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
        long_cap: int,
        short_cap: int,
    ) -> None:
        cfg = self.TV_SATELLITE_CONFIG.get(product)
        if cfg is None:
            return
        core_pos, min_hold, bias_room, sell_edge, buy_edge, clip = cfg
        item = memory.get("products", {}).get(product, {})
        if int(item.get("long_hold_ticks", 0)) < min_hold:
            return

        prior_mean = self.TIME_VALUE_MEAN[product]
        prior_std = self.TIME_VALUE_STD[product]
        live_mean = float(item.get("tv_cycle_mean", prior_mean))
        live_std = math.sqrt(max(1e-6, float(item.get("tv_cycle_var", prior_std * prior_std))))
        tv_count = int(item.get("tv_count", 0))
        if tv_count < self.TV_MATURE_COUNT:
            return
        flow_bias = float(item.get("otm_sell_bias", 0.0))
        flow_push = max(-bias_room, min(bias_room, flow_bias))
        if live_mean >= prior_mean - 0.25 * prior_std and flow_push < 0.75:
            return

        bid_tv = book["best_bid"] - intrinsic
        ask_tv = book["best_ask"] - intrinsic
        projected = position.get(product, 0) + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)

        dynamic_core = core_pos
        dynamic_core -= int(round(28.0 * max(flow_push, 0.0)))
        dynamic_core += int(round(12.0 * max(-flow_push, 0.0)))
        dynamic_core = max(120, min(long_cap - clip, dynamic_core))

        upper = live_mean + max(0.35, 0.55 * live_std, sell_edge - 0.15 * max(flow_push, 0.0))
        lower = live_mean - max(0.25, 0.45 * live_std, buy_edge + 0.10 * max(flow_push, 0.0))

        if projected > dynamic_core and bid_tv >= upper:
            sell_qty = min(clip, projected - dynamic_core, book["bid_vol"])
            if sell_qty > 0:
                self._add_order(
                    product, book["best_bid"], -sell_qty,
                    position, result, used,
                    long_cap_override=long_cap,
                    short_cap_override=short_cap,
                )

        projected = position.get(product, 0) + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)
        if projected >= dynamic_core and projected < long_cap and ask_tv <= lower:
            buy_qty = min(clip, long_cap - projected, book["ask_vol"])
            if buy_qty > 0:
                self._add_order(
                    product, book["best_ask"], buy_qty,
                    position, result, used,
                    long_cap_override=long_cap,
                    short_cap_override=short_cap,
                )

    def _trade_tv_surface_rotation(
        self,
        books: Dict[str, Dict[str, Any]],
        spot_mid: float,
        memory: Dict[str, Any],
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        if "VEV_5300" not in books or "VEV_5400" not in books:
            return
        left_item = memory.get("products", {}).get("VEV_5300", {})
        right_item = memory.get("products", {}).get("VEV_5400", {})
        if min(int(left_item.get("long_hold_ticks", 0)), int(right_item.get("long_hold_ticks", 0))) < self.TV_SURFACE_MIN_HOLD:
            return

        surface = memory.get("surface", {})
        count = int(surface.get("vev_5300_5400_count", 0))
        if count < self.TV_SURFACE_MIN_COUNT:
            return

        mean = float(surface.get("vev_5300_5400_mean", 0.0))
        std = math.sqrt(max(1e-6, float(surface.get("vev_5300_5400_var", 16.0))))
        edge = max(self.TV_SURFACE_EDGE_FLOOR, self.TV_SURFACE_STD_MULT * std)

        left_tv = books["VEV_5300"]["best_bid"] - max(spot_mid - self.STRIKES["VEV_5300"], 0.0)
        right_tv = books["VEV_5400"]["best_bid"] - max(spot_mid - self.STRIKES["VEV_5400"], 0.0)
        spread = left_tv - right_tv

        if spread >= mean + edge:
            projected = position.get("VEV_5300", 0) + used.get("VEV_5300", {}).get("buy", 0) - used.get("VEV_5300", {}).get("sell", 0)
            core = self.TV_SURFACE_CORE["VEV_5300"]
            sell_qty = min(self.TV_SURFACE_CLIP, max(0, projected - core), books["VEV_5300"]["bid_vol"])
            if sell_qty > 0:
                self._add_order(
                    "VEV_5300", books["VEV_5300"]["best_bid"], -sell_qty,
                    position, result, used,
                    long_cap_override=self.TIME_VALUE_LONG_CAP["VEV_5300"],
                    short_cap_override=self.TIME_VALUE_SHORT_CAP["VEV_5300"],
                )

    def _take_book_levels(
        self,
        product: str,
        book: Dict[str, Any],
        side: str,
        qty: int,
        limit_price: float,
        levels: int,
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
        long_cap: int,
        short_cap: int,
    ) -> None:
        if qty <= 0:
            return
        filled = 0
        if side == "buy":
            for price, volume in book["ask_levels"][:levels]:
                if price > limit_price or filled >= qty:
                    break
                take = min(volume, qty - filled)
                before = used.get(product, {}).get("buy", 0)
                self._add_order(
                    product, price, take, position, result, used,
                    long_cap_override=long_cap, short_cap_override=short_cap,
                )
                filled += max(0, used.get(product, {}).get("buy", 0) - before)
        else:
            for price, volume in book["bid_levels"][:levels]:
                if price < limit_price or filled >= qty:
                    break
                take = min(volume, qty - filled)
                before = used.get(product, {}).get("sell", 0)
                self._add_order(
                    product, price, -take, position, result, used,
                    long_cap_override=long_cap, short_cap_override=short_cap,
                )
                filled += max(0, used.get(product, {}).get("sell", 0) - before)

    def _hedge_option_delta(
        self,
        book: Dict[str, Any],
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        if self.VFE_OPTION_HEDGE_RATIO <= 0:
            return
        net_delta = 0.0
        for product, delta in self.OPTION_DELTAS.items():
            net_delta += delta * position.get(product, 0)

        target = int(round(-self.VFE_OPTION_HEDGE_RATIO * net_delta))
        target = max(-self.VFE_OPTION_HEDGE_CLIP, min(self.VFE_OPTION_HEDGE_CLIP, target))

        vfe_used = used.get(self.UNDERLYING, {"buy": 0, "sell": 0})
        current = position.get(self.UNDERLYING, 0) + vfe_used.get("buy", 0) - vfe_used.get("sell", 0)
        diff = target - current
        if abs(diff) < self.VFE_OPTION_HEDGE_TRIGGER:
            return

        qty = min(abs(diff), self.VFE_OPTION_HEDGE_SLICE)
        if diff > 0:
            self._add_order(
                self.UNDERLYING, book["best_ask"], min(qty, book["ask_vol"]),
                position, result, used
            )
        else:
            self._add_order(
                self.UNDERLYING, book["best_bid"], -min(qty, book["bid_vol"]),
                position, result, used
            )

    def _take_if_clear(
        self,
        product: str,
        book: Dict[str, Any],
        fair: float,
        edge: float,
        size: int,
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        if book["best_ask"] <= fair - edge:
            self._add_order(product, book["best_ask"], min(size, book["ask_vol"]),
                            position, result, used)
        if book["best_bid"] >= fair + edge:
            self._add_order(product, book["best_bid"], -min(size, book["bid_vol"]),
                            position, result, used)

    def _passive_quote(
        self,
        product: str,
        book: Dict[str, Any],
        fair: float,
        edge: float,
        base_size: int,
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
        allow_buy: bool = True,
        allow_sell: bool = True,
        buy_base_size: Optional[int] = None,
        sell_base_size: Optional[int] = None,
        long_cap_override: Optional[int] = None,
        short_cap_override: Optional[int] = None,
    ) -> None:
        pos = position.get(product, 0)
        cap = self.RISK_CAPS.get(product, self.LIMITS[product])
        if cap <= 0:
            return

        buy_base = base_size if buy_base_size is None else buy_base_size
        sell_base = base_size if sell_base_size is None else sell_base_size
        buy_size = max(1, int(buy_base * max(0.25, 1.0 - max(pos, 0) / cap)))
        sell_size = max(1, int(sell_base * max(0.25, 1.0 + min(pos, 0) / cap)))

        bid_candidate = int(math.floor(fair - edge))
        ask_candidate = int(math.ceil(fair + edge))
        if book["spread"] > 1:
            bid_price = min(book["best_bid"] + 1, bid_candidate)
            ask_price = max(book["best_ask"] - 1, ask_candidate)
        else:
            bid_price = min(book["best_bid"], bid_candidate)
            ask_price = max(book["best_ask"], ask_candidate)

        if allow_buy and bid_price >= 0 and bid_price < book["best_ask"] and fair - bid_price >= 0.75 * edge:
            self._add_order(product, bid_price, buy_size, position, result, used,
                            long_cap_override=long_cap_override,
                            short_cap_override=short_cap_override)
        if allow_sell and ask_price > book["best_bid"] and ask_price - fair >= 0.75 * edge:
            self._add_order(product, ask_price, -sell_size, position, result, used,
                            long_cap_override=long_cap_override,
                            short_cap_override=short_cap_override)

    def _add_order(
        self,
        product: str,
        price: int,
        qty: int,
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
        long_cap_override: Optional[int] = None,
        short_cap_override: Optional[int] = None,
    ) -> None:
        if qty == 0 or product not in self.LIMITS:
            return
        cap = min(self.LIMITS[product], self.RISK_CAPS.get(product, self.LIMITS[product]))
        long_raw = self.LONG_CAPS.get(product, cap) if long_cap_override is None else long_cap_override
        short_raw = self.SHORT_CAPS.get(product, cap) if short_cap_override is None else short_cap_override
        long_cap = min(self.LIMITS[product], long_raw)
        short_cap = min(self.LIMITS[product], short_raw)
        if cap <= 0 or long_cap <= 0 or short_cap <= 0:
            return

        pos = position.get(product, 0)
        product_used = used.setdefault(product, {"buy": 0, "sell": 0})

        if qty > 0:
            room = min(self.LIMITS[product] - pos - product_used["buy"],
                       long_cap - pos - product_used["buy"])
            qty = min(qty, max(0, room))
            if qty <= 0:
                return
            product_used["buy"] += qty
        else:
            room = min(self.LIMITS[product] + pos - product_used["sell"],
                       short_cap + pos - product_used["sell"])
            sell_qty = min(-qty, max(0, room))
            if sell_qty <= 0:
                return
            qty = -sell_qty
            product_used["sell"] += sell_qty

        result.setdefault(product, []).append(Order(product, int(price), int(qty)))