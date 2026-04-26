try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    class Order:
        def __init__(self, symbol, price, quantity):
            self.symbol = symbol; self.price = price; self.quantity = quantity
    class OrderDepth:
        def __init__(self):
            self.buy_orders = {}; self.sell_orders = {}
    class TradingState:
        def __init__(self):
            self.timestamp = 0; self.traderData = ""; self.position = {}; self.order_depths = {}

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
        "HYDROGEL_PACK": 60,
    }

    SHORT_CAPS = {
        "HYDROGEL_PACK": 16,
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
    # Change 6: adaptive HP thresholds via EMA + offsets (offsets = threshold - hist_mean)
    HYDRO_LONG_ENTRY_OFFSET   = -29   # 9950 - 9979
    HYDRO_LONG_EXIT_OFFSET    =   6   # 9985 - 9979
    HYDRO_SHORT_ENTRY_OFFSET  =   1   # 9980 - 9979
    HYDRO_SHORT_EXIT_OFFSET   = -29   # 9950 - 9979
    HYDRO_EMA_ALPHA           = 0.0005
    HYDRO_EMA_HIST_MEAN       = 9979.0
    HYDRO_EMA_BOOTSTRAP_TICKS = 1000

    # Change 3: EMA-offset swing thresholds. Offsets derived from day-2 mean.
    PRICE_SWING_OFFSETS = {
        "VELVETFRUIT_EXTRACT": {
            "long_entry": -13, "long_exit": 12, "short_entry": 12, "short_exit": -16,
            "cap": 200, "levels": 2,
        },
        "VEV_4000": {
            "long_entry": -5, "long_exit": 6, "short_entry": 1, "short_exit": -9,
            "cap": 300, "levels": 2,
        },
        "VEV_4500": {
            "long_entry": -10, "long_exit": 1, "short_entry": 1, "short_exit": -14,
            "cap": 300, "levels": 2,
        },
        "VEV_5000": {
            "long_entry": -14, "long_exit": 1, "short_entry": 6, "short_exit": -17,
            "cap": 300, "levels": 2,
        },
        "VEV_5100": {
            "long_entry": -16, "long_exit": 3, "short_entry": 3, "short_exit": -16,
            "cap": 300, "levels": 2,
        },
        "VEV_5200": {
            "long_entry": -10, "long_exit": 3, "short_entry": 3, "short_exit": -10,
            "cap": 300, "levels": 2,
        },
    }
    # Hardcoded fallback used before swing EMA bootstraps (< SWING_EMA_BOOTSTRAP ticks)
    PRICE_SWING_FALLBACK = {
        "VELVETFRUIT_EXTRACT": (5246, 5271, 5271, 5243, 200, 2),
        "VEV_4000": (1256, 1266, 1262, 1256, 300, 2),
        "VEV_4500": (755, 766, 766, 751, 300, 2),
        "VEV_5000": (253, 269, 274, 250, 300, 2),
        "VEV_5100": (162, 181, 181, 162, 300, 2),
        "VEV_5200": (92, 105, 105, 92, 300, 2),
    }
    SWING_EMA_ALPHA     = 0.0005
    SWING_EMA_BOOTSTRAP = 200

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
    # Change 4: VEV_5500 short entry z lowered from 0.80 to 0.45
    TIME_VALUE_SHORT_ENTRY_Z = {
        "VEV_5200": 0.65,
        "VEV_5300": 0.70,
        "VEV_5400": 0.50,
        "VEV_5500": 0.45,
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

    # Change 1: full delta hedging, all active strikes included
    OPTION_DELTAS = {
        "VEV_4000": 1.00,
        "VEV_4500": 1.00,
        "VEV_5000": 0.95,
        "VEV_5100": 0.85,
        "VEV_5200": 0.56,
        "VEV_5300": 0.33,
        "VEV_5400": 0.13,
        "VEV_5500": 0.06,
    }
    VFE_OPTION_HEDGE_RATIO   = 1.0
    VFE_OPTION_HEDGE_TRIGGER = 12
    VFE_OPTION_HEDGE_CLIP    = 150
    VFE_OPTION_HEDGE_SLICE   = 25

    DEEP_ITM_TAKE_SIZE = {
        "VEV_4000": 8,
    }
    DEEP_ITM_QUOTE_SIZE = {
        "VEV_4000": 6,
    }

    TV_EMA_ALPHA    = 1.00
    TV_PRIOR_WEIGHT = 0.60

    LOG_EVERY = 1000

    # Change 2: 5300/5400/5500 butterfly arb
    FLY_5300_5400_5500_MEAN_PRIOR = 24.36
    FLY_5300_5400_5500_STD_PRIOR  = 1.0
    FLY_EWMA_ALPHA = 0.05
    FLY_PRIOR_W    = 0.6
    FLY_ENTRY_Z    = 0.75
    FLY_EXIT_Z     = 0.2
    FLY_MAX_QTY    = 100
    FLY_LEG_SIZE   = 20

    # Change 5: end-of-window position flatten
    ROUND_END_TS       = 1_000_000
    FLATTEN_BEGIN_TS   = 850_000
    FLATTEN_FINAL_FRAC = 0.15

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

        # Change 3: update slow swing EMAs for all products in PRICE_SWING_OFFSETS
        for product in self.PRICE_SWING_OFFSETS:
            if product in books:
                self._update_swing_ema(memory, product, books[product]["mid"])

        # Change 6: update HP EMA
        if "HYDROGEL_PACK" in books:
            self._update_hp_ema(memory, books["HYDROGEL_PACK"]["mid"])

        # Change 5: compute flatten factor (1.0 before FLATTEN_BEGIN_TS, scales to FLATTEN_FINAL_FRAC at ROUND_END_TS)
        progress = max(0.0, (state.timestamp - self.FLATTEN_BEGIN_TS) /
                       (self.ROUND_END_TS - self.FLATTEN_BEGIN_TS))
        progress = min(1.0, progress)
        flatten_factor = 1.0 - progress * (1.0 - self.FLATTEN_FINAL_FRAC)

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

        # Step 6: HYDROGEL_PACK with adaptive thresholds (Change 6) and scaled cap (Change 5)
        if "HYDROGEL_PACK" in fairs:
            self._trade_hydro(
                "HYDROGEL_PACK", books["HYDROGEL_PACK"], fairs["HYDROGEL_PACK"],
                memory, state.position, result, used,
                cap_scale=flatten_factor,
            )

        # Step 7: VFE and VEV_4000/4500/5000/5100/5200 via price-swing with EMA offsets (Changes 3, 5)
        if self.UNDERLYING in fairs and self.UNDERLYING not in self.PRICE_SWING_OFFSETS:
            self._trade_spot(
                self.UNDERLYING, books[self.UNDERLYING], fairs[self.UNDERLYING],
                state.position, result, used
            )

        for product in self.PRICE_SWING_OFFSETS:
            if product in books:
                self._trade_price_swing(
                    product, books[product], memory, state.position, result, used,
                    cap_scale=flatten_factor,
                )

        for product in ("VEV_4000",):
            if product in fairs and product not in self.PRICE_SWING_OFFSETS:
                self._trade_deep_itm_parity(
                    product, books[product], fairs[product], state.position, result, used
                )

        if self.UNDERLYING in books:
            spot_mid = books[self.UNDERLYING]["mid"]

            # Step 8: VEV_5300/5400/5500 time-value with K=5500 z fix (Change 4) and scaled cap (Change 5)
            for product in self.TIME_VALUE_OPTIONS:
                if product in books:
                    self._trade_time_value_call(
                        product, books[product], spot_mid, memory,
                        state.position, result, used,
                        cap_scale=flatten_factor,
                    )

            # Step 9: butterfly arb on 5300/5400/5500 (Change 2), qty scaled (Change 5)
            fly_legs = ("VEV_5300", "VEV_5400", "VEV_5500")
            if all(p in books for p in fly_legs):
                fly_max = max(1, int(self.FLY_MAX_QTY * flatten_factor))
                self._trade_fly_5300_5400_5500(
                    books, state.position, memory, result, used, max_qty=fly_max
                )

            # Step 10: delta hedge — NOT scaled; remains fully active (Change 1)
            self._hedge_option_delta(
                books[self.UNDERLYING], state.position, result, used
            )

        # Periodic log with netΔ (Change 1e)
        tick = memory.get("tick", 0) + 1
        memory["tick"] = tick
        if tick % self.LOG_EVERY == 0:
            net_delta = sum(
                self.OPTION_DELTAS.get(p, 0.0) * state.position.get(p, 0)
                for p in self.OPTION_DELTAS
            )
            print(
                f"t={state.timestamp} tick={tick} netΔ={net_delta:.1f} "
                f"fly_pos={memory.get('fly_pos', 0)} flatten={flatten_factor:.3f}"
            )

        trader_data = json.dumps(memory, separators=(",", ":"))
        return result, 0, trader_data

    # ── memory helpers ────────────────────────────────────────────────────────

    def _load_memory(self, trader_data: str) -> Dict[str, Any]:
        if trader_data:
            try:
                decoded = json.loads(trader_data)
                if isinstance(decoded, dict) and isinstance(decoded.get("products"), dict):
                    return decoded
            except Exception:
                pass
        return {"products": {}}

    def _update_swing_ema(self, memory: Dict[str, Any], product: str, mid: float) -> None:
        item = memory.setdefault("swing_ema", {}).setdefault(product, {"mid": mid, "count": 0})
        if item["count"] == 0:
            item["mid"] = mid
        else:
            item["mid"] = (1.0 - self.SWING_EMA_ALPHA) * item["mid"] + self.SWING_EMA_ALPHA * mid
        item["count"] = item["count"] + 1

    def _update_hp_ema(self, memory: Dict[str, Any], mid: float) -> None:
        item = memory.setdefault("hp_ema", {"mid": self.HYDRO_EMA_HIST_MEAN, "count": 0})
        if item["count"] == 0:
            item["mid"] = self.HYDRO_EMA_HIST_MEAN
        else:
            item["mid"] = (1.0 - self.HYDRO_EMA_ALPHA) * item["mid"] + self.HYDRO_EMA_ALPHA * mid
        item["count"] = item["count"] + 1

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
            return
        last = float(item.get("last", mid))
        item["prev"] = last
        item["ema"] = 0.10 * mid + 0.90 * float(item.get("ema", mid))
        item["slow"] = 0.01 * mid + 0.99 * float(item.get("slow", mid))
        item["vol"] = 0.08 * abs(mid - last) + 0.92 * float(item.get("vol", 1.0))
        item["last"] = mid

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

    # ── trading strategies ────────────────────────────────────────────────────

    def _trade_hydro(
        self,
        product: str,
        book: Dict[str, Any],
        fair: float,
        memory: Dict[str, Any],
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
        cap_scale: float = 1.0,
    ) -> None:
        if self.HYDRO_SWING_ENABLED:
            self._trade_hydro_swing(product, book, memory, position, result, used, cap_scale=cap_scale)
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
        cap_scale: float = 1.0,
    ) -> None:
        pos = position.get(product, 0)
        cap = max(1, int(self.HYDRO_SWING_CAP * cap_scale))
        size = self.HYDRO_SWING_SIZE
        # Taper entry size as inventory builds; exits always use full size
        entry_size = max(8, int(size * max(0.5, 1.0 - 0.5 * abs(pos) / max(cap, 1))))

        # Change 6: adaptive thresholds from HP EMA
        hp_item = memory.get("hp_ema", {})
        hp_count = int(hp_item.get("count", 0))
        if hp_count < self.HYDRO_EMA_BOOTSTRAP_TICKS:
            ema = self.HYDRO_EMA_HIST_MEAN
        else:
            ema = float(hp_item.get("mid", self.HYDRO_EMA_HIST_MEAN))
        long_entry  = ema + self.HYDRO_LONG_ENTRY_OFFSET
        long_exit   = ema + self.HYDRO_LONG_EXIT_OFFSET
        short_entry = ema + self.HYDRO_SHORT_ENTRY_OFFSET
        short_exit  = ema + self.HYDRO_SHORT_EXIT_OFFSET

        if pos > 0 and book["best_bid"] >= long_exit:
            sell_qty = min(pos, book["bid_vol"], size)
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

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)

        if book["best_ask"] <= long_entry and projected < cap:
            buy_room = cap - projected
            buy_qty = min(buy_room, book["ask_vol"], entry_size)
            self._add_order(
                product, book["best_ask"], buy_qty, position, result, used,
                long_cap_override=cap, short_cap_override=cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)

        if book["best_bid"] >= short_entry and projected > -cap:
            sell_room = cap + projected
            sell_qty = min(sell_room, book["bid_vol"], entry_size)
            self._add_order(
                product, book["best_bid"], -sell_qty, position, result, used,
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
        cap_scale: float = 1.0,
    ) -> None:
        offsets = self.PRICE_SWING_OFFSETS[product]
        swing_item = memory.get("swing_ema", {}).get(product, {})
        count = int(swing_item.get("count", 0))

        if count < self.SWING_EMA_BOOTSTRAP:
            # Use hardcoded fallback before EMA has converged
            le, lx, se, sx, cap_base, levels = self.PRICE_SWING_FALLBACK[product]
            cap = max(1, int(cap_base * cap_scale))
        else:
            ema = float(swing_item["mid"])
            le = ema + offsets["long_entry"]
            lx = ema + offsets["long_exit"]
            se = ema + offsets["short_entry"]
            sx = ema + offsets["short_exit"]
            cap = max(1, int(offsets["cap"] * cap_scale))
            levels = offsets["levels"]

        pos = position.get(product, 0)

        if pos > 0:
            self._take_book_levels(
                product, book, "sell", min(pos, cap), lx, levels,
                position, result, used, cap, cap,
            )
        if pos < 0:
            self._take_book_levels(
                product, book, "buy", min(-pos, cap), sx, levels,
                position, result, used, cap, cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)
        if projected < cap:
            self._take_book_levels(
                product, book, "buy", cap - projected, le, levels,
                position, result, used, cap, cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)
        if projected > -cap:
            self._take_book_levels(
                product, book, "sell", cap + projected, se, levels,
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
        cap_scale: float = 1.0,
    ) -> None:
        long_cap = max(0, int(self.TIME_VALUE_LONG_CAP.get(product, 0) * cap_scale))
        short_cap = max(0, int(self.TIME_VALUE_SHORT_CAP.get(product, 0) * cap_scale))
        if long_cap <= 0 and short_cap <= 0:
            return

        pos = position.get(product, 0)
        strike = self.STRIKES[product]
        intrinsic = max(spot_mid - strike, 0.0)
        mean, std = self._adaptive_tv_stats(memory, product)
        item = memory.get("products", {}).get(product, {})
        tv_count = int(item.get("tv_count", 0))

        # Adaptive z: scale by (live_std / prior_std)^0.3 so calmer regimes
        # tighten bands (more trades) and volatile regimes widen them (fewer whipsaws).
        prior_std = self.TIME_VALUE_STD[product]
        live_std_raw = math.sqrt(max(1e-6, float(item.get("tv_var", prior_std ** 2))))
        vol_z_scale = max(0.4, min(2.5, live_std_raw / prior_std)) ** 0.3
        entry_z = self.TIME_VALUE_ENTRY_Z[product] * vol_z_scale
        short_entry_z = self.TIME_VALUE_SHORT_ENTRY_Z[product] * vol_z_scale

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

    def _trade_fly_5300_5400_5500(
        self,
        books: Dict[str, Any],
        position: Dict[str, int],
        memory: Dict[str, Any],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
        max_qty: int = 100,
    ) -> None:
        fv = (books["VEV_5300"]["mid"]
              - 2.0 * books["VEV_5400"]["mid"]
              + books["VEV_5500"]["mid"])

        # Update EWMA stats
        fly_mem = memory.setdefault("fly_5300_5400_5500", {
            "mean": self.FLY_5300_5400_5500_MEAN_PRIOR,
            "var": self.FLY_5300_5400_5500_STD_PRIOR ** 2,
            "count": 0,
        })
        a = self.FLY_EWMA_ALPHA
        old_mean = float(fly_mem["mean"])
        old_var  = float(fly_mem["var"])
        new_mean = (1.0 - a) * old_mean + a * fv
        diff = fv - old_mean
        new_var = (1.0 - a) * old_var + a * diff * diff
        fly_mem["mean"] = new_mean
        fly_mem["var"]  = max(new_var, 1e-6)
        fly_mem["count"] = int(fly_mem.get("count", 0)) + 1

        # Blend live EWMA with historical prior
        w = self.FLY_PRIOR_W
        blended_mean = w * self.FLY_5300_5400_5500_MEAN_PRIOR + (1.0 - w) * new_mean
        blended_std  = max(
            w * self.FLY_5300_5400_5500_STD_PRIOR,
            math.sqrt(max(1e-6, new_var)),
        )

        z = (fv - blended_mean) / max(blended_std, 1e-6)
        fly_pos = int(memory.get("fly_pos", 0))

        if z > self.FLY_ENTRY_Z:
            target_fly_pos = -max_qty   # fly expensive: sell wings, buy body
        elif z < -self.FLY_ENTRY_Z:
            target_fly_pos = max_qty    # fly cheap: buy wings, sell body
        elif abs(z) < self.FLY_EXIT_Z and fly_pos != 0:
            target_fly_pos = 0          # near mean: exit
        else:
            target_fly_pos = fly_pos    # hold

        diff_qty = target_fly_pos - fly_pos
        size = max(-self.FLY_LEG_SIZE, min(self.FLY_LEG_SIZE, diff_qty))

        if size != 0:
            leg = abs(size)
            if size > 0:
                # Buy wings (5300 + 5500 cheap): buy 5300, sell 2x5400, buy 5500
                self._take_book_levels(
                    "VEV_5300", books["VEV_5300"], "buy", leg,
                    books["VEV_5300"]["best_ask"], 1,
                    position, result, used, 300, 300,
                )
                self._take_book_levels(
                    "VEV_5400", books["VEV_5400"], "sell", 2 * leg,
                    books["VEV_5400"]["best_bid"], 1,
                    position, result, used, 300, 300,
                )
                self._take_book_levels(
                    "VEV_5500", books["VEV_5500"], "buy", leg,
                    books["VEV_5500"]["best_ask"], 1,
                    position, result, used, 300, 300,
                )
            else:
                # Sell wings (5300 + 5500 expensive): sell 5300, buy 2x5400, sell 5500
                self._take_book_levels(
                    "VEV_5300", books["VEV_5300"], "sell", leg,
                    books["VEV_5300"]["best_bid"], 1,
                    position, result, used, 300, 300,
                )
                self._take_book_levels(
                    "VEV_5400", books["VEV_5400"], "buy", 2 * leg,
                    books["VEV_5400"]["best_ask"], 1,
                    position, result, used, 300, 300,
                )
                self._take_book_levels(
                    "VEV_5500", books["VEV_5500"], "sell", leg,
                    books["VEV_5500"]["best_bid"], 1,
                    position, result, used, 300, 300,
                )
            memory["fly_pos"] = fly_pos + size

    # ── execution helpers ─────────────────────────────────────────────────────

    def _take_book_levels(
        self,
        product: str,
        book: Dict[str, Any],
        side: str,
        qty: int,
        limit_price,
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


if __name__ == "__main__":
    import sys

    class _Depth:
        def __init__(self, bids, asks):
            self.buy_orders = bids
            self.sell_orders = {p: -v for p, v in asks.items()}

    class _State:
        def __init__(self, ts, td, pos, depths):
            self.timestamp = ts
            self.traderData = td
            self.position = dict(pos)
            self.order_depths = depths

    def _make_depths():
        return {
            "HYDROGEL_PACK":       _Depth({9979: 50, 9975: 30}, {9981: 50, 9985: 30}),
            "VELVETFRUIT_EXTRACT": _Depth({5259: 50, 5255: 30}, {5261: 50, 5265: 30}),
            "VEV_4000":            _Depth({1261: 50},           {1263: 50}),
            "VEV_4500":            _Depth({765: 50},            {767: 50}),
            "VEV_5000":            _Depth({267: 50},            {270: 50}),
            "VEV_5100":            _Depth({178: 50},            {181: 50}),
            "VEV_5200":            _Depth({92: 50},             {95: 50}),
            "VEV_5300":            _Depth({46: 50, 45: 30},     {48: 50, 49: 30}),
            "VEV_5400":            _Depth({15: 50, 14: 30},     {17: 50, 18: 30}),
            "VEV_5500":            _Depth({6: 50, 5: 30},       {8: 50, 9: 30}),
        }

    try:
        trader = Trader()
        td = ""
        pos: Dict[str, int] = {}

        for i in range(5):
            state = _State(i * 10000, td, pos, _make_depths())
            orders, conv, td = trader.run(state)

        mem = json.loads(td)
        assert "products" in mem, "missing: products"
        assert "fly_5300_5400_5500" in mem, "missing: fly_5300_5400_5500"
        assert "hp_ema" in mem, "missing: hp_ema"
        assert "swing_ema" in mem, "missing: swing_ema"
        assert json.loads(json.dumps(mem)) == mem, "traderData json round-trip failed"

        print("TRADER OK")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"TRADER FAILED: {exc}")
        sys.exit(1)
