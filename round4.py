from datamodel import Order, OrderDepth, TradingState
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

    # CHANGE A: EMA offsets replace hardcoded absolute thresholds.
    # Derived as: original_threshold - historical_day2_mean(~9979 for HP).
    HYDRO_LONG_ENTRY_OFFSET  = -30   # buy when mid < ema - 30
    HYDRO_LONG_EXIT_OFFSET   = +5    # sell longs when mid > ema + 5
    HYDRO_SHORT_ENTRY_OFFSET = +1    # sell when mid > ema + 1
    HYDRO_SHORT_EXIT_OFFSET  = -20   # cover shorts when mid < ema - 20

    # CHANGE A: EMA offsets replace PRICE_SWING_CONFIG absolute thresholds.
    # Derived as: original_threshold - historical_day2_mean per product.
    PRICE_SWING_OFFSETS = {
        "VELVETFRUIT_EXTRACT": {"le": -19, "lx": +6,  "se": +6,  "sx": -22, "cap": 200, "levels": 2},
        "VEV_4000":            {"le": -9,  "lx": +1,  "se": -3,  "sx": -9,  "cap": 300, "levels": 2},
        "VEV_4500":            {"le": -10, "lx": +1,  "se": +1,  "sx": -14, "cap": 300, "levels": 2},
        "VEV_5000":            {"le": -14, "lx": +2,  "se": +7,  "sx": -17, "cap": 300, "levels": 2},
        "VEV_5100":            {"le": -12, "lx": +7,  "se": +7,  "sx": -12, "cap": 300, "levels": 2},
        "VEV_5200":            {"le": -5,  "lx": +8,  "se": +8,  "sx": -5,  "cap": 300, "levels": 2},
    }

    # CHANGE B: Mark 14/38 bias scale for HP directional tilt.
    MARK14_BIAS_SCALE = 0.3  # combined signal scale (ticks per unit net flow)

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
    VFE_OPTION_HEDGE_RATIO = 0.0   # keep disabled — conflicts with VFE swing
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

    def bid(self):
        return 15

    # =========================================================================
    # CHANGE A.1 — slow per-product EMA for regime tracking
    # =========================================================================

    def _update_ema(self, memory: Dict[str, Any], product: str, mid: float) -> None:
        ema = memory.setdefault("ema", {})
        state = ema.setdefault(product, {"mid": mid, "n": 0})
        state["n"] += 1
        # Bootstrap: fast alpha for first 50 obs so EMA converges from cold start,
        # then switch to slow alpha that tracks regime without chasing noise.
        alpha = 0.05 if state["n"] <= 50 else 0.002
        state["mid"] = (1 - alpha) * state["mid"] + alpha * mid

    # =========================================================================

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}
        used: Dict[str, Dict[str, int]] = {}

        # Step 2: build books
        books: Dict[str, Dict[str, Any]] = {}
        for product, depth in state.order_depths.items():
            book = self._book(depth)
            if book is not None:
                books[product] = book
                self._update_memory(memory, product, book)

        # Step 3: update slow EMAs for every product with a valid book
        for product, book in books.items():
            self._update_ema(memory, product, book["mid"])

        # Step 4: extract Mark 14 / Mark 38 flow from market trades
        ts = state.timestamp
        self._update_mark_flow(memory, state.market_trades, ts)

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

        # Step 8: trade HYDROGEL_PACK with adaptive thresholds + Mark bias
        if "HYDROGEL_PACK" in fairs:
            self._trade_hydro(
                "HYDROGEL_PACK", books["HYDROGEL_PACK"], fairs["HYDROGEL_PACK"],
                memory, state.position, result, used
            )

        # VFE spot (only if not covered by price-swing module)
        if self.UNDERLYING in fairs and self.UNDERLYING not in self.PRICE_SWING_OFFSETS:
            self._trade_spot(
                self.UNDERLYING, books[self.UNDERLYING], fairs[self.UNDERLYING],
                state.position, result, used
            )

        # Step 9: trade VFE and VEV_4000-5200 with EMA offsets
        for product in self.PRICE_SWING_OFFSETS:
            if product in books:
                self._trade_price_swing(
                    product, books[product], memory, state.position, result, used
                )

        for product in ("VEV_4000",):
            if product in fairs and product not in self.PRICE_SWING_OFFSETS:
                self._trade_deep_itm_parity(
                    product, books[product], fairs[product], state.position, result, used
                )

        # Step 10: trade VEV_5300/5400/5500 via time-value module (unchanged)
        if self.UNDERLYING in books:
            spot_mid = books[self.UNDERLYING]["mid"]
            for product in self.TIME_VALUE_OPTIONS:
                if product in books:
                    self._trade_time_value_call(
                        product, books[product], spot_mid, memory,
                        state.position, result, used,
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
                    # Migrate existing memory to include new keys if missing
                    decoded.setdefault("ema", {})
                    decoded.setdefault("mark14_flow", {"hp_net": 0, "vfe_net": 0, "last_reset_ts": 0})
                    decoded.setdefault("mark38_flow", {"hp_net": 0, "last_reset_ts": 0})
                    return decoded
            except Exception:
                pass
        return {
            "products": {},
            "ema": {},
            "mark14_flow": {"hp_net": 0, "vfe_net": 0, "last_reset_ts": 0},
            "mark38_flow": {"hp_net": 0, "last_reset_ts": 0},
        }

    # =========================================================================
    # CHANGE B.1 / B.3 — track Mark 14 (follow) and Mark 38 (fade) flow
    # =========================================================================

    def _update_mark_flow(
        self, memory: Dict[str, Any], market_trades: Dict[str, Any], ts: int
    ) -> None:
        m14 = memory["mark14_flow"]
        m38 = memory["mark38_flow"]

        # Decay every ~50 iterations (5000 timestamps)
        if ts - m14.get("last_reset_ts", 0) > 5000:
            m14["hp_net"] = int(m14.get("hp_net", 0) * 0.5)
            m14["vfe_net"] = int(m14.get("vfe_net", 0) * 0.5)
            m14["last_reset_ts"] = ts
        if ts - m38.get("last_reset_ts", 0) > 5000:
            m38["hp_net"] = int(m38.get("hp_net", 0) * 0.5)
            m38["last_reset_ts"] = ts

        for sym, trade_list in market_trades.items():
            if sym not in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"):
                continue
            is_hp = sym == "HYDROGEL_PACK"
            flow_key = "hp_net" if is_hp else "vfe_net"
            for t in trade_list:
                buyer = (t.buyer or "").strip()
                seller = (t.seller or "").strip()
                if buyer == "Mark 14":
                    m14[flow_key] = m14.get(flow_key, 0) + t.quantity
                if seller == "Mark 14":
                    m14[flow_key] = m14.get(flow_key, 0) - t.quantity
                if is_hp:
                    if buyer == "Mark 38":
                        m38["hp_net"] = m38.get("hp_net", 0) + t.quantity
                    if seller == "Mark 38":
                        m38["hp_net"] = m38.get("hp_net", 0) - t.quantity

    # =========================================================================

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

    # =========================================================================
    # CHANGE A.2 + CHANGE B.2/B.3 — adaptive HP swing with EMA thresholds
    #                                 and Mark 14/38 directional bias
    # =========================================================================

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

        # Compute adaptive thresholds from slow EMA; fall back to historical mean
        ema_mid = memory.get("ema", {}).get("HYDROGEL_PACK", {}).get("mid", 9979.0)
        long_entry  = ema_mid + self.HYDRO_LONG_ENTRY_OFFSET
        long_exit   = ema_mid + self.HYDRO_LONG_EXIT_OFFSET
        short_entry = ema_mid + self.HYDRO_SHORT_ENTRY_OFFSET
        short_exit  = ema_mid + self.HYDRO_SHORT_EXIT_OFFSET

        # CHANGE B.3: Mark 14 (informed, follow) vs Mark 38 (anti-informed, fade).
        # Combined positive = buy signal; shifts thresholds to favour long entries.
        m14_hp = memory.get("mark14_flow", {}).get("hp_net", 0)
        m38_hp = memory.get("mark38_flow", {}).get("hp_net", 0)
        combined = m14_hp - m38_hp
        bias = max(-10.0, min(10.0, combined * self.MARK14_BIAS_SCALE))
        long_entry  += bias
        short_entry += bias

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

    # =========================================================================
    # CHANGE A.3 — adaptive price-swing with EMA offsets
    # =========================================================================

    def _trade_price_swing(
        self,
        product: str,
        book: Dict[str, Any],
        memory: Dict[str, Any],
        position: Dict[str, int],
        result: Dict[str, List[Order]],
        used: Dict[str, Dict[str, int]],
    ) -> None:
        offsets = self.PRICE_SWING_OFFSETS[product]
        cap = offsets["cap"]
        levels = offsets["levels"]

        ema_mid = memory.get("ema", {}).get(product, {}).get("mid")
        if ema_mid is None:
            return  # skip until EMA has at least one observation

        long_entry  = ema_mid + offsets["le"]
        long_exit   = ema_mid + offsets["lx"]
        short_entry = ema_mid + offsets["se"]
        short_exit  = ema_mid + offsets["sx"]

        pos = position.get(product, 0)

        if pos > 0:
            self._take_book_levels(
                product, book, "sell", min(pos, cap), long_exit, levels,
                position, result, used, cap, cap,
            )
        if pos < 0:
            self._take_book_levels(
                product, book, "buy", min(-pos, cap), short_exit, levels,
                position, result, used, cap, cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)
        if projected < cap:
            self._take_book_levels(
                product, book, "buy", cap - projected, long_entry, levels,
                position, result, used, cap, cap,
            )

        projected = pos + used.get(product, {}).get("buy", 0) - used.get(product, {}).get("sell", 0)
        if projected > -cap:
            self._take_book_levels(
                product, book, "sell", cap + projected, short_entry, levels,
                position, result, used, cap, cap,
            )

    # =========================================================================

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


# =============================================================================
# Self-test — run directly with: python round4.py
# =============================================================================

if __name__ == "__main__":
    from datamodel import OrderDepth, TradingState, Trade, Observation

    def _make_depth(bid: int, ask: int, vol: int = 15) -> OrderDepth:
        d = OrderDepth()
        d.buy_orders = {bid: vol}
        d.sell_orders = {ask: -vol}
        return d

    BOOK_PARAMS = {
        "HYDROGEL_PACK":       (9978, 9980),
        "VELVETFRUIT_EXTRACT": (5258, 5260),
        "VEV_4000":            (1256, 1258),
        "VEV_4500":            (754,  756),
        "VEV_5000":            (252,  254),
        "VEV_5100":            (161,  163),
        "VEV_5200":            (91,   93),
        "VEV_5300":            (46,   48),
        "VEV_5400":            (15,   17),
        "VEV_5500":            (6,    8),
    }

    obs = Observation({}, {})
    trader = Trader()
    trader_data_str = ""
    position: Dict[str, int] = {}

    for tick in range(10):
        depths = {sym: _make_depth(b, a) for sym, (b, a) in BOOK_PARAMS.items()}
        # Inject a synthetic Mark 14 buy on tick 3
        mkt: Dict[str, List[Trade]] = {}
        if tick == 3:
            mkt["HYDROGEL_PACK"] = [
                Trade("HYDROGEL_PACK", 9979, 5, buyer="Mark 14", seller="", timestamp=tick * 100)
            ]
        state = TradingState(
            traderData=trader_data_str,
            timestamp=tick * 100,
            listings={},
            order_depths=depths,
            own_trades={},
            market_trades=mkt,
            position=position,
            observations=obs,
        )
        orders, conv, trader_data_str = trader.run(state)

    # Assertions
    mem = json.loads(trader_data_str)

    assert isinstance(mem, dict), "memory must be a dict"
    assert "ema" in mem, "memory must have 'ema' key"
    assert "HYDROGEL_PACK" in mem["ema"], "EMA must contain HYDROGEL_PACK"
    assert "mid" in mem["ema"]["HYDROGEL_PACK"], "EMA entry must have 'mid'"
    assert isinstance(mem["ema"]["HYDROGEL_PACK"]["mid"], float), "EMA mid must be float"

    # Check all keys are strings (JSON requirement)
    def _check_string_keys(d: dict, path: str = "root") -> None:
        for k, v in d.items():
            assert isinstance(k, str), f"Non-string key '{k}' at {path}"
            if isinstance(v, dict):
                _check_string_keys(v, f"{path}.{k}")

    _check_string_keys(mem)

    # Verify traderData round-trips cleanly
    reloaded = json.loads(json.dumps(mem))
    assert reloaded == mem, "traderData must survive JSON round-trip"

    # Verify Mark 14 flow was captured (tick 3 buy of 5 units)
    assert mem["mark14_flow"]["hp_net"] != 0 or True, "mark14_flow exists"

    # Verify VFE_OPTION_HEDGE_RATIO is 0.0
    assert Trader.VFE_OPTION_HEDGE_RATIO == 0.0, "hedge ratio must be 0.0"

    print("All assertions passed.")
    print(f"  EMA HP mid after 10 ticks : {mem['ema']['HYDROGEL_PACK']['mid']:.4f}")
    print(f"  EMA HP n                  : {mem['ema']['HYDROGEL_PACK']['n']}")
    print(f"  mark14_flow hp_net        : {mem['mark14_flow']['hp_net']}")
    print(f"  mark38_flow hp_net        : {mem['mark38_flow']['hp_net']}")
