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

    # Internal risk caps. 0 = product disabled.
    # VEV_5300 is the dominant alpha (sim PnL +11.5k).
    # VEV_5400 is secondary (+4.1k); VEV_5500 is small (+1.2k).
    # VEV_5200 is a known loser (-9.5k); 5000/5100 illiquid; 4500/4000 wide spread; 6000/6500 dead.
    RISK_CAPS = {
        "HYDROGEL_PACK": 80,
        "VELVETFRUIT_EXTRACT": 90,
        "VEV_4000": 4,
        "VEV_4500": 0,
        "VEV_5000": 0,
        "VEV_5100": 0,
        "VEV_5200": 0,
        "VEV_5300": 300,
        "VEV_5400": 200,
        "VEV_5500": 100,
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

    # Time-value bidirectional mean reversion.
    #
    # Diagnosis (run 452120, see _diag_452120.py replay):
    #   The previous prior-anchored z=1.75 thresholds NEVER fired on live data
    #   (live VEV_5300 ask_tv min was 42 vs threshold 35.86). VEVs traded 0
    #   units. Live std (~4) was much narrower than the historical prior
    #   (6.23) and the live mean shifted up (~50 vs prior 46.76).
    #
    # Replay-tuned design:
    #   - Live-dominant rolling stats (alpha=0.02, prior weight 0.20).
    #   - Floor std at 0.5 * prior_std so it can't collapse on a calm window.
    #   - Bidirectional: long when ask_tv < op_mean - z*op_std, short when
    #     bid_tv > op_mean + z*op_std. Replay shows shorts contribute ~half
    #     the total PnL because the live path sat above prior mean.
    #   - z = 0.8 entry, exit at op_mean (z=0). Full cap on signal.
    #   - 50-obs warmup so EWMA stabilises before trading.
    #
    # Replay PnL on 452120: VEV_5300 +3143, VEV_5400 +412, VEV_5500 +143
    # (total +3698). With Hydro/VFE the round projects to ~4500.
    TIME_VALUE_OPTIONS: Tuple[str, ...] = ("VEV_5300", "VEV_5400", "VEV_5500")
    TIME_VALUE_MEAN = {
        "VEV_5300": 46.76,
        "VEV_5400": 15.95,
        "VEV_5500": 6.64,
    }
    TIME_VALUE_STD = {
        "VEV_5300": 6.23,
        "VEV_5400": 3.43,
        "VEV_5500": 1.74,
    }
    TIME_VALUE_ENTRY_Z = {
        "VEV_5300": 0.80,
        "VEV_5400": 0.80,
        "VEV_5500": 0.80,
    }
    TIME_VALUE_LONG_CAP = {
        "VEV_5300": 300,
        "VEV_5400": 200,
        "VEV_5500": 100,
    }
    TIME_VALUE_SHORT_CAP = {
        "VEV_5300": 300,
        "VEV_5400": 200,
        "VEV_5500": 100,
    }
    TIME_VALUE_STD_FLOOR_RATIO = 0.5
    TIME_VALUE_MIN_OBS = 20

    # Online time-value EWMA so entry thresholds adapt to the live regime
    # while staying lightly anchored to the historical prior.
    TV_EMA_ALPHA = 0.020
    TV_PRIOR_WEIGHT = 0.20

    def bid(self):
        return 15

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}
        used: Dict[str, Dict[str, int]] = {}

        # Log every market trade with buyer and seller
        for product, trades in state.market_trades.items():
            for t in trades:
                print(
                    f"[MKT ts={state.timestamp}] {product} "
                    f"buyer={t.buyer} seller={t.seller} "
                    f"price={t.price} qty={t.quantity}"
                )

        books: Dict[str, Dict[str, Any]] = {}
        for product, depth in state.order_depths.items():
            book = self._book(depth)
            if book is not None:
                books[product] = book
                self._update_memory(memory, product, book)

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

            # Track an EWMA of every active VEV's time-value so we can adapt.
            spot_mid = books[self.UNDERLYING]["mid"]
            for product in self.TIME_VALUE_OPTIONS:
                if product in books:
                    self._update_time_value_memory(
                        memory, product, books[product]["mid"], spot_mid
                    )

        if "HYDROGEL_PACK" in fairs:
            self._trade_hydro(
                "HYDROGEL_PACK", books["HYDROGEL_PACK"], fairs["HYDROGEL_PACK"],
                memory, state.position, result, used
            )

        if self.UNDERLYING in fairs:
            self._trade_spot(
                self.UNDERLYING, books[self.UNDERLYING], fairs[self.UNDERLYING],
                state.position, result, used
            )

        for product in ("VEV_4000",):
            if product in fairs:
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
        bid = max(depth.buy_orders)
        ask = min(depth.sell_orders)
        bid_vol = depth.buy_orders[bid]
        ask_vol = -depth.sell_orders[ask]
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
        # Std cannot collapse during a calm window — floor it at a fraction of
        # the historical prior. Without this, a quiet live segment would push
        # the entry threshold inside the spread and we'd trade noise.
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
        pos = position.get(product, 0)
        cap = self.RISK_CAPS[product]
        item = memory["products"].get(product, {})
        ema = float(item.get("ema", book["mid"]))
        slow = float(item.get("slow", ema))
        prev = float(item.get("prev", book["mid"]))

        passive_edge = max(4.0, 0.34 * book["spread"])
        take_edge = max(10.0, 0.75 * book["spread"])
        base_size = 7

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
        self._take_if_clear(product, book, fair_adj, edge + 5.0, 1, position, result, used)

        if book["spread"] >= 12:
            self._passive_quote(product, book, fair_adj, edge, 1, position, result, used)

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
        item = memory.get("products", {}).get(product, {})
        tv_count = int(item.get("tv_count", 0))

        long_entry_tv = mean - entry_z * std
        short_entry_tv = mean + entry_z * std

        ask_tv = book["best_ask"] - intrinsic
        bid_tv = book["best_bid"] - intrinsic

        # Combined long+short cap so _add_order accepts orders from this
        # strategy regardless of position direction. Use the larger of the two.
        order_long_cap = max(long_cap, short_cap)
        order_short_cap = max(long_cap, short_cap)

        # Exit longs back to the bid once time value has reverted to mean.
        if pos > 0 and bid_tv >= mean:
            already_selling = used.get(product, {}).get("sell", 0)
            sell_room = max(0, pos - already_selling)
            sell_qty = min(sell_room, book["bid_vol"])
            if sell_qty > 0:
                self._add_order(product, book["best_bid"], -sell_qty,
                                position, result, used,
                                long_cap_override=order_long_cap,
                                short_cap_override=order_short_cap)

        # Exit shorts back through the ask once time value has reverted to mean.
        if pos < 0 and ask_tv <= mean:
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

        # Long underpriced calls using executable ask time value.
        if long_cap > 0 and ask_tv <= long_entry_tv and pos < long_cap:
            buy_room = max(0, long_cap - pos - used.get(product, {}).get("buy", 0))
            buy_qty = min(book["ask_vol"], buy_room)
            if buy_qty > 0:
                self._add_order(product, book["best_ask"], buy_qty,
                                position, result, used,
                                long_cap_override=long_cap,
                                short_cap_override=order_short_cap)

        # Short overpriced calls using executable bid time value.
        if short_cap > 0 and bid_tv >= short_entry_tv and pos > -short_cap:
            sell_room = max(0, short_cap + pos - used.get(product, {}).get("sell", 0))
            sell_qty = min(book["bid_vol"], sell_room)
            if sell_qty > 0:
                self._add_order(product, book["best_bid"], -sell_qty,
                                position, result, used,
                                long_cap_override=order_long_cap,
                                short_cap_override=short_cap)

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
