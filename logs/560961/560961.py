"""
IMC Prosperity 4 — Round 5 trader.

50 products in 10 categories of 5; hard position limit 10 per product.

Active surface (after empirical narrowing on days 2-4):
  PEBBLES_XL    — basket-implied fair value (basket sum = 50,000, std 2.8).
                  Only XL is the noisy leg; other 4 PEBBLES are stable, so
                  trading them on basket_dev fades random shocks. Backtest
                  contribution: ~+40k over 3 days.
  ROBOT_DISHES  — bid-ask bounce fade (return AR(1) = -0.232, strongest in
                  the round). cap=5/size=5 gives best Sharpe-like ratio.
                  Backtest contribution: ~+126k over 3 days.

Skip universe (everything else): the cap-10 + adverse-selection math kills
passive MM and weak-signal takes on the remaining 48 products. SNACKPACK
(spec's "best MM category") and bounce signals on IRONING/EVE/CHOC all
backtested net-negative. The full strategy infrastructure (snackpack_mm,
vanilla_mm, etc.) remains wired up so individual products can be re-enabled
via PRODUCT_CONFIG without code changes.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState


LIMIT = 10
POS_HARD_CAP = 8
BASKET_TARGET_SUM = 50_000

PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
SNACKPACKS = [
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
]

ALL_PRODUCTS = PEBBLES + SNACKPACKS + [
    f"{cat}_{s}" for cat, suffixes in {
        "GALAXY_SOUNDS": ["DARK_MATTER", "BLACK_HOLES", "PLANETARY_RINGS", "SOLAR_WINDS", "SOLAR_FLAMES"],
        "SLEEP_POD":     ["SUEDE", "LAMB_WOOL", "POLYESTER", "NYLON", "COTTON"],
        "MICROCHIP":     ["CIRCLE", "OVAL", "SQUARE", "RECTANGLE", "TRIANGLE"],
        "ROBOT":         ["VACUUMING", "MOPPING", "DISHES", "LAUNDRY", "IRONING"],
        "UV_VISOR":      ["YELLOW", "AMBER", "ORANGE", "RED", "MAGENTA"],
        "TRANSLATOR":    ["SPACE_GRAY", "ASTRO_BLACK", "ECLIPSE_CHARCOAL", "GRAPHITE_MIST", "VOID_BLUE"],
        "PANEL":         ["1X2", "2X2", "1X4", "2X4", "4X4"],
        "OXYGEN_SHAKE":  ["MORNING_BREATH", "EVENING_BREATH", "MINT", "CHOCOLATE", "GARLIC"],
    }.items() for s in suffixes
]

POSITION_LIMITS: Dict[str, int] = {p: LIMIT for p in ALL_PRODUCTS}

# High-AR(1000) random-walk-ish products: reduce default size and pos cap
HIGH_AR1000 = {
    "GALAXY_SOUNDS_BLACK_HOLES", "SLEEP_POD_POLYESTER",
    "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
    "UV_VISOR_AMBER", "OXYGEN_SHAKE_GARLIC",
}

# strategy: "pebbles_basket" | "snackpack_mm" | "bounce" | "vanilla_mm" | "skip"
#
# v2 lesson from day 2-4 backtest: passive quoting at limit 10 across 50 products
# creates systematic adverse selection (~50-130k loss per product). Only ROBOT_DISHES
# (+148k aggregate) and PEBBLES_XL (+55k) were positive. Strategy is to skip
# anything without a proven edge and keep MM only where the edge clears the cost.

PRODUCT_CONFIG: Dict[str, Dict[str, Any]] = {}

# Tier S — PEBBLES: only XL trades the basket strategy. The other 4 legs are
# stable mids; noise in basket_dev originates from XL, so trading XS/S/M/L on
# basket_dev is fading random shocks → pure adverse selection.
PRODUCT_CONFIG["PEBBLES_XL"] = {"strategy": "pebbles_basket"}

# Tier A — SNACKPACK skipped. Both v1 (full MM) and v2 (take-only) lost money
# consistently across all 5 legs. The wide-spread MM thesis didn't survive
# the cap-10 adverse-selection regime in this codebase. Revisit if a more
# sophisticated fair-value model is added.

# Tier B — bounce fade (only DISHES proven; IRONING/EVE/CHOC bounce signals
# don't clear adverse-selection costs at limit 10)
PRODUCT_CONFIG["ROBOT_DISHES"]  = {"strategy": "bounce", "coef": 0.20, "quote_off": 1, "take_thresh": 3, "size": 5, "pos_cap": 5}

# Everyone else: skip. No quotes, no takes. The cap-10 + adverse-selection math
# kills passive MM on products without strong directional or basket edge.
for p in ALL_PRODUCTS:
    PRODUCT_CONFIG.setdefault(p, {"strategy": "skip"})


# ───────────────────────── Constants for strategies ──────────────────────────

# SNACKPACK MM
SP_FV_ALPHA      = 0.05
SP_QUOTE_OFFSET  = 3
SP_TAKE_THRESH   = 5
SP_SKEW_TIERS    = [(3, 0.5), (5, 1.5), (7, 3.0), (9, 6.0)]
SP_PAIR_ALPHA    = 0.01
SP_PAIR_THRESH   = 30
SP_PAIR_COEF     = 0.30
RASP_FV_FAST_A   = 0.10
RASP_FV_SLOW_A   = 0.005
RASP_BLEND       = 0.50
RASP_QUOTE_OFF   = 2
RASP_TAKE_THRESH = 3

# Tier C / bounce defaults
DEFAULT_FV_ALPHA   = 0.05
DEFAULT_SKEW_TIERS = [(4, 1.0), (7, 3.0), (9, 6.0)]
BOUNCE_SKEW_TIERS  = [(4, 1.0), (7, 3.0), (9, 6.0)]

# PEBBLES
PEB_TAKE_THRESH = 1
PEB_QUOTE_OFF   = 2
PEB_SKEW_PER_UNIT = 1.0


# ───────────────────────── Trader ────────────────────────────────────────────

class Trader:
    LIMITS = POSITION_LIMITS

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}
        used: Dict[str, Dict[str, int]] = {}

        # Phase 1 — parse books and refresh per-product memory
        books: Dict[str, Dict[str, Any]] = {}
        for product, depth in state.order_depths.items():
            book = self._book(depth)
            if book is None:
                continue
            books[product] = book
            self._update_memory(memory, product, book)

        # Phase 2 — group-level memory updates (SNACKPACK pair sum EMA)
        if "SNACKPACK_CHOCOLATE" in books and "SNACKPACK_VANILLA" in books:
            pair_sum = books["SNACKPACK_CHOCOLATE"]["mid"] + books["SNACKPACK_VANILLA"]["mid"]
            prev = memory.get("pair_ema")
            memory["pair_ema"] = pair_sum if prev is None else (
                SP_PAIR_ALPHA * pair_sum + (1.0 - SP_PAIR_ALPHA) * prev
            )

        # Phase 3 — group dispatch (PEBBLES basket fires once per tick)
        if all(p in books for p in PEBBLES):
            self._trade_pebbles(books, state.position, result, used)

        # Phase 4 — per-product dispatch
        for product, book in books.items():
            cfg = PRODUCT_CONFIG.get(product)
            if not cfg or cfg["strategy"] == "pebbles_basket":
                continue
            strat = cfg["strategy"]
            if strat == "skip":
                continue
            if strat == "snackpack_mm":
                self._trade_snackpack(product, book, books, memory, cfg, state.position, result, used)
            elif strat == "bounce":
                self._trade_bounce(product, book, memory, cfg, state.position, result, used)
            elif strat == "vanilla_mm":
                self._trade_vanilla(product, book, memory, cfg, state.position, result, used)

        memory["tick"] = memory.get("tick", 0) + 1
        return result, 0, json.dumps(memory, separators=(",", ":"))

    # ────────────── Strategy: PEBBLES per-leg implied fair value ──────────────

    def _trade_pebbles(self, books, position, result, used):
        mids = [books[p]["mid"] for p in PEBBLES]
        total = sum(mids)
        basket_dev = total - BASKET_TARGET_SUM
        # Safety: if deviation is wildly out of bounds, the books are inconsistent
        if abs(basket_dev) > 200:
            return

        # Trade only legs whose config opts in (currently just XL — see PRODUCT_CONFIG)
        active_legs = [p for p in PEBBLES if PRODUCT_CONFIG.get(p, {}).get("strategy") == "pebbles_basket"]
        for leg in active_legs:
            book = books[leg]
            mid = book["mid"]
            implied_fair = mid - basket_dev
            pos = position.get(leg, 0)
            fair_adj = implied_fair - PEB_SKEW_PER_UNIT * pos

            # Take: lift ask if ≤ fair − 1
            if book["best_ask"] <= fair_adj - PEB_TAKE_THRESH:
                qty = min(book["ask_vol"], POS_HARD_CAP - pos)
                if qty > 0:
                    self._add_order(leg, book["best_ask"], qty, position, result, used,
                                    long_cap_override=POS_HARD_CAP, short_cap_override=POS_HARD_CAP)
            if book["best_bid"] >= fair_adj + PEB_TAKE_THRESH:
                qty = min(book["bid_vol"], POS_HARD_CAP + pos)
                if qty > 0:
                    self._add_order(leg, book["best_bid"], -qty, position, result, used,
                                    long_cap_override=POS_HARD_CAP, short_cap_override=POS_HARD_CAP)

            # Make: passive quotes around implied fair, BUT only when basket
            # deviation provides a real edge — quoting at near-zero dev just
            # feeds adverse selection.
            if abs(basket_dev) >= 2:
                # Bias quotes toward fading the deviation: if basket > 50000
                # (legs are rich), only post asks; if basket < 50000, only bids.
                allow_buy = basket_dev <= 0   # basket cheap → buy legs
                allow_sell = basket_dev >= 0  # basket rich → sell legs
                self._passive_quote(leg, book, fair_adj, edge=PEB_QUOTE_OFF, base_size=3,
                                    position=position, result=result, used=used,
                                    long_cap_override=POS_HARD_CAP, short_cap_override=POS_HARD_CAP,
                                    allow_buy=allow_buy, allow_sell=allow_sell)

    # ────────────── Strategy: SNACKPACK market making ─────────────────────────

    def _trade_snackpack(self, product, book, books, memory, cfg, position, result, used):
        item = memory["products"][product]

        if cfg.get("rasp_override"):
            mid = book["mid"]
            fast = item.get("fv_fast")
            slow = item.get("fv_slow")
            item["fv_fast"] = mid if fast is None else RASP_FV_FAST_A * mid + (1 - RASP_FV_FAST_A) * fast
            item["fv_slow"] = mid if slow is None else RASP_FV_SLOW_A * mid + (1 - RASP_FV_SLOW_A) * slow
            fv = RASP_BLEND * item["fv_fast"] + (1 - RASP_BLEND) * item["fv_slow"]
            quote_off = RASP_QUOTE_OFF
            take_thresh = cfg.get("take_thresh", RASP_TAKE_THRESH)
        else:
            fv = self._fv_ema(item, book["mid"], SP_FV_ALPHA)
            quote_off = SP_QUOTE_OFFSET
            take_thresh = cfg.get("take_thresh", SP_TAKE_THRESH)

        # Pair tilt for CHOC / VAN
        if cfg.get("pair_leg") and memory.get("pair_ema") is not None:
            choc = books.get("SNACKPACK_CHOCOLATE")
            van = books.get("SNACKPACK_VANILLA")
            if choc and van:
                pair_sum = choc["mid"] + van["mid"]
                pair_dev = pair_sum - memory["pair_ema"]
                if abs(pair_dev) > SP_PAIR_THRESH:
                    fv -= SP_PAIR_COEF * pair_dev / 2.0  # bias both legs toward fade

        pos = position.get(product, 0)
        fv -= self._skew_offset(pos, SP_SKEW_TIERS)

        size = cfg.get("size", 6)
        pos_cap = cfg.get("pos_cap", POS_HARD_CAP)

        # Take if mispriced beyond threshold
        self._take_if_clear(product, book, fv, take_thresh, size, position, result, used,
                            pos_cap=pos_cap)

        # Make passively (gated by cfg["do_make"])
        if cfg.get("do_make", True):
            self._passive_quote(product, book, fv, edge=quote_off, base_size=size,
                                position=position, result=result, used=used,
                                long_cap_override=pos_cap, short_cap_override=pos_cap)

    # ────────────── Strategy: bid-ask bounce fade ─────────────────────────────

    def _trade_bounce(self, product, book, memory, cfg, position, result, used):
        item = memory["products"][product]
        mid = book["mid"]
        last = item.get("last_mid")
        last_ret = (mid - last) if last is not None else 0.0
        item["last_mid"] = mid

        fv = self._fv_ema(item, mid, DEFAULT_FV_ALPHA)
        fv -= cfg["coef"] * last_ret

        pos = position.get(product, 0)
        fv -= self._skew_offset(pos, BOUNCE_SKEW_TIERS)

        size = cfg.get("size", 6)
        pos_cap = cfg.get("pos_cap", POS_HARD_CAP)
        spread = book["spread"]
        edge = cfg.get("quote_off", max(2, spread // 4))
        take_thresh = cfg.get("take_thresh", max(3, spread // 3))

        self._take_if_clear(product, book, fv, take_thresh, size, position, result, used,
                            pos_cap=pos_cap)
        if cfg.get("do_make", True):
            self._passive_quote(product, book, fv, edge=edge, base_size=size,
                                position=position, result=result, used=used,
                                long_cap_override=pos_cap, short_cap_override=pos_cap)

    # ────────────── Strategy: vanilla EMA market making ───────────────────────

    def _trade_vanilla(self, product, book, memory, cfg, position, result, used):
        item = memory["products"][product]
        fv = self._fv_ema(item, book["mid"], DEFAULT_FV_ALPHA)
        pos = position.get(product, 0)
        fv -= self._skew_offset(pos, DEFAULT_SKEW_TIERS)

        # Reduce size for high-AR(1000) random-walk-ish products
        default_size = 5 if product in HIGH_AR1000 else 6
        default_cap = 6 if product in HIGH_AR1000 else POS_HARD_CAP

        size = cfg.get("size", default_size)
        pos_cap = cfg.get("pos_cap", default_cap)
        spread = book["spread"]
        edge = max(2, spread // 4)
        take_thresh = max(3, spread // 3)

        self._take_if_clear(product, book, fv, take_thresh, size, position, result, used,
                            pos_cap=pos_cap)
        if cfg.get("do_make", True):
            self._passive_quote(product, book, fv, edge=edge, base_size=size,
                                position=position, result=result, used=used,
                                long_cap_override=pos_cap, short_cap_override=pos_cap)

    # ────────────── Primitives (book parser, EMA, skew, take, quote, add) ─────

    def _book(self, depth: OrderDepth) -> Optional[Dict[str, Any]]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid_levels = sorted(
            [(p, v) for p, v in depth.buy_orders.items() if v > 0], reverse=True
        )
        ask_levels = sorted(
            [(p, -v) for p, v in depth.sell_orders.items() if v < 0]
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
            "best_bid": bid, "best_ask": ask,
            "bid_vol": bid_vol, "ask_vol": ask_vol,
            "mid": mid, "spread": ask - bid, "micro": micro,
            "bid_levels": bid_levels, "ask_levels": ask_levels,
        }

    def _update_memory(self, memory, product, book):
        item = memory.setdefault("products", {}).setdefault(product, {})
        if "fv_ema" not in item:
            item["fv_ema"] = book["mid"]

    def _fv_ema(self, item, mid, alpha):
        prev = item.get("fv_ema")
        if prev is None:
            item["fv_ema"] = mid
        else:
            item["fv_ema"] = alpha * mid + (1.0 - alpha) * prev
        return item["fv_ema"]

    @staticmethod
    def _skew_offset(pos: int, tiers):
        # tiers ascending by |pos| threshold; returns signed tick offset.
        # Positive position → positive offset (subtracted from fair → leans sell).
        offset = 0.0
        magnitude = abs(pos)
        for thresh, ticks in tiers:
            if magnitude >= thresh:
                offset = ticks
        return math.copysign(offset, pos) if pos else 0.0

    def _load_memory(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return {"products": {}, "tick": 0}
        try:
            mem = json.loads(trader_data)
            mem.setdefault("products", {})
            return mem
        except (json.JSONDecodeError, TypeError):
            return {"products": {}, "tick": 0}

    def _take_if_clear(self, product, book, fair, edge, size, position, result, used, pos_cap):
        if book["best_ask"] <= fair - edge:
            qty = min(size, book["ask_vol"])
            if qty > 0:
                self._add_order(product, book["best_ask"], qty, position, result, used,
                                long_cap_override=pos_cap, short_cap_override=pos_cap)
        if book["best_bid"] >= fair + edge:
            qty = min(size, book["bid_vol"])
            if qty > 0:
                self._add_order(product, book["best_bid"], -qty, position, result, used,
                                long_cap_override=pos_cap, short_cap_override=pos_cap)

    def _passive_quote(self, product, book, fair, edge, base_size, position, result, used,
                       long_cap_override=None, short_cap_override=None,
                       allow_buy=True, allow_sell=True):
        pos = position.get(product, 0)
        cap = self.LIMITS.get(product, LIMIT)
        if cap <= 0:
            return

        buy_size = max(1, int(base_size * max(0.25, 1.0 - max(pos, 0) / cap)))
        sell_size = max(1, int(base_size * max(0.25, 1.0 + min(pos, 0) / cap)))

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

    def _add_order(self, product, price, qty, position, result, used,
                   long_cap_override=None, short_cap_override=None):
        if qty == 0 or product not in self.LIMITS:
            return
        cap = self.LIMITS[product]
        long_cap = cap if long_cap_override is None else min(cap, long_cap_override)
        short_cap = cap if short_cap_override is None else min(cap, short_cap_override)
        if cap <= 0 or long_cap <= 0 or short_cap <= 0:
            return

        pos = position.get(product, 0)
        product_used = used.setdefault(product, {"buy": 0, "sell": 0})

        if qty > 0:
            room = min(cap - pos - product_used["buy"],
                       long_cap - pos - product_used["buy"])
            qty = min(qty, max(0, room))
            if qty <= 0:
                return
            product_used["buy"] += qty
        else:
            room = min(cap + pos - product_used["sell"],
                       short_cap + pos - product_used["sell"])
            sell_qty = min(-qty, max(0, room))
            if sell_qty <= 0:
                return
            qty = -sell_qty
            product_used["sell"] += sell_qty

        result.setdefault(product, []).append(Order(product, int(price), int(qty)))