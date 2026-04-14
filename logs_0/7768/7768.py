from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
import json

LIMIT = 80
PRODUCT = "TOMATOES"


class Trader:
    """
    TOMATOES market-making strategy — Prosperity 4 Round 0
    
    Core signals:
        1. L2-anchored microprice  — L2 mid is the stable anchor (~20 vol/side),
           L1 book imbalance tilts it (corr ≈ +0.34 with next-tick return).
        2. Bimodal spread regime   — wide (13-14, ~93%) vs tight (5-9, ~7%).
           In wide regime we quote 1 tick inside BBO for aggressive capture.
           In tight regime we take mispriced liquidity (it reverts in 1-2 ticks).
        3. Inventory-tiered quoting — four tiers from neutral MM through to
           aggressive position offloading near the ±80 limit.
    """

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _extract_book(order_depth: OrderDepth):
        """
        Return sorted bid/ask lists as (price, abs_volume) tuples.
        Bids: highest-first.  Asks: lowest-first.
        sell_orders carry negative volumes in the Prosperity datamodel,
        so we flip the sign here once for the rest of the algo.
        """
        bids = sorted(
            [(p, v) for p, v in order_depth.buy_orders.items()],
            key=lambda x: -x[0],
        )
        asks = sorted(
            [(p, abs(v)) for p, v in order_depth.sell_orders.items()],
            key=lambda x: x[0],
        )
        return bids, asks

    @staticmethod
    def _compute_fair_value(bids, asks) -> float:
        """
        Microprice anchored on L2 mid, tilted by L1 volume imbalance.

            L2_mid      = (bid_price_2 + ask_price_2) / 2
            L1_imb      = (bid_vol_1 − ask_vol_1) / (bid_vol_1 + ask_vol_1)
            microprice  = L2_mid  +  L1_imb × (L1_spread / 2)

        Falls back to L1 mid when L2 is absent.
        """
        bid1_p, bid1_v = bids[0]
        ask1_p, ask1_v = asks[0]
        spread = ask1_p - bid1_p

        # L1 imbalance (positive → bid-heavy → price likely ticks up)
        total_l1 = bid1_v + ask1_v
        l1_imb = (bid1_v - ask1_v) / total_l1 if total_l1 > 0 else 0.0

        # L2 mid — use L2 if present, else fall back to L1 mid
        if len(bids) >= 2 and len(asks) >= 2:
            bid2_p, _ = bids[1]
            ask2_p, _ = asks[1]
            l2_mid = (bid2_p + ask2_p) / 2.0
        else:
            l2_mid = (bid1_p + ask1_p) / 2.0

        microprice = l2_mid + l1_imb * (spread / 2.0)
        return microprice

    @staticmethod
    def _inventory_params(position: int) -> Tuple[float, int, int, int, int]:
        """
        Given current position, return:
            skew            – additive FV adjustment (ticks, signed)
            buy_quote_size  – max lots to post on the bid side
            sell_quote_size – max lots to post on the ask side
            buy_take_edge   – extra edge (in ticks) required before sweeping asks
            sell_take_edge  – extra edge required before sweeping bids

        Four tiers:
            |pos| ≤  40  →  normal MM, no skew
            |pos| ≤  60  →  moderate skew, reduce accumulating side
            |pos| ≤  70  →  offloading at/near FV
            |pos| >  70  →  aggressive offloading, cross spread if needed
        """
        ap = abs(position)
        sign = 1 if position > 0 else -1  # +1 when long

        if ap <= 40:
            return (0.0,  20, 20,  0, 0)

        elif ap <= 60:
            # linear ramp: 0→3 tick skew, shrink accumulating side
            frac = (ap - 40) / 20.0
            skew = -sign * frac * 3.0
            if position > 0:
                return (skew, max(3, int(20 * (1 - frac))), 25, 0, 0)
            else:
                return (skew, 25, max(3, int(20 * (1 - frac))), 0, 0)

        elif ap <= 70:
            # offloading: 3→6 tick skew, mostly one-sided
            frac = (ap - 60) / 10.0
            skew = -sign * (3.0 + frac * 3.0)
            if position > 0:
                return (skew, 0, 30, 2, 0)
            else:
                return (skew, 30, 0, 0, 2)

        else:
            # emergency: 6→10 tick skew, cross spread to flatten
            frac = min((ap - 70) / 10.0, 1.0)
            skew = -sign * (6.0 + frac * 4.0)
            # Keep MM quote small (passive backstop); offload phase does the heavy lifting
            if position > 0:
                return (skew, 0, 20, 4, 0)
            else:
                return (skew, 20, 0, 0, 4)

    # ------------------------------------------------------------------ run
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # ---- load persisted state ----
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except (json.JSONDecodeError, TypeError):
                trader_data = {}

        # ---- skip if TOMATOES not in this tick's products ----
        if PRODUCT not in state.order_depths:
            return result, conversions, json.dumps(trader_data)

        order_depth = state.order_depths[PRODUCT]
        bids, asks = self._extract_book(order_depth)

        if not bids or not asks:
            result[PRODUCT] = []
            return result, conversions, json.dumps(trader_data)

        # ---- book snapshot ----
        best_bid, best_bid_vol = bids[0]
        best_ask, best_ask_vol = asks[0]
        spread = best_ask - best_bid

        position = state.position.get(PRODUCT, 0)
        buy_capacity  = LIMIT - position       # room to go longer
        sell_capacity = LIMIT + position       # room to go shorter

        # ---- fair value ----
        microprice = self._compute_fair_value(bids, asks)

        # EMA for slow drift tracking (used as secondary sanity check)
        ema = trader_data.get("ema", microprice)
        ema = 0.04 * microprice + 0.96 * ema
        trader_data["ema"] = ema

        # ---- inventory-adjusted fair value ----
        skew, mm_buy_sz, mm_sell_sz, buy_take_edge, sell_take_edge = \
            self._inventory_params(position)
        fair = microprice + skew

        # ---- detect spread regime ----
        wide_regime = spread >= 10

        orders: List[Order] = []

        # ==============================================================
        # PHASE 1 — TAKE  (sweep mispriced resting orders)
        # ==============================================================
        # Buy any ask priced below fair - buy_take_edge
        # Sell into any bid priced above fair + sell_take_edge
        # In tight regime (spread ≤ 9, transient) we tighten the edge
        #   requirement by 1 tick to capture fleeting opportunities.
        # ----------------------------------------------------------

        tight_bonus = 1 if not wide_regime else 0   # more willing in tight regime

        for ask_p, ask_v in asks:
            if buy_capacity <= 0:
                break
            threshold = fair - buy_take_edge + tight_bonus
            if ask_p <= threshold:
                qty = min(ask_v, buy_capacity)
                orders.append(Order(PRODUCT, ask_p, qty))  # positive = buy
                buy_capacity -= qty

        for bid_p, bid_v in bids:
            if sell_capacity <= 0:
                break
            threshold = fair + sell_take_edge - tight_bonus
            if bid_p >= threshold:
                qty = min(bid_v, sell_capacity)
                orders.append(Order(PRODUCT, bid_p, -qty))  # negative = sell
                sell_capacity -= qty

        # ==============================================================
        # PHASE 2 — POST  (passive market-making quotes)
        # ==============================================================
        # Wide regime  → 1 tick inside BBO  (capture ~12 of 14 tick spread)
        # Tight regime → at BBO             (don't improve already-tight book)
        #
        # Prices are clamped so we never buy above or sell below our
        # adjusted fair value, ensuring every fill is +EV vs our estimate.
        # ----------------------------------------------------------

        if wide_regime:
            raw_buy_price  = best_bid + 1
            raw_sell_price = best_ask - 1
        else:
            raw_buy_price  = best_bid
            raw_sell_price = best_ask

        # Clamp: never post a buy above fair-1, never post a sell below fair+1
        post_buy_price  = min(raw_buy_price,  int(fair - 1))
        post_sell_price = max(raw_sell_price,  int(fair + 1))

        # Ensure prices are integers (Prosperity requires int prices)
        post_buy_price  = int(post_buy_price)
        post_sell_price = int(post_sell_price)

        # Post buy quote
        buy_qty = min(mm_buy_sz, buy_capacity)
        if buy_qty > 0:
            orders.append(Order(PRODUCT, post_buy_price, buy_qty))
            buy_capacity -= buy_qty

        # Post sell quote
        sell_qty = min(mm_sell_sz, sell_capacity)
        if sell_qty > 0:
            orders.append(Order(PRODUCT, post_sell_price, -sell_qty))
            sell_capacity -= sell_qty

        # ==============================================================
        # PHASE 3 — EMERGENCY OFFLOAD  (only when |pos| > 60)
        # ==============================================================
        # Post a second, more aggressive offloading order closer to
        # (or even through) the fair value to actively flatten inventory.
        # This sits alongside the Phase 2 quotes and provides extra
        # fill probability on the reducing side.
        # ----------------------------------------------------------

        abs_pos = abs(position)

        # Max ticks we allow offload to cross through fair value.
        # Scales with urgency: 2 ticks in offload tier, up to 5 in emergency.
        if abs_pos > 70:
            max_cross = min(5, int(2 + (abs_pos - 70) * 0.3))
        else:
            max_cross = 2

        if abs_pos > 60 and position > 0 and sell_capacity > 0:
            # Long → need to sell.
            # Offload tier: post at fair (willing to sell at FV to flatten).
            # Emergency tier: cross toward best_bid, but floor at fair - max_cross.
            offload_price = int(round(fair))
            if abs_pos > 70:
                offload_price = min(offload_price, best_bid)
            # Guardrail: never sell more than max_cross ticks below fair
            offload_price = max(offload_price, int(fair) - max_cross)
            offload_qty = min(abs_pos - 40, sell_capacity)
            if offload_qty > 0 and offload_price != post_sell_price:
                orders.append(Order(PRODUCT, offload_price, -offload_qty))

        elif abs_pos > 60 and position < 0 and buy_capacity > 0:
            # Short → need to buy.
            offload_price = int(round(fair))
            if abs_pos > 70:
                offload_price = max(offload_price, best_ask)
            # Guardrail: never buy more than max_cross ticks above fair
            offload_price = min(offload_price, int(fair) + max_cross)
            offload_qty = min(abs_pos - 40, buy_capacity)
            if offload_qty > 0 and offload_price != post_buy_price:
                orders.append(Order(PRODUCT, offload_price, offload_qty))

        # ---- done ----
        result[PRODUCT] = orders
        return result, conversions, json.dumps(trader_data)