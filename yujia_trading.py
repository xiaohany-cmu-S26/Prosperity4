'''from datamodel import OrderDepth, TradingState, Order
from typing import List, Optional
import math

EMERALD_FAIR_VALUE = 10000
POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80
}
DEFAULT_LIMIT = 80


class Trader:
    # Tight-regime L2 skew params
    TIGHT_SPREAD_THRESHOLD = 9
    TIGHT_L2_LAMBDA = 0.35
    TIGHT_L2_CAP = 3

    def micro_dev_l2(self, order_depth: OrderDepth) -> Optional[float]:
        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        asks = sorted(order_depth.sell_orders.keys())
        if len(bids) < 2 or len(asks) < 2:
            return None

        bid_price_2 = bids[1]
        ask_price_2 = asks[1]
        bid_volume_2 = abs(order_depth.buy_orders[bid_price_2])
        ask_volume_2 = abs(order_depth.sell_orders[ask_price_2])

        total_vol = bid_volume_2 + ask_volume_2
        if total_vol == 0:
            return None

        return (bid_price_2 * ask_volume_2 + ask_price_2 * bid_volume_2) / total_vol

    def get_spread(self, order_depth: OrderDepth) -> Optional[int]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys()) - max(order_depth.buy_orders.keys())

    def get_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

    def tight_l2_skew_value(self, order_depth: OrderDepth) -> Optional[float]:
        """
        Tight-regime alpha:
        FV_adj = mid + lambda * (mp_l2 - mid), capped to +/- TIGHT_L2_CAP ticks.

        This is NOT used as a full directional fair value.
        It is only used to bias passive quotes / unwind side.
        """
        mid = self.get_mid(order_depth)
        if mid is None:
            return None

        mp_l2 = self.micro_dev_l2(order_depth)
        if mp_l2 is None:
            return mid

        raw_dev = mp_l2 - mid
        adj_dev = self.TIGHT_L2_LAMBDA * raw_dev

        if adj_dev > self.TIGHT_L2_CAP:
            adj_dev = self.TIGHT_L2_CAP
        elif adj_dev < -self.TIGHT_L2_CAP:
            adj_dev = -self.TIGHT_L2_CAP

        return mid + adj_dev

    def get_fair_value(self, product: str, order_depth: OrderDepth, position: int) -> Optional[float]:
        if product == "EMERALDS":
            fv = EMERALD_FAIR_VALUE
        else:
            mid = self.get_mid(order_depth)
            if mid is None:
                return None

            spread = self.get_spread(order_depth)
            if spread is None:
                return None

            # Wide spread: use stronger L2 microprice FV
            if spread >= 10:
                mp_l2 = self.micro_dev_l2(order_depth)
                fv = mp_l2 if mp_l2 is not None else mid
            else:
                # Tight spread: do NOT fully replace mid with microprice
                # Use only a shrunk / capped version
                fv = self.tight_l2_skew_value(order_depth)

        if fv is None:
            return None

        # Inventory skew on FV for EMERALDS only.
        # TOMATOES inventory handling is done in quote logic.
        if product == "EMERALDS":
            if abs(position) < 40:
                skew = 0.0
            elif abs(position) < 60:
                skew = 0.05 * position
            else:
                skew = 0.2 * position
            fv -= skew

        return fv

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, DEFAULT_LIMIT)

            fv = self.get_fair_value(product, order_depth, position)
            mid = self.get_mid(order_depth)

            if fv is None or mid is None:
                result[product] = orders
                continue

            buy_capacity = limit - position
            sell_capacity = limit + position

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            is_wide = spread >= 10
            is_tight = spread <= self.TIGHT_SPREAD_THRESHOLD

            # ── Step 1: Taking (Aggressive) ───────────────────────────
            #
            # Wide spread: take freely when price crosses FV.
            # Tight spread: only take to UNWIND inventory.
            #
            # Important:
            # Tight-regime L2 alpha is NOT used for directional taking.
            # Use mid as the anchor for unwind-taking, not the skewed FV.

            take_anchor = fv if is_wide or product == "EMERALDS" else mid

            if product == "EMERALDS" or is_wide:
                # Full taking in both directions
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price > take_anchor or buy_capacity <= 0:
                        break
                    vol = abs(order_depth.sell_orders[ask_price])
                    qty = min(buy_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty

                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price < take_anchor or sell_capacity <= 0:
                        break
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty

            elif product == "TOMATOES":
                # Tight spread: only take to unwind, anchored to mid
                if position < 0:
                    # Short -> buy to cover
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if ask_price > take_anchor or buy_capacity <= 0:
                            break
                        vol = abs(order_depth.sell_orders[ask_price])
                        qty = min(buy_capacity, vol, abs(position))
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            buy_capacity -= qty

                elif position > 0:
                    # Long -> sell to flatten
                    for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                        if bid_price < take_anchor or sell_capacity <= 0:
                            break
                        vol = order_depth.buy_orders[bid_price]
                        qty = min(sell_capacity, vol, position)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            sell_capacity -= qty

            # ── Step 2: Making (Passive) ──────────────────────────────
            conservative_bid = best_bid + 1
            conservative_ask = best_ask - 1

            if product == "TOMATOES":
                abs_pos = abs(position)

                # Tight-regime quote anchor:
                # use the shrunk / capped L2 skew only to bias quotes
                quote_anchor = fv if is_wide else self.tight_l2_skew_value(order_depth)
                if quote_anchor is None:
                    quote_anchor = mid

                # quote skew relative to mid
                quote_skew = quote_anchor - mid

                if abs_pos <= 39:
                    # Pure MM with small tight-regime skew
                    if is_tight:
                        our_bid = min(conservative_bid, math.floor(quote_anchor))
                        our_ask = max(conservative_ask, math.ceil(quote_anchor))
                    else:
                        our_bid = conservative_bid
                        our_ask = conservative_ask

                elif abs_pos <= 63:
                    # Moderate inventory:
                    # unwinding side uses quote_anchor +/- 3
                    if position > 0:
                        # Long: prioritize selling inventory
                        our_bid = conservative_bid

                        if is_tight:
                            our_ask = max(conservative_ask, math.ceil(quote_anchor) + 2)
                        else:
                            our_ask = int(fv) + 3

                    else:
                        # Short: prioritize buying back inventory
                        if is_tight:
                            our_bid = min(conservative_bid, math.floor(quote_anchor) - 2)
                        else:
                            our_bid = int(fv) - 3

                        our_ask = conservative_ask

                else:
                    # Critical inventory:
                    # unwinding side as close to quote_anchor/FV as possible
                    if position > 0:
                        our_bid = conservative_bid
                        anchor = quote_anchor if is_tight else fv
                        our_ask = math.ceil(anchor) if anchor != int(anchor) else int(anchor) + 1
                    else:
                        anchor = quote_anchor if is_tight else fv
                        our_bid = math.floor(anchor) if anchor != int(anchor) else int(anchor) - 1
                        our_ask = conservative_ask

                # Tight spread cap:
                # never let tight L2 skew force us to cross or over-chase
                if is_tight:
                    our_bid = min(our_bid, conservative_bid)
                    our_ask = max(our_ask, conservative_ask)

                # Floor / ceiling: stay live, but do not quote through book
                our_bid = max(our_bid, best_bid)
                our_ask = min(our_ask, best_ask)

            else:
                # EMERALDS: conservative with FV cap
                fv_bid = int(fv) - 1
                fv_ask = int(fv) + 1
                our_bid = min(conservative_bid, fv_bid)
                our_ask = max(conservative_ask, fv_ask)

            # Safety: don't cross the book
            if our_bid >= best_ask:
                our_bid = best_bid
            if our_ask <= best_bid:
                our_ask = best_ask

            # Place orders if on correct side of the quote anchor / FV
            if buy_capacity > 0 and our_bid < fv:
                orders.append(Order(product, our_bid, buy_capacity))

            if sell_capacity > 0 and our_ask > fv:
                orders.append(Order(product, our_ask, -sell_capacity))

            result[product] = orders

        traderData = "V5_TIGHT_L2_SKEW"
        conversions = 0
        return result, conversions, traderData
'''


from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Optional, Dict, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
EMERALD_FAIR_VALUE = 10000

POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80,
}
DEFAULT_LIMIT = 80

# Midwall: use top N levels by volume to estimate fair value
MIDWALL_TOP_N = 1

# OBI signal: how many levels deep to look
OBI_DEPTH = 3

# ── Skew Config ───────────────────────────────────────────────────────────────
# Max ticks the entire bid/ask quote window can shift due to OBI signal
SIGNAL_SKEW_MAX: Dict[str, float] = {
    "EMERALDS": 1.0,   # stable, small signal skew
    "TOMATOES": 2.0,   # volatile, allow larger signal skew
}

# Inventory skew: how aggressively to lean against accumulated position.
# Applied as: inventory_skew = -INVENTORY_SKEW_FACTOR * position  (in ticks)
# Tomatoes: more aggressive because it's volatile and we want to mean-revert fast
INVENTORY_SKEW_CONFIG: Dict[str, List[Tuple[float, float]]] = {
    # (position_ratio_threshold, skew_factor) — first match wins
    "EMERALDS": [
        (0.30, 0.00),   # <30% full: pure market-make, don't skew at all
        (0.55, 0.04),   # 30-55%: mild lean
        (0.75, 0.12),   # 55-75%: moderate push
        (1.00, 0.30),   # >75%: aggressive dump/cover
    ],
    "TOMATOES": [
        (0.30, 0.01),
        (0.55, 0.05),
        (0.75, 0.15),
        (1.00, 0.30),   # near-limit: very aggressive clearing
    ],
}

# Spread: how wide to quote around fair value when making
# 1 = quote at best_bid+1 / best_ask-1 (tight), higher = wider / more conservative
# Keeping tight to maximise fill rate and PnL
MAKE_SPREAD_TICKS: Dict[str, int] = {
    "EMERALDS": 1,
    "TOMATOES": 1,
}


class Trader:

    # ── Fair Value ─────────────────────────────────────────────────────────────

    def midwall(self, order_depth: OrderDepth, top_n: int = MIDWALL_TOP_N) -> Optional[float]:
        """
        Volume-weighted mid from the heaviest quote levels on each side.
        More robust than plain mid when the book is thin or skewed.
        """
        def wall_price(levels: dict) -> Optional[float]:
            if not levels:
                return None
            top = sorted(levels.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            total_vol = sum(abs(v) for _, v in top)
            return None if total_vol == 0 else sum(p * abs(v) for p, v in top) / total_vol

        bid_wall = wall_price(order_depth.buy_orders)
        ask_wall = wall_price(order_depth.sell_orders)

        if bid_wall is None and ask_wall is None:
            return None
        if bid_wall is None:
            return ask_wall
        if ask_wall is None:
            return bid_wall
        return (bid_wall + ask_wall) / 2

    def get_true_fv(self, product: str, order_depth: OrderDepth) -> Optional[float]:
        """
        Raw fair value — NO inventory skew.
        Used for Taking decisions so we never miss a genuine arbitrage.
        """
        if product == "EMERALDS":
            return float(EMERALD_FAIR_VALUE)
        return self.midwall(order_depth)

    # ── OBI Signal ─────────────────────────────────────────────────────────────

    def obi(self, order_depth: OrderDepth, depth: int = OBI_DEPTH) -> float:
        """
        Order Book Imbalance in [-1, +1].
        Positive  → more bid volume than ask → bullish short-term pressure.
        Negative  → more ask volume → bearish pressure.
        """
        bids = sorted(order_depth.buy_orders.items(), reverse=True)[:depth]
        asks = sorted(order_depth.sell_orders.items())[:depth]
        bid_vol = sum(abs(v) for _, v in bids)
        ask_vol = sum(abs(v) for _, v in asks)
        total = bid_vol + ask_vol
        return 0.0 if total == 0 else (bid_vol - ask_vol) / total

    # ── Skew Calculation ───────────────────────────────────────────────────────

    def inventory_skew_factor(self, product: str, position: int, limit: int) -> float:
        """
        Return the per-tick skew factor for inventory management.
        Larger position → larger factor → lean harder against it.
        """
        config = INVENTORY_SKEW_CONFIG.get(product, INVENTORY_SKEW_CONFIG["TOMATOES"])
        pos_ratio = abs(position) / limit if limit > 0 else 0.0
        for threshold, factor in config:
            if pos_ratio <= threshold:
                return factor
        return config[-1][1]

    def compute_total_skew(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        limit: int,
    ) -> float:
        """
        Total quote skew in ticks:
          total_skew = signal_skew + inventory_skew

        Positive total_skew → shift both bid & ask UP   (bullish: buy more, sell dearer)
        Negative total_skew → shift both bid & ask DOWN (bearish: sell more, buy cheaper)

        The two components often partially cancel, which is healthy:
        e.g. OBI is bullish (+) but you're already long (inventory skew is −)
        → the net skew is tempered, preventing over-accumulation.
        """
        # 1. Direction signal from OBI
        obi_score = self.obi(order_depth)  # [-1, +1]
        max_sig = SIGNAL_SKEW_MAX.get(product, 1.0)
        signal_skew = obi_score * max_sig  # in ticks

        # 2. Inventory lean — note the NEGATIVE sign:
        #    long position → skew down to encourage selling
        #    short position → skew up to encourage buying
        inv_factor = self.inventory_skew_factor(product, position, limit)
        inventory_skew = -inv_factor * position  # in ticks

        return signal_skew + inventory_skew

    # ── Main Loop ──────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, DEFAULT_LIMIT)

            # ── Fair Values ───────────────────────────────────────────────────
            # true_fv: for Taking  (no bias — captures every genuine edge)
            # skew_fv: for Making  (biased to manage inventory & direction)
            true_fv = self.get_true_fv(product, order_depth)
            if true_fv is None:
                result[product] = orders
                continue

            total_skew_ticks = self.compute_total_skew(product, order_depth, position, limit)
            skew_fv = true_fv + total_skew_ticks  # skewed fair value for quoting

            buy_capacity  = limit - position
            sell_capacity = limit + position

            # ── Step 1: Taking (Aggressive) ───────────────────────────────────
            # Use TRUE fair value — never leave money on the table because of
            # inventory. If the price is genuinely cheap/expensive, take it.

            # Buy underpriced asks
            if order_depth.sell_orders:
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price >= true_fv:
                        break
                    if buy_capacity <= 0:
                        break
                    vol = abs(order_depth.sell_orders[ask_price])
                    qty = min(buy_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty

            # Sell overpriced bids
            if order_depth.buy_orders:
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price <= true_fv:
                        break
                    if sell_capacity <= 0:
                        break
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty

            # ── Step 2: Making (Passive, Skewed) ─────────────────────────────
            # Quote bid/ask around the SKEWED fair value.
            # Both sides shift together → spread stays constant, centre moves.

            best_bid = max(order_depth.buy_orders.keys())  if order_depth.buy_orders  else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            spread_ticks = MAKE_SPREAD_TICKS.get(product, 1)

            # ── Quote Bid ─────────────────────────────────────────────────────
            if buy_capacity > 0 and best_bid is not None:
                # Base: improve best bid by spread_ticks
                base_bid = best_bid + spread_ticks

                # Apply skew: positive skew → bid higher (want to buy)
                our_bid = round(base_bid + total_skew_ticks)

                # Clamp: never cross the ask (avoid self-trade / accidental taking)
                if best_ask is not None and our_bid >= best_ask:
                    our_bid = best_ask - 1

                # Gate: only post if strictly below skewed FV
                # (ensures we always buy at a discount to our current view)
                if our_bid < skew_fv:
                    orders.append(Order(product, our_bid, buy_capacity))

            # ── Quote Ask ─────────────────────────────────────────────────────
            if sell_capacity > 0 and best_ask is not None:
                # Base: undercut best ask by spread_ticks
                base_ask = best_ask - spread_ticks

                # Apply skew: positive skew → ask higher (sell dearer when bullish)
                our_ask = round(base_ask + total_skew_ticks)

                # Clamp: never cross the bid
                if best_bid is not None and our_ask <= best_bid:
                    our_ask = best_bid + 1

                # Gate: only post if strictly above skewed FV
                if our_ask > skew_fv:
                    orders.append(Order(product, our_ask, -sell_capacity))

            result[product] = orders

        traderData = "SKEW_V1"
        conversions = 0
        return result, conversions, traderData