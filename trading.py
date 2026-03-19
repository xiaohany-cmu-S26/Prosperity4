from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Optional
import math

EMERALD_FAIR_VALUE = 10000
POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80
}
DEFAULT_LIMIT = 80


class Trader:

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

    def get_fair_value(self, product: str, order_depth: OrderDepth, position: int) -> Optional[float]:
        if product == "EMERALDS":
            fv = EMERALD_FAIR_VALUE
        else:
            mid = self.get_mid(order_depth)
            if mid is None:
                return None

            spread = self.get_spread(order_depth)

            # L2 microprice only during wide spread (>= 10).
            # During tight spread it blows up (std 6.15 vs 0.35).
            if spread is not None and spread >= 10:
                mp_l2 = self.micro_dev_l2(order_depth)
                fv = mp_l2 if mp_l2 is not None else mid
            else:
                fv = mid

        if fv is None:
            return None

        # Inventory skew on FV for EMERALDS only.
        # TOMATOES skew is handled via asymmetric quoting in run().
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
            if fv is None:
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

            # ── Step 1: Taking (Aggressive) ───────────────────────────
            #
            # Wide spread: take freely when price crosses FV.
            # Tight spread: take ONLY to unwind inventory.
            #   If we're short, we can lift asks below FV (buy to cover).
            #   If we're long, we can hit bids above FV (sell to flatten).
            #   Never take in the direction that worsens inventory.

            if product == "EMERALDS" or is_wide:
                # Full taking in both directions
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price > fv or buy_capacity <= 0:
                        break
                    vol = abs(order_depth.sell_orders[ask_price])
                    qty = min(buy_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty

                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price < fv or sell_capacity <= 0:
                        break
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty

            elif product == "TOMATOES":
                # Tight spread: only take to unwind
                if position < 0:
                    # Short -> buy to cover
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if ask_price > fv or buy_capacity <= 0:
                            break
                        vol = abs(order_depth.sell_orders[ask_price])
                        qty = min(buy_capacity, vol, abs(position))
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            buy_capacity -= qty

                elif position > 0:
                    # Long -> sell to flatten
                    for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                        if bid_price < fv or sell_capacity <= 0:
                            break
                        vol = order_depth.buy_orders[bid_price]
                        qty = min(sell_capacity, vol, position)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            sell_capacity -= qty

            # ── Step 2: Making (Passive) ──────────────────────────────
            #
            # Default: conservative bid+1 / ask-1 (proven profitable).
            #
            # Asymmetric inventory management for TOMATOES:
            #   The UNWINDING side quotes more aggressively based on
            #   inventory level, while the accumulating side stays at
            #   bid+1 / ask-1.
            #
            #   |pos| < 40:  both sides at bid+1 / ask-1 (pure MM)
            #   |pos| 40-63: unwinding side moves to FV ± 3
            #   |pos| > 63:  unwinding side quotes AT fair value
            #
            # Tight spread cap always applies (min/max with bid+1/ask-1)
            # to prevent microprice blowup from crossing the book.

            conservative_bid = best_bid + 1
            conservative_ask = best_ask - 1

            if product == "TOMATOES":
                abs_pos = abs(position)

                if abs_pos <= 63:
                    # Pure MM: symmetric conservative
                    our_bid = conservative_bid
                    our_ask = conservative_ask

                elif abs_pos <= 39:
                    # Moderate inventory: unwinding side at FV ± 3
                    if position > 0:
                        # Long: ask aggressively, bid conservatively
                        our_bid = conservative_bid
                        our_ask = int(fv) + 3
                    else:
                        # Short: bid aggressively, ask conservatively
                        our_bid = int(fv) - 3
                        our_ask = conservative_ask

                else:
                    # Critical inventory (>63): unwinding side as close
                    # to fair value as possible while staying live.
                    # ceil(fv) for asks guarantees ask > fv when fv is
                    # not an integer; floor(fv) + 1 when it is.
                    if position > 0:
                        our_bid = conservative_bid
                        our_ask = math.ceil(fv) if fv != int(fv) else int(fv) + 1
                    else:
                        our_bid = math.floor(fv) if fv != int(fv) else int(fv) - 1
                        our_ask = conservative_ask

                # Tight spread cap: never more aggressive than bid+1/ask-1
                if not is_wide:
                    our_bid = min(our_bid, conservative_bid)
                    our_ask = max(our_ask, conservative_ask)

                # Floor: never quote behind the bots
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

            # Place orders if on correct side of FV
            if buy_capacity > 0 and our_bid < fv:
                orders.append(Order(product, our_bid, buy_capacity))

            if sell_capacity > 0 and our_ask > fv:
                orders.append(Order(product, our_ask, -sell_capacity))

            result[product] = orders

        traderData = "V4_SKEW_FIX"
        conversions = 0
        return result, conversions, traderData