from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Optional

EMERALD_FAIR_VALUE = 10000
# Updated based on prompt: 80 for both EMERALDS and TOMATOES
POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80
}
DEFAULT_LIMIT = 80
MIDWALL_TOP_N = 1    # number of highest-volume levels to use per side


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

    def midwall(self, order_depth: OrderDepth, top_n: int = MIDWALL_TOP_N) -> Optional[float]:
        """
        Estimate fair value from the deepest (highest-volume) quotes on each side.
        """
        def wall_price(levels: dict) -> Optional[float]:
            if not levels:
                return None
            # Sort by absolute volume descending, take the heaviest top_n levels
            top = sorted(levels.items(), key=lambda x: abs(
                x[1]), reverse=True)[:top_n]
            total_vol = sum(abs(v) for _, v in top)
            if total_vol == 0:
                return None
            return sum(p * abs(v) for p, v in top) / total_vol

        bid_wall = wall_price(order_depth.buy_orders)
        ask_wall = wall_price(order_depth.sell_orders)

        if bid_wall is None and ask_wall is None:
            return None
        if bid_wall is None:
            return ask_wall
        if ask_wall is None:
            return bid_wall

        return (bid_wall + ask_wall) / 2

    def get_spread(self, order_depth: OrderDepth) -> Optional[int]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return best_ask - best_bid

    def get_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def get_fair_value(self, product: str, order_depth: OrderDepth, position: int) -> Optional[float]:
        if product == "EMERALDS":
            fv = EMERALD_FAIR_VALUE
        else:
            # Start from midprice as baseline
            mid = self.get_mid(order_depth)
            if mid is None:
                return None

            spread = self.get_spread(order_depth)

            # Only use L2 microprice adjustment during wide spread (>= 10).
            # During tight spread, L2 microprice blows up (std 6.15 vs 0.35)
            # because L1 compresses while L2 stays wide, producing spurious
            # deviations of +/-8 ticks that cause toxic aggressive fills.
            if spread is not None and spread >= 10:
                mp_l2 = self.micro_dev_l2(order_depth)
                if mp_l2 is not None:
                    fv = mp_l2  # L2 microprice IS the fair value estimate
                else:
                    fv = mid
            else:
                # Tight spread regime: fall back to simple mid
                fv = mid

        if fv is None:
            return None

        # ── Inventory Skew ────────────────────────────────────────────────────
        skew_factor = 0.0

        if product == "EMERALDS":
            if abs(position) < 40:
                skew_factor = 0.0
            elif abs(position) < 60:
                skew_factor = 0.05
            else:
                skew_factor = 0.2
        elif product == "TOMATOES":
            skew_factor = 0.1

        fv_adj = fv - (skew_factor * position)
        return fv_adj

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # 1. Get current position and limit
            position = state.position.get(product, 0)
            # Default to 80 if product not generic
            limit = POSITION_LIMITS.get(product, DEFAULT_LIMIT)

            # 2. Calculate Fair Value
            fv = self.get_fair_value(product, order_depth, position)
            if fv is None:
                result[product] = orders
                continue

            # 3. Calculate Trading Capacity
            buy_capacity = limit - position
            sell_capacity = limit + position

            # ── Step 1: Taking (Aggressive) ───────────────────────────────────
            # Buy any asks < FV — but only during wide spread.
            # During tight spread, FV estimate is unreliable, so we never take.
            spread = None
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid_price = max(order_depth.buy_orders.keys())
                best_ask_price = min(order_depth.sell_orders.keys())
                spread = best_ask_price - best_bid_price

            take_allowed = (product == "EMERALDS") or (spread is not None and spread >= 10)

            if take_allowed and len(order_depth.sell_orders) != 0:
                best_asks = sorted(order_depth.sell_orders.keys())
                for ask_price in best_asks:
                    if ask_price > fv:
                        break

                    if buy_capacity <= 0:
                        break

                    vol_at_price = abs(order_depth.sell_orders[ask_price])
                    take_qty = min(buy_capacity, vol_at_price)

                    if take_qty > 0:
                        orders.append(Order(product, ask_price, take_qty))
                        buy_capacity -= take_qty

            if take_allowed and len(order_depth.buy_orders) != 0:
                best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
                for bid_price in best_bids:
                    if bid_price < fv:
                        break

                    if sell_capacity <= 0:
                        break

                    vol_at_price = order_depth.buy_orders[bid_price]
                    take_qty = min(sell_capacity, vol_at_price)

                    if take_qty > 0:
                        orders.append(Order(product, bid_price, -take_qty))
                        sell_capacity -= take_qty

            # ── Step 2: Making (Passive) ─────────────────────────────────────
            # Place limit orders to capture spread.
            # Key invariant: our_bid <= best_bid + 1 and our_ask >= best_ask - 1
            # This prevents passive quotes from ever crossing into the spread
            # during tight-spread regimes when FV might overshoot.

            if buy_capacity > 0:
                best_bid = max(order_depth.buy_orders.keys()
                               ) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()
                               ) if order_depth.sell_orders else float('inf')

                # FV-derived bid: we want to buy below FV
                fv_bid = int(fv) - 1

                # Conservative bid: improve best bid by 1
                conservative_bid = best_bid + 1

                # Cap: never bid more aggressively than best_bid + 1
                our_bid = min(conservative_bid, fv_bid)

                # Safety: don't cross the book
                if our_bid >= best_ask:
                    our_bid = best_bid  # join instead

                if our_bid < fv:
                    orders.append(Order(product, our_bid, buy_capacity))

            if sell_capacity > 0:
                best_ask = min(order_depth.sell_orders.keys()
                               ) if order_depth.sell_orders else float('inf')
                best_bid = max(order_depth.buy_orders.keys()
                               ) if order_depth.buy_orders else 0

                # FV-derived ask: we want to sell above FV
                fv_ask = int(fv) + 1
                if fv == int(fv):
                    fv_ask = int(fv) + 1  # ensure strictly above FV

                # Conservative ask: undercut best ask by 1
                conservative_ask = best_ask - 1

                # Cap: never ask less aggressively than best_ask - 1
                our_ask = max(conservative_ask, fv_ask)

                # Safety: don't cross the book
                if our_ask <= best_bid:
                    our_ask = best_ask  # join instead

                if our_ask > fv:
                    orders.append(Order(product, our_ask, -sell_capacity))

            result[product] = orders

        traderData = "OPTIMIZED_V3"
        conversions = 0
        return result, conversions, traderData