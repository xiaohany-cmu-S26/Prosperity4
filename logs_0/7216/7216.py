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

    def get_fair_value(self, product: str, order_depth: OrderDepth, position: int) -> Optional[float]:
        if product == "EMERALDS":
            fv = EMERALD_FAIR_VALUE
        else:
            # TOMATOES and others use Midwall
            fv = self.midwall(order_depth)

        if fv is None:
            return None

        # ── Inventory Skew Optimization ───────────────────────────────────────
        # Adjust fair value based on current position to mean-revert.

        # With larger position limits (80), we can absorb more inventory.
        # But we still want to clear inventory when skewed heavily.
        skew_factor = 0.0

        if product == "EMERALDS":
            # Stable product.
            if abs(position) < 40:
                skew_factor = 0.0   # Pure Market Making
            elif abs(position) < 60:
                skew_factor = 0.05  # Moderate Inventory Management
            else:
                skew_factor = 0.2   # Aggressive Dumping/Covering
        elif product == "TOMATOES":
            # Volatile. 0.1 * 80 -> 8 ticks.
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
            # Buy any asks < FV
            if len(order_depth.sell_orders) != 0:
                best_asks = sorted(order_depth.sell_orders.keys())
                for ask_price in best_asks:
                    if ask_price > fv:
                        break  # Asks are sorted asc, so we can stop

                    if buy_capacity <= 0:
                        break

                    # Calculate quantity to take
                    vol_at_price = abs(order_depth.sell_orders[ask_price])
                    take_qty = min(buy_capacity, vol_at_price)

                    if take_qty > 0:
                        orders.append(Order(product, ask_price, take_qty))
                        buy_capacity -= take_qty
                        # print(f"TAKE BUY {product} {take_qty} @ {ask_price} (FV {fv:.2f})")

            # Sell any bids > FV
            if len(order_depth.buy_orders) != 0:
                best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
                for bid_price in best_bids:
                    if bid_price < fv:
                        break  # Bids are sorted desc, so we can stop

                    if sell_capacity <= 0:
                        break

                    vol_at_price = order_depth.buy_orders[bid_price]
                    take_qty = min(sell_capacity, vol_at_price)

                    if take_qty > 0:
                        orders.append(Order(product, bid_price, -take_qty))
                        sell_capacity -= take_qty
                        # print(f"TAKE SELL {product} {take_qty} @ {bid_price} (FV {fv:.2f})")

            # ── Step 2: Making — Hybrid Spread Regime ───────────────────────
            # Passive layer: inside bot spread for large per-fill edge
            # Active layer: tight around FV for frequent, low-adverse-selection fills
            # Allocation shifts toward active layer as position utilization rises.

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if best_bid is not None and best_ask is not None:
                pos_utilization = abs(position) / limit  # 0.0 – 1.0

                # Active layer gets more volume as position grows
                if pos_utilization < 0.5:
                    active_frac = 0.3
                elif pos_utilization < 0.75:
                    active_frac = 0.6
                else:
                    active_frac = 1.0  # Near limit: only FV-centered quotes

                passive_frac = 1.0 - active_frac

                # ── Passive layer: best_bid + 1 / best_ask - 1 ──────────
                passive_bid = best_bid + 1
                passive_ask = best_ask - 1

                # Don't cross the book
                if passive_bid >= best_ask:
                    passive_bid = best_bid
                if passive_ask <= best_bid:
                    passive_ask = best_ask

                # ── Active layer: FV - 2 / FV + 2 ───────────────────────
                active_bid = int(round(fv)) - 2
                active_ask = int(round(fv)) + 2

                # ── Allocate quantities ──────────────────────────────────
                passive_buy_qty = int(buy_capacity * passive_frac)
                active_buy_qty = buy_capacity - passive_buy_qty
                passive_sell_qty = int(sell_capacity * passive_frac)
                active_sell_qty = sell_capacity - passive_sell_qty

                # Place passive bids
                if passive_buy_qty > 0 and passive_bid < fv:
                    orders.append(Order(product, passive_bid, passive_buy_qty))

                # Place active bids
                if active_buy_qty > 0 and active_bid < fv:
                    orders.append(Order(product, active_bid, active_buy_qty))

                # Place passive asks
                if passive_sell_qty > 0 and passive_ask > fv:
                    orders.append(Order(product, passive_ask, -passive_sell_qty))

                # Place active asks
                if active_sell_qty > 0 and active_ask > fv:
                    orders.append(Order(product, active_ask, -active_sell_qty))

            result[product] = orders

        traderData = "OPTIMIZED_V3"
        conversions = 0
        return result, conversions, traderData