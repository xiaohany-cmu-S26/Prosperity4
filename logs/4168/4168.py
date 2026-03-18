from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

EMERALD_FAIR_VALUE = 10000
POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80
}
DEFAULT_LIMIT = 80

INFORMED_TRADER_ID = 'Olivia'
TOMATOES_INFORMED_WINDOW = 500   # ticks to stay in signal after Olivia trades
TOMATOES_INFORMED_TARGET = 64    # ~80% of 80 position limit


class Trader:

    def get_walls(self, order_depth: OrderDepth):
        """
        Frankfurt-style wall mid: outermost bid and ask define the fair value range.
        bid_wall = min(buy_orders), ask_wall = max(sell_orders), wall_mid = midpoint.
        """
        bid_wall = ask_wall = wall_mid = None
        try:
            bid_wall = min(order_depth.buy_orders.keys())
        except Exception:
            pass
        try:
            ask_wall = max(order_depth.sell_orders.keys())
        except Exception:
            pass
        try:
            wall_mid = (bid_wall + ask_wall) / 2
        except Exception:
            pass
        return bid_wall, wall_mid, ask_wall

    def get_emerald_orders(self, order_depth: OrderDepth, position: int, limit: int) -> List[Order]:
        """Emeralds: fixed fair value + aggressive taking + passive making with inventory skew."""
        orders = []
        fv = EMERALD_FAIR_VALUE

        if abs(position) < 40:
            skew = 0.0
        elif abs(position) < 60:
            skew = 0.05
        else:
            skew = 0.2
        fv_adj = fv - skew * position

        buy_capacity = limit - position
        sell_capacity = limit + position

        # Taking
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price > fv_adj or buy_capacity <= 0:
                break
            vol = min(buy_capacity, abs(order_depth.sell_orders[ask_price]))
            orders.append(Order("EMERALDS", ask_price, vol))
            buy_capacity -= vol

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price < fv_adj or sell_capacity <= 0:
                break
            vol = min(sell_capacity, order_depth.buy_orders[bid_price])
            orders.append(Order("EMERALDS", bid_price, -vol))
            sell_capacity -= vol

        # Making
        if buy_capacity > 0 and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
            our_bid = best_bid + 1
            if our_bid >= best_ask:
                our_bid = best_bid
            if our_bid < fv_adj:
                orders.append(Order("EMERALDS", our_bid, buy_capacity))

        if sell_capacity > 0 and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            our_ask = best_ask - 1
            if our_ask <= best_bid:
                our_ask = best_ask
            if our_ask > fv_adj:
                orders.append(Order("EMERALDS", our_ask, -sell_capacity))

        return orders

    def get_tomato_orders(self, order_depth: OrderDepth, position: int, limit: int,
                          timestamp: int, current_data: dict) -> List[Order]:
        """
        Tomatoes: Kelp-style passive market making using outer-wall mid as fair value.

        Strategy (translated from FrankfurtHedgehogs DynamicTrader):
        - bid_wall = min(buy_orders), ask_wall = max(sell_orders) — outer edges of the book
        - wall_mid = (bid_wall + ask_wall) / 2 — drifting fair value estimate
        - Default: quote at bid_wall+1 / ask_wall-1 (join inner edge, no queue front-running)
        - Spread safeguard: if quote lands within 1 tick of mid, fall back to the wall itself
        - Olivia signal: if Olivia bought/sold recently, aggressively take to TOMATOES_INFORMED_TARGET
        """
        orders = []

        bid_wall, wall_mid, ask_wall = self.get_walls(order_depth)
        if wall_mid is None:
            return orders

        buy_capacity = limit - position
        sell_capacity = limit + position

        # ── Olivia informed signal (already updated in run() from market_trades) ──
        informed_bought_ts = current_data.get('TOM_IBT', None)
        informed_sold_ts = current_data.get('TOM_IST', None)

        # Determine current direction bias
        informed_direction = 0  # NEUTRAL
        if informed_bought_ts is not None and informed_sold_ts is None:
            informed_direction = 1   # LONG
        elif informed_sold_ts is not None and informed_bought_ts is None:
            informed_direction = -1  # SHORT
        elif informed_bought_ts is not None and informed_sold_ts is not None:
            if informed_sold_ts > informed_bought_ts:
                informed_direction = -1
            elif informed_bought_ts > informed_sold_ts:
                informed_direction = 1

        # ── Bid side ─────────────────────────────────────────────────────────
        bid_price = bid_wall + 1
        bid_volume = buy_capacity

        if informed_bought_ts is not None and informed_bought_ts + TOMATOES_INFORMED_WINDOW >= timestamp:
            # Olivia bought recently → aggressively take to target long
            if position < TOMATOES_INFORMED_TARGET:
                bid_price = ask_wall  # cross spread to take liquidity
                bid_volume = TOMATOES_INFORMED_TARGET - position
        else:
            # Spread safeguard: if quote within 1 tick of mid and we're biased short, step back
            if wall_mid - bid_price < 1 and informed_direction == -1 and position > -TOMATOES_INFORMED_TARGET:
                bid_price = bid_wall

        if bid_volume > 0 and buy_capacity > 0:
            orders.append(Order("TOMATOES", int(bid_price), min(bid_volume, buy_capacity)))

        # ── Ask side ─────────────────────────────────────────────────────────
        ask_price = ask_wall - 1
        ask_volume = sell_capacity

        if informed_sold_ts is not None and informed_sold_ts + TOMATOES_INFORMED_WINDOW >= timestamp:
            # Olivia sold recently → aggressively take to target short
            if position > -TOMATOES_INFORMED_TARGET:
                ask_price = bid_wall  # cross spread to take liquidity
                ask_volume = TOMATOES_INFORMED_TARGET + position

        # Spread safeguard: if quote within 1 tick of mid and we're biased long, step back
        if ask_price - wall_mid < 1 and informed_direction == 1 and position < TOMATOES_INFORMED_TARGET:
            ask_price = ask_wall

        if ask_volume > 0 and sell_capacity > 0:
            orders.append(Order("TOMATOES", int(ask_price), -min(ask_volume, sell_capacity)))

        return orders

    def run(self, state: TradingState):
        result = {}

        # Load persisted state
        try:
            last_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            last_data = {}

        new_data = {}

        # Update Olivia signal from market trades
        for product, trades in state.market_trades.items():
            for trade in trades:
                if trade.buyer == INFORMED_TRADER_ID:
                    key = f'{product[:3]}_IBT'
                    new_data[key] = trade.timestamp
                if trade.seller == INFORMED_TRADER_ID:
                    key = f'{product[:3]}_IST'
                    new_data[key] = trade.timestamp

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, DEFAULT_LIMIT)

            if product == "EMERALDS":
                result[product] = self.get_emerald_orders(order_depth, position, limit)

            elif product == "TOMATOES":
                # Merge persisted state with freshly updated signals for this tick
                current_data = {**last_data, **new_data}
                result[product] = self.get_tomato_orders(
                    order_depth, position, limit, state.timestamp, current_data
                )

        try:
            traderData = json.dumps(new_data)
        except Exception:
            traderData = ""

        conversions = 0
        return result, conversions, traderData