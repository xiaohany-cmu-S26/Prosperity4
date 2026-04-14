import math
from datamodel import OrderDepth, TradingState, Order
from typing import List

POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 40
}
DEFAULT_LIMIT = 80


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
        """
        Emeralds: StaticTrader logic from FrankfurtHedgehogs.
        Taking: hit asks <= wall_mid-1 (always) or at wall_mid to cover a short.
                hit bids >= wall_mid+1 (always) or at wall_mid to unload a long.
        Making: base at bid_wall+1 / ask_wall-1, then overbid/undercut the best
                thick level (volume > 1) that is still on the correct side of mid.
        """
        orders = []

        bid_wall, wall_mid, ask_wall = self.get_walls(order_depth)
        if wall_mid is None:
            return orders

        buy_capacity = limit - position
        sell_capacity = limit + position

        sell_orders = sorted(order_depth.sell_orders.items())          # asc: best ask first
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)  # desc: best bid first

        # ── 1. Taking ────────────────────────────────────────────────────────
        for sp, sv in sell_orders:
            if buy_capacity <= 0:
                break
            sv = abs(sv)
            if sp <= wall_mid - 1:
                vol = min(buy_capacity, sv)
                orders.append(Order("EMERALDS", sp, vol))
                buy_capacity -= vol
            elif sp <= wall_mid and position < 0:
                vol = min(sv, abs(position), buy_capacity)
                if vol > 0:
                    orders.append(Order("EMERALDS", sp, vol))
                    buy_capacity -= vol
            else:
                break

        for bp, bv in buy_orders:
            if sell_capacity <= 0:
                break
            if bp >= wall_mid + 1:
                vol = min(sell_capacity, bv)
                orders.append(Order("EMERALDS", bp, -vol))
                sell_capacity -= vol
            elif bp >= wall_mid and position > 0:
                vol = min(bv, position, sell_capacity)
                if vol > 0:
                    orders.append(Order("EMERALDS", bp, -vol))
                    sell_capacity -= vol
            else:
                break

        # ── 2. Making ────────────────────────────────────────────────────────
        bid_price = int(bid_wall + 1)  # base
        ask_price = int(ask_wall - 1)  # base

        # Overbidding: jump ahead of the best thick bid below wall_mid
        for bp, bv in buy_orders:
            overbidding_price = bp + 1
            if bv > 1 and overbidding_price < wall_mid:
                bid_price = max(bid_price, overbidding_price)
                break
            elif bp < wall_mid:
                bid_price = max(bid_price, bp)
                break

        # Undercutting: undercut the best thick ask above wall_mid
        for sp, sv in sell_orders:
            sv = abs(sv)
            underbidding_price = sp - 1
            if sv > 1 and underbidding_price > wall_mid:
                ask_price = min(ask_price, underbidding_price)
                break
            elif sp > wall_mid:
                ask_price = min(ask_price, sp)
                break

        if buy_capacity > 0:
            orders.append(Order("EMERALDS", bid_price, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order("EMERALDS", ask_price, -sell_capacity))

        return orders

    def get_tomato_orders(self, order_depth: OrderDepth, position: int, limit: int) -> List[Order]:
        """
        Tomatoes:
        Taking: buy any ask strictly below wall_mid (positive edge);
                sell any bid strictly above wall_mid (positive edge).
        Making: overbid best_bid by 1 tick, undercut best_ask by 1 tick,
                capped at floor/ceil of wall_mid (zero edge max, never negative edge).
        """
        orders = []

        _, wall_mid, _ = self.get_walls(order_depth)
        if wall_mid is None:
            return orders

        buy_capacity = limit - position
        sell_capacity = limit + position

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        sell_orders = sorted(order_depth.sell_orders.items())              # asc: best ask first
        buy_orders  = sorted(order_depth.buy_orders.items(), reverse=True) # desc: best bid first

        # ── 1. Taking ────────────────────────────────────────────────────────
        for sp, sv in sell_orders:
            if buy_capacity <= 0:
                break
            sv = abs(sv)
            if sp < wall_mid:
                vol = min(buy_capacity, sv)
                orders.append(Order("TOMATOES", sp, vol))
                buy_capacity -= vol
            else:
                break

        for bp, bv in buy_orders:
            if sell_capacity <= 0:
                break
            if bp > wall_mid:
                vol = min(sell_capacity, bv)
                orders.append(Order("TOMATOES", bp, -vol))
                sell_capacity -= vol
            else:
                break

        # ── 2. Making ────────────────────────────────────────────────────────
        bid_price = min(best_bid + 1, math.floor(wall_mid))
        ask_price = max(best_ask - 1, math.ceil(wall_mid))

        if buy_capacity > 0:
            orders.append(Order("TOMATOES", bid_price, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order("TOMATOES", ask_price, -sell_capacity))

        return orders

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, DEFAULT_LIMIT)

            if product == "EMERALDS":
                result[product] = self.get_emerald_orders(order_depth, position, limit)

            elif product == "TOMATOES":
                result[product] = self.get_tomato_orders(order_depth, position, limit)

        conversions = 0
        return result, conversions, ""