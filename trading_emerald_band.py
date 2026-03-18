from datamodel import OrderDepth, TradingState, Order
from typing import List

POSITION_LIMIT = 80
BUY_THRESHOLD  = 9992    # buy if ask <= this
SELL_THRESHOLD = 10008   # sell if bid >= this


class Trader:

    def run(self, state: TradingState):
        result = {}
        orders: List[Order] = []

        order_depth: OrderDepth = state.order_depths.get("EMERALDS")
        if order_depth is None:
            return {"EMERALDS": orders}, 0, ""

        position = state.position.get("EMERALDS", 0)
        buy_capacity  = POSITION_LIMIT - position
        sell_capacity = POSITION_LIMIT + position

        # Buy any ask at or below BUY_THRESHOLD (only better than market = already on book)
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price > BUY_THRESHOLD:
                break
            if buy_capacity <= 0:
                break
            available = -order_depth.sell_orders[ask_price]
            qty = min(available, buy_capacity)
            orders.append(Order("EMERALDS", ask_price, qty))
            buy_capacity -= qty
            print(f"BUY  {qty}x EMERALDS @ {ask_price}")

        # Sell any bid at or above SELL_THRESHOLD
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price < SELL_THRESHOLD:
                break
            if sell_capacity <= 0:
                break
            available = order_depth.buy_orders[bid_price]
            qty = min(available, sell_capacity)
            orders.append(Order("EMERALDS", bid_price, -qty))
            sell_capacity -= qty
            print(f"SELL {qty}x EMERALDS @ {bid_price}")

        # Post passive quotes at the band edges with remaining capacity,
        # but only if the best market quote hasn't already crossed our level
        # (no point quoting a bid at 10008 if the best ask is already ≤ 10008).
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        if buy_capacity > 0 and (best_ask is None or best_ask > BUY_THRESHOLD):
            orders.append(Order("EMERALDS", BUY_THRESHOLD, buy_capacity))
            print(f"QUOTE BID {buy_capacity}x EMERALDS @ {BUY_THRESHOLD}")

        if sell_capacity > 0 and (best_bid is None or best_bid < SELL_THRESHOLD):
            orders.append(Order("EMERALDS", SELL_THRESHOLD, -sell_capacity))
            print(f"QUOTE ASK {sell_capacity}x EMERALDS @ {SELL_THRESHOLD}")

        result["EMERALDS"] = orders
        return result, 0, ""
