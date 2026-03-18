from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

FAIR_VALUE = 10000
POSITION_LIMIT = 40  # max units long or short


class Trader:

    def run(self, state: TradingState):
        result = {}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            position = state.position.get(product, 0)

            # How much capacity we have left on each side
            buy_capacity  = POSITION_LIMIT - position   # max we can still buy
            sell_capacity = POSITION_LIMIT + position   # max we can still sell

            # ── Step 1: Take profitable quotes already on the book ────────────
            # Buy every ask that is priced below fair value
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price >= FAIR_VALUE:
                    break
                if buy_capacity <= 0:
                    break
                available = -order_depth.sell_orders[ask_price]  # stored negative
                qty = min(available, buy_capacity)
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                print(f"TAKE BUY  {qty}x {product} @ {ask_price}")

            # Sell every bid that is priced above fair value
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price <= FAIR_VALUE:
                    break
                if sell_capacity <= 0:
                    break
                available = order_depth.buy_orders[bid_price]
                qty = min(available, sell_capacity)
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                print(f"TAKE SELL {qty}x {product} @ {bid_price}")

            # ── Step 2: Post quotes a tick better than the current best ───────
            # Our bid: 1 tick above the current best bid (beat the queue)
            if order_depth.buy_orders and buy_capacity > 0:
                best_bid = max(order_depth.buy_orders.keys())
                our_bid  = best_bid + 1
                # Don't quote above fair value
                if our_bid < FAIR_VALUE:
                    orders.append(Order(product, our_bid, buy_capacity))
                    print(f"QUOTE BID {buy_capacity}x {product} @ {our_bid}")

            # Our ask: 1 tick below the current best ask (beat the queue)
            if order_depth.sell_orders and sell_capacity > 0:
                best_ask = min(order_depth.sell_orders.keys())
                our_ask  = best_ask - 1
                # Don't quote below fair value
                if our_ask > FAIR_VALUE:
                    orders.append(Order(product, our_ask, -sell_capacity))
                    print(f"QUOTE ASK {sell_capacity}x {product} @ {our_ask}")

            result[product] = orders

        traderData = ""
        conversions = 0
        return result, conversions, traderData