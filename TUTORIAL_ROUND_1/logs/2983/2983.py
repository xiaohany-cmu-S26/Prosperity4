from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import math

POSITION_LIMIT = 50


class Trader:

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            print(f"product: {product}, state={state.order_depths}")
            # acceptable_price = 10  # Participant should calculate this value
            if product == "EMERALDS":
                acceptable_buy_price = 10000
                acceptable_sell_price = 10002
            else:
                acceptable_buy_price = 0
                acceptable_sell_price = math.inf
            print("Acceptable buy price: " + str(acceptable_buy_price) +
                  ", Acceptable sell price: " + str(acceptable_sell_price))
            print("Buy Orders: " + str(order_depth.buy_orders) +
                  ", Sell orders : " + str(order_depth.sell_orders))

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(
                    order_depth.sell_orders.items())[0]
                if int(best_ask) <= acceptable_buy_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -
                                  min(POSITION_LIMIT, best_ask_amount)))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(
                    order_depth.buy_orders.items())[0]
                if int(best_bid) >= acceptable_sell_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -
                                  min(POSITION_LIMIT, best_bid_amount)))

            result[product] = orders

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData