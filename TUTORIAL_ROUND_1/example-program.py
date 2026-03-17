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
            # MARKET MAKING STRATEGY
            if product == "EMERALDS":
                # Place a resting limit order to BUY at 10000
                print("Placing buy order at 10000")
                orders.append(Order(product, 10000, 10))

                # Place a resting limit order to SELL at 10002
                print("Placing sell order at 10002")
                orders.append(Order(product, 10002, -10))

            result[product] = orders

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
