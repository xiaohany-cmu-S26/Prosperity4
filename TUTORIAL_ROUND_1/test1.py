from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import numpy as np
import math
import statistics

POSITION_LIMIT = 50
RSI_PERIOD = 14  # Typical RSI period


class Trader:

    def __init__(self):
        self.previous_prices = []  # To store historical prices for RSI calculation

    def calculate_rsi(self, prices: List[float]) -> float:
        if len(prices) < RSI_PERIOD:
            return 50  # Neutral RSI if not enough data

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains[-RSI_PERIOD:])
        avg_loss = np.mean(losses[-RSI_PERIOD:])

        if avg_loss == 0:
            return 100  # Avoid division by zero

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Update historical prices
            if len(order_depth.sell_orders) != 0:
                best_ask, _ = list(order_depth.sell_orders.items())[0]
                self.previous_prices.append(float(best_ask))
            elif len(order_depth.buy_orders) != 0:
                best_bid, _ = list(order_depth.buy_orders.items())[0]
                self.previous_prices.append(float(best_bid))

            # Calculate RSI
            rsi = self.calculate_rsi(self.previous_prices)

            # Adjust acceptable prices based on RSI
            if product == "EMERALDS":
                base_price = 10000
            else:
                base_price = 2000

            if rsi < 30:  # Oversold condition
                acceptable_buy_price = base_price * 1.02  # Increase buy price
                acceptable_sell_price = base_price * 1.05  # Increase sell price
            elif rsi > 70:  # Overbought condition
                acceptable_buy_price = base_price * 0.98  # Decrease buy price
                acceptable_sell_price = base_price * 0.95  # Decrease sell price
            else:
                acceptable_buy_price = base_price
                acceptable_sell_price = base_price * 1.05

            print("Acceptable buy price: " + str(acceptable_buy_price) +
                  ", Acceptable sell price: " + str(acceptable_sell_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) +
                  ", Sell order depth : " + str(len(order_depth.sell_orders)))

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(
                    order_depth.sell_orders.items())[0]
                if float(best_ask) < acceptable_buy_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -
                                  min(POSITION_LIMIT, best_ask_amount)))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(
                    order_depth.buy_orders.items())[0]
                if float(best_bid) > acceptable_sell_price:
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
