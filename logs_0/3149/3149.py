from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Optional

EMERALD_FAIR_VALUE = 10000
POSITION_LIMIT = 80  # max units long or short
MIDWALL_TOP_N = 3    # number of highest-volume levels to use per side
EMERALD_SKEW_THRESHOLD = 60  # if |position| exceeds this, trade at fair value to revert
SKEW_THRESHOLD = 55          # same threshold applied to all other products


def midwall(order_depth: OrderDepth, top_n: int = MIDWALL_TOP_N) -> Optional[float]:
    """
    Estimate fair value from the deepest (highest-volume) quotes on each side.

    For each side, pick the top_n levels by volume and compute their
    volume-weighted average price (the "wall"). Fair value is the mean of
    the bid-wall and ask-wall prices.
    """
    def wall_price(levels: dict) -> Optional[float]:
        if not levels:
            return None
        # Sort by absolute volume descending, take the heaviest top_n levels
        top = sorted(levels.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
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


def fair_value(product: str, order_depth: OrderDepth) -> Optional[float]:
    if product == "EMERALDS":
        return EMERALD_FAIR_VALUE
    return midwall(order_depth)


class Trader:

    def run(self, state: TradingState):
        result = {}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            position = state.position.get(product, 0)

            fv = fair_value(product, order_depth)
            if fv is None:
                result[product] = orders
                continue

            print(f"FAIR VALUE {product}: {fv:.2f}")

            # How much capacity we have left on each side
            buy_capacity  = POSITION_LIMIT - position   # max we can still buy
            sell_capacity = POSITION_LIMIT + position   # max we can still sell

            # ── Step 1: Take profitable quotes already on the book ────────────
            # Buy every ask that is priced below fair value
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price >= fv:
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
                if bid_price <= fv:
                    break
                if sell_capacity <= 0:
                    break
                available = order_depth.buy_orders[bid_price]
                qty = min(available, sell_capacity)
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                print(f"TAKE SELL {qty}x {product} @ {bid_price}")

            # ── Step 2: Post quotes a tick better than the current best ───────
            # If inventory is too skewed, post directly at fair value to revert.
            skew_threshold = EMERALD_SKEW_THRESHOLD if product == "EMERALDS" else SKEW_THRESHOLD
            fv_int = int(round(fv))
            if abs(position) > skew_threshold:
                if position > 0 and sell_capacity > 0:
                    # Too long — offer to sell at fair value
                    orders.append(Order(product, fv_int, -sell_capacity))
                    print(f"SKEW ASK  {sell_capacity}x {product} @ {fv_int} (pos={position})")
                elif position < 0 and buy_capacity > 0:
                    # Too short — bid at fair value
                    orders.append(Order(product, fv_int, buy_capacity))
                    print(f"SKEW BID  {buy_capacity}x {product} @ {fv_int} (pos={position})")
            else:
                # Our bid: 1 tick above the current best bid (beat the queue)
                if order_depth.buy_orders and buy_capacity > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    our_bid  = best_bid + 1
                    # Don't quote above fair value
                    if our_bid < fv:
                        orders.append(Order(product, our_bid, buy_capacity))
                        print(f"QUOTE BID {buy_capacity}x {product} @ {our_bid}")

                # Our ask: 1 tick below the current best ask (beat the queue)
                if order_depth.sell_orders and sell_capacity > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    our_ask  = best_ask - 1
                    # Don't quote below fair value
                    if our_ask > fv:
                        orders.append(Order(product, our_ask, -sell_capacity))
                        print(f"QUOTE ASK {sell_capacity}x {product} @ {our_ask}")

            result[product] = orders

        traderData = ""
        conversions = 0
        return result, conversions, traderData