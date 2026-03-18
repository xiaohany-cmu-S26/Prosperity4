from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Optional

POSITION_LIMIT = 80   # max units long or short
MIDWALL_TOP_N = 3     # number of highest-volume levels to use per side

# Per-product skew thresholds: if |position| exceeds the value, post at fair value to revert.
# Products not listed here fall back to DEFAULT_SKEW_THRESHOLD.
DEFAULT_SKEW_THRESHOLD = 40
SKEW_THRESHOLDS: dict[str, int] = {
    "EMERALDS": 68,
    "TOMATOES": 30,
}


def midwall(order_depth: OrderDepth, top_n: int = MIDWALL_TOP_N) -> Optional[float]:
    """
    Wall Mid: the average of the bid wall and ask wall prices.

    The walls are the price levels with the deepest liquidity on each side,
    identified by finding the level with the highest volume. These walls
    represent where designated market makers quote, and their average provides
    a stable, low-noise estimate of the true underlying price.
    """
    def wall_price(levels: dict) -> Optional[float]:
        if not levels:
            return None
        # Find the single price level with the highest volume (the wall)
        max_level = max(levels.items(), key=lambda x: abs(x[1]))
        return max_level[0]

    bid_wall = wall_price(order_depth.buy_orders)
    ask_wall = wall_price(order_depth.sell_orders)

    if bid_wall is None and ask_wall is None:
        return None
    if bid_wall is None:
        return ask_wall
    if ask_wall is None:
        return bid_wall
    return (bid_wall + ask_wall) / 2


def fair_value(order_depth: OrderDepth) -> Optional[float]:
    return midwall(order_depth)


class Trader:

    def run(self, state: TradingState):
        result = {}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            position = state.position.get(product, 0)

            fv = fair_value(order_depth)
            if fv is None:
                result[product] = orders
                continue

            print(f"FAIR VALUE {product}: {fv:.2f}")

            # How much capacity we have left on each side
            buy_capacity  = POSITION_LIMIT - position   # max we can still buy
            sell_capacity = POSITION_LIMIT + position   # max we can still sell

            # ── Step 1: Take profitable quotes already on the book ────────────
            # Buy every ask that is strictly below fair value (no loss)
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price >= fv:
                    break
                if buy_capacity <= 0:
                    break
                available = -order_depth.sell_orders[ask_price]  # stored negative
                qty = min(available, buy_capacity)
                # Validate we're not taking at a loss
                if ask_price < fv:
                    orders.append(Order(product, ask_price, qty))
                    buy_capacity -= qty
                    print(f"TAKE BUY  {qty}x {product} @ {ask_price} (FV: {fv:.2f})")
                else:
                    print(f"REJECT BUY  @ {ask_price}: at/above FV {fv:.2f}")

            # Sell every bid that is strictly above fair value (no loss)
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price <= fv:
                    break
                if sell_capacity <= 0:
                    break
                available = order_depth.buy_orders[bid_price]
                qty = min(available, sell_capacity)
                # Validate we're not taking at a loss
                if bid_price > fv:
                    orders.append(Order(product, bid_price, -qty))
                    sell_capacity -= qty
                    print(f"TAKE SELL {qty}x {product} @ {bid_price} (FV: {fv:.2f})")
                else:
                    print(f"REJECT SELL @ {bid_price}: at/below FV {fv:.2f}")

            # ── Step 2: Post quotes a tick better than the current best ───────
            # If inventory is too skewed, post at fair value (rounded) to revert,
            # but only if rounding doesn't create a loss.
            fv_int = int(round(fv))
            skew_threshold = SKEW_THRESHOLDS.get(product, DEFAULT_SKEW_THRESHOLD)
            if abs(position) > skew_threshold:
                if position > 0 and sell_capacity > 0:
                    # Too long — offer to sell. Use higher of (fv_int, fv) to avoid loss.
                    sell_price = max(fv_int, int(fv))
                    orders.append(Order(product, sell_price, -sell_capacity))
                    print(f"SKEW ASK  {sell_capacity}x {product} @ {sell_price} (pos={position}, FV: {fv:.2f})")
                elif position < 0 and buy_capacity > 0:
                    # Too short — bid at fair value. Use lower of (fv_int, fv) to avoid loss.
                    buy_price = min(fv_int, int(fv))
                    orders.append(Order(product, buy_price, buy_capacity))
                    print(f"SKEW BID  {buy_capacity}x {product} @ {buy_price} (pos={position}, FV: {fv:.2f})")
            else:
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

                # Aggressive quoting strategy
                if spread is not None and spread < 3:
                    # Tight spread: post at fair value with reduced size for higher fill rate
                    fv_int = int(round(fv))
                    qty_tight = max(1, buy_capacity // 2)  # post half size in tight spreads

                    if buy_capacity > 0:
                        orders.append(Order(product, fv_int, qty_tight))
                        print(f"AGG BID {qty_tight}x {product} @ {fv_int} (tight spread={spread}, FV: {fv:.2f})")

                    if sell_capacity > 0:
                        orders.append(Order(product, fv_int, -qty_tight))
                        print(f"AGG ASK {qty_tight}x {product} @ {fv_int} (tight spread={spread}, FV: {fv:.2f})")
                else:
                    # Normal spread: post +1/-1 ticks from market with full capacity
                    # Our bid: 1 tick above the current best bid (beat the queue)
                    if best_bid is not None and buy_capacity > 0:
                        our_bid = best_bid + 1
                        if our_bid < fv:
                            orders.append(Order(product, our_bid, buy_capacity))
                            print(f"QUOTE BID {buy_capacity}x {product} @ {our_bid} (FV: {fv:.2f})")
                        else:
                            print(f"REJECT BID QUOTE @ {our_bid}: at/above FV {fv:.2f}")

                    # Our ask: 1 tick below the current best ask (beat the queue)
                    if best_ask is not None and sell_capacity > 0:
                        our_ask = best_ask - 1
                        if our_ask > fv:
                            orders.append(Order(product, our_ask, -sell_capacity))
                            print(f"QUOTE ASK {sell_capacity}x {product} @ {our_ask} (FV: {fv:.2f})")
                        else:
                            print(f"REJECT ASK QUOTE @ {our_ask}: at/below FV {fv:.2f}")

            result[product] = orders

        traderData = ""
        conversions = 0
        return result, conversions, traderData