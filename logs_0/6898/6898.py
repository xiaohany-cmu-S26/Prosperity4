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
    def __init__(self):
        # Price history for trend detection (TOMATOES)
        self.price_history = []  # List of (timestamp, price) tuples
        self.trend_direction = None  # +1 for up, -1 for down, None for no trend
        self.trend_change_tick = None  # When current trend started
        self.mean_delta = 0.0  # Average delta during current trend

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

    def update_trend(self, timestamp: int, current_price: float) -> None:
        """Track price history and detect meaningful trend changes for TOMATOES."""
        self.price_history.append((timestamp, current_price))

        # Keep only last 70 ticks for trend detection
        if len(self.price_history) > 70:
            self.price_history.pop(0)

        # Need at least 8 ticks to detect a trend change
        if len(self.price_history) < 8:
            return

        # Compute deltas for last 60 ticks
        if len(self.price_history) < 60:
            deltas_60 = [self.price_history[i+1][1] - self.price_history[i][1]
                        for i in range(len(self.price_history) - 1)]
        else:
            recent = self.price_history[-60:]
            deltas_60 = [recent[i+1][1] - recent[i][1] for i in range(len(recent) - 1)]

        if not deltas_60:
            return

        # Determine current trend direction (last 7 ticks)
        recent_deltas = deltas_60[-7:]
        recent_trend = 1 if sum(recent_deltas) > 0 else -1

        # Detect meaningful direction change (7+ consecutive ticks in same direction)
        if self.trend_direction != recent_trend:
            # Check if recent trend is sustained (7 ticks)
            if len([d for d in recent_deltas if (recent_trend > 0 and d > 0) or (recent_trend < 0 and d < 0)]) >= 6:
                # Direction change detected
                self.trend_direction = recent_trend
                self.trend_change_tick = len(self.price_history) - 1

        # Calculate mean delta during trending periods (only after trend change or during trend)
        if self.trend_direction is not None and self.trend_change_tick is not None:
            ticks_since_change = len(self.price_history) - 1 - self.trend_change_tick
            if ticks_since_change >= 7:  # Trend has lasted 7+ ticks
                self.mean_delta = sum(deltas_60) / len(deltas_60) if deltas_60 else 0.0

    def get_fair_value(self, product: str, order_depth: OrderDepth, position: int, timestamp: int = 0, current_price: float = 0) -> Optional[float]:
        if product == "EMERALDS":
            fv = EMERALD_FAIR_VALUE
        else:
            # TOMATOES and others use Midwall
            fv = self.midwall(order_depth)

        if fv is None:
            return None

        # ── Trend-based Delta Adjustment (TOMATOES only) ─────────────────────
        if product == "TOMATOES" and current_price > 0:
            self.update_trend(timestamp, current_price)
            # Add trend delta to fair value
            fv += self.mean_delta

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

            # Compute current mid price for trend detection
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                current_price = (best_bid + best_ask) / 2
            else:
                current_price = 0

            # 2. Calculate Fair Value (with trend delta for TOMATOES)
            fv = self.get_fair_value(product, order_depth, position, timestamp=state.timestamp, current_price=current_price)
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

            # ── Step 2: Making (Passive) ─────────────────────────────────────
            # Place limit orders to capture spread

            # Quote Bid
            if buy_capacity > 0:
                best_bid = max(order_depth.buy_orders.keys()
                               ) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()
                               ) if order_depth.sell_orders else float('inf')

                # Default: Improve best bid by 1
                our_bid = best_bid + 1

                # Check spread constraint: If spread is 1 tick, we can't improve without crossing.
                # If best_ask - best_bid <= 1, our_bid (best_bid+1) >= best_ask.
                # Since we already attempted to take at best_ask in Step 1 (if it was < FV),
                # If we are here, best_ask >= FV.
                # So if we quote at best_ask, we are bidding >= FV, which is bad (unless skewed).
                # But logical check: We only quote if `our_bid < fv`.

                # Special handling for tight spread: Join the queue instead of doing nothing?
                if our_bid >= best_ask:
                    our_bid = best_bid  # Join

                if our_bid < fv:
                    orders.append(Order(product, our_bid, buy_capacity))

            # Quote Ask
            if sell_capacity > 0:
                best_ask = min(order_depth.sell_orders.keys()
                               ) if order_depth.sell_orders else float('inf')
                best_bid = max(order_depth.buy_orders.keys()
                               ) if order_depth.buy_orders else 0

                # Default: Undercut best ask by 1
                our_ask = best_ask - 1

                # Check spread constraint
                if our_ask <= best_bid:
                    our_ask = best_ask  # Join

                if our_ask > fv:
                    orders.append(Order(product, our_ask, -sell_capacity))

            result[product] = orders

        traderData = "OPTIMIZED_V3"
        conversions = 0
        return result, conversions, traderData