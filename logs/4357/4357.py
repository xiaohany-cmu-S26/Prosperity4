import json
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Optional

EMERALD_FAIR_VALUE = 10000
POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80
}
DEFAULT_LIMIT = 80
MIDWALL_TOP_N = 1

# TOMATOES EMA parameters
EMA_ALPHA = 0.08          # slow EMA to track drift (1 = fully current, 0 = no update)
AC1_COEFF = 0.3           # fraction of last return to shade FV (exploits -0.44 AC1)
IMBALANCE_COEFF = 2.0     # ticks of FV adjustment per unit of volume imbalance ratio
BASE_HALF_SPREAD = 1      # minimum half-spread for making quotes
SPREAD_INV_COEFF = 0.03   # widen half-spread by this much per unit of abs(position)


class Trader:

    def midwall(self, order_depth: OrderDepth, top_n: int = MIDWALL_TOP_N) -> Optional[float]:
        def wall_price(levels: dict) -> Optional[float]:
            if not levels:
                return None
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

    def volume_imbalance(self, order_depth: OrderDepth) -> float:
        """Returns (bid_vol - ask_vol) / (bid_vol + ask_vol), range [-1, 1]."""
        bid_vol = sum(abs(v) for v in order_depth.buy_orders.values())
        ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def get_fair_value(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        ema: Optional[float],
        last_mid: Optional[float],
    ) -> Optional[float]:

        if product == "EMERALDS":
            fv = float(EMERALD_FAIR_VALUE)

            # Tiered inventory skew
            if abs(position) < 40:
                skew_factor = 0.0
            elif abs(position) < 60:
                skew_factor = 0.05
            else:
                skew_factor = 0.2

            return fv - skew_factor * position

        else:
            # ── TOMATOES (and generic products) ──────────────────────────────
            current_mid = self.midwall(order_depth)
            if current_mid is None:
                return None

            # 1. EMA-anchored fair value
            if ema is None:
                ema = current_mid
            fv = EMA_ALPHA * current_mid + (1.0 - EMA_ALPHA) * ema

            # 2. Autocorrelation reversion signal (AC1 ≈ -0.44)
            if last_mid is not None:
                last_return = current_mid - last_mid
                fv -= AC1_COEFF * last_return

            # 3. Volume imbalance directional nudge
            imbalance = self.volume_imbalance(order_depth)
            fv += IMBALANCE_COEFF * imbalance

            # 4. Tiered inventory skew
            abs_pos = abs(position)
            if abs_pos < 30:
                skew_factor = 0.0
            elif abs_pos < 55:
                skew_factor = 0.1
            else:
                skew_factor = 0.2

            fv -= skew_factor * position
            return fv

    def run(self, state: TradingState):
        # ── Load persisted state ──────────────────────────────────────────────
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        result = {}
        new_state = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, DEFAULT_LIMIT)

            # Pull per-product persisted values
            prod_state = saved.get(product, {})
            ema = prod_state.get("ema")
            last_mid = prod_state.get("last_mid")

            # Compute fair value
            fv = self.get_fair_value(product, order_depth, position, ema, last_mid)
            if fv is None:
                result[product] = orders
                new_state[product] = prod_state
                continue

            # Update EMA and last_mid for next tick
            current_mid = self.midwall(order_depth)
            if product != "EMERALDS" and current_mid is not None:
                new_ema = EMA_ALPHA * current_mid + (1.0 - EMA_ALPHA) * (ema if ema is not None else current_mid)
                new_state[product] = {"ema": new_ema, "last_mid": current_mid}
            else:
                new_state[product] = prod_state

            buy_capacity = limit - position
            sell_capacity = limit + position

            # ── Step 1: Taking (Aggressive) ───────────────────────────────────
            if order_depth.sell_orders:
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price > fv or buy_capacity <= 0:
                        break
                    vol = abs(order_depth.sell_orders[ask_price])
                    qty = min(buy_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty

            if order_depth.buy_orders:
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price < fv or sell_capacity <= 0:
                        break
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty

            # ── Step 2: Making (Passive) ─────────────────────────────────────
            # Dynamic half-spread: widen with inventory size
            half_spread = BASE_HALF_SPREAD + int(abs(position) * SPREAD_INV_COEFF)

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if buy_capacity > 0 and best_bid is not None:
                our_bid = best_bid + 1
                if best_ask is not None and our_bid >= best_ask:
                    our_bid = best_bid  # Join rather than cross
                # Enforce minimum distance from FV
                our_bid = min(our_bid, int(fv) - half_spread)
                if our_bid < fv:
                    orders.append(Order(product, our_bid, buy_capacity))

            if sell_capacity > 0 and best_ask is not None:
                our_ask = best_ask - 1
                if best_bid is not None and our_ask <= best_bid:
                    our_ask = best_ask  # Join
                our_ask = max(our_ask, int(fv) + half_spread)
                if our_ask > fv:
                    orders.append(Order(product, our_ask, -sell_capacity))

            result[product] = orders

        traderData = json.dumps(new_state)
        conversions = 0
        return result, conversions, traderData