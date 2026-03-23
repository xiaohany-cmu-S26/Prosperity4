from datamodel import OrderDepth, TradingState, Order
from typing import List, Optional
import math

EMERALD_FAIR_VALUE = 10000
POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80
}
DEFAULT_LIMIT = 80


class Trader:
    TIGHT_SPREAD_THRESHOLD = 9

    # reduce quote size under tight spread 
    def tight_size_multiplier(self, spread: int) -> float:
        if spread <= 5:
            return 0.2
        elif spread <= 7:
            return 0.3
        else:
            return 0.5

    def micro_dev_l2(self, order_depth: OrderDepth) -> Optional[float]:
        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        asks = sorted(order_depth.sell_orders.keys())
        if len(bids) < 2 or len(asks) < 2:
            return None
        bid_price_2 = bids[1]
        ask_price_2 = asks[1]
        bid_volume_2 = abs(order_depth.buy_orders[bid_price_2])
        ask_volume_2 = abs(order_depth.sell_orders[ask_price_2])
        total_vol = bid_volume_2 + ask_volume_2
        if total_vol == 0:
            return None
        return (bid_price_2 * ask_volume_2 + ask_price_2 * bid_volume_2) / total_vol

    def get_spread(self, order_depth: OrderDepth) -> Optional[int]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys()) - max(order_depth.buy_orders.keys())

    def get_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

    def get_fair_value(self, product: str, order_depth: OrderDepth, position: int) -> Optional[float]:
        if product == "EMERALDS":
            fv = EMERALD_FAIR_VALUE
        else:
            mid = self.get_mid(order_depth)
            if mid is None:
                return None

            spread = self.get_spread(order_depth)

            # Wide spread only: use L2 microprice
            if spread is not None and spread >= 10:
                mp_l2 = self.micro_dev_l2(order_depth)
                fv = mp_l2 if mp_l2 is not None else mid
            else:
                fv = mid

        if fv is None:
            return None

        # Inventory skew only for EMERALDS
        if product == "EMERALDS":
            if abs(position) < 40:
                skew = 0.0
            elif abs(position) < 60:
                skew = 0.05 * position
            else:
                skew = 0.2 * position
            fv -= skew

        return fv

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, DEFAULT_LIMIT)

            fv = self.get_fair_value(product, order_depth, position)
            if fv is None:
                result[product] = orders
                continue

            buy_capacity = limit - position
            sell_capacity = limit + position

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            is_wide = spread >= 10
            is_tight = spread <= self.TIGHT_SPREAD_THRESHOLD

            # ── Step 1: Taking (Aggressive) ───────────────────────────
            if product == "EMERALDS" or is_wide:
                # Full taking in both directions
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price > fv or buy_capacity <= 0:
                        break
                    vol = abs(order_depth.sell_orders[ask_price])
                    qty = min(buy_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        buy_capacity -= qty

                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price < fv or sell_capacity <= 0:
                        break
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, vol)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty

            elif product == "TOMATOES":
                # Tight spread: aggressive only to unwind
                if position < 0:
                    for ask_price in sorted(order_depth.sell_orders.keys()):
                        if ask_price > fv or buy_capacity <= 0:
                            break
                        vol = abs(order_depth.sell_orders[ask_price])
                        qty = min(buy_capacity, vol, abs(position))
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            buy_capacity -= qty

                elif position > 0:
                    for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                        if bid_price < fv or sell_capacity <= 0:
                            break
                        vol = order_depth.buy_orders[bid_price]
                        qty = min(sell_capacity, vol, position)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            sell_capacity -= qty

            # ── Step 2: Making (Passive) ──────────────────────────────
            conservative_bid = best_bid + 1
            conservative_ask = best_ask - 1

            if product == "TOMATOES":
                abs_pos = abs(position)

                # -----------------------------
                # Tight spread defensive mode
                # cancel symmetric quotes
                # reduce size
                # only unwind inventory
                # -----------------------------
                if is_tight:
                    size_mult = self.tight_size_multiplier(spread)

                    if position > 0:
                        # long inventory -> only sell to unwind
                        unwind_qty = min(position, sell_capacity)
                        quote_qty = max(1, int(unwind_qty * size_mult))

                        # more aggressive ask for unwind
                        if abs_pos <= 39:
                            our_ask = conservative_ask
                        elif abs_pos <= 63:
                            our_ask = max(conservative_ask, int(fv) + 3)
                        else:
                            our_ask = max(conservative_ask, math.ceil(fv) if fv != int(fv) else int(fv) + 1)

                        # safety
                        if our_ask <= best_bid:
                            our_ask = best_ask

                        if quote_qty > 0 and our_ask > fv:
                            orders.append(Order(product, our_ask, -quote_qty))

                    elif position < 0:
                        # short inventory -> only buy to unwind
                        unwind_qty = min(abs(position), buy_capacity)
                        quote_qty = max(1, int(unwind_qty * size_mult))

                        # more aggressive bid for unwind
                        if abs_pos <= 39:
                            our_bid = conservative_bid
                        elif abs_pos <= 63:
                            our_bid = min(conservative_bid, int(fv) - 3)
                        else:
                            our_bid = min(conservative_bid, math.floor(fv) if fv != int(fv) else int(fv) - 1)

                        # safety
                        if our_bid >= best_ask:
                            our_bid = best_bid

                        if quote_qty > 0 and our_bid < fv:
                            orders.append(Order(product, our_bid, quote_qty))

                    else:
                        # flat inventory: no symmetric quotes in tight regime
                        pass

                # -----------------------------
                # Wide spread normal MM mode
                # -----------------------------
                else:
                    if abs_pos <= 39:
                        # Pure MM: symmetric conservative
                        our_bid = conservative_bid
                        our_ask = conservative_ask

                    elif abs_pos <= 63:
                        # Moderate inventory: unwinding side more aggressive
                        if position > 0:
                            our_bid = conservative_bid
                            our_ask = int(fv) + 3
                        else:
                            our_bid = int(fv) - 3
                            our_ask = conservative_ask

                    else:
                        # Critical inventory: unwinding side near fair value
                        if position > 0:
                            our_bid = conservative_bid
                            our_ask = math.ceil(fv) if fv != int(fv) else int(fv) + 1
                        else:
                            our_bid = math.floor(fv) if fv != int(fv) else int(fv) - 1
                            our_ask = conservative_ask

                    # safety
                    our_bid = max(our_bid, best_bid)
                    our_ask = min(our_ask, best_ask)

                    if our_bid >= best_ask:
                        our_bid = best_bid
                    if our_ask <= best_bid:
                        our_ask = best_ask

                    # wide spread: normal full-size MM
                    if buy_capacity > 0 and our_bid < fv:
                        orders.append(Order(product, our_bid, buy_capacity))

                    if sell_capacity > 0 and our_ask > fv:
                        orders.append(Order(product, our_ask, -sell_capacity))

            else:
                # EMERALDS
                fv_bid = int(fv) - 1
                fv_ask = int(fv) + 1
                our_bid = min(conservative_bid, fv_bid)
                our_ask = max(conservative_ask, fv_ask)

                if our_bid >= best_ask:
                    our_bid = best_bid
                if our_ask <= best_bid:
                    our_ask = best_ask

                if buy_capacity > 0 and our_bid < fv:
                    orders.append(Order(product, our_bid, buy_capacity))

                if sell_capacity > 0 and our_ask > fv:
                    orders.append(Order(product, our_ask, -sell_capacity))

            result[product] = orders

        traderData = "V5_TIGHT_UNWIND"
        conversions = 0
        return result, conversions, traderData