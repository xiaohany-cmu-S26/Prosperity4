import json
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Optional, Tuple

# ── Per-product configuration ─────────────────────────────────────────────────
# fair_value:    "midwall" (EMA of best-bid/ask mid) or a float (hardcoded)
# ema_alpha:     EMA smoothing factor (only used when fair_value="midwall")
# skew_tiers:    list of (position_threshold, skew_factor)
#                Skew adjusts FV away from current position to manage inventory.
# quote_offset:  ticks outside best bid/ask for passive quotes (book-relative)
# fv_spread:     if set, post passive quotes at base_fv ± fv_spread (FV-relative)
#                overrides quote_offset for passive making
# max_passive_qty: max units per passive order
PRODUCT_CONFIG = {
    "INTARIAN_PEPPER_ROOT": {
        "limit": 80,
        # FV rises at exactly 0.001/timestamp-unit (= 0.1/tick = 1000/day).
        # Anchor to the first observed mid each day and extrapolate linearly.
        "fair_value": "linear_trend",
        "trend_slope": 0.001,
        "skew_tiers": [
            (40, 0.0),   # |pos| <  40  -> pure market-making
            (60, 0.05),  # |pos| <  60  -> light inventory skew
            (80, 0.2),   # |pos| >= 60  -> aggressive inventory clear
        ],
        # Post 1 tick inside the best bid/ask (quote_offset=-1 → bid=best_bid+1,
        # ask=best_ask-1). Structural MMs rest at best_bid/best_ask with queue
        # priority; posting inside gives us queue priority over them.
        # Spread is typically ~13 ticks, so 11-tick round-trip is still profitable.
        # max_passive_qty=20: structural MM resting volume peaks at 25; with
        # queue priority (posting inside) we can now fill larger incoming orders.
        "quote_offset": -1,
        "max_passive_qty": 20,
    },
    "ASH_COATED_OSMIUM": {
        "limit": 80,
        # True fair value is anchored at 10000 — strong mean reversion confirmed
        # empirically (lag-1 autocorr = -0.49, E[return | far from 10k] is large).
        "fair_value": 10000,
        "skew_tiers": [
            (80, 0.1),   # flat skew across all positions
        ],
        # Post passive quotes at fixed levels 6 ticks from FV (bid=9994, ask=10006).
        # Structural MMs anchor at ±7 (9993/10007); posting at ±6 puts us 1 tick
        # inside, giving queue priority. Also captures compressed-market asks at
        # 10006 that the previous ±7 quote missed entirely.
        "fv_spread": 6,
        "max_passive_qty": 12,
    },
}

DEFAULT_LIMIT = 80


class Trader:

    # ── Fair-value helpers ────────────────────────────────────────────────────

    def raw_mid(self, order_depth: OrderDepth) -> Optional[float]:
        """Best-bid / best-ask mid (top of book)."""
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return None
        if not order_depth.buy_orders:
            return float(min(order_depth.sell_orders.keys()))
        if not order_depth.sell_orders:
            return float(max(order_depth.buy_orders.keys()))
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

    def skew_factor(self, position: int, tiers: list) -> float:
        for threshold, factor in tiers:
            if abs(position) < threshold:
                return factor
        return tiers[-1][1]

    def get_base_fv(
        self,
        product: str,
        order_depth: OrderDepth,
        ema_state: Dict[str, float],
        timestamp: int,
    ) -> Optional[float]:
        """
        Returns the base (pre-skew) fair value.
        Fixed float         -> use directly.
        'linear_trend'      -> init_fv + slope * (ts - init_ts), anchored at
                               the first observed mid each day.
        'midwall' (default) -> EMA of top-of-book mid.
        """
        cfg    = PRODUCT_CONFIG.get(product, {})
        fv_cfg = cfg.get("fair_value", "midwall")

        if isinstance(fv_cfg, (int, float)):
            return float(fv_cfg)

        if fv_cfg == "linear_trend":
            slope       = cfg.get("trend_slope", 0.0)
            init_fv_key = product + "_init_fv"
            init_ts_key = product + "_init_ts"
            mid = self.raw_mid(order_depth)
            if mid is None:
                return None
            if init_fv_key not in ema_state:
                ema_state[init_fv_key] = mid
                ema_state[init_ts_key] = float(timestamp)
            return ema_state[init_fv_key] + slope * (timestamp - ema_state[init_ts_key])

        # EMA of mid-price
        mid = self.raw_mid(order_depth)
        if mid is None:
            return None
        alpha    = cfg.get("ema_alpha", 1.0)
        prev_ema = ema_state.get(product, mid)
        ema      = alpha * mid + (1 - alpha) * prev_ema
        ema_state[product] = ema
        return ema

    # ── Order-generation helpers ──────────────────────────────────────────────

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fv: float,
        buy_capacity: int,
        sell_capacity: int,
    ) -> Tuple[List[Order], int, int]:
        """Aggressive taking: lift asks below FV, hit bids above FV."""
        orders: List[Order] = []

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price >= fv or buy_capacity <= 0:
                break
            qty = min(buy_capacity, abs(order_depth.sell_orders[ask_price]))
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price <= fv or sell_capacity <= 0:
                break
            qty = min(sell_capacity, order_depth.buy_orders[bid_price])
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty

        return orders, buy_capacity, sell_capacity

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        base_fv: float,
        skewed_fv: float,
        buy_capacity: int,
        sell_capacity: int,
        max_qty: int,
        fv_spread: Optional[int],
        quote_offset: int,
    ) -> List[Order]:
        """
        Passive making.
        If fv_spread is set: quote at base_fv +/- fv_spread (FV-anchored levels).
        Otherwise: quote quote_offset ticks outside the best bid/ask.
        In both cases, orders are only posted on the correct side of skewed_fv.
        """
        orders: List[Order] = []
        fv_int = int(round(base_fv))

        if fv_spread is not None:
            # FV-anchored quotes
            bid_price = fv_int - fv_spread
            ask_price = fv_int + fv_spread
        else:
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else fv_int - 2
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else fv_int + 2
            bid_price = best_bid - quote_offset
            ask_price = best_ask + quote_offset

        if buy_capacity > 0 and bid_price < skewed_fv:
            orders.append(Order(product, bid_price, min(buy_capacity, max_qty)))

        if sell_capacity > 0 and ask_price > skewed_fv:
            orders.append(Order(product, ask_price, -min(sell_capacity, max_qty)))

        return orders

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, state: TradingState):
        try:
            ema_state: Dict[str, float] = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            ema_state = {}

        result = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []

            cfg          = PRODUCT_CONFIG.get(product, {})
            limit        = cfg.get("limit", DEFAULT_LIMIT)
            tiers        = cfg.get("skew_tiers", [(DEFAULT_LIMIT, 0.0)])
            quote_offset = cfg.get("quote_offset", 0)
            fv_spread    = cfg.get("fv_spread", None)
            max_passive  = cfg.get("max_passive_qty", limit)
            position     = state.position.get(product, 0)

            base_fv = self.get_base_fv(product, order_depth, ema_state, state.timestamp)
            if base_fv is None:
                result[product] = orders
                continue

            sf        = self.skew_factor(position, tiers)
            skewed_fv = base_fv - sf * position

            buy_capacity  = limit - position
            sell_capacity = limit + position

            # Step 1 – Aggressive taking (uses skewed FV)
            taken, buy_capacity, sell_capacity = self.take_orders(
                product, order_depth, skewed_fv, buy_capacity, sell_capacity
            )
            orders.extend(taken)

            # Step 2 – Passive making
            orders.extend(self.make_orders(
                product, order_depth, base_fv, skewed_fv,
                buy_capacity, sell_capacity,
                max_passive, fv_spread, quote_offset,
            ))

            result[product] = orders

        return result, 0, json.dumps(ema_state)
