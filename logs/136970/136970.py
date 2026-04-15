from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import json
from collections import deque
from dataclasses import dataclass
import numpy as np


# ── Per-product configuration ─────────────────────────────────────────────────
#
# ASH_COATED_OSMIUM:
#   - True FV anchors to book mid (EMA), not fixed 10000.
#     Structural MM quotes 9993/10009 → implied mid ≈ 10001, not 10000.
#   - fv_spread tightened 6→4 so passive bids/asks fill symmetrically.
#   - max_passive_qty reduced to improve fill balance between sides.
#   - Asymmetric quoting enabled with tight moderate/critical thresholds
#     to prevent the persistent-short buildup seen in the 136519 run.
#
# INTARIAN_PEPPER_ROOT:
#   - Near-perfect linear trend (slope ≈ 0.001/ts, R²≈1).
#   - We add a directional long bias: min_long_position means the algorithm
#     only posts passive sells when position > min_long_position, preventing
#     the net-short drift seen in the 136519 run.
#   - spread_aware_taking disabled so we always buy into the trend.

PRODUCT_CONFIG = {
    "INTARIAN_PEPPER_ROOT": {
        "limit": 80,
        # Deterministic linear trend anchored at the first mid each day.
        # Empirically verified: slope 0.001/ts, R²≈1.0.
        "fair_value": "linear_trend",
        "trend_slope": 0.001,

        # L2 microprice reduces top-of-book noise on the wide spread.
        "use_l2_microprice": True,
        "wide_spread_threshold": 10,

        "use_factors": False,
        "factor_fv_skew": 0.0,

        # Inventory: always stay net long to ride the uptrend.
        # Sells only posted when position > min_long_position.
        "min_long_position": 40,

        # Symmetric skew tiers (only active above min_long_position).
        "skew_tiers": [(80, 0.0)],
        "asymmetric_quoting": True,
        "asym_moderate_pos": 40,
        "asym_critical_pos": 70,
        "asym_offset": 2,

        # Post 1 tick inside the best quotes.
        "quote_offset": -1,
        "max_passive_qty": None,

        # Wide spread: take freely to build long position.
        # Tight spread: only take in unwind direction (but min_long prevents over-selling).
        "spread_aware_taking": False,
    },

    "ASH_COATED_OSMIUM": {
        "limit": 80,
        # EMA of book mid as FV — tracks the true structural mid (≈10001 offset)
        # instead of the stale fixed 10000 anchor.
        "fair_value": "midwall",
        "ema_alpha": 0.05,   # slow EMA to filter tick noise, fast enough to track shifts

        # Compressed spread on ASH → L2 microprice blows up, keep disabled.
        "use_l2_microprice": False,
        "wide_spread_threshold": 10,

        "use_factors": False,
        "factor_fv_skew": 0.0,

        # Tiered FV skew for inventory management.
        "skew_tiers": [
            (30, 0.0),    # |pos| < 30 → pure MM
            (55, 0.08),   # |pos| < 55 → light skew
            (80, 0.25),   # |pos| ≥ 55 → aggressive clear
        ],

        # Asymmetric quoting prevents the persistent-short buildup.
        # When long/short beyond moderate, skew the passive quote toward FV.
        "asymmetric_quoting": True,
        "asym_moderate_pos": 30,
        "asym_critical_pos": 55,
        "asym_offset": 2,

        # Tighter passive spread (4 vs old 6) → both sides fill more symmetrically.
        "fv_spread": 4,
        "max_passive_qty": 8,

        # Mean reversion is strong → take any cross-FV opportunity unconditionally.
        "spread_aware_taking": False,
    },
}

DEFAULT_LIMIT = 80


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    # ── Book helpers ─────────────────────────────────────────────────────────

    def raw_mid(self, order_depth: OrderDepth) -> Optional[float]:
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return None
        if not order_depth.buy_orders:
            return float(min(order_depth.sell_orders.keys()))
        if not order_depth.sell_orders:
            return float(max(order_depth.buy_orders.keys()))
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

    def micro_dev_l2(self, order_depth: OrderDepth) -> Optional[float]:
        """L2 microprice: VWAP of 2nd-best bid and 2nd-best ask."""
        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        asks = sorted(order_depth.sell_orders.keys())
        if len(bids) < 2 or len(asks) < 2:
            return None
        bid_p2, ask_p2 = bids[1], asks[1]
        bid_v2 = abs(order_depth.buy_orders[bid_p2])
        ask_v2 = abs(order_depth.sell_orders[ask_p2])
        total = bid_v2 + ask_v2
        if total == 0:
            return None
        return (bid_p2 * ask_v2 + ask_p2 * bid_v2) / total

    def get_spread(self, order_depth: OrderDepth) -> Optional[int]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys()) - max(order_depth.buy_orders.keys())

    def skew_factor(self, position: int, tiers: list) -> float:
        for threshold, factor in tiers:
            if abs(position) < threshold:
                return factor
        return tiers[-1][1]

    # ── Fair value ───────────────────────────────────────────────────────────

    def get_base_fv(
        self,
        product: str,
        order_depth: OrderDepth,
        persistent: Dict,
        timestamp: int,
    ) -> Optional[float]:
        cfg    = PRODUCT_CONFIG.get(product, {})
        fv_cfg = cfg.get("fair_value", "midwall")
        ema_state = persistent.setdefault("ema", {})

        if isinstance(fv_cfg, (int, float)):
            base = float(fv_cfg)

        elif fv_cfg == "linear_trend":
            slope       = cfg.get("trend_slope", 0.0)
            init_fv_key = product + "_init_fv"
            init_ts_key = product + "_init_ts"
            mid = self.raw_mid(order_depth)
            if mid is None:
                return None
            if init_fv_key not in ema_state:
                ema_state[init_fv_key] = mid
                ema_state[init_ts_key] = float(timestamp)
            base = ema_state[init_fv_key] + slope * (timestamp - ema_state[init_ts_key])

        else:  # "midwall" — EMA of book mid
            mid = self.raw_mid(order_depth)
            if mid is None:
                return None
            alpha    = cfg.get("ema_alpha", 1.0)
            prev_ema = ema_state.get(product, mid)
            base     = alpha * mid + (1 - alpha) * prev_ema
            ema_state[product] = base

        if cfg.get("use_l2_microprice", False):
            spread = self.get_spread(order_depth)
            if spread is not None and spread >= cfg.get("wide_spread_threshold", 10):
                mp_l2 = self.micro_dev_l2(order_depth)
                if mp_l2 is not None:
                    base = mp_l2

        return base

    # ── Taking (aggressive) ──────────────────────────────────────────────────

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fv: float,
        buy_capacity: int,
        sell_capacity: int,
        position: int,
        spread_aware: bool,
        is_wide: bool,
        min_long: int,
    ) -> Tuple[List[Order], int, int]:
        """
        Take asks below FV and bids above FV.

        min_long: if set, never sell below min_long position via taking.
        spread_aware + tight spread → only take in the unwind direction.
        """
        orders: List[Order] = []

        if spread_aware and not is_wide:
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
                    # Respect min_long: don't sell below the minimum long floor.
                    max_sell = max(0, position - min_long)
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, vol, max_sell)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty
            return orders, buy_capacity, sell_capacity

        # Unconditional taking: buys below FV, sells above FV (respecting min_long).
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
            max_sell = max(0, position - min_long)
            vol = order_depth.buy_orders[bid_price]
            qty = min(sell_capacity, vol, max_sell)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty

        return orders, buy_capacity, sell_capacity

    # ── Making (passive) ─────────────────────────────────────────────────────

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        base_fv: float,
        skewed_fv: float,
        buy_capacity: int,
        sell_capacity: int,
        position: int,
        cfg: Dict,
        is_wide: bool,
        min_long: int,
    ) -> List[Order]:
        orders: List[Order] = []
        fv_spread    = cfg.get("fv_spread", None)
        quote_offset = cfg.get("quote_offset", 0)
        max_qty      = cfg.get("max_passive_qty")

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if fv_spread is not None:
            fv_int = int(round(base_fv))
            our_bid = fv_int - fv_spread
            our_ask = fv_int + fv_spread
        else:
            if best_bid is None or best_ask is None:
                return orders
            conservative_bid = best_bid - quote_offset
            conservative_ask = best_ask + quote_offset

            if cfg.get("asymmetric_quoting", False):
                abs_pos  = abs(position)
                moderate = cfg.get("asym_moderate_pos", 40)
                critical = cfg.get("asym_critical_pos", 63)
                asym_off = cfg.get("asym_offset", 3)

                if abs_pos < moderate:
                    our_bid = conservative_bid
                    our_ask = conservative_ask

                elif abs_pos <= critical:
                    if position > 0:
                        our_bid = conservative_bid
                        our_ask = int(base_fv) + asym_off
                    else:
                        our_bid = int(base_fv) - asym_off
                        our_ask = conservative_ask

                else:
                    if position > 0:
                        our_bid = conservative_bid
                        our_ask = (
                            math.ceil(base_fv)
                            if base_fv != int(base_fv)
                            else int(base_fv) + 1
                        )
                    else:
                        our_bid = (
                            math.floor(base_fv)
                            if base_fv != int(base_fv)
                            else int(base_fv) - 1
                        )
                        our_ask = conservative_ask

                if not is_wide:
                    our_bid = min(our_bid, conservative_bid)
                    our_ask = max(our_ask, conservative_ask)

                our_bid = max(our_bid, best_bid)
                our_ask = min(our_ask, best_ask)
            else:
                our_bid = conservative_bid
                our_ask = conservative_ask

        # Safety: never cross the book.
        if best_bid is not None and best_ask is not None:
            if our_bid >= best_ask:
                our_bid = best_bid
            if our_ask <= best_bid:
                our_ask = best_ask

        # Post only on the profitable side of the skewed FV.
        if buy_capacity > 0 and our_bid < skewed_fv:
            qty = buy_capacity if max_qty is None else min(buy_capacity, max_qty)
            orders.append(Order(product, our_bid, qty))

        # Passive ask: respect min_long — don't post asks that would push us below floor.
        max_sell_passive = max(0, position - min_long) if min_long > 0 else sell_capacity
        effective_sell_cap = min(sell_capacity, max_sell_passive)
        if effective_sell_cap > 0 and our_ask > skewed_fv:
            qty = effective_sell_cap if max_qty is None else min(effective_sell_cap, max_qty)
            orders.append(Order(product, our_ask, -qty))

        return orders

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self, state: TradingState):
        try:
            persistent: Dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            persistent = {}

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            cfg = PRODUCT_CONFIG.get(product, {})

            limit    = cfg.get("limit", DEFAULT_LIMIT)
            tiers    = cfg.get("skew_tiers", [(DEFAULT_LIMIT, 0.0)])
            position = state.position.get(product, 0)
            min_long = cfg.get("min_long_position", 0)

            base_fv = self.get_base_fv(product, order_depth, persistent, state.timestamp)
            if base_fv is None:
                result[product] = orders
                continue

            # Tiered inventory skew applied on top of base FV.
            sf        = self.skew_factor(position, tiers)
            skewed_fv = base_fv - sf * position

            buy_capacity  = limit - position
            sell_capacity = limit + position

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread   = best_ask - best_bid
            is_wide  = spread >= cfg.get("wide_spread_threshold", 10)

            # Step 1 — aggressive taking (uses skewed FV)
            taken, buy_capacity, sell_capacity = self.take_orders(
                product, order_depth, skewed_fv,
                buy_capacity, sell_capacity, position,
                cfg.get("spread_aware_taking", False), is_wide,
                min_long,
            )
            orders.extend(taken)

            # Step 2 — passive making
            orders.extend(self.make_orders(
                product, order_depth, base_fv, skewed_fv,
                buy_capacity, sell_capacity, position, cfg, is_wide,
                min_long,
            ))

            result[product] = orders

        return result, 0, json.dumps(persistent)