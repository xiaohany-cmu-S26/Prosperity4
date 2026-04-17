import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Optional, Tuple

# ── Product Configuration ──────────────────────────────────────────────────────
#
#  INTARIAN_PEPPER_ROOT  ("IPR")
#    Price trends linearly upward at 0.001 ticks/timestamp.  Strategy:
#    1. Ramp-up: aggressively lift the ask until we hold buy_to_target_position=75.
#    2. Hold / light-MM: protect the 75-unit floor; recycle spread on marginal
#       units (pos 75–80) via asymmetric quoting.
#
#  ASH_COATED_OSMIUM  ("ACO")
#    Mean-reverts around 10 000 (lag-1 autocorr ≈ −0.49).
#    Quoting uses TWO passive tiers per side on each tick:
#      • Primary:   1 tick INSIDE the book (quote_offset = −1)
#                   → queue priority over resting structural MMs
#      • Secondary: AT the book (quote_offset = 0, qty = at_book_qty)
#                   → fills if structural MMs are exhausted at that level
#    Inventory is managed through tiered skew + asymmetric quoting.

PRODUCT_CONFIG = {
    "INTARIAN_PEPPER_ROOT": {
        "limit": 80,

        "fair_value":  "linear_trend",
        "trend_slope": 0.001,

        "use_l2_microprice":     True,
        "wide_spread_threshold": 10,

        # Ramp-up: lift ask until we hold 75 units.
        # Economics: trend pays ~0.001 × remaining ticks; worth chasing up to 13 ticks
        # above FV (≈ half-spread) until ts ≈ 87 000.
        "buy_to_target_position": 75,
        "core_build_window":      1000,   # ramp 0→75 over the first 1 000 ts
        "core_build_per_tick":    60,
        "buy_to_ask_max_offset":  13,

        # Hold floor: never let passive sells push position below 75.
        # This leaves 5 marginal units (76–80) available for light MM.
        "min_long_position": 75,

        # No inventory skew needed — trend is monotone.
        "skew_tiers": [(80, 0.0)],

        # Light MM on the 5 marginal units above the floor.
        # When pos ≥ 75: sell at FV + 2 (moderate), FV + 1 (critical).
        "asymmetric_quoting": True,
        "asym_moderate_pos":  75,
        "asym_critical_pos":  78,
        "asym_offset":        2,

        "quote_offset":    -1,   # 1 tick inside best bid/ask for the buy leg
        "max_passive_qty": None,

        "also_quote_at_book": False,  # IPR: single-tier quoting is fine
        "at_book_qty":        0,

        "spread_aware_taking": False,
        "buy_to_ask_max_offset": 13,
    },

    "ASH_COATED_OSMIUM": {
        "limit": 80,
        "fair_value": "midwall",
        "ema_alpha":  0.05,

        "use_l2_microprice":     False,
        "wide_spread_threshold": 10,

        # No FV skew and no asymmetric quoting.
        # Insight: the position-limit capacity constraint handles inventory
        # naturally — as we go long, buy_capacity shrinks and sell_capacity grows,
        # posting fewer bids and more asks automatically.
        # Adding skew or asymmetric quoting reduces per-fill edge without a
        # commensurate volume benefit on this strongly mean-reverting product.
        "skew_tiers": [(80, 0.0)],
        "asymmetric_quoting": False,

        # Single-tier passive quoting: full capacity at best_bid+1 / best_ask-1.
        # This places us as the unique best bid/ask, ahead of the structural MM
        # (~9992/10008), capturing every incoming market order (~7 tick edge vs FV).
        # Adding layers at n=0 or n≥2 strictly hurts: any order at a better price
        # for the counterparty fills before n=1, reducing per-fill edge, and those
        # lots cannot also be used at n=1 since total capacity is fixed.
        "quote_offset":    -1,    # best_bid+1 / best_ask-1
        "max_passive_qty": None,  # post full remaining capacity

        # Size override disabled — capacity constraint is sufficient inventory mgmt.
        "size_imbalance_threshold": 80,
        "size_heavy_qty": 14,
        "size_light_qty":  4,

        "spread_aware_taking":    False,
        "buy_to_target_position": None,
        "buy_to_ask_max_offset":  None,
        "min_long_position":      0,
    },
}

DEFAULT_LIMIT = 80


# ══════════════════════════════════════════════════════════════════════════════
#  TRADER
# ══════════════════════════════════════════════════════════════════════════════

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
        """VWAP of the 2nd-level bid and ask — reduces noise for wide-spread products."""
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

    def target_core_position(
        self,
        product: str,
        cfg: Dict,
        persistent: Dict,
        timestamp: int,
        final_target: Optional[int],
    ) -> Optional[int]:
        if final_target is None:
            return None
        build_window = cfg.get("core_build_window", 0)
        if build_window <= 0:
            return int(final_target)
        core_state = persistent.setdefault("core_build", {})
        start_key = product + "_start_ts"
        if start_key not in core_state:
            core_state[start_key] = timestamp
        elapsed = max(0, timestamp - core_state[start_key])
        progress = min(1.0, elapsed / build_window)
        target = int(math.floor(final_target * progress))
        if progress >= 1.0:
            target = int(final_target)
        return max(0, min(int(final_target), target))

    # ── Fair value ───────────────────────────────────────────────────────────

    def get_base_fv(
        self,
        product: str,
        order_depth: OrderDepth,
        persistent: Dict,
        timestamp: int,
    ) -> Optional[float]:
        cfg = PRODUCT_CONFIG.get(product, {})
        fv_cfg = cfg.get("fair_value", "midwall")
        ema_state = persistent.setdefault("ema", {})

        if isinstance(fv_cfg, (int, float)):
            base = float(fv_cfg)

        elif fv_cfg == "linear_trend":
            slope = cfg.get("trend_slope", 0.0)
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
            alpha = cfg.get("ema_alpha", 1.0)
            prev_ema = ema_state.get(product, mid)
            base = alpha * mid + (1 - alpha) * prev_ema
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
        buy_to_target: Optional[int],
        core_build_per_tick: Optional[int],
        buy_to_ask_max_offset: Optional[float],
        timestamp: int,
        trend_slope: float,
        total_ticks: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []

        # Buy-to-target ramp: lift asks up to FV + effective_offset.
        if buy_to_target is not None and position < buy_to_target and buy_capacity > 0:
            remaining_time = max(0, total_ticks - timestamp)
            dynamic_max = trend_slope * remaining_time
            effective_offset = (
                min(buy_to_ask_max_offset, dynamic_max)
                if buy_to_ask_max_offset is not None
                else dynamic_max
            )
            buy_threshold = fv + effective_offset
            need = min(buy_capacity, buy_to_target - position)
            if core_build_per_tick is not None:
                need = min(need, core_build_per_tick)
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price > buy_threshold or buy_capacity <= 0 or need <= 0:
                    break
                vol = abs(order_depth.sell_orders[ask_price])
                qty = min(buy_capacity, vol, need)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_capacity -= qty
                    need -= qty
            return orders, buy_capacity, sell_capacity

        # Standard taking: cross FV.
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price >= fv or buy_capacity <= 0:
                break
            vol = abs(order_depth.sell_orders[ask_price])
            qty = min(buy_capacity, vol)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid_price <= fv or sell_capacity <= 0:
                break
            max_sell = max(0, position - min_long) if min_long > 0 else sell_capacity
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
        buy_qty_override: Optional[int] = None,
        sell_qty_override: Optional[int] = None,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return orders

        # Max sell constrained by min_long floor.
        max_sell_passive = (
            max(0, position - min_long) if min_long > 0 else sell_capacity
        )
        effective_sell_cap = min(sell_capacity, max_sell_passive)
        remaining_buy  = buy_capacity
        remaining_sell = effective_sell_cap

        # ── Multi-layer book-anchored quoting ────────────────────────────
        # bid at best_bid+n / ask at best_ask-n for each (n, qty) in quote_layers.
        # Quotes are book-anchored so they track the live book with no EMA lag.
        # skewed_fv acts as a sanity filter: we only post bids below our FV
        # estimate and asks above it, ensuring positive edge on every fill.
        # With skew_factor=0, skewed_fv = base_fv and the filter rarely binds
        # for a product with a wide, stable spread like ACO.
        if cfg.get("quote_layers") is not None:
            imb_thresh  = cfg.get("size_imbalance_threshold")
            heavy_scale = cfg.get("heavy_scale", 1.0)
            light_scale = cfg.get("light_scale", 1.0)
            abs_pos     = abs(position)
            imbalanced  = imb_thresh is not None and abs_pos >= imb_thresh

            for n, base_qty in cfg["quote_layers"]:
                bid_price = best_bid + n
                ask_price = best_ask - n

                if imbalanced:
                    if position < 0:
                        b_qty = max(1, round(base_qty * heavy_scale))
                        s_qty = max(0, round(base_qty * light_scale))
                    else:
                        b_qty = max(0, round(base_qty * light_scale))
                        s_qty = max(1, round(base_qty * heavy_scale))
                else:
                    b_qty, s_qty = base_qty, base_qty

                # Bid: buying below FV; must not cross best_ask.
                if remaining_buy > 0 and bid_price < skewed_fv and bid_price < best_ask:
                    qty = min(remaining_buy, b_qty)
                    if qty > 0:
                        orders.append(Order(product, bid_price, qty))
                        remaining_buy -= qty

                # Ask: selling above FV; must not cross best_bid.
                if remaining_sell > 0 and ask_price > skewed_fv and ask_price > best_bid:
                    qty = min(remaining_sell, s_qty)
                    if qty > 0:
                        orders.append(Order(product, ask_price, -qty))
                        remaining_sell -= qty

            return orders

        # ── Legacy single-tier path (IPR) ────────────────────────────────
        quote_offset = cfg.get("quote_offset", 0)
        max_qty      = cfg.get("max_passive_qty")

        primary_bid = best_bid - quote_offset   # bid+1 when quote_offset=-1
        primary_ask = best_ask + quote_offset   # ask-1 when quote_offset=-1

        if cfg.get("asymmetric_quoting", False):
            abs_pos  = abs(position)
            moderate = cfg.get("asym_moderate_pos", 40)
            critical = cfg.get("asym_critical_pos", 63)
            asym_off = cfg.get("asym_offset", 3)

            if abs_pos < moderate:
                our_bid = primary_bid
                our_ask = primary_ask

            elif abs_pos <= critical:
                if position > 0:
                    our_bid = primary_bid
                    our_ask = int(base_fv) + asym_off
                else:
                    our_bid = int(base_fv) - asym_off
                    our_ask = primary_ask

            else:
                if position > 0:
                    our_bid = primary_bid
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
                    our_ask = primary_ask

            if not is_wide:
                our_bid = min(our_bid, primary_bid)
                our_ask = max(our_ask, primary_ask)

            our_bid = max(our_bid, best_bid)
            our_ask = min(our_ask, best_ask)
        else:
            our_bid = primary_bid
            our_ask = primary_ask

        if our_bid >= best_ask:
            our_bid = best_bid
        if our_ask <= best_bid:
            our_ask = best_ask

        def resolve_qty(capacity, override, max_q):
            if override is not None:
                return min(capacity, override)
            if max_q is not None:
                return min(capacity, max_q)
            return capacity

        if remaining_buy > 0 and our_bid < skewed_fv:
            qty = resolve_qty(remaining_buy, buy_qty_override, max_qty)
            if qty > 0:
                orders.append(Order(product, our_bid, qty))

        if remaining_sell > 0 and our_ask > skewed_fv:
            qty = resolve_qty(remaining_sell, sell_qty_override, max_qty)
            if qty > 0:
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

            target_buy_to_position = self.target_core_position(
                product, cfg, persistent, state.timestamp,
                cfg.get("buy_to_target_position"),
            )

            base_fv = self.get_base_fv(product, order_depth, persistent, state.timestamp)
            if base_fv is None:
                result[product] = orders
                continue

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

            # Step 1 — aggressive taking (includes buy-to-target ramp for IPR).
            taken, buy_capacity, sell_capacity = self.take_orders(
                product, order_depth, skewed_fv,
                buy_capacity, sell_capacity, position,
                cfg.get("spread_aware_taking", False), is_wide,
                min_long,
                buy_to_target=target_buy_to_position,
                core_build_per_tick=cfg.get("core_build_per_tick"),
                buy_to_ask_max_offset=cfg.get("buy_to_ask_max_offset"),
                timestamp=state.timestamp,
                trend_slope=cfg.get("trend_slope", 0.0),
                total_ticks=100_000,
            )
            orders.extend(taken)

            # Step 2 — passive making.
            # Legacy size override applies only to single-tier products (IPR).
            # Multi-layer products (ACO) handle sizing internally via heavy/light_scale.
            buy_qty_override  = None
            sell_qty_override = None
            if not cfg.get("quote_layers"):
                imb_thresh = cfg.get("size_imbalance_threshold")
                if imb_thresh is not None and abs(position) >= imb_thresh:
                    heavy = cfg.get("size_heavy_qty", 14)
                    light = cfg.get("size_light_qty", 4)
                    if position < 0:
                        buy_qty_override  = heavy
                        sell_qty_override = light
                    else:
                        buy_qty_override  = light
                        sell_qty_override = heavy

            orders.extend(self.make_orders(
                product, order_depth, base_fv, skewed_fv,
                buy_capacity, sell_capacity, position,
                cfg, is_wide, min_long,
                buy_qty_override=buy_qty_override,
                sell_qty_override=sell_qty_override,
            ))

            result[product] = orders

        return result, 0, json.dumps(persistent)
