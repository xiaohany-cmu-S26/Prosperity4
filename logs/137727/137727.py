from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import json
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  PRODUCT CONFIGURATION  ─  Manual Tuning Guide
# ══════════════════════════════════════════════════════════════════════════════
#
# This file is parameterised so a trader with a finance background can adjust
# strategy behaviour without touching any code.  Read the explanations below,
# then edit only the numbers in PRODUCT_CONFIG.
#
# ─── WHAT EACH PRODUCT DOES ───────────────────────────────────────────────────
#
#  INTARIAN_PEPPER_ROOT  ("PEPPER")
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │ Price drifts upward at a perfectly-linear rate (~+1 tick per 1 000 ts). │
#  │ Like a bond that accrues value.  The correct strategy is to hold the    │
#  │ maximum long position for as long as possible.                          │
#  │                                                                         │
#  │ Two phases:                                                              │
#  │  1. RAMP-UP  (pos < buy_to_target_position)                             │
#  │     Aggressively BUY at the ask even when ask > FV, because the future  │
#  │     trend gain outweighs the spread cost.  Break-even: pay up to        │
#  │     trend_slope × remaining_time ticks — which is ~80 ticks at ts=0    │
#  │     and still ~8 ticks at ts=92 000.  Use buy_to_ask_max_offset to      │
#  │     cap how much above FV we'll chase the ask.                          │
#  │                                                                         │
#  │  2. HOLD / LIGHT-MM  (pos ≥ min_long_position)                          │
#  │     Post passive asks above min_long_position and recycle the spread.   │
#  │     Never let position fall below min_long_position.                    │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  ASH_COATED_OSMIUM  ("ASH")
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │ Pure mean-reversion around 10 000.  Lag-1 autocorrelation ≈ -0.49:     │
#  │ whatever direction it moved last tick, it likely reverts next tick.     │
#  │                                                                         │
#  │ The structural market-maker quotes 9 993 / 10 009 (spread = 16 ticks,   │
#  │ mid ≈ 10 001).  Our passive bid sits 1 tick inside their bid and our    │
#  │ ask 1 tick inside their ask via quote_offset = -1.                      │
#  │                                                                         │
#  │ Inventory risk is handled in two layers:                                │
#  │  • Tiered FV skew  → shifts the "fair value" used for taking/making     │
#  │    as we accumulate inventory, discouraging us from adding to a large   │
#  │    position. (Think: DV01 hedging — the more we hold, the lower our     │
#  │    willingness to add.)                                                 │
#  │  • Asymmetric quoting  → when we exceed moderate_pos, the quote on the  │
#  │    unwinding side moves aggressively toward FV to attract counterflows. │
#  │  • Inventory-aware sizing  → posts more lots on the depleted side to    │
#  │    rebalance faster (like increasing the hedge ratio when off-sides).   │
#  └─────────────────────────────────────────────────────────────────────────┘
#
# ─── TUNING KNOBS ─────────────────────────────────────────────────────────────
#
#  PEPPER — key levers (highest impact first):
#
#  buy_to_target_position  [int, 0–80]
#    Target long position during the ramp-up phase.  Once reached we stop
#    paying the premium to lift the ask.
#    ↑ Higher → build larger position faster (more trend alpha, more spread cost)
#    Recommended: 70.  Max: 80 (but hard to fill immediately at limit).
#
#  buy_to_ask_max_offset   [int, ticks above FV we'll accept]
#    How far above fair value we're willing to chase the ask during ramp-up.
#    Economics: worth paying up to  trend_slope × remaining_time  ticks.
#    At ts=0 that's 0.001 × 100 000 = 100.  At ts=87 000 it's 13.
#    Safe value: 13 (= half-spread) → always profitable before ts=87 000.
#    ↑ Higher → fill faster, higher spread cost per unit.
#    Recommended: 13.
#
#  min_long_position       [int, 0–79]
#    Hard floor: never post a passive ask or take a bid that would push us
#    below this position.  Set equal to buy_to_target_position for simplest
#    behaviour (stay at target forever).  Keep 5–10 below buy_to_target so
#    we still MM a small band above the floor.
#    ↑ Higher → more trend capture, less MM spread income.
#    Recommended: 65.
#
#  asym_moderate_pos / asym_critical_pos   [int]
#    Inventory thresholds where the passive ask moves aggressively toward FV.
#    Set relative to min_long_position.  E.g. moderate = min_long + 5,
#    critical = min_long + 15.  Only relevant above min_long.
#    Recommended: 70 / 78.
#
#  trend_slope             [float, ticks per timestamp]
#    Estimated drift of the price per unit of time.  Empirically 0.001000.
#    Do NOT change unless you have new data from a different simulation seed.
#
#  ASH — key levers (highest impact first):
#
#  asym_moderate_pos       [int, 1–79]
#    Position beyond which the passive bid/ask tilts toward FV to unwind.
#    Lower → triggers earlier, faster mean-reversion to flat.
#    Previous value 30 was too high; ASH spends 14.8% of time at –15 to –20
#    without the asymmetric logic ever engaging.
#    Recommended: 10.  (Think: start hedging at 12.5% of limit.)
#
#  asym_critical_pos       [int, > asym_moderate_pos]
#    Emergency unwind level.  Quote on the unwinding side moves to FV±1.
#    Recommended: 25.
#
#  asym_offset             [int, ticks]
#    How far from FV the moderate passive quote is placed on the unwind side.
#    Smaller → more aggressive unwind, smaller spread earned on that side.
#    Recommended: 2.
#
#  size_imbalance_threshold [int]
#    When |position| ≥ this, switch to asymmetric lot sizing.
#    Set slightly above asym_moderate_pos to avoid premature size changes.
#    Recommended: 12.
#
#  size_heavy_qty / size_light_qty   [int, lots]
#    When imbalanced, post size_heavy_qty on the depleted side and
#    size_light_qty on the full side.  Together they must respect the
#    position limit and realistic fill rates.
#    Recommended: heavy=14, light=3.
#
#  ema_alpha               [float, 0–1]
#    EMA smoothing for the "midwall" fair-value estimator.
#    Lower → FV tracks the book mid more slowly (less noise, more lag).
#    0.05 = 20-tick half-life; 0.10 = 9-tick half-life.
#    Recommended: 0.05.
#
#  skew_tiers              [list of (threshold, factor)]
#    Inventory skew for the FV used in taking: skewed_fv = base_fv − factor × pos.
#    factor=0.08 at 80 units → shifts FV by 6.4 ticks (≈ half the bid-ask spread).
#    Think of it as a risk-limit mechanism: the bigger the book, the wider the
#    "acceptable" price to unwind.
#    To disable entirely: [(80, 0.0)].

PRODUCT_CONFIG = {
    "INTARIAN_PEPPER_ROOT": {
        "limit": 80,

        # ── Fair-value model ──────────────────────────────────────────────
        # Perfectly-linear uptrend confirmed empirically (R²≈1, slope≈0.001/ts).
        "fair_value":   "linear_trend",
        "trend_slope":  0.001,          # ticks per timestamp — do not change

        # Spread is typically 13 ticks → use L2 microprice to reduce noise.
        "use_l2_microprice":      True,
        "wide_spread_threshold":  10,

        "use_factors":    False,
        "factor_fv_skew": 0.0,

        # ── Ramp-up phase: aggressive buy-to-target ───────────────────────
        # During ramp-up we lift the ask even when it's above FV, because the
        # expected trend gain (slope × remaining_time) exceeds the spread cost.
        # Economics: worth paying up to ~13 ticks extra until ts ≈ 87 000.
        # Key finding: position was only reaching ~60 at ts=47700 without this.
        # With this fix, we expect pos=70 by ts≈5000, gaining ~2400 extra PnL.
        "buy_to_target_position": 70,   # lift ask until we hold this many units
        "buy_to_ask_max_offset":  13,   # max ticks above FV we'll chase the ask

        # ── Hold phase: protect the long position ────────────────────────
        # Never allow selling below this position.
        # Raised from 60 → 65 to hold 5 more units on average.
        "min_long_position": 65,

        # no inventory skew needed (trend is monotone)
        "skew_tiers": [(80, 0.0)],

        # Asymmetric quoting above min_long: sell gently at FV + small offset.
        # Only activates when pos > asym_moderate_pos (= 70 here), i.e. 5 units
        # above the floor.  This captures the bid-ask spread on the marginal units.
        "asymmetric_quoting":  True,
        "asym_moderate_pos":   70,      # = min_long + 5
        "asym_critical_pos":   78,      # ≈ limit − 2
        "asym_offset":         2,

        "quote_offset": -1,          # post 1 tick inside best bid/ask
        "max_passive_qty": None,        # no per-order cap during MM phase

        "spread_aware_taking": False,   # always take cross-FV; we want exposure
    },

    "ASH_COATED_OSMIUM": {
        "limit": 80,

        # ── Fair-value model ──────────────────────────────────────────────
        # Structural MM quotes 9993/10009 (mid≈10001, not 10000).  EMA of book
        # mid tracks this correctly and avoids the fixed-10000 anchor bias.
        "fair_value":  "midwall",
        "ema_alpha":   0.05,            # slow EMA; ~20-tick half-life

        "use_l2_microprice":     False,  # L2 blows up on compressed spread
        "wide_spread_threshold": 10,

        "use_factors":    False,
        "factor_fv_skew": 0.0,

        # ── Inventory skew (for taking decisions) ────────────────────────
        # Shifts the FV used in take_orders so we pay less to add when large.
        # Like a delta-hedge: the deeper in the book, the wider the required edge.
        "skew_tiers": [
            (10, 0.0),   # |pos| <  10 → no skew (pure MM)
            (35, 0.08),  # |pos| <  35 → light skew
            (80, 0.25),  # |pos| ≥  35 → aggressive skew
        ],

        # ── Asymmetric quoting ────────────────────────────────────────────
        # BUG FIX (vs 136519): fv_spread was overriding this block.  Now that
        # fv_spread is removed, this logic actually runs.
        #
        # KEY FIX (vs 137265): threshold lowered from 30 → 10.
        # ASH spends 14.8% of time at pos=−15 to −20, where the old threshold
        # never triggered.  Lowering to 10 moves our passive bid from 9994
        # (= best_bid+1) to 9999 (= FV−2) at |pos|=15, which is 5 ticks more
        # attractive to sellers and allows much faster short recovery.
        "asymmetric_quoting":  True,
        # start tilting at ±10 units (12.5% of limit)
        "asym_moderate_pos":   10,
        "asym_critical_pos":   25,      # emergency unwind at ±25 units
        "asym_offset":         2,       # ticks from FV for moderate quote

        # quote_offset=-1: our passive bid/ask sit 1 tick inside the structural
        # MM spread (9993/10009), making us the best price in the book.
        "quote_offset": -1,

        # ── Inventory-aware lot sizing ────────────────────────────────────
        # When we're unbalanced, post more on the side we want filled.
        # Like increasing a hedge ratio when a position grows: heavier on the
        # side that reduces exposure, lighter on the side that adds to it.
        "max_passive_qty":          8,  # lots per side when balanced
        "size_imbalance_threshold": 12,  # switch to asymmetric sizing above this
        "size_heavy_qty":          14,  # lots on depleted (want-to-fill) side
        "size_light_qty":           3,  # lots on over-full side

        "spread_aware_taking": False,   # mean reversion → always take cross-FV

        # Pepper-style buy-to-target not needed for ASH (no trend).
        "buy_to_target_position": None,
        "buy_to_ask_max_offset":  None,
    },
}

DEFAULT_LIMIT = 80


# ══════════════════════════════════════════════════════════════════════════════
#  TRADER  (no user-tunable parameters below this line)
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
            base = ema_state[init_fv_key] + slope * \
                (timestamp - ema_state[init_ts_key])

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
        buy_to_ask_max_offset: Optional[float],
        timestamp: int,
        trend_slope: float,
        total_ticks: int,
    ) -> Tuple[List[Order], int, int]:
        """
        Take asks below FV and bids above FV.

        buy_to_target / buy_to_ask_max_offset:
          When position < buy_to_target, we also lift asks that are above FV
          but within buy_to_ask_max_offset ticks of FV.  This lets us build a
          trend position by paying up to the break-even spread.
        """
        orders: List[Order] = []

        # ── Buy-to-target: lift asks even above FV during ramp-up ──────────
        if buy_to_target is not None and position < buy_to_target and buy_capacity > 0:
            # Dynamic offset: how much above FV can we afford to pay?
            # Max cost we'll accept = slope × remaining_time (break-even).
            remaining = max(0, total_ticks - timestamp)
            dynamic_max = trend_slope * remaining  # e.g. 0.001*80000 = 80 at ts=20000
            effective_offset = min(
                buy_to_ask_max_offset, dynamic_max) if buy_to_ask_max_offset else dynamic_max
            buy_threshold = fv + effective_offset
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price > buy_threshold or buy_capacity <= 0:
                    break
                # Only take as many as needed to reach target.
                remaining_to_target = buy_to_target - position
                vol = abs(order_depth.sell_orders[ask_price])
                qty = min(buy_capacity, vol, remaining_to_target)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_capacity -= qty
            return orders, buy_capacity, sell_capacity

        # ── Standard taking path ───────────────────────────────────────────
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
                    max_sell = max(0, position - min_long)
                    vol = order_depth.buy_orders[bid_price]
                    qty = min(sell_capacity, vol, max_sell)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        sell_capacity -= qty
            return orders, buy_capacity, sell_capacity

        # Unconditional taking.
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
        buy_qty_override: Optional[int] = None,
        sell_qty_override: Optional[int] = None,
    ) -> List[Order]:
        orders: List[Order] = []
        quote_offset = cfg.get("quote_offset", 0)
        max_qty = cfg.get("max_passive_qty")

        best_bid = max(order_depth.buy_orders.keys()
                       ) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()
                       ) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return orders

        conservative_bid = best_bid - quote_offset
        conservative_ask = best_ask + quote_offset

        if cfg.get("asymmetric_quoting", False):
            abs_pos = abs(position)
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

        # Never cross the book.
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

        if buy_capacity > 0 and our_bid < skewed_fv:
            qty = resolve_qty(buy_capacity, buy_qty_override, max_qty)
            if qty > 0:
                orders.append(Order(product, our_bid, qty))

        # Passive asks respect the min_long floor.
        max_sell_passive = max(
            0, position - min_long) if min_long > 0 else sell_capacity
        effective_sell_cap = min(sell_capacity, max_sell_passive)
        if effective_sell_cap > 0 and our_ask > skewed_fv:
            qty = resolve_qty(effective_sell_cap, sell_qty_override, max_qty)
            if qty > 0:
                orders.append(Order(product, our_ask, -qty))

        return orders

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self, state: TradingState):
        try:
            persistent: Dict = json.loads(
                state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            persistent = {}

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            cfg = PRODUCT_CONFIG.get(product, {})

            limit = cfg.get("limit", DEFAULT_LIMIT)
            tiers = cfg.get("skew_tiers", [(DEFAULT_LIMIT, 0.0)])
            position = state.position.get(product, 0)
            min_long = cfg.get("min_long_position", 0)

            base_fv = self.get_base_fv(
                product, order_depth, persistent, state.timestamp)
            if base_fv is None:
                result[product] = orders
                continue

            sf = self.skew_factor(position, tiers)
            skewed_fv = base_fv - sf * position

            buy_capacity = limit - position
            sell_capacity = limit + position

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid
            is_wide = spread >= cfg.get("wide_spread_threshold", 10)

            # Step 1 — aggressive taking (includes buy-to-target for PEPPER).
            taken, buy_capacity, sell_capacity = self.take_orders(
                product, order_depth, skewed_fv,
                buy_capacity, sell_capacity, position,
                cfg.get("spread_aware_taking", False), is_wide,
                min_long,
                buy_to_target=cfg.get("buy_to_target_position"),
                buy_to_ask_max_offset=cfg.get("buy_to_ask_max_offset"),
                timestamp=state.timestamp,
                trend_slope=cfg.get("trend_slope", 0.0),
                total_ticks=100_000,
            )
            orders.extend(taken)

            # Step 2 — passive making with inventory-aware sizing for ASH.
            buy_qty_override = None
            sell_qty_override = None
            imb_thresh = cfg.get("size_imbalance_threshold")
            if imb_thresh is not None and abs(position) >= imb_thresh:
                heavy = cfg.get("size_heavy_qty", 14)
                light = cfg.get("size_light_qty", 4)
                if position < 0:
                    buy_qty_override = heavy
                    sell_qty_override = light
                else:
                    buy_qty_override = light
                    sell_qty_override = heavy

            orders.extend(self.make_orders(
                product, order_depth, base_fv, skewed_fv,
                buy_capacity, sell_capacity, position, cfg, is_wide,
                min_long,
                buy_qty_override=buy_qty_override,
                sell_qty_override=sell_qty_override,
            ))

            result[product] = orders

        return result, 0, json.dumps(persistent)