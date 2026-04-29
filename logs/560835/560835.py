"""
probe5.py — IMC Prosperity 4 Round 5 diagnostic probe.

Low-risk information-gathering trader. Output is a structured traderData JSON
that we parse offline to set production parameters. PnL is intentionally
near zero (-1k..+1k seashells expected over a 10k-tick day).

Six questions (priority order):
  Q1 fill rate by offset N in {1,2,3,5,8,12} for 12 active products
  Q2 PEBBLES live basket behavior (sum=50000, |dev|>=8 tradeability)
  Q3 SNACKPACK CHOC+VAN pair regime drift
  Q4 counterparty IDs populated in live?
  Q5 order-book depth past level 1
  Q6 market-impact-at-limit-10 (next-tick mid move after 1-lot take)

Hard constraints:
  R1 |pos[p]| <= 2 for every product
  R2 <= 12 active orders per tick
  R3 size>1 only in flatten phase 3
  R4 takes only on PEBBLES |basket_dev|>=8 (cd 100t)
                  and SNACKPACK pair |sum-ema|>=80 (cd 200t)
  R5 phase 1 = [0,1000)  phase 2 = [1000,9500)  phase 3 = [9500,10000)
  R6 expected total PnL impact in [-1000,+1000]

Submission format: this file is shipped as the production submission for the
day we run the probe. Imports limited to datamodel + numpy + math + json +
collections (numpy is unused; kept allowable per spec).
"""

from datamodel import Order, OrderDepth, TradingState, Trade
from typing import Dict, List, Tuple
import json
import math
from collections import deque

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

ACTIVE: List[str] = [
    "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
    "SNACKPACK_RASPBERRY", "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",
    "ROBOT_DISHES", "OXYGEN_SHAKE_EVENING_BREATH",
    "GALAXY_SOUNDS_DARK_MATTER", "UV_VISOR_AMBER",
    "SLEEP_POD_NYLON", "PANEL_4X4",
]

PEBBLES: List[str] = [
    "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
]
PEBBLES_TARGET_SUM = 50000
PEBBLES_DEV_THRESHOLD = 8
PEBBLES_COOLDOWN = 100

PAIR_LEGS = ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA")
PAIR_DEV_THRESHOLD = 80
PAIR_COOLDOWN = 200
PAIR_EMA_ALPHA = 0.01  # ema = ema*0.99 + x*0.01

OFFSETS: List[int] = [1, 2, 3, 5, 8, 12]
WINDOW = 50  # ticks per (product, offset) cell in fill-rate probe

POS_CAP = 2
MAX_ORDERS_PER_TICK = 12

PHASE1_END = 1000
PHASE2_END = 9500
RUN_END = 10000

EVENT_LOG_CAP = 500
TRADERDATA_BUDGET = 9000  # bytes


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _best_bid(od: OrderDepth):
    if not od or not od.buy_orders:
        return None, 0
    p = max(od.buy_orders.keys())
    return p, od.buy_orders[p]


def _best_ask(od: OrderDepth):
    if not od or not od.sell_orders:
        return None, 0
    p = min(od.sell_orders.keys())
    return p, od.sell_orders[p]


def _mid(od: OrderDepth):
    bp, _ = _best_bid(od)
    ap, _ = _best_ask(od)
    if bp is None or ap is None:
        return None
    return (bp + ap) / 2.0


def _depth_levels(od: OrderDepth):
    """Return (n_buy_levels, n_sell_levels)."""
    nb = len(od.buy_orders) if od and od.buy_orders else 0
    ns = len(od.sell_orders) if od and od.sell_orders else 0
    return nb, ns


# --------------------------------------------------------------------------
# Trader
# --------------------------------------------------------------------------

class Trader:

    def __init__(self):
        # Fill-rate stats: {product: {offset_str: [n_placed, n_filled, total_pnl]}}
        self.fill_stats: Dict[str, Dict[str, List[float]]] = {}
        # Track outstanding probe orders so we can attribute fills.
        # Map (product, side, offset) -> list of (placed_tick, price)
        self.probe_open: Dict[str, List[List]] = {}

        # Basket arb tracker
        self.basket = {
            "events": 0,
            "traded": 0,
            "pnl": 0.0,
            "open_pos": {},  # product -> qty (signed)
            "open_avg": {},  # product -> avg entry px
            "last_traded_tick": -10000,
        }

        # Pair tracker (CHOC+VAN sum vs EMA)
        self.pair = {
            "events": 0,
            "traded": 0,
            "pnl": 0.0,
            "ema": None,
            "open_pos": {PAIR_LEGS[0]: 0, PAIR_LEGS[1]: 0},
            "open_avg": {PAIR_LEGS[0]: 0.0, PAIR_LEGS[1]: 0.0},
            "last_traded_tick": -10000,
        }

        # Spread observers: {product: [sum_spread, sum_bvol, sum_avol, n]}
        self.spread_stats: Dict[str, List[float]] = {}

        # Depth observers: {product: [n_with_l2, n_with_l3, n_total]}
        self.depth: Dict[str, List[int]] = {}

        # Counterparty observation
        self.cp = {
            "buyers": [],   # list of unique strings (kept small)
            "sellers": [],
            "n_pop": 0,
            "n_total": 0,
        }

        # Compact event log (FIFO bounded)
        self.event_log: List[list] = []

        # Round-robin probe schedule indices
        self.probe_window_idx = 0  # advances every WINDOW ticks of phase 2

        # Track last own-trade times we've consumed (avoid double-counting fills)
        self._last_seen_own = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _restore(self, td: str):
        try:
            d = json.loads(td)
        except Exception:
            return
        self.fill_stats = d.get("fill_stats", {}) or {}
        self.basket["events"] = d.get("basket_events", 0)
        self.basket["traded"] = d.get("basket_traded", 0)
        self.basket["pnl"] = d.get("basket_pnl", 0.0)
        self.pair["events"] = d.get("pair_events", 0)
        self.pair["traded"] = d.get("pair_traded", 0)
        self.pair["pnl"] = d.get("pair_pnl", 0.0)
        ema = d.get("pair_ema")
        if ema is not None:
            self.pair["ema"] = float(ema)
        self.spread_stats = d.get("spread_raw", {}) or {}
        self.depth = d.get("depth_observed", {}) or {}
        cp = d.get("buyer_seller_seen") or {}
        self.cp["buyers"] = list(cp.get("buyers", []))
        self.cp["sellers"] = list(cp.get("sellers", []))
        self.cp["n_pop"] = cp.get("n_pop_trades", 0)
        self.cp["n_total"] = cp.get("n_total_trades", 0)
        self.event_log = d.get("event_log", []) or []

    def _encode(self, tick: int, phase: int) -> str:
        # Compact spread output: round to 2 dp.
        spread_out = {}
        for p, agg in self.spread_stats.items():
            n = agg[3] if agg[3] > 0 else 1
            spread_out[p] = [
                round(agg[0] / n, 2),
                round(agg[1] / n, 2),
                round(agg[2] / n, 2),
            ]

        ema_val = self.pair["ema"]
        payload = {
            "tick": tick,
            "phase": phase,
            "fill_stats": self.fill_stats,
            "basket_events": self.basket["events"],
            "basket_traded": self.basket["traded"],
            "basket_pnl": round(self.basket["pnl"], 2),
            "pair_events": self.pair["events"],
            "pair_traded": self.pair["traded"],
            "pair_pnl": round(self.pair["pnl"], 2),
            "pair_ema": round(ema_val, 2) if ema_val is not None else None,
            "spread_stats": spread_out,
            # Keep raw so cross-day restore preserves running averages.
            "spread_raw": self.spread_stats,
            "buyer_seller_seen": {
                "buyers": self.cp["buyers"][:50],
                "sellers": self.cp["sellers"][:50],
                "n_pop_trades": self.cp["n_pop"],
                "n_total_trades": self.cp["n_total"],
            },
            "depth_observed": self.depth,
            "event_log": self.event_log[-EVENT_LOG_CAP:],
        }
        s = json.dumps(payload, separators=(",", ":"))
        if len(s) > TRADERDATA_BUDGET:
            payload["event_log"] = []
            s = json.dumps(payload, separators=(",", ":"))
        if len(s) > TRADERDATA_BUDGET:
            payload["spread_raw"] = {}
            s = json.dumps(payload, separators=(",", ":"))
        if len(s) > TRADERDATA_BUDGET:
            payload["spread_stats"] = {}
            payload["depth_observed"] = {}
            s = json.dumps(payload, separators=(",", ":"))
        return s

    # ------------------------------------------------------------------
    # Observation (cheap; runs every tick)
    # ------------------------------------------------------------------

    def _observe(self, state: TradingState):
        for prod, od in state.order_depths.items():
            bp, bv = _best_bid(od)
            ap, av = _best_ask(od)
            if bp is None or ap is None:
                continue
            agg = self.spread_stats.setdefault(prod, [0.0, 0.0, 0.0, 0])
            agg[0] += (ap - bp)
            agg[1] += abs(bv)
            agg[2] += abs(av)
            agg[3] += 1
            nb, ns = _depth_levels(od)
            d = self.depth.setdefault(prod, [0, 0, 0])
            if nb >= 2 or ns >= 2:
                d[0] += 1
            if nb >= 3 or ns >= 3:
                d[1] += 1
            d[2] += 1

        # Counterparty observation across ALL products' market trades.
        for prod, trades in (state.market_trades or {}).items():
            for t in trades:
                self.cp["n_total"] += 1
                buyer = getattr(t, "buyer", None) or ""
                seller = getattr(t, "seller", None) or ""
                if buyer or seller:
                    self.cp["n_pop"] += 1
                if buyer and buyer not in self.cp["buyers"]:
                    if len(self.cp["buyers"]) < 100:
                        self.cp["buyers"].append(buyer)
                if seller and seller not in self.cp["sellers"]:
                    if len(self.cp["sellers"]) < 100:
                        self.cp["sellers"].append(seller)

    # ------------------------------------------------------------------
    # Pair EMA update — runs every tick; used by pair-trade probe in phase 2
    # ------------------------------------------------------------------

    def _update_pair_ema(self, state: TradingState):
        c_od = state.order_depths.get(PAIR_LEGS[0])
        v_od = state.order_depths.get(PAIR_LEGS[1])
        if c_od is None or v_od is None:
            return None
        cm = _mid(c_od)
        vm = _mid(v_od)
        if cm is None or vm is None:
            return None
        s = cm + vm
        if self.pair["ema"] is None:
            self.pair["ema"] = s
        else:
            self.pair["ema"] = self.pair["ema"] * (1.0 - PAIR_EMA_ALPHA) + s * PAIR_EMA_ALPHA
        return s

    # ------------------------------------------------------------------
    # Own-trade ingestion: attribute fills to either a fill-probe quote, or
    # to an open basket/pair position. Update PnL accordingly.
    # ------------------------------------------------------------------

    def _ingest_own_trades(self, state: TradingState, tick: int):
        for prod, trades in (state.own_trades or {}).items():
            for t in trades:
                ts = getattr(t, "timestamp", 0)
                # Skip already-seen prior-tick trades.
                if ts < self._last_seen_own.get(prod, -1):
                    continue
                qty = t.quantity
                if qty == 0:
                    continue
                side = "BUY" if qty > 0 else "SELL"
                self._attribute_fill(prod, side, abs(qty), t.price, tick)
            if trades:
                self._last_seen_own[prod] = max(getattr(x, "timestamp", 0) for x in trades)

    def _attribute_fill(self, prod: str, side: str, qty: int, price: float, tick: int):
        # 1. Try to match a basket leg position
        if prod in PEBBLES and prod in self.basket["open_pos"]:
            cur = self.basket["open_pos"][prod]
            avg = self.basket["open_avg"].get(prod, 0.0)
            signed = qty if side == "BUY" else -qty
            new = cur + signed
            # If reducing toward zero, realize PnL on the closed portion.
            if cur != 0 and (cur > 0) != (signed > 0):
                close_qty = min(abs(cur), abs(signed))
                pnl = (price - avg) * close_qty if cur > 0 else (avg - price) * close_qty
                self.basket["pnl"] += pnl
                if cur > 0:
                    new = cur - close_qty if signed < 0 else cur
                else:
                    new = cur + close_qty if signed > 0 else cur
                # Append a settlement row to event log
                self._log_event(["BASKET_CLOSE", tick, prod, round(price, 2), round(pnl, 2)])
            else:
                # Same direction — average up.
                if abs(cur + signed) > 0:
                    avg = (avg * abs(cur) + price * qty) / abs(cur + signed)
                self.basket["open_avg"][prod] = avg
                new = cur + signed
            if new == 0:
                self.basket["open_pos"].pop(prod, None)
                self.basket["open_avg"].pop(prod, None)
            else:
                self.basket["open_pos"][prod] = new
            return

        # 2. Try to match a pair leg position
        if prod in PAIR_LEGS:
            cur = self.pair["open_pos"].get(prod, 0)
            avg = self.pair["open_avg"].get(prod, 0.0)
            signed = qty if side == "BUY" else -qty
            if cur != 0 and (cur > 0) != (signed > 0):
                close_qty = min(abs(cur), abs(signed))
                pnl = (price - avg) * close_qty if cur > 0 else (avg - price) * close_qty
                self.pair["pnl"] += pnl
                if cur > 0:
                    new = cur - close_qty if signed < 0 else cur
                else:
                    new = cur + close_qty if signed > 0 else cur
                self._log_event(["PAIR_CLOSE", tick, prod, round(price, 2), round(pnl, 2)])
            else:
                new = cur + signed
                if abs(new) > 0:
                    avg = (avg * abs(cur) + price * qty) / abs(new)
                self.pair["open_avg"][prod] = avg
            self.pair["open_pos"][prod] = new
            return

        # 3. Otherwise, attribute to a fill-rate probe (best effort: match by side+product)
        self._attribute_probe_fill(prod, side, qty, price, tick)

    def _attribute_probe_fill(self, prod: str, side: str, qty: int, price: float, tick: int):
        # We tagged outstanding probe orders by (prod, side, offset).
        # Find the offset whose recorded price matches; mark filled.
        opens = self.probe_open.get(prod, [])
        for entry in opens:
            # entry = [side, offset, price, placed_tick]
            if entry[0] != side:
                continue
            if int(entry[2]) != int(price):
                continue
            offset = entry[1]
            stats = self.fill_stats.setdefault(prod, {})
            cell = stats.setdefault(str(offset), [0, 0, 0.0])
            cell[1] += qty
            # PnL contribution: we don't immediately close this leg, but we can
            # mark notional edge vs a reference. For now, record signed price
            # so the offline parser can compute realized edge from the price
            # series. Use a simple proxy: edge per fill = offset (the gap from
            # mid we quoted at). This is the *target* edge before adverse
            # selection; realized edge requires the next-tick mid which we
            # also log.
            cell[2] += float(offset) * qty
            self._log_event(["FILL", tick, prod, side, offset, int(price)])
            entry[2] = None  # mark consumed
            return
        # Unattributed fill (e.g. flatten phase): no fill-probe match.

    # ------------------------------------------------------------------
    # Phase 2(A) — fill-rate probe
    # ------------------------------------------------------------------

    def _fill_rate_probe(self, state: TradingState, tick: int, orders: Dict[str, List[Order]]):
        # Each (product, offset) cell holds the floor for WINDOW ticks.
        # Total cells = 12 * 6 = 72 → cycle length = 72 windows = 3600 ticks.
        # Phase 2 length = 8500 ticks → 2.36 cycles → ~2 visits per cell.
        elapsed = tick - PHASE1_END
        window_idx = elapsed // WINDOW
        prod_idx = window_idx % len(ACTIVE)
        offset_idx = (window_idx // len(ACTIVE)) % len(OFFSETS)
        prod = ACTIVE[prod_idx]
        offset = OFFSETS[offset_idx]

        od = state.order_depths.get(prod)
        if od is None:
            return
        m = _mid(od)
        if m is None:
            return
        bp, _ = _best_bid(od)
        ap, _ = _best_ask(od)

        pos = state.position.get(prod, 0)
        product_orders = orders.setdefault(prod, [])
        stats = self.fill_stats.setdefault(prod, {})
        cell = stats.setdefault(str(offset), [0, 0, 0.0])
        opens = self.probe_open.setdefault(prod, [])

        bid_px = int(math.floor(m - offset))
        ask_px = int(math.ceil(m + offset))

        # Don't sit inside the wrong side of the book.
        if ap is not None and bid_px >= ap:
            bid_px = None
        if bp is not None and ask_px <= bp:
            ask_px = None

        # Respect position cap (R1) and per-tick order budget (R2).
        if bid_px is not None and pos < POS_CAP and self._order_count(orders) < MAX_ORDERS_PER_TICK:
            product_orders.append(Order(prod, bid_px, 1))
            cell[0] += 1
            opens.append(["BUY", offset, bid_px, tick])
            self._log_event(["MISS_OR_FILL_PEND", tick, prod, "BUY", offset, bid_px])
        if ask_px is not None and pos > -POS_CAP and self._order_count(orders) < MAX_ORDERS_PER_TICK:
            product_orders.append(Order(prod, ask_px, -1))
            cell[0] += 1
            opens.append(["SELL", offset, ask_px, tick])
            self._log_event(["MISS_OR_FILL_PEND", tick, prod, "SELL", offset, ask_px])

        # Garbage-collect old open entries (older than 2*WINDOW ticks → never filled).
        cutoff = tick - 2 * WINDOW
        if opens:
            self.probe_open[prod] = [e for e in opens if e[2] is not None and e[3] >= cutoff]

    # ------------------------------------------------------------------
    # Phase 2(B) — basket arb sampling
    # ------------------------------------------------------------------

    def _basket_arb_probe(self, state: TradingState, tick: int, orders: Dict[str, List[Order]]):
        mids = {}
        for p in PEBBLES:
            od = state.order_depths.get(p)
            if od is None:
                return
            mm = _mid(od)
            if mm is None:
                return
            mids[p] = mm
        basket_sum = sum(mids.values())
        dev = basket_sum - PEBBLES_TARGET_SUM
        if abs(dev) < PEBBLES_DEV_THRESHOLD:
            return
        self.basket["events"] += 1
        if tick - self.basket["last_traded_tick"] < PEBBLES_COOLDOWN:
            return
        # If sum > 50000, basket is rich → SELL the leg whose bid is most above
        # implied fair (= leg_mid - dev/5). If sum < 50000, BUY the leg whose
        # ask is most below implied fair.
        share = dev / 5.0
        best_leg = None
        best_score = 0.0
        best_px = None
        for p in PEBBLES:
            od = state.order_depths[p]
            implied_fair = mids[p] - share
            if dev > 0:
                bp, _ = _best_bid(od)
                if bp is None:
                    continue
                edge = bp - implied_fair
                if edge > best_score:
                    best_score = edge
                    best_leg = p
                    best_px = bp
            else:
                ap, _ = _best_ask(od)
                if ap is None:
                    continue
                edge = implied_fair - ap
                if edge > best_score:
                    best_score = edge
                    best_leg = p
                    best_px = ap
        if best_leg is None or best_px is None:
            return
        pos = state.position.get(best_leg, 0)
        if dev > 0 and pos <= -POS_CAP:
            return
        if dev < 0 and pos >= POS_CAP:
            return
        if self._order_count(orders) >= MAX_ORDERS_PER_TICK:
            return
        side_qty = -1 if dev > 0 else 1
        orders.setdefault(best_leg, []).append(Order(best_leg, int(best_px), side_qty))
        self.basket["traded"] += 1
        self.basket["last_traded_tick"] = tick
        # Pre-register the open position; _ingest_own_trades will refine.
        self.basket["open_pos"][best_leg] = self.basket["open_pos"].get(best_leg, 0) + side_qty
        self.basket["open_avg"][best_leg] = float(best_px)
        self._log_event(["BASKET", tick, round(dev, 2), best_leg, int(best_px)])

    # ------------------------------------------------------------------
    # Phase 2(B) — pair trade sampling (CHOC + VAN)
    # ------------------------------------------------------------------

    def _pair_trade_probe(self, state: TradingState, tick: int, orders: Dict[str, List[Order]]):
        if self.pair["ema"] is None:
            return
        c_od = state.order_depths.get(PAIR_LEGS[0])
        v_od = state.order_depths.get(PAIR_LEGS[1])
        if c_od is None or v_od is None:
            return
        cm = _mid(c_od)
        vm = _mid(v_od)
        if cm is None or vm is None:
            return
        s = cm + vm
        dev = s - self.pair["ema"]
        if abs(dev) < PAIR_DEV_THRESHOLD:
            return
        self.pair["events"] += 1
        if tick - self.pair["last_traded_tick"] < PAIR_COOLDOWN:
            return

        # Direction: dev > 0 → sell both; dev < 0 → buy both.
        if dev > 0:
            c_px, _ = _best_bid(c_od)
            v_px, _ = _best_bid(v_od)
            qty = -1
        else:
            c_px, _ = _best_ask(c_od)
            v_px, _ = _best_ask(v_od)
            qty = 1
        if c_px is None or v_px is None:
            return
        cpos = state.position.get(PAIR_LEGS[0], 0)
        vpos = state.position.get(PAIR_LEGS[1], 0)
        if qty > 0 and (cpos >= POS_CAP or vpos >= POS_CAP):
            return
        if qty < 0 and (cpos <= -POS_CAP or vpos <= -POS_CAP):
            return
        if self._order_count(orders) + 2 > MAX_ORDERS_PER_TICK:
            return

        orders.setdefault(PAIR_LEGS[0], []).append(Order(PAIR_LEGS[0], int(c_px), qty))
        orders.setdefault(PAIR_LEGS[1], []).append(Order(PAIR_LEGS[1], int(v_px), qty))
        self.pair["traded"] += 1
        self.pair["last_traded_tick"] = tick
        for p, px in ((PAIR_LEGS[0], c_px), (PAIR_LEGS[1], v_px)):
            self.pair["open_pos"][p] = self.pair["open_pos"].get(p, 0) + qty
            self.pair["open_avg"][p] = float(px)
        self._log_event(["PAIR", tick, round(dev, 2), "SHORT" if qty < 0 else "LONG",
                         int(c_px), int(v_px)])

    # ------------------------------------------------------------------
    # Phase 3 — flatten
    # ------------------------------------------------------------------

    def _flatten(self, state: TradingState, orders: Dict[str, List[Order]]):
        for prod, pos in state.position.items():
            if pos == 0:
                continue
            od = state.order_depths.get(prod)
            if od is None:
                continue
            if pos > 0:
                bp, _ = _best_bid(od)
                if bp is None:
                    continue
                # Cross by 1 contract per tick (R3 allows size>1 here, but
                # 1/tick is gentler on the book and still flattens within 500t
                # for any |pos|<=2 we permit ourselves to hold).
                qty = min(pos, 1)
                orders.setdefault(prod, []).append(Order(prod, int(bp), -qty))
            else:
                ap, _ = _best_ask(od)
                if ap is None:
                    continue
                qty = min(-pos, 1)
                orders.setdefault(prod, []).append(Order(prod, int(ap), qty))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _order_count(self, orders: Dict[str, List[Order]]) -> int:
        return sum(len(v) for v in orders.values())

    def _log_event(self, ev: list):
        self.event_log.append(ev)
        if len(self.event_log) > EVENT_LOG_CAP:
            # FIFO eviction
            self.event_log = self.event_log[-EVENT_LOG_CAP:]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, state: TradingState):
        if state.traderData:
            self._restore(state.traderData)

        tick = state.timestamp
        orders: Dict[str, List[Order]] = {}

        # 1. Cheap observers (every tick, every phase).
        self._observe(state)
        self._update_pair_ema(state)
        self._ingest_own_trades(state, tick)

        # 2. Phase branch.
        if tick < PHASE1_END:
            phase = 1
        elif tick < PHASE2_END:
            phase = 2
            self._fill_rate_probe(state, tick, orders)
            self._basket_arb_probe(state, tick, orders)
            self._pair_trade_probe(state, tick, orders)
        else:
            phase = 3
            self._flatten(state, orders)

        # 3. Hard safety net: enforce |pos+pending|<=POS_CAP after orders built.
        orders = self._enforce_pos_cap(state, orders)

        # 4. Encode traderData.
        td = self._encode(tick, phase)

        return orders, 0, td

    def _enforce_pos_cap(self, state: TradingState, orders: Dict[str, List[Order]]):
        out: Dict[str, List[Order]] = {}
        for prod, ords in orders.items():
            pos = state.position.get(prod, 0)
            running = pos
            kept: List[Order] = []
            for o in ords:
                new = running + o.quantity
                # Allow overshoot only in flatten phase (handled by caller logic).
                if abs(new) > POS_CAP and not (state.timestamp >= PHASE2_END):
                    continue
                kept.append(o)
                running = new
            if kept:
                out[prod] = kept
        return out