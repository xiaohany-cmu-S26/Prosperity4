"""
probe_r4.py — Mark-bot behavioral fingerprinting probe for Round 4.

KEY INTEL FROM OFFICIAL TOURNAMENT HINT (Rook-E1):
  "Your counterparties have been identified. All of them carry the name Mark."

All bots in round 4 are named "Mark" (or Mark_N variants). Individual
name-based fingerprinting therefore tells us NOTHING about which bot is
acting. Every entry will say buyer="Mark" or seller="Mark".

Instead, we fingerprint by BEHAVIOR:
  - Size pattern (tiny/small/mid/large per product)
  - Side bias (buy-heavy vs sell-heavy vs balanced)
  - Timing regularity (every N iterations? triggered by price level?)
  - Price discipline (always at touch? crosses spread? mid-prints?)
  - Product specialization (HP-only? options-only? chain-wide?)
  - Forward-return alpha (informed = price moves their way after they trade)

Rook's three Mark archetypes:
  1. MARKET MAKER — rhythmic, both sides, spread-collecting, high frequency
  2. LIQUIDITY TAKER — opportunistic, directional, triggered by price level
  3. LARGE PARTICIPANT — less frequent, big size, movements foreshadow price

We probe to reveal which archetype is present on each product.

Round 4 also hints at CHOOSER OPTIONS — exotic instruments where we choose
call or put before a deadline. This probe also logs any unrecognized
products that appear, so we can identify them immediately.

Round size: 10,000 iterations = 999,900 timestamp range.
Forward-return horizons: h=1000, 5000, 20000 timestamps (10/50/200 iters).

Submission format: class Trader with run(state) -> (orders, conv, traderData).
"""

from datamodel import Order, OrderDepth, TradingState, Trade
from typing import Any, Dict, List, Optional, Tuple
import json


# ============================================================================
# CONFIGURATION
# ============================================================================
PRODUCTS = [
    "HYDROGEL_PACK", "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100",
    "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500",
    "VEV_6000", "VEV_6500",
]

# Hard safety caps. Probe will not exceed these in either direction.
SAFE_CAP = {p: 25 for p in PRODUCTS}

# Probe order sizes. Small — we want a lot of small samples, not a few big ones.
PROBE_SIZE_CROSS = 1   # cross_one probe size (we lose half-spread per fill)
PROBE_SIZE_JOIN  = 2   # join-the-touch probe size (passive)

# Forward-return horizons. h is in TIMESTAMPS, not iterations.
# 1000 ts = 10 iterations, 5000 ts = 50 iterations, 20000 ts = 200 iterations.
# A 10K-iteration round is 999_900 timestamps, so h=20000 is measurable for
# the first ~9800 iterations.
FORWARD_HORIZONS_TS = [1000, 5000, 20000]
HORIZON_KEYS = [str(h) for h in FORWARD_HORIZONS_TS]   # JSON-safe string keys

# Probe schedule: cycle types and products
PROBE_TYPES = ["cross_one", "join_touch"]
PROBE_CYCLE_TICKS = 200   # one probe (per cycle position) every 200 ts = 2 iterations

# Snapshot to stdout every N iterations
SNAPSHOT_EVERY_TS = 10_000   # 100 timestamps = 1 iter; so every 100 iters we snapshot

# Final-tick detection: the LAST iteration's timestamp in a 10K-iter round
FINAL_TS_THRESHOLD = 999_800   # at this point we are within 1-2 iterations of the end

# traderData byte budget
TRADER_DATA_BUDGET = 7000

# Tracked-trades buffer length
TRACKED_TRADES_MAX = 100   # rolling buffer per product


def size_bucket(qty: int) -> str:
    """Coarse bucket for size — useful even with named counterparties because
    a CP that does both small and large size may be running mixed strategies."""
    if qty <= 2:   return "tiny"
    if qty <= 8:   return "small"
    if qty <= 20:  return "mid"
    return "large"


# ============================================================================
# Trader
# ============================================================================
class Trader:

    LIMITS = {
        "HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300, "VEV_5100": 300,
        "VEV_5200": 300, "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        orders: Dict[str, List[Order]] = {}
        ts = state.timestamp
        memory["n_iters"] = memory.get("n_iters", 0) + 1

        # Build books and decode order depths
        books = {}
        for product, depth in state.order_depths.items():
            book = self._book(depth)
            if book is not None:
                books[product] = book

        # ---- Observe market trades (passive — primary value) ----
        self._observe_market_trades(memory, ts, state.market_trades, books)

        # ---- Observe own trades (fill attribution) ----
        self._observe_own_trades(memory, ts, state.own_trades, books)

        # ---- Update forward returns for past tracked trades ----
        self._update_forward_returns(memory, ts, books)

        # ---- Auto-flatten if any product position drifts past safety cap ----
        self._auto_flatten(state.position, books, orders)

        # ---- Light active probing ----
        self._emit_probes(memory, ts, state.position, books, orders)

        # ---- Periodic snapshot ----
        if ts > 0 and ts % SNAPSHOT_EVERY_TS == 0:
            self._emit_snapshot(memory, ts, state.position, books)

        # ---- Final-tick report card ----
        if ts >= FINAL_TS_THRESHOLD and not memory.get("report_emitted"):
            self._emit_report_card(memory, ts)
            memory["report_emitted"] = True

        # ---- Persist memory ----
        memory["last_ts"] = ts
        return orders, 0, self._serialize_memory(memory)

    # ====================================================================
    # State management
    # ====================================================================
    def _load_memory(self, td: str) -> Dict[str, Any]:
        if td:
            try:
                m = json.loads(td)
                if isinstance(m, dict):
                    return m
            except Exception:
                pass
        return {
            "last_ts": -1,
            "n_iters": 0,
            "report_emitted": False,
            # BEHAVIORAL CLUSTERS — keyed by (sym, size_bucket, side) as a string.
            # Since all bots are named "Mark" (confirmed by Rook-E1 hint),
            # we cannot use name to distinguish bots. Instead, we cluster by
            # observed behavior and infer which archetype (MM, taker, whale)
            # is active on each (product, size, side) combination.
            "clusters": {},    # "SYM|bucket|side" -> stats
            # TIMING ANALYSIS — per product, list of timestamps when ANY trade
            # occurred. Used to detect periodic/rhythmic Mark bots vs triggered.
            "trade_ts_by_sym": {p: [] for p in PRODUCTS},
            # RAW CP NAMES seen (to detect if "Mark" is literal or has variants)
            "cp_names_seen": {},   # name -> count
            # Per-product summary stats
            "products": {p: {"n": 0, "spread_sum": 0.0, "mid_sum": 0.0,
                              "last_book": None, "tracked_trades": []}
                          for p in PRODUCTS},
            # Open probe orders awaiting fill
            "open_probes": {},
            # CP name → aggregated stats (all Mark variants will collapse here)
            "cp": {},
            # NEW PRODUCTS — log anything we don't recognize
            "unknown_products": {},
        }

    def _serialize_memory(self, m: Dict[str, Any]) -> str:
        # Trim oversized lists
        for p in PRODUCTS:
            pd = m.get("products", {}).get(p, {})
            tt = pd.get("tracked_trades", [])
            if len(tt) > TRACKED_TRADES_MAX:
                pd["tracked_trades"] = tt[-TRACKED_TRADES_MAX:]
        # Cap counterparties by recency × frequency
        cps = m.get("cp", {})
        if len(cps) > 80:
            top = sorted(cps.items(),
                         key=lambda kv: -(kv[1].get("n", 0) + 0.001 * kv[1].get("last_ts", 0)))[:80]
            m["cp"] = dict(top)
        # Cap open probes
        op = m.get("open_probes", {})
        # First: drop probes older than ~1000 ts (they've been auto-canceled
        # by the matching engine since unfilled probes don't persist)
        last = m.get("last_ts", 0)
        op_keep = {k: v for k, v in op.items() if last - v.get("ts", 0) <= 1000}
        m["open_probes"] = op_keep
        op = op_keep
        if len(op) > 30:
            keep = sorted(op.items(), key=lambda kv: -kv[1].get("ts", 0))[:30]
            m["open_probes"] = dict(keep)

        out = json.dumps(m, separators=(",", ":"))
        # Trim in escalating tiers if over budget. Each tier is more
        # aggressive. Stop as soon as we fit.
        for tier in range(4):
            if len(out) <= TRADER_DATA_BUDGET:
                break
            if tier == 0:
                # Tier 0: cut tracked_trades to 30 per product
                for p in PRODUCTS:
                    if p in m.get("products", {}):
                        m["products"][p]["tracked_trades"] = \
                            m["products"][p]["tracked_trades"][-30:]
                cps = m.get("cp", {})
                if len(cps) > 40:
                    top = sorted(cps.items(), key=lambda kv: -kv[1].get("n", 0))[:40]
                    m["cp"] = dict(top)
            elif tier == 1:
                # Tier 1: cut tracked_trades to 15 per product, 25 CPs max
                for p in PRODUCTS:
                    if p in m.get("products", {}):
                        m["products"][p]["tracked_trades"] = \
                            m["products"][p]["tracked_trades"][-15:]
                cps = m.get("cp", {})
                if len(cps) > 25:
                    top = sorted(cps.items(), key=lambda kv: -kv[1].get("n", 0))[:25]
                    m["cp"] = dict(top)
            elif tier == 2:
                # Tier 2: drop tracked_trades entirely (lose forward-return
                # tracking for newest trades — but the per-CP aggregations are
                # already accumulated and survive)
                for p in PRODUCTS:
                    if p in m.get("products", {}):
                        m["products"][p]["tracked_trades"] = []
            else:
                # Tier 3: drop low-frequency CPs and per-product products dict
                cps = m.get("cp", {})
                if len(cps) > 15:
                    top = sorted(cps.items(), key=lambda kv: -kv[1].get("n", 0))[:15]
                    m["cp"] = dict(top)
                # Strip per-CP product breakdown to save bytes (we still have
                # totals in fwd_ret etc.)
                for cp_st in m.get("cp", {}).values():
                    if "products" in cp_st and len(cp_st.get("products", {})) > 3:
                        prod_items = sorted(cp_st["products"].items(), key=lambda kv: -kv[1])[:3]
                        cp_st["products"] = dict(prod_items)
            out = json.dumps(m, separators=(",", ":"))
        return out

    def _book(self, depth: OrderDepth) -> Optional[Dict[str, Any]]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        bid = max(depth.buy_orders)
        ask = min(depth.sell_orders)
        bv = depth.buy_orders[bid]
        av = -depth.sell_orders[ask]
        if bv <= 0 or av <= 0 or bid >= ask:
            return None
        return {
            "best_bid": bid, "best_ask": ask, "bid_vol": bv, "ask_vol": av,
            "mid": 0.5 * (bid + ask),
            "spread": ask - bid,
        }

    # ====================================================================
    # Behavioral cluster stat update
    # ====================================================================
    def _cluster_key(self, sym: str, qty: int, side: str) -> str:
        """Build a string key for this behavioral cluster.
        Since all bots are 'Mark', we group by observable behavior."""
        return f"{sym}|{size_bucket(qty)}|{side}"

    def _update_cluster(self, memory: Dict, sym: str, side: str,
                         qty: int, vs_mid: float, ts: int,
                         cp_name: str) -> str:
        """Record one observation into the appropriate behavioral cluster.
        Returns the cluster key."""
        key = self._cluster_key(sym, qty, side)
        clusters = memory.setdefault("clusters", {})
        e = clusters.setdefault(key, {
            "n": 0,
            "buy_count": 0, "sell_count": 0,
            "qty_sum": 0,
            "vs_mid_sum": 0.0, "vs_mid_sq_sum": 0.0,
            "cp_names": {},    # name -> count (to confirm all are Mark variants)
            # Forward-return aggregations. STRING keys for JSON safety.
            "fwd_ret_sum": {hk: 0.0 for hk in HORIZON_KEYS},
            "fwd_ret_n":   {hk: 0   for hk in HORIZON_KEYS},
            "first_ts": ts, "last_ts": ts,
            # Timing: last 20 timestamps for periodicity detection
            "recent_ts": [],
        })
        e["n"] += 1
        e["qty_sum"] += qty
        if side == "buy_aggr":
            e["buy_count"] += 1
        elif side == "sell_aggr":
            e["sell_count"] += 1
        e["vs_mid_sum"] += vs_mid
        e["vs_mid_sq_sum"] += vs_mid * vs_mid
        e["last_ts"] = ts
        # Track cp_name to verify Rook's claim
        if cp_name:
            cn = e.setdefault("cp_names", {})
            cn[cp_name] = cn.get(cp_name, 0) + 1
        # Timing ring buffer (keep last 20)
        rt = e.setdefault("recent_ts", [])
        rt.append(ts)
        if len(rt) > 20:
            e["recent_ts"] = rt[-20:]
        # Defensive: ensure fwd dicts well-formed
        for d_name, default in (("fwd_ret_sum", 0.0), ("fwd_ret_n", 0)):
            d = e.get(d_name)
            if not isinstance(d, dict):
                e[d_name] = {hk: default for hk in HORIZON_KEYS}
            else:
                for hk in HORIZON_KEYS:
                    if hk not in d:
                        d[hk] = default
        # Track all raw CP names seen globally (to see Mark variants)
        if cp_name:
            cn_global = memory.setdefault("cp_names_seen", {})
            cn_global[cp_name] = cn_global.get(cp_name, 0) + 1
        return key

    def _update_cp(self, memory: Dict, name: str, sym: str, side: str,
                   qty: int, vs_mid: float, ts: int) -> None:
        """Record one trade observation into the named-CP tracking dict."""
        cps = memory.setdefault("cp", {})
        e = cps.setdefault(name, {
            "n": 0, "buy_count": 0, "sell_count": 0,
            "buy_qty": 0, "sell_qty": 0, "qty_sum": 0,
            "vs_mid_sum": 0.0, "vs_mid_sq_sum": 0.0,
            "fwd_ret_sum": {hk: 0.0 for hk in HORIZON_KEYS},
            "fwd_ret_n":   {hk: 0   for hk in HORIZON_KEYS},
            "products": {}, "size_dist": {},
            "first_ts": ts, "last_ts": ts,
        })
        e["n"] += 1
        e["qty_sum"] += qty
        e["vs_mid_sum"] += vs_mid
        e["vs_mid_sq_sum"] += vs_mid * vs_mid
        e["last_ts"] = ts
        if side == "buy_aggr":
            e["buy_count"] += 1
            e["buy_qty"] = e.get("buy_qty", 0) + qty
        elif side == "sell_aggr":
            e["sell_count"] += 1
            e["sell_qty"] = e.get("sell_qty", 0) + qty
        prods = e.setdefault("products", {})
        prods[sym] = prods.get(sym, 0) + 1
        sb = size_bucket(qty)
        sd = e.setdefault("size_dist", {})
        sd[sb] = sd.get(sb, 0) + 1
        # Ensure forward-return dicts stay well-formed across JSON round-trips
        for d_name, default in (("fwd_ret_sum", 0.0), ("fwd_ret_n", 0)):
            d = e.get(d_name)
            if not isinstance(d, dict):
                e[d_name] = {hk: default for hk in HORIZON_KEYS}
            else:
                for hk in HORIZON_KEYS:
                    if hk not in d:
                        d[hk] = default

    # ====================================================================
    # Market trade observation
    # ====================================================================
    def _observe_market_trades(self, memory: Dict, ts: int,
                                 market_trades: Dict[str, List[Trade]],
                                 books: Dict) -> None:
        for sym, trade_list in market_trades.items():
            if not trade_list:
                continue
            book = books.get(sym)
            if book is None:
                continue
            pd = memory["products"].setdefault(sym, {
                "n": 0, "spread_sum": 0.0, "mid_sum": 0.0,
                "last_book": None, "tracked_trades": []})
            prev = pd.get("last_book") or {}
            prev_bid = prev.get("best_bid", book["best_bid"])
            prev_ask = prev.get("best_ask", book["best_ask"])
            for t in trade_list:
                # Classify aggression based on prior touch when available
                if t.price >= prev_ask:
                    side = "buy_aggr"
                elif t.price <= prev_bid:
                    side = "sell_aggr"
                else:
                    side = "mid_print"
                vs_mid = float(t.price) - book["mid"]
                # Extract counterparty names. Skip blanks and SUBMISSION.
                buyer = (t.buyer or "").strip()
                seller = (t.seller or "").strip()
                # Normalize: SUBMISSION on one side means the OTHER side is
                # external. But this function processes market_trades, which
                # by spec excludes our trades — so SUBMISSION shouldn't appear.
                # Still, defensive check.
                cps_seen: List[Tuple[str, str]] = []   # (name, role)
                if buyer and buyer != "SUBMISSION":
                    self._update_cp(memory, buyer, sym, "buy_aggr",
                                     t.quantity, vs_mid, ts)
                    cps_seen.append((buyer, "buyer"))
                if seller and seller != "SUBMISSION":
                    self._update_cp(memory, seller, sym, "sell_aggr",
                                     t.quantity, vs_mid, ts)
                    cps_seen.append((seller, "seller"))

                # Track this trade for forward-return computation
                pd["tracked_trades"].append({
                    "ts": ts, "side": side, "qty": int(t.quantity),
                    "price": float(t.price), "mid": book["mid"],
                    "buyer": buyer, "seller": seller,
                    "ret": {hk: None for hk in HORIZON_KEYS},
                })

                # Stream a real-time event for any named-counterparty trade
                if cps_seen:
                    print(json.dumps({
                        "type": "cp_trade", "ts": ts, "sym": sym,
                        "buyer": buyer, "seller": seller,
                        "qty": int(t.quantity), "price": float(t.price),
                        "mid": book["mid"], "spread": book["spread"],
                        "side": side,
                    }, separators=(",", ":")))
                pd["n"] += 1

            pd["last_book"] = {"best_bid": book["best_bid"],
                                "best_ask": book["best_ask"]}
            pd["mid_sum"] += book["mid"]
            pd["spread_sum"] += book["spread"]

    # ====================================================================
    # Own trade observation (fill attribution)
    # ====================================================================
    def _observe_own_trades(self, memory: Dict, ts: int,
                              own_trades: Dict[str, List[Trade]],
                              books: Dict) -> None:
        if not own_trades:
            return
        for sym, trade_list in own_trades.items():
            book = books.get(sym)
            mid = book["mid"] if book else None
            for t in trade_list:
                we_bought = (t.buyer or "").upper() == "SUBMISSION"
                counterparty = (t.seller or "") if we_bought else (t.buyer or "")
                counterparty = counterparty.strip()
                # Match to open probe by (sym, price)
                opens = memory.get("open_probes", {})
                matched = None
                for pid, info in list(opens.items()):
                    if info.get("sym") == sym and info.get("price") == int(t.price):
                        matched = info
                        info["filled_qty"] = info.get("filled_qty", 0) + int(t.quantity)
                        if info["filled_qty"] >= info["qty"]:
                            del opens[pid]
                        break

                # Record this CP — they traded WITH us (so from their POV they
                # were on the opposite side of us). If we bought, they sold, so
                # for them this is a sell_aggr (they hit our bid? No — we crossed
                # to their ask). Actually, when we cross to ask, the seller is
                # NOT being aggressive — they're providing liquidity. So:
                #   we_bought + we_crossed → seller is liquidity provider (passive)
                #   we_bought + they_crossed our bid → seller was aggressive
                # Without knowing which order arrived first, we can't tell.
                # Safest: log as "matched_with_us" rather than aggression-side.
                if counterparty:
                    self._update_cp(memory, counterparty, sym, "matched_with_us",
                                     t.quantity,
                                     (float(t.price) - mid) if mid is not None else 0.0,
                                     ts)

                # Stream the fill event with full diagnostic context
                print(json.dumps({
                    "type": "own_fill", "ts": ts, "sym": sym,
                    "we_bought": we_bought, "qty": int(t.quantity),
                    "price": float(t.price),
                    "counterparty": counterparty,
                    "mid": mid, "spread": book["spread"] if book else None,
                    "probe_id": matched.get("probe_id") if matched else None,
                    "probe_type": matched.get("type") if matched else None,
                }, separators=(",", ":")))

    # ====================================================================
    # Forward-return computation
    # ====================================================================
    def _update_forward_returns(self, memory: Dict, ts: int,
                                  books: Dict) -> None:
        """For trades captured earlier, fill in forward returns once enough
        timestamps have elapsed. Aggregate signed returns into the relevant
        counterparties' rolling stats.

        Sign convention: positive return = mid moved UP after the trade.
        For an INFORMED counterparty:
          - if they buy_aggr and price goes up afterward → positive signed
          - if they sell_aggr and price goes down afterward → positive signed
        Therefore signed_return[h] for buy_aggr = +ret, for sell_aggr = -ret.
        High signed_return → CP is informed.
        """
        for sym, pd in memory.get("products", {}).items():
            book = books.get(sym)
            if book is None:
                continue
            mid_now = book["mid"]
            for tr in pd.get("tracked_trades", []):
                elapsed = ts - tr["ts"]
                ret_dict = tr.get("ret")
                if not isinstance(ret_dict, dict):
                    ret_dict = {hk: None for hk in HORIZON_KEYS}
                    tr["ret"] = ret_dict
                for h, hk in zip(FORWARD_HORIZONS_TS, HORIZON_KEYS):
                    if ret_dict.get(hk) is None and elapsed >= h:
                        raw_ret = mid_now - tr["mid"]
                        ret_dict[hk] = raw_ret
                        # Aggregate signed return into both counterparties' stats
                        side = tr.get("side")
                        if side == "buy_aggr":
                            signed = raw_ret      # buyer benefits if price up
                        elif side == "sell_aggr":
                            signed = -raw_ret     # seller benefits if price down
                        else:
                            signed = 0.0
                        # Buyer credit: if buy_aggr and price went up, buyer informed
                        # Seller credit: if sell_aggr and price went down, seller informed
                        for cp_name, role in (("buyer", "buyer"), ("seller", "seller")):
                            name = tr.get(cp_name, "")
                            if not name or name == "SUBMISSION":
                                continue
                            cps = memory.get("cp", {})
                            if name not in cps:
                                continue
                            e = cps[name]
                            # The signed return is the same magnitude for both
                            # parties but flipped depending on who initiated.
                            # If side='buy_aggr', buyer was the aggressor:
                            #   buyer informed if signed > 0 (price went up)
                            #   seller is the passive liquidity provider —
                            #   their "fill quality" is captured in vs_mid,
                            #   not in forward return.
                            # We credit the AGGRESSOR with the signed return.
                            if (side == "buy_aggr" and role == "buyer") or \
                               (side == "sell_aggr" and role == "seller"):
                                # Defensive: ensure dicts well-formed
                                if not isinstance(e.get("fwd_ret_sum"), dict):
                                    e["fwd_ret_sum"] = {k: 0.0 for k in HORIZON_KEYS}
                                if not isinstance(e.get("fwd_ret_n"), dict):
                                    e["fwd_ret_n"] = {k: 0 for k in HORIZON_KEYS}
                                e["fwd_ret_sum"][hk] = e["fwd_ret_sum"].get(hk, 0.0) + signed
                                e["fwd_ret_n"][hk] = e["fwd_ret_n"].get(hk, 0) + 1

    # ====================================================================
    # Active probes — light, observation-focused
    # ====================================================================
    def _emit_probes(self, memory: Dict, ts: int,
                       position: Dict[str, int], books: Dict,
                       orders: Dict[str, List[Order]]) -> None:
        """One probe per cycle position. Cycle through products and probe
        types. Skip if at safety cap."""
        # Cycle: each cycle slot probes one (product, probe_type) combo.
        # Total slots = len(PRODUCTS) * len(PROBE_TYPES) = 12 * 2 = 24.
        # Cycle period = PROBE_CYCLE_TICKS = 200 ts per slot.
        # Full cycle: 24 * 200 = 4800 ts ≈ 48 iterations.
        # Over 10K iterations we get ~208 cycles of all probes.
        slot_count = len(PRODUCTS) * len(PROBE_TYPES)
        slot = (ts // PROBE_CYCLE_TICKS) % slot_count
        prod_idx = slot % len(PRODUCTS)
        type_idx = (slot // len(PRODUCTS)) % len(PROBE_TYPES)
        sym = PRODUCTS[prod_idx]
        ptype = PROBE_TYPES[type_idx]
        if sym not in books:
            return
        book = books[sym]
        pos = position.get(sym, 0)
        cap = SAFE_CAP[sym]
        if abs(pos) >= cap:
            return

        # Side: alternate based on cycle index to balance buy/sell exposure
        side = "buy" if ((ts // PROBE_CYCLE_TICKS) % 2 == 0) else "sell"

        probe_orders: List[Tuple[int, int]] = []   # (price, signed_qty)
        if ptype == "cross_one":
            size = PROBE_SIZE_CROSS
            if side == "buy":
                probe_orders.append((book["best_ask"], +size))
            else:
                probe_orders.append((book["best_bid"], -size))
        elif ptype == "join_touch":
            size = PROBE_SIZE_JOIN
            # Skip on dead floor-priced products
            if sym in ("VEV_6000", "VEV_6500"):
                return
            if side == "buy":
                probe_orders.append((book["best_bid"], +size))
            else:
                probe_orders.append((book["best_ask"], -size))

        # Apply safety cap
        adjusted: List[Tuple[int, int]] = []
        for px, qty in probe_orders:
            if qty > 0:
                room = max(0, cap - pos)
                qty = min(qty, room)
            else:
                room = max(0, cap + pos)
                qty = -min(-qty, room)
            if qty != 0:
                adjusted.append((int(px), int(qty)))
        if not adjusted:
            return

        ord_list = orders.setdefault(sym, [])
        probe_id = f"p{ts}_{sym}_{ptype}_{side}"
        opens = memory.setdefault("open_probes", {})
        for px, qty in adjusted:
            ord_list.append(Order(sym, px, qty))
            sub_id = f"{probe_id}_{px}"
            opens[sub_id] = {
                "probe_id": probe_id, "type": ptype,
                "sym": sym, "price": px, "qty": abs(qty),
                "side": "buy" if qty > 0 else "sell",
                "filled_qty": 0, "ts": ts,
            }

        # Stream probe-emitted event
        print(json.dumps({
            "type": "probe", "ts": ts, "sym": sym, "ptype": ptype,
            "side": side, "orders": adjusted,
            "book": {"bid": book["best_bid"], "ask": book["best_ask"],
                     "bv": book["bid_vol"], "av": book["ask_vol"]},
        }, separators=(",", ":")))

    # ====================================================================
    # Auto-flatten safety
    # ====================================================================
    def _auto_flatten(self, position: Dict[str, int], books: Dict,
                       orders: Dict[str, List[Order]]) -> None:
        for sym, pos in position.items():
            cap = SAFE_CAP.get(sym, 0)
            if cap == 0 or abs(pos) <= cap // 2:
                continue
            if sym not in books:
                continue
            book = books[sym]
            target_trim = max(1, abs(pos) // 4)
            if pos > 0:
                qty = min(target_trim, book["bid_vol"])
                orders.setdefault(sym, []).append(Order(sym, book["best_bid"], -qty))
            else:
                qty = min(target_trim, book["ask_vol"])
                orders.setdefault(sym, []).append(Order(sym, book["best_ask"], qty))

    # ====================================================================
    # Periodic snapshot
    # ====================================================================
    def _emit_snapshot(self, memory: Dict, ts: int,
                        position: Dict[str, int], books: Dict) -> None:
        # Per-product summary
        prod_summary = {}
        for sym in PRODUCTS:
            pd = memory.get("products", {}).get(sym, {})
            n = pd.get("n", 1) or 1
            prod_summary[sym] = {
                "n": pd.get("n", 0),
                "avg_spread": round(pd.get("spread_sum", 0.0) / n, 2),
                "avg_mid": round(pd.get("mid_sum", 0.0) / n, 2),
                "pos": position.get(sym, 0),
                "book": (books[sym]["best_bid"], books[sym]["best_ask"])
                        if sym in books else None,
            }

        # Top 10 counterparties by trade count
        cps = memory.get("cp", {})
        top = sorted(cps.items(), key=lambda kv: -kv[1].get("n", 0))[:10]
        cp_summary = []
        for name, st in top:
            n = max(1, st["n"])
            avg_vs_mid = st["vs_mid_sum"] / n
            fwd: Dict[str, Any] = {}
            fwd_sum = st.get("fwd_ret_sum", {}) or {}
            fwd_n = st.get("fwd_ret_n", {}) or {}
            for hk in HORIZON_KEYS:
                rn = fwd_n.get(hk, 0)
                if rn > 0:
                    fwd[hk] = round(fwd_sum.get(hk, 0.0) / rn, 3)
                else:
                    fwd[hk] = None
            cp_summary.append({
                "name": name, "n": n,
                "buy": st.get("buy_count", 0), "sell": st.get("sell_count", 0),
                "avg_vs_mid": round(avg_vs_mid, 3),
                "fwd": fwd,
                "products": dict(sorted(st.get("products", {}).items(),
                                          key=lambda kv: -kv[1])[:5]),
            })

        print(json.dumps({
            "type": "snapshot", "ts": ts,
            "n_iters": memory.get("n_iters", 0),
            "n_cps": len(cps),
            "products": prod_summary,
            "top_cps": cp_summary,
        }, separators=(",", ":")))

    # ====================================================================
    # Final-tick report card — the primary deliverable
    # ====================================================================
    def _emit_report_card(self, memory: Dict, ts: int) -> None:
        """Dump a complete, ranked report of all observed counterparties.
        This is the single most important output — analyze offline."""
        cps = memory.get("cp", {})

        # Build report rows
        rows = []
        for name, st in cps.items():
            n = max(1, st["n"])
            avg_vs_mid = st["vs_mid_sum"] / n
            # Compute std of vs_mid
            mean_sq = st["vs_mid_sq_sum"] / n
            var = max(0.0, mean_sq - avg_vs_mid * avg_vs_mid)
            std_vs_mid = var ** 0.5
            # Forward returns
            fwd: Dict[str, Any] = {}
            fwd_n: Dict[str, int] = {}
            fwd_sum = st.get("fwd_ret_sum", {}) or {}
            fwd_n_dict = st.get("fwd_ret_n", {}) or {}
            for hk in HORIZON_KEYS:
                rn = fwd_n_dict.get(hk, 0)
                if rn > 0:
                    fwd[hk] = fwd_sum.get(hk, 0.0) / rn
                else:
                    fwd[hk] = None
                fwd_n[hk] = rn
            # Most active product
            prods = st.get("products", {})
            top_prod, top_prod_n = (None, 0)
            if prods:
                top_prod, top_prod_n = max(prods.items(), key=lambda kv: kv[1])
            # Side bias
            buy_qty = st.get("buy_qty", 0); sell_qty = st.get("sell_qty", 0)
            total_qty = buy_qty + sell_qty
            buy_share = buy_qty / total_qty if total_qty > 0 else 0.5

            rows.append({
                "name": name,
                "n": st["n"],
                "buy_count": st.get("buy_count", 0),
                "sell_count": st.get("sell_count", 0),
                "buy_qty": buy_qty, "sell_qty": sell_qty,
                "buy_share": round(buy_share, 3),
                "avg_vs_mid": round(avg_vs_mid, 3),
                "std_vs_mid": round(std_vs_mid, 3),
                "fwd_ret_h1k":  round(fwd.get("1000"), 3) if fwd.get("1000") is not None else None,
                "fwd_ret_h5k":  round(fwd.get("5000"), 3) if fwd.get("5000") is not None else None,
                "fwd_ret_h20k": round(fwd.get("20000"), 3) if fwd.get("20000") is not None else None,
                "fwd_n_h1k": fwd_n.get("1000", 0),
                "fwd_n_h5k": fwd_n.get("5000", 0),
                "fwd_n_h20k": fwd_n.get("20000", 0),
                "top_product": top_prod,
                "top_product_n": top_prod_n,
                "n_products": len(prods),
                "products": prods,
                "size_dist": st.get("size_dist", {}),
                "first_ts": st.get("first_ts", 0),
                "last_ts": st.get("last_ts", 0),
            })

        # Sort by trade count
        rows.sort(key=lambda r: -r["n"])

        # Emit one big report-card event with all rows
        print(json.dumps({
            "type": "report_card", "ts": ts,
            "n_iters": memory.get("n_iters", 0),
            "n_cps_total": len(rows),
            "rows": rows,
        }, separators=(",", ":")))

        # Also emit a compact "TLDR" with just the most actionable info per CP
        tldr = []
        for r in rows:
            f5k = r["fwd_ret_h5k"]
            n5k = r["fwd_n_h5k"]
            if n5k >= 3 and f5k is not None and abs(f5k) >= 0.5:
                verdict = "INFORMED — follow" if f5k > 0 else "ANTI-INFORMED — fade"
            elif n5k >= 5:
                verdict = "noise — neutral"
            else:
                verdict = "insufficient data"
            tldr.append({
                "name": r["name"], "n": r["n"], "verdict": verdict,
                "fwd_5k": f5k, "top_prod": r["top_product"],
                "buy_share": r["buy_share"],
            })
        print(json.dumps({
            "type": "report_tldr", "ts": ts, "rows": tldr,
        }, separators=(",", ":")))


# ============================================================================
# Self-test — exercise the whole pipeline without datamodel-specific
# constructors. Skip if datamodel isn't available.
# ============================================================================
if __name__ == "__main__":
    try:
        from datamodel import OrderDepth, TradingState
    except ImportError:
        print("datamodel not available, skipping self-test")
        raise SystemExit(0)

    def make_book(bid, ask, bv=20, av=20):
        d = OrderDepth()
        d.buy_orders = {bid: bv, bid - 2: bv * 2}
        d.sell_orders = {ask: -av, ask + 2: -av * 2}
        return d

    state = TradingState(
        traderData="",
        timestamp=0,
        listings={},
        order_depths={p: make_book(100, 105, 20, 20) for p in PRODUCTS},
        own_trades={}, market_trades={},
        position={p: 0 for p in PRODUCTS},
        observations=None,
    )

    t = Trader()

    print("=== Running probe_r4 self-test (40 ticks) ===")
    print("    Tests JSON round-trip, named-CP capture, forward returns, report card.")

    for tick in range(40):
        state.timestamp = tick * 100   # advance 100 ts per iteration

        # Inject named-counterparty trades on a few ticks
        if tick == 3:
            state.market_trades = {
                "VEV_5400": [Trade("VEV_5400", 105, 5, "Olivia", "Caesar", 300)],
                "HYDROGEL_PACK": [Trade("HYDROGEL_PACK", 100, 12, "Smart_Sam", "Noisy_Ned", 300)],
            }
        elif tick == 8:
            state.market_trades = {
                "VEV_5400": [Trade("VEV_5400", 100, 8, "Caesar", "Olivia", 800)],
            }
        elif tick == 15:
            state.market_trades = {
                "HYDROGEL_PACK": [Trade("HYDROGEL_PACK", 105, 15, "Noisy_Ned", "Smart_Sam", 1500)],
            }
        else:
            state.market_trades = {}

        result, conv, td = t.run(state)
        assert isinstance(result, dict)
        assert isinstance(td, str)
        # Check JSON round-trips cleanly
        decoded = json.loads(td)
        # Sanity: counterparty dict should always exist
        assert "cp" in decoded
        # If CPs have been captured, every fwd_ret_n key must be a string
        for cp_name, cp_st in decoded.get("cp", {}).items():
            for d_name in ("fwd_ret_sum", "fwd_ret_n"):
                d = cp_st.get(d_name, {})
                for k in d.keys():
                    assert isinstance(k, str), f"{d_name} has int key {k!r}"
        state.traderData = td

    # Final tick: trigger report card
    state.timestamp = 999_900
    state.market_trades = {}
    result, conv, td = t.run(state)
    decoded = json.loads(td)

    print("\n=== Self-test summary ===")
    print(f"Final traderData size: {len(td)} bytes")
    print(f"Counterparties captured: {len(decoded.get('cp', {}))}")
    print(f"Report emitted: {decoded.get('report_emitted')}")
    print(f"Sample CPs: {list(decoded.get('cp', {}).keys())}")

    # Show one CP's stats
    for name in ("Olivia", "Caesar", "Smart_Sam", "Noisy_Ned"):
        if name in decoded.get("cp", {}):
            st = decoded["cp"][name]
            print(f"\n  {name}:")
            print(f"    n={st['n']}  buy/sell={st.get('buy_count',0)}/{st.get('sell_count',0)}")
            print(f"    fwd_ret_n: {st.get('fwd_ret_n')}")
            print(f"    products: {st.get('products')}")
            break

    print("\nPROBE_R4 OK")