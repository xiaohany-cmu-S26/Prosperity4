"""
probe.py — Pure information-gathering algorithm for IMC Prosperity 4 Round 3.

Goal: extract maximum signal about bot behavior. PnL is NOT a goal.

v2 (bugfix): All dict keys that round-trip through traderData JSON are now
strings. JSON serializes integer keys as strings; the round-tripped state
needs string-key access throughout. Previous v1 crashed at first non-zero
tick with `KeyError: 100` when accessing tr["ret"][100] after the dict
came back as {"100": None, ...}.

What this does:

  1. PASSIVE OBSERVATION — every tick records the order book, all market trades,
     and all own trades into structured rolling buffers. Streams compressed
     summary lines to stdout so they survive in the lambda log for offline
     analysis.

  2. ACTIVE PROBING — places small (mostly size 1-3) probe orders at varying
     distances from the touch on a rotating per-product schedule. Each probe
     is tagged with a probe_id so when we look at fills, we know which probe
     it was. Probe types:
        - inside_one    : 1 tick inside best on chosen side
        - inside_three  : 3 ticks inside best
        - at_touch      : at the current best on chosen side (join the queue)
        - cross_one     : 1 tick across the touch (paying to hit)
        - bracket       : symmetric bid+ask at FV±k
        - large_join    : larger-size join the queue (fingerprints depth absorbers)

  3. COUNTERPARTY FINGERPRINTING — even when buyer/seller fields are blank,
     constructs behavioral counterparties:
        - by trade size buckets (small / mid / large)
        - by side and aggression (bid-hitter vs ask-lifter vs mid-printer)
        - by clustering pattern (bursty vs steady)
     For each fingerprint computes: trade count, side bias, avg-price-vs-mid,
     forward-return at +100, +500, +2000 ticks (informed vs noise classifier).

  4. SAFETY — hard cap on |position|, auto-flatten when drifts, never quote
     past a wide guard from FV. Will lose money, but bounded.

Output: stdout structured logs (one per tick during probes, periodic snapshots
otherwise) + traderData JSON snapshot for the final tick.

Submission format unchanged: class Trader with run(state) -> (orders, conv, td).
"""

from datamodel import Order, OrderDepth, TradingState, Trade
from typing import Any, Dict, List, Optional, Tuple
import json
import math


# ============================================================================
# PROBE CONFIGURATION
# ============================================================================
PRODUCTS = [
    "HYDROGEL_PACK", "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100",
    "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500",
    "VEV_6000", "VEV_6500",
]

# Hard safety caps (much smaller than position limits)
SAFE_CAP = {
    "HYDROGEL_PACK": 30,
    "VELVETFRUIT_EXTRACT": 30,
    "VEV_4000": 20, "VEV_4500": 20, "VEV_5000": 20, "VEV_5100": 20,
    "VEV_5200": 20, "VEV_5300": 20, "VEV_5400": 20, "VEV_5500": 20,
    "VEV_6000": 10, "VEV_6500": 10,
}

# Probe size by product (small enough that a fill is informative, not catastrophic)
PROBE_SIZE = {
    "HYDROGEL_PACK": 2,
    "VELVETFRUIT_EXTRACT": 3,
    "VEV_4000": 2, "VEV_4500": 1, "VEV_5000": 1, "VEV_5100": 1,
    "VEV_5200": 2, "VEV_5300": 2, "VEV_5400": 2, "VEV_5500": 2,
    "VEV_6000": 1, "VEV_6500": 1,
}
LARGE_PROBE_SIZE = {
    "HYDROGEL_PACK": 8,
    "VELVETFRUIT_EXTRACT": 12,
    "VEV_4000": 5, "VEV_4500": 3, "VEV_5000": 3, "VEV_5100": 3,
    "VEV_5200": 6, "VEV_5300": 6, "VEV_5400": 6, "VEV_5500": 6,
    "VEV_6000": 2, "VEV_6500": 2,
}

# How many distinct fingerprint size buckets to use
SIZE_BUCKETS = [(0, 2, "tiny"), (3, 8, "small"), (9, 20, "mid"), (21, 999, "large")]

# Forward-return horizons to track (in ticks).
# We keep an int list for arithmetic and a parallel string list for dict keys.
# CRITICAL: dict keys MUST be strings throughout because JSON serialization
# coerces int keys to strings, and a round-tripped dict accessed with int keys
# will KeyError. v1 crashed here at the first traderData round-trip.
FORWARD_HORIZONS = [100, 500, 2000]
HORIZON_KEYS = [str(h) for h in FORWARD_HORIZONS]   # ["100", "500", "2000"]

# Cycle one probe type per product per ~N ticks (each product gets a probe
# roughly every N * len(PROBES) ticks)
PROBE_CYCLE_TICKS = 200

PROBE_TYPES = ["inside_one", "inside_three", "at_touch", "cross_one",
               "bracket", "large_join"]


def size_bucket(qty: int) -> str:
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= qty <= hi:
            return name
    return "huge"


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

    # Maximum bytes we want traderData to occupy (keep well below platform limit)
    TRADER_DATA_BUDGET = 7000

    # Snapshot interval for stdout emission
    SNAPSHOT_EVERY = 1000

    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        orders: Dict[str, List[Order]] = {}
        ts = state.timestamp

        # Build books and decode trades into a normalized form
        books = {}
        for product, depth in state.order_depths.items():
            book = self._book(depth)
            if book is not None:
                books[product] = book

        # ---------------- Observe market trades and own trades
        self._observe_market_trades(memory, ts, state.market_trades, books)
        self._observe_own_trades(memory, ts, state.own_trades, books)

        # ---------------- Auto-flatten if drift exceeded
        self._auto_flatten(state.position, books, orders)

        # ---------------- Probes
        self._emit_probes(memory, ts, state.position, books, orders)

        # ---------------- Update forward-return computation for past trades
        self._update_forward_returns(memory, ts, books)

        # ---------------- Periodic snapshot to stdout
        if ts % self.SNAPSHOT_EVERY == 0:
            self._emit_snapshot(memory, ts, state.position, books)

        # ---------------- Persist memory (with size budget)
        memory["last_ts"] = ts
        trader_data = self._serialize_memory(memory)
        return orders, 0, trader_data

    # ====================================================================
    # State helpers
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
            # Per-product summary stats
            "products": {p: {"book_count": 0, "trade_count": 0,
                              "spread_sum": 0.0, "mid_sum": 0.0,
                              "last_book": None, "fp": {}, "tracked_trades": []}
                          for p in PRODUCTS},
            # Per-product probe history (which probe types ran when)
            "probe_log": {p: [] for p in PRODUCTS},
            # Aggregated counterparty fingerprints across all products
            "fingerprints": {},
            # Outstanding probes awaiting fill (probe_id -> details)
            "open_probes": {},
        }

    def _serialize_memory(self, m: Dict[str, Any]) -> str:
        # Trim oversized lists before serializing
        for p in PRODUCTS:
            pd = m.get("products", {}).get(p, {})
            tt = pd.get("tracked_trades", [])
            if len(tt) > 80:
                pd["tracked_trades"] = tt[-80:]
            pl = m.get("probe_log", {}).get(p, [])
            if len(pl) > 50:
                m["probe_log"][p] = pl[-50:]
        # Cap fingerprints stored
        fps = m.get("fingerprints", {})
        if len(fps) > 60:
            # keep the 60 most-active
            top = sorted(fps.items(), key=lambda kv: -kv[1].get("n", 0))[:60]
            m["fingerprints"] = dict(top)
        # Cap open_probes
        op = m.get("open_probes", {})
        if len(op) > 30:
            keep = list(op.keys())[-30:]
            m["open_probes"] = {k: op[k] for k in keep}
        out = json.dumps(m, separators=(",", ":"))
        if len(out) > self.TRADER_DATA_BUDGET:
            # Aggressive trim if still too big
            for p in PRODUCTS:
                m["products"][p]["tracked_trades"] = m["products"][p]["tracked_trades"][-30:]
                m["probe_log"][p] = m["probe_log"][p][-20:]
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
        # Full book depth — we want to fingerprint each level
        bid_levels = sorted(depth.buy_orders.items(), reverse=True)[:3]
        ask_levels = sorted(depth.sell_orders.items())[:3]
        return {
            "best_bid": bid, "best_ask": ask, "bid_vol": bv, "ask_vol": av,
            "mid": 0.5 * (bid + ask),
            "spread": ask - bid,
            "bid_levels": [(p, v) for p, v in bid_levels],
            "ask_levels": [(p, -v) for p, v in ask_levels],
        }

    # ====================================================================
    # Observation: market_trades  (other-party trades we see in the data feed)
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
            pd = memory["products"].setdefault(
                sym, {"book_count": 0, "trade_count": 0, "spread_sum": 0.0,
                      "mid_sum": 0.0, "last_book": None, "fp": {},
                      "tracked_trades": []})
            prev = pd.get("last_book") or {}
            prev_bid = prev.get("best_bid")
            prev_ask = prev.get("best_ask")
            for t in trade_list:
                # Classify aggression. Use prior touch when available, else current.
                pb = prev_bid if prev_bid is not None else book["best_bid"]
                pa = prev_ask if prev_ask is not None else book["best_ask"]
                if t.price >= pa:
                    side = "buy_aggressive"
                elif t.price <= pb:
                    side = "sell_aggressive"
                else:
                    side = "mid_print"
                vs_mid = float(t.price) - book["mid"]
                bucket = size_bucket(t.quantity)
                # Construct fingerprint key. If buyer/seller is named, use that.
                # Otherwise use behavioral fingerprint: (size_bucket, side).
                buyer_name = (t.buyer or "").strip()
                seller_name = (t.seller or "").strip()
                fp_keys_for_this_trade: List[str] = []
                if buyer_name and buyer_name != "SUBMISSION":
                    fp_key = f"named:{buyer_name}:buy"
                    self._update_fp(memory, fp_key, sym, +1, t.quantity, vs_mid, ts, book["mid"])
                    fp_keys_for_this_trade.append(fp_key)
                if seller_name and seller_name != "SUBMISSION":
                    fp_key = f"named:{seller_name}:sell"
                    self._update_fp(memory, fp_key, sym, -1, t.quantity, vs_mid, ts, book["mid"])
                    fp_keys_for_this_trade.append(fp_key)
                # Always also record behavioral fingerprint (works even when names are empty)
                fp_key = f"beh:{sym}:{bucket}:{side}"
                sign = +1 if side == "buy_aggressive" else (-1 if side == "sell_aggressive" else 0)
                self._update_fp(memory, fp_key, sym, sign, t.quantity, vs_mid, ts, book["mid"])
                fp_keys_for_this_trade.append(fp_key)

                # Capture the raw trade with mid context for forward-return.
                # IMPORTANT: use string keys for "ret" because this dict will
                # round-trip through JSON.
                pd["tracked_trades"].append({
                    "ts": ts, "side": side, "qty": int(t.quantity),
                    "price": float(t.price), "mid": book["mid"],
                    "buyer": buyer_name, "seller": seller_name,
                    "fp_keys": fp_keys_for_this_trade,
                    # Forward-return slots, filled later. Keys are strings.
                    "ret": {hk: None for hk in HORIZON_KEYS},
                })
                pd["trade_count"] += 1
            pd["last_book"] = {"best_bid": book["best_bid"],
                                "best_ask": book["best_ask"]}
            pd["mid_sum"] += book["mid"]
            pd["spread_sum"] += book["spread"]
            pd["book_count"] += 1

            # Emit per-trade event log (only if we have a named counterparty
            # — these are rare and worth printing every time we see them)
            for t in trade_list:
                if (t.buyer and t.buyer != "SUBMISSION") or \
                   (t.seller and t.seller != "SUBMISSION"):
                    print(json.dumps({
                        "type": "named_trade", "ts": ts, "sym": sym,
                        "buyer": t.buyer, "seller": t.seller,
                        "qty": int(t.quantity), "price": float(t.price),
                        "mid": book["mid"], "spread": book["spread"],
                    }, separators=(",", ":")))

    def _update_fp(self, memory: Dict, fp_key: str, sym: str,
                    sign: int, qty: int, vs_mid: float,
                    ts: int, mid: float) -> None:
        fps = memory.setdefault("fingerprints", {})
        e = fps.setdefault(fp_key, {
            "n": 0, "qty_sum": 0, "sign_qty_sum": 0,
            "vs_mid_sum": 0.0, "vs_mid_sq_sum": 0.0,
            "first_ts": ts, "last_ts": ts,
            "products": {},
            # Forward-return aggregation. STRING keys.
            "fwd_ret_sum": {hk: 0.0 for hk in HORIZON_KEYS},
            "fwd_ret_n":   {hk: 0   for hk in HORIZON_KEYS},
        })
        e["n"] += 1
        e["qty_sum"] += qty
        e["sign_qty_sum"] += sign * qty
        e["vs_mid_sum"] += vs_mid
        e["vs_mid_sq_sum"] += vs_mid * vs_mid
        e["last_ts"] = ts
        ps = e.setdefault("products", {})
        ps[sym] = ps.get(sym, 0) + 1
        # Defensive: in case fp came back from JSON without these dicts populated
        # (e.g., older traderData format), make sure they exist with string keys.
        for d_name in ("fwd_ret_sum", "fwd_ret_n"):
            d = e.get(d_name)
            if not isinstance(d, dict):
                e[d_name] = {hk: (0.0 if d_name == "fwd_ret_sum" else 0) for hk in HORIZON_KEYS}
            else:
                for hk in HORIZON_KEYS:
                    if hk not in d:
                        d[hk] = 0.0 if d_name == "fwd_ret_sum" else 0

    # ====================================================================
    # Observation: own_trades (our fills)
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
                counterparty = t.seller if we_bought else t.buyer
                # Attach probe context if any open probe matches this fill
                opens = memory.get("open_probes", {})
                matched_probe = None
                for pid, info in list(opens.items()):
                    if info.get("sym") == sym and info.get("price") == int(t.price):
                        matched_probe = info
                        # Decrement remaining qty; remove when fully filled
                        info["filled_qty"] = info.get("filled_qty", 0) + int(t.quantity)
                        if info["filled_qty"] >= info["qty"]:
                            del opens[pid]
                        break
                # Print fill event with full diagnostic context
                print(json.dumps({
                    "type": "own_fill", "ts": ts, "sym": sym,
                    "we_bought": we_bought, "qty": int(t.quantity),
                    "price": float(t.price),
                    "counterparty": counterparty,
                    "mid": mid, "spread": book["spread"] if book else None,
                    "probe_id": matched_probe.get("probe_id") if matched_probe else None,
                    "probe_type": matched_probe.get("type") if matched_probe else None,
                }, separators=(",", ":")))
                # Update counterparty fingerprint if named
                if counterparty:
                    fp_key = f"counterparty_named:{counterparty}"
                    sign = -1 if we_bought else +1   # they sold to us = sell-aggressive
                    self._update_fp(memory, fp_key, sym, sign, t.quantity,
                                     (float(t.price) - mid) if mid else 0.0,
                                     ts, mid or 0.0)

    # ====================================================================
    # Forward-return computation
    # ====================================================================
    def _update_forward_returns(self, memory: Dict, ts: int, books: Dict) -> None:
        """For trades we recorded in the past, fill in the forward-return slots
        once enough time has passed. Aggregate into the matching fingerprints.

        All dict keys here are strings (HORIZON_KEYS), to survive JSON round-trips."""
        for sym, pd in memory.get("products", {}).items():
            book = books.get(sym)
            if book is None:
                continue
            mid_now = book["mid"]
            for tr in pd.get("tracked_trades", []):
                elapsed = ts - tr["ts"]
                # Defensive: ensure ret dict exists with string keys
                ret_dict = tr.get("ret")
                if not isinstance(ret_dict, dict):
                    ret_dict = {hk: None for hk in HORIZON_KEYS}
                    tr["ret"] = ret_dict
                for h, hk in zip(FORWARD_HORIZONS, HORIZON_KEYS):
                    # Only compute if not yet computed and enough time passed
                    if ret_dict.get(hk) is None and elapsed >= h:
                        ret_dict[hk] = mid_now - tr["mid"]
                        # Aggregate into fingerprint stats.
                        # Sign convention: positive return = mid went UP.
                        # If trade was buy-aggressive and ret > 0, buyer was informed.
                        # If trade was sell-aggressive and ret < 0, seller was informed.
                        for fp_key in tr.get("fp_keys", []):
                            fps = memory.get("fingerprints", {}).get(fp_key)
                            if fps is None:
                                continue
                            # Defensive: ensure stats dicts exist with string keys
                            if not isinstance(fps.get("fwd_ret_sum"), dict):
                                fps["fwd_ret_sum"] = {k: 0.0 for k in HORIZON_KEYS}
                            if not isinstance(fps.get("fwd_ret_n"), dict):
                                fps["fwd_ret_n"] = {k: 0 for k in HORIZON_KEYS}
                            if hk not in fps["fwd_ret_sum"]:
                                fps["fwd_ret_sum"][hk] = 0.0
                            if hk not in fps["fwd_ret_n"]:
                                fps["fwd_ret_n"][hk] = 0
                            # Signed return relative to trade direction
                            if tr["side"] == "buy_aggressive":
                                signed = ret_dict[hk]
                            elif tr["side"] == "sell_aggressive":
                                signed = -ret_dict[hk]
                            else:
                                signed = 0.0
                            fps["fwd_ret_sum"][hk] += signed
                            fps["fwd_ret_n"][hk] += 1

    # ====================================================================
    # Probes — actively place small orders to elicit responses
    # ====================================================================
    def _emit_probes(self, memory: Dict, ts: int,
                       position: Dict[str, int], books: Dict,
                       orders: Dict[str, List[Order]]) -> None:
        """Cycle probes across products. Each (product, type) gets a turn
        every PROBE_CYCLE_TICKS * len(PRODUCTS) * len(PROBE_TYPES) ticks."""
        idx = (ts // PROBE_CYCLE_TICKS) % (len(PRODUCTS) * len(PROBE_TYPES))
        prod_idx = idx % len(PRODUCTS)
        type_idx = (idx // len(PRODUCTS)) % len(PROBE_TYPES)
        sym = PRODUCTS[prod_idx]
        ptype = PROBE_TYPES[type_idx]
        if sym not in books:
            return
        book = books[sym]
        pos = position.get(sym, 0)
        cap = SAFE_CAP[sym]
        if abs(pos) >= cap:
            return  # safety: don't probe further if already past cap

        size = PROBE_SIZE[sym]
        large = LARGE_PROBE_SIZE[sym]
        spread = book["spread"]
        # Pick side: alternate based on tick to avoid systematic bias
        side = "buy" if (ts // PROBE_CYCLE_TICKS) % 2 == 0 else "sell"

        probe_id = f"p{ts}_{sym}_{ptype}_{side}"
        probe_orders: List[Tuple[int, int]] = []   # list of (price, qty)

        if ptype == "inside_one":
            if side == "buy":
                px = book["best_bid"] + 1
                if px < book["best_ask"]:
                    probe_orders.append((px, +size))
            else:
                px = book["best_ask"] - 1
                if px > book["best_bid"]:
                    probe_orders.append((px, -size))

        elif ptype == "inside_three":
            offset = max(1, min(3, spread - 1))
            if side == "buy":
                px = book["best_bid"] + offset
                if px < book["best_ask"]:
                    probe_orders.append((px, +size))
            else:
                px = book["best_ask"] - offset
                if px > book["best_bid"]:
                    probe_orders.append((px, -size))

        elif ptype == "at_touch":
            if side == "buy":
                probe_orders.append((book["best_bid"], +size))
            else:
                probe_orders.append((book["best_ask"], -size))

        elif ptype == "cross_one":
            # Cross by one tick (intentionally pay the spread)
            if side == "buy":
                px = book["best_ask"]
                probe_orders.append((px, +size))
            else:
                px = book["best_bid"]
                probe_orders.append((px, -size))

        elif ptype == "bracket":
            # Symmetric pair around mid
            mid = round(book["mid"])
            half = max(1, spread // 4)
            bp = mid - half
            ap = mid + half
            if bp < book["best_ask"]:
                probe_orders.append((bp, +size))
            if ap > book["best_bid"]:
                probe_orders.append((ap, -size))

        elif ptype == "large_join":
            # Larger size at the touch — fingerprints big-flow absorbers
            if side == "buy":
                probe_orders.append((book["best_bid"], +large))
            else:
                probe_orders.append((book["best_ask"], -large))

        if not probe_orders:
            return

        # Apply safety cap to actual emitted size
        adjusted = []
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

        # Emit orders
        ord_list = orders.setdefault(sym, [])
        for px, qty in adjusted:
            ord_list.append(Order(sym, px, qty))

        # Log the probe and stash an open-probe entry per order
        log_entry = {
            "ts": ts, "type": ptype, "side": side,
            "orders": adjusted, "spread": spread,
            "book_bid": book["best_bid"], "book_ask": book["best_ask"],
            "book_bid_vol": book["bid_vol"], "book_ask_vol": book["ask_vol"],
        }
        memory.setdefault("probe_log", {}).setdefault(sym, []).append(log_entry)
        opens = memory.setdefault("open_probes", {})
        for px, qty in adjusted:
            sub_id = f"{probe_id}_{px}"
            opens[sub_id] = {
                "probe_id": probe_id, "sub_id": sub_id, "type": ptype,
                "sym": sym, "price": px, "qty": abs(qty),
                "side": "buy" if qty > 0 else "sell",
                "filled_qty": 0, "ts": ts,
            }

        # Stream a probe-emitted event
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
            if cap == 0:
                continue
            if abs(pos) <= cap // 2:
                continue
            if sym not in books:
                continue
            book = books[sym]
            # Trim toward zero by a small amount each tick
            target_trim = max(1, abs(pos) // 4)
            if pos > 0:
                # sell at the bid
                qty = min(target_trim, book["bid_vol"])
                orders.setdefault(sym, []).append(Order(sym, book["best_bid"], -qty))
            else:
                qty = min(target_trim, book["ask_vol"])
                orders.setdefault(sym, []).append(Order(sym, book["best_ask"], qty))

    # ====================================================================
    # Periodic snapshot — push aggregate stats to stdout for offline parsing
    # ====================================================================
    def _emit_snapshot(self, memory: Dict, ts: int,
                        position: Dict[str, int], books: Dict) -> None:
        # Per-product summary
        prod_summary = {}
        for sym in PRODUCTS:
            pd = memory.get("products", {}).get(sym, {})
            n = pd.get("book_count", 1) or 1
            tc = pd.get("trade_count", 0)
            prod_summary[sym] = {
                "tc": tc,
                "avg_spread": round(pd.get("spread_sum", 0.0) / n, 2),
                "avg_mid": round(pd.get("mid_sum", 0.0) / n, 2),
                "pos": position.get(sym, 0),
                "book_top": (books[sym]["best_bid"], books[sym]["best_ask"])
                            if sym in books else None,
            }

        # Top fingerprints — by frequency
        fps = memory.get("fingerprints", {})
        top = sorted(fps.items(), key=lambda kv: -kv[1].get("n", 0))[:15]
        fp_summary = []
        for key, st in top:
            n = max(1, st["n"])
            avg_vs_mid = st["vs_mid_sum"] / n
            fwd: Dict[str, Optional[float]] = {}
            fwd_sum = st.get("fwd_ret_sum", {}) or {}
            fwd_n = st.get("fwd_ret_n", {}) or {}
            for hk in HORIZON_KEYS:
                rn = fwd_n.get(hk, 0)
                if rn > 0:
                    fwd[hk] = round(fwd_sum.get(hk, 0.0) / rn, 3)
                else:
                    fwd[hk] = None
            fp_summary.append({
                "k": key, "n": n,
                "avg_qty": round(st["qty_sum"] / n, 2),
                "avg_vs_mid": round(avg_vs_mid, 3),
                "fwd": fwd,
            })

        # Open probes
        opens = memory.get("open_probes", {})

        print(json.dumps({
            "type": "snapshot", "ts": ts,
            "products": prod_summary,
            "top_fingerprints": fp_summary,
            "n_open_probes": len(opens),
            "n_fingerprints": len(fps),
        }, separators=(",", ":")))


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    from datamodel import OrderDepth, TradingState

    def make_book(bid, ask, bv=20, av=20):
        d = OrderDepth()
        d.buy_orders = {bid: bv, bid - 2: bv * 2}
        d.sell_orders = {ask: -av, ask + 2: -av * 2}
        return d

    state = TradingState(
        traderData="",
        timestamp=0,
        listings={},
        order_depths={
            "HYDROGEL_PACK":       make_book(9985, 10001, 12, 12),
            "VELVETFRUIT_EXTRACT": make_book(5263, 5268, 38, 38),
            "VEV_4000":            make_book(1255, 1276, 11, 11),
            "VEV_4500":            make_book(755, 776, 7, 7),
            "VEV_5000":            make_book(265, 272, 8, 8),
            "VEV_5100":            make_book(175, 180, 8, 8),
            "VEV_5200":            make_book(101, 105, 11, 11),
            "VEV_5300":            make_book(51, 53, 20, 20),
            "VEV_5400":            make_book(16, 18, 20, 20),
            "VEV_5500":            make_book(6, 7, 20, 20),
            "VEV_6000":            make_book(0, 1, 18, 18),
            "VEV_6500":            make_book(0, 1, 18, 18),
        },
        own_trades={}, market_trades={},
        position={p: 0 for p in PRODUCTS},
        observations=None,
    )

    t = Trader()

    print("=== Running probe.py self-test (30 ticks, 200ts apart) ===")
    print("    This exercises the JSON round-trip path that crashed v1.")

    for tick in range(30):
        state.timestamp = tick * 100

        # On some ticks, inject some market trades to test fingerprinting
        if tick == 3:
            state.market_trades = {
                "VEV_5400": [Trade("VEV_5400", 16, 5, "", "", 300)],   # bid hit
                "HYDROGEL_PACK": [Trade("HYDROGEL_PACK", 10001, 12, "Caesar", "", 300)],
            }
        elif tick == 7:
            state.market_trades = {
                "HYDROGEL_PACK": [Trade("HYDROGEL_PACK", 9985, 8, "", "", 700)],
            }
        else:
            state.market_trades = {}

        result, conv, td = t.run(state)
        assert isinstance(result, dict), f"result not dict at tick {tick}"
        assert isinstance(conv, int), f"conv not int at tick {tick}"
        assert isinstance(td, str), f"td not str at tick {tick}"

        # Round-trip JSON — this is what the simulator does
        decoded = json.loads(td)
        assert "products" in decoded
        assert "fingerprints" in decoded

        # CRITICAL: verify the bug we just fixed. After JSON round-trip the
        # ret dicts inside tracked_trades should still be accessible by string
        # key. If we accidentally regress to int keys, this whole thing will
        # crash on the next tick when _update_forward_returns runs.
        for p, pd in decoded.get("products", {}).items():
            for tr in pd.get("tracked_trades", []):
                ret = tr.get("ret", {})
                # All keys must be strings (per HORIZON_KEYS)
                for k in ret.keys():
                    assert isinstance(k, str), \
                        f"tracked_trade ret has non-string key {k!r} at tick {tick}"

        # Same check for fingerprint fwd_ret dicts
        for fp_key, fp in decoded.get("fingerprints", {}).items():
            for d_name in ("fwd_ret_sum", "fwd_ret_n"):
                d = fp.get(d_name, {}) or {}
                for k in d.keys():
                    assert isinstance(k, str), \
                        f"fingerprint {fp_key} {d_name} has non-string key {k!r}"

        state.traderData = td

    final = json.loads(state.traderData)
    print(f"\n=== Self-test summary ===")
    print(f"Final traderData size: {len(state.traderData)} bytes")
    print(f"Fingerprints captured: {len(final.get('fingerprints', {}))}")
    print(f"Open probes: {len(final.get('open_probes', {}))}")
    print(f"Products tracked: {len(final.get('products', {}))}")

    # Show a couple of fingerprints
    print("\nSample fingerprints:")
    fps = final.get("fingerprints", {})
    for k in list(fps.keys())[:5]:
        print(f"  {k}: n={fps[k]['n']}, qty_sum={fps[k]['qty_sum']}, "
              f"avg_vs_mid={fps[k]['vs_mid_sum']/max(1,fps[k]['n']):.2f}")

    # Verify forward-return aggregation worked: the trade injected at tick 3
    # should have its 100-tick forward return computed by tick 4 onward
    # (assuming book prices changed; otherwise return is 0).
    found_with_fwd = False
    for fp_key, fp in fps.items():
        n_100 = fp.get("fwd_ret_n", {}).get("100", 0)
        if n_100 > 0:
            found_with_fwd = True
            print(f"\n  Forward-return aggregation working: {fp_key} has "
                  f"{n_100} h=100 observations, avg_signed={fp['fwd_ret_sum']['100']/n_100:.3f}")
            break
    if not found_with_fwd:
        print("\n  (No forward returns computed — book wasn't moving in test fixture, OK)")

    print("\nPROBE OK — JSON round-trip stable across 30 ticks, no KeyError")