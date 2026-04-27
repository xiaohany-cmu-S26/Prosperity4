"""
trading_aco.py — ACO trading algorithm driven by a trained RL Q-table.

Loads the Q-table from aco_policy.json (produced by rl_aco.py).
Only trades ASH_COATED_OSMIUM; INTARIAN_PEPPER_ROOT is ignored.

Order placement (matches the training fill model exactly):
  BuyStrong  → aggressive buy  at best_ask          (always fills)
  BuyLight   → passive   buy   at best_bid + 1       (queue-priority inside-spread bid)
  Hold       → tiny symmetric MM at best_bid / ask   (spread capture, low conviction)
  SellLight  → passive   sell  at best_ask - 1       (queue-priority inside-spread ask)
  SellStrong → aggressive sell at best_bid            (always fills)

Run against the backtester:
    python backtest.py --algo trading_aco.py --day 0    # test set only
    python backtest.py --algo trading_aco.py             # all round-1 days

Train the policy first if aco_policy.json doesn't exist:
    python rl_aco.py
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState

# ── Load trained Q-table at import time ──────────────────────────────────────

_POLICY_PATH = Path(__file__).parent / "aco_policy.json"

with open(_POLICY_PATH, encoding="utf-8") as _f:
    _p = json.load(_f)

# Raw Q-table and metadata
_Q          = _p["Q"]          # list[list[float]], shape (N_STATES, N_ACTIONS)
_POS_EDGES  = _p["POS_EDGES"]
_DEV_EDGES  = _p["DEV_EDGES"]
_SPR_EDGES  = _p["SPR_EDGES"]
_MOM_EDGES  = _p["MOM_EDGES"]
_N_DEV      = _p["N_DEV"]
_N_SPR      = _p["N_SPR"]
_N_MOM      = _p["N_MOM"]
_FAIR_VALUE = float(_p["FAIR_VALUE"])

# ── Product configuration (read by backtest.py for position limits) ───────────

PRODUCT        = "ASH_COATED_OSMIUM"
DEFAULT_LIMIT  = 80

PRODUCT_CONFIG = {
    PRODUCT: {"limit": DEFAULT_LIMIT},
}

# ── State encoding (must match rl_aco.py exactly) ────────────────────────────

def _bucket(v: float, edges: list) -> int:
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def _encode_state(pos: int, mid: float, spread: float, mom: float) -> int:
    p = _bucket(float(pos), _POS_EDGES)
    d = _bucket(mid - _FAIR_VALUE, _DEV_EDGES)
    s = _bucket(spread, _SPR_EDGES)
    m = _bucket(mom, _MOM_EDGES)
    return ((p * _N_DEV + d) * _N_SPR + s) * _N_MOM + m


def _best_action(si: int) -> int:
    row = _Q[si]
    return max(range(len(row)), key=lambda i: row[i])


# ── Order sizing ──────────────────────────────────────────────────────────────
#
# Sizes must match those used in training (rl_aco.py apply_action):
#   BuyStrong / SellStrong : 10 lots   (aggressive, crosses the spread)
#   BuyLight  / SellLight  : 5 lots    (passive, posts on the book)
#   Hold                   : small symmetric MM quotes (2 lots each side)
#     The Hold action in training meant "no fill"; here we add tiny passive
#     MM quotes so the algo still captures spread in low-conviction states.

BUY_STRONG_QTY  = 10
BUY_LIGHT_QTY   = 5
SELL_LIGHT_QTY  = 5
SELL_STRONG_QTY = 10
HOLD_MM_QTY     = 2   # small passive MM when RL says Hold


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:
    """
    On each tick:
      1. Extract mid price, spread, 5-tick momentum from the order book.
      2. Encode (position, deviation, spread, momentum) into a state index.
      3. Look up the greedy action from the Q-table.
      4. Place orders according to the action:
           0 BuyStrong  → aggressive buy  at best_ask
           1 BuyLight   → passive   buy   at best_bid
           2 Hold       → small passive MM at best_bid and best_ask
           3 SellLight  → passive   sell  at best_ask
           4 SellStrong → aggressive sell at best_bid
    """

    def run(self, state: TradingState):
        # ── Restore persistent state ─────────────────────────────────────────
        try:
            persistent: dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            persistent = {}

        result: Dict[str, List[Order]] = {}

        # Only trade ACO; skip any other products present in the state.
        if PRODUCT not in state.order_depths:
            return result, 0, json.dumps(persistent)

        od: OrderDepth = state.order_depths[PRODUCT]

        if not od.buy_orders or not od.sell_orders:
            return result, 0, json.dumps(persistent)

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2.0
        spread   = float(best_ask - best_bid)

        pos      = state.position.get(PRODUCT, 0)
        buy_cap  = DEFAULT_LIMIT - pos
        sell_cap = DEFAULT_LIMIT + pos

        # ── Update mid-price history for 5-tick momentum ─────────────────────
        history: List[float] = persistent.get("mid_history", [])
        history.append(mid)
        if len(history) > 10:
            history = history[-10:]
        persistent["mid_history"] = history

        mom = mid - (history[-5] if len(history) >= 5 else history[0])

        # ── Encode state and choose greedy action ─────────────────────────────
        si     = _encode_state(pos, mid, spread, mom)
        action = _best_action(si)

        orders: List[Order] = []

        if action == 0:
            # BuyStrong: aggressive buy at best_ask
            qty = min(BUY_STRONG_QTY, buy_cap)
            if qty > 0:
                orders.append(Order(PRODUCT, best_ask, qty))

        elif action == 1:
            # BuyLight: passive inside-spread bid at best_bid+1 (queue priority)
            order_price = best_bid + 1
            if order_price >= best_ask:
                order_price = best_ask  # spread too narrow, treat as aggressive
            qty = min(BUY_LIGHT_QTY, buy_cap)
            if qty > 0:
                orders.append(Order(PRODUCT, order_price, qty))

        elif action == 2:
            # Hold: small symmetric MM quotes at best_bid / best_ask
            # (posts on both sides; fills only if a market trade arrives)
            mm_qty = min(HOLD_MM_QTY, buy_cap, sell_cap)
            if mm_qty > 0 and best_bid < best_ask:
                orders.append(Order(PRODUCT, best_bid,  mm_qty))
                orders.append(Order(PRODUCT, best_ask, -mm_qty))

        elif action == 3:
            # SellLight: passive inside-spread ask at best_ask-1 (queue priority)
            order_price = best_ask - 1
            if order_price <= best_bid:
                order_price = best_bid  # spread too narrow, treat as aggressive
            qty = min(SELL_LIGHT_QTY, sell_cap)
            if qty > 0:
                orders.append(Order(PRODUCT, order_price, -qty))

        elif action == 4:
            # SellStrong: aggressive sell at best_bid
            qty = min(SELL_STRONG_QTY, sell_cap)
            if qty > 0:
                orders.append(Order(PRODUCT, best_bid, -qty))

        result[PRODUCT] = orders
        return result, 0, json.dumps(persistent)
