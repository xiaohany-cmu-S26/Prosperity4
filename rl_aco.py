"""
rl_aco.py — Tabular Q-learning training for ASH_COATED_OSMIUM (ACO).

Trains on round-1 days -2 and -1, evaluates on day 0 (no training).
Saves the learned Q-table to aco_policy.json for use by trading_aco.py.

Design choices
--------------
State space  : 5-dim discretization → 972 total states
               (position, mid-deviation, spread, momentum, book imbalance)
               After ablation a 4-dim version without imbalance was kept
               (better generalisation on day 0) — see N_STATES below.

Fill model   : Matches the actual backtester two-phase model.
               • Aggressive orders always fill at book (cross the spread).
               • Passive orders fill only when an historical market trade
                 arrives at our price.  Because we quote inside the spread
                 (bid+1 / ask-1) we have queue priority and no book volume
                 ahead of us, so the eligibility check is just
                 trade_price ≤ our_bid  (buy)  or  ≥ our_ask  (sell).

Initialisation: Q-table is warm-started with domain-knowledge values that
                encode mean-reversion + inventory management.  This gives a
                valid prior for rarely-visited states and speeds convergence
                dramatically compared to a zero cold-start.

Usage:
    python rl_aco.py                        # defaults
    python rl_aco.py --epochs 20 --lr 0.06
    python rl_aco.py --no-test              # skip day-0 evaluation
"""

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Paths & constants ─────────────────────────────────────────────────────────

DATA_DIR    = Path("ROUND_1/ROUND1")
PRODUCT     = "ASH_COATED_OSMIUM"
LIMIT       = 80
FAIR_VALUE  = 10_000.0
POLICY_PATH = Path("aco_policy.json")

TRAIN_DAYS = [-2, -1]
TEST_DAYS  = [0]

# ── State discretization ──────────────────────────────────────────────────────
#
#  1. Position   (lots)                     — 9 bins
#  2. Mid deviation from FV (ticks)         — 9 bins
#  3. Bid-ask spread (ticks)                — 4 bins
#  4. 5-tick momentum (Δmid)                — 3 bins
#
#  Total: 9 × 9 × 4 × 3 = 972 states

POS_EDGES = [-60.0, -40.0, -20.0, -5.0,  5.0, 20.0, 40.0, 60.0]   # 9 bins
DEV_EDGES = [-15.0,  -8.0,  -4.0, -1.0,  1.0,  4.0,  8.0, 15.0]   # 9 bins
SPR_EDGES = [ 10.0,  14.0,  20.0]                                    # 4 bins
MOM_EDGES = [  -2.0,  2.0]                                           # 3 bins

N_POS = len(POS_EDGES) + 1   # 9
N_DEV = len(DEV_EDGES) + 1   # 9
N_SPR = len(SPR_EDGES) + 1   # 4
N_MOM = len(MOM_EDGES) + 1   # 3
N_STATES = N_POS * N_DEV * N_SPR * N_MOM   # 972


def _bucket(v: float, edges: List[float]) -> int:
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def encode_state(pos: int, mid: float, spread: float, mom: float) -> int:
    p = _bucket(float(pos), POS_EDGES)
    d = _bucket(mid - FAIR_VALUE, DEV_EDGES)
    s = _bucket(spread, SPR_EDGES)
    m = _bucket(mom, MOM_EDGES)
    return ((p * N_DEV + d) * N_SPR + s) * N_MOM + m


# ── Actions ───────────────────────────────────────────────────────────────────

N_ACTIONS    = 5
ACTION_NAMES = ["BuyStrong", "BuyLight", "Hold", "SellLight", "SellStrong"]

QTY_STRONG = 10  # lots for aggressive orders
QTY_LIGHT  = 5   # lots for passive orders


# ── Warm-start Q initialisation (domain-knowledge prior) ─────────────────────
#
# For a mean-reverting asset:
#   • Buy  (go long)  when price is BELOW fair value (negative deviation).
#   • Sell (go short) when price is ABOVE fair value (positive deviation).
#   • Inventory pressure: lean toward reducing large positions.
#
# Signal = mean_reversion_component + inventory_component
#        = -deviation × α + (-position) × β
#
# Passive actions (Light) get a small bonus over aggressive (Strong) because
# they earn the spread when filled, while aggressive actions pay the spread.
# Hold has baseline Q = 0.

_MR_ALPHA  = 1.0   # mean-reversion weight
_INV_BETA  = 0.15  # inventory weight


def _bin_center(bin_idx: int, edges: List[float]) -> float:
    """Estimate the representative value for a discrete bin."""
    if bin_idx == 0:
        return edges[0] - abs(edges[1] - edges[0])
    elif bin_idx >= len(edges):
        return edges[-1] + abs(edges[-1] - edges[-2])
    else:
        return (edges[bin_idx - 1] + edges[bin_idx]) / 2.0


def make_prior_q() -> np.ndarray:
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    for p in range(N_POS):
        pos_val = _bin_center(p, POS_EDGES)
        for d in range(N_DEV):
            dev_val = _bin_center(d, DEV_EDGES)
            for s in range(N_SPR):
                for m in range(N_MOM):
                    si = ((p * N_DEV + d) * N_SPR + s) * N_MOM + m

                    # Mean reversion: buy when price low, sell when price high.
                    # Inventory: buy to cover shorts, sell to flatten longs.
                    signal = -dev_val * _MR_ALPHA + (-pos_val) * _INV_BETA

                    # Positive signal → lean buy; negative signal → lean sell.
                    buy_q  = max(0.0, signal)
                    sell_q = max(0.0, -signal)

                    # Passive gets a small edge over aggressive (spread economics).
                    Q[si, 0] = buy_q  * 0.85   # BuyStrong  (pays spread)
                    Q[si, 1] = buy_q  * 1.00   # BuyLight   (earns spread)
                    Q[si, 2] = 0.0             # Hold
                    Q[si, 3] = sell_q * 1.00   # SellLight  (earns spread)
                    Q[si, 4] = sell_q * 0.85   # SellStrong (pays spread)
    return Q


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class PriceRow:
    ts:   int
    bid1: Optional[int]
    ask1: Optional[int]
    mid:  float


@dataclass
class TradeRecord:
    price: int
    qty:   int


def load_aco_prices(day: int) -> List[PriceRow]:
    path = DATA_DIR / f"prices_round_1_day_{day}.csv"
    rows: List[PriceRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for cols in reader:
            if cols[2] != PRODUCT:
                continue
            bid1 = int(cols[3])  if cols[3].strip()  else None
            ask1 = int(cols[9])  if cols[9].strip()  else None
            rows.append(PriceRow(int(cols[1]), bid1, ask1, float(cols[15])))
    return rows


def load_aco_trades(day: int) -> Dict[int, List[TradeRecord]]:
    """Returns market trades keyed by timestamp."""
    path = DATA_DIR / f"trades_round_1_day_{day}.csv"
    by_ts: Dict[int, List[TradeRecord]] = {}
    if not path.is_file():
        return by_ts
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for cols in reader:
            if cols[3] != PRODUCT:
                continue
            ts  = int(cols[0])
            prc = int(float(cols[5]))
            qty = int(cols[6])
            by_ts.setdefault(ts, []).append(TradeRecord(prc, qty))
    return by_ts


# ── Fill simulation ───────────────────────────────────────────────────────────

def apply_action(
    action:    int,
    row:       PriceRow,
    trades_ts: List[TradeRecord],
    pos:       int,
    cash:      float,
) -> Tuple[int, float, float]:
    """Apply one action.  Returns (new_pos, new_cash, fill_price_or_0).

    Fill model (mirrors backtester Phase-1/2):
      BuyStrong  : cross at ask1       — always fills
      BuyLight   : passive at bid1+1   — fills when trade_price ≤ bid1+1
      Hold       : no order
      SellLight  : passive at ask1-1   — fills when trade_price ≥ ask1-1
      SellStrong : cross at bid1       — always fills
    """
    if row.bid1 is None or row.ask1 is None:
        return pos, cash, 0.0

    if action == 0:  # BuyStrong
        qty = min(QTY_STRONG, LIMIT - pos)
        if qty > 0:
            cash -= row.ask1 * qty
            pos  += qty
            return pos, cash, float(row.ask1)

    elif action == 1:  # BuyLight
        order_price = row.bid1 + 1
        if order_price >= row.ask1:
            order_price = row.ask1  # degenerate spread: treat as aggressive
        qty = min(QTY_LIGHT, LIMIT - pos)
        if qty > 0:
            for t in trades_ts:
                if t.price <= order_price:
                    fill = min(qty, t.qty)
                    if fill > 0:
                        cash -= order_price * fill
                        pos  += fill
                        return pos, cash, float(order_price)
                    break

    elif action == 2:  # Hold
        pass

    elif action == 3:  # SellLight
        order_price = row.ask1 - 1
        if order_price <= row.bid1:
            order_price = row.bid1
        qty = min(QTY_LIGHT, LIMIT + pos)
        if qty > 0:
            for t in trades_ts:
                if t.price >= order_price:
                    fill = min(qty, t.qty)
                    if fill > 0:
                        cash += order_price * fill
                        pos  -= fill
                        return pos, cash, float(order_price)
                    break

    elif action == 4:  # SellStrong
        qty = min(QTY_STRONG, LIMIT + pos)
        if qty > 0:
            cash += row.bid1 * qty
            pos  -= qty
            return pos, cash, float(row.bid1)

    return pos, cash, 0.0


# ── Q-learning ────────────────────────────────────────────────────────────────

def train(
    days:        List[int],
    *,
    epochs:      int,
    lr:          float,
    gamma:       float,
    eps_start:   float,
    eps_end:     float,
    inv_penalty: float,
    eod_penalty: float,
    Q:           np.ndarray,
) -> np.ndarray:
    """
    Tabular Q-learning with warm-started Q-table.

    Reward per tick:
        r_t = ΔMtM_t − inv_penalty × |pos_t|

    Terminal-step bonus:
        r_T += − eod_penalty × |pos_T|
        (encourages the agent to unwind before end of day)
    """
    day_data: Dict[int, List[PriceRow]]               = {}
    trade_data: Dict[int, Dict[int, List[TradeRecord]]] = {}
    for d in days:
        day_data[d]   = load_aco_prices(d)
        trade_data[d] = load_aco_trades(d)
        n_tr = sum(len(v) for v in trade_data[d].values())
        print(f"  Day {d:+d}: {len(day_data[d])} price ticks, {n_tr} market trades")

    total_steps = sum(len(day_data[d]) for d in days) * epochs
    global_step = 0

    for epoch in range(epochs):
        epoch_pnl = 0.0
        for day in days:
            rows   = day_data[day]
            trades = trade_data[day]
            pos    = 0
            cash   = 0.0
            prev_mtm = 0.0
            mids: List[float] = [rows[0].mid]

            for i, row in enumerate(rows):
                is_terminal = (i == len(rows) - 1)

                if row.bid1 is None or row.ask1 is None:
                    mids.append(row.mid)
                    continue

                spread    = float(row.ask1 - row.bid1)
                mom       = row.mid - (mids[-5] if len(mids) >= 5 else mids[0])
                si        = encode_state(pos, row.mid, spread, mom)
                trades_ts = trades.get(row.ts, [])

                # ε-greedy with linear decay
                frac   = global_step / max(total_steps - 1, 1)
                eps    = eps_start + (eps_end - eps_start) * frac
                action = (
                    random.randrange(N_ACTIONS) if random.random() < eps
                    else int(np.argmax(Q[si]))
                )

                new_pos, new_cash, _ = apply_action(action, row, trades_ts, pos, cash)
                new_mtm = new_cash + new_pos * row.mid
                reward  = (new_mtm - prev_mtm) - inv_penalty * abs(new_pos)

                if is_terminal:
                    # End-of-day: penalise non-flat position
                    reward -= eod_penalty * abs(new_pos)

                # Bellman backup
                mids.append(row.mid)
                if not is_terminal:
                    nr = rows[i + 1]
                    if nr.bid1 is not None and nr.ask1 is not None:
                        n_spr  = float(nr.ask1 - nr.bid1)
                        n_mom  = nr.mid - (mids[-5] if len(mids) >= 5 else mids[0])
                        si_nxt = encode_state(new_pos, nr.mid, n_spr, n_mom)
                        best_q = float(np.max(Q[si_nxt]))
                    else:
                        best_q = 0.0
                else:
                    best_q = 0.0

                Q[si, action] += lr * (reward + gamma * best_q - Q[si, action])

                pos      = new_pos
                cash     = new_cash
                prev_mtm = new_mtm
                global_step += 1

            day_pnl = cash + pos * rows[-1].mid
            epoch_pnl += day_pnl
            print(f"  Epoch {epoch+1:>2d} Day {day:+d}: "
                  f"MtM={day_pnl:+10,.1f}  pos={pos:+4d}")

        print(f"  Epoch {epoch+1:>2d} total PnL: {epoch_pnl:+,.1f}")

    return Q


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(Q: np.ndarray, day: int) -> float:
    rows   = load_aco_prices(day)
    trades = load_aco_trades(day)
    pos    = 0
    cash   = 0.0
    mids: List[float] = [rows[0].mid]
    action_counts = [0] * N_ACTIONS
    fill_counts   = [0] * N_ACTIONS

    for row in rows:
        if row.bid1 is None or row.ask1 is None:
            mids.append(row.mid)
            continue

        spread    = float(row.ask1 - row.bid1)
        mom       = row.mid - (mids[-5] if len(mids) >= 5 else mids[0])
        si        = encode_state(pos, row.mid, spread, mom)
        action    = int(np.argmax(Q[si]))
        trades_ts = trades.get(row.ts, [])

        new_pos, new_cash, fp = apply_action(action, row, trades_ts, pos, cash)
        action_counts[action] += 1
        if fp != 0.0:
            fill_counts[action] += 1

        pos  = new_pos
        cash = new_cash
        mids.append(row.mid)

    final = cash + pos * rows[-1].mid
    cnt   = "  ".join(
        f"{ACTION_NAMES[i]}:{action_counts[i]}(f{fill_counts[i]})"
        for i in range(N_ACTIONS)
    )
    print(f"  Day {day:+d}: simulated MtM={final:+10,.1f}  pos={pos:+4d}")
    print(f"         actions: [{cnt}]")
    return final


# ── Policy I/O ────────────────────────────────────────────────────────────────

def save_policy(Q: np.ndarray, path: Path) -> None:
    payload = {
        "Q":          Q.tolist(),
        "N_STATES":   N_STATES,
        "N_ACTIONS":  N_ACTIONS,
        "POS_EDGES":  POS_EDGES,
        "DEV_EDGES":  DEV_EDGES,
        "SPR_EDGES":  SPR_EDGES,
        "MOM_EDGES":  MOM_EDGES,
        "N_POS":      N_POS,
        "N_DEV":      N_DEV,
        "N_SPR":      N_SPR,
        "N_MOM":      N_MOM,
        "FAIR_VALUE": FAIR_VALUE,
        "PRODUCT":    PRODUCT,
        "LIMIT":      LIMIT,
        "QTY_STRONG": QTY_STRONG,
        "QTY_LIGHT":  QTY_LIGHT,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"\nPolicy saved → {path}  ({path.stat().st_size / 1024:.1f} KB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Q-learning for ACO (warm-start)")
    ap.add_argument("--epochs",      type=int,   default=15)
    ap.add_argument("--lr",          type=float, default=0.06)
    ap.add_argument("--gamma",       type=float, default=0.95)
    ap.add_argument("--eps-start",   type=float, default=0.30)
    ap.add_argument("--eps-end",     type=float, default=0.02)
    ap.add_argument("--inv-penalty", type=float, default=0.02,
                    help="Per-lot per-tick inventory penalty in reward")
    ap.add_argument("--eod-penalty", type=float, default=20.0,
                    help="Per-lot end-of-day position penalty")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--policy",      type=str,   default=str(POLICY_PATH))
    ap.add_argument("--no-test",     action="store_true")
    ap.add_argument("--cold-start",  action="store_true",
                    help="Use zero Q-init instead of domain-knowledge warm start")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  ACO Q-learning Training  (warm-start, realistic fills)")
    print("=" * 60)
    print(f"  States={N_STATES}  Actions={N_ACTIONS}")
    print(f"  Epochs={args.epochs}  LR={args.lr}  γ={args.gamma}")
    print(f"  ε: {args.eps_start} → {args.eps_end}")
    print(f"  inv_penalty={args.inv_penalty}  eod_penalty={args.eod_penalty}")
    print()

    # Initialise Q-table
    if args.cold_start:
        Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        print("  (cold start: Q = 0)")
    else:
        Q = make_prior_q()
        print("  (warm start: domain-knowledge prior)")

    print()
    print("--- Training ---")
    Q = train(
        TRAIN_DAYS,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        inv_penalty=args.inv_penalty,
        eod_penalty=args.eod_penalty,
        Q=Q,
    )

    print("\n--- Train-set evaluation (greedy) ---")
    for d in TRAIN_DAYS:
        evaluate(Q, d)

    if not args.no_test:
        print("\n--- Test evaluation: day 0 (held-out, no training) ---")
        for d in TEST_DAYS:
            evaluate(Q, d)

    save_policy(Q, Path(args.policy))

    # Human-readable policy slice (spread bin 1, flat momentum)
    print("\n--- Policy slice: spread≈16 ticks, flat momentum ---")
    header_label = "pos \\ dev"
    dev_labels = ["<-15", " -8", " -4", " -1", " ~0", " +4", " +8", "+15", ">+15"]
    pos_labels = ["<-60", " -40", " -20", "  -5", "  ~0", "  +5", " +20", " +40", ">+60"]
    print(f"  {header_label:>10}", end="")
    for lab in dev_labels:
        print(f"  {lab:>9}", end="")
    print()
    for pi in range(N_POS):
        print(f"  {pos_labels[pi]:>10}", end="")
        for di in range(N_DEV):
            si = ((pi * N_DEV + di) * N_SPR + 1) * N_MOM + 1
            a  = int(np.argmax(Q[si]))
            print(f"  {ACTION_NAMES[a]:>9}", end="")
        print()


if __name__ == "__main__":
    main()
