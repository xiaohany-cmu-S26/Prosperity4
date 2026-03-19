"""
Local backtester for IMC Prosperity trading strategies.

Replays tutorial round CSV data through trading.py's Trader.run(), simulating
order matching against the historical order book.

Matching model:
  - Aggressive fills: orders that cross the current book fill immediately
  - Passive fills: resting orders that improve the book get filled when market
    trades occur at prices that would have crossed our quotes (we'd be filled
    first since our price is better for the counterparty)

Usage:
  python3 backtest.py                          # Run both days
  python3 backtest.py --day -1                 # Run day -1 only
  python3 backtest.py --verbose                # Print every fill
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    Trade,
    TradingState,
)
from trading import Trader, POSITION_LIMITS, DEFAULT_LIMIT

DATA_DIR = Path("TUTORIAL_ROUND_1")


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_prices(path: Path) -> Dict[int, Dict[str, dict]]:
    """Parse prices CSV into {timestamp: {product: row_dict}}."""
    snapshots: Dict[int, Dict[str, dict]] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ts = int(row["timestamp"])
            product = row["product"]
            snapshots.setdefault(ts, {})[product] = row
    return snapshots


def load_trades(path: Path) -> Dict[int, List[dict]]:
    """Parse trades CSV into {timestamp: [trade_dict, ...]}."""
    trades: Dict[int, List[dict]] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ts = int(row["timestamp"])
            trades.setdefault(ts, []).append(row)
    return trades


def build_order_depth(row: dict) -> OrderDepth:
    """Build an OrderDepth from a prices CSV row."""
    od = OrderDepth()
    for level in range(1, 4):
        bp = row.get(f"bid_price_{level}", "")
        bv = row.get(f"bid_volume_{level}", "")
        ap = row.get(f"ask_price_{level}", "")
        av = row.get(f"ask_volume_{level}", "")
        if bp and bv:
            od.buy_orders[int(float(bp))] = int(float(bv))
        if ap and av:
            od.sell_orders[int(float(ap))] = -int(float(av))  # negative volume
    return od


def get_mid(row: dict) -> float:
    mp = row.get("mid_price", "")
    return float(mp) if mp else 0.0


# ── Order Matching Engine ────────────────────────────────────────────────────

def match_aggressive(
    orders: List[Order],
    order_depth: OrderDepth,
    product: str,
    position: int,
    limit: int,
) -> Tuple[List[Trade], List[Order]]:
    """
    Match orders aggressively against the book. Returns (fills, passive_remainder).
    Orders that don't immediately cross the book are returned as passive orders.
    """
    fills: List[Trade] = []
    passive: List[Order] = []
    current_pos = position

    for order in orders:
        if order.quantity > 0:
            # BUY: match against asks
            available_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
            remaining = order.quantity
            buy_room = limit - current_pos
            remaining = min(remaining, buy_room)
            if remaining <= 0:
                continue

            filled_any = False
            for ask_price, ask_vol in available_asks:
                if order.price < ask_price:
                    break
                fill_qty = min(remaining, abs(ask_vol))
                if fill_qty <= 0:
                    continue
                fills.append(Trade(
                    symbol=product, price=ask_price, quantity=fill_qty,
                    buyer="SUBMISSION", seller="BOT", timestamp=0,
                ))
                current_pos += fill_qty
                remaining -= fill_qty
                filled_any = True
                # Update book
                order_depth.sell_orders[ask_price] += fill_qty
                if order_depth.sell_orders[ask_price] == 0:
                    del order_depth.sell_orders[ask_price]
                if remaining <= 0:
                    break

            # Remaining becomes a passive resting order
            if remaining > 0:
                passive.append(Order(product, order.price, remaining))

        elif order.quantity < 0:
            # SELL: match against bids
            available_bids = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])
            remaining = abs(order.quantity)
            sell_room = limit + current_pos
            remaining = min(remaining, sell_room)
            if remaining <= 0:
                continue

            for bid_price, bid_vol in available_bids:
                if order.price > bid_price:
                    break
                fill_qty = min(remaining, bid_vol)
                if fill_qty <= 0:
                    continue
                fills.append(Trade(
                    symbol=product, price=bid_price, quantity=fill_qty,
                    buyer="BOT", seller="SUBMISSION", timestamp=0,
                ))
                current_pos -= fill_qty
                remaining -= fill_qty
                order_depth.buy_orders[bid_price] -= fill_qty
                if order_depth.buy_orders[bid_price] == 0:
                    del order_depth.buy_orders[bid_price]
                if remaining <= 0:
                    break

            if remaining > 0:
                passive.append(Order(product, order.price, -remaining))

    return fills, passive


def match_passive(
    resting_orders: Dict[str, List[Order]],
    market_trades: List[dict],
    positions: Dict[str, int],
) -> List[Trade]:
    """
    Fill resting passive orders using market trade data.

    Logic: if a market trade occurred at price P for product X:
      - Our resting BUY at price >= P would have been filled first (better bid)
      - Our resting SELL at price <= P would have been filled first (better ask)
    Fill at OUR price (price improvement for the counterparty).
    Volume is capped at the market trade quantity.
    """
    fills: List[Trade] = []

    for td in market_trades:
        sym = td["symbol"]
        trade_price = int(float(td["price"]))
        trade_qty = int(float(td["quantity"]))

        if sym not in resting_orders or not resting_orders[sym]:
            continue

        limit = POSITION_LIMITS.get(sym, DEFAULT_LIMIT)
        remaining_volume = trade_qty
        new_resting = []

        for order in resting_orders[sym]:
            if remaining_volume <= 0:
                new_resting.append(order)
                continue

            if order.quantity > 0:
                # Resting BUY: fill if our bid >= trade price (we're a better bid)
                if order.price >= trade_price:
                    buy_room = limit - positions.get(sym, 0)
                    fill_qty = min(order.quantity, remaining_volume, buy_room)
                    if fill_qty > 0:
                        fills.append(Trade(
                            symbol=sym, price=order.price, quantity=fill_qty,
                            buyer="SUBMISSION", seller="BOT", timestamp=0,
                        ))
                        positions[sym] = positions.get(sym, 0) + fill_qty
                        remaining_volume -= fill_qty
                        leftover = order.quantity - fill_qty
                        if leftover > 0:
                            new_resting.append(Order(sym, order.price, leftover))
                    else:
                        new_resting.append(order)
                else:
                    new_resting.append(order)

            elif order.quantity < 0:
                # Resting SELL: fill if our ask <= trade price (we're a better ask)
                if order.price <= trade_price:
                    sell_room = limit + positions.get(sym, 0)
                    fill_qty = min(abs(order.quantity), remaining_volume, sell_room)
                    if fill_qty > 0:
                        fills.append(Trade(
                            symbol=sym, price=order.price, quantity=fill_qty,
                            buyer="BOT", seller="SUBMISSION", timestamp=0,
                        ))
                        positions[sym] = positions.get(sym, 0) - fill_qty
                        remaining_volume -= fill_qty
                        leftover = abs(order.quantity) - fill_qty
                        if leftover > 0:
                            new_resting.append(Order(sym, order.price, -leftover))
                    else:
                        new_resting.append(order)
                else:
                    new_resting.append(order)
            else:
                new_resting.append(order)

        resting_orders[sym] = new_resting

    return fills


# ── PnL Tracking ─────────────────────────────────────────────────────────────

class PnLTracker:
    def __init__(self):
        self.position: Dict[str, int] = {}
        self.cash: Dict[str, float] = {}
        self.buy_count: Dict[str, int] = {}
        self.sell_count: Dict[str, int] = {}
        self.buy_volume: Dict[str, int] = {}
        self.sell_volume: Dict[str, int] = {}
        self.fill_log: List[dict] = []

    def record_fill(self, trade: Trade):
        product = trade.symbol
        self.position.setdefault(product, 0)
        self.cash.setdefault(product, 0.0)
        self.buy_count.setdefault(product, 0)
        self.sell_count.setdefault(product, 0)
        self.buy_volume.setdefault(product, 0)
        self.sell_volume.setdefault(product, 0)

        if trade.buyer == "SUBMISSION":
            self.position[product] += trade.quantity
            self.cash[product] -= trade.price * trade.quantity
            self.buy_count[product] += 1
            self.buy_volume[product] += trade.quantity
        elif trade.seller == "SUBMISSION":
            self.position[product] -= trade.quantity
            self.cash[product] += trade.price * trade.quantity
            self.sell_count[product] += 1
            self.sell_volume[product] += trade.quantity

        self.fill_log.append({
            "timestamp": trade.timestamp,
            "product": product,
            "side": "BUY" if trade.buyer == "SUBMISSION" else "SELL",
            "price": trade.price,
            "qty": trade.quantity,
        })

    def pnl(self, mid_prices: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for product in set(list(self.position.keys()) + list(self.cash.keys())):
            pos = self.position.get(product, 0)
            cash = self.cash.get(product, 0.0)
            mid = mid_prices.get(product, 0.0)
            result[product] = cash + pos * mid
        return result

    def total_pnl(self, mid_prices: Dict[str, float]) -> float:
        return sum(self.pnl(mid_prices).values())


# ── Main Backtest Loop ───────────────────────────────────────────────────────

def run_backtest(day: int, verbose: bool = False) -> Tuple[PnLTracker, Dict[str, float]]:
    prices_path = DATA_DIR / f"prices_round_0_day_{day}.csv"
    trades_path = DATA_DIR / f"trades_round_0_day_{day}.csv"

    if not prices_path.exists():
        raise FileNotFoundError(f"Missing {prices_path}")

    snapshots = load_prices(prices_path)
    market_trades_data = load_trades(trades_path) if trades_path.exists() else {}

    timestamps = sorted(snapshots.keys())
    products = sorted({p for ts_data in snapshots.values() for p in ts_data})

    trader = Trader()
    tracker = PnLTracker()
    trader_data = ""
    positions: Dict[str, int] = {p: 0 for p in products}
    own_trades_history: Dict[str, List[Trade]] = {p: [] for p in products}
    resting_orders: Dict[str, List[Order]] = {p: [] for p in products}
    mid_prices: Dict[str, float] = {}

    listings = {p: Listing(symbol=p, product=p, denomination="XIRECS") for p in products}

    print(f"\n{'='*70}")
    print(f"  BACKTEST: Day {day} | {len(timestamps)} ticks | Products: {', '.join(products)}")
    print(f"{'='*70}")

    for ts in timestamps:
        ts_data = snapshots[ts]

        # ── 1. Fill resting passive orders using market trades at this tick ──
        mkt_trades_now = market_trades_data.get(ts, [])
        if mkt_trades_now:
            passive_fills = match_passive(resting_orders, mkt_trades_now, positions)
            for fill in passive_fills:
                fill.timestamp = ts
                tracker.record_fill(fill)
                own_trades_history.setdefault(fill.symbol, []).append(fill)

                if verbose:
                    side = "BUY" if fill.buyer == "SUBMISSION" else "SELL"
                    print(f"  [{ts:>6}] {side:>4} {fill.symbol:<10} {fill.quantity:>3} @ {fill.price:>6} [passive] | pos={positions[fill.symbol]:>4}")

        # ── 2. Build order depths and market state ───────────────────────────
        order_depths: Dict[str, OrderDepth] = {}
        for product in products:
            if product in ts_data:
                order_depths[product] = build_order_depth(ts_data[product])
                mid_prices[product] = get_mid(ts_data[product])
            else:
                order_depths[product] = OrderDepth()

        # Build market trades for TradingState
        market_trades_for_state: Dict[str, List[Trade]] = {p: [] for p in products}
        for td in mkt_trades_now:
            sym = td["symbol"]
            if sym in market_trades_for_state:
                market_trades_for_state[sym].append(Trade(
                    symbol=sym,
                    price=int(float(td["price"])),
                    quantity=int(float(td["quantity"])),
                    buyer=td.get("buyer", ""),
                    seller=td.get("seller", ""),
                    timestamp=ts,
                ))

        # ── 3. Run trader ────────────────────────────────────────────────────
        state = TradingState(
            traderData=trader_data,
            timestamp=ts,
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades_history,
            market_trades=market_trades_for_state,
            position=dict(positions),
            observations=Observation({}, {}),
        )

        try:
            result, conversions, trader_data = trader.run(state)
        except Exception as e:
            print(f"  [ERROR] Timestamp {ts}: {e}")
            continue

        # Reset own_trades for next tick
        own_trades_history = {p: [] for p in products}

        # ── 4. Match aggressive orders, collect passive remainder ────────────
        resting_orders = {p: [] for p in products}

        for product, orders in result.items():
            if not orders:
                continue
            limit = POSITION_LIMITS.get(product, DEFAULT_LIMIT)

            # Fresh copy of order book for matching
            if product in ts_data:
                match_od = build_order_depth(ts_data[product])
            else:
                continue

            aggressive_fills, passive_remainder = match_aggressive(
                orders, match_od, product, positions.get(product, 0), limit,
            )

            for fill in aggressive_fills:
                fill.timestamp = ts
                tracker.record_fill(fill)
                own_trades_history.setdefault(product, []).append(fill)

                if fill.buyer == "SUBMISSION":
                    positions[product] = positions.get(product, 0) + fill.quantity
                else:
                    positions[product] = positions.get(product, 0) - fill.quantity

                if verbose:
                    side = "BUY" if fill.buyer == "SUBMISSION" else "SELL"
                    print(f"  [{ts:>6}] {side:>4} {product:<10} {fill.quantity:>3} @ {fill.price:>6} [aggressive] | pos={positions[product]:>4}")

            resting_orders[product] = passive_remainder

        # ── Periodic summary ─────────────────────────────────────────────────
        if ts % 50000 == 0 and ts > 0:
            pnl = tracker.total_pnl(mid_prices)
            pos_str = " | ".join(f"{p}={positions.get(p,0):+d}" for p in products)
            print(f"  [{ts:>6}] PnL: {pnl:>8.0f} | {pos_str}")

    # ── Final Summary ────────────────────────────────────────────────────────
    final_pnl = tracker.pnl(mid_prices)
    total = sum(final_pnl.values())

    print(f"\n{'─'*70}")
    print(f"  RESULTS: Day {day}")
    print(f"{'─'*70}")
    print(f"  {'Product':<10} {'Pos':>5} {'PnL':>9} {'Buys':>6} {'Sells':>6} {'BuyVol':>7} {'SellVol':>8}")
    print(f"  {'─'*10} {'─'*5} {'─'*9} {'─'*6} {'─'*6} {'─'*7} {'─'*8}")
    for product in products:
        pos = positions.get(product, 0)
        pnl = final_pnl.get(product, 0.0)
        buys = tracker.buy_count.get(product, 0)
        sells = tracker.sell_count.get(product, 0)
        bvol = tracker.buy_volume.get(product, 0)
        svol = tracker.sell_volume.get(product, 0)
        print(f"  {product:<10} {pos:>5} {pnl:>9.0f} {buys:>6} {sells:>6} {bvol:>7} {svol:>8}")
    print(f"  {'─'*10} {'─'*5} {'─'*9}")
    print(f"  {'TOTAL':<10} {'':>5} {total:>9.0f}")
    print(f"{'─'*70}\n")

    return tracker, mid_prices


def main():
    parser = argparse.ArgumentParser(description="Local backtester for IMC Prosperity")
    parser.add_argument("--day", type=int, choices=[-1, -2], help="Day to backtest (-1 or -2). Default: both")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print every fill")
    args = parser.parse_args()

    days = [args.day] if args.day else [-2, -1]
    grand_total = 0.0

    for day in days:
        prices_path = DATA_DIR / f"prices_round_0_day_{day}.csv"
        if not prices_path.exists():
            print(f"Skipping day {day}: {prices_path} not found")
            continue
        tracker, mid_prices = run_backtest(day, verbose=args.verbose)
        grand_total += tracker.total_pnl(mid_prices)

    if len(days) > 1:
        print(f"{'='*70}")
        print(f"  COMBINED PnL: {grand_total:>9.0f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
