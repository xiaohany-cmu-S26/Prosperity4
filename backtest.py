"""
Local backtester for IMC Prosperity trading strategies.

Adapted from the jmerle/imc-prosperity-3-backtester project for Prosperity 4.
Replays tutorial round CSV data through any algo file's Trader.run(), simulating
order matching against the historical order book and market trades.

Matching model:
  - Aggressive fills: orders that cross the current book fill immediately
  - Market trade fills: remaining order volume matched against historical market
    trades at prices favorable to the trader's quote

Usage:
  python3 backtest.py                          # Run trading.py on all available days
  python3 backtest.py --algo my_strat.py       # Run a different algo file
  python3 backtest.py --day -1                 # Run day -1 only
  python3 backtest.py --match-trades worse     # Stricter trade matching
  python3 backtest.py --print                  # Print trader stdout
  python3 backtest.py --no-out                 # Skip saving output log
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    Symbol,
    Trade,
    TradingState,
)

DEFAULT_POSITION_LIMIT = 80


# ── Algo Loading ──────────────────────────────────────────────────────────────

def load_algo(algo_path: Path):
    """Dynamically load a trading algo module from a .py file.

    Returns (Trader class, position_limits dict, default_limit int).
    """
    spec = importlib.util.spec_from_file_location("algo", algo_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "Trader"):
        raise AttributeError(f"{algo_path} has no Trader class")

    position_limits = getattr(mod, "POSITION_LIMITS", {})
    default_limit = getattr(mod, "DEFAULT_LIMIT", DEFAULT_POSITION_LIMIT)

    return mod.Trader, position_limits, default_limit


# ── Data Models ───────────────────────────────────────────────────────────────

class TradeMatchingMode(str, Enum):
    all = "all"
    worse = "worse"
    none = "none"


@dataclass
class MarketTrade:
    trade: Trade
    buy_quantity: int
    sell_quantity: int


@dataclass
class PriceRow:
    day: int
    timestamp: int
    product: Symbol
    bid_prices: List[int]
    bid_volumes: List[int]
    ask_prices: List[int]
    ask_volumes: List[int]
    mid_price: float
    profit_loss: float


@dataclass
class BacktestData:
    round_num: int
    day_num: int
    prices: Dict[int, Dict[Symbol, PriceRow]]
    trades: Dict[int, Dict[Symbol, List[Trade]]]
    products: List[Symbol]
    profit_loss: Dict[Symbol, float]


@dataclass
class SandboxLogRow:
    timestamp: int
    sandbox_log: str
    lambda_log: str

    def with_offset(self, timestamp_offset: int) -> "SandboxLogRow":
        return SandboxLogRow(
            self.timestamp + timestamp_offset,
            self.sandbox_log,
            self.lambda_log.replace(
                f"[[{self.timestamp},", f"[[{self.timestamp + timestamp_offset},"
            ),
        )

    def __str__(self) -> str:
        return json.dumps(
            {
                "sandboxLog": self.sandbox_log,
                "lambdaLog": self.lambda_log,
                "timestamp": self.timestamp,
            },
            indent=2,
        ) + "\n"


@dataclass
class ActivityLogRow:
    columns: List[Any]

    @property
    def timestamp(self) -> int:
        return self.columns[1]

    def with_offset(
        self, timestamp_offset: int, profit_loss_offset: float = 0.0
    ) -> "ActivityLogRow":
        new_columns = self.columns[:]
        new_columns[1] += timestamp_offset
        new_columns[-1] += profit_loss_offset
        return ActivityLogRow(new_columns)

    def __str__(self) -> str:
        return ";".join(map(str, self.columns))


@dataclass
class TradeRow:
    trade: Trade

    @property
    def timestamp(self) -> int:
        return self.trade.timestamp

    def with_offset(self, timestamp_offset: int) -> "TradeRow":
        return TradeRow(
            Trade(
                self.trade.symbol,
                self.trade.price,
                self.trade.quantity,
                self.trade.buyer,
                self.trade.seller,
                self.trade.timestamp + timestamp_offset,
            )
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.trade.timestamp,
            "buyer": self.trade.buyer,
            "seller": self.trade.seller,
            "symbol": self.trade.symbol,
            "currency": "SEASHELLS",
            "price": self.trade.price,
            "quantity": self.trade.quantity,
        }

    def __str__(self) -> str:
        t = self.trade
        return (
            f'  {{\n'
            f'    "timestamp": {t.timestamp},\n'
            f'    "buyer": "{t.buyer}",\n'
            f'    "seller": "{t.seller}",\n'
            f'    "symbol": "{t.symbol}",\n'
            f'    "currency": "SEASHELLS",\n'
            f'    "price": {t.price},\n'
            f'    "quantity": {t.quantity}\n'
            f'  }}'
        )


@dataclass
class BacktestResult:
    round_num: int
    day_num: int
    sandbox_logs: List[SandboxLogRow] = field(default_factory=list)
    activity_logs: List[ActivityLogRow] = field(default_factory=list)
    trades: List[TradeRow] = field(default_factory=list)


# ── Data Loading ──────────────────────────────────────────────────────────────

def find_data_file(
    data_dir: Path, round_num: int, day_num: int, prefix: str
) -> Optional[Path]:
    """Try flat layout first, then P3-style subdirectory layout."""
    flat = data_dir / f"{prefix}_round_{round_num}_day_{day_num}.csv"
    if flat.is_file():
        return flat
    nested = data_dir / f"round{round_num}" / f"{prefix}_round_{round_num}_day_{day_num}.csv"
    if nested.is_file():
        return nested
    return None


def get_column_values(columns: List[str], indices: List[int]) -> List[int]:
    values = []
    for index in indices:
        value = columns[index]
        if value == "":
            break
        values.append(int(value))
    return values


def has_day_data(data_dir: Path, round_num: int, day_num: int) -> bool:
    return find_data_file(data_dir, round_num, day_num, "prices") is not None


def detect_days(data_dir: Path, round_num: int) -> List[int]:
    """Scan data directory for available day numbers."""
    pattern = re.compile(rf"prices_round_{round_num}_day_(-?\d+)\.csv")
    days = []

    # Scan flat layout
    if data_dir.is_dir():
        for f in data_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                days.append(int(m.group(1)))

    # Scan nested layout
    nested = data_dir / f"round{round_num}"
    if nested.is_dir():
        for f in nested.iterdir():
            m = pattern.match(f.name)
            if m:
                d = int(m.group(1))
                if d not in days:
                    days.append(d)

    return sorted(days)


def read_day_data(
    data_dir: Path, round_num: int, day_num: int
) -> BacktestData:
    prices_path = find_data_file(data_dir, round_num, day_num, "prices")
    if prices_path is None:
        raise ValueError(
            f"Prices data not available for round {round_num} day {day_num}"
        )

    prices = []
    for line in prices_path.read_text(encoding="utf-8").splitlines()[1:]:
        columns = line.split(";")
        prices.append(
            PriceRow(
                day=int(columns[0]),
                timestamp=int(columns[1]),
                product=columns[2],
                bid_prices=get_column_values(columns, [3, 5, 7]),
                bid_volumes=get_column_values(columns, [4, 6, 8]),
                ask_prices=get_column_values(columns, [9, 11, 13]),
                ask_volumes=get_column_values(columns, [10, 12, 14]),
                mid_price=float(columns[15]),
                profit_loss=float(columns[16]),
            )
        )

    trades = []
    trades_path = find_data_file(data_dir, round_num, day_num, "trades")
    if trades_path is not None:
        for line in trades_path.read_text(encoding="utf-8").splitlines()[1:]:
            columns = line.split(";")
            trades.append(
                Trade(
                    symbol=columns[3],
                    price=int(float(columns[5])),
                    quantity=int(columns[6]),
                    buyer=columns[1],
                    seller=columns[2],
                    timestamp=int(columns[0]),
                )
            )

    # Build indexed structures
    prices_by_ts: Dict[int, Dict[Symbol, PriceRow]] = defaultdict(dict)
    for row in prices:
        prices_by_ts[row.timestamp][row.product] = row

    trades_by_ts: Dict[int, Dict[Symbol, List[Trade]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trade in trades:
        trades_by_ts[trade.timestamp][trade.symbol].append(trade)

    products = sorted(set(row.product for row in prices))
    profit_loss = {product: 0.0 for product in products}

    return BacktestData(
        round_num=round_num,
        day_num=day_num,
        prices=prices_by_ts,
        trades=trades_by_ts,
        products=products,
        profit_loss=profit_loss,
    )


# ── Runner / Matching Engine ─────────────────────────────────────────────────

def prepare_state(state: TradingState, data: BacktestData) -> None:
    for product in data.products:
        order_depth = OrderDepth()
        row = data.prices[state.timestamp][product]

        for price, volume in zip(row.bid_prices, row.bid_volumes):
            order_depth.buy_orders[price] = volume

        for price, volume in zip(row.ask_prices, row.ask_volumes):
            order_depth.sell_orders[price] = -volume

        state.order_depths[product] = order_depth
        state.listings[product] = Listing(product, product, "SEASHELLS")

    state.observations = Observation({}, {})


def type_check_orders(orders: Dict[Symbol, List[Order]]) -> None:
    for key, value in orders.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Orders key '{key}' is of type {type(key)}, expected a str"
            )
        for order in value:
            if not isinstance(order.symbol, str):
                raise ValueError(
                    f"Order symbol of '{order}' is of type {type(order.symbol)}, expected a str"
                )
            if not isinstance(order.price, int):
                raise ValueError(
                    f"Order price of '{order}' is of type {type(order.price)}, expected an int"
                )
            if not isinstance(order.quantity, int):
                raise ValueError(
                    f"Order quantity of '{order}' is of type {type(order.quantity)}, expected an int"
                )


def create_activity_logs(
    state: TradingState,
    data: BacktestData,
    result: BacktestResult,
) -> None:
    for product in data.products:
        row = data.prices[state.timestamp][product]

        product_profit_loss = data.profit_loss[product]

        position = state.position.get(product, 0)
        if position != 0:
            product_profit_loss += position * row.mid_price

        bp_len = len(row.bid_prices)
        bv_len = len(row.bid_volumes)
        ap_len = len(row.ask_prices)
        av_len = len(row.ask_volumes)

        columns = [
            result.day_num,
            state.timestamp,
            product,
            row.bid_prices[0] if bp_len > 0 else "",
            row.bid_volumes[0] if bv_len > 0 else "",
            row.bid_prices[1] if bp_len > 1 else "",
            row.bid_volumes[1] if bv_len > 1 else "",
            row.bid_prices[2] if bp_len > 2 else "",
            row.bid_volumes[2] if bv_len > 2 else "",
            row.ask_prices[0] if ap_len > 0 else "",
            row.ask_volumes[0] if av_len > 0 else "",
            row.ask_prices[1] if ap_len > 1 else "",
            row.ask_volumes[1] if av_len > 1 else "",
            row.ask_prices[2] if ap_len > 2 else "",
            row.ask_volumes[2] if av_len > 2 else "",
            row.mid_price,
            product_profit_loss,
        ]

        result.activity_logs.append(ActivityLogRow(columns))


def enforce_limits(
    state: TradingState,
    data: BacktestData,
    orders: Dict[Symbol, List[Order]],
    sandbox_row: SandboxLogRow,
    limits: Dict[str, int],
    default_limit: int,
) -> None:
    sandbox_log_lines = []
    for product in data.products:
        product_orders = orders.get(product, [])
        product_position = state.position.get(product, 0)
        limit = limits.get(product, default_limit)

        total_long = sum(o.quantity for o in product_orders if o.quantity > 0)
        total_short = sum(abs(o.quantity) for o in product_orders if o.quantity < 0)

        if (
            product_position + total_long > limit
            or product_position - total_short < -limit
        ):
            sandbox_log_lines.append(
                f"Orders for product {product} exceeded limit of {limit} set"
            )
            orders.pop(product, None)

    if sandbox_log_lines:
        sandbox_row.sandbox_log += "\n" + "\n".join(sandbox_log_lines)


def match_buy_order(
    state: TradingState,
    data: BacktestData,
    order: Order,
    market_trades: List[MarketTrade],
    trade_matching_mode: TradeMatchingMode,
) -> List[Trade]:
    trades = []

    order_depth = state.order_depths[order.symbol]
    price_matches = sorted(
        price for price in order_depth.sell_orders.keys() if price <= order.price
    )

    for price in price_matches:
        volume = min(order.quantity, abs(order_depth.sell_orders[price]))

        trades.append(
            Trade(order.symbol, price, volume, "SUBMISSION", "", state.timestamp)
        )

        state.position[order.symbol] = state.position.get(order.symbol, 0) + volume
        data.profit_loss[order.symbol] -= price * volume

        order_depth.sell_orders[price] += volume
        if order_depth.sell_orders[price] == 0:
            order_depth.sell_orders.pop(price)

        order.quantity -= volume
        if order.quantity == 0:
            return trades

    if trade_matching_mode == TradeMatchingMode.none:
        return trades

    for market_trade in market_trades:
        if (
            market_trade.sell_quantity == 0
            or market_trade.trade.price > order.price
            or (
                market_trade.trade.price == order.price
                and trade_matching_mode == TradeMatchingMode.worse
            )
        ):
            continue

        volume = min(order.quantity, market_trade.sell_quantity)

        trades.append(
            Trade(
                order.symbol,
                order.price,
                volume,
                "SUBMISSION",
                market_trade.trade.seller,
                state.timestamp,
            )
        )

        state.position[order.symbol] = state.position.get(order.symbol, 0) + volume
        data.profit_loss[order.symbol] -= order.price * volume

        market_trade.sell_quantity -= volume

        order.quantity -= volume
        if order.quantity == 0:
            return trades

    return trades


def match_sell_order(
    state: TradingState,
    data: BacktestData,
    order: Order,
    market_trades: List[MarketTrade],
    trade_matching_mode: TradeMatchingMode,
) -> List[Trade]:
    trades = []

    order_depth = state.order_depths[order.symbol]
    price_matches = sorted(
        (price for price in order_depth.buy_orders.keys() if price >= order.price),
        reverse=True,
    )

    for price in price_matches:
        volume = min(abs(order.quantity), order_depth.buy_orders[price])

        trades.append(
            Trade(order.symbol, price, volume, "", "SUBMISSION", state.timestamp)
        )

        state.position[order.symbol] = state.position.get(order.symbol, 0) - volume
        data.profit_loss[order.symbol] += price * volume

        order_depth.buy_orders[price] -= volume
        if order_depth.buy_orders[price] == 0:
            order_depth.buy_orders.pop(price)

        order.quantity += volume
        if order.quantity == 0:
            return trades

    if trade_matching_mode == TradeMatchingMode.none:
        return trades

    for market_trade in market_trades:
        if (
            market_trade.buy_quantity == 0
            or market_trade.trade.price < order.price
            or (
                market_trade.trade.price == order.price
                and trade_matching_mode == TradeMatchingMode.worse
            )
        ):
            continue

        volume = min(abs(order.quantity), market_trade.buy_quantity)

        trades.append(
            Trade(
                order.symbol,
                order.price,
                volume,
                market_trade.trade.buyer,
                "SUBMISSION",
                state.timestamp,
            )
        )

        state.position[order.symbol] = state.position.get(order.symbol, 0) - volume
        data.profit_loss[order.symbol] += order.price * volume

        market_trade.buy_quantity -= volume

        order.quantity += volume
        if order.quantity == 0:
            return trades

    return trades


def match_order(
    state: TradingState,
    data: BacktestData,
    order: Order,
    market_trades: List[MarketTrade],
    trade_matching_mode: TradeMatchingMode,
) -> List[Trade]:
    if order.quantity > 0:
        return match_buy_order(state, data, order, market_trades, trade_matching_mode)
    elif order.quantity < 0:
        return match_sell_order(state, data, order, market_trades, trade_matching_mode)
    else:
        return []


def match_orders(
    state: TradingState,
    data: BacktestData,
    orders: Dict[Symbol, List[Order]],
    result: BacktestResult,
    trade_matching_mode: TradeMatchingMode,
) -> None:
    market_trades = {
        product: [MarketTrade(t, t.quantity, t.quantity) for t in trades]
        for product, trades in data.trades[state.timestamp].items()
    }

    for product in data.products:
        new_trades = []

        for order in orders.get(product, []):
            new_trades.extend(
                match_order(
                    state,
                    data,
                    order,
                    market_trades.get(product, []),
                    trade_matching_mode,
                )
            )

        if new_trades:
            state.own_trades[product] = new_trades
            result.trades.extend([TradeRow(trade) for trade in new_trades])

    for product, trades in market_trades.items():
        for trade in trades:
            trade.trade.quantity = min(trade.buy_quantity, trade.sell_quantity)

        remaining = [t.trade for t in trades if t.trade.quantity > 0]
        state.market_trades[product] = remaining
        result.trades.extend([TradeRow(trade) for trade in remaining])


# ── Main Backtest Loop ────────────────────────────────────────────────────────

def run_backtest(
    trader,
    data_dir: Path,
    round_num: int,
    day_num: int,
    limits: Dict[str, int],
    default_limit: int,
    print_output: bool,
    trade_matching_mode: TradeMatchingMode,
) -> BacktestResult:
    data = read_day_data(data_dir, round_num, day_num)

    os.environ["PROSPERITY4BT_ROUND"] = str(round_num)
    os.environ["PROSPERITY4BT_DAY"] = str(day_num)

    trader_data = ""
    state = TradingState(
        traderData=trader_data,
        timestamp=0,
        listings={},
        order_depths={},
        own_trades={},
        market_trades={},
        position={},
        observations=Observation({}, {}),
    )

    result = BacktestResult(round_num=round_num, day_num=day_num)

    timestamps = sorted(data.prices.keys())
    total = len(timestamps)

    for i, timestamp in enumerate(timestamps):
        state.timestamp = timestamp
        state.traderData = trader_data

        prepare_state(state, data)

        stdout = StringIO()

        if print_output:
            # No Tee — just capture and print after
            with redirect_stdout(stdout):
                orders, conversions, trader_data = trader.run(state)
            captured = stdout.getvalue()
            if captured:
                sys.stdout.write(captured)
        else:
            with redirect_stdout(stdout):
                orders, conversions, trader_data = trader.run(state)

        sandbox_row = SandboxLogRow(
            timestamp=timestamp,
            sandbox_log="",
            lambda_log=stdout.getvalue().rstrip(),
        )
        result.sandbox_logs.append(sandbox_row)

        type_check_orders(orders)
        create_activity_logs(state, data, result)
        enforce_limits(state, data, orders, sandbox_row, limits, default_limit)
        match_orders(state, data, orders, result, trade_matching_mode)

        # Progress indicator
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(
                f"\r  Progress: {i + 1}/{total} timestamps "
                f"({100 * (i + 1) // total}%)",
                end="",
                flush=True,
            )

    print()  # newline after progress
    return result


# ── Output Generation ─────────────────────────────────────────────────────────

def write_log_output(output_file: Path, merged: BacktestResult) -> None:
    """Write P3-style .log file (Sandbox logs / Activities log / Trade History)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w+", encoding="utf-8") as f:
        f.write("Sandbox logs:\n")
        for row in merged.sandbox_logs:
            f.write(str(row))

        f.write("\n\n\nActivities log:\n")
        f.write(
            "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;"
            "bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;"
            "ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;"
            "profit_and_loss\n"
        )
        f.write("\n".join(map(str, merged.activity_logs)))

        f.write("\n\n\n\n\nTrade History:\n")
        f.write("[\n")
        f.write(",\n".join(map(str, merged.trades)))
        f.write("]")


def write_json_output(output_file: Path, merged: BacktestResult) -> None:
    """Write JSON matching IMC server log format (for visualize_log.py)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build activitiesLog CSV string
    header = (
        "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;"
        "bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;"
        "ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;"
        "profit_and_loss"
    )
    activities_csv = header + "\n" + "\n".join(map(str, merged.activity_logs))

    # Build graphLog: total PnL per timestamp
    pnl_by_ts: Dict[int, float] = defaultdict(float)
    for row in merged.activity_logs:
        pnl_by_ts[row.timestamp] += row.columns[-1]

    graph_lines = ["timestamp;value"]
    for ts in sorted(pnl_by_ts.keys()):
        graph_lines.append(f"{ts};{pnl_by_ts[ts]}")
    graph_log = "\n".join(graph_lines)

    # Final positions
    positions_by_product: Dict[str, int] = {}
    if merged.activity_logs:
        last_ts = merged.activity_logs[-1].timestamp
        # Compute from trades
        for tr in merged.trades:
            t = tr.trade
            if t.buyer == "SUBMISSION":
                positions_by_product[t.symbol] = positions_by_product.get(t.symbol, 0) + t.quantity
            elif t.seller == "SUBMISSION":
                positions_by_product[t.symbol] = positions_by_product.get(t.symbol, 0) - t.quantity

    positions = [
        {"symbol": sym, "quantity": qty}
        for sym, qty in sorted(positions_by_product.items())
    ]

    # Total profit
    total_profit = 0.0
    if merged.activity_logs:
        last_ts = merged.activity_logs[-1].timestamp
        for row in reversed(merged.activity_logs):
            if row.timestamp != last_ts:
                break
            total_profit += row.columns[-1]

    # Trade history (only our trades)
    trade_history = [
        tr.to_dict()
        for tr in merged.trades
        if tr.trade.buyer == "SUBMISSION" or tr.trade.seller == "SUBMISSION"
    ]

    output = {
        "round": str(merged.round_num),
        "status": "FINISHED",
        "profit": total_profit,
        "activitiesLog": activities_csv,
        "graphLog": graph_log,
        "tradeHistory": trade_history,
        "positions": positions,
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output, f)


# ── CLI Helpers ───────────────────────────────────────────────────────────────

def print_day_summary(result: BacktestResult) -> None:
    last_timestamp = result.activity_logs[-1].timestamp

    product_lines = []
    total_profit = 0.0

    for row in reversed(result.activity_logs):
        if row.timestamp != last_timestamp:
            break
        product = row.columns[2]
        profit = row.columns[-1]
        product_lines.append(f"  {product}: {profit:,.0f}")
        total_profit += profit

    for line in reversed(product_lines):
        print(line)
    print(f"  Total profit: {total_profit:,.0f}")


def merge_results(
    a: BacktestResult,
    b: BacktestResult,
    merge_profit_loss: bool,
    merge_timestamps: bool,
) -> BacktestResult:
    sandbox_logs = a.sandbox_logs[:]
    activity_logs = a.activity_logs[:]
    trades = a.trades[:]

    if merge_timestamps:
        a_last_timestamp = a.activity_logs[-1].timestamp
        timestamp_offset = a_last_timestamp + 100
    else:
        timestamp_offset = 0

    sandbox_logs.extend(
        [row.with_offset(timestamp_offset) for row in b.sandbox_logs]
    )
    trades.extend([row.with_offset(timestamp_offset) for row in b.trades])

    if merge_profit_loss:
        profit_loss_offsets: Dict[str, float] = defaultdict(float)
        a_last_ts = a.activity_logs[-1].timestamp
        for row in reversed(a.activity_logs):
            if row.timestamp != a_last_ts:
                break
            profit_loss_offsets[row.columns[2]] = row.columns[-1]

        activity_logs.extend(
            [
                row.with_offset(timestamp_offset, profit_loss_offsets[row.columns[2]])
                for row in b.activity_logs
            ]
        )
    else:
        activity_logs.extend(
            [row.with_offset(timestamp_offset, 0) for row in b.activity_logs]
        )

    return BacktestResult(
        a.round_num, a.day_num, sandbox_logs, activity_logs, trades
    )


def print_overall_summary(results: List[BacktestResult]) -> None:
    print("\nProfit summary:")
    total_profit = 0.0
    for result in results:
        last_ts = result.activity_logs[-1].timestamp
        profit = 0.0
        for row in reversed(result.activity_logs):
            if row.timestamp != last_ts:
                break
            profit += row.columns[-1]
        print(f"  Round {result.round_num} day {result.day_num}: {profit:,.0f}")
        total_profit += profit
    print(f"  Total profit: {total_profit:,.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local backtester for IMC Prosperity strategies"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="trading.py",
        help="Path to algo file (default: trading.py)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="TUTORIAL_ROUND_1",
        help="Path to data directory (default: TUTORIAL_ROUND_1)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=0,
        help="Round number (default: 0)",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=None,
        help="Day to backtest (e.g., -1). Default: all available days",
    )
    parser.add_argument(
        "--match-trades",
        type=str,
        choices=["all", "worse", "none"],
        default="all",
        help="How to match orders against market trades (default: all)",
    )
    parser.add_argument(
        "--merge-pnl",
        action="store_true",
        help="Merge profit and loss across days",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_output",
        help="Print trader stdout while running",
    )
    parser.add_argument(
        "--no-out",
        action="store_true",
        help="Skip saving output log",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Custom output file path (default: logs/backtest_<algo>.json)",
    )
    args = parser.parse_args()

    algo_path = Path(args.algo)
    # Auto-add .py extension if missing
    if not algo_path.suffix:
        algo_path = algo_path.with_suffix(".py")
    # Try logs/ prefix if not found
    if not algo_path.exists():
        candidate = Path("logs") / algo_path
        if candidate.exists():
            algo_path = candidate
    if not algo_path.exists():
        print(f"Error: algo file not found: {algo_path}")
        sys.exit(1)

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    algo_name = algo_path.stem
    print(f"Loading algo: {algo_path}")
    trader_cls, position_limits, default_limit = load_algo(algo_path)

    round_num = args.round
    trade_matching_mode = TradeMatchingMode(args.match_trades)

    # Determine days to run
    if args.day is not None:
        days = [args.day]
        if not has_day_data(data_dir, round_num, args.day):
            print(f"Error: no data for round {round_num} day {args.day}")
            sys.exit(1)
    else:
        days = detect_days(data_dir, round_num)
        if not days:
            print(f"Error: no data found in {data_dir} for round {round_num}")
            sys.exit(1)

    print(
        f"Round {round_num} | Days: {days} | "
        f"Match mode: {trade_matching_mode.value} | "
        f"Limits: {position_limits or {f'default={default_limit}'}}"
    )

    results = []
    for day_num in days:
        print(f"\n{'='*60}")
        print(f"  Round {round_num} Day {day_num}")
        print(f"{'='*60}")

        trader = trader_cls()
        result = run_backtest(
            trader,
            data_dir,
            round_num,
            day_num,
            position_limits,
            default_limit,
            args.print_output,
            trade_matching_mode,
        )

        print_day_summary(result)
        results.append(result)

    if len(results) > 1:
        print_overall_summary(results)

    # Write output
    if not args.no_out and results:
        if args.out:
            output_path = Path(args.out)
        else:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            output_path = log_dir / f"backtest_{algo_name}.json"

        merged = reduce(
            lambda a, b: merge_results(a, b, args.merge_pnl, True), results
        )
        write_json_output(output_path, merged)

        # Also write .log for P3-style visualizer compatibility
        log_path = output_path.with_suffix(".log")
        write_log_output(log_path, merged)

        print(f"\nOutput saved to {output_path}")
        print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
