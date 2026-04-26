"""
Local backtester for IMC Prosperity trading strategies.

Matching model mirrors the Rust backtester (prosperity_rust_backtester):
  Phase 1 – Aggressive fill: orders that cross the current book fill immediately
             at the book price.
  Phase 2 – Market trade fill: remaining unfilled volume is matched against
             historical market trades.  Queue position is modelled: the existing
             book volume at each price level is consumed ahead of our order
             before we can fill there.

Additional parameters (mirroring Rust CLI flags):
  --queue-penetration  fraction of each market-trade's volume that is
                       reachable (default 1.0 = full volume)
  --slippage-bps       basis-point price adjustment applied to every fill
                       (default 0.0 = no slippage)

Usage:
  python backtest.py                          # Run trading.py on all available days
  python backtest.py --algo my_strat.py       # Run a different algo file
  python backtest.py --day -1                 # Run day -1 only
  python backtest.py --match-trades worse     # Stricter trade matching
  python backtest.py --queue-penetration 0.5  # Only 50% of market-trade volume reachable
  python backtest.py --slippage-bps 1         # 1 bps fill-price slippage
  python backtest.py --print                  # Print trader stdout
  python backtest.py --no-out                 # Skip saving output log
  python backtest.py --data prosperity_rust_backtester-main/datasets --round 1
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

# ── Position limits (mirrors Rust backtester's hardcoded table) ───────────────

KNOWN_POSITION_LIMITS: Dict[str, int] = {
    # Round 3
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
    # Round 1 / 2
    "EMERALDS": 80,
    "TOMATOES": 80,
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM": 80,
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75,
}
DEFAULT_POSITION_LIMIT = 100


# ── Algo Loading ──────────────────────────────────────────────────────────────

def load_algo(algo_path: Path):
    """Dynamically load a trading algo module from a .py file.

    Returns (Trader class, position_limits dict, default_limit int).
    Limits from the algo override the known-limits table; unknown products
    fall back to the known table, then to DEFAULT_POSITION_LIMIT.
    """
    spec = importlib.util.spec_from_file_location("algo", algo_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "Trader"):
        raise AttributeError(f"{algo_path} has no Trader class")

    default_limit = getattr(mod, "DEFAULT_LIMIT", DEFAULT_POSITION_LIMIT)

    # Collect explicit limits from the algo (POSITION_LIMITS or PRODUCT_CONFIG).
    algo_limits: Dict[str, int] = {}
    if hasattr(mod, "POSITION_LIMITS"):
        algo_limits = mod.POSITION_LIMITS
    elif hasattr(mod, "PRODUCT_CONFIG"):
        algo_limits = {
            product: cfg.get("limit", default_limit)
            for product, cfg in mod.PRODUCT_CONFIG.items()
        }

    # Pick up option limits declared as OPTION_STRIKES + OPTION_LIMIT.
    if hasattr(mod, "OPTION_STRIKES") and hasattr(mod, "OPTION_LIMIT"):
        option_lim = int(mod.OPTION_LIMIT)
        for product in mod.OPTION_STRIKES:
            algo_limits.setdefault(product, option_lim)

    # Merge: algo limits win; fall through to known table for everything else.
    def limit_for(product: str) -> int:
        if product in algo_limits:
            return algo_limits[product]
        return KNOWN_POSITION_LIMITS.get(product, default_limit)

    return mod.Trader, limit_for, default_limit


# ── Data Models ───────────────────────────────────────────────────────────────

class TradeMatchingMode(str, Enum):
    all = "all"
    worse = "worse"
    none = "none"


@dataclass
class MatchingConfig:
    mode: TradeMatchingMode = TradeMatchingMode.all
    # Fraction of each market trade's volume available to us (0–1).
    # Mirrors --queue-penetration in the Rust backtester.
    queue_penetration: float = 1.0
    # Basis-point slippage applied to every fill price.
    # Mirrors --price-slippage-bps in the Rust backtester.
    slippage_bps: float = 0.0


@dataclass
class _MutableMarketTrade:
    """Wrapper that tracks remaining fillable quantity of a market trade."""
    trade: Trade
    quantity: int  # decremented as we consume it


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
            "currency": "XIRECS",
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
            f'    "currency": "XIRECS",\n'
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
    """Try flat layout, then common subdirectory name conventions."""
    filename = f"{prefix}_round_{round_num}_day_{day_num}.csv"
    flat = data_dir / filename
    if flat.is_file():
        return flat
    for subdir_name in (f"round{round_num}", f"ROUND{round_num}"):
        nested = data_dir / subdir_name / filename
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
    pattern = re.compile(rf"prices_round_{round_num}_day_(-?\d+)\.csv")
    days = []

    if data_dir.is_dir():
        for f in data_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                days.append(int(m.group(1)))

    for subdir_name in (f"round{round_num}", f"ROUND{round_num}"):
        nested = data_dir / subdir_name
        if nested.is_dir():
            for f in nested.iterdir():
                m = pattern.match(f.name)
                if m:
                    d = int(m.group(1))
                    if d not in days:
                        days.append(d)

    return sorted(days)


def read_day_data(data_dir: Path, round_num: int, day_num: int) -> BacktestData:
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


# ── Runner / Matching Engine ──────────────────────────────────────────────────

def prepare_state(state: TradingState, data: BacktestData) -> None:
    for product in data.products:
        order_depth = OrderDepth()
        row = data.prices[state.timestamp][product]

        for price, volume in zip(row.bid_prices, row.bid_volumes):
            order_depth.buy_orders[price] = volume

        for price, volume in zip(row.ask_prices, row.ask_volumes):
            order_depth.sell_orders[price] = -volume

        state.order_depths[product] = order_depth
        state.listings[product] = Listing(product, product, "XIRECS")

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
        pnl = data.profit_loss[product]
        position = state.position.get(product, 0)
        if position != 0:
            pnl += position * row.mid_price

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
            pnl,
        ]
        result.activity_logs.append(ActivityLogRow(columns))


def enforce_limits(
    state: TradingState,
    data: BacktestData,
    orders: Dict[Symbol, List[Order]],
    sandbox_row: SandboxLogRow,
    limit_for,
    default_limit: int,
) -> None:
    sandbox_log_lines = []
    for product in data.products:
        product_orders = orders.get(product, [])
        product_position = state.position.get(product, 0)
        limit = limit_for(product)

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


# ── Core matching helpers (mirrors Rust runner.rs) ───────────────────────────

def _slippage_price(price: int, is_buy: bool, bps: float) -> int:
    """Apply basis-point slippage to a fill price."""
    if bps <= 0.0:
        return price
    factor = 1.0 + bps / 10_000.0
    adjusted = price * factor if is_buy else price / factor
    return round(adjusted)


def _queue_penetration_available(quantity: int, penetration: float) -> int:
    """Scale a market trade's volume by queue_penetration, ensuring at least 1
    when penetration > 0 and quantity > 0 (mirrors Rust queue_penetration_available)."""
    if penetration <= 0.0:
        return 0
    raw = quantity * penetration
    available = round(raw)
    if quantity > 0 and penetration > 0.0 and available == 0:
        available = 1
    return max(0, available)


def _market_trade_duplicates_touch(
    trade: Trade,
    best_bid: Optional[int],
    best_ask: Optional[int],
) -> bool:
    """Return True if this historical trade would be double-counted by the
    book-crossing phase.  Mirrors Rust market_trade_duplicates_touch."""
    if trade.buyer == "SUBMISSION" and best_ask is not None and trade.price >= best_ask:
        return True
    if trade.seller == "SUBMISSION" and best_bid is not None and trade.price <= best_bid:
        return True
    return False


def _eligible_trade_price(
    order_price: int,
    trade_price: int,
    quantity: int,
    mode: TradeMatchingMode,
) -> bool:
    """Return True if this market trade is eligible to fill our order.
    Mirrors Rust eligible_trade_price."""
    if mode == TradeMatchingMode.none:
        return False
    if quantity > 0:
        return trade_price <= order_price if mode == TradeMatchingMode.all else trade_price < order_price
    if quantity < 0:
        return trade_price >= order_price if mode == TradeMatchingMode.all else trade_price > order_price
    return False


def match_orders(
    state: TradingState,
    data: BacktestData,
    orders: Dict[Symbol, List[Order]],
    result: BacktestResult,
    config: MatchingConfig,
) -> None:
    """
    Two-phase matching engine mirroring the Rust backtester.

    Phase 1 – Book crossing: fills immediately at the book price for any order
              that crosses the current order depth.
    Phase 2 – Market trade matching: remaining unfilled volume is matched
              against historical market trades.  Queue position is modelled:
              existing book volume at each price is consumed ahead of our order
              before we can fill there (buy_queue / sell_queue).
    """
    for product in data.products:
        order_depth = state.order_depths[product]
        product_orders = orders.get(product, [])

        # Best bid/ask for duplicate-touch filter.
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        # Queue-ahead volumes: tracks book depth that sits in front of our
        # passive order at each price level.  Initialized from the current book
        # and decremented as market trades consume it.
        buy_queue: Dict[int, int] = dict(order_depth.buy_orders)           # price -> +volume
        sell_queue: Dict[int, int] = {                                       # price -> +volume
            p: abs(v) for p, v in order_depth.sell_orders.items()
        }

        # Build market-trade availability list (with penetration scaling,
        # deduplication filter, and mutable quantity tracking).
        raw_trades = data.trades[state.timestamp].get(product, [])
        market_available: List[_MutableMarketTrade] = []
        for t in raw_trades:
            if _market_trade_duplicates_touch(t, best_bid, best_ask):
                continue
            avail = _queue_penetration_available(t.quantity, config.queue_penetration)
            if avail > 0:
                market_available.append(_MutableMarketTrade(t, avail))

        own_trades: List[Trade] = []

        for order in product_orders:
            remaining = order.quantity

            # ── Phase 1: aggressive book crossing ────────────────────────────
            if remaining > 0:
                # Buy order: sweep ask levels at or below our price.
                for ask_price in sorted(order_depth.sell_orders):
                    if remaining <= 0:
                        break
                    if ask_price > order.price:
                        break
                    vol = order_depth.sell_orders[ask_price]
                    if vol >= 0:
                        continue
                    fill = min(remaining, abs(vol))
                    trade_price = _slippage_price(ask_price, True, config.slippage_bps)
                    own_trades.append(Trade(
                        order.symbol, trade_price, fill,
                        "SUBMISSION", "", state.timestamp,
                    ))
                    state.position[order.symbol] = state.position.get(order.symbol, 0) + fill
                    data.profit_loss[order.symbol] -= trade_price * fill
                    order_depth.sell_orders[ask_price] += fill
                    if order_depth.sell_orders[ask_price] == 0:
                        del order_depth.sell_orders[ask_price]
                    remaining -= fill

            elif remaining < 0:
                # Sell order: sweep bid levels at or above our price.
                for bid_price in sorted(order_depth.buy_orders, reverse=True):
                    if remaining >= 0:
                        break
                    if bid_price < order.price:
                        break
                    vol = order_depth.buy_orders[bid_price]
                    if vol <= 0:
                        continue
                    fill = min(-remaining, vol)
                    trade_price = _slippage_price(bid_price, False, config.slippage_bps)
                    own_trades.append(Trade(
                        order.symbol, trade_price, fill,
                        "", "SUBMISSION", state.timestamp,
                    ))
                    state.position[order.symbol] = state.position.get(order.symbol, 0) - fill
                    data.profit_loss[order.symbol] += trade_price * fill
                    order_depth.buy_orders[bid_price] -= fill
                    if order_depth.buy_orders[bid_price] == 0:
                        del order_depth.buy_orders[bid_price]
                    remaining += fill

            # ── Phase 2: market trade matching with queue position model ──────
            if remaining != 0 and config.mode != TradeMatchingMode.none:
                for mt in market_available:
                    if remaining == 0:
                        break
                    if mt.quantity <= 0:
                        continue
                    if not _eligible_trade_price(
                        order.price, mt.trade.price, remaining, config.mode
                    ):
                        continue

                    # Queue position: consume book volume ahead of us at the
                    # same price before our order can fill.
                    if remaining > 0 and mt.trade.price == order.price:
                        ahead = buy_queue.get(order.price, 0)
                        consumed = min(mt.quantity, ahead)
                        mt.quantity -= consumed
                        new_ahead = ahead - consumed
                        if new_ahead <= 0:
                            buy_queue.pop(order.price, None)
                        else:
                            buy_queue[order.price] = new_ahead

                    elif remaining < 0 and mt.trade.price == order.price:
                        ahead = sell_queue.get(order.price, 0)
                        consumed = min(mt.quantity, ahead)
                        mt.quantity -= consumed
                        new_ahead = ahead - consumed
                        if new_ahead <= 0:
                            sell_queue.pop(order.price, None)
                        else:
                            sell_queue[order.price] = new_ahead

                    if mt.quantity <= 0:
                        continue

                    fill = min(abs(remaining), mt.quantity)
                    execution_price = _slippage_price(
                        order.price, remaining > 0, config.slippage_bps
                    )

                    if remaining > 0:
                        own_trades.append(Trade(
                            order.symbol, execution_price, fill,
                            "SUBMISSION", mt.trade.seller, state.timestamp,
                        ))
                        state.position[order.symbol] = state.position.get(order.symbol, 0) + fill
                        data.profit_loss[order.symbol] -= execution_price * fill
                        remaining -= fill
                    else:
                        own_trades.append(Trade(
                            order.symbol, execution_price, fill,
                            mt.trade.buyer, "SUBMISSION", state.timestamp,
                        ))
                        state.position[order.symbol] = state.position.get(order.symbol, 0) - fill
                        data.profit_loss[order.symbol] += execution_price * fill
                        remaining += fill

                    mt.quantity -= fill

        # Commit own trades to state and result.
        if own_trades:
            state.own_trades[product] = own_trades
            result.trades.extend(TradeRow(t) for t in own_trades)

        # Remaining market trades flow into state.market_trades.
        remaining_market = [mt.trade for mt in market_available if mt.quantity > 0]
        state.market_trades[product] = remaining_market
        result.trades.extend(TradeRow(t) for t in remaining_market)


# ── Main Backtest Loop ────────────────────────────────────────────────────────

def run_backtest(
    trader,
    data_dir: Path,
    round_num: int,
    day_num: int,
    limit_for,
    default_limit: int,
    print_output: bool,
    config: MatchingConfig,
) -> BacktestResult:
    data = read_day_data(data_dir, round_num, day_num)

    os.environ["PROSPERITY4BT_ROUND"] = str(round_num)
    os.environ["PROSPERITY4BT_DAY"] = str(day_num)
    # Round 3 historical data: day_0 = TTE8 (tutorial), day_1 = TTE7 (R1), day_2 = TTE6 (R2).
    # The live Round 3 submission starts at TTE5, so this env var is absent there.
    if round_num == 3:
        os.environ["PROSPERITY4BT_TTE_START"] = str(8 - day_num)
    else:
        os.environ.pop("PROSPERITY4BT_TTE_START", None)

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
        state.own_trades = {}
        state.market_trades = {}

        prepare_state(state, data)

        stdout = StringIO()

        if print_output:
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
        enforce_limits(state, data, orders, sandbox_row, limit_for, default_limit)
        match_orders(state, data, orders, result, config)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(
                f"\r  Progress: {i + 1}/{total} timestamps "
                f"({100 * (i + 1) // total}%)",
                end="",
                flush=True,
            )

    print()
    return result


# ── Output Generation ─────────────────────────────────────────────────────────

def write_log_output(output_file: Path, merged: BacktestResult) -> None:
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
    output_file.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;"
        "bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;"
        "ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;"
        "profit_and_loss"
    )
    activities_csv = header + "\n" + "\n".join(map(str, merged.activity_logs))

    pnl_by_ts: Dict[int, float] = defaultdict(float)
    for row in merged.activity_logs:
        pnl_by_ts[row.timestamp] += row.columns[-1]

    graph_lines = ["timestamp;value"]
    for ts in sorted(pnl_by_ts):
        graph_lines.append(f"{ts};{pnl_by_ts[ts]}")
    graph_log = "\n".join(graph_lines)

    positions_by_product: Dict[str, int] = {}
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

    total_profit = 0.0
    if merged.activity_logs:
        last_ts = merged.activity_logs[-1].timestamp
        for row in reversed(merged.activity_logs):
            if row.timestamp != last_ts:
                break
            total_profit += row.columns[-1]

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

    sandbox_logs.extend(row.with_offset(timestamp_offset) for row in b.sandbox_logs)
    trades.extend(row.with_offset(timestamp_offset) for row in b.trades)

    if merge_profit_loss:
        profit_loss_offsets: Dict[str, float] = defaultdict(float)
        a_last_ts = a.activity_logs[-1].timestamp
        for row in reversed(a.activity_logs):
            if row.timestamp != a_last_ts:
                break
            profit_loss_offsets[row.columns[2]] = row.columns[-1]

        activity_logs.extend(
            row.with_offset(timestamp_offset, profit_loss_offsets[row.columns[2]])
            for row in b.activity_logs
        )
    else:
        activity_logs.extend(
            row.with_offset(timestamp_offset, 0) for row in b.activity_logs
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


def _resolve_data_dir(raw: str) -> Path:
    """Resolve the data directory, falling back to the Rust backtester's
    datasets/ folder when the given path doesn't exist."""
    p = Path(raw)
    if p.is_dir():
        return p
    # Try relative to this file
    here = Path(__file__).parent
    candidate = here / raw
    if candidate.is_dir():
        return candidate
    # Fall back to bundled datasets
    rust_datasets = here / "prosperity_rust_backtester-main" / "datasets"
    if rust_datasets.is_dir():
        return rust_datasets
    return p  # will fail later with a clear error


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
        default="ROUND_3",
        help=(
            "Path to data directory (default: ROUND_3). "
            "Also accepts prosperity_rust_backtester-main/datasets for the "
            "bundled Rust backtester datasets."
        ),
    )
    parser.add_argument(
        "--round",
        type=int,
        default=3,
        help="Round number (default: 3)",
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
        help=(
            "Market trade matching mode (default: all). "
            "'all' matches at trade_price <= order_price (buy) or >= (sell). "
            "'worse' requires strictly better price. 'none' disables market matching."
        ),
    )
    parser.add_argument(
        "--queue-penetration",
        type=float,
        default=1.0,
        dest="queue_penetration",
        help=(
            "Fraction of each market trade's volume reachable by our order "
            "(default: 1.0). Mirrors --queue-penetration in the Rust backtester."
        ),
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        dest="slippage_bps",
        help=(
            "Basis-point price slippage applied to every fill "
            "(default: 0.0). Mirrors --price-slippage-bps in the Rust backtester."
        ),
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
    if not algo_path.suffix:
        algo_path = algo_path.with_suffix(".py")
    if not algo_path.exists():
        candidate = Path("logs") / algo_path
        if candidate.exists():
            algo_path = candidate
    if not algo_path.exists():
        print(f"Error: algo file not found: {algo_path}")
        sys.exit(1)

    data_dir = _resolve_data_dir(args.data)
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    algo_name = algo_path.stem
    print(f"Loading algo: {algo_path}")
    trader_cls, limit_for, default_limit = load_algo(algo_path)

    round_num = args.round
    config = MatchingConfig(
        mode=TradeMatchingMode(args.match_trades),
        queue_penetration=args.queue_penetration,
        slippage_bps=args.slippage_bps,
    )

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
        f"Match mode: {config.mode.value} | "
        f"Queue penetration: {config.queue_penetration} | "
        f"Slippage bps: {config.slippage_bps}"
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
            limit_for,
            default_limit,
            args.print_output,
            config,
        )

        print_day_summary(result)
        results.append(result)

    if len(results) > 1:
        print_overall_summary(results)

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

        log_path = output_path.with_suffix(".log")
        write_log_output(log_path, merged)

        print(f"\nOutput saved to {output_path}")
        print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
