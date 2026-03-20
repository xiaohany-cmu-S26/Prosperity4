# Prosperity 4 Trading Algorithm

Market-making bot for IMC's Prosperity 4 tournament.

## Files

| File | Purpose |
|------|---------|
| `trading.py` | Main trading algorithm |
| `backtest.py` | Local backtester — replays tutorial round data through any algo file |
| `visualize_log.py` | Interactive chart generation from trade logs |
| `datamodel.py` | IMC Prosperity data structures (OrderDepth, TradingState, etc.) |
| `TUTORIAL_ROUND_1/` | Tutorial round market data (prices + trades CSVs) |
| `logs/` | Trade logs, backtest CSVs, and HTML visualizations |

## Strategy

### Fair Value: Microprice Deviation L2

Uses second-level order book data to estimate fair value:

```
FV = (bid_price_2 × ask_volume_2 + ask_price_2 × bid_volume_2) / (bid_volume_2 + ask_volume_2)
```

This is only used when the spread is wide (≥ 10 ticks). During tight spreads, the simple mid price is used instead — L2 microprice has high variance in tight regimes (std 6.15 vs 0.35).

### Spread Regime Detection

The algorithm behaves differently based on spread width:

**Wide spread (≥ 10 ticks):**
- Full aggressive taking in both directions when price crosses FV
- L2 microprice for fair value

**Tight spread (< 10 ticks):**
- Aggressive taking only to unwind existing inventory (never adds to position)
- Simple mid price for fair value

### Inventory Management

**EMERALDS:** FV skew based on position size:
- `|pos| < 40`: no skew
- `|pos| 40–60`: FV shifted by `0.05 × position`
- `|pos| > 60`: FV shifted by `0.2 × position`

**TOMATOES:** Asymmetric quoting — the unwinding side quotes more aggressively:
- `|pos| ≤ 63`: symmetric conservative (bid+1 / ask-1)
- `|pos| > 63`: unwinding side quotes at fair value

### Making (Passive Quotes)

- Default: post at `best_bid + 1` / `best_ask - 1`
- EMERALDS: capped by `FV ± 1` to avoid quoting through fair value
- Safety checks prevent crossing the book

## Backtester

Replays tutorial round CSV data through any algo's `Trader.run()` with a two-phase matching engine:

1. **Passive fills**: resting orders from the previous tick get filled when market trades occur at prices that would cross our quotes
2. **Aggressive fills**: orders that cross the current book fill immediately

### Usage

```bash
# Run default algo (trading.py) on both days
python3 backtest.py

# Run a specific algo file
python3 backtest.py --algo my_strat.py

# Single day, verbose fill logging
python3 backtest.py --day -1 --verbose

# Disable CSV log output
python3 backtest.py --no-log

# Custom log directory
python3 backtest.py --log-dir results/
```

### CSV Log Output

The backtester writes per-tick CSV logs to `logs/backtest_{algo}_day_{day}.csv` with columns:

| Column | Description |
|--------|-------------|
| `day`, `timestamp`, `product` | Tick identifier |
| `bid/ask_price/volume_1-3` | Order book snapshot |
| `mid_price`, `profit_and_loss` | Market state |
| `position` | Current position after fills |
| `our_bid/ask`, `our_bid/ask_vol` | Quotes placed this tick |
| `fill_side`, `fill_price`, `fill_qty`, `fill_type` | Fill details (aggressive/passive) |

### Algo File Requirements

Any `.py` file with:
- A `Trader` class with a `run(state: TradingState)` method returning `(result, conversions, traderData)`
- Optional `POSITION_LIMITS` dict and `DEFAULT_LIMIT` int (defaults to 80)

## Visualizer

```bash
# Dashboard with all logs (dropdown selector)
python3 visualize_log.py

# Single log visualization
python3 visualize_log.py --log 3179/3179
```

Features:
- Order book depth, spread, and price charts
- PnL with inventory overlay (secondary y-axis)
- Total PnL trace across products
- Dynamic sidebar metrics that update on zoom/pan

## Configuration

Key constants in `trading.py`:

```python
EMERALD_FAIR_VALUE = 10000
POSITION_LIMITS = {"EMERALDS": 80, "TOMATOES": 80}
DEFAULT_LIMIT = 80
```

---

Last updated: March 2026
