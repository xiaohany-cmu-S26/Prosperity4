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

Adapted from [jmerle/imc-prosperity-3-backtester](https://github.com/jmerle/imc-prosperity-3-backtester) for Prosperity 4. Replays tutorial round CSV data through any algo's `Trader.run()` with a two-phase matching engine:

1. **Order depth matching**: orders that cross the current book fill immediately at the book price
2. **Market trade matching**: remaining unfilled volume is matched against historical market trades at favorable prices

Trade matching mode controls phase 2: `all` (default) matches trades at prices equal to or better than your quote, `worse` only matches strictly better prices, `none` skips market trade matching entirely.

### Usage

```bash
# Run default algo (trading.py) on all available days
python3 backtest.py

# Run a specific algo file
python3 backtest.py --algo my_strat.py

# Single day
python3 backtest.py --day -1

# Stricter matching (only match market trades at worse prices)
python3 backtest.py --match-trades worse

# Print trader stdout while running
python3 backtest.py --print

# Custom data directory and round
python3 backtest.py --data ROUND_1 --round 1

# Skip saving output
python3 backtest.py --no-out

# Merge PnL across days
python3 backtest.py --merge-pnl
```

### Output

The backtester writes two files to `logs/`:
- `backtest_{algo}.json` — JSON format compatible with `visualize_log.py`
- `backtest_{algo}.log` — P3-style log format (Sandbox logs / Activities log / Trade History)

### Algo File Requirements

Any `.py` file with:
- A `Trader` class with a `run(state: TradingState)` method returning `(orders, conversions, traderData)`
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
