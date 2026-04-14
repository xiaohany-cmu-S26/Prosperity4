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

Each product is configured via `PRODUCT_CONFIG` in `trading.py`. The algo has two phases per tick:

1. **Aggressive taking** — lift any ask below `skewed_fv`, hit any bid above `skewed_fv`
2. **Passive making** — post resting quotes on both sides of `skewed_fv`

### Fair Value

| Product | Method | Detail |
|---------|--------|--------|
| `INTARIAN_PEPPER_ROOT` | `linear_trend` | Anchors to first observed mid each day, then extrapolates at +0.001/timestamp |
| `ASH_COATED_OSMIUM` | Fixed float | Hardcoded at 10000 (strong mean reversion, lag-1 autocorr ≈ −0.49) |

### Passive Quote Placement

**`quote_offset` (book-relative):** quote at `best_bid − offset` / `best_ask + offset`.  
Negative offset (e.g. `−1`) posts *inside* the spread, giving queue priority over resting structural MMs.

**`fv_spread` (FV-relative):** quote at `FV ± spread` regardless of the book.  
Used when structural MMs anchor at a known fixed distance from FV, so we can undercut them.

| Product | Quote mode | Value | Rationale |
|---------|-----------|-------|-----------|
| `INTARIAN_PEPPER_ROOT` | `quote_offset` | `−1` | Post 1 tick inside best bid/ask for queue priority; typical ~13-tick spread → 11-tick round-trip |
| `ASH_COATED_OSMIUM` | `fv_spread` | `6` | Structural MMs at ±7 (9993/10007); ±6 undercuts them and captures compressed-market asks |

### Inventory Skew

Both products use `skew_tiers`: at each tier, `skewed_fv = base_fv − skew_factor × position`. Aggressive taking and passive quote filtering both use `skewed_fv`, biasing the algo toward reducing large positions.

| Product | Tiers |
|---------|-------|
| `INTARIAN_PEPPER_ROOT` | `\|pos\| < 40`: 0.0 · `\|pos\| < 60`: 0.05 · else: 0.2 |
| `ASH_COATED_OSMIUM` | flat 0.1 across all positions |

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

All per-product tuning is in `PRODUCT_CONFIG` at the top of `trading.py`. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `limit` | int | Position limit (long and short) |
| `fair_value` | float \| `"linear_trend"` \| `"midwall"` | FV method |
| `trend_slope` | float | Used by `linear_trend` — FV increment per timestamp unit |
| `skew_tiers` | list of `(threshold, factor)` | Inventory skew ladder |
| `quote_offset` | int | Ticks outside best bid/ask (`−1` = 1 tick inside) |
| `fv_spread` | int | Fixed ticks from FV for passive quotes (overrides `quote_offset`) |
| `max_passive_qty` | int | Max units per passive order |

---

Last updated: April 2026
