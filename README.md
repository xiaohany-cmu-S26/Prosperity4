# Prosperity 4 Trading Algorithm

Market-making and arbitrage bot for IMC's Prosperity 4 tournament, with aggressive liquidity-providing strategies.

## Overview

This codebase implements a multi-product trading algorithm that:
- **Market makes** on multiple products (EMERALDS, TOMATOES) using Wall Mid fair value estimation
- **Manages inventory** by detecting skewed positions and forcing reversion trades
- **Aggressively liquidity-provides** in tight spreads while queue-beating in wider spreads
- **Prevents losses** by validating all trades against fair value before execution

## Files

| File | Purpose |
|------|---------|
| `trading.py` | Main trading algorithm with market making logic |
| `trading_emerald_band.py` | Alternative EMERALDS-only strategy with fixed bid/ask bands |
| `visualize_log.py` | Parse and visualize trade logs with interactive charts |
| `read_log.py` | Utility to inspect raw trade data |
| `datamodel.py` | Trade state and order data structures |
| `logs/` | Historical runs and trade visualizations |

## Strategy

### Fair Value Estimation: Wall Mid

The algorithm estimates true underlying price using **Wall Mid**:
- **Bid Wall** = the price level with highest buy volume
- **Ask Wall** = the price level with highest sell volume
- **Wall Mid** = (Bid Wall + Ask Wall) / 2

This is more stable than raw mid price because market makers quote at the walls, not at overbids/undercuts.

### Three Trading Steps

#### Step 1: Take Profitable Quotes
- Buy all asks **strictly below** Wall Mid
- Sell all bids **strictly above** Wall Mid
- Prevents taking trades at a loss
- Runs regardless of inventory state

#### Step 2a: Inventory Reversion (Skewed Position)
When `|position| > SKEW_THRESHOLD`:
- **Too long** (position > 0): Post ask at Wall Mid to encourage selling
- **Too short** (position < 0): Post bid at Wall Mid to encourage buying
- Prevents building extreme inventory imbalances

Per-product thresholds (configured in `SKEW_THRESHOLDS`):
```python
SKEW_THRESHOLDS: dict[str, int] = {
    "EMERALDS": 68,
    "TOMATOES": 30,
}
```

#### Step 2b: Market Making (Normal Inventory)
When inventory is within thresholds:

**Wide spreads (≥3 ticks):**
- Post bid at `best_bid + 1`, ask at `best_ask - 1` (queue-beating)
- Full remaining capacity on each side
- Attempt to capture wide spreads

**Tight spreads (<3 ticks):**
- Post both bid and ask at Wall Mid with half capacity
- Acts as liquidity provider when market is efficient
- Higher fill probability

### Loss Prevention

All trades are validated:
- Takes: `ask_price < FV` (buy) and `bid_price > FV` (sell)
- Quotes: `our_bid < FV` and `our_ask > FV`
- Skew reversion: Rounding safe to avoid fractional FV losses

Rejected trades are logged with reason for debugging.

## Configuration

Edit `trading.py` constants:

```python
POSITION_LIMIT = 80                    # Max position long/short per product
MIDWALL_TOP_N = 3                      # (unused, kept for reference)
DEFAULT_SKEW_THRESHOLD = 40            # Default inventory reversion threshold
SKEW_THRESHOLDS = {...}                # Per-product overrides
```

## Running

### View Trade Results
```bash
# Generate interactive chart for a single run
python visualize_log.py --log 3179/3179

# Generate dashboard with all runs (dropdown to switch)
python visualize_log.py

# Open in browser (automatic)
```

### Inspect Trade Data
```bash
python read_log.py
```

## Logs

Trade data is stored in `logs/`:
- `.json`: Structured activities and trades
- `.log`: Full trade history
- `.html`: Interactive visualizations

Run ID is typically `<timestamp>_<timestamp>`.

## Key Metrics

When analyzing runs, look for:
- **Total PnL**: Sum of final PnL across all products
- **Trade count**: Higher ≠ better (quality > quantity)
- **PnL by product**: Evaluate each separately (different market regimes)
- **Position at end**: Should be near 0 (inventory managed well)

## Alternative Strategies

### Fixed Band Strategy (EMERALDS only)
See `trading_emerald_band.py` for a simpler mean-reversion strategy:
- Buy if ask ≤ 9992
- Sell if bid ≥ 10008
- Post passive quotes at band edges

Good for:
- Testing specific price levels
- Hedging specific inventory targets
- Comparing against dynamic Wall Mid approach

## Tuning Guide

Start with these levers:

1. **Spread check threshold** (line ~124): Change `< 3` to adjust aggressive quoting
2. **Skew thresholds** (lines 10-13): Lower = more aggressive inventory reversion
3. **Quote size in tight spreads** (line 126): Adjust `// 2` to control position risk
4. **MIDWALL_TOP_N** (line 5): If using weighted wall, adjust levels considered

Run backtests after each change to measure impact.

## Common Issues

### No trades executed
- Check spread is positive (best_ask > best_bid)
- Verify fair value is reasonable relative to market prices
- Look for "REJECT" logs indicating loss prevention

### Excessive inventory
- Lower `SKEW_THRESHOLD` to force reversion sooner
- Increase quote size in normal quoting block
- Check if one side of market is consistently stale

### Zero PnL despite trades
- Verify entry and exit prices balance
- Check for position unwinds at bad prices
- Ensure Wall Mid is stable (not whipsawing)

## References

- **Wall Mid concept**: IMC Prosperity official documentation
- **Market making basics**: See recommendations in code comments
- **Inventory management**: Standard market maker risk control

---

Last updated: March 2026
