"""
Sweep ASH_COATED_OSMIUM (ACO) parameters in trading.py against the full round-1 dataset.
Patches PRODUCT_CONFIG in-process — no subprocess overhead.
Baseline total PnL (with optimised IPR): 293,661.
"""
import importlib.util, copy
from pathlib import Path
from itertools import product as cartesian
from collections import defaultdict

# ── Load backtest helpers ─────────────────────────────────────────────────────
bt_spec = importlib.util.spec_from_file_location("backtest", "backtest.py")
bt = importlib.util.module_from_spec(bt_spec)
bt_spec.loader.exec_module(bt)

# ── Load trading module ───────────────────────────────────────────────────────
tr_spec = importlib.util.spec_from_file_location("trading", "trading.py")
tr = importlib.util.module_from_spec(tr_spec)
tr_spec.loader.exec_module(tr)

DATA_DIR = Path("prosperity_rust_backtester-main/datasets")
DAYS = [-2, -1, 0]
ROUND = 1
CFG = bt.MatchingConfig()

def day_pnl(result):
    last = defaultdict(float)
    for row in result.activity_logs:
        parts = str(row).split(";")
        if len(parts) >= 17:
            last[parts[2]] = float(parts[16])
    return sum(last.values())

def run_all_days(trader):
    total = 0
    for day in DAYS:
        res = bt.run_backtest(
            trader, DATA_DIR, ROUND, day,
            lambda p: bt.KNOWN_POSITION_LIMITS.get(p, 80), 80,
            False, CFG,
        )
        total += day_pnl(res)
    return total

# ── Parameter grid ────────────────────────────────────────────────────────────
GRID = {
    "take_fv_skew_buy":           [2.5, 3.0, 3.5, 4.0, 4.5],
    "take_fv_skew_sell":          [1.5, 2.0, 2.5, 3.0],
    "take_signal_threshold_buy":  [0.65, 0.70, 0.75, 0.80],
    "ema_anchor_weight":          [0.20, 0.25, 0.30],
    "microprice_weight":          [0.25, 0.30, 0.35, 0.40],
}

BASE = copy.deepcopy(tr.PRODUCT_CONFIG["ASH_COATED_OSMIUM"])

results = []
keys = list(GRID.keys())
combos = list(cartesian(*[GRID[k] for k in keys]))
print(f"Testing {len(combos)} combinations…\n")

for i, vals in enumerate(combos):
    params = dict(zip(keys, vals))
    for k, v in params.items():
        tr.PRODUCT_CONFIG["ASH_COATED_OSMIUM"][k] = v

    trader = tr.Trader()
    pnl = run_all_days(trader)
    results.append((pnl, params))

    if (i + 1) % 100 == 0 or (i + 1) == len(combos):
        best_so_far = max(r[0] for r in results)
        print(f"  [{i+1}/{len(combos)}] best so far: {best_so_far:,.0f}")

# Restore original config
tr.PRODUCT_CONFIG["ASH_COATED_OSMIUM"].update(BASE)

# ── Report top 20 ─────────────────────────────────────────────────────────────
results.sort(key=lambda x: x[0], reverse=True)
print("\n=== Top 20 parameter sets ===")
print(
    f"{'PnL':>10}  {'skew_buy':>8}  {'skew_sell':>9}  "
    f"{'thr_buy':>7}  {'ema_aw':>6}  {'micro_w':>7}"
)
for pnl, p in results[:20]:
    print(
        f"{pnl:>10,.0f}  {p['take_fv_skew_buy']:>8.2f}  "
        f"{p['take_fv_skew_sell']:>9.2f}  "
        f"{p['take_signal_threshold_buy']:>7.2f}  "
        f"{p['ema_anchor_weight']:>6.2f}  "
        f"{p['microprice_weight']:>7.2f}"
    )

print(f"\nBaseline: 293,661")
print(f"Best:     {results[0][0]:,.0f}  params: {results[0][1]}")
