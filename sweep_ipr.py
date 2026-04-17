"""
Sweep IPR parameters in trading.py against the full round-1 dataset.
Patches PRODUCT_CONFIG in-process — no subprocess overhead.
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
    "min_long_position":     [72, 74, 75, 76, 77, 78, 79, 80],
    "buy_to_ask_max_offset": [10, 13, 15, 18, 20],
    "core_build_per_tick":   [40, 60, 80],
    "asym_moderate_pos":     [73, 75, 77, 80],
    "asym_offset":           [1, 2, 3],
}

BASE = copy.deepcopy(tr.PRODUCT_CONFIG["INTARIAN_PEPPER_ROOT"])

results = []
keys = list(GRID.keys())
combos = list(cartesian(*[GRID[k] for k in keys]))
print(f"Testing {len(combos)} combinations…\n")

for i, vals in enumerate(combos):
    params = dict(zip(keys, vals))
    for k, v in params.items():
        tr.PRODUCT_CONFIG["INTARIAN_PEPPER_ROOT"][k] = v
    tr.PRODUCT_CONFIG["INTARIAN_PEPPER_ROOT"]["soft_min_long_position"] = params["min_long_position"]

    trader = tr.Trader()
    pnl = run_all_days(trader)
    results.append((pnl, params))

    if (i + 1) % 100 == 0 or (i + 1) == len(combos):
        best_so_far = max(r[0] for r in results)
        print(f"  [{i+1}/{len(combos)}] best so far: {best_so_far:,.0f}")

# Restore original config
tr.PRODUCT_CONFIG["INTARIAN_PEPPER_ROOT"].update(BASE)

# ── Report top 20 ─────────────────────────────────────────────────────────────
results.sort(key=lambda x: x[0], reverse=True)
print("\n=== Top 20 parameter sets ===")
print(f"{'PnL':>10}  {'min_long':>8}  {'max_off':>7}  {'per_tick':>8}  {'asym_mod':>8}  {'asym_off':>8}")
for pnl, p in results[:20]:
    print(
        f"{pnl:>10,.0f}  {p['min_long_position']:>8}  "
        f"{p['buy_to_ask_max_offset']:>7}  {p['core_build_per_tick']:>8}  "
        f"{p['asym_moderate_pos']:>8}  {p['asym_offset']:>8}"
    )

print(f"\nBaseline: 291,527")
print(f"Best:     {results[0][0]:,.0f}  params: {results[0][1]}")
