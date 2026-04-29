# SNACKPACK Phase 1 — Negative Result Summary

**TL;DR:** Per-leg directional ML prediction on SNACKPACK fails the kill criterion. The basket-deviation autocorrelation that motivated the pilot is real (lag-1 = +0.970) but the *direction* of mean reversion at trading horizons (H=5/20/100) **flips sign across the 3 days of data**. A model trained on days 2–3 would lose money on day 4.

## What we tested

- 5 SNACKPACK legs (CHOCOLATE, VANILLA, PISTACHIO, STRAWBERRY, RASPBERRY) × 3 horizons (H=5, 20, 100 ticks).
- Features: OFI windows, queue imbalance L1/L3, micro-price, realized vol, signed trade flow, **cross-leg basket deviations** (cv_dev = CHOC+VAN deviation from EMA, psr_dev = PIS+STR−2·RASP deviation).
- Two splits tried: (a) day 2 train / day 3 val / day 4 test, (b) days 2+3 pooled / day 4 test.

## Findings

### 1. Phase 0 found two distinct basket structures
- CHOC ↔ VAN: r = **−0.915** (the pair `round5.py` already exploits)
- RASP ↔ (PIS+STR): r = **−0.83 / −0.92**, with PIS ↔ STR at **+0.913**
- CHOC+VAN deviation lag-1 autocorr = **+0.970** (looks like a strong slow mean-reverting signal)

### 2. The 0.97 autocorr does NOT translate to predictable returns

Diagnostic IC of `−cv_dev` predicting `fwd_ret_20_CHOC + fwd_ret_20_VAN`:

| Day | IC |
|---|---|
| 2 | **+0.034** (mean-revert)|
| 3 | **−0.038** (anti-revert!)|
| 4 | **+0.070** (mean-revert)|

Same flip on `psr_dev`: −0.033 / +0.086 / +0.072 across days 2/3/4. The basket *level* is slow-moving (high autocorr), but its *trajectory* over the next 20 ticks is regime-dependent.

### 3. LightGBM val/test IC fails kill criterion

V1 (day 2 train / day 3 val / day 4 test, H=20):
| Leg | val IC | test IC |
|---|---|---|
| CHOCOLATE | +0.034 | **−0.059** |
| VANILLA | +0.019 | −0.017 |
| PISTACHIO | +0.005 | **−0.061** |
| STRAWBERRY | −0.016 | −0.028 |
| RASPBERRY | −0.027 | +0.030 |

**1/5 legs** met the threshold. Best_iter was 1–3 across all legs (model can't even improve on a single tree without hurting val).

V2 (pooled days 2+3 train, day 4 test):
| H | legs ≥ 0.03 | median |IC| | max |IC| |
|---|---|---|---|
| 5 | 2/5 | 0.023 | 0.036 |
| 20 | 2/5 | 0.025 | 0.042 |
| 100 | 3/5 | 0.031 | 0.145 |

H=100 looks like a partial pass (3/5), but **all 5 test ICs are negative** while val ICs are positive — the model learned a relationship that inverts on day 4. Acting on these predictions would produce −0.145 IC trades on day 4. That's worse than random.

### 4. Why this happened

- 3 days is ~1.5 regime samples. Cross-day stability of the alpha cannot be established with this much data.
- The pure structural alpha (basket sum mean-reverts) is real but its *time scale* and *direction* shift between days. The simple `−0.30 · pair_dev / 2.0` tilt already in `round5.py` works on average across days, but pre-conditioning it on additional features doesn't help — the features themselves shift sign.
- Position limit 10 means even a barely-positive signal can't pay for the spread cost.

## Recommendation: pivot or stop, do not proceed to backtest A/B

The kill criterion exists exactly for this scenario. Two productive directions:

1. **Pivot category — ROBOT.** ROBOT_DISHES already earns +148k via simple bounce-fade in `round5.py`. Other ROBOT variants are documented as "net-negative" with the same fixed strategy. ML could (a) detect *which* variants are in a bouncy regime currently, or (b) tune the bounce coefficient per variant per regime. This is a single-leg directional problem with a known proof-of-concept (DISHES).

2. **Stop and write up.** Honest finding: with 3 days and limit-10, structural basket alpha looks attractive in autocorrelation but is not stable enough to extract via ML. The existing `round5.py` heuristic pair-tilt already captures what's reliably exploitable. This is a useful negative result for the post-mortem.

Phase 2 (per-category transformer) is **deferred indefinitely** — adding model capacity won't fix regime non-stationarity.
