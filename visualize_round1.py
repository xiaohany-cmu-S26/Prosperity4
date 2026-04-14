"""
Visualize all Round 1 raw market data.

Reads ROUND_1/ROUND1/prices_round_1_day_*.csv and
         ROUND_1/ROUND1/trades_round_1_day_*.csv

and produces an interactive Plotly HTML showing mid price, bid/ask spreads,
market trades, and trade volume for each product across every day.

Usage:
  python visualize_round1.py
  python visualize_round1.py --data ROUND_1   # alternate data root
  python visualize_round1.py --no-open        # skip auto-open in browser
"""

import argparse
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT    = Path("ROUND_1")
ROUND_NUM    = 1
VIZ_DIR      = Path("logs/visuals")
OUTPUT_NAME  = "round1_market_data.html"

COLORS        = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
BID_COLOR     = "#26C281"
ASK_COLOR     = "#EF5350"
TRADE_COLOR   = "#FFD700"
DAY_COLORS    = ["#444466", "#446644", "#664444"]   # subtle bg tints per day

TIMESTAMPS_PER_DAY = 1_000_000   # spacing between days on the x-axis


def to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Data loading ──────────────────────────────────────────────────────────────

def find_subdir(data_root: Path, round_num: int) -> Path:
    """Accept both ROUND_1/round1 and ROUND_1/ROUND1 layouts."""
    for name in (f"ROUND{round_num}", f"round{round_num}"):
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find round subdir in {data_root} "
        f"(tried ROUND{round_num} and round{round_num})"
    )


def load_prices(data_root: Path, round_num: int) -> pd.DataFrame:
    subdir = find_subdir(data_root, round_num)
    frames = []
    for p in sorted(subdir.glob(f"prices_round_{round_num}_day_*.csv")):
        df = pd.read_csv(p, sep=";", dtype=str)
        frames.append(df)
        print(f"  Prices: {p.name} ({len(df)} rows)")
    if not frames:
        raise FileNotFoundError(f"No price files found in {subdir}")
    combined = pd.concat(frames, ignore_index=True)
    num_cols = [
        "day", "timestamp",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
        "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2",
        "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss",
    ]
    for col in num_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined.sort_values(["day", "timestamp"]).reset_index(drop=True)


def load_trades(data_root: Path, round_num: int) -> pd.DataFrame:
    subdir = find_subdir(data_root, round_num)
    frames = []
    for p in sorted(subdir.glob(f"trades_round_{round_num}_day_*.csv")):
        df = pd.read_csv(p, sep=";", dtype=str)
        # Infer day from filename
        day_str = p.stem.split("_day_")[-1]
        df["day"] = int(day_str)
        frames.append(df)
        print(f"  Trades: {p.name} ({len(df)} rows)")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    for col in ("timestamp", "price", "quantity"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined.sort_values(["day", "timestamp"]).reset_index(drop=True)


def add_plot_ts(df: pd.DataFrame, days: list) -> pd.DataFrame:
    """Add a continuous `plot_ts` column that offsets each day."""
    day_order = {d: i for i, d in enumerate(days)}
    df = df.copy()
    df["plot_ts"] = df["timestamp"] + df["day"].map(day_order) * TIMESTAMPS_PER_DAY
    return df


# ── Chart building ────────────────────────────────────────────────────────────

def build_figure(prices: pd.DataFrame, trades: pd.DataFrame) -> go.Figure:
    products = sorted(prices["product"].unique())
    days     = sorted(prices["day"].unique())
    n        = len(products)

    # Rows: price panel per product + trade volume panel
    specs = [[{"secondary_y": True}]] * n + [[{"secondary_y": False}]]
    subplot_titles = list(products) + ["Trade Volume"]
    row_heights    = [0.35] * n + [0.20]

    fig = make_subplots(
        rows=n + 1, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        vertical_spacing=0.05,
        specs=specs,
    )

    # ── Day boundary shading & tick labels ────────────────────────────────────
    tick_vals, tick_text = [], []
    for di, day in enumerate(days):
        x0 = di * TIMESTAMPS_PER_DAY
        x1 = x0 + TIMESTAMPS_PER_DAY - 1
        mid = (x0 + x1) // 2
        tick_vals.append(mid)
        tick_text.append(f"Day {day}")

        bg = DAY_COLORS[di % len(DAY_COLORS)]
        for row in range(1, n + 2):
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=to_rgba(bg, 0.25),
                line_width=0,
                row=row, col=1,
            )
        # Vertical divider between days
        if di > 0:
            for row in range(1, n + 2):
                fig.add_vline(
                    x=x0, line_width=1,
                    line_dash="dash", line_color="#888",
                    row=row, col=1,
                )

    # ── Per-product price panels ───────────────────────────────────────────────
    for i, product in enumerate(products):
        row   = i + 1
        color = COLORS[i % len(COLORS)]
        pdf   = prices[prices["product"] == product].copy()

        # ── Bid/ask band ─────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=pdf["plot_ts"], y=pdf["bid_price_1"],
            name=f"Bid₁",
            line=dict(color=BID_COLOR, width=0.7),
            opacity=0.5, legendgroup=f"{product}_book", showlegend=(i == 0),
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=pdf["plot_ts"], y=pdf["ask_price_1"],
            name=f"Ask₁",
            line=dict(color=ASK_COLOR, width=0.7),
            fill="tonexty", fillcolor=to_rgba(BID_COLOR, 0.08),
            opacity=0.5, legendgroup=f"{product}_book", showlegend=(i == 0),
        ), row=row, col=1, secondary_y=False)

        # ── Mid price ─────────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=pdf["plot_ts"], y=pdf["mid_price"],
            name=f"{product} mid",
            line=dict(color=color, width=2),
            legendgroup=product,
            hovertemplate=f"<b>{product}</b><br>t=%{{x}}<br>mid=%{{y:.1f}}<extra></extra>",
        ), row=row, col=1, secondary_y=False)

        # ── Spread (ask1 - bid1) on secondary axis ────────────────────────────
        spread = pdf["ask_price_1"] - pdf["bid_price_1"]
        fig.add_trace(go.Scatter(
            x=pdf["plot_ts"], y=spread,
            name=f"{product} spread",
            line=dict(color=to_rgba(color, 0.45), width=1, dash="dot"),
            legendgroup=product,
            hovertemplate=f"spread=%{{y:.0f}}<extra></extra>",
        ), row=row, col=1, secondary_y=True)

        # ── Market trades ─────────────────────────────────────────────────────
        if not trades.empty and "symbol" in trades.columns:
            tdf = trades[trades["symbol"] == product].copy()
            if not tdf.empty:
                fig.add_trace(go.Scatter(
                    x=tdf["plot_ts"], y=tdf["price"],
                    name=f"{product} trades",
                    mode="markers",
                    marker=dict(
                        size=tdf["quantity"].clip(upper=30) / 3 + 4,
                        color=TRADE_COLOR, opacity=0.75,
                        symbol="circle",
                        line=dict(width=0.8, color="white"),
                    ),
                    legendgroup=product,
                    customdata=tdf[["quantity", "buyer", "seller"]].fillna("—"),
                    hovertemplate=(
                        f"<b>{product} trade</b><br>"
                        "t=%{x}<br>price=%{y}<br>"
                        "qty=%{customdata[0]}<br>"
                        "buyer=%{customdata[1]}<br>"
                        "seller=%{customdata[2]}<extra></extra>"
                    ),
                ), row=row, col=1, secondary_y=False)

        # Price range annotation per product
        # Row 1 axis ref is "x domain"; rows 2+ are "x2 domain", "x3 domain", …
        xref_str = "x domain" if row == 1 else f"x{row} domain"
        yref_str = "y domain" if row == 1 else f"y{row} domain"
        lo, hi = pdf["mid_price"].min(), pdf["mid_price"].max()
        fig.add_annotation(
            x=0.01, y=0.98,
            xref=xref_str, yref=yref_str,
            text=f"<b>{product}</b>  range {lo:.0f}–{hi:.0f}",
            showarrow=False, font=dict(size=11, color=color),
            bgcolor="rgba(0,0,0,0.5)", bordercolor=color, borderwidth=1,
            xanchor="left", yanchor="top",
        )

        fig.update_yaxes(title_text="Price",  row=row, col=1, secondary_y=False)
        fig.update_yaxes(
            title_text="Spread", row=row, col=1, secondary_y=True,
            showgrid=False, rangemode="nonnegative",
        )

    # ── Trade volume panel ────────────────────────────────────────────────────
    vol_row = n + 1
    if not trades.empty:
        for i, product in enumerate(products):
            color = COLORS[i % len(COLORS)]
            tdf = trades[trades["symbol"] == product].copy()
            if tdf.empty:
                continue
            # Bin into 5000-ts buckets
            tdf["bucket"] = (tdf["plot_ts"] // 5000) * 5000
            vol = tdf.groupby("bucket")["quantity"].sum().reset_index()
            fig.add_trace(go.Bar(
                x=vol["bucket"], y=vol["quantity"],
                name=f"{product} vol",
                marker_color=to_rgba(color, 0.65),
                legendgroup=f"{product}_vol",
            ), row=vol_row, col=1)

    fig.update_yaxes(title_text="Volume", row=vol_row, col=1)
    fig.update_xaxes(
        title_text="Timestamp",
        tickvals=tick_vals, ticktext=tick_text,
        row=vol_row, col=1,
    )

    # ── Summary stats annotation ──────────────────────────────────────────────
    lines = [f"<b>Round {ROUND_NUM} — Days {', '.join(str(d) for d in days)}</b>"]
    for product in products:
        pdf = prices[prices["product"] == product]
        n_ticks = len(pdf)
        avg_spread = (pdf["ask_price_1"] - pdf["bid_price_1"]).mean()
        if not trades.empty and "symbol" in trades.columns:
            n_trades = len(trades[trades["symbol"] == product])
        else:
            n_trades = 0
        lines.append(
            f"{product}:  {n_ticks:,} ticks · "
            f"avg spread {avg_spread:.1f} · {n_trades:,} trades"
        )
    fig.add_annotation(
        x=1, y=1, xref="paper", yref="paper",
        text="<br>".join(lines),
        showarrow=False, font=dict(size=11, color="white"),
        bgcolor="rgba(20,20,40,0.8)", bordercolor="#666", borderwidth=1,
        xanchor="right", yanchor="top",
        align="left",
    )

    fig.update_layout(
        title=dict(text=f"Round {ROUND_NUM} Market Data", font=dict(size=20)),
        height=380 * n + 240,
        template="plotly_dark",
        hovermode="x unified",
        barmode="stack",
        legend=dict(orientation="v", x=1.02, y=0.98),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Round 1 raw market data")
    parser.add_argument("--data",    default=str(DATA_ROOT), help="Data root dir")
    parser.add_argument("--output",  default=None,           help="Output HTML path")
    parser.add_argument("--no-open", action="store_true",    help="Skip browser open")
    args = parser.parse_args()

    data_root = Path(args.data)
    if not data_root.is_dir():
        print(f"Error: data directory not found: {data_root}")
        raise SystemExit(1)

    print("Loading price data...")
    prices = load_prices(data_root, ROUND_NUM)
    prices = add_plot_ts(prices, sorted(prices["day"].unique()))
    print(f"  {len(prices):,} rows | Products: {sorted(prices['product'].unique())}")

    print("Loading trade data...")
    try:
        trades = load_trades(data_root, ROUND_NUM)
        trades = add_plot_ts(trades, sorted(prices["day"].unique()))
        print(f"  {len(trades):,} trades")
    except FileNotFoundError:
        trades = pd.DataFrame()
        print("  No trade files found — skipping trade overlays")

    print("Building chart...")
    fig = build_figure(prices, trades)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else VIZ_DIR / OUTPUT_NAME
    fig.write_html(str(out), include_plotlyjs="cdn", auto_open=False)
    print(f"Saved: {out}")

    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
