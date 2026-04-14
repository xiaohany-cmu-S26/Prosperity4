"""
Visualize backtester output CSVs.

Usage:
  python visualize_backtest.py --algo 3197
    -> reads logs/backtest_3197_day_*.csv and writes logs/visuals/backtest_3197_viz.html

  python visualize_backtest.py --algo trading
  python visualize_backtest.py --algo 3197 --no-open
"""

import argparse
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOG_DIR = Path("logs")
VIZ_DIR = LOG_DIR / "visuals"

COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
BID_COLOR = "#26C281"
ASK_COLOR = "#EF5350"


def to_rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def load_backtest_csvs(algo_name: str) -> pd.DataFrame:
    """Load and concatenate all backtest CSVs for an algo."""
    paths = sorted(LOG_DIR.glob(f"backtest_{algo_name}_day_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No backtest CSVs found for algo '{algo_name}' in {LOG_DIR}/\n"
            f"Expected files like: backtest_{algo_name}_day_-1.csv"
        )
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep=";", dtype=str)
        frames.append(df)
        print(f"  Loaded {p.name} ({len(df)} rows)")
    combined = pd.concat(frames, ignore_index=True)

    # Parse numeric columns
    numeric_cols = [
        "day", "timestamp",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
        "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2",
        "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss", "position",
        "our_bid", "our_bid_vol", "our_ask", "our_ask_vol",
        "fill_price", "fill_qty",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    return combined.sort_values(["day", "timestamp"])


def build_chart(df: pd.DataFrame, algo_name: str) -> go.Figure:
    products = sorted(df["product"].unique())
    n = len(products)
    days = sorted(df["day"].unique())
    day_label = " & ".join(f"Day {d}" for d in days)

    # Rows: one per product + PnL panel
    specs = [[{"secondary_y": True}]] * n + [[{"secondary_y": False}]]
    fig = make_subplots(
        rows=n + 1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[*products, "PnL"],
        row_heights=[0.32] * n + [0.28],
        vertical_spacing=0.06,
        specs=specs,
    )

    for i, product in enumerate(products):
        row = i + 1
        color = COLORS[i % len(COLORS)]
        pdf = df[df["product"] == product].copy().sort_values("timestamp")

        # ── Mid price ───────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=pdf["timestamp"], y=pdf["mid_price"],
            name=f"{product} mid",
            line=dict(color=color, width=1.8),
            legendgroup=product,
        ), row=row, col=1, secondary_y=False)

        # ── Bid/ask spread ──────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=pdf["timestamp"], y=pdf["bid_price_1"],
            name=f"{product} bid1",
            line=dict(color=BID_COLOR, width=0.8, dash="dot"),
            opacity=0.55, legendgroup=product, showlegend=False,
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=pdf["timestamp"], y=pdf["ask_price_1"],
            name=f"{product} ask1",
            line=dict(color=ASK_COLOR, width=0.8, dash="dot"),
            opacity=0.55, fill="tonexty", fillcolor=to_rgba(BID_COLOR, 0.06),
            legendgroup=product, showlegend=False,
        ), row=row, col=1, secondary_y=False)

        # ── Our quoted bid/ask ──────────────────────────────────────────────
        our_bid = pdf[pdf["our_bid"].notna() & (pdf["our_bid"] != "")]
        our_ask = pdf[pdf["our_ask"].notna() & (pdf["our_ask"] != "")]

        if not our_bid.empty:
            fig.add_trace(go.Scatter(
                x=our_bid["timestamp"], y=our_bid["our_bid"],
                name=f"{product} our bid",
                mode="markers",
                marker=dict(symbol="line-ew", size=6, color=BID_COLOR, opacity=0.7,
                            line=dict(width=1.5, color=BID_COLOR)),
                legendgroup=product, showlegend=True,
            ), row=row, col=1, secondary_y=False)

        if not our_ask.empty:
            fig.add_trace(go.Scatter(
                x=our_ask["timestamp"], y=our_ask["our_ask"],
                name=f"{product} our ask",
                mode="markers",
                marker=dict(symbol="line-ew", size=6, color=ASK_COLOR, opacity=0.7,
                            line=dict(width=1.5, color=ASK_COLOR)),
                legendgroup=product, showlegend=True,
            ), row=row, col=1, secondary_y=False)

        # ── Fill markers ────────────────────────────────────────────────────
        fills = pdf[pdf["fill_side"].notna() & (pdf["fill_side"] != "")]
        buys  = fills[fills["fill_side"] == "BUY"]
        sells = fills[fills["fill_side"] == "SELL"]

        for side_df, sym, mcolor, label in [
            (buys,  "triangle-up",   "#00CC44", "BUY fill"),
            (sells, "triangle-down", "#FF4444", "SELL fill"),
        ]:
            if side_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=side_df["timestamp"], y=side_df["fill_price"],
                mode="markers+text",
                marker=dict(symbol=sym, size=11, color=mcolor,
                            line=dict(width=1.5, color="white")),
                text=side_df["fill_qty"].astype(int).astype(str),
                textposition="top center",
                textfont=dict(size=8, color=mcolor),
                name=f"{product} {label}",
                legendgroup=product,
                customdata=side_df[["fill_qty", "fill_type"]],
                hovertemplate=(
                    f"<b>{product} {label}</b><br>"
                    "t=%{x}<br>price=%{y}<br>"
                    "qty=%{customdata[0]}<br>type=%{customdata[1]}<extra></extra>"
                ),
            ), row=row, col=1, secondary_y=False)

        # ── Position (secondary y) ──────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=pdf["timestamp"], y=pdf["position"],
            name=f"{product} position",
            line=dict(color=to_rgba(color, 0.5), width=1.2, dash="dash"),
            legendgroup=product,
        ), row=row, col=1, secondary_y=True)

        fig.update_yaxes(title_text="Price",    row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Position", row=row, col=1, secondary_y=True,
                         showgrid=False, zeroline=True, zerolinecolor="#555")

    # ── PnL panel ──────────────────────────────────────────────────────────────
    pnl_row = n + 1
    total_pnl = None
    for i, product in enumerate(products):
        color = COLORS[i % len(COLORS)]
        pdf = df[df["product"] == product].sort_values("timestamp")
        pnl_series = pdf["profit_and_loss"]

        fig.add_trace(go.Scatter(
            x=pdf["timestamp"], y=pnl_series,
            name=f"{product} PnL",
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=to_rgba(color, 0.1),
            legendgroup=f"{product}_pnl",
        ), row=pnl_row, col=1)

        if total_pnl is None:
            total_pnl = pnl_series.reset_index(drop=True)
            total_ts = pdf["timestamp"].reset_index(drop=True)
        else:
            total_pnl = total_pnl.add(pnl_series.reset_index(drop=True), fill_value=0)

    if len(products) > 1 and total_pnl is not None:
        fig.add_trace(go.Scatter(
            x=total_ts, y=total_pnl,
            name="TOTAL PnL",
            line=dict(color="white", width=2.5),
            legendgroup="total_pnl",
        ), row=pnl_row, col=1)

    # Final PnL annotation
    if total_pnl is not None:
        final_val = total_pnl.iloc[-1] if not total_pnl.empty else 0.0
        fig.add_annotation(
            x=1, y=1, xref="paper", yref="paper",
            text=f"<b>Final PnL: {final_val:+,.0f}</b>",
            showarrow=False,
            font=dict(size=14, color="#FFD700"),
            bgcolor="#1e1e1e", bordercolor="#FFD700", borderwidth=1,
            xanchor="right", yanchor="top",
        )

    fig.update_yaxes(title_text="PnL",       row=pnl_row, col=1)
    fig.update_xaxes(title_text="Timestamp", row=pnl_row, col=1)

    fig.update_layout(
        title=dict(
            text=f"Backtest — {algo_name} | {day_label}",
            font=dict(size=18),
        ),
        height=320 * n + 280,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="v", x=1.02, y=1),
        barmode="overlay",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize backtester CSV output")
    parser.add_argument("--algo", required=True, help="Algo name (e.g. 3197 or trading)")
    parser.add_argument("--output", help="Output HTML path")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    args = parser.parse_args()

    print(f"Loading backtest data for algo: {args.algo}")
    df = load_backtest_csvs(args.algo)
    print(f"  Total rows: {len(df)} | Products: {sorted(df['product'].unique())}")

    fig = build_chart(df, args.algo)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else VIZ_DIR / f"backtest_{args.algo}_viz.html"
    fig.write_html(str(output), include_plotlyjs="cdn", auto_open=False)
    print(f"Saved: {output}")

    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
