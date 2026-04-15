"""Visualize tutorial round prices with wall mid and trades overlay."""

import json
from pathlib import Path
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert hex color to rgba string."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def compute_wall_mid(bid_series, ask_series):
    """Compute wall mid as average of bid and ask wall prices."""
    return (bid_series + ask_series) / 2


def build_chart(day: int) -> go.Figure:
    """Build price chart with wall mid and trades for a day."""
    prices_file = Path(f"TUTORIAL_ROUND_1/prices_round_0_day_{day}.csv")
    trades_file = Path(f"TUTORIAL_ROUND_1/trades_round_0_day_{day}.csv")

    # Load data
    prices = pd.read_csv(prices_file, sep=";")
    trades = pd.read_csv(trades_file, sep=";")

    products = sorted(prices["product"].unique())
    n = len(products)

    # Create subplots
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=products,
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    colors = ["#636EFA", "#EF553B"]

    for i, product in enumerate(products):
        row = i + 1
        color = colors[i % len(colors)]

        # Filter data for product
        df = prices[prices["product"] == product].copy().sort_values("timestamp")

        # Compute wall mid (bid_price_1 to ask_price_1)
        df["wall_mid"] = compute_wall_mid(df["bid_price_1"], df["ask_price_1"])

        # Plot bid/ask spread
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bid_price_1"],
                name=f"{product} bid",
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.5,
                legendgroup=product,
                showlegend=True,
            ),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["ask_price_1"],
                name=f"{product} ask",
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.5,
                fill="tonexty",
                fillcolor=hex_to_rgba(color, 0.08),
                legendgroup=product,
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Plot wall mid
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["wall_mid"],
                name=f"{product} wall mid",
                line=dict(color=color, width=2.5),
                legendgroup=product,
                showlegend=True,
            ),
            row=row,
            col=1,
        )

        # Overlay trades
        product_trades = trades[trades["symbol"] == product].copy()
        if not product_trades.empty:
            # Determine trade direction (all trades in tutorial are market sells to us)
            # buyer/seller columns show who initiated, if empty = market trade
            product_trades["price"] = product_trades["price"].astype(float)

            fig.add_trace(
                go.Scatter(
                    x=product_trades["timestamp"],
                    y=product_trades["price"],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=12,
                        color="#FFD700",
                        line=dict(width=1, color="#FFD700"),
                    ),
                    name=f"{product} trades",
                    text=product_trades["quantity"].astype(str),
                    textposition="top center",
                    textfont=dict(size=9, color="#FFD700"),
                    legendgroup=product,
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{product} trade</b><br>"
                        "t=%{x}<br>price=%{y}<br>qty=%{text}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text="Price", row=row, col=1)

    fig.update_layout(
        title=dict(text=f"Tutorial Round — Day {day}", font=dict(size=18)),
        height=300 * n + 100,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1),
    )

    fig.update_xaxes(title_text="Timestamp", row=n, col=1)

    return fig


def main():
    """Generate visualizations for both days."""
    for day in [-1, -2]:
        print(f"Generating chart for day {day}...")
        fig = build_chart(day)

        output_file = f"logs/tutorial_round_day_{day}_viz.html"
        fig.write_html(output_file, include_plotlyjs="cdn", auto_open=False)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
