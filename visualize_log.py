"""
Visualize IMC Prosperity simulation log.
Usage: python visualize_log.py [log_id]   e.g.  python visualize_log.py 2957
Opens an interactive HTML chart in your browser.
"""

import json
import sys
from io import StringIO
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOG_DIR = "logs"


def load(log_id: str) -> dict:
    path = f"{LOG_DIR}/{log_id}.log"
    with open(path) as f:
        return json.load(f)


def build_chart(data: dict, log_id: str) -> go.Figure:
    # --- activities (market data) ---
    acts = pd.read_csv(StringIO(data["activitiesLog"]), sep=";")
    products = sorted(acts["product"].unique())

    # --- trade history ---
    trades = pd.DataFrame(data.get("tradeHistory", []))

    n = len(products)
    # rows: one per product (price + trades) + 1 PnL row
    row_heights = [0.35] * n + [0.3]
    fig = make_subplots(
        rows=n + 1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[*products, "PnL"],
        row_heights=row_heights,
        vertical_spacing=0.06,
    )

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    for i, product in enumerate(products):
        row = i + 1
        color = colors[i % len(colors)]
        df = acts[acts["product"] == product].sort_values("timestamp")

        # Mid price line
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["mid_price"],
                name=f"{product} mid",
                line=dict(color=color, width=1.5),
                legendgroup=product,
            ),
            row=row,
            col=1,
        )

        # Best bid / ask as shaded area
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bid_price_1"],
                name=f"{product} bid",
                line=dict(color=color, width=0.5, dash="dot"),
                opacity=0.5,
                legendgroup=product,
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["ask_price_1"],
                name=f"{product} ask",
                line=dict(color=color, width=0.5, dash="dot"),
                opacity=0.5,
                fill="tonexty",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
                legendgroup=product,
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Trade markers
        if not trades.empty:
            pt = trades[trades["symbol"] == product].copy()
            if not pt.empty:
                # Determine buy/sell from buyer/seller fields
                # If neither buyer nor seller is set, we can't tell direction — show as neutral
                def classify(r):
                    if r["buyer"] == "SUBMISSION":
                        return "buy"
                    if r["seller"] == "SUBMISSION":
                        return "sell"
                    return "unknown"

                pt["side"] = pt.apply(classify, axis=1)

                for side, marker_symbol, marker_color, label in [
                    ("buy",     "triangle-up",   "#00CC44", "BUY"),
                    ("sell",    "triangle-down",  "#FF4444", "SELL"),
                    ("unknown", "circle",         "#FFAA00", "TRADE"),
                ]:
                    sub = pt[pt["side"] == side]
                    if sub.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=sub["timestamp"],
                            y=sub["price"],
                            mode="markers+text",
                            marker=dict(symbol=marker_symbol, size=10, color=marker_color,
                                        line=dict(width=1, color="white")),
                            text=sub["quantity"].astype(str),
                            textposition="top center",
                            textfont=dict(size=9),
                            name=f"{product} {label}",
                            legendgroup=product,
                            hovertemplate=(
                                f"<b>{product} {label}</b><br>"
                                "t=%{x}<br>price=%{y}<br>"
                                "qty=%{text}<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=1,
                    )

        fig.update_yaxes(title_text="Price", row=row, col=1)

    # --- PnL chart (one line per product) ---
    pnl_row = n + 1
    for i, product in enumerate(products):
        df = acts[acts["product"] == product].sort_values("timestamp")
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["profit_and_loss"],
                name=f"{product} PnL",
                line=dict(color=color, width=2),
                legendgroup=product + "_pnl",
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
            ),
            row=pnl_row,
            col=1,
        )

    fig.update_yaxes(title_text="PnL", row=pnl_row, col=1)
    fig.update_xaxes(title_text="Timestamp", row=pnl_row, col=1)

    total_trades = len(trades) if not trades.empty else 0
    fig.update_layout(
        title=dict(
            text=f"IMC Prosperity — Log {log_id}  |  {total_trades} trades",
            font=dict(size=18),
        ),
        height=350 * (n + 1),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1),
    )
    return fig


def print_trade_table(data: dict) -> None:
    trades = pd.DataFrame(data.get("tradeHistory", []))
    if trades.empty:
        print("No trades found.")
        return
    print(f"\n{'='*65}")
    print(f"  Trade History  ({len(trades)} trades)")
    print(f"{'='*65}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(trades[["timestamp", "symbol", "price", "quantity", "buyer", "seller"]].to_string(index=False))


if __name__ == "__main__":
    log_id = sys.argv[1] if len(sys.argv) > 1 else "2957"

    data = load(log_id)
    print_trade_table(data)

    fig = build_chart(data, log_id)
    out = f"logs/{log_id}_viz.html"
    fig.write_html(out, auto_open=True)
    print(f"\nChart saved → {out}  (opened in browser)")
