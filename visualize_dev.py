"""
Extended visualizer — dev use only, not for submission.

Adds on top of visualize_log:
  - Separate "PnL Breakdown" panel: total / unrealized / realized per product
  - Numerical PnL summary table (final cumulative values)

Usage:
  python visualize_dev.py
    -> builds logs/viz_dashboard_dev.html with dropdown

  python visualize_dev.py --log 3066/3066
    -> builds logs/3066_3066_viz_dev.html
"""

import argparse
import webbrowser
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualize_log import (
    LOG_DIR,
    list_log_files,
    load_log,
    build_volume_summary,
    build_dashboard_html,
    to_rgba,
    classify_trade_side,
    safe_file_name,
)


# ── PnL computation ───────────────────────────────────────────────────────────

def compute_unrealized_pnl(trades: pd.DataFrame, acts: pd.DataFrame, product: str) -> pd.Series:
    """
    Unrealized PnL = position * mid_price − cost_basis_of_open_position.
    Average-cost accounting; handles position flips.
    Returns a Series aligned to the product rows of `acts`.
    """
    my_trades = (
        trades[trades["symbol"] == product]
        .assign(
            signed_qty=lambda df: df.apply(
                lambda r: r["quantity"] if r["buyer"] == "SUBMISSION"
                else -r["quantity"] if r["seller"] == "SUBMISSION"
                else 0,
                axis=1,
            )
        )
        .query("signed_qty != 0")
        .sort_values("timestamp")
        [["timestamp", "price", "signed_qty"]]
        .reset_index(drop=True)
    )

    position, cost_basis = 0.0, 0.0
    snapshots: list[tuple[int, float, float]] = []

    for _, tr in my_trades.iterrows():
        qty, price = tr["signed_qty"], tr["price"]
        new_pos = position + qty
        if position == 0:
            cost_basis = qty * price
        elif (position > 0) == (qty > 0):
            cost_basis += qty * price
        else:
            if abs(new_pos) < abs(position):
                cost_basis = cost_basis * (new_pos / position) if position else 0.0
            else:
                cost_basis = new_pos * price
        position = new_pos
        snapshots.append((int(tr["timestamp"]), position, cost_basis))

    product_acts = acts[acts["product"] == product].sort_values("timestamp")
    unrealized, pos, cb, snap_idx = [], 0.0, 0.0, 0

    for _, row in product_acts.iterrows():
        ts = int(row["timestamp"])
        while snap_idx < len(snapshots) and snapshots[snap_idx][0] <= ts:
            _, pos, cb = snapshots[snap_idx]
            snap_idx += 1
        unrealized.append(pos * row["mid_price"] - cb if pos != 0 else 0.0)

    return pd.Series(unrealized, index=product_acts.index)


# ── Chart builder ─────────────────────────────────────────────────────────────

def build_chart_dev(data: dict, log_label: str) -> go.Figure:
    acts = pd.read_csv(StringIO(data["activitiesLog"]), sep=";").sort_values("timestamp")
    products = sorted(acts["product"].unique())
    trades = pd.DataFrame(data.get("tradeHistory", []))
    n = len(products)

    has_trades = (
        not trades.empty
        and {"symbol", "buyer", "seller", "price", "quantity"}.issubset(trades.columns)
    )

    # Pre-compute per-product PnL breakdown
    pnl: dict[str, dict] = {}
    for product in products:
        df = acts[acts["product"] == product].sort_values("timestamp")
        total = df["profit_and_loss"].values
        unreal = (
            compute_unrealized_pnl(trades, acts, product).values
            if has_trades else [0.0] * len(df)
        )
        pnl[product] = {
            "ts":         df["timestamp"].values,
            "total":      total,
            "unrealized": unreal,
            "realized":   total - unreal,
        }

    # Rows: n product panels + Total PnL + PnL Breakdown + Summary table
    specs = (
        [[{"secondary_y": True}]] * n
        + [[{"secondary_y": False}]]          # Total PnL
        + [[{"secondary_y": False}]]          # PnL Breakdown
        + [[{"type": "table"}]]               # Summary table
    )
    fig = make_subplots(
        rows=n + 3,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[*products, "Total PnL", "PnL Breakdown (total / unrealized / realized)", ""],
        row_heights=[0.25] * n + [0.18, 0.18, 0.14],
        vertical_spacing=0.04,
        specs=specs,
    )

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    # ── Product panels (price + bid/ask + volume bars + trade markers) ────────
    TRADE_STYLES = {
        "buy":     ("triangle-up",  "#00CC44", 13, 2,  "MY BUY"),
        "sell":    ("triangle-down", "#FF4444", 13, 2,  "MY SELL"),
        "unknown": ("circle",        "#888888",  7, 0,  "TRADE"),
    }

    for i, product in enumerate(products):
        row = i + 1
        color = colors[i % len(colors)]
        df = acts[acts["product"] == product].sort_values("timestamp")

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["mid_price"],
            name=f"{product} mid",
            line=dict(color=color, width=1.6),
            legendgroup=product,
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["bid_price_1"],
            name=f"{product} bid",
            line=dict(color=color, width=0.8, dash="dot"),
            opacity=0.45, legendgroup=product, showlegend=False,
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["ask_price_1"],
            name=f"{product} ask",
            line=dict(color=color, width=0.8, dash="dot"),
            opacity=0.45, fill="tonexty", fillcolor=to_rgba(color, 0.08),
            legendgroup=product, showlegend=False,
        ), row=row, col=1, secondary_y=False)

        if has_trades:
            pt = trades[trades["symbol"] == product].copy()
            if not pt.empty:
                vol = pt.groupby("timestamp", as_index=False)["quantity"].sum().sort_values("timestamp")
                fig.add_trace(go.Bar(
                    x=vol["timestamp"], y=vol["quantity"],
                    name=f"{product} volume/tick",
                    marker_color=to_rgba(color, 0.35),
                    legendgroup=product, showlegend=False,
                    hovertemplate=f"<b>{product} volume</b><br>t=%{{x}}<br>vol=%{{y}}<extra></extra>",
                ), row=row, col=1, secondary_y=True)

                if {"buyer", "seller", "price"}.issubset(pt.columns):
                    pt["side"] = pt.apply(classify_trade_side, axis=1)
                    for side, (symbol, mcolor, size, outline_w, label) in TRADE_STYLES.items():
                        subset = pt[pt["side"] == side]
                        if subset.empty:
                            continue
                        is_mine = side in ("buy", "sell")
                        fig.add_trace(go.Scatter(
                            x=subset["timestamp"], y=subset["price"],
                            mode="markers+text" if is_mine else "markers",
                            marker=dict(
                                symbol=symbol, size=size, color=mcolor,
                                line=dict(width=outline_w, color="white") if outline_w else dict(width=0),
                            ),
                            text=subset["quantity"].astype(str) if is_mine else None,
                            textposition="top center", textfont=dict(size=9),
                            name=f"{product} {label}",
                            legendgroup=product,
                            opacity=1.0 if is_mine else 0.55,
                            customdata=subset["quantity"],
                            hovertemplate=(
                                f"<b>{product} {label}</b><br>"
                                "t=%{x}<br>price=%{y}<br>qty=%{customdata}<extra></extra>"
                            ),
                        ), row=row, col=1, secondary_y=False)

        fig.update_yaxes(title_text="Price",  row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Volume", row=row, col=1, secondary_y=True, rangemode="tozero")

    # ── Total PnL panel ───────────────────────────────────────────────────────
    pnl_row = n + 1
    for i, product in enumerate(products):
        color = colors[i % len(colors)]
        d = pnl[product]
        fig.add_trace(go.Scatter(
            x=d["ts"], y=d["total"],
            name=f"{product} PnL",
            line=dict(color=color, width=2),
            legendgroup=f"{product}_pnl",
            fill="tozeroy", fillcolor=to_rgba(color, 0.12),
        ), row=pnl_row, col=1)
    fig.update_yaxes(title_text="PnL", row=pnl_row, col=1)
    fig.update_xaxes(title_text="Timestamp", row=pnl_row, col=1)

    # ── PnL Breakdown panel ───────────────────────────────────────────────────
    breakdown_row = n + 2
    for i, product in enumerate(products):
        color = colors[i % len(colors)]
        d = pnl[product]
        fig.add_trace(go.Scatter(
            x=d["ts"], y=d["total"],
            name=f"{product} total",
            line=dict(color=color, width=2),
            legendgroup=f"{product}_bd",
        ), row=breakdown_row, col=1)
        fig.add_trace(go.Scatter(
            x=d["ts"], y=d["unrealized"],
            name=f"{product} unrealized",
            line=dict(color=color, width=1.4, dash="dash"),
            legendgroup=f"{product}_bd", opacity=0.8,
        ), row=breakdown_row, col=1)
        fig.add_trace(go.Scatter(
            x=d["ts"], y=d["realized"],
            name=f"{product} realized",
            line=dict(color=color, width=1.4, dash="dot"),
            legendgroup=f"{product}_bd", opacity=0.8,
        ), row=breakdown_row, col=1)
    fig.update_yaxes(title_text="PnL", row=breakdown_row, col=1)
    fig.update_xaxes(title_text="Timestamp", row=breakdown_row, col=1)

    # ── Summary table ─────────────────────────────────────────────────────────
    table_row = n + 3
    rows = [
        (product, pnl[product]["total"][-1], pnl[product]["unrealized"][-1], pnl[product]["realized"][-1])
        for product in products
        if len(pnl[product]["total"]) > 0
    ]
    grand = ("TOTAL", sum(r[1] for r in rows), sum(r[2] for r in rows), sum(r[3] for r in rows))
    rows.append(grand)

    def fmt(v): return f"{v:+.2f}"
    def cell_colors(vals):
        return ["#2a3d2a" if v >= 0 else "#3d2a2a" for v in vals]

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Product</b>", "<b>Total PnL</b>", "<b>Unrealized</b>", "<b>Realized</b>"],
            fill_color="#1e1e1e", font=dict(color="white", size=13),
            align=["left", "right", "right", "right"], height=30,
        ),
        cells=dict(
            values=[
                [r[0] for r in rows],
                [fmt(r[1]) for r in rows],
                [fmt(r[2]) for r in rows],
                [fmt(r[3]) for r in rows],
            ],
            fill_color=[
                ["#252525"] * len(rows),
                cell_colors([r[1] for r in rows]),
                cell_colors([r[2] for r in rows]),
                cell_colors([r[3] for r in rows]),
            ],
            font=dict(color="white", size=12),
            align=["left", "right", "right", "right"],
            height=28,
        ),
    ), row=table_row, col=1)

    total_trades = len(trades) if not trades.empty else 0
    fig.update_layout(
        title=dict(text=f"IMC Prosperity — Log {log_label} | {total_trades} trades", font=dict(size=18)),
        height=300 * n + 200 + 200 + 160,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1),
        barmode="overlay",
    )
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dev visualizer with PnL breakdown")
    parser.add_argument("--log", help="Log key (e.g. 3066/3066)")
    parser.add_argument("--output", help="Output HTML path")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = list_log_files(LOG_DIR)
    if not logs:
        raise FileNotFoundError("No .json logs found under logs/")

    if args.log:
        if args.log not in logs:
            available = ", ".join(sorted(logs.keys()))
            raise KeyError(f"Unknown log '{args.log}'. Available: {available}")

        data = load_log(logs[args.log])
        fig = build_chart_dev(data, args.log)
        default_name = f"{safe_file_name(args.log)}_viz_dev.html"
        output = Path(args.output) if args.output else LOG_DIR / default_name
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output), include_plotlyjs="cdn", auto_open=not args.no_open)
        print(f"Saved: {output}")
        return

    figures: dict[str, dict] = {}
    summaries: dict[str, list[dict]] = {}
    for log_key, path in logs.items():
        data = load_log(path)
        fig = build_chart_dev(data, log_key)
        figures[log_key] = fig.to_plotly_json()
        summaries[log_key] = build_volume_summary(data)

    output = Path(args.output) if args.output else LOG_DIR / "viz_dashboard_dev.html"
    build_dashboard_html(figures, summaries, output)
    print(f"Saved dashboard: {output}")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
