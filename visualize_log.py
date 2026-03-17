"""
General IMC Prosperity log visualizer.

Usage:
  python visualize_log.py
    -> builds logs/viz_dashboard.html with a dropdown to choose any log

  python visualize_log.py --log 2957
    -> builds logs/2957_viz.html for one log
"""

import argparse
import json
import re
import webbrowser
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOG_DIR = Path("logs")


def list_log_files(log_dir: Path) -> dict[str, Path]:
    files = sorted(log_dir.rglob("*.json"))
    logs: dict[str, Path] = {}
    for path in files:
        key = path.relative_to(log_dir).with_suffix("").as_posix()
        logs[key] = path
    return logs


def load_log(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    # The .json export omits tradeHistory; the sibling .log file contains it.
    if not data.get("tradeHistory"):
        sibling = path.with_suffix(".log")
        if sibling.exists():
            with sibling.open() as f:
                log_data = json.load(f)
            if log_data.get("tradeHistory"):
                data["tradeHistory"] = log_data["tradeHistory"]
    return data


def to_rgba(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def classify_trade_side(row: pd.Series) -> str:
    if row.get("buyer") == "SUBMISSION":
        return "buy"
    if row.get("seller") == "SUBMISSION":
        return "sell"
    return "unknown"


def build_chart(data: dict, log_label: str) -> go.Figure:
    acts = pd.read_csv(StringIO(data["activitiesLog"]), sep=";")
    acts = acts.sort_values("timestamp")
    products = sorted(acts["product"].unique())
    trades = pd.DataFrame(data.get("tradeHistory", []))

    n = len(products)
    specs = [[{"secondary_y": True}] for _ in range(n)] + [[{"secondary_y": False}]]
    fig = make_subplots(
        rows=n + 1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[*products, "PnL"],
        row_heights=[0.3] * n + [0.28],
        vertical_spacing=0.06,
        specs=specs,
    )

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    for i, product in enumerate(products):
        row = i + 1
        color = colors[i % len(colors)]
        df = acts[acts["product"] == product].copy().sort_values("timestamp")

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["mid_price"],
                name=f"{product} mid",
                line=dict(color=color, width=1.6),
                legendgroup=product,
            ),
            row=row,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bid_price_1"],
                name=f"{product} bid",
                line=dict(color=color, width=0.8, dash="dot"),
                opacity=0.45,
                legendgroup=product,
                showlegend=False,
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["ask_price_1"],
                name=f"{product} ask",
                line=dict(color=color, width=0.8, dash="dot"),
                opacity=0.45,
                fill="tonexty",
                fillcolor=to_rgba(color, 0.08),
                legendgroup=product,
                showlegend=False,
            ),
            row=row,
            col=1,
            secondary_y=False,
        )

        if not trades.empty and {"symbol", "timestamp", "quantity"}.issubset(trades.columns):
            product_trades = trades[trades["symbol"] == product].copy()
            if not product_trades.empty:
                volume_per_tick = (
                    product_trades.groupby("timestamp", as_index=False)["quantity"].sum()
                    .sort_values("timestamp")
                )
                fig.add_trace(
                    go.Bar(
                        x=volume_per_tick["timestamp"],
                        y=volume_per_tick["quantity"],
                        name=f"{product} volume/tick",
                        marker_color=to_rgba(color, 0.35),
                        legendgroup=product,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{product} volume</b><br>"
                            "t=%{x}<br>volume=%{y}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                    secondary_y=True,
                )

                if {"buyer", "seller", "price"}.issubset(product_trades.columns):
                    product_trades["side"] = product_trades.apply(classify_trade_side, axis=1)
                    # (symbol, color, size, outline_width, label)
                    # My trades: large solid triangles with thick white outline
                    # Market trades: small dim circle, no outline
                    TRADE_STYLES = {
                        "buy":     ("triangle-up",   "#00CC44", 13, 2,   "MY BUY"),
                        "sell":    ("triangle-down",  "#FF4444", 13, 2,   "MY SELL"),
                        "unknown": ("circle",         "#888888",  7, 0,   "TRADE"),
                    }
                    for side, (symbol, marker_color, size, outline_w, label) in TRADE_STYLES.items():
                        subset = product_trades[product_trades["side"] == side]
                        if subset.empty:
                            continue
                        is_mine = side in ("buy", "sell")
                        fig.add_trace(
                            go.Scatter(
                                x=subset["timestamp"],
                                y=subset["price"],
                                mode="markers+text" if is_mine else "markers",
                                marker=dict(
                                    symbol=symbol,
                                    size=size,
                                    color=marker_color,
                                    line=dict(width=outline_w, color="white") if outline_w else dict(width=0),
                                ),
                                text=subset["quantity"].astype(str) if is_mine else None,
                                textposition="top center",
                                textfont=dict(size=9),
                                name=f"{product} {label}",
                                legendgroup=product,
                                opacity=1.0 if is_mine else 0.55,
                                hovertemplate=(
                                    f"<b>{product} {label}</b><br>"
                                    "t=%{x}<br>price=%{y}<br>qty=%{customdata}<extra></extra>"
                                ),
                                customdata=subset["quantity"],
                            ),
                            row=row,
                            col=1,
                            secondary_y=False,
                        )

        fig.update_yaxes(title_text="Price", row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Volume", row=row, col=1, secondary_y=True, rangemode="tozero")

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
                legendgroup=f"{product}_pnl",
                fill="tozeroy",
                fillcolor=to_rgba(color, 0.12),
            ),
            row=pnl_row,
            col=1,
        )

    total_trades = len(trades) if not trades.empty else 0
    fig.update_yaxes(title_text="PnL", row=pnl_row, col=1)
    fig.update_xaxes(title_text="Timestamp", row=pnl_row, col=1)
    fig.update_layout(
        title=dict(text=f"IMC Prosperity — Log {log_label} | {total_trades} trades", font=dict(size=18)),
        height=340 * (n + 1),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1),
        barmode="overlay",
    )
    return fig


def build_volume_summary(data: dict) -> list[dict]:
    trades = pd.DataFrame(data.get("tradeHistory", []))
    if trades.empty or not {"symbol", "quantity"}.issubset(trades.columns):
        return []

    summary = (
        trades.groupby("symbol", as_index=False)
        .agg(total_volume=("quantity", "sum"), trade_count=("quantity", "size"))
        .sort_values("symbol")
    )
    return summary.to_dict(orient="records")


def build_dashboard_html(figures: dict[str, dict], summaries: dict[str, list[dict]], output_path: Path) -> None:
    keys = sorted(figures.keys())
    if not keys:
        raise ValueError("No logs available to visualize.")

    fig_json = json.dumps(figures)
    summaries_json = json.dumps(summaries)
    option_html = "\n".join([f'<option value="{k}">{k}</option>' for k in keys])

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>IMC Prosperity Log Dashboard</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {{ background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; }}
    .toolbar {{ padding:12px 16px; border-bottom:1px solid #333; display:flex; gap:10px; align-items:center; }}
    select {{ background:#1e1e1e; color:#eee; border:1px solid #444; border-radius:6px; padding:6px 10px; min-width:220px; }}
        #chart {{ width:100%; min-height:700px; }}
        .summary {{ padding: 10px 16px 20px; }}
        table {{ border-collapse: collapse; width: 460px; max-width: 100%; background:#171717; }}
        th, td {{ border: 1px solid #333; padding: 7px 10px; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
        th {{ background:#202020; }}
  </style>
</head>
<body>
  <div class=\"toolbar\">
    <label for=\"logSelect\">Choose log:</label>
    <select id=\"logSelect\">{option_html}</select>
  </div>
  <div id=\"chart\"></div>
    <div class="summary">
        <h3 style="margin:8px 0 10px;">Trade Volume by Product</h3>
        <table>
            <thead>
                <tr>
                    <th>Product</th>
                    <th>Total Volume</th>
                    <th>Trade Count</th>
                </tr>
            </thead>
            <tbody id="volumeTableBody"></tbody>
        </table>
    </div>

  <script>
    const figures = {fig_json};
        const summaries = {summaries_json};
    const select = document.getElementById('logSelect');
    const chart = document.getElementById('chart');
        const volumeTableBody = document.getElementById('volumeTableBody');

        function renderSummary(logKey) {{
            const rows = summaries[logKey] || [];
            if (!rows.length) {{
                volumeTableBody.innerHTML = '<tr><td colspan="3" style="text-align:center;">No trades</td></tr>';
                return;
            }}

            volumeTableBody.innerHTML = rows.map((row) => (
                `<tr>
                    <td>${{row.symbol}}</td>
                    <td>${{row.total_volume}}</td>
                    <td>${{row.trade_count}}</td>
                </tr>`
            )).join('');
        }}

    function draw(logKey) {{
      const fig = figures[logKey];
      Plotly.react(chart, fig.data, fig.layout, {{responsive: true}});
            renderSummary(logKey);
    }}

    select.addEventListener('change', (e) => draw(e.target.value));
    draw(select.value || Object.keys(figures)[0]);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def safe_file_name(log_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", log_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build IMC Prosperity log visualizations")
    parser.add_argument("--log", help="Log key to visualize (e.g. 2957 or 3015/3015)")
    parser.add_argument("--output", help="Output HTML path")
    parser.add_argument("--no-open", action="store_true", help="Do not open the generated HTML file")
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

        fig = build_chart(load_log(logs[args.log]), args.log)
        default_name = f"{safe_file_name(args.log)}_viz.html"
        output = Path(args.output) if args.output else LOG_DIR / default_name
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output), include_plotlyjs="cdn", auto_open=not args.no_open)
        print(f"Saved single-log visualization: {output}")
        return

    figures: dict[str, dict] = {}
    summaries: dict[str, list[dict]] = {}
    for log_key, path in logs.items():
        data = load_log(path)
        fig = build_chart(data, log_key)
        figures[log_key] = fig.to_plotly_json()
        summaries[log_key] = build_volume_summary(data)

    output = Path(args.output) if args.output else LOG_DIR / "viz_dashboard.html"
    build_dashboard_html(figures, summaries, output)
    print(f"Saved dashboard: {output}")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
