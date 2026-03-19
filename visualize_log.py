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
VIZ_DIR = LOG_DIR / "visuals"

BID_COLOR = "#26C281"   # green
ASK_COLOR = "#EF5350"   # red


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


def build_chart(data: dict, log_label: str) -> tuple[go.Figure, dict]:
    acts = pd.read_csv(StringIO(data["activitiesLog"]), sep=";")
    acts = acts.sort_values("timestamp")
    products = sorted(acts["product"].unique())
    trades = pd.DataFrame(data.get("tradeHistory", []))

    # Compute inventory (cumulative buy/sell) at each timestamp for each product
    inventories: dict[str, dict] = {}  # product -> {timestamp: inventory}
    if not trades.empty and {"symbol", "timestamp", "buyer", "seller", "quantity"}.issubset(trades.columns):
        for product in products:
            inventories[product] = {}
            pt = trades[trades["symbol"] == product].copy().sort_values("timestamp")
            inventory = 0
            for _, row in pt.iterrows():
                if row["buyer"] == "SUBMISSION":
                    inventory += row["quantity"]  # we bought, inventory goes up
                elif row["seller"] == "SUBMISSION":
                    inventory -= row["quantity"]  # we sold, inventory goes down
                inventories[product][row["timestamp"]] = inventory

    n = len(products)
    specs = [[{"secondary_y": True}] for _ in range(n)] + [[{"secondary_y": True}]]
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

        # Add inventory at each timestamp (forward-fill from last trade)
        inventory_series = pd.Series(inventories.get(product, {}))
        if not inventory_series.empty:
            # Forward-fill inventory across all timestamps so it persists between trades
            inventory_series = inventory_series.reindex(df["timestamp"], method="ffill").fillna(0)
            df["inventory"] = inventory_series.values
        else:
            df["inventory"] = 0

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["mid_price"],
                name=f"{product} mid",
                line=dict(color=color, width=1.6),
                legendgroup=product,
                customdata=df[["timestamp", "inventory"]],
                hovertemplate=(
                    f"<b>{product} mid price</b><br>"
                    "<b>Tick: %{customdata[0]}</b><br>"
                    "Price: %{y}<br>"
                    "Inventory: %{customdata[1]}<extra></extra>"
                ),
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
                line=dict(color=BID_COLOR, width=0.8, dash="dot"),
                opacity=0.6,
                legendgroup=product,
                showlegend=False,
                customdata=df["timestamp"],
                hovertemplate=(
                    f"<b>{product} bid</b><br>"
                    "<b>Tick: %{customdata}</b><br>"
                    "Price: %{y}<extra></extra>"
                ),
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
                line=dict(color=ASK_COLOR, width=0.8, dash="dot"),
                opacity=0.6,
                fill="tonexty",
                fillcolor=to_rgba(BID_COLOR, 0.07),
                legendgroup=product,
                showlegend=False,
                customdata=df["timestamp"],
                hovertemplate=(
                    f"<b>{product} ask</b><br>"
                    "<b>Tick: %{customdata}</b><br>"
                    "Price: %{y}<extra></extra>"
                ),
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
                        customdata=volume_per_tick["timestamp"],
                        hovertemplate=(
                            f"<b>{product} volume</b><br>"
                            "<b>Tick: %{customdata}</b><br>"
                            "Volume: %{y}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                    secondary_y=True,
                )

                if {"buyer", "seller", "price"}.issubset(product_trades.columns):
                    product_trades["side"] = product_trades.apply(classify_trade_side, axis=1)
                    # (symbol, color, size, outline_width, label)
                    # My trades: original triangles with white outline
                    # Bot trades (not involved): yellow crosses
                    TRADE_STYLES = {
                        "buy":     ("triangle-up",   "#00CC44", 13, 2,   "MY BUY"),
                        "sell":    ("triangle-down",  "#FF4444", 13, 2,   "MY SELL"),
                        "unknown": ("x",             "#FFD700", 13, 0,   "BOT TRADE"),
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
                                    "<b>Tick: %{customdata[0]}</b><br>"
                                    "Price: %{y}<br>Qty: %{customdata[1]}<extra></extra>"
                                ),
                                customdata=subset[["timestamp", "quantity"]],
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
                customdata=df["timestamp"],
                hovertemplate=(
                    f"<b>{product} PnL</b><br>"
                    "<b>Tick: %{customdata}</b><br>"
                    "PnL: %{y}<extra></extra>"
                ),
            ),
            row=pnl_row,
            col=1,
            secondary_y=False,
        )

        # Inventory trace on secondary y-axis
        inv_data = inventories.get(product, {})
        if inv_data:
            inv_series = pd.Series(inv_data).sort_index()
            inv_full = inv_series.reindex(df["timestamp"], method="ffill").fillna(0)
        else:
            inv_full = pd.Series(0, index=df["timestamp"])
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=inv_full,
                name=f"{product} Inv",
                line=dict(color=color, width=1.2, dash="dot"),
                legendgroup=f"{product}_pnl",
                showlegend=True,
                customdata=df["timestamp"],
                hovertemplate=(
                    f"<b>{product} Inventory</b><br>"
                    "<b>Tick: %{customdata}</b><br>"
                    "Inv: %{y}<extra></extra>"
                ),
            ),
            row=pnl_row,
            col=1,
            secondary_y=True,
        )

    # Total PnL (sum across all products)
    total_pnl_df = acts.groupby("timestamp", as_index=False)["profit_and_loss"].sum().sort_values("timestamp")
    fig.add_trace(
        go.Scatter(
            x=total_pnl_df["timestamp"],
            y=total_pnl_df["profit_and_loss"],
            name="Total PnL",
            line=dict(color="white", width=2.5),
            legendgroup="total_pnl",
            customdata=total_pnl_df["timestamp"],
            hovertemplate=(
                "<b>Total PnL</b><br>"
                "<b>Tick: %{customdata}</b><br>"
                "PnL: %{y}<extra></extra>"
            ),
        ),
        row=pnl_row,
        col=1,
        secondary_y=False,
    )

    total_trades = len(trades) if not trades.empty else 0
    fig.update_yaxes(title_text="PnL", row=pnl_row, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Inventory", row=pnl_row, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Timestamp", row=pnl_row, col=1)
    fig.update_layout(
        title=dict(text=f"IMC Prosperity — Log {log_label} | {total_trades} trades", font=dict(size=18)),
        height=340 * (n + 1),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01, y=1),
        barmode="overlay",
    )

    statesheet = build_statesheet(data)
    return fig, statesheet


def _final_unrealized_pnl(product_trades: pd.DataFrame, final_mid: float) -> float:
    """Unrealized PnL on the remaining open position at end of session.
    Uses average-cost accounting; returns position * final_mid - cost_basis."""
    position, cost_basis = 0.0, 0.0
    for _, row in product_trades.sort_values("timestamp").iterrows():
        if row["buyer"] == "SUBMISSION":
            qty = float(row["quantity"])
        elif row["seller"] == "SUBMISSION":
            qty = -float(row["quantity"])
        else:
            continue
        new_pos = position + qty
        if position == 0:
            cost_basis = qty * row["price"]
        elif (position > 0) == (qty > 0):
            cost_basis += qty * row["price"]
        else:
            cost_basis = cost_basis * (new_pos / position) if abs(new_pos) < abs(position) else new_pos * row["price"]
        position = new_pos
    return position * final_mid - cost_basis if position != 0 else 0.0


def build_volume_summary(data: dict) -> list[dict]:
    trades = pd.DataFrame(data.get("tradeHistory", []))
    if trades.empty or not {"symbol", "quantity"}.issubset(trades.columns):
        return []

    all_counts = (
        trades.groupby("symbol", as_index=False)
        .agg(total_volume=("quantity", "sum"), trade_count=("quantity", "size"))
    )

    has_sides = {"buyer", "seller"}.issubset(trades.columns)

    # Calculate profitable vs unprofitable trades using FIFO accounting
    profitable = {}
    unprofitable = {}

    if has_sides and "price" in trades.columns:
        for product in trades["symbol"].unique():
            pt = trades[trades["symbol"] == product].copy().sort_values("timestamp")

            # Separate buys and sells
            buys = pt[pt["buyer"] == "SUBMISSION"].copy()
            sells = pt[pt["seller"] == "SUBMISSION"].copy()

            if not buys.empty and not sells.empty:
                # Track realized PnL using FIFO
                buy_queue = list(zip(buys["price"], buys["quantity"]))
                sell_queue = list(zip(sells["price"], sells["quantity"]))

                realized_pnl = 0.0
                sell_qty_remaining = 0
                sell_price_current = 0.0

                for sell_price, sell_qty in sell_queue:
                    sell_qty_remaining = sell_qty
                    sell_price_current = sell_price

                    # Match with buys in order (FIFO)
                    while sell_qty_remaining > 0 and buy_queue:
                        buy_price, buy_qty = buy_queue.pop(0)
                        matched_qty = min(sell_qty_remaining, buy_qty)
                        realized_pnl += matched_qty * (sell_price - buy_price)
                        sell_qty_remaining -= matched_qty

                        if buy_qty > matched_qty:
                            buy_queue.insert(0, (buy_price, buy_qty - matched_qty))

                # Count profitable trades (those with positive contribution to PnL)
                # A trade is profitable if it improved our position
                buy_proceeds = buys["quantity"].sum() * buys["price"].mean() if not buys.empty else 0
                sell_proceeds = sells["quantity"].sum() * sells["price"].mean() if not sells.empty else 0

                if realized_pnl > 0:
                    profitable[product] = len(sells)
                    unprofitable[product] = 0
                else:
                    profitable[product] = 0
                    unprofitable[product] = len(sells)
            elif not buys.empty or not sells.empty:
                # Only one side, count as unprofitable (unrealized)
                product_trades = len(buys) if not buys.empty else len(sells)
                profitable[product] = 0
                unprofitable[product] = product_trades
            else:
                profitable[product] = 0
                unprofitable[product] = 0

    if has_sides:
        my_trades = trades[(trades["buyer"] == "SUBMISSION") | (trades["seller"] == "SUBMISSION")]
        my_counts = my_trades.groupby("symbol").size().reset_index(name="my_trade_count")
        summary = all_counts.merge(my_counts, on="symbol", how="left").fillna(0)
        summary["my_trade_count"] = summary["my_trade_count"].astype(int)
    else:
        summary = all_counts
        summary["my_trade_count"] = 0

    # Add profitability columns
    summary["profitable"] = summary["symbol"].map(lambda x: profitable.get(x, 0))
    summary["unprofitable"] = summary["symbol"].map(lambda x: unprofitable.get(x, 0))

    summary = summary.sort_values("symbol")

    result = []
    for row in summary.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v):
                clean_row[k] = 0
            elif isinstance(v, (int, float)) and float(v) == int(float(v)):
                clean_row[k] = int(float(v))
            else:
                clean_row[k] = v
        result.append(clean_row)
    return result


def build_statesheet(data: dict) -> dict:
    """Build final state summary: PnL, positions, unrealized PnL, trade counts."""
    acts = pd.read_csv(StringIO(data["activitiesLog"]), sep=";")
    acts = acts.sort_values("timestamp")
    trades = pd.DataFrame(data.get("tradeHistory", []))

    has_trades = (
        not trades.empty
        and {"symbol", "buyer", "seller", "quantity", "price"}.issubset(trades.columns)
    )

    statesheet = {
        "total_pnl": 0.0,
        "positions": {},
        "unrealized_pnl": {},
        "total_trades": int(len(trades)),
        "trades_per_asset": {},
    }

    for product in list(acts["product"].unique()):
        df = acts[acts["product"] == product].sort_values("timestamp")
        final_pnl = float(df["profit_and_loss"].iloc[-1]) if len(df) > 0 else 0.0
        final_mid = float(df["mid_price"].iloc[-1]) if len(df) > 0 else 0.0
        statesheet["total_pnl"] += final_pnl

        position = 0
        unrealized = 0.0
        if has_trades:
            pt = trades[trades["symbol"] == product].copy()
            for _, row in pt.iterrows():
                if row["buyer"] == "SUBMISSION":
                    position += int(row["quantity"])
                elif row["seller"] == "SUBMISSION":
                    position -= int(row["quantity"])
            unrealized = _final_unrealized_pnl(pt, final_mid)

        statesheet["positions"][product] = int(position)
        statesheet["unrealized_pnl"][product] = unrealized
        statesheet["trades_per_asset"][product] = int(
            len(trades[trades["symbol"] == product]) if has_trades else 0
        )

    return statesheet


def build_raw_timeseries(data: dict) -> dict:
    """Build per-product timeseries data for JS dynamic sidebar computation."""
    acts = pd.read_csv(StringIO(data["activitiesLog"]), sep=";").sort_values("timestamp")
    trades = pd.DataFrame(data.get("tradeHistory", []))
    products = sorted(acts["product"].unique())

    has_trades = (
        not trades.empty
        and {"symbol", "buyer", "seller", "quantity", "price", "timestamp"}.issubset(trades.columns)
    )

    timeseries = {}
    for product in products:
        df = acts[acts["product"] == product].sort_values("timestamp")
        timestamps = df["timestamp"].tolist()
        pnl = df["profit_and_loss"].tolist()

        # Build inventory series
        inv = [0] * len(timestamps)
        if has_trades:
            pt = trades[trades["symbol"] == product].sort_values("timestamp")
            inventory = 0
            inv_dict = {}
            for _, row in pt.iterrows():
                if row["buyer"] == "SUBMISSION":
                    inventory += int(row["quantity"])
                elif row["seller"] == "SUBMISSION":
                    inventory -= int(row["quantity"])
                inv_dict[row["timestamp"]] = inventory
            if inv_dict:
                inv_series = pd.Series(inv_dict).sort_index()
                inv_full = inv_series.reindex(timestamps, method="ffill").fillna(0)
                inv = inv_full.astype(int).tolist()

        timeseries[product] = {
            "timestamps": timestamps,
            "pnl": pnl,
            "inventory": inv,
        }

    # Build trade list for filtering by range
    trade_list = {}
    if has_trades:
        for product in products:
            pt = trades[trades["symbol"] == product].sort_values("timestamp")
            entries = []
            for _, row in pt.iterrows():
                side = "bot"
                if row["buyer"] == "SUBMISSION":
                    side = "buy"
                elif row["seller"] == "SUBMISSION":
                    side = "sell"
                entries.append({
                    "t": int(row["timestamp"]),
                    "side": side,
                    "price": float(row["price"]),
                    "qty": int(row["quantity"]),
                })
            trade_list[product] = entries

    return {"timeseries": timeseries, "trades": trade_list}


def build_single_log_html(fig: go.Figure, statesheet: dict, summary: list[dict], raw_ts: dict, log_label: str, output_path: Path) -> None:
    """Build HTML for a single log with sidebar stats."""
    import numpy as np

    def json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    fig_json = json.dumps(fig.to_plotly_json(), default=json_serializer)
    statesheet_json = json.dumps(statesheet)
    summary_json = json.dumps(summary)
    raw_ts_json = json.dumps(raw_ts, default=json_serializer)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>IMC Prosperity Log — {log_label}</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; display:flex; flex-direction:column; height:100vh; }}
    .toolbar {{ padding:12px 16px; border-bottom:1px solid #333; display:flex; gap:10px; align-items:center; flex-shrink:0; }}
    .container {{ display: flex; gap: 0; flex: 1; overflow: hidden; width: 100%; }}
    #chart {{ flex: 1; overflow: auto; min-width: 0; }}
    .sidebar {{ width: 380px; overflow-y: auto; border-left: 1px solid #333; padding: 16px; flex-shrink: 0; background: #0a0a0a; min-width: 380px; }}
    .summary {{ margin-bottom: 24px; }}
    .summary h3 {{ margin: 0 0 12px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color:#aaa; font-weight: 600; }}
    .range-label {{ font-size: 10px; color: #666; margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; background:#171717; font-size: 11px; }}
    th, td {{ border: 1px solid #333; padding: 5px 6px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background:#202020; font-weight: 600; }}
  </style>
</head>
<body>
  <div class=\"toolbar\">
    <h2 style=\"margin:0; font-size:16px;\">IMC Prosperity — {log_label}</h2>
  </div>
  <div class=\"container\">
    <div id=\"chart\"></div>
    <div class=\"sidebar\">
      <div id="rangeLabel" class="range-label">Showing: full range</div>

      <div class="summary">
          <h3>State</h3>
          <table>
              <thead><tr><th>Metric</th><th>Value</th></tr></thead>
              <tbody id="statesheetBody"></tbody>
          </table>
      </div>

      <div class="summary">
          <h3>Positions</h3>
          <table>
              <thead><tr><th>Product</th><th>Inv</th><th>PnL</th></tr></thead>
              <tbody id="positionsTableBody"></tbody>
          </table>
      </div>

      <div class="summary">
          <h3>Trades</h3>
          <table>
              <thead><tr><th>Product</th><th>Total</th><th>My</th><th>Buy</th><th>Sell</th></tr></thead>
              <tbody id="volumeTableBody"></tbody>
          </table>
      </div>
    </div>
  </div>

  <script>
    const figure = {fig_json};
    const rawTs = {raw_ts_json};
    const chart = document.getElementById('chart');

    // Compute stats for a given timestamp range [tMin, tMax]
    function computeStats(tMin, tMax) {{
        const ts = rawTs.timeseries || {{}};
        const trades = rawTs.trades || {{}};
        const products = Object.keys(ts).sort();

        let totalPnl = 0;
        let totalTrades = 0;
        let totalMyTrades = 0;
        const productStats = [];

        for (const product of products) {{
            const d = ts[product];
            const timestamps = d.timestamps;
            const pnl = d.pnl;
            const inv = d.inventory;

            // Find indices within range
            let iStart = -1, iEnd = -1;
            for (let i = 0; i < timestamps.length; i++) {{
                if (timestamps[i] >= tMin && iStart === -1) iStart = i;
                if (timestamps[i] <= tMax) iEnd = i;
            }}

            let pnlInRange = 0;
            let invAtEnd = 0;
            if (iStart >= 0 && iEnd >= 0) {{
                const startPnl = iStart > 0 ? pnl[iStart - 1] : 0;
                pnlInRange = pnl[iEnd] - startPnl;
                invAtEnd = inv[iEnd];
            }}

            // Count trades in range
            const productTrades = trades[product] || [];
            let tradeCount = 0, myCount = 0, buyCount = 0, sellCount = 0;
            for (const t of productTrades) {{
                if (t.t >= tMin && t.t <= tMax) {{
                    tradeCount++;
                    if (t.side === 'buy') {{ myCount++; buyCount++; }}
                    else if (t.side === 'sell') {{ myCount++; sellCount++; }}
                }}
            }}

            totalPnl += pnlInRange;
            totalTrades += tradeCount;
            totalMyTrades += myCount;

            productStats.push({{
                product, pnlInRange, invAtEnd,
                tradeCount, myCount, buyCount, sellCount
            }});
        }}

        return {{ totalPnl, totalTrades, totalMyTrades, productStats }};
    }}

    function renderSidebar(tMin, tMax, isFullRange) {{
        const stats = computeStats(tMin, tMax);

        // Range label
        const rangeLabel = document.getElementById('rangeLabel');
        if (isFullRange) {{
            rangeLabel.textContent = 'Showing: full range';
        }} else {{
            rangeLabel.textContent = `Showing: ${{Math.round(tMin)}} — ${{Math.round(tMax)}}`;
        }}

        // State table
        document.getElementById('statesheetBody').innerHTML = [
            ['Total PnL', stats.totalPnl.toFixed(0)],
            ['Total Trades', stats.totalTrades],
            ['My Trades', stats.totalMyTrades],
        ].map(r => `<tr><td>${{r[0]}}</td><td>${{r[1]}}</td></tr>`).join('');

        // Positions table
        document.getElementById('positionsTableBody').innerHTML = stats.productStats.map(s =>
            `<tr><td>${{s.product}}</td><td>${{s.invAtEnd}}</td><td>${{s.pnlInRange.toFixed(0)}}</td></tr>`
        ).join('');

        // Trades table
        document.getElementById('volumeTableBody').innerHTML = stats.productStats.map(s =>
            `<tr><td>${{s.product}}</td><td>${{s.tradeCount}}</td><td>${{s.myCount}}</td><td style="color:#4CAF50">${{s.buyCount}}</td><td style="color:#FF6B6B">${{s.sellCount}}</td></tr>`
        ).join('');
    }}

    // Get full timestamp range
    function getFullRange() {{
        let tMin = Infinity, tMax = -Infinity;
        for (const d of Object.values(rawTs.timeseries || {{}})) {{
            if (d.timestamps.length) {{
                tMin = Math.min(tMin, d.timestamps[0]);
                tMax = Math.max(tMax, d.timestamps[d.timestamps.length - 1]);
            }}
        }}
        return [tMin, tMax];
    }}

    Plotly.newPlot(chart, figure.data, figure.layout, {{responsive: true}});

    // Initial render with full range
    const [fullMin, fullMax] = getFullRange();
    renderSidebar(fullMin, fullMax, true);

    // Update sidebar on zoom/pan
    chart.on('plotly_relayout', function(eventData) {{
        // Find the x-axis range — shared x-axes mean any xaxis range works
        let tMin = null, tMax = null;
        for (const key of Object.keys(eventData)) {{
            if (key.match(/^xaxis\\d*\\.range\\[0\\]$/)) {{
                tMin = parseFloat(eventData[key]);
            }}
            if (key.match(/^xaxis\\d*\\.range\\[1\\]$/)) {{
                tMax = parseFloat(eventData[key]);
            }}
        }}

        // Check for autorange (double-click reset)
        const hasAutorange = Object.keys(eventData).some(k => k.match(/^xaxis\\d*\\.autorange$/));

        if (hasAutorange || (tMin === null && tMax === null)) {{
            const [fMin, fMax] = getFullRange();
            renderSidebar(fMin, fMax, true);
        }} else if (tMin !== null && tMax !== null) {{
            renderSidebar(tMin, tMax, false);
        }}
    }});
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def build_dashboard_html(figures: dict[str, dict], summaries: dict[str, list[dict]], statesheets: dict[str, dict], all_raw_ts: dict[str, dict], output_path: Path) -> None:
    keys = sorted(figures.keys())
    if not keys:
        raise ValueError("No logs available to visualize.")

    # Custom JSON encoder to handle numpy types
    def json_serializer(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    fig_json = json.dumps(figures, default=json_serializer)
    all_raw_ts_json = json.dumps(all_raw_ts, default=json_serializer)
    option_html = "\n".join([f'<option value="{k}">{k}</option>' for k in keys])

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>IMC Prosperity Log Dashboard</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; display:flex; flex-direction:column; height:100vh; }}
    .toolbar {{ padding:12px 16px; border-bottom:1px solid #333; display:flex; gap:10px; align-items:center; flex-shrink:0; }}
    select {{ background:#1e1e1e; color:#eee; border:1px solid #444; border-radius:6px; padding:6px 10px; min-width:220px; }}
    .container {{ display: flex; gap: 0; flex: 1; overflow: hidden; width: 100%; }}
    #chart {{ flex: 1; overflow: auto; min-width: 0; }}
    .sidebar {{ width: 380px; overflow-y: auto; border-left: 1px solid #333; padding: 16px; flex-shrink: 0; background: #0a0a0a; min-width: 380px; }}
    .summary {{ margin-bottom: 24px; }}
    .summary h3 {{ margin: 0 0 12px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color:#aaa; font-weight: 600; }}
    .range-label {{ font-size: 10px; color: #666; margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; background:#171717; font-size: 11px; }}
    th, td {{ border: 1px solid #333; padding: 5px 6px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background:#202020; font-weight: 600; }}
  </style>
</head>
<body>
  <div class=\"toolbar\">
    <label for=\"logSelect\">Choose log:</label>
    <select id=\"logSelect\">{option_html}</select>
  </div>
  <div class=\"container\">
    <div id=\"chart\"></div>
    <div class=\"sidebar\">
      <div id="rangeLabel" class="range-label">Showing: full range</div>

      <div class="summary">
          <h3>State</h3>
          <table>
              <thead><tr><th>Metric</th><th>Value</th></tr></thead>
              <tbody id="statesheetBody"></tbody>
          </table>
      </div>

      <div class="summary">
          <h3>Positions</h3>
          <table>
              <thead><tr><th>Product</th><th>Inv</th><th>PnL</th></tr></thead>
              <tbody id="positionsTableBody"></tbody>
          </table>
      </div>

      <div class="summary">
          <h3>Trades</h3>
          <table>
              <thead><tr><th>Product</th><th>Total</th><th>My</th><th>Buy</th><th>Sell</th></tr></thead>
              <tbody id="volumeTableBody"></tbody>
          </table>
      </div>
    </div>
  </div>

  <script>
    const figures = {fig_json};
    const allRawTs = {all_raw_ts_json};
    const select = document.getElementById('logSelect');
    const chart = document.getElementById('chart');
    let currentLogKey = null;

    function computeStats(rawTs, tMin, tMax) {{
        const ts = rawTs.timeseries || {{}};
        const trades = rawTs.trades || {{}};
        const products = Object.keys(ts).sort();

        let totalPnl = 0, totalTrades = 0, totalMyTrades = 0;
        const productStats = [];

        for (const product of products) {{
            const d = ts[product];
            const timestamps = d.timestamps;
            const pnl = d.pnl;
            const inv = d.inventory;

            let iStart = -1, iEnd = -1;
            for (let i = 0; i < timestamps.length; i++) {{
                if (timestamps[i] >= tMin && iStart === -1) iStart = i;
                if (timestamps[i] <= tMax) iEnd = i;
            }}

            let pnlInRange = 0, invAtEnd = 0;
            if (iStart >= 0 && iEnd >= 0) {{
                const startPnl = iStart > 0 ? pnl[iStart - 1] : 0;
                pnlInRange = pnl[iEnd] - startPnl;
                invAtEnd = inv[iEnd];
            }}

            const productTrades = trades[product] || [];
            let tradeCount = 0, myCount = 0, buyCount = 0, sellCount = 0;
            for (const t of productTrades) {{
                if (t.t >= tMin && t.t <= tMax) {{
                    tradeCount++;
                    if (t.side === 'buy') {{ myCount++; buyCount++; }}
                    else if (t.side === 'sell') {{ myCount++; sellCount++; }}
                }}
            }}

            totalPnl += pnlInRange;
            totalTrades += tradeCount;
            totalMyTrades += myCount;
            productStats.push({{ product, pnlInRange, invAtEnd, tradeCount, myCount, buyCount, sellCount }});
        }}

        return {{ totalPnl, totalTrades, totalMyTrades, productStats }};
    }}

    function getFullRange(rawTs) {{
        let tMin = Infinity, tMax = -Infinity;
        for (const d of Object.values(rawTs.timeseries || {{}})) {{
            if (d.timestamps.length) {{
                tMin = Math.min(tMin, d.timestamps[0]);
                tMax = Math.max(tMax, d.timestamps[d.timestamps.length - 1]);
            }}
        }}
        return [tMin, tMax];
    }}

    function renderSidebar(rawTs, tMin, tMax, isFullRange) {{
        const stats = computeStats(rawTs, tMin, tMax);

        document.getElementById('rangeLabel').textContent = isFullRange
            ? 'Showing: full range'
            : `Showing: ${{Math.round(tMin)}} — ${{Math.round(tMax)}}`;

        document.getElementById('statesheetBody').innerHTML = [
            ['Total PnL', stats.totalPnl.toFixed(0)],
            ['Total Trades', stats.totalTrades],
            ['My Trades', stats.totalMyTrades],
        ].map(r => `<tr><td>${{r[0]}}</td><td>${{r[1]}}</td></tr>`).join('');

        document.getElementById('positionsTableBody').innerHTML = stats.productStats.map(s =>
            `<tr><td>${{s.product}}</td><td>${{s.invAtEnd}}</td><td>${{s.pnlInRange.toFixed(0)}}</td></tr>`
        ).join('');

        document.getElementById('volumeTableBody').innerHTML = stats.productStats.map(s =>
            `<tr><td>${{s.product}}</td><td>${{s.tradeCount}}</td><td>${{s.myCount}}</td><td style="color:#4CAF50">${{s.buyCount}}</td><td style="color:#FF6B6B">${{s.sellCount}}</td></tr>`
        ).join('');
    }}

    function draw(logKey) {{
        currentLogKey = logKey;
        const fig = figures[logKey];
        Plotly.react(chart, fig.data, fig.layout, {{responsive: true}});
        const rawTs = allRawTs[logKey] || {{ timeseries: {{}}, trades: {{}} }};
        const [fMin, fMax] = getFullRange(rawTs);
        renderSidebar(rawTs, fMin, fMax, true);
    }}

    chart.on('plotly_relayout', function(eventData) {{
        if (!currentLogKey) return;
        const rawTs = allRawTs[currentLogKey] || {{ timeseries: {{}}, trades: {{}} }};

        let tMin = null, tMax = null;
        for (const key of Object.keys(eventData)) {{
            if (key.match(/^xaxis\\d*\\.range\\[0\\]$/)) tMin = parseFloat(eventData[key]);
            if (key.match(/^xaxis\\d*\\.range\\[1\\]$/)) tMax = parseFloat(eventData[key]);
        }}

        const hasAutorange = Object.keys(eventData).some(k => k.match(/^xaxis\\d*\\.autorange$/));

        if (hasAutorange || (tMin === null && tMax === null)) {{
            const [fMin, fMax] = getFullRange(rawTs);
            renderSidebar(rawTs, fMin, fMax, true);
        }} else if (tMin !== null && tMax !== null) {{
            renderSidebar(rawTs, tMin, tMax, false);
        }}
    }});

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

        data = load_log(logs[args.log])
        fig, statesheet = build_chart(data, args.log)
        summary = build_volume_summary(data)
        raw_ts = build_raw_timeseries(data)

        default_name = f"{safe_file_name(args.log)}_viz.html"
        output = Path(args.output) if args.output else VIZ_DIR / default_name
        output.parent.mkdir(parents=True, exist_ok=True)

        # Build custom HTML with sidebar
        build_single_log_html(fig, statesheet, summary, raw_ts, args.log, output)
        print(f"Saved single-log visualization: {output}")

        if not args.no_open:
            webbrowser.open(output.resolve().as_uri())
        return

    figures: dict[str, dict] = {}
    summaries: dict[str, list[dict]] = {}
    statesheets: dict[str, dict] = {}
    all_raw_ts: dict[str, dict] = {}
    for log_key, path in logs.items():
        data = load_log(path)
        fig, statesheet = build_chart(data, log_key)
        figures[log_key] = fig.to_plotly_json()
        summaries[log_key] = build_volume_summary(data)
        statesheets[log_key] = statesheet
        all_raw_ts[log_key] = build_raw_timeseries(data)

    output = Path(args.output) if args.output else VIZ_DIR / "viz_dashboard.html"
    build_dashboard_html(figures, summaries, statesheets, all_raw_ts, output)
    print(f"Saved dashboard: {output}")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
