"""
Read and display IMC Prosperity simulation log (JSON format).
Usage: python read_log.py [log_id]   e.g.  python read_log.py 2957
"""

import json
import sys
from io import StringIO
import pandas as pd

LOG_DIR = "logs"

def load(log_id: str) -> dict:
    path = f"{LOG_DIR}/{log_id}.json"
    with open(path) as f:
        return json.load(f)

def parse_csv(raw: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(raw), sep=";")

def show_summary(data: dict) -> None:
    print("=" * 60)
    print(f"  Round   : {data['round']}")
    print(f"  Status  : {data['status']}")
    print(f"  Profit  : {data['profit']:,.2f}")
    print("  Positions:")
    for p in data.get("positions", []):
        print(f"    {p['symbol']}: {p['quantity']}")
    print("=" * 60)

def show_activities(data: dict, product=None, tail: int = 20) -> None:
    df = parse_csv(data["activitiesLog"])
    if product:
        df = df[df["product"] == product.upper()]
    print(f"\n--- Activities Log (last {tail} rows{', product=' + product if product else ''}) ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.tail(tail).to_string(index=False))

def show_pnl_by_product(data: dict) -> None:
    df = parse_csv(data["activitiesLog"])
    # Last PnL per day per product
    summary = (
        df.sort_values(["day", "timestamp"])
        .groupby(["day", "product"])["profit_and_loss"]
        .last()
        .unstack(fill_value=0)
    )
    print("\n--- Final PnL per product per day ---")
    print(summary.to_string())
    print(f"\n  Total PnL: {summary.values.sum():,.2f}")

def show_graph_log(data: dict, tail: int = 20) -> None:
    df = parse_csv(data["graphLog"])
    print(f"\n--- Graph Log (last {tail} rows) ---")
    print(df.tail(tail).to_string(index=False))

def show_products(data: dict) -> None:
    df = parse_csv(data["activitiesLog"])
    products = sorted(df["product"].unique())
    print(f"\n  Products in log: {', '.join(products)}")

if __name__ == "__main__":
    log_id = sys.argv[1] if len(sys.argv) > 1 else "2957"
    product_filter = sys.argv[2] if len(sys.argv) > 2 else None

    data = load(log_id)
    show_summary(data)
    show_products(data)
    show_pnl_by_product(data)
    show_graph_log(data)
    show_activities(data, product=product_filter)