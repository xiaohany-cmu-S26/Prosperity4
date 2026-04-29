"""Round 5 CSV loaders. One responsibility: read raw CSVs into tidy DataFrames."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

ROUND5_DIR = Path(__file__).resolve().parents[1] / "ROUND_5"
DAYS = (2, 3, 4)


def load_prices(days: Iterable[int] = DAYS, products: Optional[List[str]] = None) -> pd.DataFrame:
    """Load price/order-book snapshots. Schema: 1 row per (day, timestamp, product).

    Returns columns: day, timestamp, product, bid_price_1..3, bid_volume_1..3,
    ask_price_1..3, ask_volume_1..3, mid_price, profit_and_loss.
    """
    frames = []
    for d in days:
        df = pd.read_csv(ROUND5_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        if products is not None:
            df = df[df["product"].isin(products)]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)


def load_trades(days: Iterable[int] = DAYS, products: Optional[List[str]] = None) -> pd.DataFrame:
    """Load market trades. Round 5 buyer/seller fields are empty (anonymized)."""
    frames = []
    for d in days:
        df = pd.read_csv(ROUND5_DIR / f"trades_round_5_day_{d}.csv", sep=";")
        df.insert(0, "day", d)
        if products is not None:
            df = df[df["symbol"].isin(products)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["day", "timestamp"]).reset_index(drop=True)


SNACKPACKS = [
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_RASPBERRY",
]

ROBOTS = [
    "ROBOT_DISHES",
    "ROBOT_IRONING",
    "ROBOT_LAUNDRY",
    "ROBOT_MOPPING",
    "ROBOT_VACUUMING",
]


def pivot_mid(prices: pd.DataFrame, products: List[str]) -> pd.DataFrame:
    """Wide-format mid price: index=(day, timestamp), columns=product, values=mid_price."""
    sub = prices[prices["product"].isin(products)]
    return sub.pivot_table(index=["day", "timestamp"], columns="product", values="mid_price")
