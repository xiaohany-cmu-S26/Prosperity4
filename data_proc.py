'''keep only the rows for TOMATOES in the prices and trades files'''

import pandas as pd

files = [
    ("TUTORIAL_ROUND_1/prices_round_0_day_-1.csv", "product"),
    ("TUTORIAL_ROUND_1/prices_round_0_day_-2.csv", "product"),
    ("TUTORIAL_ROUND_1/trades_round_0_day_-1.csv", "symbol"),
    ("TUTORIAL_ROUND_1/trades_round_0_day_-2.csv", "symbol"),
]

for filename, col in files:

    df = pd.read_csv(filename, sep=";")

    df = df[df[col] == "TOMATOES"]

    new_name = filename.replace(".csv", "_tomatoes.csv")

    df.to_csv(
        new_name,
        sep=";",
        index=False
    )

    print(f"{filename} → {new_name} ({len(df)} rows)")