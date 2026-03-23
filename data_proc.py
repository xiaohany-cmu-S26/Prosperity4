'''keep only the rows for TOMATOES in the prices and trades files'''

import pandas as pd

files = [
    ("TUTORIAL_ROUND_1/prices_round_0_day_-1.csv", "product"),
    ("TUTORIAL_ROUND_1/prices_round_0_day_-2.csv", "product"),
    ("TUTORIAL_ROUND_1/trades_round_0_day_-1.csv", "symbol"),
    ("TUTORIAL_ROUND_1/trades_round_0_day_-2.csv", "symbol"),
]

'''
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
'''

'''visualize original data'''
import pandas as pd
import matplotlib.pyplot as plt

# 修改为你的真实路径
file_path = "TUTORIAL_ROUND_1/trades_round_0_day_-2_tomatoes.csv"

df = pd.read_csv(file_path, sep=";")

# 如果文件还是包含 EMERALDS，需要先过滤
if "symbol" in df.columns:
    df = df[df["symbol"] == "TOMATOES"]

# 按时间排序（重要）
df = df.sort_values("timestamp")

plt.figure()
plt.plot(df["timestamp"], df["price"])
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.title("TOMATOES Trades Time Series (Day -2)")
plt.show()