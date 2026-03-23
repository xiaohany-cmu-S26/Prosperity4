# import pandas as pd

# df = pd.read_csv("logs/backtest_trading_day_-1.csv", sep=";")


# df_tomatoes = df[df["product"] == "TOMATOES"]

# toxic_up_count = (df_tomatoes["toxic_up"] == True).sum()
# toxic_down_count = (df_tomatoes["toxic_down"] == True).sum()

# print("toxic_up_count:", toxic_up_count)
# print("toxic_down_count:", toxic_down_count)

# print("total_toxic_events:", toxic_up_count + toxic_down_count)

import pandas as pd

# 修改成你的 CSV 路径
file_path = "logs/backtest_trading_day_-1.csv"

# 读取 CSV
df = pd.read_csv(file_path, sep=";")

# 如果只想看 TOMATOES
df = df[df["product"] == "TOMATOES"]

# 转成 numeric（防止字符串）
df["ask_volume_1"] = pd.to_numeric(df["ask_volume_1"], errors="coerce")

print("ask_volume_1 statistics:")
print("------------------------")

print("max:", df["ask_volume_1"].max())
print("min:", df["ask_volume_1"].min())
print("mean:", df["ask_volume_1"].mean())

print("\nTop 10 largest ask_volume_1:")
print(
    df.sort_values("ask_volume_1", ascending=False)
      [["timestamp", "ask_volume_1"]]
      .head(10)
)