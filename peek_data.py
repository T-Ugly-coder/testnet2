import pandas as pd
df = pd.read_parquet('data_store/BTCUSDT_1h_backtest_7y.parquet')
print("Columns:", df.columns.tolist())
print("\nFirst row:\n", df.iloc[0])
