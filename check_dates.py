import pandas as pd
import glob
import os

files = glob.glob("data_store/BTCUSDT_*.parquet")
results = []

for f in files:
    try:
        df = pd.read_parquet(f)
        # Find the time column (usually contains 'time', 'date', 'timestamp')
        time_col = next((col for col in df.columns if any(x in col.lower() for x in ['time', 'date', 'timestamp'])), None)

        if time_col:
            start = df[time_col].min()
            end = df[time_col].max()
            results.append(f"{f}: {start} to {end}")
        else:
            results.append(f"{f}: No time column found")
    except Exception as e:
        results.append(f"{f}: Error {e}")

print("\n".join(results))
