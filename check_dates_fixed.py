import pandas as pd
import glob
from datetime import datetime

files = glob.glob("data_store/BTCUSDT_*.parquet")
results = []

for f in files:
    try:
        df = pd.read_parquet(f)
        # Use ts_ms as the time column
        if 'ts_ms' in df.columns:
            start_ms = df['ts_ms'].min()
            end_ms = df['ts_ms'].max()

            start_date = datetime.fromtimestamp(start_ms / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
            end_date = datetime.fromtimestamp(end_ms / 1000.0).strftime('%Y-%m-%d %H:%M:%S')

            results.append(f"{f}: {start_date} to {end_date}")
        else:
            results.append(f"{f}: No 'ts_ms' column found")
    except Exception as e:
        results.append(f"{f}: Error {e}")

print("\n".join(results))
