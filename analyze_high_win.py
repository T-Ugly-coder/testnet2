import json
import glob
import os
from pathlib import Path

data_dir = Path(r"C:\Users\vaibh\OneDrive\Documents\tryingnew\data_store\high_win_80_v1")
files = glob.glob(str(data_dir / "*.json"))

strategies_1h = []
strategies_4h = []

for f in files:
    with open(f, 'r') as jf:
        try:
            data = json.load(jf)
            # The data is usually a list of strategies or a dict containing a list
            # Based on typical outputs of this tool, it's likely a list of best trials
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'best_trials' in data:
                items = data['best_trials']
            else:
                continue

            for item in items:
                # Identify timeframe from filename or item
                if "tf-1h" in f:
                    strategies_1h.append(item)
                elif "tf-4h" in f:
                    strategies_4h.append(item)
        except Exception as e:
            print(f"Error reading {f}: {e}")

def get_top_5(strategies):
    # Sort by rank_score descending, handling None
    sorted_strats = sorted(
        strategies,
        key=lambda x: x.get("rank_score") if x.get("rank_score") is not None else -float('inf'),
        reverse=True
    )
    return sorted_strats[:5]

top_1h = get_top_5(strategies_1h)
top_4h = get_top_5(strategies_4h)

print("=== TOP 5 1H STRATEGIES ===")
for i, s in enumerate(top_1h, 1):
    print(f"{i}. Score: {s.get('rank_score')} | WinRate: {s.get('win_rate')} | PF: {s.get('profit_factor')} | Expectancy: {s.get('expectancy_r')}")

print("\n=== TOP 5 4H STRATEGIES ===")
for i, s in enumerate(top_4h, 1):
    print(f"{i}. Score: {s.get('rank_score')} | WinRate: {s.get('win_rate')} | PF: {s.get('profit_factor')} | Expectancy: {s.get('expectancy_r')}")
