import json
import glob
import os
from pathlib import Path

def extract_best_strategies(data_dir):
    all_strategies = []
    files = glob.glob(str(Path(data_dir) / "*.json"))

    for f in files:
        with open(f, 'r') as jf:
            try:
                data = json.load(jf)
                # Look for the best trial in the file
                trial = data.get('best_trial', {})
                if not trial:
                    # Some versions might store it as 'best_params' and 'best_score'
                    # but without the full trial stats. We need the stats.
                    continue

                # Add metadata from filename
                trial['filename'] = f
                all_strategies.append(trial)
            except Exception as e:
                print(f"Error reading {f}: {e}")

    return all_strategies

def main():
    data_dir = r"C:\Users\vaibh\OneDrive\Documents\tryingnew\data_store\high_win_80_v1"
    strategies = extract_best_strategies(data_dir)

    # Separate by timeframe
    tf_1h = [s for s in strategies if "tf-1h" in s['filename']]
    tf_4h = [s for s in strategies if "tf-4h" in s['filename']]

    def get_top_5(strats):
        # Sort by win_rate primarily, then profit_factor as a tie-breaker
        # This shows the "highest win rate" ones regardless of filters
        return sorted(
            strats,
            key=lambda x: (x.get('win_rate', 0), x.get('profit_factor', 0)),
            reverse=True
        )[:5]

    top_1h = get_top_5(tf_1h)
    top_4h = get_top_5(tf_4h)

    print("=== BEST FOUND 1H (Regardless of Filters) ===")
    if not top_1h:
        print("No trial data found in 1h files.")
    else:
        print(f"{'Rank':<5} | {'WinRate':<10} | {'PF':<8} | {'Exp':<8} | {'Seed':<6}")
        for i, s in enumerate(top_1h, 1):
            wr = s.get('win_rate', 0)
            pf = s.get('profit_factor', 0)
            exp = s.get('expectancy_r', 0)
            seed = s.get('seed', 'N/A')
            print(f"{i:<5} | {wr:<10.2%} | {pf:<8.2f} | {exp:<8.2f} | {seed:<6}")

    print("\n=== BEST FOUND 4H (Regardless of Filters) ===")
    if not top_4h:
        print("No trial data found in 4h files.")
    else:
        print(f"{'Rank':<5} | {'WinRate':<10} | {'PF':<8} | {'Exp':<8} | {'Seed':<6}")
        for i, s in enumerate(top_4h, 1):
            wr = s.get('win_rate', 0)
            pf = s.get('profit_factor', 0)
            exp = s.get('expectancy_r', 0)
            seed = s.get('seed', 'N/A')
            print(f"{i:<5} | {wr:<10.2%} | {pf:<8.2f} | {exp:<8.2f} | {seed:<6}")

if __name__ == "__main__":
    main()
