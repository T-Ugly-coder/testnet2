import json
import glob
import os
from pathlib import Path
from datetime import datetime

# Configuration matching your run
WIN_RATE_TARGET = 0.80
WIN_RATE_WEIGHT = 25
MIN_FORWARD_WIN_RATE = 0.75
MIN_HOLDOUT_WIN_RATE = 0.75
MIN_FORWARD_PF = 1.10
MIN_HOLDOUT_PF = 1.15
MIN_FORWARD_EXPECTANCY = 0.05
MIN_HOLDOUT_EXPECTANCY = 0.05
MAX_TRAIN_HOLDOUT_PF_RATIO = 1.5
MAX_TRAIN_HOLDOUT_SCORE_GAP = 3.0

def calculate_rank_score(trial):
    """
    Replicates the scoring logic from auto_algo_finder.py
    """
    # Basic metrics
    win_rate = trial.get('win_rate', 0)
    pf = trial.get('profit_factor', 0)
    exp = trial.get('expectancy_r', 0)

    # Hard Guardrails
    if win_rate < MIN_HOLDOUT_WIN_RATE: return None
    if pf < MIN_HOLDOUT_PF: return None
    if exp < MIN_HOLDOUT_EXPECTANCY: return None

    # Train vs Holdout Gap
    train_pf = trial.get('train_pf', 0)
    holdout_pf = trial.get('holdout_pf', 0)
    if holdout_pf > 0 and (train_pf / holdout_pf) > MAX_TRAIN_HOLDOUT_PF_RATIO:
        return None

    # Score calculation
    # (Simplified version of the ranking logic from the script)
    score = (win_rate * WIN_RATE_WEIGHT) + (pf * 2.0) + (exp * 5.0)
    return score

def extract_best_strategies(data_dir):
    all_strategies = []
    files = glob.glob(str(Path(data_dir) / "*.json"))

    for f in files:
        with open(f, 'r') as jf:
            try:
                data = json.load(jf)
                # In the raw output files, 'best_params' is the champion of that specific seed/target
                # We extract the best candidate from each file
                if 'best_params' in data:
                    # Since the files only saved the 'best_params' but not the full stats
                    # for that specific trial in the JSON root, we have to check if
                    # the trial stats were saved alongside.
                    # If only params were saved, we can't score them without re-running.
                    # However, usually, these files also contain a 'best_trial' object.

                    trial = data.get('best_trial', {})
                    if trial:
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

    results_1h = []
    results_4h = []

    for s in tf_1h:
        score = calculate_rank_score(s)
        if score:
            results_1h.append({**s, 'calculated_score': score})

    for s in tf_4h:
        score = calculate_rank_score(s)
        if score:
            results_4h.append({**s, 'calculated_score': score})

    # Sort by the calculated score
    top_1h = sorted(results_1h, key=lambda x: x['calculated_score'], reverse=True)[:5]
    top_4h = sorted(results_4h, key=lambda x: x['calculated_score'], reverse=True)[:5]

    print("=== TOP 5 1H STRATEGIES ===")
    if not top_1h:
        print("No strategies passed the strict guardrails in the saved data.")
    else:
        print(f"{'Rank':<5} | {'Score':<8} | {'WinRate':<10} | {'PF':<8} | {'Exp':<8} | {'Seed':<6}")
            # Formatting here is wrong in a string, will fix in final print

    # Re-printing correctly
    for i, s in enumerate(top_1h, 1):
        print(f"{i:<5} | {s['calculated_score']:<8.2f} | {s['win_rate']:<10.2%} | {s['profit_factor']:<8.2f} | {s['expectancy_r']:<8.2f} | {s.get('seed', 'N/A')}")

    print("\n=== TOP 5 4H STRATEGIES ===")
    if not top_4h:
        print("No strategies passed the strict guardrails in the saved data.")
    else:
        for i, s in enumerate(top_4h, 1):
            print(f"{i:<5} | {s['calculated_score']:<8.2f} | {s['win_rate']:<10.2%} | {s['profit_factor']:<8.2f} | {s['expectancy_r']:<8.2f} | {s.get('seed', 'N/A')}")

if __name__ == "__main__":
    main()
