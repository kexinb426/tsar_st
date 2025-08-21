# scripts/aggregate_results.py
import os
import json
import pandas as pd
from collections import defaultdict

def aggregate_results(results_dir='results'):
    """
    Scans all experiment subdirectories in the results folder,
    aggregates the scores.json files, and prints a summary table.
    """
    all_results = []

    print(f"🔍 Scanning for results in '{results_dir}'...")

    # Find all subdirectories in the results folder
    for exp_run_name in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_run_name)
        
        # Check if it's a directory
        if not os.path.isdir(exp_path):
            continue

        scores_path = os.path.join(exp_path, 'scores.json')
        
        # Check if a scores.json file exists in the directory
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                scores = json.load(f)
            
            # Flatten the nested JSON into a single dictionary
            flat_scores = {}
            for category, metrics in scores.items():
                for metric_name, value in metrics.items():
                    # Create a clean metric name like "CEFR_weighted_f1"
                    clean_metric_name = f"{category.split(' ')[0]}_{metric_name}"
                    flat_scores[clean_metric_name] = value
            
            # Use the directory name as the unique identifier for the run
            flat_scores['Experiment Run'] = exp_run_name
            all_results.append(flat_scores)

    if not all_results:
        print("❌ No 'scores.json' files found. Run some experiments first!")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns to have the experiment name first
    cols = ['Experiment Run'] + [col for col in df.columns if col != 'Experiment Run']
    df = df[cols]
    
    # Set a more readable index
    df.set_index('Experiment Run', inplace=True)
    
    # Sort the table by the index (experiment name)
    df.sort_index(inplace=True)

    # --- Display and Save Results ---
    print("\n--- Experiment Results Summary ---")
    # Print the full table to the console
    print(df.to_string())

    # Save the table to a CSV file for easy access
    summary_csv_path = 'a_misc./agg_results/results_summary_08201235.csv'
    df.to_csv(summary_csv_path)
    print(f"\n✅ Summary table saved to '{summary_csv_path}'")


if __name__ == "__main__":
    aggregate_results()
