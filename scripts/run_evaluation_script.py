# evaluate.py

import argparse
import json
import yaml
import os

# Import your existing evaluation function
from src.evaluation.metrics import run_evaluation

def main():
    """
    Standalone script to run evaluation on a pre-existing simplification file.
    """
    parser = argparse.ArgumentParser(description="Run evaluation on a generated simplification file.")
    
    # --- Define Command-Line Arguments ---
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the system's simplification output file (e.g., results/.../simplifications.jsonl)"
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default=None,
        help="Path to the reference data file. If not provided, it will be read from config.yaml."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the score files. Defaults to the same directory as the input file."
    )
    
    args = parser.parse_args()
    
    print("⚖️  Starting standalone evaluation...")

    # --- Determine file paths ---
    system_output_path = args.input_file
    
    reference_data_path = args.reference_file
    if not reference_data_path:
        print("-> No reference file provided, reading path from config.yaml...")
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        reference_data_path = config['paths']['evaluation_data']
    
    output_dir = args.output_dir
    if not output_dir:
        # Default to saving scores in the same folder as the input file
        output_dir = os.path.dirname(system_output_path)
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"-> Evaluating System Output: {system_output_path}")
    print(f"-> Against Reference Data: {reference_data_path}")
    print(f"-> Saving scores to: {output_dir}")
    
    # --- Run the evaluation ---
    agg_scores, individual_scores = run_evaluation(
        system_output_path=system_output_path,
        reference_data_path=reference_data_path
    )

    # --- Save the scores ---
    agg_scores_path = os.path.join(output_dir, 'scores_standalone.json')
    with open(agg_scores_path, 'w') as f:
        json.dump(agg_scores, f, indent=4)
    print(f"✅ Aggregated scores saved to: {agg_scores_path}")
    
    ind_scores_path = os.path.join(output_dir, 'individual_scores_standalone.jsonl')
    with open(ind_scores_path, 'w') as f:
        for score_record in individual_scores:
            f.write(json.dumps(score_record) + '\n')
    print(f"✅ Individual scores saved to: {ind_scores_path}")

    print("\n📊 Final Scores:")
    print(json.dumps(agg_scores, indent=2))

if __name__ == "__main__":
    main()