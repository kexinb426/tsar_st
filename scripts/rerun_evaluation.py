# scripts/rerun_evaluation.py
import os
import sys
import json
import yaml
import numpy as np
from transformers import pipeline
from sklearn.metrics import f1_score, root_mean_squared_error

# Add the root directory to the Python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# We can import the main evaluation function and also its internal helpers
from src.evaluation.metrics import run_evaluation, _read_jsonl, _get_cefr_predictions

def rerun_full_evaluation(results_folder_path):
    """
    Re-runs the full evaluation for a specific, existing results folder
    and overwrites its score files.
    """
    print(f"🔄 Re-running FULL evaluation for: {results_folder_path}")

    # Define paths based on the input folder
    simplifications_path = os.path.join(results_folder_path, 'simplifications.jsonl')
    config_path = os.path.join(results_folder_path, 'config.yaml')
    agg_scores_path = os.path.join(results_folder_path, 'scores.json')
    ind_scores_path = os.path.join(results_folder_path, 'individual_scores.jsonl')

    # Check if the necessary files exist
    if not os.path.exists(simplifications_path) or not os.path.exists(config_path):
        print(f"❌ Error: Cannot find 'simplifications.jsonl' or 'config.yaml' in the specified directory.")
        return

    # Load the original config to get the path to the evaluation data
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    reference_data_path = config['paths']['evaluation_data']

    # Run the (now corrected) evaluation function
    agg_scores, individual_scores = run_evaluation(
        system_output_path=simplifications_path,
        reference_data_path=reference_data_path
    )

    # Overwrite the old aggregated scores file
    with open(agg_scores_path, 'w') as f:
        json.dump(agg_scores, f, indent=4)
    print(f"✅ Aggregated scores overwritten at: {agg_scores_path}")
    
    # Overwrite the old individual scores file
    with open(ind_scores_path, 'w') as f:
        for score_record in individual_scores:
            f.write(json.dumps(score_record) + '\n')
    print(f"✅ Individual scores overwritten at: {ind_scores_path}")

    print(f"📊 New Scores: {json.dumps(agg_scores, indent=2)}")


def rerun_cefr_only(results_folder_path):
    """
    Re-runs ONLY the CEFR evaluation for a results folder,
    updates the score files without re-calculating other metrics.
    """
    print(f"🔄 Re-running CEFR-ONLY evaluation for: {results_folder_path}")

    # 1. Define paths
    simplifications_path = os.path.join(results_folder_path, 'simplifications.jsonl')
    config_path = os.path.join(results_folder_path, 'config.yaml')
    agg_scores_path = os.path.join(results_folder_path, 'scores.json')
    ind_scores_path = os.path.join(results_folder_path, 'individual_scores.jsonl')

    if not all(os.path.exists(p) for p in [simplifications_path, config_path, agg_scores_path, ind_scores_path]):
        print(f"❌ Error: One or more required files are missing. Cannot run CEFR-only update.")
        return

    # 2. Load necessary data
    print("   -> Loading data...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    reference_data_path = config['paths']['evaluation_data']
    
    sys_data = _read_jsonl(simplifications_path)
    ref_data_full = _read_jsonl(reference_data_path)
    individual_scores_old = _read_jsonl(ind_scores_path)
    
    # Align reference data
    ref_map = {entry['text_id']: entry for entry in ref_data_full}
    ref_data = [ref_map[entry['text_id']] for entry in sys_data]
    
    simplified_texts = [entry['simplified'] for entry in sys_data]
    target_cefr_levels = [entry['target_cefr'].upper() for entry in ref_data]

    # 3. Re-run CEFR classification
    print("   -> Loading CEFR models and re-classifying...")
    cefr_labeler1 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr")
    cefr_labeler2 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr")
    cefr_labeler3 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang-Cefr")
    cefr_models = [cefr_labeler1, cefr_labeler2, cefr_labeler3]
    
    new_cefr_predictions = _get_cefr_predictions(simplified_texts, cefr_models)

    # 4. Update the individual scores file
    print("   -> Updating individual scores...")
    CEFR_LABELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    LABEL2IDX = {label: idx for idx, label in enumerate(CEFR_LABELS)}
    
    ind_scores_map = {score['text_id']: score for score in individual_scores_old}
    new_individual_scores = []
    for i, pred in enumerate(new_cefr_predictions):
        text_id = sys_data[i]['text_id']
        updated_score = ind_scores_map[text_id]
        
        true_cefr_idx = LABEL2IDX.get(target_cefr_levels[i], -1)
        pred_cefr_idx = LABEL2IDX.get(pred['label'], -1)
        
        updated_score['predicted_cefr'] = pred['label']
        updated_score['predicted_cefr_confidence'] = round(pred['score'], 4)
        updated_score['cefr_adj_accuracy'] = 1 if abs(true_cefr_idx - pred_cefr_idx) <= 1 else 0
        
        new_individual_scores.append(updated_score)

    # 5. Re-aggregate and update the main scores file
    print("   -> Updating aggregated scores...")
    with open(agg_scores_path, 'r') as f:
        agg_scores_old = json.load(f)

    agg_predicted_cefr = [d['predicted_cefr'] for d in new_individual_scores]
    agg_adj_accuracy = [d['cefr_adj_accuracy'] for d in new_individual_scores]
    agg_true_idx = np.array([LABEL2IDX.get(l, -1) for l in target_cefr_levels])
    agg_pred_idx = np.array([LABEL2IDX.get(l, -1) for l in agg_predicted_cefr])

    new_cefr_compliance = {
        "weighted_f1": round(f1_score(target_cefr_levels, agg_predicted_cefr, average='weighted', labels=CEFR_LABELS, zero_division=0), 4),
        "adj_accuracy": round(np.mean(agg_adj_accuracy), 4),
        "rmse": round(root_mean_squared_error(agg_true_idx, agg_pred_idx), 4)
    }
    
    agg_scores_old['CEFR Compliance'] = new_cefr_compliance
    new_agg_scores = agg_scores_old

    # 6. Overwrite the files
    with open(agg_scores_path, 'w') as f:
        json.dump(new_agg_scores, f, indent=4)
    print(f"✅ Aggregated scores updated at: {agg_scores_path}")
    
    with open(ind_scores_path, 'w') as f:
        for score_record in new_individual_scores:
            f.write(json.dumps(score_record) + '\n')
    print(f"✅ Individual scores updated at: {ind_scores_path}")

    print(f"📊 New Scores: {json.dumps(new_agg_scores, indent=2)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/rerun_evaluation.py <path_to_results_folder> [--cefr-only]")
        print("  python scripts/rerun_evaluation.py --all [--cefr-only]")
        sys.exit(1)

    target = sys.argv[1]
    cefr_only_mode = '--cefr-only' in sys.argv

    # Determine which function to call based on the mode
    evaluation_function = rerun_cefr_only if cefr_only_mode else rerun_full_evaluation

    if target == '--all':
        print("🚀 Running evaluation for all result directories...")
        results_dir = 'results'
        if not os.path.isdir(results_dir):
            print(f"❌ Error: Results directory '{results_dir}' not found.")
            sys.exit(1)
            
        subdirectories = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        
        if not subdirectories:
            print("No result directories found to process.")
            sys.exit(0)

        for subdir_name in subdirectories:
            folder_path = os.path.join(results_dir, subdir_name)
            try:
                evaluation_function(folder_path)
                print("-" * 50)
            except Exception as e:
                print(f"❌ Failed to process {folder_path}: {e}")
                print("-" * 50)

    else: # It's a single folder path
        folder_path = target
        if not os.path.isdir(folder_path):
            print(f"❌ Error: Directory not found at '{folder_path}'")
            sys.exit(1)
        evaluation_function(folder_path)

# full evaluation: python scripts/rerun_evaluation.py results/gpt4o_cot_20250727-173349/
# just CEFR part: python scripts/rerun_evaluation.py results/gpt4o_cot_20250727-173349/ --cefr-only
# just CEFR FOR EVERYTHING UNDER results/: python scripts/rerun_evaluation.py --all --cefr-only