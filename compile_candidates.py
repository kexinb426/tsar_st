import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm # For a nice progress bar

# --- 1. SETUP: Define file and directory paths ---
BASE_PATH = Path("/clwork/kexin/tsar_st/")
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"

# Input files
ORIGINALS_FILE = DATA_PATH / "input/documents.jsonl"
REFERENCES_FILE = DATA_PATH / "references/human_simplifications.jsonl"

# Output file
COMPILED_OUTPUT_FILE = RESULTS_PATH / "compiled_results.json"
FLAT_CSV_OUTPUT_FILE = RESULTS_PATH / "compiled_results_flat.csv"

# --- 2. HELPER FUNCTIONS ---
def load_jsonl(file_path):
    """Loads a .jsonl file into a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def create_lookup(data, key_field, value_field):
    """Creates a dictionary for quick lookups from a list of dicts."""
    return {item[key_field]: item[value_field] for item in data}

# --- 3. LOAD STATIC DATA (Originals & References) ---
print("Loading original documents and human references...")
originals_data = load_jsonl(ORIGINALS_FILE)
references_data = load_jsonl(REFERENCES_FILE)

# Create lookup tables for fast access
originals_lookup = create_lookup(originals_data, 'text_id', 'original')
references_lookup = create_lookup(references_data, 'text_id', 'reference')

print(f"Loaded {len(originals_lookup)} originals and {len(references_lookup)} references.")

# --- 4. COMPILE RESULTS FROM ALL RUNS ---
compiled_data = {}

# Find all subdirectories in the results path
run_dirs = [d for d in RESULTS_PATH.iterdir() if d.is_dir()]
print(f"\nFound {len(run_dirs)} potential experiment runs. Processing...")

for run_dir in tqdm(run_dirs, desc="Processing Runs"):
    run_name = run_dir.name
    simplifications_path = run_dir / "simplifications.jsonl"
    scores_path = run_dir / "individual_scores.jsonl"

    # Ensure both required files exist for a given run
    if not (simplifications_path.exists() and scores_path.exists()):
        # print(f"Skipping '{run_name}': missing simplifications or scores file.")
        continue

    simplifications = load_jsonl(simplifications_path)
    scores = load_jsonl(scores_path)
    scores_lookup = {score['text_id']: score for score in scores}

    # Process each simplified sentence in the run
    for simpl in simplifications:
        text_id = simpl['text_id']
        
        # Initialize entry for this text_id if it's the first time we see it
        if text_id not in compiled_data:
            compiled_data[text_id] = {
                'original': originals_lookup.get(text_id),
                'reference': references_lookup.get(text_id),
                'target_cefr': scores_lookup.get(text_id, {}).get('target_cefr'),
                'candidates': []
            }
        
        # Combine simplification with its scores
        candidate_scores = scores_lookup.get(text_id, {})
        candidate_info = {
            'run_name': run_name,
            'simplified_text': simpl.get('simplified'),
            **candidate_scores # Unpack all scores into this dict
        }
        
        compiled_data[text_id]['candidates'].append(candidate_info)

# --- 5. SAVE THE COMPILED DATA ---
print(f"\nSaving compiled data for {len(compiled_data)} unique text_ids to {COMPILED_OUTPUT_FILE}")
with open(COMPILED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(compiled_data, f, indent=2)

print("Compilation complete.")

# # --- 6. ANALYSIS: FLATTEN DATA FOR PANDAS ---
# print("\n--- Analysis with Pandas ---")
# flat_data = []
# for text_id, data in compiled_data.items():
#     for candidate in data['candidates']:
#         row = {
#             'text_id': text_id,
#             'target_cefr': data['target_cefr'],
#             'original': data['original'],
#             'reference': data['reference'],
#             **candidate
#         }
#         flat_data.append(row)

# # Create DataFrame
# df = pd.DataFrame(flat_data)

# # Reorder columns for better readability
# cols_order = [
#     'text_id', 'target_cefr', 'run_name', 'simplified_text', 
#     'meaningbert_ref', 'meaningbert_orig', 'bertscore_f1_ref', 'bertscore_f1_orig',
#     'cefr_adj_accuracy', 'predicted_cefr', 'predicted_cefr_confidence',
#     'original', 'reference'
# ]
# # Ensure all expected columns exist before reordering
# existing_cols_order = [col for col in cols_order if col in df.columns]
# df = df[existing_cols_order]

# # Save the flat structure to a CSV for easy access in other tools (e.g., Excel)
# df.to_csv(FLAT_CSV_OUTPUT_FILE, index=False)
# print(f"Saved flat data to {FLAT_CSV_OUTPUT_FILE}")

# # Set pandas display options for wider columns
# pd.set_option('display.max_rows', 200)
# pd.set_option('display.max_colwidth', 80)

# # --- Example Analysis: View all candidates for one specific text_id ---
# print("\nExample 1: Viewing all candidates for text_id '01-a2'")
# example_id = '01-a2'
# if example_id in df['text_id'].values:
#     display(df[df['text_id'] == example_id][['run_name', 'simplified_text', 'meaningbert_ref', 'cefr_adj_accuracy']].sort_values(by='meaningbert_ref', ascending=False))
# else:
#     print(f"Could not find example text_id '{example_id}' in the data.")

# # --- Example Analysis: Find the best runs on average ---
# print("\nExample 2: Average scores per run (model/configuration)")
# avg_scores = df.groupby('run_name').agg({
#     'meaningbert_ref': 'mean',
#     'meaningbert_orig': 'mean',
#     'cefr_adj_accuracy': 'mean',
#     'predicted_cefr_confidence': 'mean'
# }).sort_values(by='meaningbert_ref', ascending=False)

# display(avg_scores)