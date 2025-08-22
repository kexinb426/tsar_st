# # scripts/calculate_difficulty.py
# import json
# import os
# import numpy as np
# from transformers import pipeline
# from collections import defaultdict

# def _read_jsonl(filepath):
#     """Helper function to read a .jsonl file."""
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return [json.loads(line) for line in f]

# def calculate_difficulty(
#     input_path='data/raw/tsar2025_trialdata.jsonl',
#     output_path='data/source_with_difficulty.jsonl'
# ):
#     """
#     Reads the full dataset, runs CEFR classifiers on the ORIGINAL texts,
#     calculates an average "difficulty vector" (probability distribution),
#     and saves a new dataset enriched with this vector.
#     """
#     print("🚀 Starting source text difficulty calculation...")

#     # 1. Load all the judging models
#     print("   -> Loading CEFR models...")
#     # top_k=None returns probabilities for all classes
#     cefr_labeler1 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr", top_k=None)
#     cefr_labeler2 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr", top_k=None)
#     cefr_labeler3 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang-Cefr", top_k=None)
#     cefr_models = [cefr_labeler1, cefr_labeler2, cefr_labeler3]
    
#     CEFR_LABELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

#     # 2. Load the dataset
#     full_dataset = _read_jsonl(input_path)
#     print(f"   -> Found {len(full_dataset)} documents to analyze.")

#     # 3. Process each document and save to the new file
#     with open(output_path, 'w', encoding='utf-8') as f_out:
#         for i, doc in enumerate(full_dataset):
#             print(f"--> Processing document {i+1}/{len(full_dataset)} (ID: {doc['text_id']})")
            
#             original_text = doc['original']

#             # Get full probability distributions from all three models
#             all_model_preds = [model(original_text)[0] for model in cefr_models]

#             # Average the distributions to get a stable difficulty vector
#             avg_probs = defaultdict(list)
#             for model_pred_list in all_model_preds:
#                 for pred in model_pred_list:
#                     avg_probs[pred['label']].append(pred['score'])
            
#             # Create the final vector in the correct order (A1, A2, ...)
#             difficulty_vector = [np.mean(avg_probs[label]) for label in CEFR_LABELS]

#             # Create the new record with the added vector
#             new_doc = doc.copy()
#             new_doc['difficulty_vector'] = [round(p, 6) for p in difficulty_vector]
            
#             f_out.write(json.dumps(new_doc) + '\n')

#     print(f"\n✅ Successfully created difficulty-scored file at: '{output_path}'")


# if __name__ == "__main__":
#     calculate_difficulty()

# scripts/calculate_difficulty.py
import json
import os
import numpy as np
from transformers import pipeline
from collections import defaultdict

def _read_jsonl(filepath):
    """Helper function to read a .jsonl file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def calculate_difficulty(
    input_path='data/raw/tsar2025_trialdata.jsonl',
    output_path='data/source_with_difficulty.jsonl'
):
    """
    Reads the full dataset, runs CEFR classifiers on the ORIGINAL texts,
    calculates an average 'difficulty vector' (optional),
    and sets predicted_label to the label from ANY classifier with the highest confidence.
    """
    print("🚀 Starting source text difficulty calculation...")

    # 1. Load all the judging models
    print("   -> Loading CEFR models...")
    cefr_labeler1 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr", top_k=None)
    cefr_labeler2 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr", top_k=None)
    cefr_labeler3 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang-Cefr", top_k=None)
    cefr_models = [cefr_labeler1, cefr_labeler2, cefr_labeler3]
    
    CEFR_LABELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

    # 2. Load the dataset
    full_dataset = _read_jsonl(input_path)
    print(f"   -> Found {len(full_dataset)} documents to analyze.")

    # 3. Process each document and save to the new file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, doc in enumerate(full_dataset):
            print(f"--> Processing document {i+1}/{len(full_dataset)} (ID: {doc['text_id']})")
            original_text = doc['original']

            # Get full probability distributions from all three models
            all_model_preds = [model(original_text)[0] for model in cefr_models]  # list of lists

            # ---- NEW LOGIC: pick the single highest-probability label across models ----
            best_label = None
            best_score = -1.0
            # (optional) track which model produced it
            best_model_idx = None

            for m_idx, preds in enumerate(all_model_preds):
                for p in preds:
                    if p['score'] > best_score:
                        best_score = p['score']
                        best_label = p['label']
                        best_model_idx = m_idx

            # Keep your averaged difficulty vector (useful for analysis)
            avg_probs = defaultdict(list)
            for preds in all_model_preds:
                for p in preds:
                    avg_probs[p['label']].append(p['score'])
            difficulty_vector = [float(np.mean(avg_probs[label])) for label in CEFR_LABELS]

            # Create the new record
            new_doc = doc.copy()
            new_doc['difficulty_vector'] = [round(p, 6) for p in difficulty_vector]
            new_doc['predicted_cefr'] = best_label                    # <- from the single highest prob
            new_doc['predicted_confidence'] = round(float(best_score), 6)  # optional but handy
            new_doc['predicted_from_model'] = best_model_idx           # optional traceability (0,1,2)

            f_out.write(json.dumps(new_doc) + '\n')

    print(f"\n✅ Successfully created difficulty-scored file at: '{output_path}'")


if __name__ == "__main__":
    calculate_difficulty()