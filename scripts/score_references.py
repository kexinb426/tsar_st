# scripts/score_references.py

"""
1. Read your full dataset.

2. Run the CEFR classifiers on every human-written reference text.

3. For each reference, find the classifier's confidence score for its correct target level (e.g., for an A2 reference, what was the confidence score for the "A2" label?).

4. Save this information to a new file: data/references_with_scores.jsonl.
"""

import json
import os
import numpy as np
from transformers import pipeline

def _read_jsonl(filepath):
    """Helper function to read a .jsonl file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def score_references(
    input_path='data/raw/tsar2025_trialdata.jsonl',
    output_path='data/references_with_scores.jsonl'
):
    """
    Reads the full dataset, runs CEFR classifiers on the reference texts,
    and saves a new dataset enriched with confidence scores.
    """
    print("🚀 Starting reference scoring process...")

    # 1. Load all the judging models
    print("   -> Loading CEFR models...")
    # We only need the classifiers for this task
    cefr_labeler1 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr", top_k=None)
    cefr_labeler2 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr", top_k=None)
    cefr_labeler3 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang-Cefr", top_k=None)
    cefr_models = [cefr_labeler1, cefr_labeler2, cefr_labeler3]

    # 2. Load the dataset
    full_dataset = _read_jsonl(input_path)
    print(f"   -> Found {len(full_dataset)} documents to score.")

    # 3. Process each document and save to the new file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, doc in enumerate(full_dataset):
            print(f"--> Processing document {i+1}/{len(full_dataset)} (ID: {doc['text_id']})")
            
            reference_text = doc['reference']
            target_level = doc['target_cefr'].upper()

            # Run all three models and get their full probability distributions
            all_model_preds = [model(reference_text)[0] for model in cefr_models]

            # Find the confidence score for the *correct* target level from each model
            scores_for_target_level = []
            for model_pred_list in all_model_preds:
                for pred in model_pred_list:
                    if pred['label'] == target_level:
                        scores_for_target_level.append(pred['score'])
                        break # Move to the next model's predictions
            
            # The final confidence is the AVERAGE score for that label across the three models
            if scores_for_target_level:
                confidence_score = np.mean(scores_for_target_level)
            else:
                confidence_score = 0.0 # Should not happen if labels are correct

            # Create the new record with the added score
            new_doc = doc.copy()
            new_doc['confidence_score'] = round(float(confidence_score), 4)
            
            f_out.write(json.dumps(new_doc) + '\n')

    print(f"\n✅ Successfully created scored reference file at: '{output_path}'")


if __name__ == "__main__":
    score_references()