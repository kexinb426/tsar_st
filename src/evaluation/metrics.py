# src/evaluation/metrics.py
import json
import os
import numpy as np
from sklearn.metrics import f1_score, root_mean_squared_error
from transformers import pipeline
import evaluate

# This helper function reads the .jsonl files
def _read_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def _get_cefr_predictions(simplifications: list, models: list):
    """
    For each simplification, runs all CEFR models.
    
    Returns a list of dictionaries, where each dictionary contains:
    - 'best_prediction': The winner-takes-all prediction ({'label': 'A2', 'score': 0.95}).
    - 'all_model_probas': Full probability distributions from each model.
    """
    all_results = []
    for text in simplifications:
        # Get full probability distributions from each model
        model_outputs = [model(text, return_all_scores=True)[0] for model in models]

        # Find the overall best prediction (winner-takes-all)
        top_preds_from_each_model = [max(output, key=lambda d: d['score']) for output in model_outputs]
        best_overall_pred = max(top_preds_from_each_model, key=lambda d: d['score'])

        # Structure the full probabilities for storage
        all_probas = {}
        for i, model_output in enumerate(model_outputs):
            # Use the model's name as a descriptive key
            model_name = os.path.basename(models[i].model.config._name_or_path)
            probas_dict = {d['label']: round(d['score'], 6) for d in model_output}
            all_probas[model_name] = probas_dict
            
        all_results.append({
            "best_prediction": best_overall_pred,
            "all_model_probas": all_probas
        })
        
    return all_results

# --- main function called from other scripts! ---
def run_evaluation(system_output_path: str, reference_data_path: str):
    """
    The main "Head Judge" function.
    It now returns two items:
    1. A dictionary of the final, averaged scores.
    2. A list of dictionaries, with detailed scores for each text instance.
    """
    print("⚖️  Starting evaluation...")

    # 1. Load all the judging models
    print("   -> Loading CEFR models...")
    cefr_labeler1 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr")
    cefr_labeler2 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr")
    cefr_labeler3 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang-Cefr")
    cefr_models = [cefr_labeler1, cefr_labeler2, cefr_labeler3]

    print("   -> Loading similarity models...")
    meaning_bert = evaluate.load("davebulaval/meaningbert")
    bertscore = evaluate.load("bertscore")
    
    # 2. Read and align data
    full_ref_data = _read_jsonl(reference_data_path)
    sys_data = _read_jsonl(system_output_path)
    ref_map = {entry['text_id']: entry for entry in full_ref_data}
    sys_ids = [entry['text_id'] for entry in sys_data]
    ref_data = [ref_map[tid] for tid in sys_ids if tid in ref_map]
    
    print(f"   -> System output has {len(sys_data)} entries. Found {len(ref_data)} matching entries in reference data.")
    assert len(ref_data) == len(sys_data), "Data alignment failed."
    
    original_texts = [entry['original'] for entry in ref_data]
    target_cefr_levels = [entry['target_cefr'].upper() for entry in ref_data]
    reference_texts = [entry['reference'] for entry in ref_data]
    simplified_texts = [entry['simplified'] for entry in sys_data]

    # 3. Calculate all scores
    print("   -> Calculating scores for each instance...")
    
    # Get the rich prediction data for each simplified text
    cefr_predictions = _get_cefr_predictions(simplified_texts, cefr_models)
    
    meaningbert_scores_org = [meaning_bert.compute(predictions=[p], references=[r])['scores'][0] for p, r in zip(simplified_texts, original_texts)]
    bertscore_results_org = bertscore.compute(references=original_texts, predictions=simplified_texts, lang="en")
    meaningbert_scores_ref = [meaning_bert.compute(predictions=[p], references=[r])['scores'][0] for p, r in zip(simplified_texts, reference_texts)]
    bertscore_results_ref = bertscore.compute(references=reference_texts, predictions=simplified_texts, lang="en")

    # 4. Assemble the detailed per-instance results
    individual_results = []
    CEFR_LABELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    LABEL2IDX = {label: idx for idx, label in enumerate(CEFR_LABELS)}
    
    for i in range(len(sys_data)):
        best_pred = cefr_predictions[i]['best_prediction']
        true_cefr_idx = LABEL2IDX.get(target_cefr_levels[i], -1)
        pred_cefr_idx = LABEL2IDX.get(best_pred['label'], -1)
        
        instance_score = {
            "text_id": sys_data[i]['text_id'],
            "target_cefr": target_cefr_levels[i],
            "predicted_cefr": best_pred['label'],
            "predicted_cefr_confidence": round(best_pred['score'], 4),
            "cefr_adj_accuracy": 1 if abs(true_cefr_idx - pred_cefr_idx) <= 1 else 0,
            "meaningbert_orig": round(meaningbert_scores_org[i] / 100, 4),
            "bertscore_f1_orig": round(bertscore_results_org['f1'][i], 4),
            "meaningbert_ref": round(meaningbert_scores_ref[i] / 100, 4),
            "bertscore_f1_ref": round(bertscore_results_ref['f1'][i], 4),
            "cefr_probas": cefr_predictions[i]['all_model_probas'] # <-- NEWLY ADDED
        }
        individual_results.append(instance_score)

    # 5. Assemble the final aggregated report card
    agg_predicted_cefr = [d['predicted_cefr'] for d in individual_results]
    agg_adj_accuracy = [d['cefr_adj_accuracy'] for d in individual_results]
    
    agg_true_idx = np.array([LABEL2IDX.get(l, -1) for l in target_cefr_levels])
    agg_pred_idx = np.array([LABEL2IDX.get(l, -1) for l in agg_predicted_cefr])

    aggregated_scores = {
        "CEFR Compliance": {
            "weighted_f1": round(f1_score(target_cefr_levels, agg_predicted_cefr, average='weighted', labels=CEFR_LABELS, zero_division=0), 4),
            "adj_accuracy": round(np.mean(agg_adj_accuracy), 4),
            "rmse": round(root_mean_squared_error(agg_true_idx, agg_pred_idx), 4)
        },
        "Meaning Preservation": {
            "MeaningBERT-Orig": round(np.mean(meaningbert_scores_org) / 100, 4),
            "BERTScore-Orig": round(np.mean(bertscore_results_org['f1']), 4)
        },
        "Similarity to References": {
            "MeaningBERT-Ref": round(np.mean(meaningbert_scores_ref) / 100, 4),
            "BERTScore-Ref": round(np.mean(bertscore_results_ref['f1']), 4)
        }
    }
    
    print("✅ Evaluation complete!")
    return aggregated_scores, individual_results