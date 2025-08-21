# scripts/profile_references.py
import os
import sys
import json
from tqdm import tqdm
from transformers import pipeline
import evaluate

# Add the root directory to the Python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import _read_jsonl, _get_cefr_predictions

def profile_source_and_references():
    """
    Analyzes the original and reference texts to generate CEFR profiles
    and calculate the similarity between them.
    """
    print("🚀 Starting analysis of source and reference texts...")

    # 1. Define paths and load data
    originals_path = '/clwork/kexin/tsar_st/data/input/documents.jsonl'
    references_path = '/clwork/kexin/tsar_st/data/references/human_simplifications.jsonl'
    output_path = 'data/src_ref_results.jsonl'

    print("   -> Loading data files...")
    originals_data = _read_jsonl(originals_path)
    references_data = _read_jsonl(references_path)

    # 2. Align data by text_id
    original_map = {item['text_id']: item['original'] for item in originals_data}
    reference_map = {item['text_id']: item['reference'] for item in references_data}
    
    common_ids = sorted(original_map.keys() & reference_map.keys())
    print(f"   -> Found {len(common_ids)} common text_ids to analyze.")

    # Create aligned lists for batch processing
    originals_list = [original_map[tid] for tid in common_ids]
    references_list = [reference_map[tid] for tid in common_ids]

    # 3. Load all necessary models
    print("   -> Loading CEFR models...")
    cefr_labeler1 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr")
    cefr_labeler2 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr")
    cefr_labeler3 = pipeline(task="text-classification", model="AbdullahBarayan/ModernBERT-base-reference_AllLang-Cefr")
    cefr_models = [cefr_labeler1, cefr_labeler2, cefr_labeler3]

    print("   -> Loading similarity models...")
    meaning_bert = evaluate.load("davebulaval/meaningbert")
    bertscore = evaluate.load("bertscore")

    # 4. Run the analyses
    print("   -> Generating CEFR profiles for original texts...")
    cefr_profiles_orig = _get_cefr_predictions(originals_list, cefr_models)

    print("   -> Generating CEFR profiles for reference texts...")
    cefr_profiles_ref = _get_cefr_predictions(references_list, cefr_models)

    print("   -> Calculating similarity scores between original and reference...")
    bertscore_results = bertscore.compute(references=originals_list, predictions=references_list, lang="en")
    meaningbert_scores = [
        meaning_bert.compute(predictions=[p], references=[r])['scores'][0]
        for p, r in tqdm(zip(references_list, originals_list), total=len(common_ids), desc="MeaningBERT")
    ]

    # 5. Assemble the final results
    print("   -> Assembling final output file...")
    all_profiles = []
    for i, text_id in enumerate(common_ids):
        profile = {
            "text_id": text_id,
            "cefr_profile_original": cefr_profiles_orig[i],
            "cefr_profile_reference": cefr_profiles_ref[i],
            "similarity_scores": {
                "meaningbert": round(meaningbert_scores[i] / 100, 4),
                "bertscore_f1": round(bertscore_results['f1'][i], 4)
            }
        }
        all_profiles.append(profile)

    # 6. Save the output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for profile in all_profiles:
            f.write(json.dumps(profile) + '\n')
    
    print(f"✅ Analysis complete! Results saved to: {output_path}")

if __name__ == "__main__":
    profile_source_and_references()