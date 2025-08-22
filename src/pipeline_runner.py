# src/pipeline_runner.py
import os
import json
import yaml
import random
import time
import math
import datetime
from collections import defaultdict
from botocore.exceptions import ClientError
from scipy.spatial.distance import cosine
# from Levenshtein import distance as levenshtein # <-- Import for edit distance

# Import our custom modules
from src.models.gpt import GPTModel
from src.models.claude import ClaudeModel
from src.models.gemma import GemmaModel
from src.evaluation.metrics import run_evaluation
from src.utils.prompt_builder import build_prompt

def _read_jsonl(filepath):
    """Helper function to read a .jsonl file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def _initialize_model(model_name, global_config):
    """Helper function to initialize a model from config."""
    model_config = global_config['models'][model_name]
    model_type = model_config.get('type')
    print(f"   -> Initializing model: '{model_name}' ({model_type})...")
    if model_type == 'gpt':
        return GPTModel(api_model_name=model_config['api_model_name'], temperature=model_config.get('temperature', 1.0))
    elif model_type == 'claude':
        return ClaudeModel(api_model_name=model_config['api_model_name'], temperature=model_config.get('temperature', 1.0))
    elif model_type == 'gemma':
        return GemmaModel(api_model_name=model_config['api_model_name'], temperature=model_config.get('temperature', 1.0))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_standard_pipeline(pipeline_config, global_config, all_data, exp_run_name, output_dir, models):
    """Executes a standard, multi-step generation pipeline using pre-initialized models."""
    first_step_config = pipeline_config['steps'][0]
    sample_size = first_step_config.get('sample_size', None)
    contrastive_config = first_step_config.get('contrastive', None)
    
    docs_to_process = all_data['input_documents']
    if contrastive_config:
        docs_to_process = [d for d in docs_to_process if d['target_cefr'].upper() in ['A2', 'B1']]
    if sample_size is not None:
        docs_to_process = docs_to_process[:sample_size]
    
    print(f"   -> Processing {len(docs_to_process)} documents through {len(pipeline_config['steps'])} steps...")

    final_results = []
    for i, doc in enumerate(docs_to_process):
        print(f"\n--> Processing document {i+1}/{len(docs_to_process)} (ID: {doc['text_id']})")
        
        intermediate_text = doc['original']
        step_outputs = {}

        for step_num, step_config in enumerate(pipeline_config['steps']):
            step_name = step_config['step_name']
            print(f"    - Running Step {step_num+1}: {step_name}")

            # Get the pre-initialized model from the dictionary
            model = models[step_config['model']]
            prompt_template_text = all_data['prompt_templates'][step_config['prompt_template']]
            
            # Look up all the analysis data for the current document
            avoid_words = all_data['avoid_words_map'].get(doc['text_id'], [])
            named_entities = all_data['named_entities_map'].get(doc['text_id'], [])
            source_cefr = all_data['source_cefr_map'].get(doc['text_id'], 'N/A')

            final_prompt = build_prompt(
                prompt_template=prompt_template_text,
                target_cefr=doc['target_cefr'].upper(),
                document_text=doc['original'],
                previous_step_output=intermediate_text,
                cefr_descriptions=all_data['cefr_descriptions'],
                cefr_instructions=all_data['cefr_instructions'],
                cefr_translate_instructions = all_data['cefr_translate_instructions'],
                cefr_simp_from_trans_instructions = all_data['cefr_simp_from_trans_instructions'],
                cefr_judge_criterias = all_data['cefr_judge_criterias'],
                avoid_words=avoid_words,
                named_entities=named_entities,
                source_predicted_cefr=source_cefr,
                standard_examples=[], a2_examples=[], b1_examples=[] 
            )

            # --- API Call with Retry Logic ---
            raw_output, clean_output = "", ""
            processed_successfully = False
            backoff_delay = 1.0
            max_backoff = 64.0
            while not processed_successfully:
                try:
                    raw_output, clean_output = model.simplify(prompt=final_prompt)
                    processed_successfully = True
                    print(f"      ✅ Success!")
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ThrottlingException':
                        print(f"      🐌 Throttled! Retrying after {backoff_delay:.2f} seconds...")
                        time.sleep(backoff_delay)
                        backoff_delay = min(max_backoff, backoff_delay * 2 + random.uniform(0, 1))
                    else:
                        print(f"      ❌ Unrecoverable AWS error: {e}")
                        break
            
            step_outputs[step_name] = {'raw_output': raw_output, 'clean_output': clean_output, 'prompt': final_prompt}
            intermediate_text = clean_output

        final_results.append({"text_id": doc['text_id'], "simplified": intermediate_text, "step_outputs": step_outputs})

    return final_results

def run_ensemble_pipeline(pipeline_config, global_config, all_data, exp_run_name, output_dir, models):
    """Executes an ensemble pipeline that filters and judges previous runs."""
    print("   -> Detected ENSEMBLE pipeline type.")
    
    candidates_by_id = defaultdict(list)
    original_text_map = {doc['text_id']: doc['original'] for doc in all_data['input_documents']}

    for run_path in pipeline_config['candidate_runs']:
        simplifications_path = os.path.join(run_path, 'simplifications.jsonl')
        scores_path = os.path.join(run_path, 'individual_scores.jsonl')
        
        if not os.path.exists(simplifications_path) or not os.path.exists(scores_path):
            print(f"   -> ⚠️ Warning: Skipping candidate run '{run_path}' (missing files).")
            continue
        
        simplifications = {s['text_id']: s for s in _read_jsonl(simplifications_path)}
        scores = {s['text_id']: s for s in _read_jsonl(scores_path)}

        for text_id, score_data in scores.items():
            if text_id in simplifications:
                candidate = {**simplifications[text_id], **score_data, 'source_run': run_path}
                candidate['original'] = original_text_map.get(text_id, "")
                candidates_by_id[text_id].append(candidate)
    
    print(f"   -> Loaded candidates for {len(candidates_by_id)} unique documents.")

    filter_step = pipeline_config['steps'][0]
    judge_step = pipeline_config['steps'][1]
    
    # Get the pre-initialized judge model
    judge_model = models[judge_step['model']]
    judge_prompt_template = all_data['prompt_templates'][judge_step['prompt_template']]

    final_results = []
    docs_to_process = [doc for doc in all_data['input_documents'] if doc['text_id'] in candidates_by_id]

    for i, doc in enumerate(docs_to_process):
        print(f"\n--> Ensembling document {i+1}/{len(docs_to_process)} (ID: {doc['text_id']})")
        
        candidates = candidates_by_id[doc['text_id']]
        criteria = filter_step.get('filter_criteria', {})
        
        filtered_candidates = candidates
        if isinstance(criteria, list):
            filtered_candidates = [c for c in candidates if all(eval(crit, {"__builtins__": {}}, c) for crit in criteria)]
        
        elif isinstance(criteria, dict) and criteria.get('strategy') == 'top_percent':
            metric_formula = criteria['metric']
            percentile = criteria['percentile']
            
            for cand in candidates:
                try:
                    cand['combined_score'] = eval(metric_formula, {"__builtins__": {}}, cand)
                except Exception as e:
                    cand['combined_score'] = -1
            
            candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            num_to_keep = math.ceil(len(candidates) * (percentile / 100.0))
            filtered_candidates = candidates[:num_to_keep]

        print(f"    - Filtered {len(candidates)} candidates down to {len(filtered_candidates)}.")

        if not filtered_candidates:
            final_results.append({"text_id": doc['text_id'], "simplified": "NO CANDIDATES PASSED FILTERING", "step_outputs": {}})
            continue

        candidate_text_for_prompt = ""
        for j, cand in enumerate(filtered_candidates):
            candidate_text_for_prompt += f"Candidate {j+1}:\n{cand['simplified']}\n\n"
        
        target_cefr = doc['target_cefr'].upper()
        judge_instruction = all_data['cefr_judge_criterias'].get(target_cefr, "")
        
        final_prompt = judge_prompt_template.format(
            document_text=doc['original'],
            target_cefr=target_cefr,
            candidate_simplifications=candidate_text_for_prompt.strip(),
            cefr_judge_criteria=judge_instruction
        )
        
        # --- API Call with Retry Logic for the Judge ---
        ensembled_output = ""
        processed_successfully = False
        backoff_delay = 1.0
        max_backoff = 64.0
        while not processed_successfully:
            try:
                _, ensembled_output = judge_model.simplify(prompt=final_prompt)
                processed_successfully = True
                print(f"      ✅ Judge Success!")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    print(f"      🐌 Judge Throttled! Retrying after {backoff_delay:.2f} seconds...")
                    time.sleep(backoff_delay)
                    backoff_delay = min(max_backoff, backoff_delay * 2 + random.uniform(0, 1))
                else:
                    print(f"      ❌ Unrecoverable AWS error for Judge: {e}")
                    break
        
        final_results.append({
            "text_id": doc['text_id'],
            "simplified": ensembled_output,
            "step_outputs": {
                "llm_judge": {"prompt": final_prompt, "clean_output": ensembled_output}
            }
        })

    return final_results


def run_pipeline(pipeline_config, global_config, all_data):
    """
    Main dispatcher. Determines the pipeline type, pre-initializes all necessary models,
    and then calls the appropriate runner function.
    """
    pipeline_name = pipeline_config['name']
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_run_name = f"{pipeline_name}_{timestamp}"

    print(f"\n🔥🔥🔥 Running Pipeline: {pipeline_name} (Run: {exp_run_name}) 🔥🔥�")

    output_dir = os.path.join(global_config['paths']['results_dir'], exp_run_name)
    os.makedirs(output_dir, exist_ok=True)
    config_copy_path = os.path.join(output_dir, 'config.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(global_config, f, default_flow_style=False)
    print(f"   -> Config saved to: {config_copy_path}")

    # --- EFFICIENT MODEL LOADING ---
    # Scan all steps in the pipeline to find the unique set of models needed.
    models_to_load = set()
    for step in pipeline_config['steps']:
        models_to_load.add(step['model'])
    
    # Initialize each required model only once.
    initialized_models = {}
    for model_name in models_to_load:
        initialized_models[model_name] = _initialize_model(model_name, global_config)
    
    print("   -> All models for this pipeline have been initialized.")

    pipeline_type = pipeline_config.get('pipeline_type', 'standard')
    
    if pipeline_type == 'ensemble':
        final_results = run_ensemble_pipeline(pipeline_config, global_config, all_data, exp_run_name, output_dir, initialized_models)
    else: # Default to standard pipeline
        final_results = run_standard_pipeline(pipeline_config, global_config, all_data, exp_run_name, output_dir, initialized_models)

    simplifications_path = os.path.join(output_dir, 'simplifications.jsonl')
    with open(simplifications_path, 'w', encoding='utf-8') as f_out:
        for result in final_results:
            f_out.write(json.dumps(result) + '\n')
    print(f"\n   -> Final simplifications saved to: {simplifications_path}")

    agg_scores, individual_scores = run_evaluation(
        system_output_path=simplifications_path,
        reference_data_path=global_config['paths']['evaluation_data']
    )

    agg_scores_path = os.path.join(output_dir, 'scores.json')
    with open(agg_scores_path, 'w') as f: json.dump(agg_scores, f, indent=4)
    print(f"   -> Aggregated scores saved to: {agg_scores_path}")
    
    ind_scores_path = os.path.join(output_dir, 'individual_scores.jsonl')
    with open(ind_scores_path, 'w') as f:
        for score_record in individual_scores: f.write(json.dumps(score_record) + '\n')
    print(f"   -> Individual scores saved to: {ind_scores_path}")

    print(f"📊 Final Scores: {json.dumps(agg_scores, indent=2)}")