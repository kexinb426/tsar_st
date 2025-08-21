# src/pipeline_runner.py
import os
import json
import yaml
import random
import time
import datetime
from collections import defaultdict
from botocore.exceptions import ClientError
from scipy.spatial.distance import cosine

# Import our custom modules
from src.models.gpt import GPTModel
from src.models.claude import ClaudeModel
from src.models.gemma import GemmaModel
from src.evaluation.metrics import run_evaluation
from src.utils.prompt_builder import build_prompt

def run_pipeline(pipeline_config, global_config, all_data):
    """
    Executes a single pipeline, which can now contain multiple steps.
    The output of each step is fed as input to the next.
    """
    pipeline_name = pipeline_config['name']
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_run_name = f"{pipeline_name}_{timestamp}"

    print(f"\n🔥🔥🔥 Running Pipeline: {pipeline_name} (Run: {exp_run_name}) 🔥🔥🔥")

    # --- Prepare results directory and save config ---
    output_dir = os.path.join(global_config['paths']['results_dir'], exp_run_name)
    os.makedirs(output_dir, exist_ok=True)
    config_copy_path = os.path.join(output_dir, 'config.yaml')
    with open(config_copy_path, 'w') as f_config:
        yaml.dump(global_config, f_config, default_flow_style=False)
    print(f"   -> Config saved to: {config_copy_path}")

    # Determine which documents to process based on the first step's config
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
        
        intermediate_text = doc['original'] # Start with the original text
        step_outputs = {} # To store outputs from each step

        # --- Loop through each step in the pipeline ---
        for step_num, step_config in enumerate(pipeline_config['steps']):
            step_name = step_config['step_name']
            print(f"    - Running Step {step_num+1}: {step_name}")

            # (All the logic for model init, prompt building, etc., now lives inside the step loop)
            model_name = step_config['model']
            model_config = global_config['models'][model_name]
            model_type = model_config.get('type')
            
            if model_type == 'gpt': model = GPTModel(api_model_name=model_config['api_model_name'], temperature=model_config.get('temperature', 1.0))
            elif model_type == 'claude': model = ClaudeModel(api_model_name=model_config['api_model_name'], temperature=model_config.get('temperature', 1.0))
            elif model_type == 'gemma': model = GemmaModel(api_model_name=model_config['api_model_name'], temperature=model_config.get('temperature', 1.0))
            else: raise ValueError(f"Unknown model type: {model_type}")

            prompt_template_text = all_data['prompt_templates'][step_config['prompt_template']]
            
            # Build the prompt for the current step
            final_prompt = build_prompt(
                prompt_template=prompt_template_text,
                target_cefr=doc['target_cefr'].upper(),
                document_text=doc['original'],
                previous_step_output=intermediate_text, # Pass the output from the last step
                cefr_descriptions=all_data['cefr_descriptions'],
                cefr_instructions=all_data['cefr_instructions'],
                cefr_translate_instructions = all_data['cefr_translate_instructions'],
                cefr_simp_from_trans_instructions = all_data['cefr_simp_from_trans_instructions'],
                # (Few-shot logic can be adapted here if needed for specific steps)
                standard_examples=[], a2_examples=[], b1_examples=[] 
            )

            # --- API Call ---
            raw_output, clean_output = model.simplify(prompt=final_prompt)
            
            # Store the results of this step
            step_outputs[step_name] = {
                'raw_output': raw_output,
                'clean_output': clean_output,
                'prompt': final_prompt
            }
            
            # The output of this step becomes the input for the next
            intermediate_text = clean_output

        # After all steps are done, store the final result for this document
        final_results.append({
            "text_id": doc['text_id'],
            "simplified": intermediate_text, # The output from the very last step
            "step_outputs": step_outputs
        })

    # --- Save final results to a single file ---
    simplifications_path = os.path.join(output_dir, 'simplifications.jsonl')
    with open(simplifications_path, 'w', encoding='utf-8') as f_out:
        for result in final_results:
            f_out.write(json.dumps(result) + '\n')
    print(f"\n   -> Final simplifications saved to: {simplifications_path}")

    # --- Run evaluation on the final output ---
    agg_scores, individual_scores = run_evaluation(
        system_output_path=simplifications_path,
        reference_data_path=global_config['paths']['evaluation_data']
    )

    # --- Save scores ---
    agg_scores_path = os.path.join(output_dir, 'scores.json')
    with open(agg_scores_path, 'w') as f: json.dump(agg_scores, f, indent=4)
    print(f"   -> Aggregated scores saved to: {agg_scores_path}")
    
    ind_scores_path = os.path.join(output_dir, 'individual_scores.jsonl')
    with open(ind_scores_path, 'w') as f:
        for score_record in individual_scores: f.write(json.dumps(score_record) + '\n')
    print(f"   -> Individual scores saved to: {ind_scores_path}")

    print(f"📊 Final Scores: {json.dumps(agg_scores, indent=2)}")