# main.py
import os
import json
import yaml
import random
import time
import datetime
from collections import defaultdict
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Import our custom modules
from src.models.gpt import GPTModel
from src.models.claude import ClaudeModel
from src.models.gemma import GemmaModel
from src.evaluation.metrics import run_evaluation
from src.utils.prompt_builder import build_prompt

# Load environment variables from .env file (for the API key)
load_dotenv()

def read_jsonl(filepath):
    """Helper function to read a .jsonl file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_json(filepath):
    """Helper function to read a standard .json file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_prompt_template(prompts_dir, filename):
    """Helper function to load a prompt from a file."""
    with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as f:
        return f.read()

def main():
    """
    Main function to run the simplification and evaluation pipeline.
    """
    # 1. Load the main configuration file
    print("🚀 Starting experiment pipeline...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    seed = config.get('random_seed', None)
    if seed is not None:
        print(f"-> Using random seed: {seed}")
        random.seed(seed)

    paths = config['paths']
    
    # 2. Load all data and prompt assets ONCE
    print("-> Loading all data and prompt assets...")
    input_documents = read_jsonl(paths['input_docs'])
    # Load the new dataset that includes confidence scores
    full_scored_dataset = read_jsonl(paths['scored_references'])
    cefr_descriptions = load_json(os.path.join(paths['prompt_assets_dir'], 'cefr_descriptions.json'))

    # Pre-sort the scored dataset by CEFR level for efficient example lookup
    examples_by_level = defaultdict(list)
    for doc in full_scored_dataset:
        examples_by_level[doc['target_cefr'].upper()].append(doc)

    # 3. Loop through each experiment defined in the config
    for experiment in config['experiments']:
        exp_name = experiment['name']
        model_name = experiment['model']
        prompt_template_file = experiment['prompt_template']
        num_few_shot = experiment.get('few_shot', 0)
        # Get the selection strategy, defaulting to 'random'
        selection_strategy = experiment.get('few_shot_selection', 'random')
        contrastive_config = experiment.get('contrastive', None)
        sample_size = experiment.get('sample_size', None)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_run_name = f"{exp_name}_{timestamp}"

        print(f"\n🔥 Running Experiment: {exp_run_name} (Selection: {selection_strategy}) 🔥")

        # 4. Initialize the correct model
        model_config = config['models'][model_name]
        model_type = model_config.get('type')

        if model_type == 'gpt':
            model = GPTModel(api_model_name=model_config['api_model_name'], temperature=model_config.get('temperature', 1.0))
        elif model_type == 'claude':
            model = ClaudeModel(api_model_name=model_config['api_model_name'], temperature=model_config.get('temperature', 1.0))
        elif model_type == 'gemma':
            model = GemmaModel(api_model_name=model_config['api_model_name'], temperature=model_config.get('temperature', 1.0))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 5. Load the base prompt template
        prompt_template_text = load_prompt_template(paths['prompts_dir'], prompt_template_file)
        
        # 6. Prepare results directory and save config
        output_dir = os.path.join(paths['results_dir'], exp_run_name)
        os.makedirs(output_dir, exist_ok=True)
        config_copy_path = os.path.join(output_dir, 'config.yaml')
        with open(config_copy_path, 'w') as f_config:
            yaml.dump(config, f_config, default_flow_style=False)
        print(f"✅ Config saved to: {config_copy_path}")

        simplifications_path = os.path.join(output_dir, 'simplifications.jsonl')
        
        # 7. Handle sampling and filtering
        docs_to_process = input_documents
        if contrastive_config:
            docs_to_process = [d for d in input_documents if d['target_cefr'].upper() in ['A2', 'B1']]
        if sample_size is not None:
            docs_to_process = docs_to_process[:sample_size]
        
        print(f"-> Simplifying {len(docs_to_process)} documents...")

        with open(simplifications_path, 'w', encoding='utf-8') as f_out:
            for i, doc in enumerate(docs_to_process):
                print(f"--> Processing document {i+1}/{len(docs_to_process)} (ID: {doc['text_id']})")
                target_cefr_upper = doc['target_cefr'].upper()
                
                # --- EXAMPLE SELECTION LOGIC ---
                standard_examples, a2_examples, b1_examples = [], [], []
                
                if num_few_shot > 0:
                    possible = examples_by_level.get(target_cefr_upper, [])
                    eligible = [ex for ex in possible if ex['text_id'] != doc['text_id']]
                    
                    if len(eligible) >= num_few_shot:
                        if selection_strategy == 'confidence':
                            # Sort by confidence score (highest first) and take the top N
                            eligible.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
                            standard_examples = eligible[:num_few_shot]
                        else: # Default to 'random'
                            standard_examples = random.sample(eligible, num_few_shot)
                
                if contrastive_config:
                    num_examples = contrastive_config.get('num_examples', 1)
                    a2_eligible = [ex for ex in examples_by_level['A2'] if ex['text_id'] != doc['text_id']]
                    b1_eligible = [ex for ex in examples_by_level['B1'] if ex['text_id'] != doc['text_id']]

                    if selection_strategy == 'confidence':
                        # Sort both lists by confidence and take the top N
                        a2_eligible.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
                        b1_eligible.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
                        a2_examples = a2_eligible[:num_examples]
                        b1_examples = b1_eligible[:num_examples]
                    else: # Default to 'random'
                        a2_examples = random.sample(a2_eligible, min(num_examples, len(a2_eligible)))
                        b1_examples = random.sample(b1_eligible, min(num_examples, len(b1_eligible)))

                # --- UNIFIED PROMPT BUILDING ---
                final_prompt = build_prompt(
                    prompt_template=prompt_template_text,
                    target_cefr=target_cefr_upper,
                    document_text=doc['original'],
                    cefr_descriptions=cefr_descriptions,
                    standard_examples=standard_examples,
                    a2_examples=a2_examples,
                    b1_examples=b1_examples
                )

                # --- API Call with Retry Logic ---
                processed_successfully = False
                backoff_delay = 1.0
                max_backoff = 64.0

                while not processed_successfully:
                    try:
                        raw_model_output, simplified_text = model.simplify(prompt=final_prompt)
                        processed_successfully = True
                        print(f"    ✅ Success!")
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'ThrottlingException':
                            print(f"    🐌 Throttled! Retrying after {backoff_delay:.2f} seconds...")
                            time.sleep(backoff_delay)
                            backoff_delay = min(max_backoff, backoff_delay * 2 + random.uniform(0, 1))
                        else:
                            print(f"    ❌ Unrecoverable AWS error: {e}")
                            raw_model_output, simplified_text = "", ""
                            break
                
                # --- Save Results ---
                example_ids = [ex['text_id'] for ex in (standard_examples + a2_examples + b1_examples)]
                output_record = {
                    "text_id": doc['text_id'],
                    "simplified": simplified_text,
                    "full_prompt": final_prompt,
                    "raw_model_output": raw_model_output,
                    "few_shot_example_ids": example_ids
                }
                f_out.write(json.dumps(output_record) + '\n')
                
                time.sleep(1)
        
        print(f"✅ Simplifications saved to: {simplifications_path}")

        # 8. Run evaluation
        agg_scores, individual_scores = run_evaluation(
            system_output_path=simplifications_path,
            reference_data_path=paths['evaluation_data']
        )

        # 9. Save scores
        agg_scores_path = os.path.join(output_dir, 'scores.json')
        with open(agg_scores_path, 'w') as f:
            json.dump(agg_scores, f, indent=4)
        print(f"✅ Aggregated scores saved to: {agg_scores_path}")
        
        ind_scores_path = os.path.join(output_dir, 'individual_scores.jsonl')
        with open(ind_scores_path, 'w') as f:
            for score_record in individual_scores:
                f.write(json.dumps(score_record) + '\n')
        print(f"✅ Individual scores saved to: {ind_scores_path}")

        print(f"📊 Final Scores: {json.dumps(agg_scores, indent=2)}")

if __name__ == "__main__":
    main()

# TORCH_COMPILE_DISABLE=1 python main.py