# main.py
import os
import json
import yaml
import random
from dotenv import load_dotenv
from collections import defaultdict

# Import our custom modules
from src.pipeline_runner import run_pipeline # <-- Import the new runner

def read_jsonl(filepath):
    """Helper function to read a .jsonl file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_json(filepath):
    """Helper function to read a standard .json file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_all_prompt_templates(prompts_dir):
    """Loads all .txt files from the prompts directory into a dictionary."""
    templates = {}
    for filename in os.listdir(prompts_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(prompts_dir, filename), 'r', encoding='utf-8') as f:
                templates[filename] = f.read()
    return templates

def main():
    """
    Main function to load configuration and data, then trigger the pipeline runner.
    """
    # 1. Load the main configuration file
    print("🚀 Starting experiment pipeline...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set the random seed for reproducibility if provided
    seed = config.get('random_seed', None)
    if seed is not None:
        print(f"-> Using random seed: {seed}")
        random.seed(seed)

    paths = config['paths']
    
    # 2. Load all data and prompt assets ONCE into a single dictionary
    print("-> Loading all data and prompt assets...")
    
    # Load all datasets
    input_documents = read_jsonl(paths['input_docs'])
    scored_references = read_jsonl(paths.get('scored_references', ''))
    difficulty_data = read_jsonl(paths.get('source_difficulty_data', ''))

    # Pre-sort examples and create lookup maps
    examples_by_level = defaultdict(list)
    for doc in scored_references: # Use scored references for confidence
        examples_by_level[doc['target_cefr'].upper()].append(doc)
        
    difficulty_map = {doc['text_id']: doc.get('difficulty_vector') for doc in difficulty_data}

    # Load all prompt assets
    all_data = {
        'input_documents': input_documents,
        'examples_by_level': examples_by_level,
        'difficulty_map': difficulty_map,
        'cefr_descriptions': load_json(os.path.join(paths['prompt_assets_dir'], 'cefr_descriptions.json')),
        'cefr_instructions': load_json(os.path.join(paths['prompt_assets_dir'], 'cefr_instructions.json')),
        'cefr_translate_instructions': load_json(os.path.join(paths['prompt_assets_dir'], 'cefr_translate_instructions.json')),
        'cefr_simp_from_trans_instructions': load_json(os.path.join(paths['prompt_assets_dir'], 'cefr_simp_from_trans_instructions.json')),
        'prompt_templates': load_all_prompt_templates(paths['prompts_dir'])
    }
    
    print("-> All assets loaded successfully.")

    # 3. Loop through each pipeline defined in the config and run it
    if 'pipelines' not in config:
        print("❌ Error: 'pipelines' key not found in config.yaml. Please update your config structure.")
        return
        
    for pipeline_config in config['pipelines']:
        run_pipeline(pipeline_config, global_config=config, all_data=all_data)

if __name__ == "__main__":
    # Load environment variables from .env file for API keys
    load_dotenv()
    main()

# TORCH_COMPILE_DISABLE=1 python main.py