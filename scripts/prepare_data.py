# prepare_data.py
import json
import os

# --- Configuration ---
# The file the organizers gave you
source_data_file = 'data/raw/tsar2025_trialdata.jsonl'

# The directories we'll create to store our organized data
input_dir = 'data/input'
reference_dir = 'data/references'

# The new files we'll create
input_file_path = os.path.join(input_dir, 'documents.jsonl')
reference_file_path = os.path.join(reference_dir, 'human_simplifications.jsonl')
# --- End Configuration ---

def prepare_data():
    """
    Reads the source data and splits it into two files:
    1. An input file for our models.
    2. A reference file for our evaluation script.
    """
    # Create the directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)

    print(f"Reading from '{source_data_file}'...")

    # Open all three files at once
    with open(source_data_file, 'r', encoding='utf-8') as f_source, \
         open(input_file_path, 'w', encoding='utf-8') as f_input, \
         open(reference_file_path, 'w', encoding='utf-8') as f_reference:

        for line in f_source:
            # Load the data from one line
            data = json.loads(line)

            # 1. Prepare the data for the AI model (the "input")
            # We need the ID, the original text, and the target level
            input_data = {
                "text_id": data["text_id"],
                "original": data["original"],
                "target_cefr": data["target_cefr"]
            }
            # Write it to the input file
            f_input.write(json.dumps(input_data) + '\n')

            # 2. Prepare the data for the judge (the "reference")
            # The judge only needs the ID and the human-written simplification
            reference_data = {
                "text_id": data["text_id"],
                "reference": data["reference"]
            }
            # Write it to the reference file
            f_reference.write(json.dumps(reference_data) + '\n')

    print(f"Successfully created input file at: '{input_file_path}'")
    print(f"Successfully created reference file at: '{reference_file_path}'")

if __name__ == "__main__":
    prepare_data()