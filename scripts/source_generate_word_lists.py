# scripts/generate_word_lists.py
import os
import json
import spacy
from cefrpy import CEFRSpaCyAnalyzer, CEFRLevel

def _read_jsonl(filepath):
    """Helper function to read a .jsonl file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def generate_word_lists(
    input_path='data/raw/tsar2025_trialdata.jsonl',
    output_path='data/source_with_word_lists.jsonl'
):
    """
    Reads the full dataset, analyzes the original texts with cefrpy,
    and saves a new dataset enriched with two lists:
    1. 'avoid_words': Words more difficult than the target CEFR level.
    2. 'named_entities': Words identified as named entities to be preserved.
    """
    print("🚀 Starting word list generation process...")

    # 1. Initialize cefrpy and spaCy
    print("   -> Loading spaCy and cefrpy models...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("❌ Error: spaCy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        return

    # Define the named entity types we want to identify and preserve
    ENTITY_TYPES_TO_SKIP_CEFR = {
        'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT',
        'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE'
    }
    text_analyzer = CEFRSpaCyAnalyzer(entity_types_to_skip=ENTITY_TYPES_TO_SKIP_CEFR)
    
    # Map CEFR string levels to numerical values for comparison
    CEFR_TO_INT = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

    # 2. Load the dataset
    full_dataset = _read_jsonl(input_path)
    print(f"   -> Found {len(full_dataset)} documents to analyze.")

    # 3. Process each document
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, doc_data in enumerate(full_dataset):
            print(f"--> Processing document {i+1}/{len(full_dataset)} (ID: {doc_data['text_id']})")
            
            original_text = doc_data['original']
            target_level_str = doc_data['target_cefr'].upper()
            target_level_int = CEFR_TO_INT.get(target_level_str, 0)

            # Analyze the document with cefrpy
            spacy_doc = nlp(original_text)
            tokens = text_analyzer.analize_doc(spacy_doc)

            avoid_words = set()
            named_entities = set()

            for token_data in tokens:
                word, _, is_skipped, level, _, _ = token_data
                
                # Clean the word
                clean_word = word.lower().strip()
                if not clean_word or not clean_word.isalpha():
                    continue

                if is_skipped:
                    # If cefrpy skipped it because it's a named entity, add it to our list
                    named_entities.add(clean_word)
                elif level and round(level) > target_level_int:
                    # If the word's level is higher than the target, add it to the avoid list
                    avoid_words.add(clean_word)

            # Create the new record
            new_doc = doc_data.copy()
            # Convert sets to sorted lists for consistent output
            new_doc['avoid_words'] = sorted(list(avoid_words))
            new_doc['named_entities'] = sorted(list(named_entities))
            
            f_out.write(json.dumps(new_doc) + '\n')

    print(f"\n✅ Successfully created file with word lists at: '{output_path}'")


if __name__ == "__main__":
    generate_word_lists()