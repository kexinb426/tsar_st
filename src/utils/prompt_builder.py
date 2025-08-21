# src/utils/prompt_builder.py

# A custom dictionary that returns an empty string for any missing keys.
class SafeDict(dict):
    def __missing__(self, key):
        return ''

def build_prompt(
    prompt_template: str,
    target_cefr: str,
    document_text: str,
    previous_step_output: str, # <-- NEW: Input from the last step
    cefr_descriptions: dict,
    cefr_instructions: dict,
    cefr_translate_instructions: dict,
    cefr_simp_from_trans_instructions: dict,
    standard_examples: list,
    a2_examples: list,
    b1_examples: list
) -> str:
    """
    Constructs the final prompt string by filling all possible placeholders in a template.
    """
    def format_examples(examples):
        """Helper to format a list of example dicts into a string."""
        text = ""
        if not examples:
            return ""
        for ex in examples:
            text += f"Complex Document: {ex['original']}\n"
            text += f"Simplified Document: {ex['reference']}\n---\n"
        return text.strip()

    # Prepare all possible placeholder values
    cefr_description = cefr_descriptions.get(target_cefr, "")
    cefr_instruction = cefr_instructions.get(target_cefr, "")
    cefr_translate_instruction = cefr_translate_instructions.get(target_cefr, "")
    cefr_simp_from_trans_instruction = cefr_simp_from_trans_instructions.get(target_cefr, "")
    standard_examples_text = format_examples(standard_examples)
    a2_examples_text = format_examples(a2_examples)
    b1_examples_text = format_examples(b1_examples)

    # Use the SafeDict to hold all possible format keys.
    format_args = SafeDict({
        'target_cefr': target_cefr,
        'document_text': document_text,
        'previous_step_output': previous_step_output, # <-- Add new placeholder value
        'cefr_description': cefr_description,
        'cefr_instruction': cefr_instruction,
        'cefr_translate_instruction': cefr_translate_instruction,
        'cefr_simp_from_trans_instruction': cefr_simp_from_trans_instruction,
        'few_shot_examples': standard_examples_text,
        'a2_examples': a2_examples_text,
        'b1_examples': b1_examples_text
    })

    return prompt_template.format_map(format_args)