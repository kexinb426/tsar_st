# src/utils/prompt_builder.py

# A custom dictionary that returns an empty string for any missing keys.
# This makes the .format_map() method robust to templates that don't use all placeholders.
class SafeDict(dict):
    def __missing__(self, key):
        return ''

def build_prompt(
    prompt_template: str,
    target_cefr: str,
    document_text: str,
    cefr_descriptions: dict,
    standard_examples: list,
    a2_examples: list,
    b1_examples: list
) -> str:
    """
    Constructs the final prompt string by filling all possible placeholders in a template.
    It safely handles templates that don't use all placeholders.
    """
    def format_examples(examples):
        """Helper to format a list of example dicts into a string."""
        text = ""
        # If the list of examples is empty, return a clear debugging message.
        if not examples:
            return "DEBUG: No examples were provided for this placeholder."
        for ex in examples:
            text += f"Complex Document: {ex['original']}\n"
            text += f"Simplified Document: {ex['reference']}\n---\n"
        return text.strip()

    # Prepare all possible placeholder values
    cefr_description = cefr_descriptions.get(target_cefr, "")
    standard_examples_text = format_examples(standard_examples)
    a2_examples_text = format_examples(a2_examples)
    b1_examples_text = format_examples(b1_examples)

    # Use the SafeDict to hold all possible format keys.
    # This dictionary will provide an empty string for any key that
    # is requested by .format_map() but not explicitly defined here.
    format_args = SafeDict({
        'target_cefr': target_cefr,
        'document_text': document_text,
        'cefr_description': cefr_description,
        'few_shot_examples': standard_examples_text,
        'a2_examples': a2_examples_text,
        'b1_examples': b1_examples_text
    })

    # .format_map() is safer than .format() because it can be used with a
    # custom dictionary subclass (like SafeDict) to gracefully handle
    # missing placeholders without raising a KeyError.
    return prompt_template.format_map(format_args)