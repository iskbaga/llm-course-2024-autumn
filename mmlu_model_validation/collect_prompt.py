def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """

    return f"""The following are multiple choice questions (with answers) about {sample['subject']}.
{sample['question']}
A. {sample['choices'][0]}
B. {sample['choices'][1]}
C. {sample['choices'][2]}
D. {sample['choices'][3]}
Answer:"""


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    prompts = []

    for ex in examples:
        prompt = create_prompt(ex)
        ans_idx = ex['answer']
        prompt += f" {chr(ord('A') + ans_idx)}"
        if add_full_example:
            prompt += f". {ex['choices'][ans_idx]}"
        prompts.append(prompt)

    # Add the sample's prompt without the answer
    prompts.append(create_prompt(sample))

    return "\n\n".join(prompts)

