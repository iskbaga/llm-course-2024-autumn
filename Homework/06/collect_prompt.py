def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """

    question = sample.get("question", "")
    subject = sample.get("subject", "")
    choices = sample.get("choices", [])
    correct_answer_index = sample.get("answer_index", -1)

    choices_text = "\n".join([f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)])

    return f"Subject: {subject}\nQuestion: {question}\nChoices:{choices_text}\nAnswer: {chr(65 + correct_answer_index)}"


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
    example_texts = []
    for example in examples:
        question = example.get("question", "")
        subject = example.get("subject", "")
        choices = example.get("choices", [])
        answer_index = example.get("answer_index", -1)

        example_text = (f"Subject: {subject}\n"
                        f"Question: {question}\n"
                        f"Choices:{choices}\n"
                        f"Answer: {chr(65 + answer_index)}")
        example_texts.append(example_text)
    return f"{example_texts}"
