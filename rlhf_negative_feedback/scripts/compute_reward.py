from torch import Tensor, no_grad

def compute_reward(reward_model, reward_tokenizer, texts: list[str], device='cpu') -> Tensor:

    inputs = reward_tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {key: value for key, value in inputs.items()}

    with no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits

    return logits[:, 0]
