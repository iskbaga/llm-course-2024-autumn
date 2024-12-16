import torch
from transformers import AutoTokenizer


def predict_by_token_id(logits: torch.Tensor, tokenizer: AutoTokenizer) -> int:
    """
    Determines the predicted choice based on the logits of the model's output.

    Args:
        logits (torch.Tensor): The logits output from the model, typically of shape (1, sequence_length, vocab_size).
        tokenizer (AutoTokenizer): The tokenizer used to encode the input prompt.

    Returns:
        int: The index of the predicted choice (0 for 'A', 1 for 'B', 2 for 'C', 3 for 'D').
    """
    ids = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in ['A', 'B', 'C', 'D']]
    return torch.argmax(logits[0, -1][ids]).item()


def get_choice_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """
    Calculates the average log probabilities of predicted tokens for a given sequence.


    Args:
        logits (torch.Tensor): A tensor of logits generated by the model, with shape (batch_size, sequence_length, vocab_size).
        input_ids (torch.Tensor): A tensor of input token IDs, with shape (batch_size, sequence_length).

    Returns:
         float: The average log probability of the predicted tokens.
    """
    probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return probs[:, :-1, :].gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).mean()
