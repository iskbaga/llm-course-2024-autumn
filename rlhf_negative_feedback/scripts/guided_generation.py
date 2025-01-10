from scripts.compute_reward import compute_reward
import torch
def generate_with_reward_guidance(
        main_model, main_tokenizer,
        reward_model, reward_tokenizer,
        N=16,
        device='cpu',
    ):

    inputs = main_tokenizer(["It was"] * N, return_tensors='pt').to(device)
    candidates = main_model.generate(**inputs, max_new_tokens= 100, do_sample=True)
    text = []

    for candidate in candidates:
        text.append(main_tokenizer.decode(candidate.flatten().cpu().numpy().tolist()))

    return text[torch.argmax(compute_reward(reward_model, reward_tokenizer, text))]