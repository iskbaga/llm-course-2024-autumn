from scripts.compute_reward import compute_reward

def eval_reward_model(reward_model, reward_tokenizer, test_dataset, target_label):
    chosen = [text['text'] for text in test_dataset if text['label'] == target_label]
    rejected = [text['text'] for text in test_dataset if text['label'] != target_label]

    summ = 0

    for x, y in zip(chosen, rejected):
        reward = compute_reward(reward_model, reward_tokenizer, [x, y])
        if reward[0] > reward[1]:
            summ += 1

    return summ/len(chosen)
