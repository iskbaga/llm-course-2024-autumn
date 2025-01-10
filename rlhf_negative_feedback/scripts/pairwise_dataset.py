from torch.utils.data import Dataset


class IMDBPairwiseDataset(Dataset):
    def __init__(self, imdb, tokenizer, accepted_label):
        super().__init__()
        self.tokenizer = tokenizer
        self.chosen_texts = [item['text'] for item in imdb if item['label'] == accepted_label]
        self.rejected_texts = [item['text'] for item in imdb if item['label'] != accepted_label]
        self.num_rejected = len(self.rejected_texts)
        assert self.chosen_texts, f"no texts with label {accepted_label}"
        self.column_names = [
            'input_ids_chosen', 'attention_mask_chosen',
            'input_ids_rejected', 'attention_mask_rejected'
        ]

    def __len__(self):
        return len(self.chosen_texts) * len(self.rejected_texts)

    def __getitem__(self, index: int):
        chosen_text = self.chosen_texts[index // self.num_rejected]
        rejected_text = self.rejected_texts[index % self.num_rejected]
        chosen_encoding = self.tokenizer(
            chosen_text, truncation=True, return_tensors="pt"
        )
        rejected_encoding = self.tokenizer(
            rejected_text, truncation=True, return_tensors="pt"
        )

        return {
            "input_ids_chosen": chosen_encoding["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_encoding["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected_encoding["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_encoding["attention_mask"].squeeze(0),
        }