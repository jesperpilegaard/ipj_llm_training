from datasets import load_dataset
from transformers import AutoTokenizer

def load_data(dataset_name, tokenizer_name, max_length):
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return dataset["train"], dataset["test"]