from datasets import load_dataset
from transformers import AutoTokenizer

def load_data(dataset_name, tokenizer_name, max_length):
    print(f"Indlæser dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    text_column = "text"
    if "review" in dataset["train"].column_names:
        text_column = "review"

    def tokenize(batch):
        return tokenizer(batch[text_column], padding="max_length", truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    
    label_column = "label" if "label" in dataset["train"].column_names else "labels"
    
    dataset.set_format("torch", columns=["input_ids", "attention_mask", label_column])

    print("Eksempel efter tokenization:", dataset["train"][0])

    return dataset["train"], dataset["test"]

# Kald funktionen med de rigtige argumenter
train_data, eval_data = load_data("imdb", "distilbert-base-uncased", 128)
