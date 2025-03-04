import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.dataset import load_data
from src.eval import compute_metrics, save_results

def train_model(model_name, train_config):
    train_data, eval_data = load_data(train_config["dataset"], model_name, train_config["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./experiments/",
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=train_config["learning_rate"],
        logging_dir="./logs/",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    trainer.train()
    
    # Evaluer modellen
    predictions = trainer.predict(eval_data)
    results = compute_metrics(predictions.predictions, predictions.label_ids)
    
    print("Evaluering:", results)
    
    # Gem resultater til CSV
    save_results(results)
