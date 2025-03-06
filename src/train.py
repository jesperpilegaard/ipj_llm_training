from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.dataset import load_data
from src.eval import compute_metrics, save_results
import warnings
warnings.filterwarnings("ignore")

def train_model(model_name, train_config):
    # Indlæs data
    train_data, eval_data = load_data(train_config["dataset"], model_name, train_config["max_length"])

    # Hent model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Definer træningsparametre
    training_args = TrainingArguments(
        output_dir="./experiments/",
        num_train_epochs=train_config["epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        evaluation_strategy="epoch",  # Evaluer efter hver epoch
        save_strategy="epoch",  # Gem model efter hver epoch
        learning_rate=train_config["learning_rate"],
        logging_dir="./logs/",
        load_best_model_at_end=True,
    )

    # Initialiser Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,  # Brug compute_metrics funktionen
    )

    # Træn modellen
    trainer.train()
    
    # Evaluer modellen
    predictions = trainer.predict(eval_data)
    results = compute_metrics(predictions)  # Beregn metrikker for evalueringsdata

    print("Evaluering:", results)
    
    # Gem resultater til CSV
    save_results(results)

# Kald funktion til at træne modellen
if __name__ == "__main__":
    train_config = {
        "dataset": "imdb",  # Brug IMDb-datasæt
        "dataset_size": 1000,  # Brug kun de første 1000 samples
        "max_length": 128,  # Maksimal længde for tokens
        "epochs": 3,  # Antal træningsepochs
        "batch_size": 2, # Batch størrelse
        "learning_rate": 2e-5  # Læringsrate
    }
    
    # Træn model
    train_model("distilbert-base-uncased", train_config)
