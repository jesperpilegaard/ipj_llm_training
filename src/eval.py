import torch
import evaluate
import pandas as pd

# Funktion til at beregne metrikker som accuracy og f1
def compute_metrics(p):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    # Konverter logits til forudsigelser (vælg den højeste værdi)
    predictions = torch.argmax(torch.tensor(p.predictions), dim=-1).numpy()  
    labels = p.label_ids  # Sande labels

    # Beregn metrikkerne
    accuracy_score = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average='weighted')

    return {
        "accuracy": accuracy_score["accuracy"],  # Returner accuracy
        "f1": f1_score["f1"],  # Returner F1-score
    }

# Funktion til at gemme resultater til CSV
def save_results(results, log_path="experiments/logs.csv"):
    df = pd.DataFrame([results])
    # Tjek om der er data, før de gemmes
    print(f"Gemmer resultater: {results}")
    df.to_csv(log_path, mode='a', header=not open(log_path).read(1), index=False)
    print(f"Resultater gemt i {log_path}")



