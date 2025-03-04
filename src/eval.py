import evaluate
import torch
import pandas as pd

def compute_metrics(predictions, labels):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    preds = torch.argmax(torch.tensor(predictions), dim=1).numpy()

    results = {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }
    
    return results

def save_results(results, log_path="experiments/logs.csv"):
    df = pd.DataFrame([results])
    df.to_csv(log_path, mode='a', header=not open(log_path).read(1), index=False)
    print(f"Resultater gemt i {log_path}")

