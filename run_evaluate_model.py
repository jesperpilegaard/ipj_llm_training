import argparse
import yaml
from src.train import train_model

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train config")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config")
    args = parser.parse_args()

    train_config = load_config(args.config)
    model_config = load_config(args.model_config)

    train_model(model_config["model_name"], train_config)
