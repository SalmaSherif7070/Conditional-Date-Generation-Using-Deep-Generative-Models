"""
Usage:
  python main.py train
  python main.py evaluate
  python main.py predict -i data/raw/example_input.txt -o output/predictions.txt
"""
import argparse
import os
import torch

from src.config import ModelConfig, TrainConfig, PathConfig
from src.data_processing import load_dataset, load_example_input, format_output_line
from src.model_1.model import AutoregressiveDateModel
from src.model_1.train import train
from src.evaluate import evaluate_model
from src.visualization import plot_loss_curves, plot_condition_breakdown, plot_loss_log_scale


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(path_cfg: PathConfig, model_cfg: ModelConfig, device: str) -> AutoregressiveDateModel:
    model = AutoregressiveDateModel(model_cfg.cond_dim, model_cfg.max_decade).to(device)
    model.load_state_dict(torch.load(path_cfg.weights_path, map_location=device))
    model.eval()
    return model


# ── Subcommands ───────────────────────────────────────────────────

def cmd_train(args):
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    path_cfg  = PathConfig()
    device    = _device()

    print(f"Device : {device.upper()}")
    train_ds, val_ds, _, _ = load_dataset(path_cfg.data_path, train_cfg.val_split, train_cfg.seed)
    print(f"Train  : {len(train_ds)} samples  |  Val/Test : {len(val_ds)} samples")

    model, history = train(train_ds, val_ds, model_cfg, train_cfg, device)

    os.makedirs(os.path.dirname(path_cfg.weights_path), exist_ok=True)
    torch.save(model.state_dict(), path_cfg.weights_path)
    print(f"\nWeights saved → {path_cfg.weights_path}")

    metrics = evaluate_model(model, val_ds, device)

    plot_loss_curves(history, path_cfg.figures_dir)
    plot_loss_log_scale(history, path_cfg.figures_dir)
    plot_condition_breakdown(metrics, path_cfg.figures_dir)


def cmd_evaluate(args):
    model_cfg = ModelConfig()
    path_cfg  = PathConfig()
    device    = _device()

    _, val_ds, _, _ = load_dataset(path_cfg.data_path)
    model   = _load_model(path_cfg, model_cfg, device)
    metrics = evaluate_model(model, val_ds, device)
    plot_condition_breakdown(metrics, path_cfg.figures_dir)


def cmd_predict(args):
    model_cfg   = ModelConfig()
    path_cfg    = PathConfig()
    device      = _device()
    input_path  = args.input  or path_cfg.example_input_path
    output_path = args.output or os.path.join(path_cfg.output_dir, "predictions.txt")

    model = _load_model(path_cfg, model_cfg, device)

    X, _ = load_example_input(input_path)
    with torch.no_grad():
        Y_gen = model.sample(X.to(device), n_samples=1, device=device).cpu()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for i, cond in enumerate(X.tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Predictions ({len(X)}) → {output_path}")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Date Generator – Model 1")
    sub    = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train",    help="Train model 1 on data/raw/data.txt")
    sub.add_parser("evaluate", help="Evaluate saved model on the val/test split")

    pred_p = sub.add_parser("predict", help="Run inference on an input file")
    pred_p.add_argument("-i", "--input",  default=None, help="Path to input conditions file")
    pred_p.add_argument("-o", "--output", default=None, help="Path to write predictions")

    args = parser.parse_args()

    if   args.command == "train":    cmd_train(args)
    elif args.command == "evaluate": cmd_evaluate(args)
    elif args.command == "predict":  cmd_predict(args)