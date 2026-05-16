"""
Usage:
  # Model 1 (Autoregressive)
  python main.py train
  python main.py evaluate
  python main.py predict -i data/raw/example_input.txt -o output/predictions.txt

  # Model 2 (Conditional GAN)
  python main.py train2
  python main.py evaluate2
  python main.py predict2 -i data/raw/example_input.txt -o output/predictions_model2.txt
"""
import argparse
import os
import torch

from src.config import (
    ModelConfig, TrainConfig, PathConfig,
    Model2Config, Train2Config, Path2Config,
)
from src.data_processing import load_dataset, load_example_input, format_output_line

# Model 1
from src.model_1.model import AutoregressiveDateModel
from src.model_1.train import train as train_model1
from src.evaluate import evaluate_model as evaluate_model1
from src.visualization import plot_loss_curves, plot_condition_breakdown, plot_loss_log_scale

# Model 2
from src.model_2.model import DateGenerator, ConditionEncoder
from src.model_2.train import train as train_model2


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Model 1 helpers ────────────────────────────────────────────────────────────

def _load_model1(path_cfg: PathConfig, model_cfg: ModelConfig, device: str) -> AutoregressiveDateModel:
    model = AutoregressiveDateModel(model_cfg.cond_dim, model_cfg.max_decade).to(device)
    model.load_state_dict(torch.load(path_cfg.weights_path, map_location=device))
    model.eval()
    return model


# ── Model 2 helpers ────────────────────────────────────────────────────────────

class _Model2Wrapper:
    """
    Thin wrapper so Model 2's generator+encoder pair fits the same
    interface as Model 1's single model object expected by
    src/evaluate.py and src/visualization.py.
    """
    def __init__(self, generator, encoder):
        self.generator = generator
        self.encoder   = encoder

    def eval(self):
        self.generator.eval()
        self.encoder.eval()

    def sample(self, X, n_samples=1, device="cpu"):
        cond = self.encoder(X)
        return self.generator.sample(cond, X, device=device)


def _load_model2(path2_cfg: Path2Config, model2_cfg: Model2Config, device: str):
    encoder = ConditionEncoder(model2_cfg.cond_dim, model2_cfg.max_decade).to(device)
    encoder.load_state_dict(torch.load(path2_cfg.encoder_path, map_location=device))
    encoder.eval()

    generator = DateGenerator(model2_cfg.z_dim, model2_cfg.cond_dim, model2_cfg.max_decade).to(device)
    generator.load_state_dict(torch.load(path2_cfg.generator_path, map_location=device))
    generator.eval()

    return _Model2Wrapper(generator, encoder)


# ── Model 1 subcommands ────────────────────────────────────────────────────────

def cmd_train(args):
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    path_cfg  = PathConfig()
    device    = _device()

    print(f"Device : {device.upper()}")
    train_ds, val_ds, _, _ = load_dataset(path_cfg.data_path, train_cfg.val_split, train_cfg.seed)
    print(f"Train  : {len(train_ds)} samples  |  Val/Test : {len(val_ds)} samples")

    model, history = train_model1(train_ds, val_ds, model_cfg, train_cfg, device)

    os.makedirs(os.path.dirname(path_cfg.weights_path), exist_ok=True)
    torch.save(model.state_dict(), path_cfg.weights_path)
    print(f"\nWeights saved → {path_cfg.weights_path}")

    metrics = evaluate_model1(model, val_ds, device)

    plot_loss_curves(history, path_cfg.figures_dir)
    plot_loss_log_scale(history, path_cfg.figures_dir)
    plot_condition_breakdown(metrics, path_cfg.figures_dir)


def cmd_evaluate(args):
    model_cfg = ModelConfig()
    path_cfg  = PathConfig()
    device    = _device()

    _, val_ds, _, _ = load_dataset(path_cfg.data_path)
    model   = _load_model1(path_cfg, model_cfg, device)
    metrics = evaluate_model1(model, val_ds, device)
    plot_condition_breakdown(metrics, path_cfg.figures_dir)


def cmd_predict(args):
    model_cfg   = ModelConfig()
    path_cfg    = PathConfig()
    device      = _device()
    input_path  = args.input  or path_cfg.example_input_path
    output_path = args.output or os.path.join(path_cfg.output_dir, "predictions.txt")

    model = _load_model1(path_cfg, model_cfg, device)

    X, _ = load_example_input(input_path)
    with torch.no_grad():
        Y_gen = model.sample(X.to(device), n_samples=1, device=device).cpu()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for i, cond in enumerate(X.tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Predictions ({len(X)}) → {output_path}")


# ── Model 2 subcommands ────────────────────────────────────────────────────────

def cmd_train2(args):
    model2_cfg = Model2Config()
    train2_cfg = Train2Config()
    path2_cfg  = Path2Config()
    device     = _device()

    print(f"Device : {device.upper()}")
    train_ds, val_ds, _, _ = load_dataset(
        path2_cfg.data_path, train2_cfg.val_split, train2_cfg.seed)
    print(f"Train  : {len(train_ds)} samples  |  Val : {len(val_ds)} samples")

    # Optional warm-start from Model 1 encoder
    m1_path = (path2_cfg.model1_weights_path
               if os.path.exists(path2_cfg.model1_weights_path) else None)
    if m1_path:
        print(f"Warm-starting encoder from Model 1 → {m1_path}")

    G, D, encoder, history = train_model2(
        train_ds, val_ds, model2_cfg, train2_cfg, device,
        model1_weights_path=m1_path,
    )

    os.makedirs(os.path.dirname(path2_cfg.generator_path), exist_ok=True)
    torch.save(G.state_dict(),       path2_cfg.generator_path)
    torch.save(D.state_dict(),       path2_cfg.discriminator_path)
    torch.save(encoder.state_dict(), path2_cfg.encoder_path)
    print(f"\nGenerator     saved → {path2_cfg.generator_path}")
    print(f"Discriminator saved → {path2_cfg.discriminator_path}")
    print(f"Encoder       saved → {path2_cfg.encoder_path}")

    model2 = _Model2Wrapper(G, encoder)
    metrics = evaluate_model1(model2, val_ds, device)

    # Adapt history keys so shared visualization functions work
    # plot_loss_curves expects: train_loss, val_loss
    vis_history = {
        "epochs":     history["epochs"],
        "train_loss": history["g_loss"],   # generator loss as "train"
        "val_loss":   history["d_loss"],   # discriminator loss as "val"
    }
    # Reuse shared visualization — saves into model_2 figures dir
    plot_loss_curves(vis_history,         path2_cfg.figures_dir)
    plot_loss_log_scale(vis_history,      path2_cfg.figures_dir)
    plot_condition_breakdown(metrics, path2_cfg.figures_dir)


def cmd_evaluate2(args):
    model2_cfg = Model2Config()
    path2_cfg  = Path2Config()
    train2_cfg = Train2Config()
    device     = _device()

    _, val_ds, _, _ = load_dataset(
        path2_cfg.data_path, train2_cfg.val_split, train2_cfg.seed)
    model2  = _load_model2(path2_cfg, model2_cfg, device)
    metrics = evaluate_model1(model2, val_ds, device)
    plot_condition_breakdown(metrics, path2_cfg.figures_dir)


def cmd_predict2(args):
    model2_cfg  = Model2Config()
    path2_cfg   = Path2Config()
    device      = _device()
    input_path  = args.input  or path2_cfg.example_input_path
    output_path = args.output or os.path.join(path2_cfg.output_dir, "predictions_model2.txt")

    model2 = _load_model2(path2_cfg, model2_cfg, device)

    X, _ = load_example_input(input_path)
    X    = X.to(device)
    with torch.no_grad():
        Y_gen = model2.sample(X, device=device).cpu()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for i, cond in enumerate(X.cpu().tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Predictions ({len(X)}) → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Date Generator – Model 1 & Model 2")
    sub    = parser.add_subparsers(dest="command", required=True)

    # Model 1
    sub.add_parser("train",    help="Train Model 1 (Autoregressive) on data/raw/data.txt")
    sub.add_parser("evaluate", help="Evaluate Model 1 on the val/test split")
    pred_p = sub.add_parser("predict", help="Run Model 1 inference on an input file")
    pred_p.add_argument("-i", "--input",  default=None, help="Path to input conditions file")
    pred_p.add_argument("-o", "--output", default=None, help="Path to write predictions")

    # Model 2
    sub.add_parser("train2",    help="Train Model 2 (Conditional GAN) on data/raw/data.txt")
    sub.add_parser("evaluate2", help="Evaluate Model 2 on the val split")
    pred2_p = sub.add_parser("predict2", help="Run Model 2 inference on an input file")
    pred2_p.add_argument("-i", "--input",  default=None, help="Path to input conditions file")
    pred2_p.add_argument("-o", "--output", default=None, help="Path to write predictions")

    args = parser.parse_args()

    dispatch = {
        "train":     cmd_train,
        "evaluate":  cmd_evaluate,
        "predict":   cmd_predict,
        "train2":    cmd_train2,
        "evaluate2": cmd_evaluate2,
        "predict2":  cmd_predict2,
    }
    dispatch[args.command](args)