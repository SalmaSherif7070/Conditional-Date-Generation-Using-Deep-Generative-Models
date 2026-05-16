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

  # Model 3 (Conditional VAE)
  python main.py train3
  python main.py evaluate3
  python main.py predict3 -i data/raw/example_input.txt -o output/model_3/predictions.txt

  # Model 4 (Energy-Based Model)
  python main.py train4
  python main.py evaluate4
  python main.py predict4 -i data/raw/example_input.txt -o output/model_4/predictions.txt
"""
import argparse
import os
import torch

from src.config import (
    ModelConfig, TrainConfig, PathConfig,
    Model2Config, Train2Config, Path2Config,
    Model3Config, Train3Config, Path3Config,
    Model4Config, Train4Config, Path4Config,
)
from src.data_processing import load_dataset, load_example_input, format_output_line

# Model 1
from src.model_1.model import AutoregressiveDateModel
from src.model_1.train import train as train_model1
from src.model_1.evaluate import evaluate_model as evaluate_model1
from src.model_1.visualization import plot_loss_curves, plot_loss_log_scale, plot_condition_breakdown

# Model 2
from src.model_2.model import DateGenerator, ConditionEncoder
from src.model_2.train import train as train_model2
from src.model_2.evaluate import evaluate_model as evaluate_model2
from src.model_2.visualization import (
    plot_loss_curves as plot_loss_curves2,
    plot_loss_log_scale as plot_loss_log_scale2,
    plot_condition_breakdown as plot_condition_breakdown2,
)

# Model 3
from src.model_3.model import DateCVAE
from src.model_3.train import train as train_model3
from src.model_3.evaluate import evaluate_model as evaluate_model3
from src.model_3.visualization import (
    plot_loss_curves as plot_loss_curves3,
    plot_loss_log_scale as plot_loss_log_scale3,
    plot_condition_breakdown as plot_condition_breakdown3,
)

# Model 4
from src.model_4.model import DateEBM
from src.model_4.train import train as train_model4
from src.model_4.evaluate import evaluate_model as evaluate_model4
from src.model_4.visualization import (
    plot_loss_curves as plot_loss_curves4,
    plot_loss_log_scale as plot_loss_log_scale4,
    plot_condition_breakdown as plot_condition_breakdown4,
)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Model 1 helpers ────────────────────────────────────────────────────────────

def _load_model1(path_cfg: PathConfig, model_cfg: ModelConfig, device: str) -> AutoregressiveDateModel:
    model = AutoregressiveDateModel(model_cfg.cond_dim, model_cfg.max_decade).to(device)
    model.load_state_dict(torch.load(path_cfg.weights_path, map_location=device))
    model.eval()
    return model


# ── Model 2 helpers ────────────────────────────────────────────────────────────

def _load_model2(path2_cfg: Path2Config, model2_cfg: Model2Config, device: str):
    encoder = ConditionEncoder(model2_cfg.cond_dim, model2_cfg.max_decade).to(device)
    encoder.load_state_dict(torch.load(path2_cfg.encoder_path, map_location=device))
    encoder.eval()

    generator = DateGenerator(model2_cfg.z_dim, model2_cfg.cond_dim, model2_cfg.max_decade).to(device)
    generator.load_state_dict(torch.load(path2_cfg.generator_path, map_location=device))
    generator.eval()

    return generator, encoder


# ── Model 3 helpers ────────────────────────────────────────────────────────────

def _load_model3(path3_cfg: Path3Config, model3_cfg: Model3Config, device: str) -> DateCVAE:
    model = DateCVAE(
        z_dim=model3_cfg.z_dim,
        cond_dim=model3_cfg.cond_dim,
        max_decade=model3_cfg.max_decade,
    ).to(device)
    model.load_state_dict(torch.load(path3_cfg.weights_path, map_location=device))
    model.eval()
    return model


# ── Model 4 helpers ────────────────────────────────────────────────────────────

def _load_model4(path4_cfg: Path4Config, model4_cfg: Model4Config, device: str) -> DateEBM:
    model = DateEBM(
        cond_dim=model4_cfg.cond_dim,
        max_decade=model4_cfg.max_decade,
        hidden_dim=model4_cfg.hidden_dim,
        n_layers=model4_cfg.n_layers,
    ).to(device)
    model.load_state_dict(torch.load(path4_cfg.weights_path, map_location=device))
    model.eval()
    return model


# ── Model 1 subcommands ────────────────────────────────────────────────────────

def cmd_train(args):
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    path_cfg  = PathConfig()
    device    = _device()

    print(f"Device : {device.upper()}")
    train_ds, val_ds, _, _ = load_dataset(path_cfg.data_path, train_cfg.val_split, train_cfg.seed)
    print(f"Train  : {len(train_ds)} samples  |  Val : {len(val_ds)} samples")

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

    G, D, encoder, history = train_model2(
        train_ds, val_ds, model2_cfg, train2_cfg, device,
    )

    os.makedirs(os.path.dirname(path2_cfg.generator_path), exist_ok=True)
    torch.save(G.state_dict(),       path2_cfg.generator_path)
    torch.save(D.state_dict(),       path2_cfg.discriminator_path)
    torch.save(encoder.state_dict(), path2_cfg.encoder_path)
    print(f"\nGenerator     saved → {path2_cfg.generator_path}")
    print(f"Discriminator saved → {path2_cfg.discriminator_path}")
    print(f"Encoder       saved → {path2_cfg.encoder_path}")

    metrics = evaluate_model2(G, encoder, val_ds, device)

    plot_loss_curves2(history,      path2_cfg.figures_dir)
    plot_loss_log_scale2(history,   path2_cfg.figures_dir)
    plot_condition_breakdown2(metrics, path2_cfg.figures_dir)


def cmd_evaluate2(args):
    model2_cfg = Model2Config()
    path2_cfg  = Path2Config()
    train2_cfg = Train2Config()
    device     = _device()

    _, val_ds, _, _ = load_dataset(
        path2_cfg.data_path, train2_cfg.val_split, train2_cfg.seed)
    generator, encoder = _load_model2(path2_cfg, model2_cfg, device)
    metrics = evaluate_model2(generator, encoder, val_ds, device)
    plot_condition_breakdown2(metrics, path2_cfg.figures_dir)


def cmd_predict2(args):
    model2_cfg  = Model2Config()
    path2_cfg   = Path2Config()
    device      = _device()
    input_path  = args.input  or path2_cfg.example_input_path
    output_path = args.output or os.path.join(path2_cfg.output_dir, "predictions_model2.txt")

    generator, encoder = _load_model2(path2_cfg, model2_cfg, device)

    X, _ = load_example_input(input_path)
    X    = X.to(device)
    with torch.no_grad():
        cond_emb = encoder(X)
        Y_gen    = generator.sample(cond_emb, X, device=device).cpu()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for i, cond in enumerate(X.cpu().tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Predictions ({len(X)}) → {output_path}")


# ── Model 3 subcommands ────────────────────────────────────────────────────────

def cmd_train3(args):
    model3_cfg = Model3Config()
    train3_cfg = Train3Config()
    path3_cfg  = Path3Config()
    device     = _device()

    print(f"Device : {device.upper()}")
    train_ds, val_ds, _, _ = load_dataset(
        path3_cfg.data_path, train3_cfg.val_split, train3_cfg.seed)
    print(f"Train  : {len(train_ds)} samples  |  Val : {len(val_ds)} samples")

    model, history = train_model3(train_ds, val_ds, model3_cfg, train3_cfg, device)

    os.makedirs(os.path.dirname(path3_cfg.weights_path), exist_ok=True)
    torch.save(model.state_dict(), path3_cfg.weights_path)
    print(f"\nWeights saved → {path3_cfg.weights_path}")

    metrics = evaluate_model3(model, val_ds, device)

    plot_loss_curves3(history,         path3_cfg.figures_dir)
    plot_loss_log_scale3(history,      path3_cfg.figures_dir)
    plot_condition_breakdown3(metrics, path3_cfg.figures_dir)


def cmd_evaluate3(args):
    model3_cfg = Model3Config()
    path3_cfg  = Path3Config()
    train3_cfg = Train3Config()
    device     = _device()

    _, val_ds, _, _ = load_dataset(
        path3_cfg.data_path, train3_cfg.val_split, train3_cfg.seed)
    model   = _load_model3(path3_cfg, model3_cfg, device)
    metrics = evaluate_model3(model, val_ds, device)
    plot_condition_breakdown3(metrics, path3_cfg.figures_dir)


def cmd_predict3(args):
    model3_cfg  = Model3Config()
    path3_cfg   = Path3Config()
    device      = _device()
    input_path  = args.input  or path3_cfg.example_input_path
    output_path = args.output or os.path.join(path3_cfg.output_dir, "predictions.txt")

    model = _load_model3(path3_cfg, model3_cfg, device)

    X, _ = load_example_input(input_path)
    X    = X.to(device)
    with torch.no_grad():
        Y_gen = model.sample(X, n_samples=1, device=device).cpu()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for i, cond in enumerate(X.cpu().tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Predictions ({len(X)}) → {output_path}")


# ── Model 4 subcommands ────────────────────────────────────────────────────────

def cmd_train4(args):
    model4_cfg = Model4Config()
    train4_cfg = Train4Config()
    path4_cfg  = Path4Config()
    device     = _device()

    print(f"Device : {device.upper()}")
    train_ds, val_ds, _, _ = load_dataset(
        path4_cfg.data_path, train4_cfg.val_split, train4_cfg.seed)
    print(f"Train  : {len(train_ds)} samples  |  Val : {len(val_ds)} samples")

    model, history = train_model4(train_ds, val_ds, model4_cfg, train4_cfg, device)

    os.makedirs(os.path.dirname(path4_cfg.weights_path), exist_ok=True)
    torch.save(model.state_dict(), path4_cfg.weights_path)
    print(f"\nWeights saved → {path4_cfg.weights_path}")

    metrics = evaluate_model4(model, val_ds, device)

    plot_loss_curves4(history,         path4_cfg.figures_dir)
    plot_loss_log_scale4(history,      path4_cfg.figures_dir)
    plot_condition_breakdown4(metrics, path4_cfg.figures_dir)


def cmd_evaluate4(args):
    model4_cfg = Model4Config()
    path4_cfg  = Path4Config()
    train4_cfg = Train4Config()
    device     = _device()

    _, val_ds, _, _ = load_dataset(
        path4_cfg.data_path, train4_cfg.val_split, train4_cfg.seed)
    model   = _load_model4(path4_cfg, model4_cfg, device)
    metrics = evaluate_model4(model, val_ds, device)
    plot_condition_breakdown4(metrics, path4_cfg.figures_dir)


def cmd_predict4(args):
    model4_cfg  = Model4Config()
    path4_cfg   = Path4Config()
    train4_cfg  = Train4Config()
    device      = _device()
    input_path  = args.input  or path4_cfg.example_input_path
    output_path = args.output or os.path.join(path4_cfg.output_dir, "predictions.txt")

    model = _load_model4(path4_cfg, model4_cfg, device)

    X, _ = load_example_input(input_path)
    X    = X.to(device)
    with torch.no_grad():
        Y_gen = model.sample(
            X,
            n_samples=1,
            device=device,
            n_mcmc_steps=train4_cfg.n_mcmc_steps,
            step_size=train4_cfg.mcmc_step_size,
            noise_std=train4_cfg.mcmc_noise,
        ).cpu()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for i, cond in enumerate(X.cpu().tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Predictions ({len(X)}) → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Date Generator – Models 1, 2, 3 & 4")
    sub    = parser.add_subparsers(dest="command", required=True)

    # Model 1
    sub.add_parser("train",    help="Train Model 1 (Autoregressive)")
    sub.add_parser("evaluate", help="Evaluate Model 1 on the val split")
    pred_p = sub.add_parser("predict", help="Run Model 1 inference")
    pred_p.add_argument("-i", "--input",  default=None)
    pred_p.add_argument("-o", "--output", default=None)

    # Model 2
    sub.add_parser("train2",    help="Train Model 2 (Conditional GAN)")
    sub.add_parser("evaluate2", help="Evaluate Model 2 on the val split")
    pred2_p = sub.add_parser("predict2", help="Run Model 2 inference")
    pred2_p.add_argument("-i", "--input",  default=None)
    pred2_p.add_argument("-o", "--output", default=None)

    # Model 3
    sub.add_parser("train3",    help="Train Model 3 (Conditional VAE)")
    sub.add_parser("evaluate3", help="Evaluate Model 3 on the val split")
    pred3_p = sub.add_parser("predict3", help="Run Model 3 inference")
    pred3_p.add_argument("-i", "--input",  default=None)
    pred3_p.add_argument("-o", "--output", default=None)

    # Model 4
    sub.add_parser("train4",    help="Train Model 4 (Energy-Based Model)")
    sub.add_parser("evaluate4", help="Evaluate Model 4 on the val split")
    pred4_p = sub.add_parser("predict4", help="Run Model 4 inference")
    pred4_p.add_argument("-i", "--input",  default=None)
    pred4_p.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    dispatch = {
        "train":     cmd_train,
        "evaluate":  cmd_evaluate,
        "predict":   cmd_predict,
        "train2":    cmd_train2,
        "evaluate2": cmd_evaluate2,
        "predict2":  cmd_predict2,
        "train3":    cmd_train3,
        "evaluate3": cmd_evaluate3,
        "predict3":  cmd_predict3,
        "train4":    cmd_train4,
        "evaluate4": cmd_evaluate4,
        "predict4":  cmd_predict4,
    }
    dispatch[args.command](args)