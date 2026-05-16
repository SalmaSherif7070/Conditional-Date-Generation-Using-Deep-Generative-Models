"""
Model 4 – Standalone Inference Script
======================================
Run from repo root:
  python src/model_4/predict.py -i data/raw/example_input.txt -o output/model_4/predictions.txt

Optionally override saved weights path via --weights flag, and control
Langevin sampling via --mcmc-steps / --step-size / --noise-std.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import torch

from src.config import Model4Config, Path4Config
from src.data_processing import load_example_input, format_output_line
from src.model_4.model import DateEBM


def main():
    parser = argparse.ArgumentParser(description="Model 4 – Energy-Based Model Inference")
    parser.add_argument("-i", "--input",      required=True,  help="Path to input conditions file")
    parser.add_argument("-o", "--output",     required=True,  help="Path to write predictions")
    parser.add_argument("--weights",          default=None,   help="Override model weights path")
    parser.add_argument("--mcmc-steps",  type=int,   default=None,
                        help="Number of Langevin steps (default: Train4Config.n_mcmc_steps)")
    parser.add_argument("--step-size",   type=float, default=None,
                        help="Langevin step size α (default: Train4Config.mcmc_step_size)")
    parser.add_argument("--noise-std",   type=float, default=None,
                        help="Langevin noise std σ (default: Train4Config.mcmc_noise)")
    args = parser.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model4_cfg = Model4Config()
    path4_cfg  = Path4Config()

    # Import Train4Config for defaults
    from src.config import Train4Config
    train4_cfg   = Train4Config()
    weights_path = args.weights      or path4_cfg.weights_path
    n_mcmc_steps = args.mcmc_steps   or train4_cfg.n_mcmc_steps
    step_size    = args.step_size    or train4_cfg.mcmc_step_size
    noise_std    = args.noise_std    or train4_cfg.mcmc_noise

    model = DateEBM(
        cond_dim=model4_cfg.cond_dim,
        max_decade=model4_cfg.max_decade,
        hidden_dim=model4_cfg.hidden_dim,
        n_layers=model4_cfg.n_layers,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    X, _ = load_example_input(args.input)
    X    = X.to(device)

    with torch.no_grad():
        Y_gen = model.sample(
            X,
            n_samples=1,
            device=device,
            n_mcmc_steps=n_mcmc_steps,
            step_size=step_size,
            noise_std=noise_std,
        ).cpu()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for i, cond in enumerate(X.cpu().tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Done. {len(X)} predictions → {args.output}")


if __name__ == "__main__":
    main()