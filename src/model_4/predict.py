"""
Model 4 – Diffusion Inference Script
======================================
Run from repo root:
  python src/model_4/predict.py -i data/raw/example_input.txt -o output/predictions.txt

Optional: --ddim-steps to control inference speed/quality tradeoff.
          More steps = better quality, slower inference (default: 50).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import torch

from src.config import Model4Config, Path4Config, Train4Config
from src.data_processing import load_example_input, format_output_line
from src.model_4.model import DateDiffusionModel


def main():
    parser = argparse.ArgumentParser(description="Model 4 – Diffusion Inference")
    parser.add_argument("-i", "--input",      required=True, help="Path to input conditions file")
    parser.add_argument("-o", "--output",     required=True, help="Path to write predictions")
    parser.add_argument("--weights",          default=None,  help="Override model weights path")
    parser.add_argument("--ddim-steps", type=int, default=None,
                        help="DDIM inference steps (default: Train4Config.ddim_steps)")
    args = parser.parse_args()

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    model4_cfg  = Model4Config()
    path4_cfg   = Path4Config()
    train4_cfg  = Train4Config()

    weights_path = args.weights    or path4_cfg.weights_path
    ddim_steps   = args.ddim_steps or train4_cfg.ddim_steps

    model = DateDiffusionModel(
        cond_dim=model4_cfg.cond_dim,
        max_decade=model4_cfg.max_decade,
        hidden_dim=model4_cfg.hidden_dim,
        n_layers=model4_cfg.n_layers,
        time_dim=model4_cfg.time_dim,
        T=model4_cfg.T,
        emb_dim=model4_cfg.emb_dim,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    X, _ = load_example_input(args.input)
    X    = X.to(device)

    with torch.no_grad():
        Y_gen = model.sample(X, n_samples=1, device=device, ddim_steps=ddim_steps).cpu()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for i, cond in enumerate(X.cpu().tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Done. {len(X)} predictions → {args.output}")


if __name__ == "__main__":
    main()