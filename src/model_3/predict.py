"""
Model 3 – Standalone Inference Script
======================================
Run from repo root:
  python src/model_3/predict.py -i data/raw/example_input.txt -o output/model_3/predictions.txt

Optionally override saved weights path via --weights flag.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import torch

from src.config import Model3Config, Path3Config
from src.data_processing import load_example_input, format_output_line
from src.model_3.model import DateCVAE


def main():
    parser = argparse.ArgumentParser(description="Model 3 – Conditional VAE Inference")
    parser.add_argument("-i", "--input",   required=True, help="Path to input conditions file")
    parser.add_argument("-o", "--output",  required=True, help="Path to write predictions")
    parser.add_argument("--weights", default=None,        help="Override model weights path")
    args = parser.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model3_cfg = Model3Config()
    path3_cfg  = Path3Config()

    weights_path = args.weights or path3_cfg.weights_path

    model = DateCVAE(
        z_dim=model3_cfg.z_dim,
        cond_dim=model3_cfg.cond_dim,
        max_decade=model3_cfg.max_decade,
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    X, _ = load_example_input(args.input)
    X    = X.to(device)

    with torch.no_grad():
        Y_gen = model.sample(X, n_samples=1, device=device).cpu()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for i, cond in enumerate(X.cpu().tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Done. {len(X)} predictions → {args.output}")


if __name__ == "__main__":
    main()