"""
Standalone inference script (assignment requirement).
Run from repo root:
  python src/model_1/predict.py -i data/raw/example_input.txt -o output/predictions.txt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import torch

from src.config import ModelConfig, PathConfig
from src.data_processing import load_example_input, format_output_line
from src.model_1.model import AutoregressiveDateModel


def main():
    parser = argparse.ArgumentParser(description="Model 1 – Inference")
    parser.add_argument("-i", "--input",  required=True, help="Path to input conditions file")
    parser.add_argument("-o", "--output", required=True, help="Path to write predictions")
    args = parser.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = ModelConfig()
    path_cfg  = PathConfig()

    model = AutoregressiveDateModel(model_cfg.cond_dim, model_cfg.max_decade).to(device)
    model.load_state_dict(torch.load(path_cfg.weights_path, map_location=device))
    model.eval()

    X, _ = load_example_input(args.input)
    with torch.no_grad():
        Y_gen = model.sample(X.to(device), n_samples=1, device=device).cpu()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for i, cond in enumerate(X.tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Done. {len(X)} predictions → {args.output}")


if __name__ == "__main__":
    main()