"""
Model 2 – Standalone Inference Script
======================================
Run from repo root:
  python src/model_2/predict.py -i data/raw/example_input.txt -o output/predictions_model2.txt

Optionally override saved weights paths via --generator, --encoder flags.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import torch

from src.config import Model2Config, Path2Config
from src.data_processing import load_example_input, format_output_line
from src.model_2.model import DateGenerator, ConditionEncoder


def main():
    parser = argparse.ArgumentParser(description="Model 2 – Conditional GAN Inference")
    parser.add_argument("-i", "--input",       required=True, help="Path to input conditions file")
    parser.add_argument("-o", "--output",      required=True, help="Path to write predictions")
    parser.add_argument("--generator",  default=None, help="Override generator weights path")
    parser.add_argument("--encoder",    default=None, help="Override encoder weights path")
    args = parser.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model2_cfg = Model2Config()
    path2_cfg  = Path2Config()

    gen_path = args.generator or path2_cfg.generator_path
    enc_path = args.encoder   or path2_cfg.encoder_path

    encoder = ConditionEncoder(model2_cfg.cond_dim, model2_cfg.max_decade).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    encoder.eval()

    generator = DateGenerator(model2_cfg.z_dim, model2_cfg.cond_dim, model2_cfg.max_decade).to(device)
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    generator.eval()

    X, _ = load_example_input(args.input)
    X    = X.to(device)

    with torch.no_grad():
        cond_emb = encoder(X)
        Y_gen    = generator.sample(cond_emb, X, device=device).cpu()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for i, cond in enumerate(X.cpu().tolist()):
            f.write(format_output_line(cond, Y_gen[i].tolist()) + "\n")

    print(f"Done. {len(X)} predictions → {args.output}")


if __name__ == "__main__":
    main()