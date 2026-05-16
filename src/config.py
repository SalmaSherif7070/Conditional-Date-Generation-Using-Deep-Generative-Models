from dataclasses import dataclass


@dataclass
class ModelConfig:
    cond_dim: int = 128
    max_decade: int = 300


@dataclass
class TrainConfig:
    n_epochs: int = 300
    batch_size: int = 256
    lr: float = 2e-3
    val_split: float = 0.2
    seed: int = 42


@dataclass
class PathConfig:
    data_path: str = "data/raw/data.txt"
    example_input_path: str = "data/raw/example_input.txt"
    output_dir: str = "output"
    weights_path: str = "models/model_1/weights.pt"
    figures_dir: str = "output/figures"