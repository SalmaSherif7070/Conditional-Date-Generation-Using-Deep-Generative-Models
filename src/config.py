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
class Model2Config:
    cond_dim: int = 128
    max_decade: int = 300
    z_dim: int = 64


@dataclass
class Train2Config:
    n_epochs: int = 200
    batch_size: int = 256
    lr_g: float = 1e-4
    lr_d: float = 4e-4
    n_critic: int = 2
    lambda_gp: float = 10.0
    tau_start: float = 2.0
    tau_end: float = 0.5
    val_split: float = 0.2
    seed: int = 42


@dataclass
class PathConfig:
    data_path: str = "data/raw/data.txt"
    example_input_path: str = "data/raw/example_input.txt"
    output_dir  = "output/model_2"
    weights_path: str = "models/model_1/weights.pt"
    figures_dir: str = "output/model_2/figures"


@dataclass
class Path2Config:
    data_path: str = "data/raw/data.txt"
    example_input_path: str = "data/raw/example_input.txt"
    output_dir: str = "output"
    generator_path: str = "models/model_2/generator.pt"
    discriminator_path: str = "models/model_2/discriminator.pt"
    encoder_path: str = "models/model_2/encoder.pt"
    figures_dir: str = "output/figures/model_2"