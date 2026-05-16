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
class Model3Config:
    cond_dim: int = 128
    max_decade: int = 300
    z_dim: int = 64


@dataclass
class Train3Config:
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    beta_max: float = 0.5
    beta_warmup_frac: float = 0.5   # fraction of epochs for β warm-up
    val_split: float = 0.2
    seed: int = 42


@dataclass
class Model4Config:
    cond_dim: int = 128       # condition embedding dimension (matches Models 1–3)
    max_decade: int = 300     # maximum decade index (matches Models 1–3)
    hidden_dim: int = 512     # energy network hidden width
    n_layers: int = 4         # number of residual blocks in the energy network


@dataclass
class Train4Config:
    n_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-4
    # Langevin / MCMC sampling hyper-parameters
    n_mcmc_steps: int = 60        # Langevin steps per negative sample
    mcmc_step_size: float = 0.1   # Langevin step size (α)
    mcmc_noise: float = 0.005     # noise std added each Langevin step
    replay_buffer_size: int = 10_000  # persistent replay buffer capacity
    replay_prob: float = 0.95         # probability of re-using a buffer sample
    # Regularisation
    l2_reg: float = 1.0           # coefficient for E(x)² regulariser
    grad_clip: float = 1.0
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


@dataclass
class Path3Config:
    data_path: str = "data/raw/data.txt"
    example_input_path: str = "data/raw/example_input.txt"
    output_dir: str = "output/model_3"
    weights_path: str = "models/model_3/weights.pt"
    figures_dir: str = "output/model_3/figures"


@dataclass
class Path4Config:
    data_path: str = "data/raw/data.txt"
    example_input_path: str = "data/raw/example_input.txt"
    output_dir: str = "output/model_4"
    weights_path: str = "models/model_4/weights.pt"
    figures_dir: str = "output/model_4/figures"