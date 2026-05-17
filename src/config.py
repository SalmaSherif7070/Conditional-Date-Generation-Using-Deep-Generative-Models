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
    beta_warmup_frac: float = 0.5
    val_split: float = 0.2
    seed: int = 42


# ── Model 4 (Conditional Diffusion) ──────────────────────────────────────────

@dataclass
class Model4Config:
    cond_dim: int   = 128   # condition embedding dim (matches Models 1–3)
    max_decade: int = 300   # max decade index       (matches Models 1–3)
    emb_dim: int    = 32    # per-component date embedding dim (4×32 = 128 total)
    hidden_dim: int = 512   # residual MLP hidden width
    n_layers: int   = 6     # number of FiLM residual blocks
    time_dim: int   = 128   # sinusoidal time embedding dim
    T: int          = 500   # forward-process timesteps


@dataclass
class Train4Config:
    n_epochs: int      = 100
    batch_size: int    = 256
    lr: float          = 1e-4
    aux_weight: float  = 0.1   # weight of auxiliary CE loss
    ddim_steps: int    = 50    # DDIM inference steps
    val_split: float   = 0.2
    seed: int          = 42


# ── Paths ─────────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    data_path: str           = "data/raw/data.txt"
    example_input_path: str  = "data/raw/example_input.txt"
    output_dir: str          = "output/model_1"
    weights_path: str        = "models/model_1/weights.pt"
    figures_dir: str         = "output/model_1/figures"


@dataclass
class Path2Config:
    data_path: str           = "data/raw/data.txt"
    example_input_path: str  = "data/raw/example_input.txt"
    output_dir: str          = "output"
    generator_path: str      = "models/model_2/generator.pt"
    discriminator_path: str  = "models/model_2/discriminator.pt"
    encoder_path: str        = "models/model_2/encoder.pt"
    figures_dir: str         = "output/figures/model_2"


@dataclass
class Path3Config:
    data_path: str           = "data/raw/data.txt"
    example_input_path: str  = "data/raw/example_input.txt"
    output_dir: str          = "output/model_3"
    weights_path: str        = "models/model_3/weights.pt"
    figures_dir: str         = "output/model_3/figures"


@dataclass
class Path4Config:
    data_path: str           = "data/raw/data.txt"
    example_input_path: str  = "data/raw/example_input.txt"
    output_dir: str          = "output/model_4"
    weights_path: str        = "models/model_4/weights.pt"
    figures_dir: str         = "output/model_4/figures"