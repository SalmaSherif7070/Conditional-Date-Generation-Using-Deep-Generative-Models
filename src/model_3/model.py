"""
Conditional VAE (CVAE) Date Generator – Model 3
=================================================
Architecture
------------
  ConditionEncoder : (dow, month_idx, leap, decade) → 128-dim condition
                     Shared structure with Models 1 & 2.

  CVAE_Encoder     : Date + Condition → (mu, logvar) in latent space
                     Recognition network q(z | x, c).

  CVAE_Decoder     : Latent z + Condition → Autoregressive Date Logits
                     Generative network p(x | z, c).
                     Autoregressive: month → decade → year-unit → day.

Training objective : ELBO = Reconstruction Loss + β * KL Divergence
                     β is annealed from 0 → beta_max over the first 50% of training.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Calendar helpers (vectorised, matches Models 1 & 2) ───────────────────────

def is_leap(y: torch.Tensor) -> torch.Tensor:
    return ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)


def days_in_month(m: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
        [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        device=m.device, dtype=torch.long,
    )
    ms = torch.clamp(m, 1, 12)
    return table[ms] + ((ms == 2) & is_leap(y)).long()


def day_of_week(d: torch.Tensor, m: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tomohiko Sakamoto's algorithm. Returns 0=Mon … 6=Sun."""
    t = torch.tensor([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], device=d.device, dtype=torch.long)
    ms = torch.clamp(m, 1, 12)
    ya = y - (ms < 3).long()
    dow_sun = (ya + ya // 4 - ya // 100 + ya // 400 + t[ms - 1] + d) % 7
    return (dow_sun + 6) % 7  # shift so 0=Mon


# ── Condition Encoder ──────────────────────────────────────────────────────────

class ConditionEncoder(nn.Module):
    """
    Encodes (dow, month_idx, leap, decade) → cond_dim-dimensional vector.
    Identical interface to Models 1 & 2 for interoperability.
    """

    def __init__(self, cond_dim: int = 128, max_decade: int = 300):
        super().__init__()
        self.cond_dim   = cond_dim
        self.max_decade = max_decade

        self.emb_dow  = nn.Embedding(7,           16)
        self.emb_mon  = nn.Embedding(12,          16)
        self.emb_leap = nn.Embedding(2,            8)
        self.emb_dec  = nn.Embedding(max_decade,  32)

        # 16+16+8+32+4(cyclical) = 76 → cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(76, cond_dim),
            nn.ReLU(),
            nn.LayerNorm(cond_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, 4) — [dow, mon_idx, leap, decade]"""
        dow_rad = X[:, 0].float() * (2 * math.pi / 7.0)
        mon_rad = X[:, 1].float() * (2 * math.pi / 12.0)
        cyc = torch.stack([
            torch.sin(dow_rad), torch.cos(dow_rad),
            torch.sin(mon_rad), torch.cos(mon_rad),
        ], dim=-1)
        h = torch.cat([
            self.emb_dow(X[:, 0]),
            self.emb_mon(X[:, 1]),
            self.emb_leap(X[:, 2]),
            self.emb_dec(X[:, 3]),
            cyc,
        ], dim=-1)
        return self.cond_mlp(h)


# ── VAE Encoder (Recognition Model) ───────────────────────────────────────────

class CVAEEncoder(nn.Module):
    """
    Recognition network q(z | date, condition).
    Encodes a real date together with its condition into a latent Gaussian.
    """

    def __init__(self, z_dim: int = 64, cond_dim: int = 128, max_decade: int = 300):
        super().__init__()
        self.z_dim = z_dim

        # Date embeddings for the recognition network
        self.emb_d_day  = nn.Embedding(32,          32)   # indices 1-31
        self.emb_d_mon  = nn.Embedding(13,          32)   # indices 1-12
        self.emb_d_dec  = nn.Embedding(max_decade,  32)
        self.emb_d_unit = nn.Embedding(10,          32)

        date_dim = 32 * 4  # 128

        self.net = nn.Sequential(
            nn.Linear(date_dim + cond_dim, 512), nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 256),               nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
        )
        self.fc_mu     = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, Y: torch.Tensor, cond_emb: torch.Tensor):
        """
        Y: (B, 3) — [day, month, year]
        Returns (mu, logvar) each of shape (B, z_dim).
        """
        d, m, y = Y[:, 0], Y[:, 1], Y[:, 2]
        dec, unit = y // 10, y % 10

        date_emb = torch.cat([
            self.emb_d_day(d.clamp(1, 31)),
            self.emb_d_mon(m.clamp(1, 12)),
            self.emb_d_dec(dec.clamp(0, self.emb_d_dec.num_embeddings - 1)),
            self.emb_d_unit(unit),
        ], dim=-1)

        h = self.net(torch.cat([date_emb, cond_emb], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)


# ── VAE Decoder (Generative Model) ────────────────────────────────────────────

class CVAEDecoder(nn.Module):
    """
    Generative network p(date | z, condition).
    Autoregressive: month → decade → year-unit → day.
    Teacher-forcing during training; constrained sampling at inference.
    """

    def __init__(self, z_dim: int = 64, cond_dim: int = 128, max_decade: int = 300):
        super().__init__()
        self.z_dim      = z_dim
        self.max_decade = max_decade

        self.trunk = nn.Sequential(
            nn.Linear(z_dim + cond_dim, 512), nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Linear(512, 512),               nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
        )

        # AR context embeddings
        self.emb_g_mon  = nn.Embedding(13,          32)   # 1-12; index 0 unused
        self.emb_g_dec  = nn.Embedding(max_decade,  32)
        self.emb_g_unit = nn.Embedding(10,          32)

        # Prediction heads
        self.head_mon  = nn.Sequential(nn.Linear(512,        128), nn.LeakyReLU(0.2), nn.Linear(128, 12))
        self.head_dec  = nn.Sequential(nn.Linear(512 + 32,   256), nn.LeakyReLU(0.2), nn.Linear(256, max_decade))
        self.head_unit = nn.Sequential(nn.Linear(512 + 64,   256), nn.LeakyReLU(0.2), nn.Linear(256, 10))
        self.head_day  = nn.Sequential(nn.Linear(512 + 96,   256), nn.LeakyReLU(0.2), nn.Linear(256, 31))

    def forward(self, z: torch.Tensor, cond_emb: torch.Tensor, Y_true: torch.Tensor):
        """
        Teacher-forcing forward pass.
        Returns (logits_mon, logits_dec, logits_unit, logits_day).
        """
        h = self.trunk(torch.cat([z, cond_emb], dim=-1))

        # Month
        logits_mon = self.head_mon(h)
        mon_ctx    = self.emb_g_mon(Y_true[:, 1].clamp(1, 12))

        # Decade
        ctx2       = torch.cat([h, mon_ctx], dim=-1)
        logits_dec = self.head_dec(ctx2)
        dec_ctx    = self.emb_g_dec((Y_true[:, 2] // 10).clamp(0, self.max_decade - 1))

        # Year unit
        ctx3        = torch.cat([ctx2, dec_ctx], dim=-1)
        logits_unit = self.head_unit(ctx3)
        unit_ctx    = self.emb_g_unit(Y_true[:, 2] % 10)

        # Day
        ctx4       = torch.cat([ctx3, unit_ctx], dim=-1)
        logits_day = self.head_day(ctx4)

        return logits_mon, logits_dec, logits_unit, logits_day

    @torch.no_grad()
    def sample(self, cond_emb: torch.Tensor, X_cond: torch.Tensor,
               device: str = "cpu") -> torch.Tensor:
        """
        Constrained autoregressive sampling — mirrors Models 1 & 2.
        Decade is taken directly from condition; leap & DOW are enforced.
        Returns (B, 3) tensor of [day, month, year].
        """
        B = cond_emb.shape[0]
        z = torch.randn(B, self.z_dim, device=device)
        h = self.trunk(torch.cat([z, cond_emb], dim=-1))

        dow_cond  = X_cond[:, 0]
        leap_cond = X_cond[:, 2]
        dec_cond  = X_cond[:, 3]

        # 1. Month (greedy — fully determined by condition)
        mon_pred = torch.argmax(self.head_mon(h), dim=-1) + 1
        mon_ctx  = self.emb_g_mon(mon_pred)

        # 2. Decade (from condition, same as Models 1 & 2)
        dec_pred = dec_cond
        ctx2     = torch.cat([h, mon_ctx], dim=-1)
        dec_ctx  = self.emb_g_dec(dec_pred)

        # 3. Year unit (constrained by leap-year condition)
        ctx3        = torch.cat([ctx2, dec_ctx], dim=-1)
        logits_unit = self.head_unit(ctx3)
        y_cands     = dec_pred.unsqueeze(1) * 10 + torch.arange(10, device=device)
        valid_units = (is_leap(y_cands).long() == leap_cond.unsqueeze(1))
        logits_unit = logits_unit.masked_fill(~valid_units, float("-inf"))
        unit_pred   = torch.multinomial(F.softmax(logits_unit, dim=-1), 1).squeeze(-1)
        y_pred      = dec_pred * 10 + unit_pred
        unit_ctx    = self.emb_g_unit(unit_pred)

        # 4. Day (constrained by valid range + day-of-week)
        ctx4       = torch.cat([ctx3, unit_ctx], dim=-1)
        logits_day = self.head_day(ctx4)
        d_cands      = torch.arange(1, 32, device=device).unsqueeze(0).expand(B, 31)
        valid_bounds = d_cands <= days_in_month(mon_pred, y_pred).unsqueeze(1)
        dow_cands    = day_of_week(d_cands, mon_pred.unsqueeze(1), y_pred.unsqueeze(1))
        valid_dow    = dow_cands == dow_cond.unsqueeze(1)
        logits_day   = logits_day.masked_fill(~(valid_bounds & valid_dow), float("-inf"))
        probs        = F.softmax(logits_day, dim=-1)
        probs        = torch.nan_to_num(probs, nan=0.0)
        probs[probs.sum(dim=-1) == 0, 0] = 1.0   # fallback for impossible constraints
        day_pred     = torch.multinomial(probs, 1).squeeze(-1) + 1

        return torch.stack([day_pred, mon_pred, y_pred], dim=-1)


# ── Full CVAE ──────────────────────────────────────────────────────────────────

class DateCVAE(nn.Module):
    """
    Full Conditional VAE combining ConditionEncoder, CVAEEncoder, CVAEDecoder.
    Exposes a unified interface compatible with the project's evaluate / predict pipeline.
    """

    def __init__(self, z_dim: int = 64, cond_dim: int = 128, max_decade: int = 300):
        super().__init__()
        self.z_dim   = z_dim
        self.encoder = CVAEEncoder(z_dim, cond_dim, max_decade)
        self.decoder = CVAEDecoder(z_dim, cond_dim, max_decade)
        # ConditionEncoder stored separately so it can be saved / loaded independently
        self.condition_encoder = ConditionEncoder(cond_dim, max_decade)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Full ELBO forward pass (training).
        Returns (logits_tuple, mu, logvar).
        """
        cond_emb      = self.condition_encoder(X)
        mu, logvar    = self.encoder(Y, cond_emb)
        z             = self.reparameterize(mu, logvar)
        logits        = self.decoder(z, cond_emb, Y)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, X: torch.Tensor, n_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """
        Constrained inference — matches Model 1 / Model 2 sample() signature.
        Returns (B * n_samples, 3) tensor of [day, month, year].
        """
        X_rep    = X.repeat_interleave(n_samples, dim=0)
        cond_emb = self.condition_encoder(X_rep)
        return self.decoder.sample(cond_emb, X_rep, device=device)


# ── ELBO Loss ─────────────────────────────────────────────────────────────────

def cvae_loss(logits, Y_true: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 1.0):
    """
    ELBO = Reconstruction Loss + β * KL Divergence.
    Reconstruction uses cross-entropy over each AR head independently.
    β-annealing controlled externally (passed in per epoch).
    """
    logits_mon, logits_dec, logits_unit, logits_day = logits
    d_true, m_true, y_true = Y_true[:, 0], Y_true[:, 1], Y_true[:, 2]

    CE = nn.CrossEntropyLoss(reduction="sum")
    recon = (
        CE(logits_mon,  m_true - 1)         +
        CE(logits_dec,  y_true // 10)        +
        CE(logits_unit, y_true % 10)         +
        CE(logits_day,  d_true - 1)
    ) / Y_true.size(0)

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Y_true.size(0)

    return recon + beta * kl, recon, kl