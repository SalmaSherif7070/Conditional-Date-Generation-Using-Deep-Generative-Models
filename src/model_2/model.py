"""
Conditional GAN Date Generator (Model 2)
=========================================
Architecture
------------
  ConditionEncoder : (dow, month_idx, leap, decade) → 128-dim embedding
                     Identical structure to AutoregressiveDateModel.encode_cond()
                     so weights can be warm-started from a trained Model 1.

  DateGenerator    : noise(z_dim) + cond_emb(128) → date logits
                     Autoregressive heads: month → decade → year-unit → day.
                     Uses Gumbel-softmax for differentiable discrete sampling.

  DateDiscriminator: date_emb + cond_emb → real/fake score
                     Spectral-norm layers for WGAN-GP stability.

Training objective: WGAN-GP + auxiliary KL supervision on every generator head.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Calendar helpers (vectorised, matches Model 1) ─────────────────────────────

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
    Architecture is identical to AutoregressiveDateModel.encode_cond() so that
    weights can be transferred from a pre-trained Model 1 checkpoint.
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

    def load_from_model1(self, ar_state_dict: dict):
        """
        Copy encoder weights from a saved Model 1 state dict.
        Model 1 uses emb_mon / emb_dec; this module uses the same names.
        """
        mapping = {
            "emb_dow":  "emb_dow",
            "emb_mon":  "emb_mon",
            "emb_leap": "emb_leap",
            "emb_dec":  "emb_dec",
            "cond_mlp": "cond_mlp",
        }
        own_sd = self.state_dict()
        for own_key, ar_key in mapping.items():
            for param_name in ("weight", "bias"):
                src = f"{ar_key}.{param_name}"
                dst = f"{own_key}.{param_name}"
                # LayerNorm uses weight/bias too
                if src in ar_state_dict and dst in own_sd:
                    own_sd[dst].copy_(ar_state_dict[src])
        # Handle cond_mlp sub-layers (Sequential index keys)
        for k, v in ar_state_dict.items():
            if k.startswith("cond_mlp."):
                if k in own_sd:
                    own_sd[k].copy_(v)
        self.load_state_dict(own_sd)
        print("[ConditionEncoder] Weights transferred from Model 1 checkpoint.")


# ── Generator ─────────────────────────────────────────────────────────────────

class DateGenerator(nn.Module):
    """
    noise(z_dim) + cond_emb(cond_dim) → date component logits.
    Autoregressive: month → decade → year-unit → day.
    Gumbel-softmax enables end-to-end gradient flow during GAN training.
    """

    def __init__(self, z_dim: int = 64, cond_dim: int = 128, max_decade: int = 300):
        super().__init__()
        self.z_dim      = z_dim
        self.max_decade = max_decade

        in_dim = z_dim + cond_dim  # 192

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 256),   nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
        )

        # AR context embeddings (inside generator)
        self.emb_g_mon  = nn.Embedding(13,          32)   # 1-12; index 0 unused
        self.emb_g_dec  = nn.Embedding(max_decade,  32)
        self.emb_g_unit = nn.Embedding(10,          32)

        # Prediction heads (mirror Model 1 architecture)
        self.head_mon  = nn.Sequential(nn.Linear(256,       64),  nn.LeakyReLU(0.2), nn.Linear(64,  12))
        self.head_dec  = nn.Sequential(nn.Linear(256 + 32,  128), nn.LeakyReLU(0.2), nn.Linear(128, max_decade))
        self.head_unit = nn.Sequential(nn.Linear(256 + 64,  128), nn.LeakyReLU(0.2), nn.Linear(128, 10))
        self.head_day  = nn.Sequential(nn.Linear(256 + 96,  256), nn.LeakyReLU(0.2), nn.Linear(256, 31))

    def forward(self, z: torch.Tensor, cond_emb: torch.Tensor,
                tau: float = 1.0, hard: bool = False):
        """
        Teacher-free forward pass using Gumbel-softmax for differentiability.
        Returns (logits_mon, logits_dec, logits_unit, logits_day,
                 soft_mon, soft_dec, soft_unit).
        """
        h = self.trunk(torch.cat([z, cond_emb], dim=-1))

        # Month
        logits_mon = self.head_mon(h)
        soft_mon   = F.gumbel_softmax(logits_mon, tau=tau, hard=hard)
        mon_ctx    = soft_mon @ self.emb_g_mon.weight[1:13]          # skip pad-0

        # Decade
        ctx2       = torch.cat([h, mon_ctx], dim=-1)
        logits_dec = self.head_dec(ctx2)
        soft_dec   = F.gumbel_softmax(logits_dec, tau=tau, hard=hard)
        dec_ctx    = soft_dec @ self.emb_g_dec.weight

        # Year unit
        ctx3        = torch.cat([ctx2, dec_ctx], dim=-1)
        logits_unit = self.head_unit(ctx3)
        soft_unit   = F.gumbel_softmax(logits_unit, tau=tau, hard=hard)
        unit_ctx    = soft_unit @ self.emb_g_unit.weight

        # Day
        ctx4       = torch.cat([ctx3, unit_ctx], dim=-1)
        logits_day = self.head_day(ctx4)

        return logits_mon, logits_dec, logits_unit, logits_day, soft_mon, soft_dec, soft_unit

    @torch.no_grad()
    def sample(self, cond_emb: torch.Tensor, X_cond: torch.Tensor,
               device: str = "cpu") -> torch.Tensor:
        """
        Constrained greedy+stochastic sampling.
        Mirrors Model 1's sample() — decade taken from condition, leap & DOW enforced.
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

        # 2. Decade (from condition, same as Model 1)
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


# ── Discriminator ─────────────────────────────────────────────────────────────

class DateDiscriminator(nn.Module):
    """
    Scores a (day, month, year) date given the condition embedding.
    Spectral normalisation on all linear layers for WGAN-GP stability.
    Exposes both a hard (integer) and soft (differentiable) forward path.
    """

    def __init__(self, cond_dim: int = 128, max_decade: int = 300):
        super().__init__()
        self.max_decade = max_decade
        SL = nn.utils.spectral_norm

        self.emb_d_day  = nn.Embedding(32,          32)   # indices 1-31
        self.emb_d_mon  = nn.Embedding(13,          32)   # indices 1-12
        self.emb_d_dec  = nn.Embedding(max_decade,  32)
        self.emb_d_unit = nn.Embedding(10,          32)

        date_dim = 32 * 4  # 128

        self.net = nn.Sequential(
            SL(nn.Linear(date_dim + cond_dim, 256)), nn.LeakyReLU(0.2),
            SL(nn.Linear(256,                 128)), nn.LeakyReLU(0.2),
            SL(nn.Linear(128,                  64)), nn.LeakyReLU(0.2),
            SL(nn.Linear(64,                    1)),
        )

    def _embed_date(self, Y: torch.Tensor) -> torch.Tensor:
        """Y: (B, 3) integer tensor [day, month, year]."""
        d, m, y = Y[:, 0], Y[:, 1], Y[:, 2]
        dec, unit = y // 10, y % 10
        return torch.cat([
            self.emb_d_day(d.clamp(1, 31)),
            self.emb_d_mon(m.clamp(1, 12)),
            self.emb_d_dec(dec.clamp(0, self.max_decade - 1)),
            self.emb_d_unit(unit),
        ], dim=-1)

    def forward(self, Y: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """Hard (integer) forward — used for real samples."""
        return self.net(torch.cat([self._embed_date(Y), cond_emb], dim=-1))

    def forward_soft(self, soft_mon: torch.Tensor, soft_dec: torch.Tensor,
                     soft_unit: torch.Tensor, soft_day: torch.Tensor,
                     cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Differentiable forward — used for generator gradient updates.
        Accepts soft one-hot vectors; bypasses embedding look-ups via matmul.
        """
        d_mon  = soft_mon  @ self.emb_d_mon.weight[1:13]
        d_dec  = soft_dec  @ self.emb_d_dec.weight
        d_unit = soft_unit @ self.emb_d_unit.weight
        d_day  = soft_day  @ self.emb_d_day.weight[1:32]
        date_emb = torch.cat([d_day, d_mon, d_dec, d_unit], dim=-1)
        return self.net(torch.cat([date_emb, cond_emb], dim=-1))