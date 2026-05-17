"""
Conditional Diffusion Model (DDPM / DDIM) Date Generator – Model 4
===================================================================
Architecture
------------
  ConditionEncoder   : (dow, month_idx, leap, decade) → cond_dim embedding.
                       Identical to Models 1–3.

  SinusoidalTimeEmb  : timestep t → time_dim via sinusoidal encoding + MLP.

  DateEmbedder       : (day, month, year) → continuous embedding for diffusion.
                       Day, month, decade, year-unit each get a learned embedding.

  DenoisingNet       : (x_t, cond_emb, time_emb) → x0_pred + discrete logits.
                       Residual MLP with FiLM conditioning (scale+shift).
                       Auxiliary discrete heads provide strong gradient signal.

  DateDiffusionModel : Full model. Forward process adds Gaussian noise (linear
                       beta schedule, T steps). Trained with "predict-x0" MSE +
                       auxiliary cross-entropy on discrete heads.
                       Inference via DDIM (50 steps default) — ~10× faster than
                       DDPM. Final discrete projection enforces all calendar
                       constraints identically to Models 1–3.

Training objective
------------------
  L = MSE(x0_pred, x0_embed) + aux_weight * CE(discrete_heads, Y)

Sampling (DDIM)
---------------
  x_T ~ N(0, I)  →  DDIM reverse for ddim_steps  →  project logits to
  valid (day, month, year) via constraint masks (dow, leap, month, decade).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Calendar helpers (identical to Models 1–3) ────────────────────────────────

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
    return (dow_sun + 6) % 7


# ── Condition Encoder ──────────────────────────────────────────────────────────

class ConditionEncoder(nn.Module):
    def __init__(self, cond_dim: int = 128, max_decade: int = 300):
        super().__init__()
        self.emb_dow  = nn.Embedding(7,           16)
        self.emb_mon  = nn.Embedding(12,          16)
        self.emb_leap = nn.Embedding(2,            8)
        self.emb_dec  = nn.Embedding(max_decade,  32)
        self.cond_mlp = nn.Sequential(
            nn.Linear(76, cond_dim), nn.ReLU(), nn.LayerNorm(cond_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        dow_rad = X[:, 0].float() * (2 * math.pi / 7.0)
        mon_rad = X[:, 1].float() * (2 * math.pi / 12.0)
        cyc = torch.stack([
            dow_rad.sin(), dow_rad.cos(), mon_rad.sin(), mon_rad.cos(),
        ], dim=-1)
        h = torch.cat([
            self.emb_dow(X[:, 0]), self.emb_mon(X[:, 1]),
            self.emb_leap(X[:, 2]), self.emb_dec(X[:, 3]), cyc,
        ], dim=-1)
        return self.cond_mlp(h)


# ── Sinusoidal Time Embedding ─────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps → (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# ── Date Embedder ─────────────────────────────────────────────────────────────

class DateEmbedder(nn.Module):
    """Maps integer date (day, month, year) → continuous embedding for diffusion."""

    def __init__(self, max_decade: int = 300, emb_dim: int = 32):
        super().__init__()
        self.emb_dim    = emb_dim
        self.max_decade = max_decade
        self.emb_day  = nn.Embedding(32,          emb_dim)
        self.emb_mon  = nn.Embedding(13,          emb_dim)
        self.emb_dec  = nn.Embedding(max_decade,  emb_dim)
        self.emb_unit = nn.Embedding(10,          emb_dim)

    @property
    def out_dim(self) -> int:
        return self.emb_dim * 4

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """Y: (B, 3) integer [day, month, year] → (B, 4*emb_dim)."""
        d, m, y = Y[:, 0], Y[:, 1], Y[:, 2]
        return torch.cat([
            self.emb_day(d.clamp(1, 31)),
            self.emb_mon(m.clamp(1, 12)),
            self.emb_dec((y // 10).clamp(0, self.max_decade - 1)),
            self.emb_unit(y % 10),
        ], dim=-1)


# ── FiLM Residual Block ───────────────────────────────────────────────────────

class _FiLMBlock(nn.Module):
    """Residual block with FiLM (scale + shift) conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm      = nn.LayerNorm(dim)
        self.net       = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        self.cond_proj = nn.Linear(cond_dim, dim * 2)   # → (scale, shift)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + self.net(h)


# ── Denoising Network ─────────────────────────────────────────────────────────

class DenoisingNet(nn.Module):
    """
    Predicts the clean embedding x0 from noisy embedding x_t, condition, and timestep.
    Also outputs discrete logits (month, decade, unit, day) as auxiliary heads —
    these provide strong gradient signal and are used directly at inference.
    """

    def __init__(
        self,
        date_dim: int,
        cond_dim: int,
        time_dim: int,
        hidden_dim: int,
        n_layers: int,
        max_decade: int,
    ):
        super().__init__()
        film_cond = cond_dim + time_dim

        self.input_proj  = nn.Linear(date_dim, hidden_dim)
        self.res_blocks  = nn.ModuleList(
            [_FiLMBlock(hidden_dim, film_cond) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, date_dim)   # x0 embedding pred

        # Discrete logit heads
        self.head_mon  = nn.Linear(hidden_dim, 12)
        self.head_dec  = nn.Linear(hidden_dim, max_decade)
        self.head_unit = nn.Linear(hidden_dim, 10)
        self.head_day  = nn.Linear(hidden_dim, 31)

    def forward(
        self,
        x_t: torch.Tensor,
        cond_emb: torch.Tensor,
        time_emb: torch.Tensor,
    ):
        """Returns (x0_pred, logits_mon, logits_dec, logits_unit, logits_day)."""
        film_cond = torch.cat([cond_emb, time_emb], dim=-1)
        h = self.input_proj(x_t)
        for block in self.res_blocks:
            h = block(h, film_cond)
        x0_pred    = self.output_proj(h)
        return x0_pred, self.head_mon(h), self.head_dec(h), self.head_unit(h), self.head_day(h)


# ── Full Diffusion Model ──────────────────────────────────────────────────────

class DateDiffusionModel(nn.Module):
    """
    Conditional DDPM/DDIM for date generation.
    Exposes the project-standard  .sample(X, n_samples, device)  interface.
    """

    def __init__(
        self,
        cond_dim: int    = 128,
        max_decade: int  = 300,
        hidden_dim: int  = 512,
        n_layers: int    = 6,
        time_dim: int    = 128,
        T: int           = 500,
        emb_dim: int     = 32,
    ):
        super().__init__()
        self.T          = T
        self.max_decade = max_decade

        self.condition_encoder = ConditionEncoder(cond_dim, max_decade)
        self.date_embedder     = DateEmbedder(max_decade, emb_dim)
        self.time_embedding    = SinusoidalTimeEmbedding(time_dim)
        self.denoiser          = DenoisingNet(
            date_dim=self.date_embedder.out_dim,
            cond_dim=cond_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            max_decade=max_decade,
        )

        # Linear beta noise schedule
        betas     = torch.linspace(1e-4, 0.02, T)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas",                   betas)
        self.register_buffer("alpha_bar",               alpha_bar)
        self.register_buffer("sqrt_alpha_bar",          alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())

    # ── Forward process ───────────────────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """x_t = √ᾱ_t · x0 + √(1−ᾱ_t) · ε."""
        if noise is None:
            noise = torch.randn_like(x0)
        s = self.sqrt_alpha_bar[t].unsqueeze(-1)
        r = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return s * x0 + r * noise

    # ── Training step ─────────────────────────────────────────────────────────

    def training_loss(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        aux_weight: float = 0.1,
    ):
        """
        Predict-x0 diffusion loss + auxiliary discrete CE.
        Returns (total_loss, mse_loss.item(), ce_loss.item()).
        """
        B      = X.shape[0]
        device = X.device

        cond_emb = self.condition_encoder(X)
        x0       = self.date_embedder(Y)

        t     = torch.randint(0, self.T, (B,), device=device)
        x_t   = self.q_sample(x0, t)
        t_emb = self.time_embedding(t)

        x0_pred, l_mon, l_dec, l_unit, l_day = self.denoiser(x_t, cond_emb, t_emb)

        mse = F.mse_loss(x0_pred, x0)

        d_t, m_t, y_t = Y[:, 0], Y[:, 1], Y[:, 2]
        ce = (
            F.cross_entropy(l_mon,  m_t - 1)    +
            F.cross_entropy(l_dec,  y_t // 10)  +
            F.cross_entropy(l_unit, y_t % 10)   +
            F.cross_entropy(l_day,  d_t - 1)
        )

        total = mse + aux_weight * ce
        return total, mse.item(), ce.item()

    # ── Constrained projection ────────────────────────────────────────────────

    def _project_to_valid_date(
        self,
        l_mon: torch.Tensor,
        l_dec: torch.Tensor,
        l_unit: torch.Tensor,
        l_day: torch.Tensor,
        X_cond: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Project denoised logits to a valid calendar date (same masks as Models 1–3)."""
        B         = l_mon.shape[0]
        dow_cond  = X_cond[:, 0]
        leap_cond = X_cond[:, 2]
        dec_cond  = X_cond[:, 3]

        # Month: fixed by condition
        mon_pred = X_cond[:, 1].long() + 1   # 0-indexed → 1-indexed

        # Decade: fixed by condition
        dec_pred = dec_cond.long()

        # Year unit: constrained by leap-year condition
        y_cands     = dec_pred.unsqueeze(1) * 10 + torch.arange(10, device=device)
        valid_units = is_leap(y_cands).long() == leap_cond.unsqueeze(1)
        l_unit_m    = l_unit.masked_fill(~valid_units, float("-inf"))
        probs_unit  = F.softmax(l_unit_m, dim=-1)
        probs_unit  = torch.nan_to_num(probs_unit, nan=0.0)
        probs_unit[probs_unit.sum(-1) == 0, 0] = 1.0
        unit_pred   = torch.multinomial(probs_unit, 1).squeeze(-1)
        y_pred      = dec_pred * 10 + unit_pred

        # Day: constrained by valid range + day-of-week
        d_cands      = torch.arange(1, 32, device=device).unsqueeze(0).expand(B, 31)
        valid_bounds = d_cands <= days_in_month(mon_pred, y_pred).unsqueeze(1)
        dow_cands    = day_of_week(d_cands, mon_pred.unsqueeze(1), y_pred.unsqueeze(1))
        valid_dow    = dow_cands == dow_cond.unsqueeze(1)
        l_day_m      = l_day.masked_fill(~(valid_bounds & valid_dow), float("-inf"))
        probs_day    = F.softmax(l_day_m, dim=-1)
        probs_day    = torch.nan_to_num(probs_day, nan=0.0)
        probs_day[probs_day.sum(-1) == 0, 0] = 1.0
        day_pred     = torch.multinomial(probs_day, 1).squeeze(-1) + 1

        return torch.stack([day_pred, mon_pred, y_pred], dim=-1)

    # ── DDIM reverse sampling ─────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        X: torch.Tensor,
        n_samples: int = 1,
        device: str    = "cpu",
        ddim_steps: int = 50,
    ) -> torch.Tensor:
        """
        DDIM reverse process. Returns (B * n_samples, 3) integer dates [day, month, year].
        Matches Models 1–3 sample() signature.
        """
        X_rep    = X.repeat_interleave(n_samples, dim=0)
        B        = X_rep.shape[0]
        cond_emb = self.condition_encoder(X_rep)
        date_dim = self.date_embedder.out_dim

        x = torch.randn(B, date_dim, device=device)

        # Build DDIM timestep sequence (high → low, inclusive of 0)
        step = max(1, self.T // ddim_steps)
        ts   = list(range(self.T - 1, 0, -step))
        if ts[-1] != 0:
            ts.append(0)

        l_mon = l_dec = l_unit = l_day = None

        for i, t_curr in enumerate(ts):
            t_batch  = torch.full((B,), t_curr, device=device, dtype=torch.long)
            t_emb    = self.time_embedding(t_batch)
            x0_pred, l_mon, l_dec, l_unit, l_day = self.denoiser(x, cond_emb, t_emb)

            if i < len(ts) - 1:
                t_prev   = ts[i + 1]
                ab_curr  = self.alpha_bar[t_curr]
                ab_prev  = self.alpha_bar[t_prev]

                eps_pred = (x - ab_curr.sqrt() * x0_pred) / (1 - ab_curr).sqrt().clamp(min=1e-8)
                x        = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_pred
            # else: last step — logits are already from the cleanest denoised state

        return self._project_to_valid_date(l_mon, l_dec, l_unit, l_day, X_rep, device)