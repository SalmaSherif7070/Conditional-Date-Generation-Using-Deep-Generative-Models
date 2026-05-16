"""
Energy-Based Model (EBM) Date Generator – Model 4
===================================================
Architecture
------------
  ConditionEncoder : (dow, month_idx, leap, decade) → 128-dim condition.
                     Identical interface to Models 1–3 for interoperability.

  DateEnergyNet    : Scalar energy function  E(date, condition) → ℝ.
                     Lower energy  ↔  more plausible (date, condition) pair.
                     Date is embedded the same way as the CVAE discriminator
                     (day, month, decade, year-unit each get a learnable
                     embedding), then concatenated with the condition vector
                     and passed through a deep residual MLP.

  DateEBM          : Wraps ConditionEncoder + DateEnergyNet and exposes the
                     project-standard  .sample(X, n_samples, device)  method
                     using Langevin MCMC with calendar constraints applied
                     after each step.

Training objective : Contrastive Divergence (CD)
    L = E(x⁺, c) − E(x⁻, c)  +  λ·[E(x⁺,c)² + E(x⁻,c)²]
  x⁺  real date from the dataset
  x⁻  negative sample drawn by running Langevin dynamics from a persistent
      replay buffer (following Du & Mordatch, 2019).

Sampling (inference)
--------------------
  Discrete Langevin with constraint projection:
    1. Initialise from a replay-buffer entry or random valid date.
    2. Repeat for n_mcmc_steps:
         a. Embed the current discrete date as continuous one-hot weights.
         b. Compute ∂E/∂embed  via autograd.
         c. Update each component logit with a gradient step + Gaussian noise.
         d. Project back to a valid integer date (greedy argmax + calendar
            constraints for decade, leap-year, day-of-week).
    3. Return the final constrained sample.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Calendar helpers (vectorised, matches Models 1–3) ─────────────────────────

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
    Identical interface to Models 1–3.
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


# ── Date Embedding (shared by energy net) ────────────────────────────────────

class DateEmbedder(nn.Module):
    """
    Embeds an integer date (day, month, year) into a fixed-size vector.
    Mirrors the discriminator embeddings used in Models 2 & 3 for consistency.
    """

    def __init__(self, max_decade: int = 300, emb_dim: int = 32):
        super().__init__()
        self.emb_dim    = emb_dim
        self.max_decade = max_decade

        self.emb_day  = nn.Embedding(32,          emb_dim)   # indices 1–31
        self.emb_mon  = nn.Embedding(13,          emb_dim)   # indices 1–12
        self.emb_dec  = nn.Embedding(max_decade,  emb_dim)
        self.emb_unit = nn.Embedding(10,          emb_dim)

    @property
    def out_dim(self) -> int:
        return self.emb_dim * 4

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """Y: (B, 3) integer [day, month, year]  →  (B, 4 * emb_dim)."""
        d, m, y = Y[:, 0], Y[:, 1], Y[:, 2]
        dec, unit = y // 10, y % 10
        return torch.cat([
            self.emb_day(d.clamp(1, 31)),
            self.emb_mon(m.clamp(1, 12)),
            self.emb_dec(dec.clamp(0, self.max_decade - 1)),
            self.emb_unit(unit),
        ], dim=-1)

    def embed_soft(
        self,
        w_day: torch.Tensor,   # (B, 31)  soft weights over days 1–31
        w_mon: torch.Tensor,   # (B, 12)  soft weights over months 1–12
        w_dec: torch.Tensor,   # (B, D)   soft weights over decades
        w_unit: torch.Tensor,  # (B, 10)  soft weights over year-units
    ) -> torch.Tensor:
        """Differentiable embedding via weighted combination — used in Langevin."""
        d_emb    = w_day  @ self.emb_day.weight[1:32]          # skip index 0
        m_emb    = w_mon  @ self.emb_mon.weight[1:13]          # skip index 0
        dec_emb  = w_dec  @ self.emb_dec.weight
        unit_emb = w_unit @ self.emb_unit.weight
        return torch.cat([d_emb, m_emb, dec_emb, unit_emb], dim=-1)


# ── Residual Block ────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ── Energy Network ────────────────────────────────────────────────────────────

class DateEnergyNet(nn.Module):
    """
    Scalar energy function  E(date_emb, cond_emb) → ℝ.

    Input  : concatenation of DateEmbedder output (128-d) + ConditionEncoder
             output (cond_dim-d).
    Output : scalar energy per sample — lower is more plausible.

    Architecture: linear projection → N residual blocks → scalar head.
    Spectral normalisation on the input/output projections keeps the energy
    landscape Lipschitz-bounded and improves Langevin mixing.
    """

    def __init__(
        self,
        date_emb_dim: int = 128,   # DateEmbedder.out_dim (4 * 32)
        cond_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        in_dim = date_emb_dim + cond_dim

        SN = nn.utils.spectral_norm
        self.proj_in = SN(nn.Linear(in_dim, hidden_dim))
        self.act_in  = nn.SiLU()

        self.res_blocks = nn.ModuleList(
            [_ResBlock(hidden_dim) for _ in range(n_layers)]
        )

        self.proj_out = SN(nn.Linear(hidden_dim, 1))

    def forward(self, date_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) energy scores."""
        h = self.act_in(self.proj_in(torch.cat([date_emb, cond_emb], dim=-1)))
        for block in self.res_blocks:
            h = block(h)
        return self.proj_out(h)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Persistent replay buffer for EBM training (Du & Mordatch, 2019).
    Stores past MCMC negative samples as integer tensors of shape (3,) = [d, m, y].
    """

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer: list[torch.Tensor] = []
        self._ptr = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, samples: torch.Tensor):
        """samples: (B, 3) CPU tensor."""
        for s in samples:
            if len(self.buffer) < self.capacity:
                self.buffer.append(s.clone())
            else:
                self.buffer[self._ptr] = s.clone()
                self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, n: int) -> torch.Tensor:
        """Returns (n, 3) tensor; raises if buffer is empty."""
        idx = torch.randint(len(self.buffer), (n,))
        return torch.stack([self.buffer[i] for i in idx])


# ── Full EBM ──────────────────────────────────────────────────────────────────

class DateEBM(nn.Module):
    """
    Full Energy-Based Model combining ConditionEncoder, DateEmbedder,
    and DateEnergyNet.

    Exposes the project-standard interface:
        .energy(X, Y)              → (B, 1) scalar energies
        .sample(X, n_samples, device) → (B * n_samples, 3) integer dates

    Langevin sampling strategy
    --------------------------
    Because dates are discrete, we work in the *logit space* of each
    component (month, decade, year-unit, day) and project back to integers
    after each step:

      logit ← logit − α · ∂E/∂embed(soft_date)  +  σ · ε,   ε ~ N(0, I)

    Calendar constraints are enforced by masking invalid logits to −∞
    before taking the argmax (exactly as in Models 1–3 during sampling).
    This gives a valid calendar date at every Langevin step.
    """

    def __init__(
        self,
        cond_dim: int = 128,
        max_decade: int = 300,
        hidden_dim: int = 512,
        n_layers: int = 4,
    ):
        super().__init__()
        self.max_decade = max_decade

        self.condition_encoder = ConditionEncoder(cond_dim, max_decade)
        self.date_embedder     = DateEmbedder(max_decade, emb_dim=32)
        self.energy_net        = DateEnergyNet(
            date_emb_dim=self.date_embedder.out_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    # ── Energy helpers ────────────────────────────────────────────────────────

    def energy(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        E(condition, date).
        X: (B, 4) condition  |  Y: (B, 3) integer date
        Returns (B, 1).
        """
        cond_emb = self.condition_encoder(X)
        date_emb = self.date_embedder(Y)
        return self.energy_net(date_emb, cond_emb)

    def energy_soft(
        self,
        cond_emb: torch.Tensor,
        w_day: torch.Tensor,
        w_mon: torch.Tensor,
        w_dec: torch.Tensor,
        w_unit: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable energy for Langevin gradient computation."""
        date_emb = self.date_embedder.embed_soft(w_day, w_mon, w_dec, w_unit)
        return self.energy_net(date_emb, cond_emb)

    # ── Constrained projection ────────────────────────────────────────────────

    @staticmethod
    def _project_to_valid_date(
        logit_mon: torch.Tensor,   # (B, 12)
        logit_dec: torch.Tensor,   # (B, max_decade)  — masked to condition
        logit_unit: torch.Tensor,  # (B, 10)
        logit_day: torch.Tensor,   # (B, 31)
        X_cond: torch.Tensor,      # (B, 4)
        device: str,
    ):
        """
        Project continuous logits to a valid (d, m, y) integer date tuple
        while enforcing all four calendar constraints.
        Returns (day, month, year) each (B,).
        """
        B         = logit_mon.shape[0]
        dow_cond  = X_cond[:, 0]
        leap_cond = X_cond[:, 2]
        dec_cond  = X_cond[:, 3]

        # 1. Month — greedy, constrained to the condition month
        mon_mask = torch.full((B, 12), float("-inf"), device=device)
        mon_mask.scatter_(1, X_cond[:, 1:2].long(), 0.0)
        mon_pred = torch.argmax(logit_mon + mon_mask, dim=-1) + 1  # 1-indexed

        # 2. Decade — fixed by condition (same as Models 1–3)
        dec_pred = dec_cond.long()

        # 3. Year unit — constrained by leap-year condition
        y_cands     = dec_pred.unsqueeze(1) * 10 + torch.arange(10, device=device)
        valid_units = is_leap(y_cands).long() == leap_cond.unsqueeze(1)
        masked_unit = logit_unit.masked_fill(~valid_units, float("-inf"))
        unit_pred   = torch.multinomial(F.softmax(masked_unit, dim=-1), 1).squeeze(-1)
        y_pred      = dec_pred * 10 + unit_pred

        # 4. Day — constrained by valid range + day-of-week
        d_cands      = torch.arange(1, 32, device=device).unsqueeze(0).expand(B, 31)
        valid_bounds = d_cands <= days_in_month(mon_pred, y_pred).unsqueeze(1)
        dow_cands    = day_of_week(d_cands, mon_pred.unsqueeze(1), y_pred.unsqueeze(1))
        valid_dow    = dow_cands == dow_cond.unsqueeze(1)
        masked_day   = logit_day.masked_fill(~(valid_bounds & valid_dow), float("-inf"))
        probs_day    = F.softmax(masked_day, dim=-1)
        probs_day    = torch.nan_to_num(probs_day, nan=0.0)
        probs_day[probs_day.sum(dim=-1) == 0, 0] = 1.0  # fallback
        day_pred     = torch.multinomial(probs_day, 1).squeeze(-1) + 1

        return day_pred, mon_pred, y_pred

    # ── Langevin MCMC ─────────────────────────────────────────────────────────

    def langevin_sample(
        self,
        X_cond: torch.Tensor,        # (B, 4) condition
        init_Y: torch.Tensor | None, # (B, 3) integer starting point (optional)
        n_steps: int = 60,
        step_size: float = 0.1,
        noise_std: float = 0.005,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Discrete Langevin dynamics in logit space with calendar-constrained
        projection at every step.

        Returns (B, 3) integer tensor of sampled dates [day, month, year].
        """
        B         = X_cond.shape[0]
        cond_emb  = self.condition_encoder(X_cond)  # fixed throughout

        # Initialise logits from the starting integer date (or uniform)
        if init_Y is not None:
            # Warm-start: one-hot logits centred on the init date
            d0, m0, y0 = init_Y[:, 0], init_Y[:, 1], init_Y[:, 2]
            dec0, unit0 = y0 // 10, y0 % 10

            logit_mon  = torch.zeros(B, 12,              device=device)
            logit_dec  = torch.zeros(B, self.max_decade, device=device)
            logit_unit = torch.zeros(B, 10,              device=device)
            logit_day  = torch.zeros(B, 31,              device=device)

            logit_mon.scatter_(1,  (m0 - 1).unsqueeze(1).long(), 3.0)
            logit_dec.scatter_(1,  dec0.unsqueeze(1).long(),      3.0)
            logit_unit.scatter_(1, unit0.unsqueeze(1).long(),     3.0)
            logit_day.scatter_(1,  (d0 - 1).unsqueeze(1).long(),  3.0)
        else:
            logit_mon  = torch.randn(B, 12,              device=device)
            logit_dec  = torch.randn(B, self.max_decade, device=device)
            logit_unit = torch.randn(B, 10,              device=device)
            logit_day  = torch.randn(B, 31,              device=device)

        for _ in range(n_steps):
            # Soft weights via softmax (differentiable)
            w_mon  = F.softmax(logit_mon,  dim=-1).requires_grad_(True)
            w_dec  = F.softmax(logit_dec,  dim=-1).requires_grad_(True)
            w_unit = F.softmax(logit_unit, dim=-1).requires_grad_(True)
            w_day  = F.softmax(logit_day,  dim=-1).requires_grad_(True)

            E = self.energy_soft(cond_emb, w_day, w_mon, w_dec, w_unit).sum()
            E.backward()

            with torch.no_grad():
                logit_mon  = (logit_mon  - step_size * w_mon.grad
                              + noise_std * torch.randn_like(logit_mon)).detach()
                logit_dec  = (logit_dec  - step_size * w_dec.grad
                              + noise_std * torch.randn_like(logit_dec)).detach()
                logit_unit = (logit_unit - step_size * w_unit.grad
                              + noise_std * torch.randn_like(logit_unit)).detach()
                logit_day  = (logit_day  - step_size * w_day.grad
                              + noise_std * torch.randn_like(logit_day)).detach()

        # Final constrained projection
        with torch.no_grad():
            day_pred, mon_pred, y_pred = self._project_to_valid_date(
                logit_mon, logit_dec, logit_unit, logit_day, X_cond, device,
            )

        return torch.stack([day_pred, mon_pred, y_pred], dim=-1)

    # ── Project-standard sample() ─────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        X: torch.Tensor,
        n_samples: int = 1,
        device: str = "cpu",
        n_mcmc_steps: int = 60,
        step_size: float = 0.1,
        noise_std: float = 0.005,
    ) -> torch.Tensor:
        """
        Constrained inference — matches Models 1–3 sample() signature.
        Returns (B * n_samples, 3) tensor of [day, month, year].

        Note: Langevin is run with gradient tracking turned on internally
        even though this method is decorated with @no_grad, because the
        Langevin loop needs gradients w.r.t. the soft logits (not model
        parameters). The outer @no_grad prevents accidental parameter
        gradient accumulation.
        """
        X_rep = X.repeat_interleave(n_samples, dim=0)

        # Temporarily enable grad for Langevin (model params won't accumulate)
        with torch.enable_grad():
            result = self.langevin_sample(
                X_rep,
                init_Y=None,
                n_steps=n_mcmc_steps,
                step_size=step_size,
                noise_std=noise_std,
                device=device,
            )
        return result

    # ── CD Loss ───────────────────────────────────────────────────────────────

    def cd_loss(
        self,
        X: torch.Tensor,       # (B, 4)  condition
        Y_pos: torch.Tensor,   # (B, 3)  real date (positive sample)
        Y_neg: torch.Tensor,   # (B, 3)  MCMC negative sample
        l2_reg: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Contrastive Divergence loss.
            L = E(x⁺, c) − E(x⁻, c)  +  λ·[E(x⁺,c)² + E(x⁻,c)²]

        The regulariser prevents the energy landscape from collapsing.
        Returns (loss, e_pos.mean(), e_neg.mean()) for logging.
        """
        e_pos = self.energy(X, Y_pos)   # (B, 1)
        e_neg = self.energy(X, Y_neg)   # (B, 1)

        cd    = (e_pos - e_neg).mean()
        reg   = l2_reg * (e_pos.pow(2) + e_neg.pow(2)).mean()
        return cd + reg, e_pos.detach().mean(), e_neg.detach().mean()