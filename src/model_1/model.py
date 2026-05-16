import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Calendar helpers ──────────────────────────────────────────────

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


# ── Model ─────────────────────────────────────────────────────────

class AutoregressiveDateModel(nn.Module):
    """
    Autoregressive conditional date generator.
    Predicts: month → year-decade → year-unit → day.
    Condition: [dow, month, leap, decade].
    """

    def __init__(self, cond_dim: int = 128, max_decade: int = 300):
        super().__init__()
        self.max_decade = max_decade

        # Condition encoders
        self.emb_dow  = nn.Embedding(7,           16)
        self.emb_mon  = nn.Embedding(12,          16)
        self.emb_leap = nn.Embedding(2,            8)
        self.emb_dec  = nn.Embedding(max_decade,  32)

        # +4 for sine/cosine cyclical features (dow + month)
        self.cond_mlp = nn.Sequential(
            nn.Linear(16 + 16 + 8 + 32 + 4, cond_dim),
            nn.ReLU(),
            nn.LayerNorm(cond_dim),
        )

        # Autoregressive context embeddings
        self.emb_seq_mon  = nn.Embedding(13,         32)
        self.emb_seq_dec  = nn.Embedding(max_decade, 32)
        self.emb_seq_unit = nn.Embedding(10,         32)

        # Prediction heads (each takes accumulated context)
        self.head_mon  = nn.Sequential(nn.Linear(cond_dim,       64),  nn.ReLU(), nn.Linear(64,  12))
        self.head_dec  = nn.Sequential(nn.Linear(cond_dim + 32,  128), nn.ReLU(), nn.Linear(128, max_decade))
        self.head_unit = nn.Sequential(nn.Linear(cond_dim + 64,  128), nn.ReLU(), nn.Linear(128, 10))
        self.head_day  = nn.Sequential(nn.Linear(cond_dim + 96,  256), nn.ReLU(), nn.Linear(256, 31))

    def encode_cond(self, X: torch.Tensor) -> torch.Tensor:
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

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """Teacher-forcing forward pass. Returns logits for each AR step."""
        cond      = self.encode_cond(X)
        m_true    = Y[:, 1]
        dec_true  = Y[:, 2] // 10
        unit_true = Y[:, 2] % 10

        l_mon  = self.head_mon(cond)
        ctx2   = torch.cat([cond, self.emb_seq_mon(m_true)], dim=-1)
        l_dec  = self.head_dec(ctx2)
        ctx3   = torch.cat([ctx2, self.emb_seq_dec(dec_true)], dim=-1)
        l_unit = self.head_unit(ctx3)
        ctx4   = torch.cat([ctx3, self.emb_seq_unit(unit_true)], dim=-1)
        l_day  = self.head_day(ctx4)

        return l_mon, l_dec, l_unit, l_day

    @torch.no_grad()
    def sample(self, X: torch.Tensor, n_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """
        Constrained autoregressive sampling.
        Returns tensor of shape (B * n_samples, 3): [day, month, year].
        """
        B_rep = X.shape[0] * n_samples
        X_rep = X.repeat_interleave(n_samples, dim=0)
        cond  = self.encode_cond(X_rep)

        dow_cond  = X_rep[:, 0]
        leap_cond = X_rep[:, 2]
        dec_cond  = X_rep[:, 3]

        # Step 1 – Month (greedy; month is fully determined by condition)
        mon_pred = torch.argmax(self.head_mon(cond), dim=-1) + 1

        # Step 2 – Decade (taken directly from condition; head_dec is auxiliary)
        dec_pred = dec_cond
        ctx2 = torch.cat([cond, self.emb_seq_mon(mon_pred)], dim=-1)

        # Step 3 – Year unit (constrained by leap year condition)
        ctx3   = torch.cat([ctx2, self.emb_seq_dec(dec_pred)], dim=-1)
        l_unit = self.head_unit(ctx3)

        y_cands     = dec_pred.unsqueeze(1) * 10 + torch.arange(10, device=device)
        valid_units = (is_leap(y_cands).long() == leap_cond.unsqueeze(1))
        l_unit      = l_unit.masked_fill(~valid_units, float("-inf"))
        unit_pred   = torch.multinomial(F.softmax(l_unit, dim=-1), 1).squeeze(-1)
        y_pred      = dec_pred * 10 + unit_pred

        # Step 4 – Day (constrained by valid range + day-of-week)
        ctx4  = torch.cat([ctx3, self.emb_seq_unit(unit_pred)], dim=-1)
        l_day = self.head_day(ctx4)

        d_cands      = torch.arange(1, 32, device=device).unsqueeze(0).expand(B_rep, 31)
        valid_bounds = d_cands <= days_in_month(mon_pred, y_pred).unsqueeze(1)
        dow_cands    = day_of_week(d_cands, mon_pred.unsqueeze(1), y_pred.unsqueeze(1))
        valid_dow    = dow_cands == dow_cond.unsqueeze(1)

        l_day = l_day.masked_fill(~(valid_bounds & valid_dow), float("-inf"))
        probs = F.softmax(l_day, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs[probs.sum(dim=-1) == 0, 0] = 1.0  # fallback for impossible constraints

        day_pred = torch.multinomial(probs, 1).squeeze(-1) + 1

        return torch.stack([day_pred, mon_pred, y_pred], dim=-1)