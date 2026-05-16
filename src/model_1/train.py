import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model_1.model import AutoregressiveDateModel, is_leap, days_in_month, day_of_week


def _target_distributions(X: torch.Tensor, Y: torch.Tensor, device: str):
    """
    Build multi-hot KL target distributions for each AR step.
    Any valid date satisfying the conditions is equally correct.
    """
    B = X.shape[0]
    dow_cond, mon_cond, leap_cond, dec_cond = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    m_true, y_true = Y[:, 1], Y[:, 2]

    # Month: one-hot from condition
    t_mon = torch.zeros(B, 12, device=device).scatter_(1, mon_cond.unsqueeze(1).long(), 1.0)

    # Decade: one-hot from condition
    t_dec = torch.zeros(B, 300, device=device).scatter_(1, dec_cond.unsqueeze(1).long(), 1.0)

    # Year unit: uniform over digits that satisfy leap constraint
    y_cands    = dec_cond.unsqueeze(1) * 10 + torch.arange(10, device=device)
    valid_unit = (is_leap(y_cands).long() == leap_cond.unsqueeze(1))
    t_unit     = valid_unit.float() / (valid_unit.sum(1, keepdim=True) + 1e-6)

    # Day: uniform over days that match dow and fit in the month
    # Uses teacher-forced m_true / y_true (consistent with forward pass)
    d_cands = torch.arange(1, 32, device=device).unsqueeze(0).expand(B, 31)
    valid_d = (
        (d_cands <= days_in_month(m_true, y_true).unsqueeze(1)) &
        (day_of_week(d_cands, m_true.unsqueeze(1), y_true.unsqueeze(1)) == dow_cond.unsqueeze(1))
    )
    t_day = valid_d.float() / (valid_d.sum(1, keepdim=True) + 1e-6)

    return t_mon, t_dec, t_unit, t_day


def _batch_loss(model, X: torch.Tensor, Y: torch.Tensor, device: str) -> torch.Tensor:
    l_mon, l_dec, l_unit, l_day = model(X, Y)
    t_mon, t_dec, t_unit, t_day = _target_distributions(X, Y, device)
    return (
        F.kl_div(F.log_softmax(l_mon,  -1), t_mon,  reduction="batchmean") +
        F.kl_div(F.log_softmax(l_dec,  -1), t_dec,  reduction="batchmean") +
        F.kl_div(F.log_softmax(l_unit, -1), t_unit, reduction="batchmean") +
        F.kl_div(F.log_softmax(l_day,  -1), t_day,  reduction="batchmean")
    )


def train(train_ds, val_ds, model_cfg, train_cfg, device: str = "cpu"):
    """
    Train Model 1. Returns (model, history).
    history keys: epochs, train_loss, val_loss.
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=min(train_cfg.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = AutoregressiveDateModel(
        cond_dim=model_cfg.cond_dim,
        max_decade=model_cfg.max_decade,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.n_epochs
    )

    history = {"epochs": [], "train_loss": [], "val_loss": []}

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}")
    print("-" * 34)

    for epoch in range(1, train_cfg.n_epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            loss = _batch_loss(model, X_b, Y_b, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b, Y_b = X_b.to(device), Y_b.to(device)
                val_loss += _batch_loss(model, X_b, Y_b, device).item()
        val_loss /= max(1, len(val_loader))

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 50 == 0 or epoch == 1 or epoch == train_cfg.n_epochs:
            print(f"{epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}")

    return model, history