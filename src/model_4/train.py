"""
Model 4 training: Conditional Diffusion (DDPM with predict-x0 objective).

Loss = MSE(x0_pred, x0_embed) + aux_weight * CE(discrete_heads, Y)

The auxiliary cross-entropy on discrete heads (month, decade, unit, day)
provides strong gradient signal that guides the denoiser to understand
calendar structure — without it, the model has to infer all structure
from the MSE in embedding space alone.

Validation metric: Condition Satisfaction Rate (CSR) via DDIM sampling.
"""
import torch
from torch.utils.data import DataLoader

from src.model_4.model import (
    DateDiffusionModel,
    is_leap, days_in_month, day_of_week,
)


def _compute_csr(Y_gen: torch.Tensor, X_cond: torch.Tensor) -> float:
    """Fraction of generated dates satisfying all four constraints."""
    d, m, y = Y_gen[:, 0], Y_gen[:, 1], Y_gen[:, 2]
    dow_c, mon_c, leap_c, dec_c = X_cond[:, 0], X_cond[:, 1], X_cond[:, 2], X_cond[:, 3]
    ok_month = (m >= 1) & (m <= 12)
    ok_year  = y > 0
    ok_day   = ok_month & ok_year & (d >= 1) & (d <= days_in_month(m, y))
    all_ok   = (
        ok_day
        & (day_of_week(d, m, y) == dow_c)
        & ((m - 1) == mon_c)
        & (is_leap(y).long() == leap_c)
        & ((y // 10) == dec_c)
    )
    return all_ok.float().mean().item()


def train(train_ds, val_ds, model4_cfg, train4_cfg, device: str = "cpu"):
    """
    Train Model 4 (Conditional Diffusion).

    Parameters
    ----------
    train_ds   : training dataset (yields (X, Y) pairs)
    val_ds     : validation dataset
    model4_cfg : Model4Config
    train4_cfg : Train4Config
    device     : "cpu" or "cuda"

    Returns
    -------
    (model, history)
    history keys: epochs, train_loss, val_loss, mse_loss, ce_loss, csr
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=min(train4_cfg.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = DateDiffusionModel(
        cond_dim=model4_cfg.cond_dim,
        max_decade=model4_cfg.max_decade,
        hidden_dim=model4_cfg.hidden_dim,
        n_layers=model4_cfg.n_layers,
        time_dim=model4_cfg.time_dim,
        T=model4_cfg.T,
        emb_dim=model4_cfg.emb_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train4_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train4_cfg.n_epochs
    )

    history = {
        "epochs": [], "train_loss": [], "val_loss": [],
        "mse_loss": [], "ce_loss": [], "csr": [],
    }

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  "
          f"{'MSE':>8}  {'CE':>8}  {'Val CSR':>8}")
    print("-" * 62)

    for epoch in range(1, train4_cfg.n_epochs + 1):
        model.train()
        epoch_total = epoch_mse = epoch_ce = 0.0
        n_batches = 0

        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()

            loss, mse, ce = model.training_loss(X_b, Y_b, aux_weight=train4_cfg.aux_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_total += loss.item()
            epoch_mse   += mse
            epoch_ce    += ce
            n_batches   += 1

        scheduler.step()

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_loss_acc          = 0.0
        all_X_val, all_Y_gen  = [], []

        with torch.no_grad():
            for X_v, Y_v in val_loader:
                X_v, Y_v = X_v.to(device), Y_v.to(device)
                l_v, _, _ = model.training_loss(X_v, Y_v, aux_weight=train4_cfg.aux_weight)
                val_loss_acc += l_v.item()

                Y_gen_v = model.sample(
                    X_v, n_samples=1, device=device,
                    ddim_steps=train4_cfg.ddim_steps,
                )
                all_X_val.append(X_v.cpu())
                all_Y_gen.append(Y_gen_v.cpu())

        val_loss_avg = val_loss_acc / max(1, len(val_loader))
        csr          = _compute_csr(torch.cat(all_Y_gen), torch.cat(all_X_val))

        t_loss = epoch_total / n_batches
        t_mse  = epoch_mse   / n_batches
        t_ce   = epoch_ce    / n_batches

        history["epochs"].append(epoch)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(val_loss_avg)
        history["mse_loss"].append(t_mse)
        history["ce_loss"].append(t_ce)
        history["csr"].append(csr)

        if epoch % 10 == 0 or epoch == 1 or epoch == train4_cfg.n_epochs:
            print(f"{epoch:>6}  {t_loss:>11.4f}  {val_loss_avg:>9.4f}  "
                  f"{t_mse:>8.4f}  {t_ce:>8.4f}  {csr:>8.3f}")

    return model, history