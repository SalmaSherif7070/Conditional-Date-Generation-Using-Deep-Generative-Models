"""
Model 3 training: Conditional VAE with β-annealed ELBO.

β is annealed from 0 → beta_max over the first 50% of epochs (warm-up),
then held constant.  This prioritises reconstruction quality early on
and lets the KL term regularise the posterior once the decoder is stable.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model_3.model import DateCVAE, cvae_loss, is_leap, days_in_month, day_of_week


# ── Condition Satisfaction Rate ───────────────────────────────────────────────

def _compute_csr(Y_gen: torch.Tensor, X_cond: torch.Tensor) -> float:
    """Fraction of generated dates satisfying all four constraints."""
    d, m, y = Y_gen[:, 0], Y_gen[:, 1], Y_gen[:, 2]
    dow_c, mon_c, leap_c, dec_c = X_cond[:, 0], X_cond[:, 1], X_cond[:, 2], X_cond[:, 3]

    ok_month = (m >= 1) & (m <= 12)
    ok_year  = y > 0
    ok_day   = ok_month & ok_year & (d >= 1) & (d <= days_in_month(m, y))

    all_ok = (
        ok_day
        & (day_of_week(d, m, y) == dow_c)
        & ((m - 1) == mon_c)
        & (is_leap(y).long() == leap_c)
        & ((y // 10) == dec_c)
    )
    return all_ok.float().mean().item()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(train_ds, val_ds, model3_cfg, train3_cfg, device: str = "cpu"):
    """
    Train Model 3 (Conditional VAE).

    Parameters
    ----------
    train_ds   : training dataset (yields (X, Y) pairs)
    val_ds     : validation dataset
    model3_cfg : Model3Config
    train3_cfg : Train3Config
    device     : "cpu" or "cuda"

    Returns
    -------
    (model, history)
    history keys: epochs, train_loss, val_loss, recon_loss, kl_loss, csr, beta
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=min(train3_cfg.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = DateCVAE(
        z_dim=model3_cfg.z_dim,
        cond_dim=model3_cfg.cond_dim,
        max_decade=model3_cfg.max_decade,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train3_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train3_cfg.n_epochs
    )

    history = {
        "epochs": [], "train_loss": [], "val_loss": [],
        "recon_loss": [], "kl_loss": [], "csr": [], "beta": [],
    }

    warmup_end = int(train3_cfg.n_epochs * train3_cfg.beta_warmup_frac)

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  "
          f"{'Recon':>8}  {'KL':>8}  {'Beta':>6}  {'Val CSR':>8}")
    print("-" * 68)

    for epoch in range(1, train3_cfg.n_epochs + 1):
        # β annealing: linear warm-up then constant
        if epoch <= warmup_end:
            beta = train3_cfg.beta_max * (epoch / max(warmup_end, 1))
        else:
            beta = train3_cfg.beta_max

        # ── Train ──────────────────────────────────────────────────
        model.train()
        epoch_loss = epoch_recon = epoch_kl = 0.0
        n_batches  = 0

        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()

            logits, mu, logvar = model(X_b, Y_b)
            loss, recon, kl    = cvae_loss(logits, Y_b, mu, logvar, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_recon += recon.item()
            epoch_kl    += kl.item()
            n_batches   += 1

        scheduler.step()

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        all_X_val, all_Y_gen = [], []

        with torch.no_grad():
            for X_v, Y_v in val_loader:
                X_v, Y_v = X_v.to(device), Y_v.to(device)
                logits_v, mu_v, logvar_v = model(X_v, Y_v)
                l_v, _, _ = cvae_loss(logits_v, Y_v, mu_v, logvar_v, beta=beta)
                val_loss += l_v.item()

                Y_gen_v = model.sample(X_v, n_samples=1, device=device)
                all_X_val.append(X_v.cpu())
                all_Y_gen.append(Y_gen_v.cpu())

        val_loss /= max(1, len(val_loader))

        X_val_cat  = torch.cat(all_X_val)
        Y_gen_cat  = torch.cat(all_Y_gen)
        csr        = _compute_csr(Y_gen_cat, X_val_cat)

        t_loss = epoch_loss  / n_batches
        t_rec  = epoch_recon / n_batches
        t_kl   = epoch_kl    / n_batches

        history["epochs"].append(epoch)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(val_loss)
        history["recon_loss"].append(t_rec)
        history["kl_loss"].append(t_kl)
        history["csr"].append(csr)
        history["beta"].append(beta)

        if epoch % 10 == 0 or epoch == 1 or epoch == train3_cfg.n_epochs:
            print(f"{epoch:>6}  {t_loss:>11.4f}  {val_loss:>9.4f}  "
                  f"{t_rec:>8.4f}  {t_kl:>8.4f}  {beta:>6.4f}  {csr:>8.3f}")

    return model, history