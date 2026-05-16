"""
Model 2 training: WGAN-GP + Auxiliary KL Supervision.

Discriminator is updated n_critic times per generator step.
Generator loss = -D(fake) + lambda_aux * KL(heads || target_distributions).
Gradient penalty is computed in embedding space for differentiability.
Gumbel temperature is annealed from tau_start → tau_end over training.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model_2.model import (
    ConditionEncoder, DateGenerator, DateDiscriminator,
    is_leap, days_in_month, day_of_week,
)


# ── Target distributions (same logic as Model 1) ──────────────────────────────

def _target_distributions(X: torch.Tensor, Y: torch.Tensor, device: str):
    """
    Multi-hot KL targets for each AR step.
    Any date satisfying the conditions is equally correct.
    """
    B = X.shape[0]
    dow_cond, mon_cond, leap_cond, dec_cond = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    m_true, y_true = Y[:, 1], Y[:, 2]

    # Month: one-hot from condition
    t_mon = torch.zeros(B, 12, device=device).scatter_(
        1, mon_cond.unsqueeze(1).long(), 1.0)

    # Decade: one-hot from condition
    t_dec = torch.zeros(B, 300, device=device).scatter_(
        1, dec_cond.unsqueeze(1).long(), 1.0)

    # Year unit: uniform over leap-valid digits
    y_cands    = dec_cond.unsqueeze(1) * 10 + torch.arange(10, device=device)
    valid_unit = is_leap(y_cands).long() == leap_cond.unsqueeze(1)
    t_unit     = valid_unit.float() / (valid_unit.sum(1, keepdim=True) + 1e-6)

    # Day: uniform over valid days matching DOW constraint
    d_cands = torch.arange(1, 32, device=device).unsqueeze(0).expand(B, 31)
    valid_d = (
        (d_cands <= days_in_month(m_true, y_true).unsqueeze(1)) &
        (day_of_week(d_cands, m_true.unsqueeze(1), y_true.unsqueeze(1))
         == dow_cond.unsqueeze(1))
    )
    t_day = valid_d.float() / (valid_d.sum(1, keepdim=True) + 1e-6)

    return t_mon, t_dec, t_unit, t_day


# ── Gradient penalty ──────────────────────────────────────────────────────────

def _gradient_penalty(D: DateDiscriminator, real_Y: torch.Tensor,
                      soft_mon: torch.Tensor, soft_dec: torch.Tensor,
                      soft_unit: torch.Tensor, soft_day: torch.Tensor,
                      cond_emb: torch.Tensor, device: str) -> torch.Tensor:
    """
    WGAN-GP in embedding space (differentiable for discrete outputs).
    Interpolates between real and fake date embeddings.
    """
    B   = real_Y.shape[0]
    eps = torch.rand(B, 1, device=device)

    # Real embeddings
    d_r    = D.emb_d_day(real_Y[:, 0].clamp(1, 31))
    m_r    = D.emb_d_mon(real_Y[:, 1].clamp(1, 12))
    dec_r  = D.emb_d_dec((real_Y[:, 2] // 10).clamp(0, D.max_decade - 1))
    unit_r = D.emb_d_unit(real_Y[:, 2] % 10)

    # Fake embeddings (soft)
    d_f    = soft_day  @ D.emb_d_day.weight[1:32]
    m_f    = soft_mon  @ D.emb_d_mon.weight[1:13]
    dec_f  = soft_dec  @ D.emb_d_dec.weight
    unit_f = soft_unit @ D.emb_d_unit.weight

    def interp(r, f):
        return (eps * r + (1 - eps) * f).requires_grad_(True)

    hat_day  = interp(d_r,    d_f)
    hat_mon  = interp(m_r,    m_f)
    hat_dec  = interp(dec_r,  dec_f)
    hat_unit = interp(unit_r, unit_f)

    interp_date = torch.cat([hat_day, hat_mon, hat_dec, hat_unit], dim=-1)
    score = D.net(torch.cat([interp_date, cond_emb.detach()], dim=-1))

    grads = torch.autograd.grad(
        outputs=score,
        inputs=[hat_day, hat_mon, hat_dec, hat_unit],
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )
    grad_cat  = torch.cat([g.view(B, -1) for g in grads], dim=-1)
    grad_norm = grad_cat.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(train_ds, val_ds, model2_cfg, train2_cfg, device: str = "cpu",
          model1_weights_path: str = None):
    """
    Train Model 2 (Conditional GAN).

    Parameters
    ----------
    train_ds, val_ds : torch Subset datasets yielding (X, Y) pairs
    model2_cfg       : Model2Config
    train2_cfg       : Train2Config
    device           : "cpu" or "cuda"
    model1_weights_path : optional path to Model 1 weights for encoder warm-start

    Returns
    -------
    (generator, discriminator, encoder, history)
    history keys: epochs, d_loss, g_loss, gp, csr, tau
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=min(train2_cfg.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # Build models
    encoder = ConditionEncoder(model2_cfg.cond_dim, model2_cfg.max_decade).to(device)
    G = DateGenerator(model2_cfg.z_dim, model2_cfg.cond_dim, model2_cfg.max_decade).to(device)
    D = DateDiscriminator(model2_cfg.cond_dim, model2_cfg.max_decade).to(device)

    # Optional: warm-start encoder from Model 1
    freeze_encoder = False
    if model1_weights_path is not None:
        try:
            ar_sd = torch.load(model1_weights_path, map_location=device)
            encoder.load_from_model1(ar_sd)
            for p in encoder.parameters():
                p.requires_grad = False
            freeze_encoder = True
        except Exception as e:
            print(f"[Warning] Could not load Model 1 weights: {e}. Training encoder from scratch.")

    g_params = list(G.parameters())
    if not freeze_encoder:
        g_params += list(encoder.parameters())

    opt_G = torch.optim.Adam(g_params,         lr=train2_cfg.lr_g, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(),   lr=train2_cfg.lr_d, betas=(0.5, 0.9))
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=train2_cfg.n_epochs)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=train2_cfg.n_epochs)

    history = {"epochs": [], "d_loss": [], "g_loss": [], "gp": [], "csr": [], "tau": []}

    print(f"\n{'Epoch':>6}  {'D Loss':>10}  {'G Loss':>10}  {'GP':>8}  {'Val CSR':>8}  {'tau':>6}")
    print("-" * 60)

    for epoch in range(1, train2_cfg.n_epochs + 1):
        G.train(); D.train()
        if not freeze_encoder:
            encoder.train()

        tau = train2_cfg.tau_start * (train2_cfg.tau_end / train2_cfg.tau_start) ** (
            (epoch - 1) / max(train2_cfg.n_epochs - 1, 1)
        )

        epoch_d, epoch_g, epoch_gp = 0.0, 0.0, 0.0
        n_batches = 0

        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            B = X_b.shape[0]
            cond_emb = encoder(X_b)

            # ── Discriminator steps ────────────────────────────────────
            for _ in range(train2_cfg.n_critic):
                z = torch.randn(B, model2_cfg.z_dim, device=device)
                lm, ld, lu, lday, sm, sd, su = G(z, cond_emb.detach(), tau=tau)
                soft_day = F.gumbel_softmax(lday, tau=tau, hard=False)

                score_real = D(Y_b, cond_emb.detach())
                score_fake = D.forward_soft(sm, sd, su, soft_day, cond_emb.detach())

                gp = _gradient_penalty(
                    D, Y_b, sm, sd, su, soft_day, cond_emb, device)
                d_loss = -score_real.mean() + score_fake.mean() + train2_cfg.lambda_gp * gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

                epoch_d  += d_loss.item()
                epoch_gp += gp.item()

            # ── Generator step ─────────────────────────────────────────
            cond_emb = encoder(X_b)
            z        = torch.randn(B, model2_cfg.z_dim, device=device)
            lm, ld, lu, lday, sm, sd, su = G(z, cond_emb, tau=tau)
            soft_day    = F.gumbel_softmax(lday, tau=tau, hard=False)
            score_fake_g = D.forward_soft(sm, sd, su, soft_day, cond_emb)

            # Auxiliary KL supervision
            t_mon, t_dec, t_unit, t_day = _target_distributions(X_b, Y_b, device)
            aux = (
                F.kl_div(F.log_softmax(lm,    -1), t_mon,  reduction="batchmean") +
                F.kl_div(F.log_softmax(ld,    -1), t_dec,  reduction="batchmean") +
                F.kl_div(F.log_softmax(lu,    -1), t_unit, reduction="batchmean") +
                F.kl_div(F.log_softmax(lday,  -1), t_day,  reduction="batchmean")
            )
            g_loss = -score_fake_g.mean() + 0.5 * aux

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            epoch_g   += g_loss.item()
            n_batches += 1

        sched_G.step()
        sched_D.step()

        # ── Validation CSR ─────────────────────────────────────────────
        G.eval(); encoder.eval()
        all_X_val, all_Y_gen = [], []
        with torch.no_grad():
            for X_v, _ in val_loader:
                X_v = X_v.to(device)
                cond_v = encoder(X_v)
                Y_v    = G.sample(cond_v, X_v, device=device)
                all_X_val.append(X_v.cpu())
                all_Y_gen.append(Y_v.cpu())

        X_val_all = torch.cat(all_X_val)
        Y_gen_all = torch.cat(all_Y_gen)
        csr = _compute_csr(Y_gen_all, X_val_all)

        d_avg  = epoch_d  / (n_batches * train2_cfg.n_critic)
        g_avg  = epoch_g  / n_batches
        gp_avg = epoch_gp / (n_batches * train2_cfg.n_critic)

        history["epochs"].append(epoch)
        history["d_loss"].append(d_avg)
        history["g_loss"].append(g_avg)
        history["gp"].append(gp_avg)
        history["csr"].append(csr)
        history["tau"].append(tau)

        if epoch % 20 == 0 or epoch == 1 or epoch == train2_cfg.n_epochs:
            print(f"{epoch:>6}  {d_avg:>10.4f}  {g_avg:>10.4f}  "
                  f"{gp_avg:>8.4f}  {csr:>8.3f}  {tau:>6.3f}")

        if not freeze_encoder:
            encoder.train()
        G.train()

    return G, D, encoder, history


def _compute_csr(Y_gen: torch.Tensor, X_cond: torch.Tensor) -> float:
    """Condition Satisfaction Rate — fraction of dates satisfying all 4 constraints."""
    d, m, y = Y_gen[:, 0], Y_gen[:, 1], Y_gen[:, 2]
    dow_c, mon_c, leap_c, dec_c = X_cond[:, 0], X_cond[:, 1], X_cond[:, 2], X_cond[:, 3]
    ok_month = (m >= 1) & (m <= 12)
    ok_year  = y > 0
    ok_day   = ok_month & ok_year & (d >= 1) & (d <= days_in_month(m, y))
    all_ok   = (
        ok_day &
        (day_of_week(d, m, y) == dow_c) &
        ((m - 1) == mon_c) &
        (is_leap(y).long() == leap_c) &
        ((y // 10) == dec_c)
    )
    return all_ok.float().mean().item()