"""
src/model_3/evaluate.py  — evaluation helper for Model 3 (CVAE).
"""
import torch
from torch.utils.data import DataLoader
from src.model_4.model import is_leap, days_in_month, day_of_week


def check_conditions(dates: torch.Tensor, conditions: torch.Tensor) -> dict:
    d, m, y = dates[:, 0], dates[:, 1], dates[:, 2]
    dow_c, mon_c, leap_c, dec_c = (
        conditions[:, 0], conditions[:, 1], conditions[:, 2], conditions[:, 3]
    )
    ok_month  = (m >= 1) & (m <= 12)
    ok_year   = y > 0
    ok_day    = ok_month & ok_year & (d >= 1) & (d <= days_in_month(m, y))
    dow_ok    = ok_day & (day_of_week(d, m, y) == dow_c)
    mon_ok    = ok_day & ((m - 1) == mon_c)
    leap_ok   = ok_day & (is_leap(y).long() == leap_c)
    decade_ok = ok_day & ((y // 10) == dec_c)
    all_ok    = dow_ok & mon_ok & leap_ok & decade_ok
    return {
        "csr":        all_ok.float().mean().item(),
        "dow_acc":    dow_ok.float().mean().item(),
        "mon_acc":    mon_ok.float().mean().item(),
        "leap_acc":   leap_ok.float().mean().item(),
        "decade_acc": decade_ok.float().mean().item(),
    }


def evaluate_model(model, val_ds, device: str, n_show: int = 15) -> dict:
    loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    all_X, all_Y = [], []
    model.eval()
    with torch.no_grad():
        for X_b, _ in loader:
            X_b = X_b.to(device)
            Y_b = model.sample(X_b, n_samples=1, device=device).cpu()
            all_X.append(X_b.cpu())
            all_Y.append(Y_b)
    metrics = check_conditions(torch.cat(all_Y), torch.cat(all_X))
    print("\n── Model 3 Evaluation ──")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    return metrics