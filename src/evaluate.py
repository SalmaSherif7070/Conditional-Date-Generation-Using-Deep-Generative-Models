import torch
from torch.utils.data import DataLoader

from src.model_1.model import is_leap, days_in_month, day_of_week
from src.data_processing import REV_DOW, REV_MON


def check_conditions(dates: torch.Tensor, conditions: torch.Tensor) -> dict:
    """
    Interval-based evaluation: a date is correct if it satisfies ALL four conditions.
    There is no single ground-truth answer; any compliant date is valid.
    """
    d, m, y = dates[:, 0], dates[:, 1], dates[:, 2]
    dow_cond, mon_cond, leap_cond, dec_cond = (
        conditions[:, 0], conditions[:, 1], conditions[:, 2], conditions[:, 3]
    )

    ok_month = (m >= 1) & (m <= 12)
    ok_year  = y > 0
    ok_day   = ok_month & ok_year & (d >= 1) & (d <= days_in_month(m, y))

    dow_ok    = ok_day & (day_of_week(d, m, y) == dow_cond)
    mon_ok    = ok_day & ((m - 1) == mon_cond)
    leap_ok   = ok_day & (is_leap(y).long() == leap_cond)
    decade_ok = ok_day & ((y // 10) == dec_cond)
    all_ok    = dow_ok & mon_ok & leap_ok & decade_ok

    return {
        "csr":        all_ok.float().mean().item(),
        "dow_acc":    dow_ok.float().mean().item(),
        "mon_acc":    mon_ok.float().mean().item(),
        "leap_acc":   leap_ok.float().mean().item(),
        "decade_acc": decade_ok.float().mean().item(),
    }


def evaluate_model(model, val_ds, device: str, n_show: int = 15) -> dict:
    """
    Run constrained sampling on the validation set and report per-condition accuracy.
    Prints a sample of generated dates with pass/fail per condition.
    """
    loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    all_X, all_Y_gen = [], []
    model.eval()
    with torch.no_grad():
        for X_b, _ in loader:
            Y_b = model.sample(X_b.to(device), n_samples=1, device=device).cpu()
            all_X.append(X_b)
            all_Y_gen.append(Y_b)

    X_all     = torch.cat(all_X)
    Y_gen_all = torch.cat(all_Y_gen)
    metrics   = check_conditions(Y_gen_all, X_all)

    print("\n── Evaluation (Interval-Based, any valid date is correct) ──")
    print(f"  CSR – all conditions satisfied : {metrics['csr']:.3f}")
    print(f"  Day-of-week accuracy           : {metrics['dow_acc']:.3f}")
    print(f"  Month accuracy                 : {metrics['mon_acc']:.3f}")
    print(f"  Leap-year accuracy             : {metrics['leap_acc']:.3f}")
    print(f"  Decade accuracy                : {metrics['decade_acc']:.3f}")

    print(f"\n── Sample Generated Outputs (first {n_show}) ──")
    header = f"{'Condition':<38}  {'Generated':>12}  {'DOW':>4}  {'MON':>4}  {'LEAP':>5}  {'DEC':>4}  {'ALL':>4}"
    print(header)
    print("-" * len(header))

    for i in range(min(n_show, len(X_all))):
        c   = X_all[i].tolist()
        gen = Y_gen_all[i].tolist()
        m_i = check_conditions(Y_gen_all[i:i+1], X_all[i:i+1])

        cond_str = (f"[{REV_DOW[c[0]]}][{REV_MON[c[1]]}]"
                    f"[{'T' if c[2] else 'F'}][{c[3]}]")
        date_str = f"{gen[0]:02d}-{gen[1]:02d}-{gen[2]:04d}"

        def tick(v): return "✓" if v == 1.0 else "✗"

        print(
            f"{cond_str:<38}  {date_str:>12}  "
            f"{tick(m_i['dow_acc']):>4}  {tick(m_i['mon_acc']):>4}  "
            f"{tick(m_i['leap_acc']):>5}  {tick(m_i['decade_acc']):>4}  "
            f"{tick(m_i['csr']):>4}"
        )

    return metrics