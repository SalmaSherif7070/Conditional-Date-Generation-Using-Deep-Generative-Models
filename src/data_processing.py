import re
import torch
from torch.utils.data import TensorDataset, Subset

DOW_MAP = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
MON_MAP = {"JAN": 0, "FEB": 1, "MAR": 2, "APR": 3, "MAY": 4, "JUN": 5,
           "JUL": 6, "AUG": 7, "SEP": 8, "OCT": 9, "NOV": 10, "DEC": 11}
REV_DOW = {v: k for k, v in DOW_MAP.items()}
REV_MON = {v: k for k, v in MON_MAP.items()}


def _parse_data_line(line: str):
    """Parse a full data line: [DOW] [MON] [LEAP] [DEC] d-m-yyyy"""
    m = re.match(r'\[(.*?)\] \[(.*?)\] \[(.*?)\] \[(.*?)\] (\S+)', line.strip())
    if not m:
        return None, None
    dow_s, mon_s, leap_s, dec_s, date_s = m.groups()
    x = [
        DOW_MAP.get(dow_s.upper(), 0),
        MON_MAP.get(mon_s.upper(), 0),
        1 if leap_s.strip().lower() == "true" else 0,
        int(dec_s),
    ]
    d, mo, y = map(int, date_s.split("-"))
    return x, [d, mo, y]


def _parse_condition_line(line: str):
    """Parse a condition-only line (no date)."""
    m = re.match(r'\[(.*?)\] \[(.*?)\] \[(.*?)\] \[(.*?)\]', line.strip())
    if not m:
        return None
    dow_s, mon_s, leap_s, dec_s = m.groups()
    return [
        DOW_MAP.get(dow_s.upper(), 0),
        MON_MAP.get(mon_s.upper(), 0),
        1 if leap_s.strip().lower() == "true" else 0,
        int(dec_s),
    ]


def load_dataset(data_path: str, val_split: float = 0.2, seed: int = 42):
    """
    Load data.txt and split 80/20. Test == validation (same 20% split).
    Returns train_ds, val_ds, full X tensor, full Y tensor.
    """
    X_list, Y_list = [], []
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            x, y = _parse_data_line(line)
            if x is not None:
                X_list.append(x)
                Y_list.append(y)

    X = torch.tensor(X_list, dtype=torch.long)
    Y = torch.tensor(Y_list, dtype=torch.long)
    dataset = TensorDataset(X, Y)

    n = len(dataset)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    train_ds = Subset(dataset, indices[:n_train])
    val_ds   = Subset(dataset, indices[n_train:])

    return train_ds, val_ds, X, Y


def load_example_input(path: str):
    """Load condition-only example_input.txt for inference."""
    conditions, raw_lines = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            c = _parse_condition_line(line)
            if c is not None:
                conditions.append(c)
                raw_lines.append(line.strip())
    return torch.tensor(conditions, dtype=torch.long), raw_lines


def format_output_line(cond: list, date: list) -> str:
    """Format a condition + date into the data.txt format."""
    d, m, y = date
    return (f"[{REV_DOW[cond[0]]}] [{REV_MON[cond[1]]}] "
            f"[{'True' if cond[2] else 'False'}] [{cond[3]}] {d}-{m}-{y}")