"""
src/model_4/visualization.py
Plots: total loss curves, MSE vs CE loss, CSR curve, per-condition accuracy.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

STYLE  = {"figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False}
C_TR   = "#4C72B0"
C_VAL  = "#DD8452"
C_MSE  = "#55A868"
C_CE   = "#C44E52"
C_CSR  = "#CCB974"
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_curves(history: dict, save_dir: str):
    """Train vs Val total loss."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["train_loss"], lw=2, color=C_TR,  label="Train Loss")
        ax.plot(history["epochs"], history["val_loss"],   lw=2, color=C_VAL, label="Val Loss", ls="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Model 4 – Diffusion Loss (Train vs Val)"); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model4_loss_curves.png"))


def plot_loss_log_scale(history: dict, save_dir: str):
    """Train vs Val loss on log scale."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(history["epochs"], history["train_loss"], lw=2, color=C_TR,  label="Train")
        ax.semilogy(history["epochs"], history["val_loss"],   lw=2, color=C_VAL, label="Val", ls="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
        ax.set_title("Model 4 – Loss Log Scale"); ax.legend(); ax.grid(alpha=0.3, which="both")
        _save(fig, os.path.join(save_dir, "model4_loss_log.png"))


def plot_mse_ce_curves(history: dict, save_dir: str):
    """MSE (embedding) vs CE (discrete heads) loss components."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["mse_loss"], lw=2, color=C_MSE, label="MSE (embedding)")
        ax.plot(history["epochs"], history["ce_loss"],  lw=2, color=C_CE,  label="CE (discrete)", ls="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Model 4 – MSE vs Cross-Entropy Loss Components")
        ax.legend(); ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model4_mse_ce_curves.png"))


def plot_csr_curve(history: dict, save_dir: str):
    """Condition Satisfaction Rate on validation set over epochs."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["csr"], lw=2, color=C_CSR, label="Val CSR")
        ax.axhline(1.0, ls=":", lw=1, color="grey", label="Perfect CSR")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Condition Satisfaction Rate")
        ax.set_title("Model 4 – Validation CSR Over Training")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(); ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model4_csr_curve.png"))


def plot_condition_breakdown(metrics: dict, save_dir: str):
    """Per-condition accuracy bar chart — same layout as Models 1–3."""
    labels = ["Day of Week", "Month", "Leap Year", "Decade", "All (CSR)"]
    values = [
        metrics["dow_acc"], metrics["mon_acc"], metrics["leap_acc"],
        metrics["decade_acc"], metrics["csr"],
    ]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=COLORS, edgecolor="white")
        ax.set_ylabel("Accuracy"); ax.set_title("Model 4 – Per-Condition Accuracy")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        _save(fig, os.path.join(save_dir, "model4_condition_breakdown.png"))