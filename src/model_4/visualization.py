"""
src/model_4/visualization.py
Three plots: CD loss curves, loss log scale, per-condition accuracy bar chart.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

STYLE  = {"figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False}
C_TR   = "#4C72B0"
C_VAL  = "#DD8452"
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_curves(history: dict, save_dir: str):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["train_loss"], lw=2, color=C_TR,  label="Train CD Loss")
        ax.plot(history["epochs"], history["val_loss"],   lw=2, color=C_VAL, label="Val CD Loss", ls="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("CD Loss")
        ax.set_title("Model 4 – Contrastive Divergence Loss"); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model4_loss_curves.png"))


def plot_loss_log_scale(history: dict, save_dir: str):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(history["epochs"], [abs(v) for v in history["train_loss"]], lw=2, color=C_TR,  label="|Train|")
        ax.semilogy(history["epochs"], [abs(v) for v in history["val_loss"]],   lw=2, color=C_VAL, label="|Val|", ls="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("|CD Loss| (log scale)")
        ax.set_title("Model 4 – Loss Log Scale"); ax.legend(); ax.grid(alpha=0.3, which="both")
        _save(fig, os.path.join(save_dir, "model4_loss_log.png"))


def plot_condition_breakdown(metrics: dict, save_dir: str):
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