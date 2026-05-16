import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

STYLE = {"figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False}


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_loss_curves(history: dict, save_dir: str):
    """Train vs validation KL-divergence loss over epochs."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["train_loss"], lw=2, label="Train Loss")
        ax.plot(history["epochs"], history["val_loss"],   lw=2, ls="--", label="Val / Test Loss")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("KL Divergence Loss", fontsize=12)
        ax.set_title("Model 1 – Training & Validation Loss", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model1_loss_curves.png"))


def plot_condition_breakdown(metrics: dict, save_dir: str):
    """Bar chart of per-condition accuracy."""
    labels = ["Day of Week", "Month", "Leap Year", "Decade", "All (CSR)"]
    values = [
        metrics["dow_acc"],
        metrics["mon_acc"],
        metrics["leap_acc"],
        metrics["decade_acc"],
        metrics["csr"],
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Model 1 – Per-Condition Accuracy", fontsize=14)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.1%}",
                ha="center", va="bottom", fontsize=10,
            )
        ax.grid(axis="y", alpha=0.3)
        _save(fig, os.path.join(save_dir, "model1_condition_breakdown.png"))


def plot_loss_log_scale(history: dict, save_dir: str):
    """Loss on log scale to better see convergence behaviour."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(history["epochs"], history["train_loss"], lw=2, label="Train Loss")
        ax.semilogy(history["epochs"], history["val_loss"],   lw=2, ls="--", label="Val / Test Loss")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("KL Divergence Loss (log scale)", fontsize=12)
        ax.set_title("Model 1 – Loss (Log Scale)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, which="both")
        _save(fig, os.path.join(save_dir, "model1_loss_log.png"))