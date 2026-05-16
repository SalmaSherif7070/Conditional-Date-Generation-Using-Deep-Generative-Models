"""
Model 4 visualisation — mirrors src/model_3/visualization.py.
Produces EBM-specific plots (CD loss, energy gap, CSR curve,
per-condition accuracy bar chart, combined overview).
"""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

STYLE = {
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

COLORS = {
    "train":  "#4C72B0",
    "val":    "#DD8452",
    "e_pos":  "#C44E52",
    "e_neg":  "#55A868",
    "gap":    "#8172B2",
    "csr":    "#CCB974",
}


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── CD loss curves ────────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, save_dir: str):
    """Train / Val Contrastive Divergence loss over epochs."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["train_loss"], lw=2,
                color=COLORS["train"], label="Train CD Loss")
        ax.plot(history["epochs"], history["val_loss"],   lw=2,
                color=COLORS["val"],   label="Val CD Loss", ls="--")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Contrastive Divergence Loss", fontsize=12)
        ax.set_title("Model 4 – CD Loss (Train vs Val)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model4_loss_curves.png"))


def plot_loss_log_scale(history: dict, save_dir: str):
    """Train / Val CD loss on a log scale (absolute values)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        # CD loss can be negative — plot |loss| on log scale
        train_abs = [abs(v) for v in history["train_loss"]]
        val_abs   = [abs(v) for v in history["val_loss"]]
        ax.semilogy(history["epochs"], train_abs, lw=2,
                    color=COLORS["train"], label="|Train CD Loss|")
        ax.semilogy(history["epochs"], val_abs,   lw=2,
                    color=COLORS["val"],   label="|Val CD Loss|", ls="--")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("|CD Loss| (log scale)", fontsize=12)
        ax.set_title("Model 4 – Loss Log Scale", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, which="both")
        _save(fig, os.path.join(save_dir, "model4_loss_log.png"))


# ── EBM-specific plots ────────────────────────────────────────────────────────

def plot_energy_curves(history: dict, save_dir: str):
    """
    E(x⁺) and E(x⁻) over training.
    A well-trained EBM should have E(x⁺) < E(x⁻).
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["e_pos"], lw=2,
                color=COLORS["e_pos"], label="E(real)  E⁺")
        ax.plot(history["epochs"], history["e_neg"], lw=2,
                color=COLORS["e_neg"], label="E(MCMC)  E⁻", ls="--")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Mean Energy", fontsize=12)
        ax.set_title("Model 4 – Energy of Real vs MCMC Samples", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model4_energy_curves.png"))


def plot_energy_gap(history: dict, save_dir: str):
    """
    Energy gap  Δ = E⁺ − E⁻  over training.
    Negative gap means real samples have lower energy than MCMC samples
    (the desired behaviour).
    """
    gap = [ep - en for ep, en in zip(history["e_pos"], history["e_neg"])]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history["epochs"], gap, lw=2, color=COLORS["gap"], label="E⁺ − E⁻")
        ax.axhline(0.0, ls=":", lw=1, color="grey", label="Zero gap")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Energy Gap", fontsize=12)
        ax.set_title("Model 4 – Energy Gap (E⁺ − E⁻)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model4_energy_gap.png"))


def plot_csr_curve(history: dict, save_dir: str):
    """Condition Satisfaction Rate on the validation set over epochs."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["csr"], lw=2,
                color=COLORS["csr"], label="Val CSR")
        ax.axhline(1.0, ls=":", lw=1, color="grey", label="Perfect CSR")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Condition Satisfaction Rate", fontsize=12)
        ax.set_title("Model 4 – Validation CSR Over Training", fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model4_csr_curve.png"))


# ── Per-condition accuracy (matches Models 1–3) ───────────────────────────────

def plot_condition_breakdown(metrics: dict, save_dir: str):
    """Bar chart of per-condition accuracy — identical layout to other models."""
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
        ax.set_title("Model 4 – Per-Condition Accuracy", fontsize=14)
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
        _save(fig, os.path.join(save_dir, "model4_condition_breakdown.png"))


def plot_combined_overview(history: dict, metrics: dict, save_dir: str):
    """2×2 overview: CD loss | CSR curve | Energy curves | condition breakdown."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Model 4 – Training Overview (Energy-Based Model)", fontsize=15)

        # Top-left: CD loss
        ax = axes[0, 0]
        ax.plot(history["epochs"], history["train_loss"], lw=2,
                color=COLORS["train"], label="Train")
        ax.plot(history["epochs"], history["val_loss"],   lw=2,
                color=COLORS["val"],   label="Val", ls="--")
        ax.set_title("CD Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # Top-right: CSR
        ax = axes[0, 1]
        ax.plot(history["epochs"], history["csr"], lw=2, color=COLORS["csr"])
        ax.axhline(1.0, ls=":", lw=1, color="grey")
        ax.set_title("Validation CSR")
        ax.set_xlabel("Epoch"); ax.set_ylabel("CSR")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(alpha=0.3)

        # Bottom-left: Energy curves
        ax = axes[1, 0]
        ax.plot(history["epochs"], history["e_pos"], lw=2,
                color=COLORS["e_pos"], label="E⁺ (real)")
        ax.plot(history["epochs"], history["e_neg"], lw=2,
                color=COLORS["e_neg"], label="E⁻ (MCMC)", ls="--")
        ax.set_title("Energy: Real vs MCMC")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Energy")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # Bottom-right: condition breakdown bar
        ax = axes[1, 1]
        labels = ["DOW", "Month", "Leap", "Decade", "All"]
        values = [
            metrics["dow_acc"], metrics["mon_acc"],
            metrics["leap_acc"], metrics["decade_acc"], metrics["csr"],
        ]
        bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white")
        ax.set_title("Per-Condition Accuracy")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        _save(fig, os.path.join(save_dir, "model4_overview.png"))