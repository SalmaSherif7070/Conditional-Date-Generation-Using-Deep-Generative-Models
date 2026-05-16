"""
Model 2 visualisation — mirrors src/visualization.py for Model 1.
Produces GAN-specific plots (D/G loss curves, GP curve, CSR over epochs,
Gumbel temperature schedule, per-condition accuracy bar chart).
"""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

STYLE = {
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Loss curves ───────────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, save_dir: str):
    """
    Discriminator and Generator loss over training epochs.
    Mirrors Model 1's plot_loss_curves() with GAN-appropriate labelling.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["d_loss"], lw=2, label="Discriminator Loss (WGAN)")
        ax.plot(history["epochs"], history["g_loss"], lw=2, ls="--", label="Generator Loss")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Model 2 – Discriminator & Generator Loss", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model2_loss_curves.png"))


def plot_loss_log_scale(history: dict, save_dir: str):
    """
    Absolute value of D / G loss on a log scale to reveal convergence behaviour.
    Mirrors Model 1's plot_loss_log_scale().
    """
    d_abs = [abs(v) for v in history["d_loss"]]
    g_abs = [abs(v) for v in history["g_loss"]]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(history["epochs"], d_abs, lw=2, label="|Discriminator Loss|")
        ax.semilogy(history["epochs"], g_abs, lw=2, ls="--", label="|Generator Loss|")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (log scale)", fontsize=12)
        ax.set_title("Model 2 – Loss (Log Scale)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, which="both")
        _save(fig, os.path.join(save_dir, "model2_loss_log.png"))


# ── GAN-specific plots ────────────────────────────────────────────────────────

def plot_gradient_penalty(history: dict, save_dir: str):
    """Gradient penalty value over epochs — useful GAN health indicator."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history["epochs"], history["gp"], lw=2, color="#C44E52", label="Gradient Penalty")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("GP Value", fontsize=12)
        ax.set_title("Model 2 – Gradient Penalty Over Training", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model2_gradient_penalty.png"))


def plot_csr_curve(history: dict, save_dir: str):
    """
    Condition Satisfaction Rate (CSR) on the validation set over epochs.
    GAN-specific equivalent of plotting validation accuracy.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["csr"], lw=2, color="#55A868", label="Val CSR")
        ax.axhline(1.0, ls=":", lw=1, color="grey", label="Perfect CSR")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Condition Satisfaction Rate", fontsize=12)
        ax.set_title("Model 2 – Validation CSR Over Training", fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model2_csr_curve.png"))


def plot_gumbel_temperature(history: dict, save_dir: str):
    """
    Gumbel temperature annealing schedule.
    Shows how the generator transitions from soft → hard discrete samples.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history["epochs"], history["tau"], lw=2, color="#8172B2", label="Gumbel τ")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Temperature (τ)", fontsize=12)
        ax.set_title("Model 2 – Gumbel-Softmax Temperature Schedule", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model2_gumbel_temperature.png"))


# ── Per-condition accuracy (matches Model 1) ──────────────────────────────────

def plot_condition_breakdown(metrics: dict, save_dir: str):
    """
    Bar chart of per-condition accuracy.
    Identical layout to Model 1's plot_condition_breakdown() for easy comparison.
    """
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
        ax.set_title("Model 2 – Per-Condition Accuracy", fontsize=14)
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
        _save(fig, os.path.join(save_dir, "model2_condition_breakdown.png"))


def plot_combined_overview(history: dict, metrics: dict, save_dir: str):
    """
    2×2 overview figure: D/G loss | CSR curve | GP | condition breakdown.
    Gives a single-glance summary of Model 2 training.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Model 2 – Training Overview (Conditional GAN)", fontsize=15)

        # Top-left: D/G loss
        ax = axes[0, 0]
        ax.plot(history["epochs"], history["d_loss"], lw=2, label="Discriminator")
        ax.plot(history["epochs"], history["g_loss"], lw=2, ls="--", label="Generator")
        ax.set_title("D & G Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # Top-right: CSR
        ax = axes[0, 1]
        ax.plot(history["epochs"], history["csr"], lw=2, color="#55A868")
        ax.axhline(1.0, ls=":", lw=1, color="grey")
        ax.set_title("Validation CSR")
        ax.set_xlabel("Epoch"); ax.set_ylabel("CSR")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(alpha=0.3)

        # Bottom-left: Gradient penalty
        ax = axes[1, 0]
        ax.plot(history["epochs"], history["gp"], lw=2, color="#C44E52")
        ax.set_title("Gradient Penalty")
        ax.set_xlabel("Epoch"); ax.set_ylabel("GP")
        ax.grid(alpha=0.3)

        # Bottom-right: condition breakdown bar
        ax = axes[1, 1]
        labels = ["DOW", "Month", "Leap", "Decade", "All"]
        values = [metrics["dow_acc"], metrics["mon_acc"],
                  metrics["leap_acc"], metrics["decade_acc"], metrics["csr"]]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
        bars = ax.bar(labels, values, color=colors, edgecolor="white")
        ax.set_title("Per-Condition Accuracy")
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        _save(fig, os.path.join(save_dir, "model2_overview.png"))