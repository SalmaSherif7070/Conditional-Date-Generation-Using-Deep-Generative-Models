"""
Model 3 visualisation — mirrors visualization patterns from Models 1 & 2.
Produces CVAE-specific plots (ELBO loss, reconstruction vs KL, β schedule,
CSR curve, per-condition accuracy bar chart, combined overview).
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
    "recon":  "#55A868",
    "kl":     "#C44E52",
    "csr":    "#8172B2",
    "beta":   "#CCB974",
}


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── ELBO loss curves ──────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, save_dir: str):
    """Train / Val ELBO loss over epochs."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["train_loss"], lw=2,
                color=COLORS["train"], label="Train ELBO Loss")
        ax.plot(history["epochs"], history["val_loss"],   lw=2,
                color=COLORS["val"],   label="Val ELBO Loss", ls="--")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("ELBO Loss", fontsize=12)
        ax.set_title("Model 3 – ELBO Loss (Train vs Val)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model3_loss_curves.png"))


def plot_loss_log_scale(history: dict, save_dir: str):
    """Train / Val ELBO loss on a log scale."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(history["epochs"], history["train_loss"], lw=2,
                    color=COLORS["train"], label="Train ELBO Loss")
        ax.semilogy(history["epochs"], history["val_loss"],   lw=2,
                    color=COLORS["val"],   label="Val ELBO Loss", ls="--")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("ELBO Loss (log scale)", fontsize=12)
        ax.set_title("Model 3 – Loss Log Scale", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, which="both")
        _save(fig, os.path.join(save_dir, "model3_loss_log.png"))


# ── CVAE-specific plots ───────────────────────────────────────────────────────

def plot_recon_kl_split(history: dict, save_dir: str):
    """Reconstruction loss vs KL divergence over training."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["recon_loss"], lw=2,
                color=COLORS["recon"], label="Reconstruction Loss")
        ax2 = ax.twinx()
        ax2.plot(history["epochs"], history["kl_loss"], lw=2,
                 color=COLORS["kl"], ls="--", label="KL Divergence")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Reconstruction Loss", fontsize=12, color=COLORS["recon"])
        ax2.set_ylabel("KL Divergence",       fontsize=12, color=COLORS["kl"])
        ax.set_title("Model 3 – Reconstruction vs KL", fontsize=14)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model3_recon_kl.png"))


def plot_beta_schedule(history: dict, save_dir: str):
    """β annealing schedule over training epochs."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history["epochs"], history["beta"], lw=2,
                color=COLORS["beta"], label="β (KL weight)")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("β", fontsize=12)
        ax.set_title("Model 3 – β Annealing Schedule", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model3_beta_schedule.png"))


def plot_csr_curve(history: dict, save_dir: str):
    """Condition Satisfaction Rate on validation set over epochs."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(history["epochs"], history["csr"], lw=2,
                color=COLORS["csr"], label="Val CSR")
        ax.axhline(1.0, ls=":", lw=1, color="grey", label="Perfect CSR")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Condition Satisfaction Rate", fontsize=12)
        ax.set_title("Model 3 – Validation CSR Over Training", fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        _save(fig, os.path.join(save_dir, "model3_csr_curve.png"))


# ── Per-condition accuracy (matches Models 1 & 2) ────────────────────────────

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
        ax.set_title("Model 3 – Per-Condition Accuracy", fontsize=14)
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
        _save(fig, os.path.join(save_dir, "model3_condition_breakdown.png"))


def plot_combined_overview(history: dict, metrics: dict, save_dir: str):
    """2×2 overview: ELBO loss | CSR curve | Recon/KL | condition breakdown."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Model 3 – Training Overview (Conditional VAE)", fontsize=15)

        # Top-left: ELBO loss
        ax = axes[0, 0]
        ax.plot(history["epochs"], history["train_loss"], lw=2,
                color=COLORS["train"], label="Train")
        ax.plot(history["epochs"], history["val_loss"],   lw=2,
                color=COLORS["val"],   label="Val", ls="--")
        ax.set_title("ELBO Loss")
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

        # Bottom-left: Recon vs KL
        ax    = axes[1, 0]
        ax2   = ax.twinx()
        ax.plot(history["epochs"], history["recon_loss"], lw=2,
                color=COLORS["recon"], label="Recon")
        ax2.plot(history["epochs"], history["kl_loss"],   lw=2,
                 color=COLORS["kl"], ls="--", label="KL")
        ax.set_title("Recon vs KL")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Recon", color=COLORS["recon"])
        ax2.set_ylabel("KL",   color=COLORS["kl"])
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9)
        ax.grid(alpha=0.3)

        # Bottom-right: condition breakdown bar
        ax = axes[1, 1]
        labels = ["DOW", "Month", "Leap", "Decade", "All"]
        values = [metrics["dow_acc"], metrics["mon_acc"],
                  metrics["leap_acc"], metrics["decade_acc"], metrics["csr"]]
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
        _save(fig, os.path.join(save_dir, "model3_overview.png"))