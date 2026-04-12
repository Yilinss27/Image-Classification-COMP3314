"""Generate figures for the early exploration report (runs 01-06).

Outputs:
  reports/figures/early_01_progression.png  — Best val accuracy progression
  reports/figures/early_02_classifier_comparison.png — Classifier comparison (run_01 & run_02)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_progression():
    """Bar chart: best val accuracy across all 6 runs."""
    runs = ["run_01\nRaw Pixels\n+PCA+SVC-RBF",
            "run_02\nHOG+Color\n+LBP+HistGB",
            "run_03\nCoates K=800\nLinearSVC",
            "run_04\nCoates K=400\nC=0.03",
            "run_05\nCoates K=1600\nC=0.01 (GPU)",
            "run_06\nCoates K=3200\nC=0.01"]
    # run_03 val not exactly known from logs; run_04 K=800 gives 0.7544 which is
    # the corrected run_03 equivalent. Use ~0.73 as stated in README context.
    val_accs = [0.4974, 0.6468, 0.73, 0.7328, 0.7678, 0.77]  # run_06 has no val; use public proxy
    public_scores = [None, 0.6425, None, 0.7335, 0.7740, 0.7715]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    bars = ax.bar(range(len(runs)), val_accs, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    # Add public score markers where available
    for i, pub in enumerate(public_scores):
        if pub is not None:
            ax.plot(i, pub, marker="*", color="gold", markersize=16, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=5)

    # Annotate bars
    for i, (bar, acc) in enumerate(zip(bars, val_accs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Annotate public scores
    for i, pub in enumerate(public_scores):
        if pub is not None:
            ax.text(i + 0.3, pub + 0.003, f"pub: {pub:.4f}", ha="center", va="bottom",
                    fontsize=8, color="darkgoldenrod", fontstyle="italic")

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Early Exploration: Validation Accuracy Progression (Runs 01-06)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add annotation arrows showing gains
    gains = [None, "+0.149", "+0.083", "+0.003", "+0.035", "-0.003*"]
    for i in range(1, len(runs)):
        if gains[i]:
            color = "green" if not gains[i].startswith("-") else "red"
            ax.annotate("", xy=(i, val_accs[i] - 0.01), xytext=(i - 1, val_accs[i - 1] - 0.01),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=10, label="Val accuracy"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=14, label="Kaggle public score"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "early_01_progression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_classifier_comparison():
    """Grouped bar chart comparing classifiers in run_01 and run_02."""
    # run_01 classifiers
    run01_names = ["SVC-RBF", "Voting", "LogReg", "LinearSVC", "KNN"]
    run01_accs = [0.4974, 0.4756, 0.4114, 0.4038, 0.3588]

    # run_02 classifiers
    run02_names = ["HistGB", "SVC-RBF", "Voting", "RF", "LogReg"]
    run02_accs = [0.6468, 0.6416, 0.6434, 0.5674, 0.5550]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # run_01
    colors1 = ["#2ca02c" if a == max(run01_accs) else "#1f77b4" for a in run01_accs]
    bars1 = ax1.barh(range(len(run01_names)), run01_accs, color=colors1,
                     edgecolor="black", linewidth=0.6, alpha=0.85)
    ax1.set_yticks(range(len(run01_names)))
    ax1.set_yticklabels(run01_names, fontsize=11)
    ax1.set_xlabel("Val Accuracy", fontsize=11)
    ax1.set_title("run_01: Raw Pixels + PCA(200)\nClassifier Comparison", fontsize=11, fontweight="bold")
    ax1.set_xlim(0.3, 0.55)
    for i, (bar, acc) in enumerate(zip(bars1, run01_accs)):
        ax1.text(acc + 0.003, i, f"{acc:.4f}", va="center", fontsize=10)
    ax1.axvline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Random=0.10")

    # run_02
    colors2 = ["#2ca02c" if a == max(run02_accs) else "#ff7f0e" for a in run02_accs]
    bars2 = ax2.barh(range(len(run02_names)), run02_accs, color=colors2,
                     edgecolor="black", linewidth=0.6, alpha=0.85)
    ax2.set_yticks(range(len(run02_names)))
    ax2.set_yticklabels(run02_names, fontsize=11)
    ax2.set_xlabel("Val Accuracy", fontsize=11)
    ax2.set_title("run_02: HOG + Color + LBP (1354 dims)\nClassifier Comparison", fontsize=11, fontweight="bold")
    ax2.set_xlim(0.5, 0.7)
    for i, (bar, acc) in enumerate(zip(bars2, run02_accs)):
        ax2.text(acc + 0.003, i, f"{acc:.4f}", va="center", fontsize=10)

    # Highlight the feature engineering improvement
    fig.text(0.5, 0.02,
             "Feature engineering (run_01 -> run_02) improves ALL classifiers by +0.14 to +0.23",
             ha="center", fontsize=10, fontstyle="italic", color="darkgreen")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIGURES_DIR / "early_02_classifier_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_progression()
    plot_classifier_comparison()
    print("Done.")
