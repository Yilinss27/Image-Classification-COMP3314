"""
Dataset analysis for COMP3314 Image Classification report.
Generates three figures:
  1. One example per class (2x5 grid)
  2. Class distribution bar chart
  3. Image statistics (mean/std distributions + average image)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
TRAIN_IMS = os.path.join(DATA, "train_ims")


def load_image(fname):
    return np.array(Image.open(os.path.join(TRAIN_IMS, fname)))


# ── Figure 1: One example per class ──────────────────────────────────────────

def fig1_class_examples():
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    np.random.seed(42)
    for cls in range(10):
        row = train_df[train_df["label"] == cls].sample(1, random_state=42).iloc[0]
        img = load_image(row["im_name"])
        axes[cls].imshow(img)
        axes[cls].set_title(f"Class {cls}", fontsize=12)
        axes[cls].axis("off")
    fig.suptitle("One Random Example per Class", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "report_01_class_examples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 2: Class distribution ─────────────────────────────────────────────

def fig2_class_distribution():
    counts = train_df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color="steelblue", edgecolor="black")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 30, str(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Class Label", fontsize=12)
    ax.set_ylabel("Number of Training Samples", fontsize=12)
    ax.set_xticks(range(10))
    ax.set_title("Training Set Class Distribution", fontsize=14)
    ax.set_ylim(0, counts.max() * 1.12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "report_02_class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 3: Image statistics ───────────────────────────────────────────────

def fig3_image_stats():
    n = len(train_df)
    means = np.empty(n, dtype=np.float32)
    stds = np.empty(n, dtype=np.float32)
    running_sum = np.zeros((32, 32, 3), dtype=np.float64)

    for i, fname in enumerate(train_df["im_name"]):
        img = load_image(fname).astype(np.float32)
        means[i] = img.mean()
        stds[i] = img.std()
        running_sum += img
        if (i + 1) % 10000 == 0:
            print(f"  processed {i + 1}/{n} images")

    avg_img = (running_sum / n).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(means, bins=60, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Mean Pixel Intensity")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) Distribution of Mean Intensity")

    axes[1].hist(stds, bins=60, color="coral", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Std Pixel Intensity")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) Distribution of Std Intensity")

    axes[2].imshow(avg_img)
    axes[2].set_title("(c) Average Training Image")
    axes[2].axis("off")

    fig.suptitle("Training Set Image Statistics", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "report_03_image_stats.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    print("=== Figure 1: Class examples ===")
    fig1_class_examples()
    print("=== Figure 2: Class distribution ===")
    fig2_class_distribution()
    print("=== Figure 3: Image statistics ===")
    fig3_image_stats()
    print("Done.")
