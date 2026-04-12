"""Comprehensive analysis and visualization for all post-run_07 experiments.

Generates figures covering:
  1. Overall progression waterfall (public score timeline)
  2. Phase A: val vs K with fine C grid + K-C* relationship
  3. Flip augmentation: before/after comparison
  4. Power normalization: 3-way comparison
  5. Failed experiments summary
  6. Ensemble: individual vs ensemble comparison
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# Figure 1: Public score progression (waterfall)
# ============================================================
submissions = [
    ("run_02\nHOG+HistGB", 0.64250, "#9E9E9E"),
    ("run_04\nCoates K=400", 0.73350, "#9E9E9E"),
    ("run_05\nK=1600 GPU", 0.77400, "#9E9E9E"),
    ("run_07\n250-config\nsweep", 0.78400, "#2196F3"),
    ("run_08\nPhase B\nlow C", 0.78750, "#4CAF50"),
    ("run_10\nPhase A\nK=10000", 0.78600, "#FF9800"),
    ("run_12\n+flip aug", 0.81550, "#E91E63"),
    ("run_14\nTTA10\n(failed)", 0.80000, "#F44336"),
    ("run_17\n+power\nnorm", 0.82700, "#9C27B0"),
]

fig, ax = plt.subplots(figsize=(14, 6))
x = range(len(submissions))
labels, scores, colors = zip(*submissions)
bars = ax.bar(x, scores, color=colors, width=0.7, edgecolor="white", linewidth=1.5)
for i, (lbl, sc, _) in enumerate(submissions):
    ax.text(i, sc + 0.003, f"{sc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Kaggle Public Score", fontsize=12)
ax.set_title("Public Score Progression — from HOG baseline to final pnorm model", fontsize=14)
ax.set_ylim(0.60, 0.85)
ax.axhline(0.82800, color="red", linestyle="--", linewidth=1.5, label="1st place (0.82800)")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "post07_01_public_progression.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'post07_01_public_progression.png'}")

# ============================================================
# Figure 2: Phase A — val vs K for P=6 with fine C grid
# ============================================================
# Data from run_10 Phase A
phase_a_p6 = {
    10000: {0.0005: 0.7772, 0.001: 0.7798, 0.0015: 0.7836, 0.002: 0.7860,
            0.0025: 0.7850, 0.003: 0.7860, 0.0035: 0.7846, 0.004: 0.7844},
    12000: {0.0005: 0.7792, 0.001: 0.7848, 0.0015: 0.7836, 0.002: 0.7826,
            0.0025: 0.7812, 0.003: 0.7808, 0.0035: 0.7810, 0.004: 0.7804},
    14000: {0.0005: 0.7816, 0.001: 0.7792, 0.0015: 0.7780, 0.002: 0.7770,
            0.0025: 0.7748, 0.003: None, 0.0035: 0.7706, 0.004: None},
}
# Add run_07 K=8000 for reference
run07_k8000 = {0.003: 0.7858, 0.005: 0.7842, 0.01: 0.7780, 0.02: 0.7626, 0.03: 0.7540}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: val vs C for each K
ax = axes[0]
c_colors = {8000: "#E53935", 10000: "#2E86C1", 12000: "#4CAF50", 14000: "#FB8C00"}
# K=8000 from run_07 (only C values that overlap with Phase A range)
k8000_phase_b = {0.0005: 0.7806, 0.001: 0.7848, 0.002: 0.7840, 0.003: 0.7858}
Cs_8k = sorted(k8000_phase_b.keys())
ax.plot(Cs_8k, [k8000_phase_b[c] for c in Cs_8k], "o-", color=c_colors[8000],
        linewidth=2, markersize=7, label="K=8000")
for K in [10000, 12000, 14000]:
    data = phase_a_p6[K]
    Cs = sorted([c for c, v in data.items() if v is not None])
    vals = [data[c] for c in Cs]
    ax.plot(Cs, vals, "o-", color=c_colors[K], linewidth=2, markersize=7, label=f"K={K}")
ax.set_xscale("log")
ax.set_xlabel("C (regularization inverse)", fontsize=12)
ax.set_ylabel("Validation accuracy", fontsize=12)
ax.set_title("Phase A: P=6 val vs C for different K\n(K→C* inverse trend)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: optimal C* vs K
ax = axes[1]
k_cstar = [(8000, 0.003), (10000, 0.002), (12000, 0.001), (14000, 0.0005)]
Ks, Cstars = zip(*k_cstar)
best_vals = [0.7858, 0.7860, 0.7848, 0.7816]
ax2 = ax.twinx()
ax.plot(Ks, Cstars, "s-", color="#E53935", linewidth=2.5, markersize=10, label="C* (optimal C)")
ax2.plot(Ks, best_vals, "o--", color="#2E86C1", linewidth=2, markersize=8, label="best val at C*")
ax.set_xlabel("Dictionary size K", fontsize=12)
ax.set_ylabel("Optimal C*", fontsize=12, color="#E53935")
ax2.set_ylabel("Best val accuracy", fontsize=12, color="#2E86C1")
ax.set_title("K→C* inverse relationship\n(larger K needs stronger regularization)", fontsize=13)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "post07_02_phase_a_analysis.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'post07_02_phase_a_analysis.png'}")

# ============================================================
# Figure 3: Flip augmentation impact
# ============================================================
configs = ["P=6 K=8000\nC=0.002", "P=7 K=6000\nC=0.002", "P=8 K=3200\nC=0.003"]
val_no_aug = [0.7836, 0.7876, 0.7816]
val_flip = [0.8122, 0.8078, 0.8006]
val_flip_pnorm = [0.8130, 0.8098, 0.7998]

x = np.arange(len(configs))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w, val_no_aug, w, label="No augmentation", color="#90CAF9")
b2 = ax.bar(x, val_flip, w, label="+ Flip augmentation", color="#2196F3")
b3 = ax.bar(x + w, val_flip_pnorm, w, label="+ Flip + Power norm", color="#0D47A1")
for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=11)
ax.set_ylabel("Validation accuracy", fontsize=12)
ax.set_title("Impact of flip augmentation and power normalization\nacross different (P, K, C) configurations", fontsize=13)
ax.set_ylim(0.77, 0.82)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "post07_03_flip_pnorm_impact.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'post07_03_flip_pnorm_impact.png'}")

# ============================================================
# Figure 4: Power norm comparison (3 normalization strategies)
# ============================================================
C_vals = [0.001, 0.002, 0.003, 0.005]
baseline = [0.8088, 0.8118, 0.8114, 0.8068]
pnorm = [0.8112, 0.8136, 0.8110, 0.8110]
pnorm_l2 = [0.8110, 0.8126, 0.8124, 0.8098]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(C_vals, baseline, "o-", linewidth=2.5, markersize=9, label="Baseline (StandardScaler only)", color="#90CAF9")
ax.plot(C_vals, pnorm, "s-", linewidth=2.5, markersize=9, label="Power norm + StandardScaler", color="#E53935")
ax.plot(C_vals, pnorm_l2, "^-", linewidth=2.5, markersize=9, label="Power norm + L2 + StandardScaler", color="#4CAF50")
ax.set_xscale("log")
ax.set_xlabel("C", fontsize=12)
ax.set_ylabel("Validation accuracy (flip aug, 90k train)", fontsize=12)
ax.set_title("Power normalization: signed sqrt consistently improves val\nL2 normalization on top does NOT help", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
# Annotate the winner
ax.annotate("Best: pnorm C=0.002\nval=0.8136 → public 0.82700",
            xy=(0.002, 0.8136), xytext=(0.004, 0.8140),
            fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E53935"),
            color="#E53935")
plt.tight_layout()
plt.savefig(FIG_DIR / "post07_04_power_norm_comparison.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'post07_04_power_norm_comparison.png'}")

# ============================================================
# Figure 5: Failed experiments — what went wrong
# ============================================================
experiments = [
    "Flip only\n(baseline)", "2× crop\n(replace orig)", "3× crop+flip\n(stack)",
    "TTA10\n(5crop×2flip)", "Two-layer\nK1=1600", "Multi-crop\nfeature avg"
]
val_results = [0.8122, 0.7912, None, None, 0.7836, 0.8076]  # None = OOM/no val
public_results = [0.81550, None, None, 0.80000, None, None]
status = ["success", "val regression", "OOM", "public regression", "no improvement", "val regression"]
colors_status = {"success": "#4CAF50", "val regression": "#FF9800",
                 "OOM": "#F44336", "public regression": "#F44336",
                 "no improvement": "#9E9E9E"}

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(experiments))
bar_vals = [v if v else 0.78 for v in val_results]
bar_colors = [colors_status[s] for s in status]
bars = ax.bar(x, bar_vals, color=bar_colors, width=0.6, edgecolor="white", linewidth=1.5)
for i, (v, p, s) in enumerate(zip(val_results, public_results, status)):
    label = f"val={v:.4f}" if v else "OOM"
    if p:
        label += f"\npublic={p:.5f}"
    label += f"\n({s})"
    y = bar_vals[i] + 0.001
    ax.text(i, y, label, ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=10)
ax.set_ylabel("Validation accuracy", fontsize=12)
ax.set_title("Augmentation & architecture experiments: most failed\nOnly flip aug succeeded; crops and multi-layer did not help", fontsize=13)
ax.axhline(0.8122, color="#4CAF50", linestyle="--", linewidth=1.5, label="Flip-only baseline (0.8122)")
ax.set_ylim(0.77, 0.82)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "post07_05_failed_experiments.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'post07_05_failed_experiments.png'}")

# ============================================================
# Figure 6: Ensemble — individual vs ensemble val
# ============================================================
models = ["P=6 K=8000\n(single)", "P=7 K=6000\n(single)", "P=8 K=3200\n(single)",
          "2-model\n(P=6+P=7)", "3-model\n(P=6+P=7+P=8)"]
vals = [0.8134, 0.8100, 0.7998, 0.8234, 0.8236]
colors_ens = ["#2196F3", "#4CAF50", "#FB8C00", "#9C27B0", "#E91E63"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(models)), vals, color=colors_ens, width=0.6,
              edgecolor="white", linewidth=1.5)
for i, v in enumerate(vals):
    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Validation accuracy (pnorm + flip aug)", fontsize=12)
ax.set_title("Ensemble gives +0.01 over best single model\n2-model and 3-model ensembles are nearly identical", fontsize=13)
ax.axhline(0.8134, color="#2196F3", linestyle=":", linewidth=1.2, alpha=0.7)
ax.set_ylim(0.79, 0.83)
ax.grid(axis="y", alpha=0.3)
# Add annotation for ensemble boost
ax.annotate("+0.0100", xy=(3, 0.8234), xytext=(3.5, 0.8200),
            fontsize=12, fontweight="bold", color="#9C27B0",
            arrowprops=dict(arrowstyle="->", color="#9C27B0"))
plt.tight_layout()
plt.savefig(FIG_DIR / "post07_06_ensemble_comparison.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'post07_06_ensemble_comparison.png'}")

# ============================================================
# Figure 7: Val vs Public score scatter (all submitted runs)
# ============================================================
submitted = [
    ("run_07 P6K8000\nC=0.003", 0.7858, 0.78400),
    ("run_08 P7K6000\nC=0.002", 0.7876, 0.78600),
    ("run_08 P6K8000\nC=0.002", 0.7836, 0.78750),
    ("run_10 P6K10000\nC=0.002", 0.7860, 0.78600),
    ("run_12 flip\n+TTA2", 0.8122, 0.81550),
    ("run_14\nTTA10", 0.8104, 0.80000),
    ("run_17 pnorm\n+TTA2", 0.8136, 0.82700),
]
fig, ax = plt.subplots(figsize=(9, 7))
for label, val, pub in submitted:
    color = "#F44336" if pub < 0.81 else "#4CAF50" if pub > 0.82 else "#2196F3"
    ax.scatter(val, pub, s=120, color=color, edgecolor="black", linewidth=1, zorder=5)
    ax.annotate(label, (val, pub), textcoords="offset points", xytext=(8, 5), fontsize=8)
# Diagonal reference
lims = [0.78, 0.83]
ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="val = public")
ax.set_xlabel("Validation accuracy", fontsize=12)
ax.set_ylabel("Kaggle public score", fontsize=12)
ax.set_title("Val vs Public: val is a noisy predictor of public\nPower norm had +0.0014 val but +0.0115 public", fontsize=13)
ax.set_xlim(0.78, 0.82)
ax.set_ylim(0.785, 0.83)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "post07_07_val_vs_public.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'post07_07_val_vs_public.png'}")

print("\n=== All 7 post-run_07 figures generated ===")
