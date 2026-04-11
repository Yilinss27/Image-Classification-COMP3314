"""Analyse run_07 cuML sweep results and produce plots + stats for the report.

Reads BOTH the final results JSON (P=5-8) AND the full stdout log (which
contains P=4 from the earlier sweep attempt). Takes the LAST occurrence of
each (P,K,C) so we get the canonical final val for each config.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "logs" / "run_07_cuml_sweep_results.json"
LOG = ROOT / "logs" / "run_07_cuml.stdout.log"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SOTA_VAL = 0.7678  # previous sklearn LinearSVC K=1600 P=6 C=0.01 public=0.77400

# Parse val= lines from the full stdout log (covers all 5 P values including P=4)
pat = re.compile(r"P=(\d+) K=(\d+) C=([\d.]+) val=([\d.]+)")
latest: dict[tuple[int, int, float], float] = {}
for line in LOG.read_text().splitlines():
    m = pat.search(line)
    if m:
        latest[(int(m[1]), int(m[2]), float(m[3]))] = float(m[4])
# Augment with final JSON (P=5-8) — same numbers, just a sanity check
for r in json.load(open(RESULTS)):
    latest[(r["P"], r["K"], r["C"])] = r["val_acc"]

df = pd.DataFrame(
    [{"P": P, "K": K, "C": C, "val_acc": v} for (P, K, C), v in latest.items()]
)
df["delta_sota"] = df["val_acc"] - SOTA_VAL
assert len(df) == 250, f"expected 250 configs, got {len(df)}"
print("n configs:", len(df))
print("P values:", sorted(df.P.unique()))
print("K values:", sorted(df.K.unique()))
print("C values:", sorted(df.C.unique()))
print()

# ---- Best per (P, K) ----
best_pk = df.loc[df.groupby(["P", "K"])["val_acc"].idxmax()].reset_index(drop=True)
best_pk = best_pk.sort_values(["P", "K"]).reset_index(drop=True)
print("=== Best per (P, K) ===")
print(best_pk[["P", "K", "C", "val_acc"]].to_string(index=False))
print()

# ---- Top 10 overall ----
top10 = df.sort_values("val_acc", ascending=False).head(10).reset_index(drop=True)
print("=== Top 10 configs ===")
print(top10[["P", "K", "C", "val_acc", "delta_sota"]].to_string(index=False))
print()

# ---- Plot 1: val acc vs K for each P (best C per (P,K)) ----
fig, ax = plt.subplots(figsize=(9.5, 6))
colors = {4: "#2E86C1", 5: "#4CAF50", 6: "#E53935", 7: "#FB8C00", 8: "#8E44AD"}
for P in sorted(df.P.unique()):
    sub = best_pk[best_pk.P == P].sort_values("K")
    ax.plot(sub.K, sub.val_acc, marker="o", linewidth=2, markersize=7,
            label=f"P={P}", color=colors[P])
ax.axhline(SOTA_VAL, color="gray", linestyle="--", linewidth=1.2,
           label=f"prior SOTA val={SOTA_VAL}")
ax.set_xlabel("Dictionary size K", fontsize=12)
ax.set_ylabel("Validation accuracy (best C)", fontsize=12)
ax.set_title("run_07 Coates-Ng sweep: val acc vs K, per patch size", fontsize=13)
ax.set_xscale("log")
ax.set_xticks([400, 800, 1600, 3200, 6000, 8000])
ax.set_xticklabels(["400", "800", "1600", "3200", "6000", "8000"])
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_val_vs_K_per_P.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / '01_val_vs_K_per_P.png'}")

# ---- Plot 2: P × K heatmap of best val_acc ----
pivot_best = best_pk.pivot(index="P", columns="K", values="val_acc")
fig, ax = plt.subplots(figsize=(10, 4.8))
im = ax.imshow(pivot_best.values, aspect="auto", cmap="RdYlGn",
               vmin=0.70, vmax=0.79)
ax.set_xticks(range(len(pivot_best.columns)))
ax.set_xticklabels(pivot_best.columns)
ax.set_yticks(range(len(pivot_best.index)))
ax.set_yticklabels([f"P={p}" for p in pivot_best.index])
ax.set_xlabel("K", fontsize=12)
ax.set_title("Best val acc per (P, K) — SOTA beaters (≥0.7678) outlined", fontsize=13)
for i in range(pivot_best.shape[0]):
    for j in range(pivot_best.shape[1]):
        v = pivot_best.values[i, j]
        color = "black" if 0.74 < v < 0.78 else "white"
        ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                color=color, fontsize=9)
        if v >= SOTA_VAL:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="navy", linewidth=2.5))
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Validation accuracy", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_heatmap_best_per_PK.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / '02_heatmap_best_per_PK.png'}")

# ---- Plot 3: val acc vs C for each P at fixed K (K=4000) — regularization study ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, K_fix in zip(axes, [4000, 8000]):
    for P in sorted(df.P.unique()):
        sub = df[(df.P == P) & (df.K == K_fix)].sort_values("C")
        ax.plot(sub.C, sub.val_acc, marker="o", linewidth=2, markersize=7,
                label=f"P={P}", color=colors[P])
    ax.axhline(SOTA_VAL, color="gray", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("C", fontsize=12)
    ax.set_ylabel("Validation accuracy", fontsize=12)
    ax.set_title(f"K={K_fix}", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle("C-regularization response by patch size", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_C_sweep_K4k_K8k.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / '03_C_sweep_K4k_K8k.png'}")

# ---- Plot 4: distribution of optimal K per patch size ----
K_opt_per_P = best_pk.loc[best_pk.groupby("P")["val_acc"].idxmax()].sort_values("P")
print("\n=== Optimal K per P ===")
print(K_opt_per_P[["P", "K", "C", "val_acc"]].to_string(index=False))
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.bar(K_opt_per_P.P.astype(str), K_opt_per_P.K,
       color=[colors[p] for p in K_opt_per_P.P])
for _, row in K_opt_per_P.iterrows():
    ax.text(str(row.P), row.K + 150, f"{int(row.K)}\nval={row.val_acc:.4f}",
            ha="center", fontsize=9)
ax.set_xlabel("Patch size P", fontsize=12)
ax.set_ylabel("Optimal K", fontsize=12)
ax.set_title("Optimal dictionary size K*(P) — shrinks as patch grows", fontsize=13)
ax.set_ylim(0, max(K_opt_per_P.K) * 1.25)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_optimal_K_per_P.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / '04_optimal_K_per_P.png'}")

# ---- Plot 5: heatmap of all (P, K, C) for P=6 (best patch) ----
P_fix = 6
sub = df[df.P == P_fix].pivot(index="K", columns="C", values="val_acc")
fig, ax = plt.subplots(figsize=(7.5, 5))
im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn", vmin=0.70, vmax=0.79)
ax.set_xticks(range(len(sub.columns)))
ax.set_xticklabels([f"{c:.3f}" for c in sub.columns])
ax.set_yticks(range(len(sub.index)))
ax.set_yticklabels(sub.index)
ax.set_xlabel("C", fontsize=12)
ax.set_ylabel("K", fontsize=12)
ax.set_title(f"P={P_fix}: full (K, C) val acc matrix", fontsize=13)
for i in range(sub.shape[0]):
    for j in range(sub.shape[1]):
        v = sub.values[i, j]
        color = "black" if 0.74 < v < 0.78 else "white"
        ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                color=color, fontsize=9)
plt.colorbar(im, ax=ax, label="Val acc")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_P6_full_KC_matrix.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / '05_P6_full_KC_matrix.png'}")

# ---- Plot 6: best-C distribution ----
best_c_counts = best_pk.groupby(["P", "C"]).size().unstack(fill_value=0)
print("\n=== Best-C distribution per P ===")
print(best_c_counts)
fig, ax = plt.subplots(figsize=(8, 4.5))
bottoms = np.zeros(len(best_c_counts))
C_list = sorted(best_c_counts.columns)
c_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(C_list)))
for C, col in zip(C_list, c_colors):
    if C in best_c_counts.columns:
        vals = best_c_counts[C].values
        ax.bar(best_c_counts.index.astype(str), vals, bottom=bottoms,
               label=f"C={C}", color=col)
        bottoms += vals
ax.set_xlabel("Patch size P", fontsize=12)
ax.set_ylabel("Count (K values where this C won)", fontsize=12)
ax.set_title("Best C per (P, K) — optimal C shifts lower as (P, K) grow", fontsize=13)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_bestC_distribution.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / '06_bestC_distribution.png'}")

# ---- Summary stats for report ----
n_sota_beaters = (best_pk.val_acc >= SOTA_VAL).sum()
print(f"\n=== Summary ===")
print(f"Total configs: {len(df)}")
print(f"Unique (P,K) combos: {len(best_pk)}")
print(f"(P,K) combos with best val ≥ {SOTA_VAL}: {n_sota_beaters}")
print(f"Mean val: {df.val_acc.mean():.4f}")
print(f"Max val: {df.val_acc.max():.4f}")
print(f"Min val: {df.val_acc.min():.4f}")
print(f"Top 10 mean val: {top10.val_acc.mean():.4f}")

# Save top10 and all-results as a CSV
df.to_csv(Path(__file__).resolve().parent / "run07_all_results.csv", index=False)
top10.to_csv(Path(__file__).resolve().parent / "run07_top10.csv", index=False)
print("Saved CSVs")
