"""Analyse run_08 Phase B (lower-C extension) and combine with run_07 baselines
to produce plots + a short report.

Phase B tested C in {0.0005, 0.001, 0.002} for 16 (P, K) cells where run_07's
winner was C=0.003. This script:

  1. Merges Phase B results with run_07's C=0.003 values for the same cells
     so each cell has 4 C points.
  2. Plots val vs C for every cell; highlights where C<0.003 wins.
  3. Produces a Δ heatmap (Phase B best - run_07 baseline) over (P, K).
  4. Reports global curve (val vs C averaged over cells, stratified by patch
     size) to read off the regularization trend.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PHASE_B_JSON = ROOT / "logs" / "run_08_phase_b_results.json"
RUN07_CSV = ROOT / "reports" / "run07_all_results.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(exist_ok=True)

PREV_SOTA = 0.7858  # run_07 overall (P=6 K=8000 C=0.003)

# 16 Phase B cells (same order as run_08_phase_b.py)
TARGETS = [
    (4, 8000),
    (5, 4000), (5, 8000),
    (6, 2400), (6, 4000), (6, 6000), (6, 8000),
    (7, 1600), (7, 4000), (7, 6000), (7, 8000),
    (8, 2400), (8, 3200), (8, 4000), (8, 6000), (8, 8000),
]

# ---- Load ----
phb = json.load(open(PHASE_B_JSON))
phb_df = pd.DataFrame(phb["results"])  # P, K, C, val_acc, delta
phb_df["source"] = "phase_b"
print(f"Phase B: {len(phb_df)} configs")

run07 = pd.read_csv(RUN07_CSV)
# Pick C=0.003 rows for the target cells only (these are the baselines)
mask = run07.C == 0.003
r07_base = run07[mask][["P", "K", "C", "val_acc"]].copy()
r07_base["delta"] = 0.0
r07_base["source"] = "run07_C0.003"
# Filter to target cells
tset = set(TARGETS)
r07_base = r07_base[r07_base.apply(lambda r: (int(r.P), int(r.K)) in tset, axis=1)]
print(f"run_07 C=0.003 baselines: {len(r07_base)}")

df = pd.concat([phb_df, r07_base], ignore_index=True)
df = df.sort_values(["P", "K", "C"]).reset_index(drop=True)
print(f"Combined: {len(df)} rows")

# Precompute per-cell best C from Phase B and from the 4-C curve
bestC = df.loc[df.groupby(["P", "K"])["val_acc"].idxmax()].reset_index(drop=True)
print("\n=== Per-cell winner across 4 Cs ===")
print(bestC[["P", "K", "C", "val_acc", "source"]].to_string(index=False))

# ---- Plot 1: val vs C for each (P,K), 4x4 grid of 16 cells ----
fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharey=False)
axes = axes.flatten()
P_colors = {4: "#2E86C1", 5: "#4CAF50", 6: "#E53935", 7: "#FB8C00", 8: "#8E44AD"}

for ax, (P, K) in zip(axes, TARGETS):
    sub = df[(df.P == P) & (df.K == K)].sort_values("C")
    color = P_colors[P]
    ax.plot(sub.C, sub.val_acc, marker="o", linewidth=2.2, markersize=8, color=color)
    ax.set_xscale("log")
    ax.set_xticks([0.0005, 0.001, 0.002, 0.003])
    ax.set_xticklabels(["5e-4", "1e-3", "2e-3", "3e-3"], fontsize=8)
    ax.grid(True, alpha=0.3)
    # Mark the winner
    wrow = sub.loc[sub.val_acc.idxmax()]
    ax.scatter([wrow.C], [wrow.val_acc], s=200, marker="*",
               color="gold", edgecolor="black", zorder=5)
    # Highlight if C<0.003 won
    base_val = float(sub[sub.C == 0.003].val_acc.iloc[0])
    c003_idx = sub.index[sub.C == 0.003][0]
    ax.axhline(base_val, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    delta = wrow.val_acc - base_val
    win = delta > 0
    title = f"P={P} K={K}"
    if win:
        title += f"  Δ={delta:+.4f} ⬆"
        ax.set_facecolor("#EEF9EE")
    else:
        title += f"  Δ={delta:+.4f}"
    ax.set_title(title, fontsize=11,
                 color="darkgreen" if win else "black",
                 fontweight="bold" if win else "normal")
    ax.set_xlabel("C", fontsize=9)
    ax.set_ylabel("val", fontsize=9)

fig.suptitle(
    "run_08 Phase B — val vs C (0.0005, 0.001, 0.002, 0.003) per (P, K) cell\n"
    "gold ★ = cell winner · green = new best with C<0.003 · gray dotted = run_07 baseline",
    fontsize=13, y=1.00,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "phase_b_01_val_vs_C_per_cell.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"\nSaved {FIG_DIR / 'phase_b_01_val_vs_C_per_cell.png'}")

# ---- Plot 2: Δ heatmap (best Phase B val - C=0.003 baseline) ----
delta_rows = []
for P, K in TARGETS:
    sub = df[(df.P == P) & (df.K == K)]
    base = float(sub[sub.C == 0.003].val_acc.iloc[0])
    pb = sub[sub.C < 0.003]
    best_pb = pb.val_acc.max()
    best_c = float(pb.loc[pb.val_acc.idxmax()].C)
    delta_rows.append({
        "P": P, "K": K,
        "baseline": base,
        "phase_b_best": best_pb,
        "phase_b_best_C": best_c,
        "delta": best_pb - base,
    })
dh = pd.DataFrame(delta_rows)
print("\n=== Δ (best Phase B - baseline C=0.003) ===")
print(dh.to_string(index=False))

# Build P×K delta matrix (only cells we tested; others = NaN)
Ps = sorted({P for P, K in TARGETS})
Ks = sorted({K for P, K in TARGETS})
mat = np.full((len(Ps), len(Ks)), np.nan)
for _, r in dh.iterrows():
    i = Ps.index(int(r.P))
    j = Ks.index(int(r.K))
    mat[i, j] = r.delta

fig, ax = plt.subplots(figsize=(10, 4.2))
# Use RdBu with 0 centered
vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(Ks)))
ax.set_xticklabels(Ks)
ax.set_yticks(range(len(Ps)))
ax.set_yticklabels([f"P={p}" for p in Ps])
ax.set_xlabel("K", fontsize=12)
ax.set_title("Phase B — best (C<0.003) vs run_07 C=0.003 baseline (Δ val)", fontsize=13)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if np.isnan(v):
            ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=10)
        else:
            color = "white" if abs(v) > 0.7 * vmax else "black"
            ax.text(j, i, f"{v:+.4f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")
plt.colorbar(im, ax=ax, label="Δ val vs baseline")
plt.tight_layout()
plt.savefig(FIG_DIR / "phase_b_02_delta_heatmap.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'phase_b_02_delta_heatmap.png'}")

# ---- Plot 3: summary curves — average val across K, stratified by P ----
fig, ax = plt.subplots(figsize=(9, 5.5))
Cs = sorted(df.C.unique())
for P in sorted(df.P.unique()):
    sub_p = df[df.P == P]
    # Average over K within this P
    agg = sub_p.groupby("C").val_acc.mean().reset_index()
    ax.plot(agg.C, agg.val_acc, marker="o", linewidth=2.2, markersize=8,
            label=f"P={P} (avg over K)", color=P_colors[P])
ax.axvline(0.003, color="gray", linestyle="--", linewidth=1,
           label="run_07 grid floor (C=0.003)")
ax.set_xscale("log")
ax.set_xticks(Cs)
ax.set_xticklabels([f"{c:g}" for c in Cs])
ax.set_xlabel("C (regularization inverse)", fontsize=12)
ax.set_ylabel("mean val acc (over tested K per P)", fontsize=12)
ax.set_title(
    "Phase B — mean val vs C by patch size\n"
    "P=7/8 curves rise as C drops (→ C=0.002–0.003 is NOT yet optimal here)\n"
    "P=4/5/6 curves rise as C grows (→ C=0.003 is optimal or C* > 0.003)",
    fontsize=12,
)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "phase_b_03_mean_val_vs_C_per_P.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'phase_b_03_mean_val_vs_C_per_P.png'}")

# ---- Plot 4: biggest win — P=8 K=8000 recovery ----
# This cell collapsed in run_07 (0.7502 at C=0.003). Phase B recovered +0.0206.
# Show this alongside P=6 K=8000 (no C<0.003 helped) to contrast the regimes.
fig, ax = plt.subplots(figsize=(9, 5.5))
for (P, K, color, lab) in [(8, 8000, "#8E44AD", "P=8 K=8000 (run_07 collapse)"),
                            (7, 6000, "#FB8C00", "P=7 K=6000 (new SOTA cell)"),
                            (6, 8000, "#E53935", "P=6 K=8000 (old SOTA cell)"),
                            (4, 8000, "#2E86C1", "P=4 K=8000 (small-patch reference)")]:
    sub = df[(df.P == P) & (df.K == K)].sort_values("C")
    ax.plot(sub.C, sub.val_acc, marker="o", linewidth=2.4, markersize=9,
            color=color, label=lab)
ax.set_xscale("log")
ax.set_xticks(Cs)
ax.set_xticklabels([f"{c:g}" for c in Cs])
ax.set_xlabel("C", fontsize=12)
ax.set_ylabel("val acc", fontsize=12)
ax.set_title(
    "Selected cells — C response at the grid edge\n"
    "P=8 K=8000 recovers +0.0206 as C drops (was not a real collapse, just C too high)",
    fontsize=12,
)
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "phase_b_04_featured_cells.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / 'phase_b_04_featured_cells.png'}")

# ---- Save combined CSV and summary stats ----
df.to_csv(FIG_DIR.parent / "phase_b_all_results.csv", index=False)
dh.to_csv(FIG_DIR.parent / "phase_b_delta_summary.csv", index=False)

print("\n=== Summary ===")
print(f"Total Phase B configs: {len(phb_df)}")
print(f"Cells improved (Δ>0): {(dh.delta > 0).sum()} / {len(dh)}")
print(f"Cells with Δ > +0.003 (meaningful): {(dh.delta > 0.003).sum()}")
print(f"Mean Δ: {dh.delta.mean():+.4f}")
print(f"Max Δ: {dh.delta.max():+.4f} at P={int(dh.loc[dh.delta.idxmax(), 'P'])} K={int(dh.loc[dh.delta.idxmax(), 'K'])}")
print(f"Min Δ: {dh.delta.min():+.4f} at P={int(dh.loc[dh.delta.idxmin(), 'P'])} K={int(dh.loc[dh.delta.idxmin(), 'K'])}")

# Which P benefits most?
by_P = dh.groupby("P").agg(
    mean_delta=("delta", "mean"),
    n_wins=("delta", lambda x: (x > 0).sum()),
    n=("delta", "size"),
).reset_index()
print("\n=== By patch size ===")
print(by_P.to_string(index=False))
