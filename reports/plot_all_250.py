"""Parse run_07 stdout log + results JSON to get all 250 (P,K,C) configs, then plot."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "run_07_cuml.stdout.log"
JSON_FILE = ROOT / "logs" / "run_07_cuml_sweep_results.json"
FIG_DIR = Path(__file__).resolve().parent / "figures"
SOTA_VAL = 0.7678

# Parse val= lines from the full stdout log. Take the LAST occurrence of each
# (P, K, C) so we get the final successful run for P=4 (which had to be re-done
# after the first sweep's disk-full crash) and also P=5–8 from the restart.
pat = re.compile(r"P=(\d+) K=(\d+) C=([\d.]+) val=([\d.]+)")
latest: dict[tuple[int, int, float], float] = {}
for line in LOG.read_text().splitlines():
    m = pat.search(line)
    if m:
        P, K, C, v = int(m[1]), int(m[2]), float(m[3]), float(m[4])
        latest[(P, K, C)] = v

# Also ingest the JSON (P=5-8 only) to confirm / augment
for r in json.load(open(JSON_FILE)):
    latest[(r["P"], r["K"], r["C"])] = r["val_acc"]

rows = [
    {"P": P, "K": K, "C": C, "val_acc": v} for (P, K, C), v in latest.items()
]
df = pd.DataFrame(rows)
print(f"total configs parsed: {len(df)}")
print("P values:", sorted(df.P.unique()))
print("K values:", sorted(df.K.unique()))
print("C values:", sorted(df.C.unique()))
print(f"(P,K) combos: {len(df.groupby(['P','K']))}")
assert len(df) == 250, f"expected 250 configs, got {len(df)}"
df.to_csv(Path(__file__).resolve().parent / "run07_all_250.csv", index=False)

# ---- One big figure: 5 subplots (one per P), each a K×C heatmap ----
P_list = sorted(df.P.unique())
K_list = sorted(df.K.unique())
C_list = sorted(df.C.unique())

fig, axes = plt.subplots(
    1, len(P_list), figsize=(5.2 * len(P_list), 9.0),
    sharey=True, constrained_layout=True,
)

vmin, vmax = 0.68, 0.79
for ax, P in zip(axes, P_list):
    sub = df[df.P == P].pivot(index="K", columns="C", values="val_acc")
    sub = sub.reindex(index=K_list, columns=C_list)
    im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(C_list)))
    ax.set_xticklabels([f"{c:.3f}" for c in C_list], rotation=45, fontsize=11)
    ax.set_yticks(range(len(K_list)))
    ax.set_yticklabels([str(k) for k in K_list], fontsize=11)
    ax.set_xlabel("C (LinearSVC reg)", fontsize=13)
    if ax is axes[0]:
        ax.set_ylabel("K (dictionary size)", fontsize=13)

    # Annotate each cell with the val number, highlight SOTA beaters
    best_val_P = sub.values.max()
    best_idx = np.unravel_index(sub.values.argmax(), sub.values.shape)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="black", fontsize=8)
                continue
            color = "black" if 0.74 <= v <= 0.78 else "white"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
            if v >= SOTA_VAL:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="navy", linewidth=1.5))
    # Star on the per-P best
    bi, bj = best_idx
    ax.plot(bj, bi, marker="*", markersize=18,
            markerfacecolor="gold", markeredgecolor="black",
            markeredgewidth=1.2, zorder=5)

    ax.set_title(f"P = {P}×{P}   (best val = {best_val_P:.4f})", fontsize=14)

# Global colorbar on the right
cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.02)
cbar.set_label("Validation accuracy", fontsize=11)

fig.suptitle(
    "run_07 cuML sweep — all 250 configurations (5 P × 10 K × 5 C)\n"
    "★ = best per-P   ·   navy outline = beats prior SOTA val = 0.7678   ·   59 of 250 configs beat SOTA",
    fontsize=15,
)
plt.savefig(FIG_DIR / "00_all_250_configs.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR / '00_all_250_configs.png'}")

# Print quick stats for sanity
print()
print("Top 15 across all 250 configs:")
print(df.sort_values("val_acc", ascending=False).head(15).to_string(index=False))
print()
n_sota = (df.val_acc >= SOTA_VAL).sum()
print(f"Configs beating SOTA 0.7678: {n_sota} / 250")
