"""Run 11: hard-vote ensemble of the top 3 current submissions.

Each sub CSV stores only hard labels (no probabilities), so this is a
majority vote per test sample. For the rare case of a 3-way tie, we fall
back to the label from the model with the highest public score, which
is the current SOTA run_08 P=6 K=8000 C=0.002 -> 0.78750.

Inputs (all already in submissions/):
  1. sub_run07_P6_K8000_C0.003.csv  val=0.7858  public=0.78400
  2. sub_run08_P7_K6000_C0.002.csv  val=0.7876  public=0.78600
  3. sub_run07_P6_K8000_C0.002.csv  val=0.7836  public=0.78750  (tie-break)

Output: submissions/sub_run11_ensemble_top3.csv
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = ROOT / "submissions"

MODELS = [
    ("run07_P6_K8000_C0.003", 0.78400),
    ("run08_P7_K6000_C0.002", 0.78600),
    ("run07_P6_K8000_C0.002", 0.78750),  # current SOTA, also tie-breaker
]


def load(name: str) -> pd.DataFrame:
    p = SUB_DIR / f"sub_{name}.csv"
    df = pd.read_csv(p)
    return df


def main():
    dfs = [load(m[0]) for m in MODELS]
    # Sanity-check: same order, same ids, same length
    n = len(dfs[0])
    assert all(len(d) == n for d in dfs), f"row count mismatch: {[len(d) for d in dfs]}"
    id_col = dfs[0].columns[0]
    lab_col = dfs[0].columns[1]
    for d in dfs[1:]:
        assert (d[id_col].values == dfs[0][id_col].values).all(), "id order mismatch"
    print(f"loaded {len(MODELS)} subs, {n} rows each")

    # Tie-breaker index: highest public, i.e. MODELS[-1]
    tb_idx = max(range(len(MODELS)), key=lambda i: MODELS[i][1])
    print(f"tie-breaker: {MODELS[tb_idx][0]} (public {MODELS[tb_idx][1]})")

    preds = [d[lab_col].values for d in dfs]  # list of 3 arrays length n

    # Agreement stats
    agree_all = sum(1 for i in range(n)
                    if preds[0][i] == preds[1][i] == preds[2][i])
    agree_any_pair = sum(1 for i in range(n)
                         if (preds[0][i] == preds[1][i]) or
                            (preds[0][i] == preds[2][i]) or
                            (preds[1][i] == preds[2][i]))
    print(f"all 3 agree:  {agree_all}/{n}  ({agree_all/n*100:.1f}%)")
    print(f"majority exists (any pair agree): {agree_any_pair}/{n}  ({agree_any_pair/n*100:.1f}%)")
    ties = n - agree_any_pair
    print(f"3-way ties (all different):       {ties}")

    # Majority vote
    ensemble = []
    per_model_flips = [0, 0, 0]
    for i in range(n):
        labs = [preds[j][i] for j in range(3)]
        cnt = Counter(labs)
        top = cnt.most_common(1)[0]
        if top[1] >= 2:
            winner = top[0]
        else:
            # 3-way tie: fall back to highest-public model
            winner = preds[tb_idx][i]
        ensemble.append(winner)
        for j in range(3):
            if preds[j][i] != winner:
                per_model_flips[j] += 1

    out = pd.DataFrame({id_col: dfs[0][id_col].values, lab_col: ensemble})
    out_path = SUB_DIR / "sub_run11_ensemble_top3.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"rows where ensemble differs from each model:")
    for (name, pub), flips in zip(MODELS, per_model_flips):
        print(f"  {name}: {flips}  ({flips/n*100:.2f}%)")


if __name__ == "__main__":
    main()
