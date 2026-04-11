"""Run 06 — LinearSVC sweep on cached K=3200 P=6 Coates features.

GPU sweep (run_05) was OOM-killed after encoding K=3200 P=6 features but before
fitting any classifier. This script reuses those cached features, runs a
val-split sweep across C values, refits the best on full 50k, writes submission.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
SUB = ROOT / "submissions"
LOG = ROOT / "logs"

K = 3200
P = 6
C_GRID = [0.001, 0.003, 0.01, 0.03]

t0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log(f"loading K={K} P={P} features")
X_tr_raw = np.load(CACHE / f"gpu_feat_train_K{K}_P{P}.npy")
X_te_raw = np.load(CACHE / f"gpu_feat_test_K{K}_P{P}.npy")
y_all    = np.load(CACHE / "train_y.npy")
names    = np.load(CACHE / "test_names.npy", allow_pickle=True).tolist()
log(f"X_tr {X_tr_raw.shape}  X_te {X_te_raw.shape}  y {y_all.shape}")

idx = np.arange(len(y_all))
tr_idx, val_idx = train_test_split(idx, test_size=0.1, stratify=y_all, random_state=0)
log(f"val split: train {len(tr_idx)}  val {len(val_idx)}")

log("StandardScaler fit on split train")
sc = StandardScaler()
X_tr_split = sc.fit_transform(X_tr_raw[tr_idx]).astype(np.float32)
X_val      = sc.transform(X_tr_raw[val_idx]).astype(np.float32)
y_tr       = y_all[tr_idx]
y_val      = y_all[val_idx]
gc.collect()
log(f"scaled: split train {X_tr_split.shape}  val {X_val.shape}")

results: dict[float, float] = {}
for C in C_GRID:
    log(f"[START] LinearSVC C={C}")
    clf = LinearSVC(C=C, dual="auto", max_iter=3000, tol=1e-5)
    clf.fit(X_tr_split, y_tr)
    val_acc = float(accuracy_score(y_val, clf.predict(X_val)))
    log(f"[END]   K={K} P={P} C={C}  val={val_acc:.4f}")
    results[C] = val_acc
    del clf
    gc.collect()

best_C = max(results, key=results.get)
best_val = results[best_C]
log(f">>> BEST K={K} P={P} C={best_C}  val={best_val:.4f}")
log(f"    all results: {results}")

del X_tr_split, X_val
gc.collect()

log("StandardScaler refit on full 50k")
sc_full = StandardScaler()
X_tr_full = sc_full.fit_transform(X_tr_raw).astype(np.float32)
del X_tr_raw
gc.collect()
X_te = sc_full.transform(X_te_raw).astype(np.float32)
del X_te_raw
gc.collect()
log(f"scaled full: train {X_tr_full.shape}  test {X_te.shape}")

log(f"refit LinearSVC on full 50k with C={best_C}")
clf_full = LinearSVC(C=best_C, dual="auto", max_iter=3000, tol=1e-5)
clf_full.fit(X_tr_full, y_all)
preds = clf_full.predict(X_te)
log(f"preds shape {preds.shape}  dist {np.bincount(preds)}")

out = SUB / f"sub_sota_K{K}_P{P}_C{best_C}.csv"
df = pd.DataFrame({"im_name": names, "label": preds.astype(int)})
df.to_csv(out, index=False)
log(f"wrote {out}")

with open(LOG / "run_06_k3200.log", "a") as f:
    f.write(f"\n=== run_06 K={K} P={P} sweep ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    for c, v in sorted(results.items()):
        f.write(f"  K={K} P={P} C={c}  val={v:.4f}\n")
    f.write(f"  BEST C={best_C} val={best_val:.4f}\n")
    f.write(f"  submission: {out.name}\n")
log("done")
