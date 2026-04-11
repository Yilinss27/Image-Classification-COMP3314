"""Run 06 — LinearSVC on cached K=3200 P=6 Coates features.

Mirrors refit_sota.py: no val split, single full-50k fit with C=0.01
(the sweet spot from the K=1600 sweep — peak val 0.7678 at C=0.01,
adjacent C=0.003 and C=0.03 both ~0.015 lower).

Memory-careful: drop raw feature arrays immediately after scaling.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
SUB = ROOT / "submissions"

ap = argparse.ArgumentParser()
ap.add_argument("--C", type=float, default=0.01)
ap.add_argument("--K", type=int, default=3200)
ap.add_argument("--P", type=int, default=6)
args = ap.parse_args()

K = args.K
P = args.P
C = args.C
OUT = SUB / f"sub_sota_K{K}_P{P}_C{C}.csv"

t0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log(f"loading K={K} P={P} features")
X_tr = np.load(CACHE / f"gpu_feat_train_K{K}_P{P}.npy")
log(f"X_tr {X_tr.shape} {X_tr.dtype} ({X_tr.nbytes / 2**30:.2f} GB)")

log("StandardScaler fit_transform on train (drop raw immediately)")
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
del X_tr
gc.collect()
log(f"X_tr_s {X_tr_s.shape} {X_tr_s.dtype}")

log("loading test features")
X_te = np.load(CACHE / f"gpu_feat_test_K{K}_P{P}.npy")
X_te_s = sc.transform(X_te).astype(np.float32)
del X_te
gc.collect()
log(f"X_te_s {X_te_s.shape}")

log("loading y and names")
y = np.load(CACHE / "train_y.npy")
names = np.load(CACHE / "test_names.npy", allow_pickle=True).tolist()

log(f"LinearSVC fit C={C} (full 50k)")
clf = LinearSVC(C=C, dual=False, max_iter=3000, tol=1e-5)
clf.fit(X_tr_s, y)
log("fit done")

log("predict test")
preds = clf.predict(X_te_s)
log(f"preds {preds.shape}  dist {np.bincount(preds)}")

df = pd.DataFrame({"im_name": names, "label": preds.astype(int)})
df.to_csv(OUT, index=False)
log(f"wrote {OUT}")
