"""Refit best config on full 50k and predict test.

Loads cached Coates features, standardizes, fits LinearSVC, writes submission.
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
SUB = ROOT / "submissions"

K = 1600
P = 6
C = 0.01
VAL = 0.7678
OUT = SUB / f"sub_sota_K{K}_P{P}_C{C}.csv"

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

log(f"loading features K={K} P={P}")
X_tr = np.load(CACHE / f"gpu_feat_train_K{K}_P{P}.npy")
X_te = np.load(CACHE / f"gpu_feat_test_K{K}_P{P}.npy")
y    = np.load(CACHE / "train_y.npy")
names = np.load(CACHE / "test_names.npy", allow_pickle=True).tolist()
log(f"X_tr {X_tr.shape} X_te {X_te.shape} y {y.shape} names {len(names)}")

log("StandardScaler fit_transform train")
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
del X_tr
gc.collect()

log("StandardScaler transform test")
X_te_s = sc.transform(X_te).astype(np.float32)
del X_te
gc.collect()

log(f"LinearSVC fit C={C}")
clf = LinearSVC(C=C, dual="auto", max_iter=3000, tol=1e-5)
clf.fit(X_tr_s, y)
log("fit done")

log("predict test")
preds = clf.predict(X_te_s)
log(f"preds {preds.shape} dist {np.bincount(preds)}")

df = pd.DataFrame({"im_name": names, "label": preds.astype(int)})
df.to_csv(OUT, index=False)
log(f"wrote {OUT}")
