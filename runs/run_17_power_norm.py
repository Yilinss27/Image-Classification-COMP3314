"""Run 17: Power normalization + L2 norm + multi-C ensemble on flip-aug features.

Step 1: Encode P=6 K=8000 flip-aug features (same as run_12)
Step 2: Apply power norm (signed sqrt) + L2 norm before StandardScaler
Step 3: Fit C in {0.001, 0.002, 0.003, 0.005}, report val for each
Step 4: Also test without power norm as control (should match run_12 ~0.8122)
Step 5: Multi-C ensemble: average decision_functions of all C values
Step 6: Full refit best config + TTA2
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LOG_DIR, SUB_DIR, Timer, get_logger, load_test_cached, load_train_cached, save_submission
from run_07_cuml_sweep import encode_images_gpu, get_or_fit_dict

from cuml.svm import LinearSVC
import cupy as cp

RUN_NAME = "run_17_power_norm"
N_PATCHES = 1_000_000
P, K = 6, 8000
POOL = 2
STRIDE = 1
BATCH_SIZE = 128
C_LIST = [0.001, 0.002, 0.003, 0.005]


def flip_horizontal(X):
    return np.ascontiguousarray(X[:, :, ::-1, :])


def power_norm(X):
    """Signed square-root: x -> sign(x) * sqrt(|x|)."""
    return np.sign(X) * np.sqrt(np.abs(X))


def l2_norm(X):
    """Per-sample L2 normalization."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return X / norms


def fit_and_eval(F_tr, y_tr, F_val, y_val, C, log, label=""):
    """Fit LinearSVC on F_tr, eval on F_val, return (acc, clf, dt)."""
    cp.get_default_memory_pool().free_all_blocks()
    t0 = time.time()
    clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                    max_iter=5000, tol=1e-4)
    clf.fit(F_tr, y_tr)
    preds = clf.predict(F_val)
    if hasattr(preds, "get"):
        preds = preds.get()
    acc = float(accuracy_score(y_val, preds))
    dt = time.time() - t0
    log.info(f"  {label} C={C:.4f}  val={acc:.4f}  ({dt:.1f}s)")
    return acc, clf, dt


def main():
    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")

    with Timer(log, "load data"):
        X, y, _ = load_train_cached(n_jobs=8)
        Xte, test_names = load_test_cached(n_jobs=8)

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0)
    y_tr, y_val = y[idx_tr], y[idx_val]
    rng = np.random.RandomState(0)

    centroids, zca_mean, zca_W = get_or_fit_dict(
        X[idx_tr], K, P, N_PATCHES, log, rng)

    # Encode flip-aug train + val
    X_tr = X[idx_tr]
    X_val = X[idx_val]
    X_tr_aug = np.concatenate([X_tr, flip_horizontal(X_tr)], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_tr])

    with Timer(log, "encode aug-train (90k)"):
        F_tr_raw = encode_images_gpu(X_tr_aug, centroids, zca_mean, zca_W,
                                     patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
    with Timer(log, "encode val (5k)"):
        F_val_raw = encode_images_gpu(X_val, centroids, zca_mean, zca_W,
                                      patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
    log.info(f"  F_tr {F_tr_raw.shape}  F_val {F_val_raw.shape}")

    # ---- Experiment A: baseline (no power norm, just StandardScaler) ----
    log.info(f"\n=== A: baseline (StandardScaler only) ===")
    sc_a = StandardScaler().fit(F_tr_raw)
    F_tr_a = sc_a.transform(F_tr_raw).astype(np.float32)
    F_val_a = sc_a.transform(F_val_raw).astype(np.float32)
    for C in C_LIST:
        fit_and_eval(F_tr_a, y_tr_aug, F_val_a, y_val, C, log, "baseline")
    del F_tr_a, F_val_a, sc_a
    gc.collect()

    # ---- Experiment B: power norm + StandardScaler ----
    log.info(f"\n=== B: power_norm + StandardScaler ===")
    F_tr_pn = power_norm(F_tr_raw)
    F_val_pn = power_norm(F_val_raw)
    sc_b = StandardScaler().fit(F_tr_pn)
    F_tr_b = sc_b.transform(F_tr_pn).astype(np.float32)
    F_val_b = sc_b.transform(F_val_pn).astype(np.float32)
    best_C_b, best_val_b = None, -1
    for C in C_LIST:
        acc, _, _ = fit_and_eval(F_tr_b, y_tr_aug, F_val_b, y_val, C, log, "pnorm")
        if acc > best_val_b:
            best_C_b, best_val_b = C, acc
    del F_tr_b, F_val_b
    gc.collect()

    # ---- Experiment C: power norm + L2 norm + StandardScaler ----
    log.info(f"\n=== C: power_norm + L2_norm + StandardScaler ===")
    F_tr_pl = l2_norm(F_tr_pn)
    F_val_pl = l2_norm(F_val_pn)
    sc_c = StandardScaler().fit(F_tr_pl)
    F_tr_c = sc_c.transform(F_tr_pl).astype(np.float32)
    F_val_c = sc_c.transform(F_val_pl).astype(np.float32)
    best_C_c, best_val_c = None, -1
    for C in C_LIST:
        acc, _, _ = fit_and_eval(F_tr_c, y_tr_aug, F_val_c, y_val, C, log, "pnorm+L2")
        if acc > best_val_c:
            best_C_c, best_val_c = C, acc
    del F_tr_c, F_val_c, sc_c
    gc.collect()

    # ---- Pick best experiment and generate submissions ----
    results = [("baseline", None, None),
               ("pnorm", best_C_b, best_val_b),
               ("pnorm+L2", best_C_c, best_val_c)]
    log.info(f"\n=== Summary ===")
    log.info(f"  pnorm best: C={best_C_b} val={best_val_b:.4f}")
    log.info(f"  pnorm+L2 best: C={best_C_c} val={best_val_c:.4f}")

    # Use the best normalization for full refit + TTA
    if best_val_b >= best_val_c:
        best_label, best_C, best_val = "pnorm", best_C_b, best_val_b
        norm_fn = power_norm
    else:
        best_label, best_C, best_val = "pnorm+L2", best_C_c, best_val_c
        norm_fn = lambda x: l2_norm(power_norm(x))

    log.info(f"\n--- Full refit: {best_label} C={best_C} ---")

    X_full_aug = np.concatenate([X, flip_horizontal(X)], axis=0)
    y_full_aug = np.concatenate([y, y])
    Xte_flip = flip_horizontal(Xte)

    with Timer(log, "encode full-aug (100k)"):
        F_full_raw = encode_images_gpu(X_full_aug, centroids, zca_mean, zca_W,
                                       patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
    del X_full_aug
    gc.collect()

    F_full_n = norm_fn(F_full_raw)
    del F_full_raw
    sc_full = StandardScaler().fit(F_full_n)
    F_full_s = sc_full.transform(F_full_n).astype(np.float32)
    del F_full_n
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    t0 = time.time()
    final_clf = LinearSVC(C=best_C, loss="squared_hinge", penalty="l2",
                          max_iter=5000, tol=1e-5)
    final_clf.fit(F_full_s, y_full_aug)
    log.info(f"  full-refit fit {time.time()-t0:.1f}s")
    del F_full_s
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # TTA2: test orig + test flip
    with Timer(log, "encode test + test-flip"):
        F_te_raw = encode_images_gpu(Xte, centroids, zca_mean, zca_W,
                                     patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
        F_te_flip_raw = encode_images_gpu(Xte_flip, centroids, zca_mean, zca_W,
                                          patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)

    F_te_s = sc_full.transform(norm_fn(F_te_raw)).astype(np.float32)
    F_te_flip_s = sc_full.transform(norm_fn(F_te_flip_raw)).astype(np.float32)
    del F_te_raw, F_te_flip_raw
    gc.collect()

    df_orig = final_clf.decision_function(F_te_s)
    df_flip = final_clf.decision_function(F_te_flip_s)
    if hasattr(df_orig, "get"): df_orig = df_orig.get()
    if hasattr(df_flip, "get"): df_flip = df_flip.get()

    preds_no = df_orig.argmax(axis=1)
    preds_tta = (df_orig + df_flip).argmax(axis=1)

    tag = best_label.replace("+", "_")
    sub1 = SUB_DIR / f"sub_run17_{tag}_C{best_C}.csv"
    sub2 = SUB_DIR / f"sub_run17_{tag}_tta_C{best_C}.csv"
    save_submission(test_names, preds_no, sub1)
    save_submission(test_names, preds_tta, sub2)
    log.info(f"  wrote {sub1.name}")
    log.info(f"  wrote {sub2.name}")

    diffs = int((preds_no != preds_tta).sum())
    log.info(f"  TTA changed {diffs}/{len(preds_no)} preds ({100*diffs/len(preds_no):.2f}%)")
    log.info(f"\n=== {RUN_NAME} done ===")
    log.info(f"best: {best_label} C={best_C} val={best_val:.4f}")


if __name__ == "__main__":
    main()
