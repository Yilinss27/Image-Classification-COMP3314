"""Run 14: flip aug C sweep + 10-view TTA.

Step 1: encode P=6 K=8000 flip-aug features (90k), fit C in {0.001, 0.002, 0.003, 0.005}
Step 2: refit best C on full 100k (50k + 50k flipped)
Step 3: 10-view TTA on test: 5 spatial crops x 2 (orig + flip), avg decision_function

Crop positions (from pad-4 reflected 40x40 image):
  center (4,4), TL (0,0), TR (0,8), BL (8,0), BR (8,8)
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

RUN_NAME = "run_14_flip_csweep_tta"
N_PATCHES = 1_000_000
POOL = 2
STRIDE = 1
BATCH_SIZE = 128
PAD = 4

P, K = 6, 8000
C_LIST = [0.001, 0.002, 0.003, 0.005]

CROP_OFFSETS = [
    (4, 4),   # center
    (0, 0),   # top-left
    (0, 8),   # top-right
    (8, 0),   # bottom-left
    (8, 8),   # bottom-right
]


def flip_horizontal(X):
    return np.ascontiguousarray(X[:, :, ::-1, :])


def pad_reflect(X, pad=PAD):
    return np.pad(X, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode="reflect")


def fixed_crop(X_padded, h_off, w_off, size=32):
    return np.ascontiguousarray(X_padded[:, h_off:h_off+size, w_off:w_off+size, :])


def main():
    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")

    with Timer(log, "load data"):
        X, y, _ = load_train_cached(n_jobs=8)
        Xte, test_names = load_test_cached(n_jobs=8)

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0)
    y_tr, y_val = y[idx_tr], y[idx_val]

    X_tr = X[idx_tr]
    X_val = X[idx_val]
    X_tr_aug = np.concatenate([X_tr, flip_horizontal(X_tr)], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_tr])
    log.info(f"flip-aug train: {X_tr_aug.shape}")

    rng = np.random.RandomState(0)
    centroids, zca_mean, zca_W = get_or_fit_dict(
        X[idx_tr], K, P, N_PATCHES, log, rng)

    # ---- Step 1: encode + C sweep ----
    with Timer(log, "encode aug-train (90k)"):
        F_tr_aug = encode_images_gpu(
            X_tr_aug, centroids, zca_mean, zca_W,
            patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
    with Timer(log, "encode val (5k)"):
        F_val = encode_images_gpu(
            X_val, centroids, zca_mean, zca_W,
            patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)

    scaler = StandardScaler().fit(F_tr_aug)
    F_tr_s = scaler.transform(F_tr_aug).astype(np.float32)
    F_val_s = scaler.transform(F_val).astype(np.float32)
    del F_tr_aug, F_val
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    best_C, best_val = None, -1
    for C in C_LIST:
        t0 = time.time()
        clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                        max_iter=5000, tol=1e-4)
        clf.fit(F_tr_s, y_tr_aug)
        preds = clf.predict(F_val_s)
        if hasattr(preds, "get"):
            preds = preds.get()
        acc = float(accuracy_score(y_val, preds))
        dt = time.time() - t0
        mark = " ⭐" if acc > best_val else ""
        log.info(f"  C={C:.4f}  val={acc:.4f}  ({dt:.1f}s){mark}")
        if acc > best_val:
            best_C, best_val = C, acc
    log.info(f"  -> best C={best_C} val={best_val:.4f}")

    del F_tr_s, F_val_s, scaler
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # ---- Step 2: full refit with best C ----
    log.info(f"\n--- Full refit C={best_C} ---")
    X_full_aug = np.concatenate([X, flip_horizontal(X)], axis=0)
    y_full_aug = np.concatenate([y, y])

    with Timer(log, "encode full-aug (100k)"):
        F_full = encode_images_gpu(
            X_full_aug, centroids, zca_mean, zca_W,
            patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
    del X_full_aug
    gc.collect()

    scaler_full = StandardScaler().fit(F_full)
    F_full_s = scaler_full.transform(F_full).astype(np.float32)
    del F_full
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

    # ---- Step 3: 10-view TTA ----
    log.info(f"\n--- 10-view TTA (5 crops x 2 flips) ---")
    Xte_pad = pad_reflect(Xte)
    all_df = []

    for i, (h, w) in enumerate(CROP_OFFSETS):
        crop_name = ["center", "TL", "TR", "BL", "BR"][i]
        Xte_crop = fixed_crop(Xte_pad, h, w)
        Xte_crop_flip = flip_horizontal(Xte_crop)

        for label, X_view in [("orig", Xte_crop), ("flip", Xte_crop_flip)]:
            with Timer(log, f"encode test {crop_name}_{label}"):
                F_te = encode_images_gpu(
                    X_view, centroids, zca_mean, zca_W,
                    patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
            F_te_s = scaler_full.transform(F_te).astype(np.float32)
            df = final_clf.decision_function(F_te_s)
            if hasattr(df, "get"):
                df = df.get()
            all_df.append(df)
            del F_te, F_te_s
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

    log.info(f"  collected {len(all_df)} views, each {all_df[0].shape}")

    # Simple TTA: average all 10 views
    avg_df = np.mean(all_df, axis=0)
    preds_tta10 = avg_df.argmax(axis=1)

    # Also: 2-view TTA (center orig + center flip) for comparison
    avg_df_2 = (all_df[0] + all_df[1]) / 2
    preds_tta2 = avg_df_2.argmax(axis=1)

    # No-TTA baseline (center crop only = original test images)
    preds_no_tta = all_df[0].argmax(axis=1)

    # Write all three subs
    sub_no = SUB_DIR / f"sub_run14_P{P}_K{K}_C{best_C}.csv"
    sub_2 = SUB_DIR / f"sub_run14_tta2_P{P}_K{K}_C{best_C}.csv"
    sub_10 = SUB_DIR / f"sub_run14_tta10_P{P}_K{K}_C{best_C}.csv"
    save_submission(test_names, preds_no_tta, sub_no)
    save_submission(test_names, preds_tta2, sub_2)
    save_submission(test_names, preds_tta10, sub_10)
    log.info(f"  wrote {sub_no.name}")
    log.info(f"  wrote {sub_2.name}")
    log.info(f"  wrote {sub_10.name}")

    d2 = int((preds_no_tta != preds_tta2).sum())
    d10 = int((preds_no_tta != preds_tta10).sum())
    d2v10 = int((preds_tta2 != preds_tta10).sum())
    log.info(f"  TTA2 vs no-TTA: {d2} diffs ({100*d2/len(preds_no_tta):.2f}%)")
    log.info(f"  TTA10 vs no-TTA: {d10} diffs ({100*d10/len(preds_no_tta):.2f}%)")
    log.info(f"  TTA10 vs TTA2: {d2v10} diffs ({100*d2v10/len(preds_no_tta):.2f}%)")

    log.info(f"\n=== {RUN_NAME} done ===")
    log.info(f"best C={best_C} val={best_val:.4f}")
    log.info(f"submissions: no-TTA, TTA2, TTA10")


if __name__ == "__main__":
    main()
