"""Run 19: Multi-P ensemble with power norm + flip aug + TTA2.

The final submission. Combines 3 diverse models:
  P=6 K=8000 C=0.002 (current public SOTA 0.82700)
  P=7 K=6000 C=0.002 (Phase B SOTA)
  P=8 K=3200 C=0.003 (run_07 top P=8)

Each model: 100k flip-aug train → pnorm → StandardScaler → cuML LinearSVC
Ensemble: average decision_functions across 3 models → TTA2 → argmax
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LOG_DIR, SUB_DIR, Timer, get_logger, load_test_cached, load_train_cached, save_submission
from run_07_cuml_sweep import encode_images_gpu, get_or_fit_dict

from cuml.svm import LinearSVC
import cupy as cp

RUN_NAME = "run_19_pnorm_ensemble"
N_PATCHES = 1_000_000
POOL = 2
STRIDE = 1
BATCH_SIZE = 128

TARGETS = [
    (6, 8000, 0.002),
    (7, 6000, 0.002),
    (8, 3200, 0.003),
]


def flip_horizontal(X):
    return np.ascontiguousarray(X[:, :, ::-1, :])


def power_norm(X):
    return np.sign(X) * np.sqrt(np.abs(X))


def main():
    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"targets: {TARGETS}")

    with Timer(log, "load data"):
        X, y, _ = load_train_cached(n_jobs=8)
        Xte, test_names = load_test_cached(n_jobs=8)

    X_full_aug = np.concatenate([X, flip_horizontal(X)], axis=0)
    y_full_aug = np.concatenate([y, y])
    Xte_flip = flip_horizontal(Xte)
    log.info(f"full aug: {X_full_aug.shape}, test: {Xte.shape}")

    rng = np.random.RandomState(0)
    all_df_orig = []
    all_df_flip = []

    for patch, K, C in TARGETS:
        log.info(f"\n### P={patch} K={K} C={C} ###")

        centroids, zca_mean, zca_W = get_or_fit_dict(
            X, K, patch, N_PATCHES, log, rng)

        with Timer(log, f"encode full-aug (100k) P={patch} K={K}"):
            F_full = encode_images_gpu(
                X_full_aug, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)

        F_full = power_norm(F_full)
        scaler = StandardScaler().fit(F_full)
        F_full_s = scaler.transform(F_full).astype(np.float32)
        del F_full
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        t0 = time.time()
        clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                        max_iter=5000, tol=1e-5)
        clf.fit(F_full_s, y_full_aug)
        log.info(f"  fit {time.time()-t0:.1f}s")
        del F_full_s
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        with Timer(log, f"encode test + test-flip P={patch} K={K}"):
            F_te = encode_images_gpu(
                Xte, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
            F_te_flip = encode_images_gpu(
                Xte_flip, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)

        F_te_s = scaler.transform(power_norm(F_te)).astype(np.float32)
        F_te_flip_s = scaler.transform(power_norm(F_te_flip)).astype(np.float32)
        del F_te, F_te_flip
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        df_orig = clf.decision_function(F_te_s)
        df_flip = clf.decision_function(F_te_flip_s)
        if hasattr(df_orig, "get"): df_orig = df_orig.get()
        if hasattr(df_flip, "get"): df_flip = df_flip.get()
        log.info(f"  df shapes: {df_orig.shape}, {df_flip.shape}")

        all_df_orig.append(df_orig)
        all_df_flip.append(df_flip)

        del F_te_s, F_te_flip_s, clf, scaler
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    # ---- Ensemble ----
    log.info(f"\n--- Ensemble ({len(TARGETS)} models + pnorm) ---")

    # TTA2 per model, then average across models
    avg_tta2 = np.mean(
        [(o + f) / 2 for o, f in zip(all_df_orig, all_df_flip)],
        axis=0)
    preds_tta2 = avg_tta2.argmax(axis=1)

    # Also: no-TTA ensemble for comparison
    avg_no_tta = np.mean(all_df_orig, axis=0)
    preds_no_tta = avg_no_tta.argmax(axis=1)

    sub1 = SUB_DIR / "sub_run19_pnorm_ensemble_noTTA.csv"
    sub2 = SUB_DIR / "sub_run19_pnorm_ensemble_tta2.csv"
    save_submission(test_names, preds_no_tta, sub1)
    save_submission(test_names, preds_tta2, sub2)
    log.info(f"wrote {sub1.name}")
    log.info(f"wrote {sub2.name}")

    diffs = int((preds_no_tta != preds_tta2).sum())
    log.info(f"noTTA vs TTA2: {diffs} diffs ({100*diffs/len(preds_no_tta):.2f}%)")

    log.info(f"\n=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
