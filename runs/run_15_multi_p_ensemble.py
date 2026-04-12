"""Run 15: Multi-P soft-vote ensemble with flip augmentation.

For each (P, K, C) target:
  1. Flip-aug full 50k -> 100k train
  2. Encode + StandardScaler + cuML LinearSVC fit
  3. Encode test + test-flip
  4. Save decision_function as .npy for ensemble

Then average decision_functions across all P values, argmax -> submission.
Also generate TTA2 (center orig + center flip avg per model, then ensemble).
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

RUN_NAME = "run_15_multi_p_ensemble"
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

        F_te_s = scaler.transform(F_te).astype(np.float32)
        F_te_flip_s = scaler.transform(F_te_flip).astype(np.float32)
        del F_te, F_te_flip
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        df_orig = clf.decision_function(F_te_s)
        df_flip = clf.decision_function(F_te_flip_s)
        if hasattr(df_orig, "get"):
            df_orig = df_orig.get()
        if hasattr(df_flip, "get"):
            df_flip = df_flip.get()
        log.info(f"  df shapes: {df_orig.shape}, {df_flip.shape}")

        all_df_orig.append(df_orig)
        all_df_flip.append(df_flip)

        del F_te_s, F_te_flip_s, clf, scaler
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    # ---- Ensemble strategies ----
    log.info(f"\n--- Ensemble ({len(TARGETS)} models) ---")

    # Strategy 1: no-TTA ensemble (avg decision_function of orig test only)
    avg_orig = np.mean(all_df_orig, axis=0)
    preds_no_tta = avg_orig.argmax(axis=1)
    sub1 = SUB_DIR / "sub_run15_ensemble_noTTA.csv"
    save_submission(test_names, preds_no_tta, sub1)
    log.info(f"  wrote {sub1.name}")

    # Strategy 2: per-model TTA2 then ensemble (each model avg orig+flip, then avg across models)
    avg_tta2 = np.mean(
        [(o + f) / 2 for o, f in zip(all_df_orig, all_df_flip)],
        axis=0)
    preds_tta2 = avg_tta2.argmax(axis=1)
    sub2 = SUB_DIR / "sub_run15_ensemble_tta2.csv"
    save_submission(test_names, preds_tta2, sub2)
    log.info(f"  wrote {sub2.name}")

    # Strategy 3: pool all decision_functions (orig + flip across all models)
    all_dfs = all_df_orig + all_df_flip  # 6 total
    avg_all = np.mean(all_dfs, axis=0)
    preds_all = avg_all.argmax(axis=1)
    sub3 = SUB_DIR / "sub_run15_ensemble_all6.csv"
    save_submission(test_names, preds_all, sub3)
    log.info(f"  wrote {sub3.name}")

    # Diff stats
    d12 = int((preds_no_tta != preds_tta2).sum())
    d13 = int((preds_no_tta != preds_all).sum())
    d23 = int((preds_tta2 != preds_all).sum())
    log.info(f"  noTTA vs TTA2: {d12} diffs ({100*d12/len(preds_no_tta):.2f}%)")
    log.info(f"  noTTA vs all6: {d13} diffs ({100*d13/len(preds_no_tta):.2f}%)")
    log.info(f"  TTA2 vs all6: {d23} diffs ({100*d23/len(preds_no_tta):.2f}%)")

    log.info(f"\n=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
