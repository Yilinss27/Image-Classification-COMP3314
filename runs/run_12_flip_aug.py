"""Run 12: horizontal flip data augmentation on the current SOTA config.

The Coates-Ng pipeline has never seen any data augmentation in this repo,
and horizontal flip is the cheapest, most-reliable classical augmentation
for natural images. Paper results suggest ~+1 pp on CIFAR-style data.

Pipeline per (P, K, C) target:
  1. Reuse the existing run_07 / run_10 dict (1M or 2M patches, unchanged)
  2. Split 50k train into 45k-sub + 5k-val (same random_state=0 as before)
  3. Flip the 45k sub horizontally -> 90k augmented train set
     (labels duplicated)
  4. GPU-encode the 90k train + 5k val + 10k test
  5. Fit cuML LinearSVC on 90k, eval on 5k val
  6. TTA: encode flipped test set too, average decision_function with
     original test, argmax
  7. Refit on full 100k (50k original + 50k flipped) and write submission

Baseline comparison: sub_run07_P6_K8000_C0.002 (val=0.7836, public=0.78750)

Val split is on the ORIGINAL 50k only — the flip is applied only to the
training partition so the val/test distributions remain unchanged.
"""
from __future__ import annotations

import argparse
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

RUN_NAME = "run_12_flip_aug"
N_PATCHES = 1_000_000  # reuse existing run_07 dicts
POOL = 2
STRIDE = 1
BATCH_SIZE = 128

# (P, K, C) targets. Start with the current public SOTA and add a second
# top candidate for comparison.
TARGETS = [
    (6, 8000, 0.002),   # public SOTA so far: 0.78750
    (7, 6000, 0.002),   # Phase B SOTA: public 0.78600
]


def flip_horizontal(X: np.ndarray) -> np.ndarray:
    """Horizontally flip a batch of (N, H, W, C) uint8 images."""
    return np.ascontiguousarray(X[:, :, ::-1, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Run just one target, e.g. P6K8000C0.002")
    ap.add_argument("--skip-refit", action="store_true",
                    help="Only run val-split fit, skip full-100k refit+submission")
    args = ap.parse_args()

    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")

    with Timer(log, "load train/test"):
        X, y, _ = load_train_cached(n_jobs=8)
        Xte, test_names = load_test_cached(n_jobs=8)
    log.info(f"train {X.shape}  test {Xte.shape}")

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0)
    y_tr = y[idx_tr]
    y_val = y[idx_val]
    log.info(f"train sub {len(idx_tr)}  val {len(idx_val)}")

    X_tr = X[idx_tr]   # (45000, 32, 32, 3)
    X_val = X[idx_val]  # (5000,  32, 32, 3)
    X_tr_aug = np.concatenate([X_tr, flip_horizontal(X_tr)], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_tr], axis=0)
    log.info(f"flip-aug train: {X_tr_aug.shape}  labels {y_tr_aug.shape}")

    X_full_aug = np.concatenate([X, flip_horizontal(X)], axis=0)
    y_full_aug = np.concatenate([y, y], axis=0)
    log.info(f"full aug (for refit): {X_full_aug.shape}")

    Xte_flip = flip_horizontal(Xte)

    rng = np.random.RandomState(0)
    results = []

    targets = TARGETS
    if args.only:
        targets = [t for t in TARGETS if f"P{t[0]}K{t[1]}C{t[2]}" == args.only]

    for patch, K, C in targets:
        log.info(f"\n### P={patch} K={K} C={C} ###")

        centroids, zca_mean, zca_W = get_or_fit_dict(
            X[idx_tr], K, patch, N_PATCHES, log, rng)

        with Timer(log, f"GPU encode aug-train (90k) K={K} P={patch}"):
            F_tr_aug = encode_images_gpu(
                X_tr_aug, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
        with Timer(log, f"GPU encode val (5k) K={K} P={patch}"):
            F_val = encode_images_gpu(
                X_val, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
        with Timer(log, f"GPU encode test (10k) K={K} P={patch}"):
            F_test = encode_images_gpu(
                Xte, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
        with Timer(log, f"GPU encode test-flipped (10k) K={K} P={patch}"):
            F_test_flip = encode_images_gpu(
                Xte_flip, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
        log.info(f"  F_tr_aug {F_tr_aug.shape}  F_val {F_val.shape}  "
                 f"F_test {F_test.shape}  F_test_flip {F_test_flip.shape}")

        scaler = StandardScaler().fit(F_tr_aug)
        F_tr_s = scaler.transform(F_tr_aug).astype(np.float32)
        F_val_s = scaler.transform(F_val).astype(np.float32)
        F_test_s = scaler.transform(F_test).astype(np.float32)
        F_test_flip_s = scaler.transform(F_test_flip).astype(np.float32)
        del F_tr_aug, F_val, F_test, F_test_flip
        gc.collect()

        t0 = time.time()
        clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                        max_iter=5000, tol=1e-4)
        clf.fit(F_tr_s, y_tr_aug)
        fit_dt = time.time() - t0
        val_preds = clf.predict(F_val_s)
        if hasattr(val_preds, "get"):
            val_preds = val_preds.get()
        val_acc = float(accuracy_score(y_val, val_preds))
        log.info(f"  aug-train val={val_acc:.4f}  (fit {fit_dt:.1f}s)")
        results.append(dict(P=patch, K=K, C=C, val_aug=val_acc))

        if not args.skip_refit:
            # Full 100k refit + test prediction with TTA
            with Timer(log, f"encode full aug (100k) K={K} P={patch}"):
                F_full_aug = encode_images_gpu(
                    X_full_aug, centroids, zca_mean, zca_W,
                    patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
            scaler_full = StandardScaler().fit(F_full_aug)
            F_full_s = scaler_full.transform(F_full_aug).astype(np.float32)
            del F_full_aug
            gc.collect()

            # Re-encode test + test-flip with full-scaler (same encoder, fresh scaler)
            with Timer(log, f"re-encode test (for full refit) K={K} P={patch}"):
                F_te = encode_images_gpu(
                    Xte, centroids, zca_mean, zca_W,
                    patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
                F_te_flip = encode_images_gpu(
                    Xte_flip, centroids, zca_mean, zca_W,
                    patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
            F_te_s = scaler_full.transform(F_te).astype(np.float32)
            F_te_flip_s = scaler_full.transform(F_te_flip).astype(np.float32)
            del F_te, F_te_flip
            gc.collect()

            t0 = time.time()
            final_clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                                  max_iter=5000, tol=1e-5)
            final_clf.fit(F_full_s, y_full_aug)
            log.info(f"  full-refit fit {time.time()-t0:.1f}s")

            # TTA: average decision functions of original and flipped test.
            # cuML LinearSVC returns a decision_function array (N, n_classes)
            # for multi-class OvR.
            df_orig = final_clf.decision_function(F_te_s)
            df_flip = final_clf.decision_function(F_te_flip_s)
            if hasattr(df_orig, "get"):
                df_orig = df_orig.get()
            if hasattr(df_flip, "get"):
                df_flip = df_flip.get()
            log.info(f"  df shapes: orig {df_orig.shape}  flip {df_flip.shape}")

            # Hard preds from just the original (no TTA) — write separate sub
            preds_orig = df_orig.argmax(axis=1)
            sub_no_tta = SUB_DIR / f"sub_run12_flip_P{patch}_K{K}_C{C}.csv"
            save_submission(test_names, preds_orig, sub_no_tta)
            log.info(f"  wrote {sub_no_tta.name}")

            # TTA preds
            preds_tta = (df_orig + df_flip).argmax(axis=1)
            sub_tta = SUB_DIR / f"sub_run12_flip_tta_P{patch}_K{K}_C{C}.csv"
            save_submission(test_names, preds_tta, sub_tta)
            log.info(f"  wrote {sub_tta.name}")

            # How often TTA changes prediction vs no-TTA
            diffs = int((preds_orig != preds_tta).sum())
            log.info(f"  TTA changed preds on {diffs}/{len(preds_tta)} rows "
                     f"({100.0*diffs/len(preds_tta):.2f}%)")

            del F_full_s, F_te_s, F_te_flip_s, final_clf, scaler_full
            gc.collect()

        del F_tr_s, F_val_s, F_test_s, F_test_flip_s, scaler
        gc.collect()

    log.info(f"\n=== {RUN_NAME} summary ===")
    for r in results:
        log.info(f"  P={r['P']} K={r['K']} C={r['C']}  aug-val={r['val_aug']:.4f}")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
