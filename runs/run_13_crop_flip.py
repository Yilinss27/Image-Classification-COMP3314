"""Run 13: random crop + horizontal flip augmentation.

Standard CIFAR-10 augmentation recipe:
  - pad 4 pixels (reflect)
  - random 32x32 crop
  - random horizontal flip

We pre-generate 2 augmented views per training image (each is a random
crop + 50% flip with an independent seed). This keeps the feature matrix
at 90k x K (same size as run_12 flip) which fits in 5090 VRAM; 4x was
attempted first but cuML s 23 GB feature copy + solver workspace OOM d
on 32 GB VRAM.

So 45k -> 90k augmented training samples. Val partition stays at 5k
original images. For full-50k refit we use the same 2x recipe to get 100k.

Target: P=6 K=8000 C=0.002 (current flip-only public SOTA 0.81550 on val 0.8122).
Baseline to beat: flip-only aug-val 0.8122.
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

RUN_NAME = "run_13_crop_flip"
N_PATCHES = 1_000_000
POOL = 2
STRIDE = 1
BATCH_SIZE = 128
PAD = 4

TARGETS = [
    (6, 8000, 0.002),
    (7, 6000, 0.002),
]


def pad_reflect(X: np.ndarray, pad: int = PAD) -> np.ndarray:
    return np.pad(X, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode="reflect")


def random_crop(X_pad: np.ndarray, seed: int, flip_prob: float = 0.5) -> np.ndarray:
    """Random 32x32 crop from a (N, 40, 40, 3) padded batch.

    With probability `flip_prob` each sample is also horizontally flipped.
    """
    rng = np.random.RandomState(seed)
    N, H, W, C = X_pad.shape
    out_h = H - 2 * PAD
    out_w = W - 2 * PAD
    assert out_h == 32 and out_w == 32
    off_h = rng.randint(0, 2 * PAD + 1, size=N)
    off_w = rng.randint(0, 2 * PAD + 1, size=N)
    flip = rng.random(N) < flip_prob
    out = np.empty((N, out_h, out_w, C), dtype=X_pad.dtype)
    for i in range(N):
        h = off_h[i]; w = off_w[i]
        crop = X_pad[i, h:h+out_h, w:w+out_w, :]
        if flip[i]:
            crop = crop[:, ::-1, :]
        out[i] = crop
    return out


def flip_horizontal(X: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(X[:, :, ::-1, :])


def build_aug_views(X: np.ndarray, seed_base: int) -> np.ndarray:
    """3x augmentation: [original, flipped, random_crop + 50% flip].

    Preserves the run_12 flip-only baseline (views 0+1) and adds one
    random-crop view on top. 45k -> 135k fits in 5090 VRAM (~17.3 GB
    at K=8000 P=6) with proper cupy mempool cleanup before fit.
    """
    v0 = X
    v1 = flip_horizontal(X)
    X_pad = pad_reflect(X)
    v2 = random_crop(X_pad, seed=seed_base, flip_prob=0.5)
    return np.concatenate([v0, v1, v2], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Run a single target, e.g. P6K8000C0.002")
    ap.add_argument("--skip-refit", action="store_true")
    args = ap.parse_args()

    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"augmentation: 3x [orig, flip, random_crop_maybe_flip]")

    with Timer(log, "load train/test"):
        X, y, _ = load_train_cached(n_jobs=8)
        Xte, test_names = load_test_cached(n_jobs=8)
    log.info(f"train {X.shape}  test {Xte.shape}")

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0)
    y_tr = y[idx_tr]
    y_val = y[idx_val]
    log.info(f"train sub {len(idx_tr)}  val {len(idx_val)}")

    X_tr = X[idx_tr]
    X_val = X[idx_val]

    with Timer(log, "build aug-views (3x) for 45k train"):
        X_tr_aug = build_aug_views(X_tr, seed_base=100)
        y_tr_aug = np.tile(y_tr, 3)
    log.info(f"aug train: {X_tr_aug.shape}  labels {y_tr_aug.shape}")

    with Timer(log, "build aug-views (3x) for full 50k (for refit)"):
        X_full_aug = build_aug_views(X, seed_base=200)
        y_full_aug = np.tile(y, 3)
    log.info(f"full aug: {X_full_aug.shape}")

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

        with Timer(log, f"GPU encode aug-train (135k) K={K} P={patch}"):
            F_tr_aug = encode_images_gpu(
                X_tr_aug, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
        with Timer(log, f"GPU encode val (5k) K={K} P={patch}"):
            F_val = encode_images_gpu(
                X_val, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
        log.info(f"  F_tr_aug {F_tr_aug.shape}  F_val {F_val.shape}")

        scaler = StandardScaler().fit(F_tr_aug)
        F_tr_s = scaler.transform(F_tr_aug).astype(np.float32)
        F_val_s = scaler.transform(F_val).astype(np.float32)
        del F_tr_aug, F_val
        gc.collect()
        # Drop lingering cupy encode buffers before the big fit.
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()

        t0 = time.time()
        clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                        max_iter=5000, tol=1e-4)
        clf.fit(F_tr_s, y_tr_aug)
        fit_dt = time.time() - t0
        val_preds = clf.predict(F_val_s)
        if hasattr(val_preds, "get"):
            val_preds = val_preds.get()
        val_acc = float(accuracy_score(y_val, val_preds))
        log.info(f"  3x-cropflip val={val_acc:.4f}  (fit {fit_dt:.1f}s)")
        results.append(dict(P=patch, K=K, C=C, val_aug=val_acc))

        if not args.skip_refit:
            with Timer(log, f"GPU encode full-aug (150k) K={K} P={patch}"):
                F_full_aug = encode_images_gpu(
                    X_full_aug, centroids, zca_mean, zca_W,
                    patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
            scaler_full = StandardScaler().fit(F_full_aug)
            F_full_s = scaler_full.transform(F_full_aug).astype(np.float32)
            del F_full_aug
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()

            with Timer(log, f"re-encode test + test-flip K={K} P={patch}"):
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

            df_orig = final_clf.decision_function(F_te_s)
            df_flip = final_clf.decision_function(F_te_flip_s)
            if hasattr(df_orig, "get"): df_orig = df_orig.get()
            if hasattr(df_flip, "get"): df_flip = df_flip.get()

            preds_orig = df_orig.argmax(axis=1)
            sub_no_tta = SUB_DIR / f"sub_run13_cropflip_P{patch}_K{K}_C{C}.csv"
            save_submission(test_names, preds_orig, sub_no_tta)
            log.info(f"  wrote {sub_no_tta.name}")

            preds_tta = (df_orig + df_flip).argmax(axis=1)
            sub_tta = SUB_DIR / f"sub_run13_cropflip_tta_P{patch}_K{K}_C{C}.csv"
            save_submission(test_names, preds_tta, sub_tta)
            log.info(f"  wrote {sub_tta.name}")

            diffs = int((preds_orig != preds_tta).sum())
            log.info(f"  TTA changed preds on {diffs}/{len(preds_tta)} rows "
                     f"({100.0*diffs/len(preds_tta):.2f}%)")

            del F_full_s, F_te_s, F_te_flip_s, final_clf, scaler_full
            gc.collect()

        del F_tr_s, F_val_s, scaler
        gc.collect()

    log.info(f"\n=== {RUN_NAME} summary ===")
    for r in results:
        log.info(f"  P={r['P']} K={r['K']} C={r['C']}  3x-cropflip val={r['val_aug']:.4f}")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
