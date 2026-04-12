"""Run 18: Multi-crop feature averaging + power norm + flip aug.

For each image, generate 5 views (original + 4 random crops with 50% flip),
encode each view separately, AVERAGE the 5 feature vectors into 1 stable
representation. Training set size stays the same (no VRAM increase).

Pipeline:
  1. For each image: 5 views → encode each → avg → 1 feature vector
  2. Flip aug: do the above for original 50k AND flipped 50k → 100k total
  3. Power norm + StandardScaler + cuML LinearSVC
  4. Test TTA2: avg decision_function of orig and flip test
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

RUN_NAME = "run_18_multicrop_avg"
P, K, C = 6, 8000, 0.002
N_PATCHES = 1_000_000
POOL = 2
STRIDE = 1
BATCH_SIZE = 128
PAD = 4
N_VIEWS = 5  # original + 4 random crops


def flip_horizontal(X):
    return np.ascontiguousarray(X[:, :, ::-1, :])


def power_norm(X):
    return np.sign(X) * np.sqrt(np.abs(X))


def pad_reflect(X, pad=PAD):
    return np.pad(X, [(0, 0), (pad, pad), (pad, pad), (0, 0)], mode="reflect")


def random_crop_batch(X_padded, rng):
    """Random 32x32 crop from (N, 40, 40, 3) padded images."""
    N, H, W, Ch = X_padded.shape
    out_h = H - 2 * PAD
    out_w = W - 2 * PAD
    off_h = rng.randint(0, 2 * PAD + 1, size=N)
    off_w = rng.randint(0, 2 * PAD + 1, size=N)
    out = np.empty((N, out_h, out_w, Ch), dtype=X_padded.dtype)
    for i in range(N):
        h, w = off_h[i], off_w[i]
        out[i] = X_padded[i, h:h+out_h, w:w+out_w, :]
    return out


def encode_multicrop_avg(X, centroids, zca_mean, zca_W, n_views, rng, log, label=""):
    """Encode each image N_VIEWS times (orig + random crops), average features."""
    N = len(X)
    feat_dim = POOL * POOL * K

    # View 0: original (no crop)
    with Timer(log, f"  {label} view 0 (original) encode"):
        F_sum = encode_images_gpu(X, centroids, zca_mean, zca_W,
                                  patch=P, stride=STRIDE, pool=POOL,
                                  batch_size=BATCH_SIZE).astype(np.float64)

    # Views 1..n_views-1: random crops
    X_padded = pad_reflect(X)
    for v in range(1, n_views):
        X_crop = random_crop_batch(X_padded, rng)
        with Timer(log, f"  {label} view {v} (crop) encode"):
            F_v = encode_images_gpu(X_crop, centroids, zca_mean, zca_W,
                                    patch=P, stride=STRIDE, pool=POOL,
                                    batch_size=BATCH_SIZE)
        F_sum += F_v.astype(np.float64)
        del X_crop, F_v
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    del X_padded
    F_avg = (F_sum / n_views).astype(np.float32)
    del F_sum
    return F_avg


def main():
    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"P={P} K={K} C={C} n_views={N_VIEWS}")

    with Timer(log, "load data"):
        X, y, _ = load_train_cached(n_jobs=8)
        Xte, test_names = load_test_cached(n_jobs=8)

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0)
    y_tr, y_val = y[idx_tr], y[idx_val]
    rng = np.random.RandomState(42)

    centroids, zca_mean, zca_W = get_or_fit_dict(
        X[idx_tr], K, P, N_PATCHES, log, np.random.RandomState(0))

    # ---- Val split: multi-crop avg on 45k train + plain 5k val ----
    log.info("\n--- Val split encoding ---")
    X_tr = X[idx_tr]
    X_val = X[idx_val]

    # Flip aug: encode orig 45k and flipped 45k, each with multi-crop avg
    with Timer(log, "multi-crop avg encode train orig (45k)"):
        F_tr_orig = encode_multicrop_avg(X_tr, centroids, zca_mean, zca_W,
                                         N_VIEWS, rng, log, "train_orig")
    with Timer(log, "multi-crop avg encode train flip (45k)"):
        F_tr_flip = encode_multicrop_avg(flip_horizontal(X_tr), centroids, zca_mean, zca_W,
                                         N_VIEWS, rng, log, "train_flip")
    F_tr_aug = np.concatenate([F_tr_orig, F_tr_flip], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_tr])
    del F_tr_orig, F_tr_flip
    log.info(f"F_tr_aug: {F_tr_aug.shape}")

    # Val: plain encode (no multi-crop, since test won't have it either)
    with Timer(log, "encode val (5k, plain)"):
        F_val = encode_images_gpu(X_val, centroids, zca_mean, zca_W,
                                  patch=P, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)

    # Power norm + scale + fit
    F_tr_aug = power_norm(F_tr_aug)
    F_val_pn = power_norm(F_val)
    sc = StandardScaler().fit(F_tr_aug)
    F_tr_s = sc.transform(F_tr_aug).astype(np.float32)
    F_val_s = sc.transform(F_val_pn).astype(np.float32)
    del F_tr_aug, F_val, F_val_pn
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    t0 = time.time()
    clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2", max_iter=5000, tol=1e-4)
    clf.fit(F_tr_s, y_tr_aug)
    preds = clf.predict(F_val_s)
    if hasattr(preds, "get"): preds = preds.get()
    val_acc = float(accuracy_score(y_val, preds))
    log.info(f"\n*** multi-crop avg val = {val_acc:.4f} (fit {time.time()-t0:.1f}s) ***")
    log.info(f"    (baseline pnorm val was 0.8136)")

    del F_tr_s, F_val_s, sc
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # ---- Full refit + TTA ----
    log.info("\n--- Full refit ---")
    with Timer(log, "multi-crop avg encode full orig (50k)"):
        F_full_orig = encode_multicrop_avg(X, centroids, zca_mean, zca_W,
                                           N_VIEWS, rng, log, "full_orig")
    with Timer(log, "multi-crop avg encode full flip (50k)"):
        F_full_flip = encode_multicrop_avg(flip_horizontal(X), centroids, zca_mean, zca_W,
                                           N_VIEWS, rng, log, "full_flip")
    F_full = np.concatenate([F_full_orig, F_full_flip], axis=0)
    y_full = np.concatenate([y, y])
    del F_full_orig, F_full_flip
    gc.collect()

    F_full = power_norm(F_full)
    sc_full = StandardScaler().fit(F_full)
    F_full_s = sc_full.transform(F_full).astype(np.float32)
    del F_full
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    t0 = time.time()
    final_clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2", max_iter=5000, tol=1e-5)
    final_clf.fit(F_full_s, y_full)
    log.info(f"full-refit fit {time.time()-t0:.1f}s")
    del F_full_s
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # Test: multi-crop avg for test too (same n_views)
    with Timer(log, "multi-crop avg encode test (10k)"):
        F_te = encode_multicrop_avg(Xte, centroids, zca_mean, zca_W,
                                    N_VIEWS, rng, log, "test_orig")
    with Timer(log, "multi-crop avg encode test-flip (10k)"):
        F_te_flip = encode_multicrop_avg(flip_horizontal(Xte), centroids, zca_mean, zca_W,
                                         N_VIEWS, rng, log, "test_flip")

    F_te_s = sc_full.transform(power_norm(F_te)).astype(np.float32)
    F_te_flip_s = sc_full.transform(power_norm(F_te_flip)).astype(np.float32)
    del F_te, F_te_flip
    gc.collect()

    do = final_clf.decision_function(F_te_s)
    df = final_clf.decision_function(F_te_flip_s)
    if hasattr(do, "get"): do = do.get()
    if hasattr(df, "get"): df = df.get()

    preds_no = do.argmax(axis=1)
    preds_tta = (do + df).argmax(axis=1)

    sub1 = SUB_DIR / f"sub_run18_mcavg_pnorm_C{C}.csv"
    sub2 = SUB_DIR / f"sub_run18_mcavg_pnorm_tta_C{C}.csv"
    save_submission(test_names, preds_no, sub1)
    save_submission(test_names, preds_tta, sub2)
    log.info(f"wrote {sub1.name}")
    log.info(f"wrote {sub2.name}")

    diffs = int((preds_no != preds_tta).sum())
    log.info(f"TTA changed {diffs}/{len(preds_no)} preds ({100*diffs/len(preds_no):.2f}%)")
    log.info(f"\n=== {RUN_NAME} done ===")
    log.info(f"multi-crop avg val = {val_acc:.4f}")


if __name__ == "__main__":
    main()
