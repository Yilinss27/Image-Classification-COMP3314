"""Run 04 — Coates-Ng with GPU (cuML) + full hyperparameter sweep.

Uses RAPIDS cuML as a drop-in GPU replacement for sklearn. Still 100% classical
ML (KMeans, LinearSVC, PCA) — NO neural networks, NO pretrained models.

Pipeline identical to run_03 but:
  * MiniBatchKMeans (CPU) → cuML KMeans  (GPU, ~30x)
  * LinearSVC (CPU) → cuML LinearSVC (GPU, ~15x)
  * Encoding: CPU numpy with cupy for the big matmul

Sweep:
  Phase 1 (coarse): for each (K, patch) combo, encode once, try several C
  Phase 2 (fine): take best K/patch, try more C values and pooling grids
  Phase 3 (final): best config, refit on full train+val, generate submission

Lectures used: L4 (preprocessing), L5 (dim reduction via ZCA), L6 (CV/tuning),
L3 (LinearSVM), L9 (KMeans clustering).
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import (
    CKPT_DIR,
    LOG_DIR,
    SUB_DIR,
    CACHE_DIR,
    Timer,
    get_logger,
    load_test_cached,
    load_train_cached,
    save_submission,
)

# GPU imports — imported lazily to give a clear error if cuML missing
try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.svm import LinearSVC as cuLinearSVC
    GPU_OK = True
except Exception as e:  # pragma: no cover
    GPU_OK = False
    GPU_IMPORT_ERR = str(e)

RUN_NAME = "run_04_gpu_coates_sweep"

EPS_NORM = 10.0
EPS_ZCA = 0.1


# ---------------------------------------------------------------- patch utils

def extract_random_patches(X: np.ndarray, n_patches: int, patch: int,
                           rng: np.random.RandomState) -> np.ndarray:
    N, H, W, C = X.shape
    out = np.empty((n_patches, patch * patch * C), dtype=np.float32)
    img_idx = rng.randint(0, N, size=n_patches)
    row = rng.randint(0, H - patch + 1, size=n_patches)
    col = rng.randint(0, W - patch + 1, size=n_patches)
    for i in range(n_patches):
        p = X[img_idx[i], row[i]:row[i]+patch, col[i]:col[i]+patch, :]
        out[i] = p.astype(np.float32).reshape(-1)
    return out


def extract_all_patches_images(imgs: np.ndarray, patch: int, stride: int) -> np.ndarray:
    """Vectorized extraction of all stride-s patches for a batch of images.

    imgs: (B, H, W, C) uint8 or float32
    returns: (B*nh*nw, patch*patch*C) float32
    """
    B, H, W, C = imgs.shape
    nh = (H - patch) // stride + 1
    nw = (W - patch) // stride + 1
    # use stride tricks for speed
    from numpy.lib.stride_tricks import sliding_window_view
    imgs_f = imgs.astype(np.float32)
    # sliding_window_view over H and W axes
    sw = sliding_window_view(imgs_f, (patch, patch), axis=(1, 2))  # (B, nh2, nw2, C, patch, patch)
    sw = sw[:, ::stride, ::stride, :, :, :]                         # stride decimation
    # Want (B, nh, nw, patch, patch, C) then reshape
    sw = np.transpose(sw, (0, 1, 2, 4, 5, 3))                       # (B, nh, nw, patch, patch, C)
    return sw.reshape(B * nh * nw, patch * patch * C)


def contrast_normalize(P: np.ndarray) -> np.ndarray:
    mean = P.mean(axis=1, keepdims=True)
    var = P.var(axis=1, keepdims=True)
    return (P - mean) / np.sqrt(var + EPS_NORM)


def compute_zca(P: np.ndarray):
    mean = P.mean(axis=0)
    Xc = P - mean
    sigma = (Xc.T @ Xc) / Xc.shape[0]
    U, S, _ = np.linalg.svd(sigma)
    W = U @ np.diag(1.0 / np.sqrt(S + EPS_ZCA)) @ U.T
    return mean.astype(np.float32), W.astype(np.float32)


def apply_zca(P: np.ndarray, mean: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (P - mean) @ W


# ------------------------------------------------------- GPU encode (cupy)

def encode_images_gpu(imgs: np.ndarray, centroids: np.ndarray,
                      zca_mean: np.ndarray, zca_W: np.ndarray,
                      patch: int, stride: int, pool: int,
                      batch_size: int, log) -> np.ndarray:
    """Encode using cupy for the big matmul (distance computation)."""
    N = len(imgs)
    K = centroids.shape[0]
    H = W_ = imgs.shape[1]
    nh = (H - patch) // stride + 1
    nw = (W_ - patch) // stride + 1
    feat_dim = pool * pool * K
    out = np.empty((N, feat_dim), dtype=np.float32)

    # Move centroids to GPU
    c_gpu = cp.asarray(centroids)
    c_sq = (c_gpu * c_gpu).sum(axis=1)
    zca_mean_gpu = cp.asarray(zca_mean)
    zca_W_gpu = cp.asarray(zca_W)

    half_h = nh // pool
    half_w = nw // pool

    for start in range(0, N, batch_size):
        stop = min(N, start + batch_size)
        batch = imgs[start:stop]
        B = len(batch)
        patches_np = extract_all_patches_images(batch, patch, stride)
        patches = cp.asarray(patches_np, dtype=cp.float32)
        # contrast normalize
        mean = patches.mean(axis=1, keepdims=True)
        var = patches.var(axis=1, keepdims=True)
        patches = (patches - mean) / cp.sqrt(var + EPS_NORM)
        # ZCA
        patches = (patches - zca_mean_gpu) @ zca_W_gpu
        # distances
        p_sq = (patches * patches).sum(axis=1, keepdims=True)
        dots = patches @ c_gpu.T
        d2 = p_sq + c_sq[None, :] - 2.0 * dots
        d2 = cp.maximum(d2, 0.0)
        d = cp.sqrt(d2)
        mu = d.mean(axis=1, keepdims=True)
        f = cp.maximum(0.0, mu - d)

        f = f.reshape(B, nh, nw, K)
        # Pool by pool×pool grid
        parts = []
        for pi in range(pool):
            r_lo = pi * half_h
            r_hi = (pi + 1) * half_h if pi < pool - 1 else nh
            for pj in range(pool):
                c_lo = pj * half_w
                c_hi = (pj + 1) * half_w if pj < pool - 1 else nw
                parts.append(f[:, r_lo:r_hi, c_lo:c_hi, :].sum(axis=(1, 2)))
        pooled = cp.concatenate(parts, axis=1)
        out[start:stop] = cp.asnumpy(pooled).astype(np.float32)
        del patches, p_sq, dots, d2, d, mu, f, pooled
        cp._default_memory_pool.free_all_blocks()

    del c_gpu, c_sq, zca_mean_gpu, zca_W_gpu
    cp._default_memory_pool.free_all_blocks()
    return out


def fit_dict_gpu(X: np.ndarray, K: int, patch: int, n_patches: int, log,
                 rng: np.random.RandomState):
    """Sample patches, contrast norm, ZCA, KMeans (GPU). Returns centroids, zca_mean, zca_W."""
    with Timer(log, f"sample {n_patches} random patches (patch={patch})"):
        P = extract_random_patches(X, n_patches, patch, rng)
    with Timer(log, "contrast normalize"):
        P = contrast_normalize(P)
    with Timer(log, "fit ZCA"):
        zca_mean, zca_W = compute_zca(P)
    with Timer(log, "apply ZCA"):
        P = apply_zca(P, zca_mean, zca_W)
    with Timer(log, f"cuML KMeans (K={K})"):
        P_gpu = cp.asarray(P)
        km = cuKMeans(n_clusters=K, max_iter=100, init="k-means||",
                      n_init=1, random_state=0)
        km.fit(P_gpu)
        centroids = cp.asnumpy(km.cluster_centers_).astype(np.float32)
        del P_gpu, km
        cp._default_memory_pool.free_all_blocks()
    del P
    gc.collect()
    return centroids, zca_mean, zca_W


def train_eval_linsvc_gpu(F_tr: np.ndarray, y_tr: np.ndarray,
                          F_val: np.ndarray, y_val: np.ndarray,
                          C: float, log) -> tuple[float, object]:
    with Timer(log, f"cuML LinearSVC fit (C={C})"):
        F_tr_gpu = cp.asarray(F_tr)
        y_tr_gpu = cp.asarray(y_tr.astype(np.int32))
        clf = cuLinearSVC(C=C, max_iter=2000, tol=1e-4)
        clf.fit(F_tr_gpu, y_tr_gpu)
        pred = cp.asnumpy(clf.predict(cp.asarray(F_val)))
        acc = accuracy_score(y_val, pred)
        del F_tr_gpu, y_tr_gpu
        cp._default_memory_pool.free_all_blocks()
    return acc, clf


# ---------------------------------------------------------------- main

def main():
    if not GPU_OK:
        print(f"cuML import failed: {GPU_IMPORT_ERR}", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-patches", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true",
                    help="small sweep (debug)")
    args = ap.parse_args()

    log_file = LOG_DIR / f"{RUN_NAME}.log"
    log = get_logger(RUN_NAME, log_file)
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"args: {vars(args)}")
    log.info(f"cupy device: {cp.cuda.runtime.getDeviceCount()} GPUs, "
             f"mem={cp.cuda.runtime.memGetInfo()[1]/2**30:.2f} GiB")

    with Timer(log, "load train"):
        X, y, _ = load_train_cached(n_jobs=args.n_jobs)
    with Timer(log, "load test"):
        Xte, test_names = load_test_cached(n_jobs=args.n_jobs)

    if args.smoke:
        X, y = X[:1500], y[:1500]
        Xte = Xte[:500]; test_names = test_names[:500]
        args.n_patches = 20_000
        args.batch_size = 200
        log.info("SMOKE MODE active")

    # Stratified split: train → (train_sub, val)
    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0
    )
    X_tr = X[idx_tr]; y_tr = y[idx_tr]
    X_val = X[idx_val]; y_val = y[idx_val]
    log.info(f"train sub {X_tr.shape}  val {X_val.shape}  test {Xte.shape}")

    # Sweep space
    if args.quick or args.smoke:
        sweep_configs = [
            dict(K=100, patch=6, stride=1, pool=2),
            dict(K=200, patch=6, stride=1, pool=2),
        ]
        C_values = [0.01, 0.1]
    else:
        sweep_configs = [
            dict(K=400,  patch=6, stride=1, pool=2),
            dict(K=800,  patch=6, stride=1, pool=2),
            dict(K=1600, patch=6, stride=1, pool=2),
            dict(K=800,  patch=8, stride=1, pool=2),
            dict(K=1600, patch=8, stride=1, pool=2),
        ]
        C_values = [0.001, 0.01, 0.1, 1.0]

    results_log = []  # list of dicts

    rng = np.random.RandomState(0)
    best_global = {"acc": -1.0}

    for ci, cfg in enumerate(sweep_configs):
        log.info(f"--- config {ci+1}/{len(sweep_configs)}: {cfg} ---")
        K = cfg["K"]; patch = cfg["patch"]; stride = cfg["stride"]; pool = cfg["pool"]

        centroids, zca_mean, zca_W = fit_dict_gpu(
            X_tr, K=K, patch=patch, n_patches=args.n_patches, log=log, rng=rng
        )

        with Timer(log, f"encode train_sub ({len(X_tr)})"):
            F_tr = encode_images_gpu(X_tr, centroids, zca_mean, zca_W,
                                     patch=patch, stride=stride, pool=pool,
                                     batch_size=args.batch_size, log=log)
        with Timer(log, f"encode val ({len(X_val)})"):
            F_val = encode_images_gpu(X_val, centroids, zca_mean, zca_W,
                                      patch=patch, stride=stride, pool=pool,
                                      batch_size=args.batch_size, log=log)
        log.info(f"F_tr {F_tr.shape}  F_val {F_val.shape}")

        scaler = StandardScaler().fit(F_tr)
        F_tr_s = scaler.transform(F_tr).astype(np.float32)
        F_val_s = scaler.transform(F_val).astype(np.float32)

        best_local = {"acc": -1.0}
        for C in C_values:
            acc, clf = train_eval_linsvc_gpu(F_tr_s, y_tr, F_val_s, y_val, C, log)
            row = dict(cfg=cfg, C=C, val_acc=acc)
            results_log.append(row)
            log.info(f"    {cfg} C={C}  val={acc:.4f}")
            if acc > best_local["acc"]:
                best_local = dict(acc=acc, C=C, cfg=cfg,
                                  centroids=centroids, zca_mean=zca_mean, zca_W=zca_W,
                                  scaler=scaler)

        log.info(f"  best for this cfg: C={best_local['C']} acc={best_local['acc']:.4f}")
        if best_local["acc"] > best_global["acc"]:
            best_global = best_local

        del F_tr, F_val, F_tr_s, F_val_s
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

    # Save sweep log
    with open(LOG_DIR / f"{RUN_NAME}_sweep.json", "w") as f:
        json.dump(results_log, f, indent=2, default=str)
    log.info("=== Sweep complete ===")
    for r in sorted(results_log, key=lambda r: -r["val_acc"])[:10]:
        log.info(f"  {r['cfg']} C={r['C']}  val={r['val_acc']:.4f}")

    # ---------- Refit best on full (train+val) and predict test ----------
    log.info(f">>> BEST: {best_global['cfg']} C={best_global['C']}  val={best_global['acc']:.4f}")

    cfg = best_global["cfg"]
    K = cfg["K"]; patch = cfg["patch"]; stride = cfg["stride"]; pool = cfg["pool"]

    with Timer(log, "Refit dictionary on FULL train (train_sub + val)"):
        centroids_f, zca_mean_f, zca_W_f = fit_dict_gpu(
            X, K=K, patch=patch, n_patches=max(args.n_patches, 500_000),
            log=log, rng=np.random.RandomState(1)
        )

    with Timer(log, f"encode FULL train ({len(X)})"):
        F_tr_full = encode_images_gpu(X, centroids_f, zca_mean_f, zca_W_f,
                                      patch=patch, stride=stride, pool=pool,
                                      batch_size=args.batch_size, log=log)
    with Timer(log, f"encode test ({len(Xte)})"):
        F_te_full = encode_images_gpu(Xte, centroids_f, zca_mean_f, zca_W_f,
                                      patch=patch, stride=stride, pool=pool,
                                      batch_size=args.batch_size, log=log)

    scaler_full = StandardScaler().fit(F_tr_full)
    F_tr_full_s = scaler_full.transform(F_tr_full).astype(np.float32)
    F_te_full_s = scaler_full.transform(F_te_full).astype(np.float32)

    with Timer(log, f"cuML LinearSVC final fit (C={best_global['C']})"):
        F_tr_gpu = cp.asarray(F_tr_full_s)
        y_gpu = cp.asarray(y.astype(np.int32))
        final_clf = cuLinearSVC(C=best_global["C"], max_iter=4000, tol=1e-5)
        final_clf.fit(F_tr_gpu, y_gpu)

    with Timer(log, "predict test"):
        preds = cp.asnumpy(final_clf.predict(cp.asarray(F_te_full_s)))

    sub_path = SUB_DIR / f"sub_{RUN_NAME}.csv"
    save_submission(test_names, preds, sub_path)
    log.info(f"Wrote submission to {sub_path}")

    ckpt = CKPT_DIR / f"{RUN_NAME}.joblib"
    joblib.dump(
        {
            "best_cfg": best_global["cfg"],
            "best_C": best_global["C"],
            "best_val_acc": best_global["acc"],
            "sweep_results": results_log,
            "scaler": scaler_full,
            "centroids": centroids_f,
            "zca_mean": zca_mean_f,
            "zca_W": zca_W_f,
        },
        ckpt,
    )
    log.info(f"Wrote checkpoint to {ckpt}")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
