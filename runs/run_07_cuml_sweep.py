"""Run 07 — parameterized Coates-Ng sweep on GPU (cupy encoding + sklearn LinearSVC).

Broad exploration grid: K × C, patch=6, 2x2 pool. Each (K, best_C) yields a
submission CSV. Reuses dictionary + feature caches across runs.

Designed for AutoDL / cloud boxes with lots of RAM (sklearn LinearSVC primal
fit on 50k × 4K-16K float32 data fits comfortably in <30 GB).

Usage:
  python runs/run_07_cuml_sweep.py \\
    --k-list 1200,1600,2000,2400,3200,4000 \\
    --c-list 0.001,0.003,0.01,0.03 \\
    --patch 6 --pool 2 --n-patches 500000
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Use cuML GPU LinearSVC — sklearn LinearSVC is pathologically slow on this
# Xeon+OpenBLAS build. cuML fits in seconds via L-BFGS on the 5090.
# Lazy import so other scripts can reuse encode_images_gpu / get_or_fit_dict
# without pulling cuML (e.g. sklearn-env refit scripts).
try:
    from cuml.svm import LinearSVC  # noqa: F401
except ImportError:
    LinearSVC = None

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

import cupy as cp

RUN_NAME_BASE = "run_07_cuml_sweep"
RUN_NAME = RUN_NAME_BASE

EPS_NORM = 10.0
EPS_ZCA = 0.1


# ---------------------------------------------------------------- patches

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


def extract_all_patches_batch(imgs: np.ndarray, patch: int, stride: int) -> np.ndarray:
    B, H, W, C = imgs.shape
    nh = (H - patch) // stride + 1
    nw = (W - patch) // stride + 1
    from numpy.lib.stride_tricks import sliding_window_view
    imgs_f = imgs.astype(np.float32)
    sw = sliding_window_view(imgs_f, (patch, patch), axis=(1, 2))
    sw = sw[:, ::stride, ::stride, :, :, :]
    sw = np.transpose(sw, (0, 1, 2, 4, 5, 3))
    return np.ascontiguousarray(sw).reshape(B * nh * nw, patch * patch * C)


def contrast_normalize_np(P: np.ndarray) -> np.ndarray:
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


def apply_zca_np(P: np.ndarray, mean: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (P - mean) @ W


# ---------------------------------------------------------------- GPU encode

def encode_images_gpu(imgs: np.ndarray, centroids: np.ndarray,
                      zca_mean: np.ndarray, zca_W: np.ndarray,
                      patch: int, stride: int, pool: int,
                      batch_size: int) -> np.ndarray:
    N = len(imgs)
    K = centroids.shape[0]
    H = W_ = imgs.shape[1]
    nh = (H - patch) // stride + 1
    nw = (W_ - patch) // stride + 1
    feat_dim = pool * pool * K
    out = np.empty((N, feat_dim), dtype=np.float32)

    c_gpu = cp.asarray(centroids, dtype=cp.float32)
    c_sq = (c_gpu * c_gpu).sum(axis=1)
    zca_mean_gpu = cp.asarray(zca_mean, dtype=cp.float32)
    zca_W_gpu = cp.asarray(zca_W, dtype=cp.float32)

    half_h = nh // pool
    half_w = nw // pool

    mempool = cp.get_default_memory_pool()

    for start in range(0, N, batch_size):
        stop = min(N, start + batch_size)
        batch = imgs[start:stop]
        B = len(batch)

        patches_np = extract_all_patches_batch(batch, patch, stride)
        patches = cp.asarray(patches_np, dtype=cp.float32)
        mean = patches.mean(axis=1, keepdims=True)
        var = patches.var(axis=1, keepdims=True)
        patches = (patches - mean) / cp.sqrt(var + EPS_NORM)
        patches = (patches - zca_mean_gpu) @ zca_W_gpu

        p_sq = (patches * patches).sum(axis=1, keepdims=True)
        dots = patches @ c_gpu.T
        d2 = p_sq + c_sq[None, :] - 2.0 * dots
        d2 = cp.maximum(d2, 0.0)
        d = cp.sqrt(d2)
        mu = d.mean(axis=1, keepdims=True)
        f = cp.maximum(0.0, mu - d)

        f = f.reshape(B, nh, nw, K)
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

        del patches, p_sq, dots, d2, d, mu, f, pooled, mean, var
        mempool.free_all_blocks()

    del c_gpu, c_sq, zca_mean_gpu, zca_W_gpu
    mempool.free_all_blocks()
    return out


# ---------------------------------------------------------------- caching

TAG = "run07"

def dict_cache_path(K, patch, n_patches):
    return CACHE_DIR / f"{TAG}_dict_K{K}_P{patch}_N{n_patches}.npz"

def feat_cache_path(split, K, patch):
    return CACHE_DIR / f"{TAG}_feat_{split}_K{K}_P{patch}.npy"


def get_or_fit_dict(X_tr, K, patch, n_patches, log, rng):
    p = dict_cache_path(K, patch, n_patches)
    for alt_tag in ("sweep", "gpu"):
        alt = CACHE_DIR / f"{alt_tag}_dict_K{K}_P{patch}_N{n_patches}.npz"
        if not p.exists() and alt.exists():
            log.info(f"reusing dict cache {alt.name}")
            d = np.load(alt)
            return d["centroids"], d["zca_mean"], d["zca_W"]
    if p.exists():
        d = np.load(p)
        log.info(f"loaded dict {p.name}")
        return d["centroids"], d["zca_mean"], d["zca_W"]

    with Timer(log, f"sample {n_patches} patches (P={patch})"):
        P = extract_random_patches(X_tr, n_patches, patch, rng)
    with Timer(log, "contrast normalize"):
        P = contrast_normalize_np(P)
    with Timer(log, "fit ZCA"):
        zca_mean, zca_W = compute_zca(P)
    with Timer(log, "apply ZCA"):
        P = apply_zca_np(P, zca_mean, zca_W)
    with Timer(log, f"MiniBatchKMeans K={K}"):
        km = MiniBatchKMeans(
            n_clusters=K, batch_size=4096, n_init=3, max_iter=300,
            random_state=0, verbose=0,
        )
        km.fit(P)
        centroids = km.cluster_centers_.astype(np.float32)
    del P; gc.collect()
    np.savez(p, centroids=centroids, zca_mean=zca_mean, zca_W=zca_W)
    log.info(f"saved dict {p.name}")
    return centroids, zca_mean, zca_W


def get_or_encode(X, split, K, patch, centroids, zca_mean, zca_W,
                  stride, pool, batch_size, log):
    p = feat_cache_path(split, K, patch)
    for alt_tag in ("sweep", "gpu"):
        alt = CACHE_DIR / f"{alt_tag}_feat_{split}_K{K}_P{patch}.npy"
        if not p.exists() and alt.exists():
            log.info(f"reusing features {alt.name}")
            return np.load(alt)
    if p.exists():
        log.info(f"loaded features {p.name}")
        return np.load(p)
    with Timer(log, f"GPU encode {split} ({len(X)} imgs, K={K}, P={patch})"):
        F = encode_images_gpu(X, centroids, zca_mean, zca_W,
                              patch=patch, stride=stride, pool=pool,
                              batch_size=batch_size)
    np.save(p, F)
    log.info(f"saved features {p.name} shape {F.shape}")
    return F


# ---------------------------------------------------------------- main

def parse_list(s, cast):
    return [cast(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k-list", default="1200,1600,2000,2400,3200,4000")
    ap.add_argument("--c-list", default="0.001,0.003,0.01,0.03")
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--patch-list", default=None,
                    help="comma-separated patch sizes; overrides --patch when set")
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--n-patches", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=400)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--parallel-c", type=int, default=4,
                    help="number of LinearSVC C fits to run in parallel per (P,K)")
    ap.add_argument("--tag", default=None,
                    help="suffix appended to RUN_NAME for log/json isolation across parallel sweeps")
    args = ap.parse_args()

    global RUN_NAME
    if args.tag:
        RUN_NAME = f"{RUN_NAME_BASE}_{args.tag}"
    log_file = LOG_DIR / f"{RUN_NAME}.log"
    log = get_logger(RUN_NAME, log_file)
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"args: {vars(args)}")

    K_list = parse_list(args.k_list, int)
    C_list = parse_list(args.c_list, float)
    P_list = parse_list(args.patch_list, int) if args.patch_list else [args.patch]

    if args.smoke:
        K_list = [100]
        C_list = [0.01]
        P_list = [args.patch]
        args.n_patches = 20_000

    dev = cp.cuda.Device(0)
    free_mem, total_mem = dev.mem_info
    log.info(f"GPU free={free_mem/2**30:.2f}/{total_mem/2**30:.2f} GiB")

    with Timer(log, "load train/test"):
        X, y, _ = load_train_cached(n_jobs=args.n_jobs)
        Xte, test_names = load_test_cached(n_jobs=args.n_jobs)
    log.info(f"train {X.shape}  test {Xte.shape}")

    if args.smoke:
        X = X[:1500]; y = y[:1500]
        Xte = Xte[:500]; test_names = test_names[:500]

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0
    )
    y_tr = y[idx_tr]; y_val = y[idx_val]
    log.info(f"train sub {len(idx_tr)}  val {len(idx_val)}")

    total_configs = len(P_list) * len(K_list) * len(C_list)
    log.info(f"sweep: {len(P_list)} P × {len(K_list)} K × {len(C_list)} C = {total_configs} configs")
    log.info(f"parallel C fits per (P,K): {args.parallel_c}")

    rng = np.random.RandomState(0)
    results = []
    best_per_pk = {}

    for patch in P_list:
        for K in K_list:
            log.info(f"\n### P={patch} K={K} ###")
            centroids, zca_mean, zca_W = get_or_fit_dict(
                X[idx_tr], K, patch, args.n_patches, log, rng
            )
            F_full_tr = get_or_encode(X, "train", K, patch,
                                      centroids, zca_mean, zca_W,
                                      args.stride, args.pool, args.batch_size, log)
            F_test = get_or_encode(Xte, "test", K, patch,
                                   centroids, zca_mean, zca_W,
                                   args.stride, args.pool, args.batch_size, log)

            F_tr = F_full_tr[idx_tr]
            F_val = F_full_tr[idx_val]
            scaler = StandardScaler().fit(F_tr)
            F_tr_s = scaler.transform(F_tr).astype(np.float32)
            F_val_s = scaler.transform(F_val).astype(np.float32)
            del F_tr, F_val; gc.collect()

            def fit_one(C, F_tr_s=F_tr_s, F_val_s=F_val_s):
                t = time.time()
                clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                                max_iter=3000, tol=1e-5)
                clf.fit(F_tr_s, y_tr)
                preds = clf.predict(F_val_s)
                if hasattr(preds, "get"):
                    preds = preds.get()
                acc = float(accuracy_score(y_val, preds))
                return C, acc, time.time() - t

            best_local = {"acc": -1.0}
            with Timer(log, f"parallel LinearSVC P={patch} K={K} ({len(C_list)} C, {args.parallel_c}-way)"):
                with ThreadPoolExecutor(max_workers=args.parallel_c) as ex:
                    for C, acc, dt in ex.map(fit_one, C_list):
                        log.info(f"  P={patch} K={K} C={C} val={acc:.4f}  ({dt:.1f}s)")
                        results.append(dict(P=patch, K=K, C=C, val_acc=acc))
                        if acc > best_local["acc"]:
                            best_local = dict(acc=acc, C=C)

            log.info(f"  [best P={patch} K={K}] C={best_local['C']} val={best_local['acc']:.4f}")
            best_per_pk[(patch, K)] = best_local

            # Refit best C on full 50k and save submission
            with Timer(log, f"refit full P={patch} K={K} C={best_local['C']}"):
                scaler_full = StandardScaler().fit(F_full_tr)
                F_full_s = scaler_full.transform(F_full_tr).astype(np.float32)
                F_te_s = scaler_full.transform(F_test).astype(np.float32)
                final_clf = LinearSVC(C=best_local["C"], loss="squared_hinge",
                                      penalty="l2", max_iter=3000, tol=1e-6)
                final_clf.fit(F_full_s, y)
                preds = final_clf.predict(F_te_s)
                if hasattr(preds, "get"):
                    preds = preds.get()

            sub_path = SUB_DIR / f"sub_run07_P{patch}_K{K}_C{best_local['C']}.csv"
            save_submission(test_names, preds, sub_path)
            log.info(f"  wrote {sub_path.name} (P={patch}, K={K}, C={best_local['C']}, val={best_local['acc']:.4f})")

            del F_full_tr, F_test, F_tr_s, F_val_s, F_full_s, F_te_s, scaler, scaler_full, final_clf
            gc.collect()

    # ---- Summary ----
    results.sort(key=lambda r: -r["val_acc"])
    summary_json = LOG_DIR / f"{RUN_NAME}_results.json"
    with open(summary_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info("\n=== sweep complete ===")
    log.info("Top 10 by val_acc:")
    for r in results[:10]:
        log.info(f"  P={r['P']} K={r['K']}  C={r['C']:.4f}  val={r['val_acc']:.4f}")
    log.info(f"\nSubmission CSVs saved to {SUB_DIR}:")
    for (patch, K), best in sorted(best_per_pk.items()):
        log.info(f"  P={patch} K={K}  C={best['C']}  val={best['acc']:.4f}")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
