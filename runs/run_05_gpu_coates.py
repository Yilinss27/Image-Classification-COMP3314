"""Run 05 — Coates-Ng on GPU via cupy for the encoding bottleneck.

The encoding step in run_03/run_04 is dominated by one big matmul per batch:
(M patches × D) @ (D × K) where M ~= 400 imgs * 729 patches = 291600.
On a 32-core CPU this takes ~5 min for 50k images at K=800. On GPU via cupy
it should be ~10-30 s — roughly 10-20x faster.

Design:
  * Dictionary learning (patch sampling + contrast norm + ZCA + KMeans) runs on
    CPU using sklearn.MiniBatchKMeans. Small compared to encoding, no reason
    to involve GPU.
  * Image encoding runs on GPU via cupy. Patches are extracted on CPU, moved
    to GPU in batches, and the distance + triangle + pool ops happen there.
  * Final classifier is sklearn LinearSVC on CPU — operates on compact
    pooled features (50k × 4K), fast.

This keeps each component simple and plays to each device's strength.
Classical ML throughout — no neural networks, no pretrained models.

Because cupy and the CPU sweep in run_04 don't share memory, this script
can run in parallel with run_04 on a different tmux session without conflict.

Sweep focus: run_04 already covers K in {400, 800, 1600} × patch in {6, 8},
so run_05 targets the more ambitious K=3200 configs (Coates paper peak).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
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

try:
    import cupy as cp
    GPU_OK = True
except Exception as e:
    GPU_OK = False
    GPU_ERR = repr(e)

RUN_NAME = "run_05_gpu_coates"

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
    """Vectorized: (B, H, W, C) → (B*nh*nw, patch*patch*C) float32."""
    B, H, W, C = imgs.shape
    nh = (H - patch) // stride + 1
    nw = (W - patch) // stride + 1
    from numpy.lib.stride_tricks import sliding_window_view
    imgs_f = imgs.astype(np.float32)
    sw = sliding_window_view(imgs_f, (patch, patch), axis=(1, 2))  # (B, nh*stride, nw*stride, C, patch, patch)
    sw = sw[:, ::stride, ::stride, :, :, :]
    sw = np.transpose(sw, (0, 1, 2, 4, 5, 3))  # (B, nh, nw, p, p, C)
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
                      batch_size: int, log) -> np.ndarray:
    """Encode images with cupy-accelerated matmul for distance computation."""
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

        # CPU: extract patches (stride tricks, fast)
        patches_np = extract_all_patches_batch(batch, patch, stride)  # (B*nh*nw, P*P*C)

        # GPU: normalize + whiten + distance + triangle + pool
        patches = cp.asarray(patches_np, dtype=cp.float32)
        mean = patches.mean(axis=1, keepdims=True)
        var = patches.var(axis=1, keepdims=True)
        patches = (patches - mean) / cp.sqrt(var + EPS_NORM)
        patches = (patches - zca_mean_gpu) @ zca_W_gpu

        # Distance ||p - c||^2 = ||p||^2 + ||c||^2 - 2 p.c
        p_sq = (patches * patches).sum(axis=1, keepdims=True)
        dots = patches @ c_gpu.T
        d2 = p_sq + c_sq[None, :] - 2.0 * dots
        d2 = cp.maximum(d2, 0.0)
        d = cp.sqrt(d2)
        mu = d.mean(axis=1, keepdims=True)
        f = cp.maximum(0.0, mu - d)  # (B*nh*nw, K)

        f = f.reshape(B, nh, nw, K)
        parts = []
        for pi in range(pool):
            r_lo = pi * half_h
            r_hi = (pi + 1) * half_h if pi < pool - 1 else nh
            for pj in range(pool):
                c_lo = pj * half_w
                c_hi = (pj + 1) * half_w if pj < pool - 1 else nw
                parts.append(f[:, r_lo:r_hi, c_lo:c_hi, :].sum(axis=(1, 2)))
        pooled = cp.concatenate(parts, axis=1)  # (B, pool*pool*K)
        out[start:stop] = cp.asnumpy(pooled).astype(np.float32)

        # Free GPU buffers for this batch
        del patches, p_sq, dots, d2, d, mu, f, pooled, mean, var
        mempool.free_all_blocks()

    del c_gpu, c_sq, zca_mean_gpu, zca_W_gpu
    mempool.free_all_blocks()
    return out


# ---------------------------------------------------------------- caching

def dict_cache_path(K: int, patch: int, n_patches: int, tag: str) -> Path:
    return CACHE_DIR / f"{tag}_dict_K{K}_P{patch}_N{n_patches}.npz"


def feat_cache_path(split: str, K: int, patch: int, tag: str) -> Path:
    return CACHE_DIR / f"{tag}_feat_{split}_K{K}_P{patch}.npy"


def get_or_fit_dict(X_tr, K, patch, n_patches, log, rng, tag):
    p = dict_cache_path(K, patch, n_patches, tag)
    # Also reuse run_04's sweep dict cache if exists
    for alt_tag in ("sweep", "gpu"):
        alt = dict_cache_path(K, patch, n_patches, alt_tag)
        if not p.exists() and alt.exists():
            log.info(f"reusing dict cache {alt.name}")
            d = np.load(alt)
            return d["centroids"], d["zca_mean"], d["zca_W"]
    if p.exists():
        log.info(f"loaded dict {p.name}")
        d = np.load(p)
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
                  stride, pool, batch_size, log, tag):
    p = feat_cache_path(split, K, patch, tag)
    # Reuse run_04 sweep feature cache if available
    for alt_tag in ("sweep", "gpu"):
        alt = feat_cache_path(split, K, patch, alt_tag)
        if not p.exists() and alt.exists():
            log.info(f"reusing features {alt.name}")
            return np.load(alt)
    if p.exists():
        log.info(f"loaded features {p.name}")
        return np.load(p)
    with Timer(log, f"GPU encode {split} ({len(X)} imgs, K={K}, P={patch})"):
        F = encode_images_gpu(X, centroids, zca_mean, zca_W,
                              patch=patch, stride=stride, pool=pool,
                              batch_size=batch_size, log=log)
    np.save(p, F)
    log.info(f"saved features {p.name} shape {F.shape}")
    return F


# ---------------------------------------------------------------- main

def main():
    if not GPU_OK:
        print(f"cupy import failed: {GPU_ERR}", file=sys.stderr)
        sys.exit(2)

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-patches", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--configs", default="gpu")
    ap.add_argument("--tag", default="gpu")
    args = ap.parse_args()

    log_file = LOG_DIR / f"{RUN_NAME}.log"
    log = get_logger(RUN_NAME, log_file)
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"args: {vars(args)}")

    # GPU info
    dev = cp.cuda.Device(0)
    free_mem, total_mem = dev.mem_info
    log.info(f"GPU: device 0, free={free_mem/2**30:.2f} GiB total={total_mem/2**30:.2f} GiB")

    with Timer(log, "load train/test"):
        X, y, _ = load_train_cached(n_jobs=args.n_jobs)
        Xte, test_names = load_test_cached(n_jobs=args.n_jobs)
    log.info(f"train {X.shape}  test {Xte.shape}")

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0
    )
    y_tr = y[idx_tr]; y_val = y[idx_val]
    log.info(f"train sub {len(idx_tr)}  val {len(idx_val)}")

    # Sweep space — complementary to CPU run_04
    if args.smoke:
        sweep = [dict(K=100, patch=6, stride=1, pool=2)]
        C_values = [0.01, 0.1]
    else:
        # Focus on high-K configs that CPU can't afford. Skip smaller K to
        # avoid duplicating run_04's cheap configs.
        sweep = [
            dict(K=1600, patch=6, stride=1, pool=2),
            dict(K=3200, patch=6, stride=1, pool=2),
            dict(K=1600, patch=8, stride=1, pool=2),
            dict(K=3200, patch=8, stride=1, pool=2),
        ]
        C_values = [0.001, 0.003, 0.01, 0.03, 0.1]

    if args.smoke:
        X = X[:1500]; y = y[:1500]
        Xte = Xte[:500]; test_names = test_names[:500]
        args.n_patches = 20_000
        idx_tr, idx_val = train_test_split(
            np.arange(len(X)), test_size=0.1, stratify=y, random_state=0
        )
        y_tr = y[idx_tr]; y_val = y[idx_val]

    log.info(f"sweep: {len(sweep)} configs x {len(C_values)} C values = "
             f"{len(sweep)*len(C_values)} experiments")

    results_log = []
    best_global = {"acc": -1.0}

    rng = np.random.RandomState(0)

    for ci, cfg in enumerate(sweep):
        log.info(f"--- config {ci+1}/{len(sweep)}: {cfg} ---")
        K = cfg["K"]; patch = cfg["patch"]; stride = cfg["stride"]; pool = cfg["pool"]

        # Fit dictionary on (subset of) training set
        centroids, zca_mean, zca_W = get_or_fit_dict(
            X[idx_tr] if not args.smoke else X,
            K=K, patch=patch, n_patches=args.n_patches, log=log, rng=rng, tag=args.tag,
        )

        # Encode full train (so we can slice later) and test
        F_full_tr = get_or_encode(X, "train", K, patch,
                                  centroids, zca_mean, zca_W,
                                  stride, pool, args.batch_size, log, tag=args.tag)
        F_test = get_or_encode(Xte, "test", K, patch,
                               centroids, zca_mean, zca_W,
                               stride, pool, args.batch_size, log, tag=args.tag)

        F_tr = F_full_tr[idx_tr]
        F_val = F_full_tr[idx_val]

        scaler = StandardScaler().fit(F_tr)
        F_tr_s = scaler.transform(F_tr).astype(np.float32)
        F_val_s = scaler.transform(F_val).astype(np.float32)
        del F_full_tr, F_tr, F_val; gc.collect()

        best_local = {"acc": -1.0}
        for C in C_values:
            with Timer(log, f"LinearSVC C={C}"):
                clf = LinearSVC(C=C, dual="auto", max_iter=2000, tol=1e-4)
                clf.fit(F_tr_s, y_tr)
                pred = clf.predict(F_val_s)
                acc = accuracy_score(y_val, pred)
            log.info(f"  K={K} P={patch} C={C}  val={acc:.4f}")
            results_log.append(dict(K=K, patch=patch, C=float(C), val_acc=float(acc)))
            if acc > best_local["acc"]:
                best_local = dict(
                    acc=acc, C=C, K=K, patch=patch, stride=stride, pool=pool,
                    centroids=centroids, zca_mean=zca_mean, zca_W=zca_W,
                    scaler=scaler, clf=clf,
                )
        log.info(f"  [best] C={best_local['C']} acc={best_local['acc']:.4f}")

        if best_local["acc"] > best_global["acc"]:
            best_global = best_local

        del F_tr_s, F_val_s, scaler; gc.collect()

    # Save sweep log
    sweep_json = LOG_DIR / f"{RUN_NAME}_sweep.json"
    with open(sweep_json, "w") as f:
        json.dump(sorted(results_log, key=lambda r: -r["val_acc"]), f, indent=2)
    log.info("=== sweep complete ===")
    log.info("Top 10:")
    for r in sorted(results_log, key=lambda r: -r["val_acc"])[:10]:
        log.info(f"  K={r['K']} P={r['patch']} C={r['C']:.3f}  val={r['val_acc']:.4f}")

    # ---- Final: refit best on FULL train+val, predict test ----
    log.info(f">>> BEST: K={best_global['K']} P={best_global['patch']} "
             f"C={best_global['C']}  val={best_global['acc']:.4f}")

    K = best_global["K"]; patch = best_global["patch"]
    stride = best_global["stride"]; pool = best_global["pool"]; C = best_global["C"]

    F_full = get_or_encode(X, "train", K, patch,
                           best_global["centroids"], best_global["zca_mean"], best_global["zca_W"],
                           stride, pool, args.batch_size, log, tag=args.tag)
    F_te = get_or_encode(Xte, "test", K, patch,
                         best_global["centroids"], best_global["zca_mean"], best_global["zca_W"],
                         stride, pool, args.batch_size, log, tag=args.tag)

    scaler_full = StandardScaler().fit(F_full)
    F_full_s = scaler_full.transform(F_full).astype(np.float32)
    F_te_s = scaler_full.transform(F_te).astype(np.float32)

    with Timer(log, f"LinearSVC final fit on FULL (C={C})"):
        final_clf = LinearSVC(C=C, dual="auto", max_iter=3000, tol=1e-5)
        final_clf.fit(F_full_s, y)

    with Timer(log, "predict test"):
        preds = final_clf.predict(F_te_s)

    sub_path = SUB_DIR / f"sub_{RUN_NAME}.csv"
    save_submission(test_names, preds, sub_path)
    log.info(f"Wrote submission to {sub_path}")

    ckpt = CKPT_DIR / f"{RUN_NAME}.joblib"
    joblib.dump(
        {
            "best_config": {k: v for k, v in best_global.items()
                             if k in ("K", "patch", "stride", "pool", "C", "acc")},
            "sweep_results": results_log,
            "scaler": scaler_full,
            "model": final_clf,
            "centroids": best_global["centroids"],
            "zca_mean": best_global["zca_mean"],
            "zca_W": best_global["zca_W"],
        },
        ckpt,
    )
    log.info(f"Wrote checkpoint to {ckpt}")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
