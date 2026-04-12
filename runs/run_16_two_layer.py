"""Run 16: Two-layer Coates-Ng unsupervised feature learning + flip aug.

Layer 1: K1=1600, P1=6 → spatial map (27,27,1600) → 2×2 pool → 6400-dim
Intermediate: 3×3 sum-pool on spatial → (9,9,1600)
Layer 2: P2=3 patches from (9,9,1600) → dim=14400 → ZCA → K2=1600
         → triangle encode → (7,7,1600) → 2×2 pool → 6400-dim
Final: concat L1+L2 = 12800-dim → StandardScaler → cuML LinearSVC

Everything streams through in batches (spatial maps never fully materialized).
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (LOG_DIR, SUB_DIR, CACHE_DIR, Timer,
                  get_logger, load_test_cached, load_train_cached, save_submission)
from run_07_cuml_sweep import (extract_all_patches_batch, contrast_normalize_np,
                                compute_zca, apply_zca_np, EPS_NORM,
                                get_or_fit_dict)
from cuml.svm import LinearSVC
import cupy as cp

RUN_NAME = "run_16_two_layer"
P1, K1 = 6, 1600
P2, K2 = 3, 1600
INTER_POOL = 3
POOL = 2
STRIDE = 1
N_PATCHES_L1 = 1_000_000
N_PATCHES_L2 = 200_000
BATCH_SIZE = 200
C = 0.002


def flip_horizontal(X):
    return np.ascontiguousarray(X[:, :, ::-1, :])


# ---- Layer-1 spatial encoding (GPU, no final pool) ----

def encode_batch_spatial_gpu(batch, centroids_gpu, c_sq, zca_mean_gpu, zca_W_gpu,
                             patch, stride):
    """Encode one batch to spatial maps (B, nh, nw, K) on GPU. Returns cupy array."""
    B = len(batch)
    H = batch.shape[1]
    nh = (H - patch) // stride + 1

    patches_np = extract_all_patches_batch(batch, patch, stride)
    patches = cp.asarray(patches_np, dtype=cp.float32)
    mean = patches.mean(axis=1, keepdims=True)
    var = patches.var(axis=1, keepdims=True)
    patches = (patches - mean) / cp.sqrt(var + EPS_NORM)
    patches = (patches - zca_mean_gpu) @ zca_W_gpu

    p_sq = (patches * patches).sum(axis=1, keepdims=True)
    dots = patches @ centroids_gpu.T
    d2 = p_sq + c_sq[None, :] - 2.0 * dots
    d2 = cp.maximum(d2, 0.0)
    d = cp.sqrt(d2)
    mu = d.mean(axis=1, keepdims=True)
    f = cp.maximum(0.0, mu - d)  # (B*nh*nw, K)

    f = f.reshape(B, nh, nh, -1)  # (B, nh, nw, K)
    return f


def spatial_sum_pool(maps_gpu, pool_size):
    """Sum-pool (B, H, W, K) → (B, H//pool, W//pool, K) on GPU."""
    B, H, W, K = maps_gpu.shape
    oh = H // pool_size
    ow = W // pool_size
    out = cp.zeros((B, oh, ow, K), dtype=cp.float32)
    for i in range(pool_size):
        for j in range(pool_size):
            out += maps_gpu[:, i::pool_size, j::pool_size, :][:, :oh, :ow, :]
    return out


def quadrant_pool(maps_gpu, pool=2):
    """2×2 quadrant sum pool (B, H, W, K) → (B, pool*pool*K) on CPU numpy."""
    B, H, W, K = maps_gpu.shape
    half_h = H // pool
    half_w = W // pool
    parts = []
    for pi in range(pool):
        r_lo = pi * half_h
        r_hi = (pi + 1) * half_h if pi < pool - 1 else H
        for pj in range(pool):
            c_lo = pj * half_w
            c_hi = (pj + 1) * half_w if pj < pool - 1 else W
            parts.append(maps_gpu[:, r_lo:r_hi, c_lo:c_hi, :].sum(axis=(1, 2)))
    return cp.concatenate(parts, axis=1)  # (B, pool*pool*K)


# ---- Layer-2 patch extraction from feature maps ----

def extract_patches_from_maps_gpu(maps_gpu, p2):
    """Extract all P2×P2 patches from (B, sh, sw, K) feature maps on GPU.
    Returns (B*oh*ow, P2*P2*K) cupy array.
    """
    B, sh, sw, K = maps_gpu.shape
    oh = sh - p2 + 1
    ow = sw - p2 + 1
    patches = cp.empty((B, oh, ow, p2, p2, K), dtype=cp.float32)
    for i in range(p2):
        for j in range(p2):
            patches[:, :, :, i, j, :] = maps_gpu[:, i:i+oh, j:j+ow, :]
    return patches.reshape(B * oh * ow, p2 * p2 * K)


# ---- Collect Layer-2 random patches (streaming) ----

def collect_l2_patches(X, L1_centroids, L1_zca_mean, L1_zca_W,
                       n_patches, log, rng):
    """Stream through images, produce L1 spatial maps, intermediate pool,
    extract random patches for L2 dictionary learning."""
    c_gpu = cp.asarray(L1_centroids, dtype=cp.float32)
    c_sq = (c_gpu * c_gpu).sum(axis=1)
    zm_gpu = cp.asarray(L1_zca_mean, dtype=cp.float32)
    zW_gpu = cp.asarray(L1_zca_W, dtype=cp.float32)

    all_patches = []
    n_collected = 0
    mempool = cp.get_default_memory_pool()

    for start in range(0, len(X), BATCH_SIZE):
        if n_collected >= n_patches:
            break
        batch = X[start:start + BATCH_SIZE]
        spatial = encode_batch_spatial_gpu(batch, c_gpu, c_sq, zm_gpu, zW_gpu,
                                           P1, STRIDE)
        inter = spatial_sum_pool(spatial, INTER_POOL)  # (B, 9, 9, K1)
        del spatial

        l2_patches = extract_patches_from_maps_gpu(inter, P2)  # (B*49, 14400)
        l2_np = cp.asnumpy(l2_patches)
        del inter, l2_patches
        mempool.free_all_blocks()

        n_take = min(n_patches - n_collected, len(l2_np))
        idx = rng.choice(len(l2_np), n_take, replace=False)
        all_patches.append(l2_np[idx])
        n_collected += n_take

        if start % (BATCH_SIZE * 20) == 0:
            log.info(f"    collected {n_collected}/{n_patches} L2 patches")

    del c_gpu, c_sq, zm_gpu, zW_gpu
    mempool.free_all_blocks()
    return np.concatenate(all_patches, axis=0)[:n_patches]


# ---- Fit Layer-2 dictionary ----

def fit_l2_dict(X_tr, L1_centroids, L1_zca_mean, L1_zca_W, log, rng):
    cache = CACHE_DIR / f"l2_dict_K1{K1}_K2{K2}_P2{P2}_ip{INTER_POOL}.npz"
    if cache.exists():
        d = np.load(cache)
        log.info(f"loaded L2 dict {cache.name}")
        return d["centroids"], d["zca_mean"], d["zca_W"]

    with Timer(log, f"collect {N_PATCHES_L2} L2 patches"):
        patches = collect_l2_patches(X_tr, L1_centroids, L1_zca_mean, L1_zca_W,
                                     N_PATCHES_L2, log, rng)
    log.info(f"  L2 patches shape: {patches.shape}")

    with Timer(log, "L2 contrast normalize"):
        mean = patches.mean(axis=1, keepdims=True)
        std = np.sqrt(patches.var(axis=1, keepdims=True) + EPS_NORM)
        patches = (patches - mean) / std

    with Timer(log, "L2 fit ZCA"):
        l2_zca_mean, l2_zca_W = compute_zca(patches)

    with Timer(log, "L2 apply ZCA"):
        patches = apply_zca_np(patches, l2_zca_mean, l2_zca_W)

    with Timer(log, f"L2 MiniBatchKMeans K2={K2}"):
        km = MiniBatchKMeans(n_clusters=K2, batch_size=4096, n_init=1,
                             max_iter=100, random_state=0, verbose=0)
        km.fit(patches)
        l2_centroids = km.cluster_centers_.astype(np.float32)

    del patches
    gc.collect()
    np.savez(cache, centroids=l2_centroids, zca_mean=l2_zca_mean, zca_W=l2_zca_W)
    log.info(f"saved L2 dict {cache.name}")
    return l2_centroids, l2_zca_mean, l2_zca_W


# ---- Full two-layer encoding ----

def encode_two_layer_gpu(imgs, L1_centroids, L1_zca_mean, L1_zca_W,
                         L2_centroids, L2_zca_mean, L2_zca_W,
                         batch_size=BATCH_SIZE):
    N = len(imgs)
    feat_dim = POOL * POOL * K1 + POOL * POOL * K2
    out = np.empty((N, feat_dim), dtype=np.float32)

    c1_gpu = cp.asarray(L1_centroids, dtype=cp.float32)
    c1_sq = (c1_gpu * c1_gpu).sum(axis=1)
    zm1_gpu = cp.asarray(L1_zca_mean, dtype=cp.float32)
    zW1_gpu = cp.asarray(L1_zca_W, dtype=cp.float32)

    c2_gpu = cp.asarray(L2_centroids, dtype=cp.float32)
    c2_sq = (c2_gpu * c2_gpu).sum(axis=1)
    zm2_gpu = cp.asarray(L2_zca_mean, dtype=cp.float32)
    zW2_gpu = cp.asarray(L2_zca_W, dtype=cp.float32)

    mempool = cp.get_default_memory_pool()

    for start in range(0, N, batch_size):
        batch = imgs[start:start + batch_size]
        B = len(batch)

        # Layer 1: spatial maps
        spatial = encode_batch_spatial_gpu(batch, c1_gpu, c1_sq, zm1_gpu, zW1_gpu,
                                           P1, STRIDE)  # (B, 27, 27, K1)
        l1_pooled = quadrant_pool(spatial, POOL)  # (B, 4*K1)

        # Intermediate pool for Layer 2
        inter = spatial_sum_pool(spatial, INTER_POOL)  # (B, 9, 9, K1)
        del spatial

        # Layer 2: extract patches, encode
        l2_patches = extract_patches_from_maps_gpu(inter, P2)  # (B*49, 14400)
        del inter

        # Contrast norm on GPU
        pmean = l2_patches.mean(axis=1, keepdims=True)
        pvar = l2_patches.var(axis=1, keepdims=True)
        l2_patches = (l2_patches - pmean) / cp.sqrt(pvar + EPS_NORM)
        # ZCA
        l2_patches = (l2_patches - zm2_gpu) @ zW2_gpu

        # Triangle encode
        p_sq = (l2_patches * l2_patches).sum(axis=1, keepdims=True)
        dots = l2_patches @ c2_gpu.T
        d2 = p_sq + c2_sq[None, :] - 2.0 * dots
        d2 = cp.maximum(d2, 0.0)
        d = cp.sqrt(d2)
        mu = d.mean(axis=1, keepdims=True)
        f2 = cp.maximum(0.0, mu - d)  # (B*oh*ow, K2)

        oh = 9 - P2 + 1  # 7
        ow = oh
        f2 = f2.reshape(B, oh, ow, K2)
        l2_pooled = quadrant_pool(f2, POOL)  # (B, 4*K2)

        combined = cp.concatenate([l1_pooled, l2_pooled], axis=1)
        out[start:start + B] = cp.asnumpy(combined)

        del l2_patches, p_sq, dots, d2, d, mu, f2, l1_pooled, l2_pooled, combined
        mempool.free_all_blocks()

    del c1_gpu, c1_sq, zm1_gpu, zW1_gpu, c2_gpu, c2_sq, zm2_gpu, zW2_gpu
    mempool.free_all_blocks()
    return out


def main():
    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"L1: P={P1} K={K1} | L2: P2={P2} K2={K2} inter_pool={INTER_POOL}")
    log.info(f"final dim = {POOL**2 * K1 + POOL**2 * K2}")

    with Timer(log, "load data"):
        X, y, _ = load_train_cached(n_jobs=8)
        Xte, test_names = load_test_cached(n_jobs=8)

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0)
    y_tr, y_val = y[idx_tr], y[idx_val]
    rng = np.random.RandomState(0)

    # ---- Layer-1 dictionary ----
    with Timer(log, "Layer-1 dict"):
        L1_c, L1_zm, L1_zW = get_or_fit_dict(X[idx_tr], K1, P1, N_PATCHES_L1, log, rng)
    log.info(f"  L1 centroids: {L1_c.shape}")

    # ---- Layer-2 dictionary ----
    with Timer(log, "Layer-2 dict"):
        L2_c, L2_zm, L2_zW = fit_l2_dict(X[idx_tr], L1_c, L1_zm, L1_zW, log, rng)
    log.info(f"  L2 centroids: {L2_c.shape}, ZCA: {L2_zW.shape}")

    # ---- Flip-aug train ----
    X_tr = X[idx_tr]
    X_val = X[idx_val]
    X_tr_aug = np.concatenate([X_tr, flip_horizontal(X_tr)], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_tr])
    log.info(f"flip-aug train: {X_tr_aug.shape}")

    with Timer(log, "encode two-layer aug-train (90k)"):
        F_tr = encode_two_layer_gpu(X_tr_aug, L1_c, L1_zm, L1_zW,
                                    L2_c, L2_zm, L2_zW)
    with Timer(log, "encode two-layer val (5k)"):
        F_val = encode_two_layer_gpu(X_val, L1_c, L1_zm, L1_zW,
                                     L2_c, L2_zm, L2_zW)
    log.info(f"  F_tr {F_tr.shape}  F_val {F_val.shape}")

    scaler = StandardScaler().fit(F_tr)
    F_tr_s = scaler.transform(F_tr).astype(np.float32)
    F_val_s = scaler.transform(F_val).astype(np.float32)
    del F_tr, F_val
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    t0 = time.time()
    clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                    max_iter=5000, tol=1e-4)
    clf.fit(F_tr_s, y_tr_aug)
    preds = clf.predict(F_val_s)
    if hasattr(preds, "get"):
        preds = preds.get()
    val_acc = float(accuracy_score(y_val, preds))
    log.info(f"  two-layer val={val_acc:.4f}  (fit {time.time()-t0:.1f}s)")

    del F_tr_s, F_val_s, scaler
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # ---- Full refit + TTA ----
    log.info(f"\n--- Full refit C={C} ---")
    X_full_aug = np.concatenate([X, flip_horizontal(X)], axis=0)
    y_full_aug = np.concatenate([y, y])

    with Timer(log, "encode two-layer full-aug (100k)"):
        F_full = encode_two_layer_gpu(X_full_aug, L1_c, L1_zm, L1_zW,
                                      L2_c, L2_zm, L2_zW)
    del X_full_aug
    gc.collect()

    scaler_full = StandardScaler().fit(F_full)
    F_full_s = scaler_full.transform(F_full).astype(np.float32)
    del F_full
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    t0 = time.time()
    final_clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                          max_iter=5000, tol=1e-5)
    final_clf.fit(F_full_s, y_full_aug)
    log.info(f"  full-refit fit {time.time()-t0:.1f}s")
    del F_full_s
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    # TTA: test orig + test flip
    with Timer(log, "encode two-layer test (10k)"):
        F_te = encode_two_layer_gpu(Xte, L1_c, L1_zm, L1_zW,
                                    L2_c, L2_zm, L2_zW)
    with Timer(log, "encode two-layer test-flip (10k)"):
        F_te_flip = encode_two_layer_gpu(flip_horizontal(Xte), L1_c, L1_zm, L1_zW,
                                         L2_c, L2_zm, L2_zW)
    F_te_s = scaler_full.transform(F_te).astype(np.float32)
    F_te_flip_s = scaler_full.transform(F_te_flip).astype(np.float32)
    del F_te, F_te_flip
    gc.collect()

    df_orig = final_clf.decision_function(F_te_s)
    df_flip = final_clf.decision_function(F_te_flip_s)
    if hasattr(df_orig, "get"): df_orig = df_orig.get()
    if hasattr(df_flip, "get"): df_flip = df_flip.get()

    preds_no_tta = df_orig.argmax(axis=1)
    preds_tta = (df_orig + df_flip).argmax(axis=1)

    sub1 = SUB_DIR / f"sub_run16_2layer_P{P1}_K{K1}_{K2}_C{C}.csv"
    sub2 = SUB_DIR / f"sub_run16_2layer_tta_P{P1}_K{K1}_{K2}_C{C}.csv"
    save_submission(test_names, preds_no_tta, sub1)
    save_submission(test_names, preds_tta, sub2)
    log.info(f"  wrote {sub1.name}")
    log.info(f"  wrote {sub2.name}")

    diffs = int((preds_no_tta != preds_tta).sum())
    log.info(f"  TTA changed {diffs}/{len(preds_no_tta)} preds ({100*diffs/len(preds_no_tta):.2f}%)")

    log.info(f"\n=== {RUN_NAME} done ===")
    log.info(f"two-layer val={val_acc:.4f} (baseline single-layer flip: 0.8122)")


if __name__ == "__main__":
    main()
