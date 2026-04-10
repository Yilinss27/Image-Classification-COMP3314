"""Run 03 — Coates & Ng (2011) unsupervised feature learning.

Method (all components are covered in the COMP3314 lectures):
  1. Extract M random 6x6 patches from training images                (sampling)
  2. Per-patch contrast normalization (subtract mean, divide by std)  (L4: preprocessing)
  3. ZCA whitening over patches                                       (L4 / L5)
  4. MiniBatchKMeans with K centroids → visual dictionary             (L9: KMeans)
  5. Encode each image:
       - extract all stride-1 6x6 patches (27*27 per image)
       - normalize + whiten with saved statistics
       - triangle activation  f_k = max(0, mu(z) - z_k),  z_k = ||patch-c_k||
       - 2x2 spatial pooling (sum pool) → feature dim = 4*K           (L5-style pooling)
  6. LinearSVC / LogisticRegression on the 4K-dim features            (L3: SVM, LogReg)

References:
  Coates, Lee, Ng. "An analysis of single-layer networks in unsupervised feature
  learning." AISTATS 2011.

Expected wall time: 3-6 h on 20-core machine with K=800.
Expected accuracy: 0.70-0.80.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
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

RUN_NAME = "run_03_coates"

PATCH = 6     # patch side
STRIDE = 1    # encoding stride
POOL = 2      # pool grid = 2x2 (= 4 quadrants)
EPS_NORM = 10.0    # patch contrast normalization eps  (Coates: 10)
EPS_ZCA = 0.1      # ZCA regularization  (Coates: 0.1 for CIFAR)


# ---------------------------------------------------------------- patch utils

def extract_random_patches(X: np.ndarray, n_patches: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample n_patches random (PATCH x PATCH x 3) patches from images X (N,32,32,3)."""
    N, H, W, C = X.shape
    P = PATCH
    pats = np.empty((n_patches, P * P * C), dtype=np.float32)
    img_idx = rng.randint(0, N, size=n_patches)
    row = rng.randint(0, H - P + 1, size=n_patches)
    col = rng.randint(0, W - P + 1, size=n_patches)
    for i in range(n_patches):
        p = X[img_idx[i], row[i] : row[i] + P, col[i] : col[i] + P, :]
        pats[i] = p.astype(np.float32).reshape(-1)
    return pats


def extract_all_patches(img: np.ndarray, stride: int = 1) -> np.ndarray:
    """All stride-s patches from one image (H,W,C) → array (nPatches, P*P*C) float32."""
    H, W, C = img.shape
    P = PATCH
    nh = (H - P) // stride + 1
    nw = (W - P) // stride + 1
    n = nh * nw
    out = np.empty((n, P * P * C), dtype=np.float32)
    k = 0
    for r in range(0, H - P + 1, stride):
        for c in range(0, W - P + 1, stride):
            out[k] = img[r : r + P, c : c + P, :].astype(np.float32).reshape(-1)
            k += 1
    return out  # (nh*nw, P*P*C)


def extract_all_patches_batch(imgs: np.ndarray, stride: int = 1) -> np.ndarray:
    """Extract all patches for a batch of images → (N*nPatches, P*P*C)."""
    return np.concatenate([extract_all_patches(im, stride) for im in imgs], axis=0)


# ---------------------------------------------------------------- normalization

def contrast_normalize(patches: np.ndarray) -> np.ndarray:
    """Per-patch brightness/contrast normalization (Coates 2011)."""
    mean = patches.mean(axis=1, keepdims=True)
    var = patches.var(axis=1, keepdims=True)
    return (patches - mean) / np.sqrt(var + EPS_NORM)


def compute_zca(patches: np.ndarray):
    """Fit ZCA whitening on contrast-normalized patches. Returns (mean, W)."""
    mean = patches.mean(axis=0)
    Xc = patches - mean
    sigma = (Xc.T @ Xc) / Xc.shape[0]
    U, S, _ = np.linalg.svd(sigma)
    W = U @ np.diag(1.0 / np.sqrt(S + EPS_ZCA)) @ U.T
    return mean.astype(np.float32), W.astype(np.float32)


def apply_zca(patches: np.ndarray, mean: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (patches - mean) @ W


# ---------------------------------------------------------------- encoding

def encode_images(imgs: np.ndarray, centroids: np.ndarray,
                  zca_mean: np.ndarray, zca_W: np.ndarray,
                  batch_size: int = 500) -> np.ndarray:
    """Encode images into 4K-dim Coates features via triangle activation + 2x2 sum pool.

    imgs: (N, 32, 32, 3) uint8
    centroids: (K, P*P*C) whitened patch space
    Returns: (N, 4*K) float32
    """
    N = len(imgs)
    K = centroids.shape[0]
    P = PATCH
    H = W_ = 32
    nh = (H - P) // STRIDE + 1  # = 27
    nw = (W_ - P) // STRIDE + 1  # = 27
    feat_dim = POOL * POOL * K
    out = np.empty((N, feat_dim), dtype=np.float32)

    # Pre-compute ||centroid||^2 for fast distance
    c_sq = (centroids * centroids).sum(axis=1)  # (K,)

    for start in range(0, N, batch_size):
        stop = min(N, start + batch_size)
        batch = imgs[start:stop]
        B = len(batch)
        # Extract all patches: (B*nh*nw, P*P*C)
        patches = extract_all_patches_batch(batch, stride=STRIDE)
        patches = contrast_normalize(patches)
        patches = apply_zca(patches, zca_mean, zca_W)
        # Distances to each centroid using ||p||^2 + ||c||^2 - 2 p·c
        p_sq = (patches * patches).sum(axis=1, keepdims=True)   # (M,1)
        dots = patches @ centroids.T                            # (M,K)
        d2 = p_sq + c_sq[None, :] - 2.0 * dots                  # (M,K)
        d2 = np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2, out=d2)                                 # reuse buffer
        # Triangle activation: f_k = max(0, mu(z) - z_k)
        mu = d.mean(axis=1, keepdims=True)                      # (M,1)
        f = np.maximum(0.0, mu - d)                             # (M,K)

        # Reshape to (B, nh, nw, K) and pool by 2x2 quadrants
        f = f.reshape(B, nh, nw, K)
        half_h = nh // 2
        half_w = nw // 2
        # 4 quadrants: TL, TR, BL, BR  (sum pool)
        tl = f[:, :half_h,  :half_w, :].sum(axis=(1, 2))
        tr = f[:, :half_h,  half_w:, :].sum(axis=(1, 2))
        bl = f[:, half_h:,  :half_w, :].sum(axis=(1, 2))
        br = f[:, half_h:,  half_w:, :].sum(axis=(1, 2))
        out[start:stop] = np.concatenate([tl, tr, bl, br], axis=1).astype(np.float32)

        del patches, p_sq, dots, d2, d, f, tl, tr, bl, br
        gc.collect()

    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=800, help="number of KMeans centroids")
    ap.add_argument("--n-patches", type=int, default=400_000,
                    help="patches sampled for KMeans training")
    ap.add_argument("--batch-size", type=int, default=400,
                    help="images per encoding batch")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()

    log_file = LOG_DIR / f"{RUN_NAME}.log"
    log = get_logger(RUN_NAME, log_file)
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"args: {vars(args)}")

    with Timer(log, "load train"):
        X, y, _ = load_train_cached(n_jobs=args.n_jobs)
    with Timer(log, "load test"):
        Xte, test_names = load_test_cached(n_jobs=args.n_jobs)
    log.info(f"train X {X.shape}   test X {Xte.shape}")

    if args.smoke:
        X, y = X[:1000], y[:1000]
        Xte = Xte[:500]; test_names = test_names[:500]
        args.k = 80
        args.n_patches = 20_000
        args.batch_size = 200
        log.info("SMOKE MODE active")
        tag = "smoke"
    else:
        tag = "full"

    rng = np.random.RandomState(0)

    # ---- Step 1: sample random patches ----
    with Timer(log, f"sample {args.n_patches} random patches"):
        patches = extract_random_patches(X, args.n_patches, rng)
    log.info(f"random patches: {patches.shape} {patches.dtype}")

    # ---- Step 2: contrast normalize ----
    with Timer(log, "contrast normalize patches"):
        patches = contrast_normalize(patches)

    # ---- Step 3: ZCA fit ----
    with Timer(log, "fit ZCA"):
        zca_mean, zca_W = compute_zca(patches)
    with Timer(log, "apply ZCA to patches"):
        patches = apply_zca(patches, zca_mean, zca_W)
    log.info(f"whitened patches: {patches.shape}")

    # ---- Step 4: MiniBatchKMeans ----
    with Timer(log, f"MiniBatchKMeans (K={args.k})"):
        kmeans = MiniBatchKMeans(
            n_clusters=args.k,
            batch_size=4096,
            n_init=3,
            max_iter=300,
            random_state=0,
            verbose=0,
        )
        kmeans.fit(patches)
        centroids = kmeans.cluster_centers_.astype(np.float32)
    log.info(f"centroids: {centroids.shape}")
    del patches
    gc.collect()

    # Save dictionary artifacts immediately
    np.savez(CACHE_DIR / f"coates_dict_{tag}.npz",
             centroids=centroids, zca_mean=zca_mean, zca_W=zca_W)
    log.info(f"saved dictionary to coates_dict_{tag}.npz")

    # ---- Step 5: encode train and test ----
    feat_cache_tr = CACHE_DIR / f"coates_feat_train_{tag}_K{args.k}.npy"
    feat_cache_te = CACHE_DIR / f"coates_feat_test_{tag}_K{args.k}.npy"

    if feat_cache_tr.exists():
        F_tr_all = np.load(feat_cache_tr)
        log.info(f"loaded cached train features {F_tr_all.shape}")
    else:
        with Timer(log, f"encode train ({len(X)} images, batch={args.batch_size})"):
            F_tr_all = encode_images(X, centroids, zca_mean, zca_W, batch_size=args.batch_size)
        log.info(f"train features {F_tr_all.shape} {F_tr_all.dtype}")
        np.save(feat_cache_tr, F_tr_all)

    if feat_cache_te.exists():
        F_te = np.load(feat_cache_te)
        log.info(f"loaded cached test features {F_te.shape}")
    else:
        with Timer(log, f"encode test ({len(Xte)} images)"):
            F_te = encode_images(Xte, centroids, zca_mean, zca_W, batch_size=args.batch_size)
        log.info(f"test features {F_te.shape}")
        np.save(feat_cache_te, F_te)

    # ---- Step 6: standardize + train classifier ----
    F_tr, F_val, y_tr, y_val = train_test_split(
        F_tr_all, y, test_size=0.1, stratify=y, random_state=0
    )
    log.info(f"split train {F_tr.shape}  val {F_val.shape}")

    with Timer(log, "StandardScaler fit"):
        scaler = StandardScaler().fit(F_tr)
        F_tr_s = scaler.transform(F_tr).astype(np.float32)
        F_val_s = scaler.transform(F_val).astype(np.float32)
        F_te_s = scaler.transform(F_te).astype(np.float32)

    results: dict[str, tuple[float, object]] = {}

    # Try a few C values for LinearSVC
    for C in (0.01, 0.1, 1.0):
        with Timer(log, f"LinearSVC fit (C={C})"):
            clf = LinearSVC(C=C, dual="auto", max_iter=3000)
            clf.fit(F_tr_s, y_tr)
            acc = accuracy_score(y_val, clf.predict(F_val_s))
            log.info(f"LinearSVC C={C}  val acc = {acc:.4f}")
            results[f"lsvc_C{C}"] = (acc, clf)

    with Timer(log, "LogReg fit"):
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000)
        lr.fit(F_tr_s, y_tr)
        acc = accuracy_score(y_val, lr.predict(F_val_s))
        log.info(f"LogReg val acc = {acc:.4f}")
        results["logreg"] = (acc, lr)

    log.info("=== Validation summary ===")
    for name, (acc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        log.info(f"  {name:12s}  {acc:.4f}")

    # ---- Refit best on full train ----
    best_name = max(results, key=lambda k: results[k][0])
    log.info(f"Best on val: {best_name} ({results[best_name][0]:.4f})")

    with Timer(log, "refit scaler on full"):
        scaler_full = StandardScaler().fit(F_tr_all)
        F_all_s = scaler_full.transform(F_tr_all).astype(np.float32)
        F_te_full = scaler_full.transform(F_te).astype(np.float32)

    with Timer(log, f"refit {best_name} on full"):
        if best_name.startswith("lsvc"):
            C = float(best_name.split("C")[-1])
            model = LinearSVC(C=C, dual="auto", max_iter=3000)
        else:
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000)
        model.fit(F_all_s, y)

    with Timer(log, "predict test"):
        preds = model.predict(F_te_full)

    sub_path = SUB_DIR / f"sub_{RUN_NAME}.csv"
    save_submission(test_names, preds, sub_path)
    log.info(f"Wrote submission to {sub_path}")

    ckpt = CKPT_DIR / f"{RUN_NAME}.joblib"
    joblib.dump(
        {
            "scaler": scaler_full,
            "model": model,
            "best_name": best_name,
            "val_results": {k: v[0] for k, v in results.items()},
            "centroids": centroids,
            "zca_mean": zca_mean,
            "zca_W": zca_W,
            "K": args.k,
        },
        ckpt,
    )
    log.info(f"Wrote checkpoint to {ckpt}")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
