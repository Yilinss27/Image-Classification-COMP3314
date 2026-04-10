"""Run 04 — Coates-Ng full hyperparameter sweep (CPU, cache-heavy).

Systematically explores (K, patch_size, C) on the Coates-Ng pipeline from
run_03. All intermediate artefacts (dictionary, encoded features) are cached
per (K, patch) config so that:
  - Re-running the script resumes where it left off.
  - Trying new C values on an already-encoded config is ~30 seconds.
  - The existing run_03 cache (K=800, patch=6) is reused for free.

Lectures used: L3 (Linear SVM), L4 (preprocessing), L5 (dim reduction via ZCA),
L6 (validation, hyperparameter tuning), L9 (KMeans).

Expected wall time: ~60-120 min for a 5-config sweep.
Expected peak accuracy: 0.70-0.80.
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

RUN_NAME = "run_04_coates_sweep"

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


def extract_all_patches_img(img: np.ndarray, patch: int, stride: int) -> np.ndarray:
    H, W, C = img.shape
    nh = (H - patch) // stride + 1
    nw = (W - patch) // stride + 1
    out = np.empty((nh * nw, patch * patch * C), dtype=np.float32)
    k = 0
    for r in range(0, H - patch + 1, stride):
        for c in range(0, W - patch + 1, stride):
            out[k] = img[r:r+patch, c:c+patch, :].astype(np.float32).reshape(-1)
            k += 1
    return out


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


def encode_images(imgs: np.ndarray, centroids: np.ndarray,
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

    c_sq = (centroids * centroids).sum(axis=1)

    half_h = nh // pool
    half_w = nw // pool

    for start in range(0, N, batch_size):
        stop = min(N, start + batch_size)
        batch = imgs[start:stop]
        B = len(batch)
        patches = np.concatenate(
            [extract_all_patches_img(im, patch, stride) for im in batch], axis=0
        )
        patches = contrast_normalize(patches)
        patches = apply_zca(patches, zca_mean, zca_W)
        p_sq = (patches * patches).sum(axis=1, keepdims=True)
        dots = patches @ centroids.T
        d2 = p_sq + c_sq[None, :] - 2.0 * dots
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2)
        mu = d.mean(axis=1, keepdims=True)
        f = np.maximum(0.0, mu - d)

        f = f.reshape(B, nh, nw, K)
        parts = []
        for pi in range(pool):
            r_lo = pi * half_h
            r_hi = (pi + 1) * half_h if pi < pool - 1 else nh
            for pj in range(pool):
                c_lo = pj * half_w
                c_hi = (pj + 1) * half_w if pj < pool - 1 else nw
                parts.append(f[:, r_lo:r_hi, c_lo:c_hi, :].sum(axis=(1, 2)))
        out[start:stop] = np.concatenate(parts, axis=1).astype(np.float32)
        del patches, p_sq, dots, d2, d, f
        if start % (batch_size * 20) == 0:
            gc.collect()

    return out


# ---------------------------------------------------------------- caching

def dict_cache_path(K: int, patch: int, n_patches: int) -> Path:
    return CACHE_DIR / f"sweep_dict_K{K}_P{patch}_N{n_patches}.npz"


def feat_cache_path(split: str, K: int, patch: int) -> Path:
    return CACHE_DIR / f"sweep_feat_{split}_K{K}_P{patch}.npy"


def get_or_fit_dict(X_tr: np.ndarray, K: int, patch: int, n_patches: int,
                    log, rng: np.random.RandomState):
    """Fit (or load) patch dictionary for given (K, patch)."""
    p = dict_cache_path(K, patch, n_patches)
    # Also reuse run_03's existing cache if available (patch=6, K matches)
    if not p.exists() and K == 800 and patch == 6:
        legacy = CACHE_DIR / "coates_dict_full.npz"
        if legacy.exists():
            log.info(f"reusing legacy dictionary {legacy.name}")
            d = np.load(legacy)
            return d["centroids"], d["zca_mean"], d["zca_W"]
    if p.exists():
        log.info(f"loaded dictionary {p.name}")
        d = np.load(p)
        return d["centroids"], d["zca_mean"], d["zca_W"]

    with Timer(log, f"sample {n_patches} random patches (patch={patch})"):
        P = extract_random_patches(X_tr, n_patches, patch, rng)
    with Timer(log, "contrast normalize patches"):
        P = contrast_normalize(P)
    with Timer(log, "fit ZCA"):
        zca_mean, zca_W = compute_zca(P)
    with Timer(log, "apply ZCA"):
        P = apply_zca(P, zca_mean, zca_W)
    with Timer(log, f"MiniBatchKMeans (K={K})"):
        km = MiniBatchKMeans(
            n_clusters=K, batch_size=4096, n_init=3, max_iter=300,
            random_state=0, verbose=0,
        )
        km.fit(P)
        centroids = km.cluster_centers_.astype(np.float32)
    del P; gc.collect()

    np.savez(p, centroids=centroids, zca_mean=zca_mean, zca_W=zca_W)
    log.info(f"saved dictionary {p.name}")
    return centroids, zca_mean, zca_W


def get_or_encode(X: np.ndarray, split: str, K: int, patch: int,
                  centroids, zca_mean, zca_W, stride: int, pool: int,
                  batch_size: int, log) -> np.ndarray:
    p = feat_cache_path(split, K, patch)
    # Reuse legacy run_03 cache for train/test at K=800 patch=6
    if not p.exists() and K == 800 and patch == 6 and split in ("train", "test"):
        legacy = CACHE_DIR / f"coates_feat_{split}_full_K{K}.npy"
        if legacy.exists():
            log.info(f"reusing legacy features {legacy.name}")
            return np.load(legacy)
    if p.exists():
        log.info(f"loaded features {p.name}")
        return np.load(p)
    with Timer(log, f"encode {split} ({len(X)} imgs, K={K}, patch={patch})"):
        F = encode_images(X, centroids, zca_mean, zca_W,
                          patch=patch, stride=stride, pool=pool,
                          batch_size=batch_size)
    np.save(p, F)
    log.info(f"saved features {p.name} shape {F.shape}")
    return F


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-patches", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--configs", default="default",
                    help="'default' or 'quick' (2 configs)")
    args = ap.parse_args()

    log_file = LOG_DIR / f"{RUN_NAME}.log"
    log = get_logger(RUN_NAME, log_file)
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"args: {vars(args)}")

    with Timer(log, "load train/test"):
        X, y, _ = load_train_cached(n_jobs=args.n_jobs)
        Xte, test_names = load_test_cached(n_jobs=args.n_jobs)

    # Fixed val split (SAME as run_03 for fair comparison)
    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0
    )
    y_tr = y[idx_tr]; y_val = y[idx_val]
    log.info(f"train sub {len(idx_tr)}  val {len(idx_val)}  test {len(Xte)}")

    # Configs to sweep
    if args.smoke:
        sweep = [dict(K=80, patch=6, stride=1, pool=2)]
        C_values = [0.01, 0.1]
    elif args.configs == "quick":
        sweep = [
            dict(K=400,  patch=6, stride=1, pool=2),
            dict(K=800,  patch=6, stride=1, pool=2),
        ]
        C_values = [0.003, 0.01, 0.03, 0.1]
    else:
        sweep = [
            dict(K=400,  patch=6, stride=1, pool=2),
            dict(K=800,  patch=6, stride=1, pool=2),
            dict(K=1600, patch=6, stride=1, pool=2),
            dict(K=800,  patch=8, stride=1, pool=2),
            dict(K=1600, patch=8, stride=1, pool=2),
        ]
        C_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

    log.info(f"sweep: {len(sweep)} configs x {len(C_values)} C values = "
             f"{len(sweep)*len(C_values)} experiments")

    if args.smoke:
        X = X[:1500]; y = y[:1500]
        Xte = Xte[:500]; test_names = test_names[:500]
        args.n_patches = 20_000
        idx_tr, idx_val = train_test_split(
            np.arange(len(X)), test_size=0.1, stratify=y, random_state=0
        )
        y_tr = y[idx_tr]; y_val = y[idx_val]

    results_log = []
    best_global = {"acc": -1.0}

    rng = np.random.RandomState(0)

    for ci, cfg in enumerate(sweep):
        log.info(f"--- config {ci+1}/{len(sweep)}: {cfg} ---")
        K = cfg["K"]; patch = cfg["patch"]; stride = cfg["stride"]; pool = cfg["pool"]

        centroids, zca_mean, zca_W = get_or_fit_dict(
            X[idx_tr] if not args.smoke else X,
            K=K, patch=patch, n_patches=args.n_patches, log=log, rng=rng
        )

        # Encode full train set once (since split is fixed, we slice later)
        F_full_tr = get_or_encode(X, "train", K, patch, centroids, zca_mean, zca_W,
                                  stride, pool, args.batch_size, log)
        F_test = get_or_encode(Xte, "test", K, patch, centroids, zca_mean, zca_W,
                               stride, pool, args.batch_size, log)

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
            log.info(f"  K={K} patch={patch} C={C}  val={acc:.4f}")
            results_log.append(dict(K=K, patch=patch, C=C, val_acc=float(acc)))
            if acc > best_local["acc"]:
                best_local = dict(
                    acc=acc, C=C, K=K, patch=patch, stride=stride, pool=pool,
                    centroids=centroids, zca_mean=zca_mean, zca_W=zca_W,
                    scaler=scaler, clf=clf,
                )

        log.info(f"  [best for cfg] C={best_local['C']} acc={best_local['acc']:.4f}")
        if best_local["acc"] > best_global["acc"]:
            best_global = best_local

        del F_tr_s, F_val_s, scaler; gc.collect()

    # Save sweep log
    sweep_json = LOG_DIR / f"{RUN_NAME}_sweep.json"
    with open(sweep_json, "w") as f:
        json.dump(sorted(results_log, key=lambda r: -r["val_acc"]), f, indent=2)
    log.info("=== sweep complete ===")
    log.info("Top 10 configs:")
    for r in sorted(results_log, key=lambda r: -r["val_acc"])[:10]:
        log.info(f"  K={r['K']} patch={r['patch']} C={r['C']:.3f}  val={r['val_acc']:.4f}")

    # ---- Refit best on FULL train+val and predict test ----
    log.info(f">>> BEST: K={best_global['K']} patch={best_global['patch']} "
             f"C={best_global['C']}  val={best_global['acc']:.4f}")

    K = best_global["K"]; patch = best_global["patch"]
    stride = best_global["stride"]; pool = best_global["pool"]
    C = best_global["C"]

    # Load full encoded train features for this (K, patch) — use cache
    F_full = get_or_encode(X, "train", K, patch,
                           best_global["centroids"], best_global["zca_mean"], best_global["zca_W"],
                           stride, pool, args.batch_size, log)
    F_te = get_or_encode(Xte, "test", K, patch,
                         best_global["centroids"], best_global["zca_mean"], best_global["zca_W"],
                         stride, pool, args.batch_size, log)

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
