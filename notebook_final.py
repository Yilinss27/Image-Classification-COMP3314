# %% [markdown]
# # COMP3314 Assignment 3 — Image Classification
#
# **Final Solution: Coates-Ng Single-Layer Unsupervised Feature Learning**
#
# Pipeline: Random patches → Contrast normalization → ZCA whitening →
# KMeans dictionary → Triangle encoding → 2×2 sum pooling →
# Power normalization → Horizontal flip augmentation → Linear SVM ensemble
#
# Two-model ensemble: P=6 K=8000 C=0.002 + P=7 K=6000 C=0.002
# with flip TTA (test-time augmentation)

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
import gc
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# cuML GPU LinearSVC and cupy for GPU-accelerated encoding
from cuml.svm import LinearSVC
import cupy as cp

# Paths — adjust DATA_DIR if your data is elsewhere
ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

print(f"Root: {ROOT}")
print(f"Data: {DATA_DIR}")

# %%
# Load training data
def load_images(img_dir, filenames):
    imgs = []
    for fn in filenames:
        img = Image.open(img_dir / fn)
        imgs.append(np.array(img))
    return np.stack(imgs)

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train: {len(train_df)} images, {train_df['label'].nunique()} classes")
print(f"Test:  {len(test_df)} images")

X_train = load_images(DATA_DIR / "train_ims", train_df["im_name"].values)
y_train = train_df["label"].values
X_test = load_images(DATA_DIR / "test_ims", test_df["im_name"].values)
test_names = test_df["im_name"].values

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}")

# %% [markdown]
# ## 2. Helper Functions
#
# Core functions for the Coates-Ng (2011) unsupervised feature learning pipeline.

# %%
EPS_NORM = 10.0  # contrast normalization epsilon
EPS_ZCA = 0.1    # ZCA whitening epsilon

def extract_random_patches(images, n_patches, patch_size, rng):
    """Extract random patches from a set of images."""
    N, H, W, C = images.shape
    patches = np.empty((n_patches, patch_size * patch_size * C), dtype=np.float32)
    for i in range(n_patches):
        idx = rng.randint(N)
        r = rng.randint(H - patch_size + 1)
        c = rng.randint(W - patch_size + 1)
        patches[i] = images[idx, r:r+patch_size, c:c+patch_size, :].ravel().astype(np.float32)
    return patches

def extract_all_patches_batch(images, patch_size, stride):
    """Extract all patches from a batch of images. Returns (N*nh*nw, patch_dim)."""
    N, H, W, C = images.shape
    nh = (H - patch_size) // stride + 1
    nw = (W - patch_size) // stride + 1
    patches = np.empty((N * nh * nw, patch_size * patch_size * C), dtype=np.float32)
    idx = 0
    for r in range(nh):
        for c in range(nw):
            p = images[:, r*stride:r*stride+patch_size, c*stride:c*stride+patch_size, :]
            patches[idx:idx+N] = p.reshape(N, -1).astype(np.float32)
            idx += N
    # Reorder so patches from same image are contiguous
    patches = patches.reshape(nh * nw, N, -1).transpose(1, 0, 2).reshape(N * nh * nw, -1)
    return patches

def compute_zca(X, eps=EPS_ZCA):
    """Compute ZCA whitening parameters from data matrix X."""
    mean = X.mean(axis=0)
    X_c = X - mean
    cov = X_c.T @ X_c / len(X_c)
    U, S, Vt = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    return mean.astype(np.float32), W.astype(np.float32)

def apply_zca(X, mean, W):
    return ((X - mean) @ W).astype(np.float32)

def flip_horizontal(X):
    """Horizontally flip images."""
    return np.ascontiguousarray(X[:, :, ::-1, :])

def power_norm(X):
    """Signed square-root power normalization."""
    return np.sign(X) * np.sqrt(np.abs(X))

# %%
def encode_images_gpu(imgs, centroids, zca_mean, zca_W,
                      patch, stride, pool, batch_size):
    """GPU-accelerated Coates-Ng triangle encoding with sum pooling."""
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

# %% [markdown]
# ## 3. Dictionary Learning
#
# Learn KMeans dictionaries for both models. Cached to disk for speed.

# %%
def get_or_fit_dict(X_train, K, patch_size, n_patches, rng):
    """Fit or load cached KMeans dictionary."""
    cache_path = CACHE_DIR / f"dict_K{K}_P{patch_size}_N{n_patches}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        print(f"  Loaded cached dict: {cache_path.name}")
        return d["centroids"], d["zca_mean"], d["zca_W"]

    print(f"  Fitting dict K={K} P={patch_size} ({n_patches} patches)...")
    t0 = time.time()
    patches = extract_random_patches(X_train, n_patches, patch_size, rng)
    # Contrast normalize
    mean = patches.mean(axis=1, keepdims=True)
    std = np.sqrt(patches.var(axis=1, keepdims=True) + EPS_NORM)
    patches = (patches - mean) / std
    # ZCA
    zca_mean, zca_W = compute_zca(patches)
    patches = apply_zca(patches, zca_mean, zca_W)
    # KMeans
    km = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=3,
                         max_iter=300, random_state=0, verbose=0)
    km.fit(patches)
    centroids = km.cluster_centers_.astype(np.float32)
    del patches; gc.collect()
    np.savez(cache_path, centroids=centroids, zca_mean=zca_mean, zca_W=zca_W)
    print(f"  Dict fitted in {time.time()-t0:.1f}s, saved to {cache_path.name}")
    return centroids, zca_mean, zca_W

# %%
N_PATCHES = 1_000_000
rng = np.random.RandomState(0)

print("=== Model 1: P=6 K=8000 ===")
centroids_1, zca_mean_1, zca_W_1 = get_or_fit_dict(X_train, 8000, 6, N_PATCHES, rng)
print(f"  Centroids: {centroids_1.shape}")

print("\n=== Model 2: P=7 K=6000 ===")
centroids_2, zca_mean_2, zca_W_2 = get_or_fit_dict(X_train, 6000, 7, N_PATCHES, rng)
print(f"  Centroids: {centroids_2.shape}")

# %% [markdown]
# ## 4. Feature Encoding with Flip Augmentation
#
# For each model:
# - Encode original 50k + horizontally flipped 50k = 100k training samples
# - Apply power normalization (signed sqrt)
# - StandardScaler
# - Fit cuML LinearSVC

# %%
X_train_flip = flip_horizontal(X_train)
X_full_aug = np.concatenate([X_train, X_train_flip], axis=0)
y_full_aug = np.concatenate([y_train, y_train])
X_test_flip = flip_horizontal(X_test)
print(f"Augmented train: {X_full_aug.shape}, labels: {y_full_aug.shape}")

# %%
# Model 1: P=6 K=8000 C=0.002
print("=== Encoding Model 1 (P=6 K=8000) ===")
t0 = time.time()
F_train_1 = encode_images_gpu(X_full_aug, centroids_1, zca_mean_1, zca_W_1,
                               patch=6, stride=1, pool=2, batch_size=128)
print(f"  Train features: {F_train_1.shape} ({time.time()-t0:.1f}s)")

F_train_1 = power_norm(F_train_1)
scaler_1 = StandardScaler().fit(F_train_1)
F_train_1_s = scaler_1.transform(F_train_1).astype(np.float32)
del F_train_1; gc.collect(); cp.get_default_memory_pool().free_all_blocks()

t0 = time.time()
clf_1 = LinearSVC(C=0.002, loss="squared_hinge", penalty="l2", max_iter=5000, tol=1e-5)
clf_1.fit(F_train_1_s, y_full_aug)
print(f"  SVM fit: {time.time()-t0:.1f}s")
del F_train_1_s; gc.collect(); cp.get_default_memory_pool().free_all_blocks()

# Test encoding
F_test_1 = encode_images_gpu(X_test, centroids_1, zca_mean_1, zca_W_1,
                              patch=6, stride=1, pool=2, batch_size=128)
F_test_flip_1 = encode_images_gpu(X_test_flip, centroids_1, zca_mean_1, zca_W_1,
                                   patch=6, stride=1, pool=2, batch_size=128)
F_test_1_s = scaler_1.transform(power_norm(F_test_1)).astype(np.float32)
F_test_flip_1_s = scaler_1.transform(power_norm(F_test_flip_1)).astype(np.float32)
del F_test_1, F_test_flip_1; gc.collect()

df_orig_1 = clf_1.decision_function(F_test_1_s)
df_flip_1 = clf_1.decision_function(F_test_flip_1_s)
if hasattr(df_orig_1, "get"): df_orig_1 = df_orig_1.get()
if hasattr(df_flip_1, "get"): df_flip_1 = df_flip_1.get()
print(f"  Decision functions: {df_orig_1.shape}")

del F_test_1_s, F_test_flip_1_s, clf_1, scaler_1
gc.collect(); cp.get_default_memory_pool().free_all_blocks()

# %%
# Model 2: P=7 K=6000 C=0.002
print("=== Encoding Model 2 (P=7 K=6000) ===")
t0 = time.time()
F_train_2 = encode_images_gpu(X_full_aug, centroids_2, zca_mean_2, zca_W_2,
                               patch=7, stride=1, pool=2, batch_size=128)
print(f"  Train features: {F_train_2.shape} ({time.time()-t0:.1f}s)")

F_train_2 = power_norm(F_train_2)
scaler_2 = StandardScaler().fit(F_train_2)
F_train_2_s = scaler_2.transform(F_train_2).astype(np.float32)
del F_train_2; gc.collect(); cp.get_default_memory_pool().free_all_blocks()

t0 = time.time()
clf_2 = LinearSVC(C=0.002, loss="squared_hinge", penalty="l2", max_iter=5000, tol=1e-5)
clf_2.fit(F_train_2_s, y_full_aug)
print(f"  SVM fit: {time.time()-t0:.1f}s")
del F_train_2_s; gc.collect(); cp.get_default_memory_pool().free_all_blocks()

# Test encoding
F_test_2 = encode_images_gpu(X_test, centroids_2, zca_mean_2, zca_W_2,
                              patch=7, stride=1, pool=2, batch_size=128)
F_test_flip_2 = encode_images_gpu(X_test_flip, centroids_2, zca_mean_2, zca_W_2,
                                   patch=7, stride=1, pool=2, batch_size=128)
F_test_2_s = scaler_2.transform(power_norm(F_test_2)).astype(np.float32)
F_test_flip_2_s = scaler_2.transform(power_norm(F_test_flip_2)).astype(np.float32)
del F_test_2, F_test_flip_2; gc.collect()

df_orig_2 = clf_2.decision_function(F_test_2_s)
df_flip_2 = clf_2.decision_function(F_test_flip_2_s)
if hasattr(df_orig_2, "get"): df_orig_2 = df_orig_2.get()
if hasattr(df_flip_2, "get"): df_flip_2 = df_flip_2.get()
print(f"  Decision functions: {df_orig_2.shape}")

del F_test_2_s, F_test_flip_2_s, clf_2, scaler_2
gc.collect(); cp.get_default_memory_pool().free_all_blocks()

# %% [markdown]
# ## 5. Ensemble + TTA + Save Submission
#
# Soft-vote ensemble: average per-model TTA2 decision functions, then argmax.

# %%
# Per-model TTA2: average orig + flip decision functions
tta_1 = (df_orig_1 + df_flip_1) / 2
tta_2 = (df_orig_2 + df_flip_2) / 2

# Ensemble: average across models
ensemble_df = (tta_1 + tta_2) / 2
predictions = ensemble_df.argmax(axis=1)

print(f"Predictions shape: {predictions.shape}")
print(f"Unique classes: {np.unique(predictions)}")
print(f"Class distribution: {np.bincount(predictions)}")

# %%
# Save submission CSV
submission = pd.DataFrame({
    "im_name": test_names,
    "label": predictions
})
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"\nSubmission saved to: {output_path}")
print(f"Shape: {submission.shape}")
print(submission.head(10))
