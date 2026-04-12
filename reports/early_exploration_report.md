# Early Exploration Report: Runs 01-06

## Model Architecture Exploration Phase

**Date:** 2026-04-11  
**Task:** CIFAR-10-style image classification (50k train / 10k test, 32x32 RGB, 10 classes)  
**Constraint:** No neural networks, no pretrained models, no extra data — only methods from COMP3314 lectures.  
**Goal:** Find the best classical-ML feature representation before launching a large-scale hyperparameter sweep.

---

## Executive Summary

Over the course of 6 experiments on April 11, 2026, we progressed from a naive raw-pixel baseline (val 0.4974) through handcrafted features (val 0.6468) to unsupervised learned features via the Coates-Ng pipeline (val 0.7678, public 0.77400). This represents a **+0.27 absolute accuracy improvement** through architectural choices alone, before any systematic hyperparameter optimization.

| Run | Approach | Best Val Acc | Public LB | Wall Time |
|-----|----------|-------------|-----------|-----------|
| run_01 | Raw pixels + PCA(200) + SVC-RBF | 0.4974 | -- | ~8 min |
| run_02 | HOG + Color + LBP + HistGB | 0.6468 | 0.64250 | ~82 min |
| run_03 | Coates K=800 P=6 + LinearSVC | ~0.73 | -- | ~7 min (encode only) |
| run_04 | Coates sweep K={400,800,1600} | 0.7544 | 0.73350 | ~4.5 hr |
| run_05 | Coates GPU K=1600 P=6 C=0.01 | 0.7678 | 0.77400 | ~4 hr |
| run_06 | Coates K=3200 P=6 C=0.01 (full refit) | -- | 0.77150 | ~2 hr |

**Narrative arc:** raw pixels (0.50) -> handcrafted features (0.65) -> unsupervised learned features (0.73) -> GPU-scaled features (0.77) -> ready for massive sweep.

---

## 1. run_01 -- Raw Pixel Baseline

### Motivation

Establish a lower bound using the simplest possible representation: flatten 32x32x3 images into 3072-dimensional vectors, reduce with PCA, and classify with standard models from lectures.

### Method

1. **Preprocessing:** Flatten images to 3072 dims, StandardScaler, PCA(200) retaining 94.96% of variance.
2. **Classifiers:** LogisticRegression, KNN(k=10), LinearSVC(C=0.1), SVC-RBF(C=10, gamma=scale), VotingClassifier (soft).
3. **Note:** SVC-RBF was trained on a 15k-sample subset due to O(n^2) cost; VotingClassifier also on same subset.

### Results

| Classifier | Val Accuracy | Training Time | Notes |
|-----------|-------------|---------------|-------|
| SVC-RBF | **0.4974** | 124.2 s | Best; 15k subset only |
| VotingClassifier | 0.4756 | 118.5 s | Ensemble of LogReg+KNN+SVC |
| LogisticRegression | 0.4114 | 4.5 s | Fast but weak |
| LinearSVC | 0.4038 | 30.7 s | Linear boundary insufficient |
| KNN (k=10) | 0.3588 | 4.2 s | Curse of dimensionality |

**Total wall time:** ~8 minutes (04:19 to 04:27).

### Analysis

- **Why so low?** Raw pixels lack invariance to any geometric or photometric transformation. A 1-pixel shift in an image creates a completely different feature vector. PCA captures global covariance structure but cannot encode local patterns like edges or textures.
- **Why SVC-RBF wins?** The RBF kernel can model nonlinear decision boundaries in the PCA space. Linear models (LogReg, LinearSVC) are constrained to hyperplanes in a space where classes are not linearly separable.
- **Why KNN is worst?** With 200 PCA dimensions, Euclidean distance becomes a poor similarity metric — the "curse of dimensionality" from Lecture 5.
- **PCA variance:** 200 components capture 94.96% of pixel variance, but variance does not equal discriminative information. Background pixels have high variance but low class relevance.

### Decision

Raw pixels are a dead end for this dataset. We need features that capture **local structure** (edges, textures) with some invariance to position and lighting. Next step: handcrafted descriptors from the computer vision literature.

---

## 2. run_02 -- Handcrafted Features (HOG + Color + LBP)

### Motivation

Use domain knowledge to design features that capture the properties raw pixels cannot: edge orientation (HOG), color distribution (histograms), and local texture (LBP). All are standard descriptors from computer vision, implementable with scikit-image.

### Method

**Feature extraction pipeline (per image):**

| Feature | Description | Dimensions |
|---------|-------------|-----------|
| HOG (grayscale) | Histogram of Oriented Gradients, 9 orientations, 8x8 cells, 2x2 blocks, L2-Hys norm | ~144 |
| HOG (R, G, B) | Same HOG on each color channel separately | ~432 |
| Color histogram | 16-bin histogram per channel, density-normalized | 48 |
| LBP histogram | Local Binary Pattern (P=8, R=1, uniform), 10 bins | 10 |
| **Total** | | **~1354** |

**Classifiers:** HistGradientBoostingClassifier (300 iters, lr=0.08), RandomForest (400 trees), LogisticRegression, SVC-RBF (20k subset), VotingClassifier.

### Results

| Classifier | Val Accuracy | Training Time | Notes |
|-----------|-------------|---------------|-------|
| HistGradientBoosting | **0.6468** | 396.2 s | Best; fast on full 45k |
| SVC-RBF | 0.6416 | 2027.4 s | Near-best but 5x slower |
| VotingClassifier | 0.6434 | 2028.2 s | Ensemble on SVM subset |
| RandomForest | 0.5674 | 104.1 s | Weak on this feature set |
| LogisticRegression | 0.5550 | 58.0 s | Linear model, good baseline |

**Total wall time:** ~82 minutes (04:27 to 05:49).  
**Feature extraction:** 12.4 s (train), 2.6 s (test) — fast with joblib parallelism.  
**Kaggle public score:** 0.64250 (first submission).

### Analysis

- **+0.15 over raw pixels.** HOG captures edge structure that makes objects recognizable regardless of exact pixel positions. Color histograms provide global chromatic information (e.g., "green" strongly suggests frog/bird on grass).
- **Why HistGB wins?** Gradient boosting excels at heterogeneous features with different scales and distributions. It automatically handles the mix of HOG (smooth, real-valued) and histograms (sparse, discrete).
- **Why RF underperforms?** Random forests with `max_features=sqrt` see only ~37 features per split. With 1354 dims of mixed character, individual features are weak discriminators — boosting's sequential correction outperforms bagging's parallel averaging.
- **Why SVC-RBF doesn't dominate?** Unlike run_01, the feature space is already more structured. The 20k subset limitation caps its potential — more data would likely help.
- **Limitations:** HOG is computed at a single scale and orientation. LBP is local (radius=1 pixel). No spatial layout is preserved after histogram pooling. The feature design reflects our priors, not the data's structure.

### Decision

Handcrafted features provide a solid +0.15 improvement, but we are limited by our ability to design them. The next logical step: **learn features from the data itself** using unsupervised methods. The Coates & Ng (2011) approach — random patches, whitening, KMeans dictionary, triangle encoding — uses only KMeans (Lecture 9) and linear classifiers (Lecture 3), fitting squarely within course constraints.

---

## 3. run_03 -- First Coates-Ng Implementation

### Motivation

Replace hand-designed features with **data-driven** ones. The Coates & Ng (2011) single-layer feature learning pipeline was shown to match or exceed deep networks on CIFAR-10 using only KMeans and a linear classifier. Every component maps to a course lecture:

- Random patch sampling (data collection)
- Contrast normalization (L4: preprocessing)
- ZCA whitening (L4/L5: decorrelation, PCA variant)
- MiniBatchKMeans (L9: clustering)
- Linear classifier (L3: SVM)

### Method

**Pipeline:**

```
Image (32x32x3)
  -> Extract all 6x6 patches stride-1 (27x27 = 729 patches per image)
  -> Per-patch contrast normalization: (x - mean) / sqrt(var + 10)
  -> ZCA whitening (global, fit on 400k random training patches)
  -> Triangle encoding: f_k = max(0, mean(distances) - dist_to_centroid_k)
  -> 2x2 spatial sum-pooling (4 quadrants)
  -> Feature vector: 4 * K = 3200 dims (for K=800)
```

**Parameters:** K=800 centroids, P=6 patch size, 400k patches for dictionary learning.  
**Classifiers:** LinearSVC at C={0.01, 0.1, 1.0}, LogisticRegression.

### Results

| Classifier | Val Accuracy | Notes |
|-----------|-------------|-------|
| LinearSVC (C=0.01) | ~0.73* | Best from logs |
| LinearSVC (C=0.1) | ~0.73* | Similar range |
| LinearSVC (C=1.0) | ~0.72* | Slight overregularization |
| LogisticRegression | ~0.72* | Comparable |

*Exact values not preserved in truncated log; run_04 with K=800 confirms val=0.7544 at C=0.01.

**Timing breakdown:**
- Dictionary learning (sample + normalize + ZCA + KMeans): ~12 s
- Encode 50k train images: 299.7 s (~5 min)
- Encode 10k test images: 71.5 s
- Classifier fit: variable (minutes to tens of minutes on this CPU)

### Analysis

- **Massive jump: +0.08 over HOG.** The learned features capture visual patterns that humans would not think to encode — texture gradients, color transitions, and micro-patterns that KMeans discovers from the data distribution.
- **Why triangle encoding?** Soft thresholding `max(0, mu - z_k)` produces sparse, non-negative features. Only centroids closer than average activate, creating a competition among dictionary elements. This is similar to ReLU in neural networks but derived from distance geometry.
- **Why 2x2 pooling?** Spatial pooling aggregates activations within each quadrant, providing coarse spatial information (e.g., "sky features in top quadrants, ground features in bottom") while adding translation invariance within each quadrant.
- **Encoding bottleneck:** The 300s encoding time is dominated by the `(B*729, 108) @ (108, K)` matmul and subsequent distance computation. This scales linearly with K and quadratically with patch count.

### Decision

The Coates pipeline clearly dominates both raw pixels and handcrafted features. Key questions:
1. How does accuracy scale with K (dictionary size)?
2. Does patch size P matter?
3. What is the optimal regularization C?

We need a systematic sweep. Next: run_04 explores (K, P, C) on CPU.

---

## 4. run_04 -- Coates Hyperparameter Sweep (CPU)

### Motivation

Systematically map the (K, P, C) landscape to understand which knob matters most. All intermediate artifacts (dictionaries, encoded features) are cached so that:
- Adding new C values to an existing (K, P) config costs only ~30 seconds.
- The run_03 K=800 cache is reused for free.

### Method

**Sweep grid:**
- K (dictionary size): {400, 800, 1600}
- P (patch size): {6, 8}
- C (LinearSVC regularization): {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}
- Total: 5 configs x 6 C values = **30 experiments**

**Fixed:** stride=1, pool=2x2, n_patches=400k, val split=10% stratified (seed=0).

### Results (K=400, P=6)

| C | Val Accuracy | LinearSVC Time |
|---|-------------|----------------|
| 0.001 | 0.7096 | 314.0 s |
| 0.003 | 0.7254 | 401.7 s |
| **0.01** | **0.7302** | 546.3 s |
| **0.03** | **0.7328** | 661.6 s |
| 0.1 | 0.7324 | 906.1 s |
| 0.3 | 0.7272 | 1156.8 s |

### Results (K=800, P=6)

| C | Val Accuracy | LinearSVC Time |
|---|-------------|----------------|
| 0.001 | 0.7362 | 708.2 s |
| 0.003 | 0.7510 | 929.5 s |
| **0.01** | **0.7544** | 1070.2 s |
| 0.03 | 0.7472 | 1443.2 s |
| 0.1 | 0.7348 | 1601.4 s |
| 0.3 | 0.7224 | 2124.5 s |

### Results (K=1600, P=6) -- partial

| C | Val Accuracy | LinearSVC Time |
|---|-------------|----------------|
| 0.001 | 0.7554 | 1386.2 s |
| 0.003 | ~0.76* | ~30 min |

*K=1600 sweep was interrupted by the parallel GPU run (run_05) which completed faster.

### Key Findings

1. **K matters most:** K=400 -> K=800 gains +0.02; K=800 -> K=1600 gains another +0.02. More centroids = richer visual vocabulary = better linear separability.
2. **Optimal C shifts with K:** K=400 peaks at C=0.03; K=800 peaks at C=0.01. Larger K produces more features, requiring less regularization (lower C) to prevent overfitting the expanded space.
3. **LinearSVC time scales super-linearly with C and K:** Higher C means harder optimization (more support vectors); higher K means wider feature space. K=800 at C=0.3 takes 35 minutes — K=1600+ configs become impractical on this CPU.
4. **Submitted K=400 C=0.03:** Public score 0.73350 — matches val closely (val 0.7328), indicating no overfitting.

### Timing Issue

The sklearn `LinearSVC` (liblinear coordinate descent) on this Xeon + OpenBLAS configuration was **pathologically slow** — a single K=1600 C=0.01 fit took 23+ minutes. The issue appeared to be poor BLAS threading on the "Haswell" OpenBLAS build. This CPU bottleneck, rather than encoding time, became the limiting factor.

### Decision

K scaling clearly helps, but CPU LinearSVC cannot handle K >= 1600 in reasonable time. Solution: move encoding to GPU (cupy) and hope the smaller classifier fit time is acceptable. Alternatively, try GPU-accelerated SVM later (foreshadowing cuML in run_07).

---

## 5. run_05 -- GPU-Accelerated Coates (cupy)

### Motivation

The encoding step is a large matrix multiplication: `(M patches, D) @ (D, K)` where M ~ batch_size * 729. On a 4 GiB GPU via cupy, this should be 10-20x faster than CPU. Target the high-K configs (K=1600, K=3200) that run_04 could not reach efficiently.

### Method

**GPU acceleration strategy:**
- Dictionary learning (KMeans): stays on CPU (small, one-time cost)
- Patch extraction: CPU (stride tricks, memory-bound)
- Contrast norm + ZCA + distance + triangle + pooling: **GPU via cupy**
- Classifier (LinearSVC): CPU (operates on compact 4K features)

**Sweep grid:**
- K: {1600, 3200}
- P: {6, 8}
- C: {0.001, 0.003, 0.01, 0.03, 0.1}
- Total: 4 configs x 5 C values = **20 experiments**

**Hardware:** GPU with 4.00 GiB total / 3.22 GiB free.

### Results (K=1600, P=6) -- the winner

| C | Val Accuracy | LinearSVC Time | Cumulative |
|---|-------------|----------------|------------|
| 0.001 | 0.7576 | 2035.2 s | Solid baseline |
| 0.003 | 0.7656 | 2053.7 s | New best |
| **0.01** | **0.7678** | 2496.7 s | **Overall best** |
| 0.03 | 0.7532 | 2624.3 s | Overfitting begins |
| 0.1 | 0.7368 | 3494.4 s | Clear overfit |

**GPU encoding time:** 447.0 s for 50k train images (K=1600) vs. 486.7 s on CPU (run_04) — surprisingly similar because this 4 GiB GPU was memory-limited, requiring smaller batch sizes.

### Results (K=3200, P=6)

| C | Val Accuracy | Notes |
|---|-------------|-------|
| 0.001 | ~0.755* | Below K=1600 peak |
| 0.003 | ~0.76* | Still below |
| 0.01 | ~0.76* | Disappointing |

*K=3200 results from the GPU sweep suggested diminishing returns, motivating run_06 as a dedicated test.

### Key Findings

1. **K=1600 P=6 C=0.01 is the new SOTA:** val 0.7678, public **0.77400** (+0.04 over run_04's public 0.73350).
2. **C=0.01 is optimal for K=1600:** Consistent with the trend from run_04 — larger K shifts optimal C lower.
3. **K=3200 shows diminishing returns:** Despite doubling the dictionary, accuracy does not meaningfully improve. Possible explanations:
   - The 108-dim patch space (6x6x3) may be saturated by 1600 centroids — additional centroids capture noise.
   - LinearSVC struggles with the expanded 12800-dim feature space (regularization issues).
   - C=0.01 may not be optimal for K=3200 (would need lower C).
4. **LinearSVC still slow:** Even with fast GPU encoding, the classifier fit at K=1600 takes 34-58 minutes per C value. This is the true bottleneck — liblinear's coordinate descent on this hardware.

### Decision

K=1600 C=0.01 is clearly the best single config found so far. Before declaring victory, test K=3200 with a full 50k refit (no val split) to confirm the diminishing returns on the public leaderboard.

---

## 6. run_06 -- K=3200 Full Refit

### Motivation

The val results for K=3200 in run_05 were ambiguous — it is possible that:
1. K=3200 needs a different C (perhaps 0.003 or lower).
2. The 45k train subset is insufficient for K=3200's 12800-dim features.

Test hypothesis (2) by refitting on the full 50k training set (no val holdout) and submitting directly to Kaggle.

### Method

```python
# Minimal script: load cached features, scale, fit, predict
X_tr = np.load("gpu_feat_train_K3200_P6.npy")   # (50000, 12800)
sc = StandardScaler().fit_transform(X_tr)
clf = LinearSVC(C=0.01, dual=False, max_iter=3000, tol=1e-5)
clf.fit(X_tr_scaled, y)
preds = clf.predict(X_te_scaled)
```

**Key difference:** `dual=False` — primal formulation is faster when n_features > n_samples/2 (12800 > 25000). This helps somewhat.

### Results

| Metric | Value |
|--------|-------|
| Feature dim | 12800 (4 x 3200) |
| Train size | 50000 (full, no val split) |
| LinearSVC time | **7347.8 s (~2 hours)** |
| Kaggle public | **0.77150** |

### Comparison with K=1600

| Config | Val Acc | Public | Feature Dim | SVC Time |
|--------|---------|--------|-------------|----------|
| K=1600 P=6 C=0.01 | 0.7678 | **0.77400** | 6400 | ~42 min |
| K=3200 P=6 C=0.01 | -- | 0.77150 | 12800 | ~122 min |

**K=3200 is 0.0025 WORSE on public despite 2x more features and 3x more compute.**

### Analysis

1. **Diminishing returns confirmed.** The 108-dim patch space (6x6x3 = 108 dimensions) is well-covered by 1600 centroids. Adding more centroids captures increasingly fine-grained distinctions that are not discriminative for 10-class classification.
2. **Possible overfitting to noise patterns.** With 12800 features and 50000 samples, the feature-to-sample ratio is 0.256. LinearSVC at C=0.01 may not regularize strongly enough — a lower C (0.003 or 0.001) might help K=3200.
3. **Training cost explosion.** 2+ hours for a single fit makes hyperparameter tuning impractical. This motivates moving to cuML's GPU-accelerated LinearSVC (implemented in run_07).
4. **Class distribution sanity check.** The prediction distribution `[986, 994, 947, 905, 955, 1050, 1096, 1032, 992, 1043]` is reasonably balanced — the model is not degenerate.

### Decision

K=3200 at C=0.01 does not beat K=1600. Rather than continuing to hand-tune (K, C) pairs one at a time, we need:
1. **GPU-accelerated classifier** (cuML) to make large-K fits feasible in seconds instead of hours.
2. **Systematic wide sweep** over P, K, and C simultaneously to find the global optimum.

This leads directly to **run_07**: a 250-config grid search on a rented RTX 5090 with cuML's GPU LinearSVC, sweeping P in {4,5,6,7,8}, K in {400..8000}, C in {0.003..0.03}.

---

## Overall Analysis

### Accuracy Progression

```
run_01:  0.4974  |████████████████████████░░░░░░░░░░░░░░░░|
run_02:  0.6468  |████████████████████████████████░░░░░░░░|
run_03:  ~0.73   |████████████████████████████████████░░░░|
run_04:  0.7328  |████████████████████████████████████░░░░|  (public 0.73350)
run_05:  0.7678  |██████████████████████████████████████░░|  (public 0.77400)
run_06:  --      |██████████████████████████████████████░░|  (public 0.77150)
```

### What Each Transition Taught Us

| Transition | Accuracy Gain | Key Insight |
|-----------|--------------|-------------|
| run_01 -> run_02 | +0.15 | Feature engineering matters more than classifier choice |
| run_02 -> run_03 | +0.08 | Learned features beat handcrafted features |
| run_03 -> run_05 | +0.04 | Scaling K (dictionary size) provides consistent gains |
| run_05 -> run_06 | -0.003 | Diminishing returns exist; need smarter exploration |

### Bottleneck Evolution

| Phase | Bottleneck | Solution |
|-------|-----------|----------|
| run_01-02 | Feature quality | Better features (HOG, then Coates) |
| run_03-04 | Dictionary size (K) limited by encoding speed | GPU encoding (cupy) |
| run_05-06 | Classifier training time (LinearSVC on CPU) | GPU classifier (cuML, in run_07) |

### Critical Design Choices

1. **Patch size P=6 dominant.** Produces 27x27 patch grid -> 729 patches per image. The 108-dim patch space (6x6x3) is small enough for efficient KMeans but rich enough to capture local structure.

2. **Triangle encoding superior to hard assignment.** Unlike one-hot (which activates only the nearest centroid), triangle activation creates a distributed, sparse code that preserves distance information. This is what makes linear classifiers work on the encoded features.

3. **2x2 spatial pooling is the right granularity.** 1x1 (no pooling) would lose all spatial information; 4x4 would create K*16 features (too many for the sample size). 2x2 gives 4 spatial bins = 4K features — a good balance.

4. **StandardScaler essential before LinearSVC.** The pooled features have very different magnitudes across quadrants and centroids. Without scaling, the SVM objective is dominated by high-magnitude features.

### Reproducibility Notes

- All runs use `random_state=0` for splits and `random_state=0` for KMeans.
- Val split is consistent: `train_test_split(..., test_size=0.1, stratify=y, random_state=0)` giving 45000/5000.
- Feature caches are keyed by (K, P, n_patches) so results are deterministic given the same data.

---

## Figures

- `reports/figures/early_01_progression.png` — Bar chart of best val accuracy across runs 01-06.
- `reports/figures/early_02_classifier_comparison.png` — Classifier comparison within run_01 and run_02.

Generated by `reports/analyze_early.py`.

---

## Conclusion

The early exploration phase established that:

1. **Unsupervised feature learning (Coates-Ng) dramatically outperforms both raw pixels and handcrafted features** for this task — a +0.27 improvement from run_01 to run_05.
2. **Dictionary size K is the most important hyperparameter**, with clear gains from 400 to 1600 but diminishing returns at 3200.
3. **The CPU LinearSVC bottleneck** prevents efficient exploration of large-K configurations, motivating the move to cuML GPU training in run_07.
4. **The pipeline is ready for systematic optimization:** all components are validated, caching infrastructure is in place, and the remaining gains likely come from (a) wider K/P/C sweeps and (b) training data augmentation (explored post-run_07).

The stage is set for the 250-config sweep (run_07) on a rented RTX 5090 with cuML, which would push accuracy to 0.7858 (val) / 0.78400 (public).
