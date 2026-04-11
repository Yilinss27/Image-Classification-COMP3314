# COMP3314 Assignment 3 — Image Classification

Classical-ML image classification on a CIFAR-10-style dataset (50k train / 10k test, 32×32 RGB, 10 classes) for the HKU COMP3314 Kaggle challenge.

**Constraints.** No neural networks, no pretrained models, no extra datasets. Only methods covered in the course lectures (preprocessing, PCA, SVM / LogReg, KMeans, ensembles, etc.).

## Validation progress

Running log of every time a new best-so-far validation accuracy was hit. Updated automatically after each periodic sweep check.

| Time (local) | Run | Config | Val acc | Kaggle public | Note |
|---|---|---|---|---|---|
| 2026-04-11 04:30 | run_01 | PCA(200) + SVC-RBF | 0.4974 | — | raw-pixel baseline |
| 2026-04-11 05:30 | run_02 | HOG + color hist + LBP → HistGB | 0.6468 | 0.64250 | first submission |
| 2026-04-11 06:18 | run_04 | Coates K=400 P=6 C=0.01 | 0.7302 | — | crossed the 0.70 bar |
| 2026-04-11 06:29 | run_04 | Coates K=400 P=6 C=0.03 | 0.7328 | **0.73350** | submitted, 50/50 locked |
| 2026-04-11 07:01 | run_05 | Coates K=1600 P=6 C=0.001 (GPU) | 0.7576 | — | GPU sweep started |
| 2026-04-11 07:35 | run_05 | Coates K=1600 P=6 C=0.003 (GPU) | 0.7656 | — | refit pending (low RAM) |
| 2026-04-11 08:17 | run_05 | Coates K=1600 P=6 C=0.01 (GPU) | 0.7678 | **0.77400** | full refit submitted |
| 2026-04-11 15:39 | run_06 | Coates K=3200 P=6 C=0.01 | — | 0.77150 | K=3200 slightly worse than K=1600 (diminishing returns) |

## run_07 cuML GPU sweep (2026-04-11 evening)

Wide `P × K × C` grid on a rented AutoDL RTX 5090 (Xeon 8470Q, 754 GiB RAM). 250 configs total:
`P ∈ {4,5,6,7,8}`, `K ∈ {400,800,1200,1600,2000,2400,3200,4000,6000,8000}`, `C ∈ {0.003,0.005,0.01,0.02,0.03}`, `pool=2×2`, `n_patches=1M`.
Classifier is **cuML's GPU `LinearSVC`** via L-BFGS — sklearn's `LinearSVC` is pathologically slow on this Xeon + OpenBLAS "Haswell" build (single K=400 fit did not finish in 8+ minutes). cuML fits each config in 2–90 s depending on feature dim.

cuML's L-BFGS solver lands ~0.3–0.5 pp below sklearn's liblinear coordinate-descent at the same `(P, K, C)`, so these val numbers are systematically slightly lower than the run_05 liblinear baselines.

Top configs so far (best val per (P, K), sorted descending — **P=4 branch complete, P=5/6/7/8 pending**):

| Rank | P | K | Best C | Val acc | Notes |
|---|---|---|---|---|---|
| 1 | 4 | 8000 | 0.003 | **0.7686** | 🎯 beats SOTA 0.7678 by 0.08 pp |
| 2 | 4 | 6000 | 0.005 | 0.7666 | |
| 3 | 4 | 4000 | 0.005 | 0.7646 | |
| 4 | 4 | 3200 | 0.005 | 0.7574 | |
| 5 | 4 | 2400 | 0.01  | 0.7546 | |
| 6 | 4 | 2000 | 0.01  | 0.7518 | |
| 7 | 4 | 1600 | 0.02  | 0.7502 | |
| 8 | 4 | 1200 | 0.03  | 0.7428 | |
| 9 | 4 | 800  | 0.03  | 0.7348 | |
| 10 | 4 | 400  | 0.03  | 0.7138 | |

P=4 monotonically improves with K all the way up to K=8000. Key finding: **cuML L-BFGS LinearSVC on P=4 K=8000 already matches/beats the sklearn liblinear K=1600 P=6 baseline**. P=5/6/7/8 stages pending — expect the best Coates patch sizes (P=6 traditionally) to push val higher still.

Reference: the current Kaggle public SOTA is 0.77400 (run_05 K=1600 P=6 C=0.01, val=0.7678), so anything `val ≥ 0.7678` here is a submission candidate once the sweep completes.

## Approach

Five progressively stronger pipelines, all built from `scikit-learn` / `scikit-image` primitives:

| Run | Pipeline | Classifier(s) | Val acc |
|---|---|---|---|
| `run_01_baseline.py` | raw pixels → StandardScaler → PCA(200) | LogReg / KNN / LinearSVC / SVC-RBF + soft VotingClassifier | ~0.50 |
| `run_02_hog.py` | HOG (gray + 3 RGB channels) + color histogram + LBP (≈1.4k dims) | HistGB / RF / LogReg / SVC-RBF + VotingClassifier | ~0.65 |
| `run_03_coates.py` | Coates & Ng (2011) single-layer feature learning: random 6×6 patches → contrast normalize → ZCA whiten → MiniBatchKMeans(K=800) → triangle encoding → 2×2 sum pool (3200 dims) | LinearSVC | ~0.73 |
| `run_04_coates_sweep.py` | Coates pipeline, sweep over `K ∈ {400, 800, 1600}`, `patch ∈ {6, 8}`, `C ∈ {1e-3 … 3e-1}` (CPU) | LinearSVC | up to ~0.75 |
| `run_05_gpu_coates.py` | Same sweep extended to `K ∈ {1600, 3200}`, GPU-accelerated patch encoding via `cupy` | LinearSVC | up to ~0.77 |
| `run_06_k3200.py` | Refit K=3200 P=6 C=0.01 on full 50k train (no val split) | LinearSVC | — (public 0.77150) |
| `run_07_cuml_sweep.py` | Wide `P × K × C` grid on 5090: P∈{4..8}, K∈{400..8000}, C∈{0.003..0.03}, cupy encoding + **cuML GPU LinearSVC** (sklearn LinearSVC is pathologically slow on the Xeon 8470Q + OpenBLAS "Haswell" build) | cuML LinearSVC | in progress |

The Coates-Ng single-layer unsupervised feature learning pipeline gives by far the best results and is the source of the final submission.

## Layout

```
runs/        entry-point scripts (one per experiment)
src/         shared data loading, logging, submission utilities
logs/        per-run stdout + structured logs with validation metrics
submissions/ generated Kaggle submission CSVs
cache/       (gitignored) cached .npy features + KMeans dictionaries
data/        (gitignored) raw train/test images and CSV label files
checkpoints/ (gitignored) pickled models
```

## Reproduction

```bash
conda create -n comp3314 python=3.11 -y
conda activate comp3314
pip install -r requirements.txt
# place competition data under data/{train.csv,test.csv,train_ims/,test_ims/}
python runs/run_02_hog.py
python runs/run_03_coates.py
python runs/run_04_coates_sweep.py
python runs/run_05_gpu_coates.py    # requires NVIDIA GPU + cupy-cuda12x
```

Each script caches intermediate features under `cache/` so re-runs (e.g. trying a different classifier) are fast.
