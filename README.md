# COMP3314 Assignment 3 — Image Classification

Classical-ML image classification on a CIFAR-10-style dataset (50k train / 10k test, 32×32 RGB, 10 classes) for the HKU COMP3314 Kaggle challenge.

**Constraints.** No neural networks, no pretrained models, no extra datasets. Only methods covered in the course lectures (preprocessing, PCA, SVM / LogReg, KMeans, ensembles, etc.).

## Approach

Five progressively stronger pipelines, all built from `scikit-learn` / `scikit-image` primitives:

| Run | Pipeline | Classifier(s) | Val acc |
|---|---|---|---|
| `run_01_baseline.py` | raw pixels → StandardScaler → PCA(200) | LogReg / KNN / LinearSVC / SVC-RBF + soft VotingClassifier | ~0.50 |
| `run_02_hog.py` | HOG (gray + 3 RGB channels) + color histogram + LBP (≈1.4k dims) | HistGB / RF / LogReg / SVC-RBF + VotingClassifier | ~0.65 |
| `run_03_coates.py` | Coates & Ng (2011) single-layer feature learning: random 6×6 patches → contrast normalize → ZCA whiten → MiniBatchKMeans(K=800) → triangle encoding → 2×2 sum pool (3200 dims) | LinearSVC | ~0.73 |
| `run_04_coates_sweep.py` | Coates pipeline, sweep over `K ∈ {400, 800, 1600}`, `patch ∈ {6, 8}`, `C ∈ {1e-3 … 3e-1}` (CPU) | LinearSVC | up to ~0.75 |
| `run_05_gpu_coates.py` | Same sweep extended to `K ∈ {1600, 3200}`, GPU-accelerated patch encoding via `cupy` | LinearSVC | up to ~0.77 |

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
