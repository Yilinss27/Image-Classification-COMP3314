"""Run 02 — HOG + Color histogram + LBP features + (SVM-RBF, HistGB, RF) + Voting.

Lectures used:
  L3: SVM (RBF), Decision Trees
  L4: StandardScaler
  L6: Pipeline, cross-validation
  L7: Ensemble (RandomForest, VotingClassifier)

Feature pipeline (all hand-crafted, no neural nets):
  - HOG on grayscale (pixels_per_cell=8, cells_per_block=2)       ~144 dims
  - HOG on each of 3 RGB channels                                  ~432 dims
  - Color histogram per channel (16 bins x 3)                       48 dims
  - LBP histogram (uniform, P=8, R=1)                               10 dims
  Total ~ 630 dims

Expected wall time: 2-4 h.
Expected accuracy: 0.55-0.65.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

RUN_NAME = "run_02_hog"

HOG_KW = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
              block_norm="L2-Hys", feature_vector=True)
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"
LBP_NBINS = LBP_P + 2  # uniform → P+2 bins
HIST_BINS = 16


def _features_one(img: np.ndarray) -> np.ndarray:
    """img: (32,32,3) uint8 → 1-D feature vector."""
    img_f = img.astype(np.float32) / 255.0
    gray = rgb2gray(img_f)

    feats = []
    feats.append(hog(gray, **HOG_KW))
    for c in range(3):
        feats.append(hog(img_f[:, :, c], **HOG_KW))

    # Color histogram
    for c in range(3):
        h, _ = np.histogram(img[:, :, c], bins=HIST_BINS, range=(0, 256), density=True)
        feats.append(h.astype(np.float32))

    # LBP on grayscale
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), LBP_P, LBP_R, LBP_METHOD)
    h, _ = np.histogram(lbp, bins=LBP_NBINS, range=(0, LBP_NBINS), density=True)
    feats.append(h.astype(np.float32))

    return np.concatenate(feats).astype(np.float32)


def _features_batch(imgs: np.ndarray) -> np.ndarray:
    return np.stack([_features_one(im) for im in imgs], axis=0)


def extract_features(X: np.ndarray, n_jobs: int, log, label: str) -> np.ndarray:
    with Timer(log, f"extract HOG features ({label}, n={len(X)})"):
        chunk = max(200, len(X) // (n_jobs * 4) + 1)
        batches = [X[i : i + chunk] for i in range(0, len(X), chunk)]
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_features_batch)(b) for b in batches
        )
        F = np.concatenate(results, axis=0)
    log.info(f"features shape {F.shape}  dtype {F.dtype}")
    return F


def features_cached(X: np.ndarray, tag: str, n_jobs: int, log) -> np.ndarray:
    cache = CACHE_DIR / f"hog_{tag}.npy"
    if cache.exists():
        F = np.load(cache)
        log.info(f"loaded cached features {cache.name} shape {F.shape}")
        return F
    F = extract_features(X, n_jobs, log, tag)
    np.save(cache, F)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--svm-subset", type=int, default=20000)
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
    log.info(f"train X {X.shape}  y {y.shape}   test X {Xte.shape}")

    if args.smoke:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X), 1000, replace=False)
        X, y = X[idx], y[idx]
        Xte = Xte[:500]; test_names = test_names[:500]
        args.svm_subset = 800
        log.info("SMOKE MODE active")
        tag_tr = "smoke_train"
        tag_te = "smoke_test"
    else:
        tag_tr = "train"
        tag_te = "test"

    F_all = features_cached(X, tag_tr, args.n_jobs, log)
    F_te = features_cached(Xte, tag_te, args.n_jobs, log)

    F_tr, F_val, y_tr, y_val = train_test_split(
        F_all, y, test_size=0.1, stratify=y, random_state=0
    )
    log.info(f"train {F_tr.shape}  val {F_val.shape}  test {F_te.shape}")

    with Timer(log, "StandardScaler fit"):
        scaler = StandardScaler().fit(F_tr)
        F_tr_s = scaler.transform(F_tr).astype(np.float32)
        F_val_s = scaler.transform(F_val).astype(np.float32)
        F_te_s = scaler.transform(F_te).astype(np.float32)

    results: dict[str, tuple[float, object]] = {}

    # ---- HistGradientBoostingClassifier on full set (fast, strong baseline) ----
    with Timer(log, "HistGB fit (full)"):
        hgb = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=None,
            l2_regularization=1.0, random_state=0
        )
        hgb.fit(F_tr_s, y_tr)
        acc = accuracy_score(y_val, hgb.predict(F_val_s))
        log.info(f"HistGB val acc = {acc:.4f}")
        results["hgb"] = (acc, hgb)

    # ---- RandomForest on full set ----
    with Timer(log, "RandomForest fit (full)"):
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", n_jobs=args.n_jobs, random_state=0
        )
        rf.fit(F_tr_s, y_tr)
        acc = accuracy_score(y_val, rf.predict(F_val_s))
        log.info(f"RF val acc = {acc:.4f}")
        results["rf"] = (acc, rf)

    # ---- LogReg (fast sanity check) ----
    with Timer(log, "LogReg fit"):
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000)
        lr.fit(F_tr_s, y_tr)
        acc = accuracy_score(y_val, lr.predict(F_val_s))
        log.info(f"LogReg val acc = {acc:.4f}")
        results["logreg"] = (acc, lr)

    # ---- SVC RBF on a subset ----
    n_sub = min(args.svm_subset, len(F_tr_s))
    sub = np.random.RandomState(1).choice(len(F_tr_s), n_sub, replace=False)
    with Timer(log, f"SVC-RBF fit ({n_sub} samples, C=10, gamma=scale)"):
        svc = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, cache_size=2048)
        svc.fit(F_tr_s[sub], y_tr[sub])
        acc = accuracy_score(y_val, svc.predict(F_val_s))
        log.info(f"SVC-RBF val acc = {acc:.4f}")
        results["svc_rbf"] = (acc, svc)

    log.info("=== Validation summary ===")
    for name, (acc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        log.info(f"  {name:12s}  {acc:.4f}")

    # ---- Voting ensemble (soft) over all four ----
    with Timer(log, "Voting (soft) fit on svm subset"):
        voting = VotingClassifier(
            estimators=[
                ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                       l2_regularization=1.0, random_state=0)),
                ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                              max_features="sqrt", n_jobs=args.n_jobs, random_state=0)),
                ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, cache_size=2048)),
            ],
            voting="soft",
        )
        voting.fit(F_tr_s[sub], y_tr[sub])
        acc = accuracy_score(y_val, voting.predict(F_val_s))
        log.info(f"Voting val acc = {acc:.4f}")
        results["voting"] = (acc, voting)

    # ---- Pick best, refit on full (train+val) ----
    best_name = max(results, key=lambda k: results[k][0])
    log.info(f"Best on val: {best_name} ({results[best_name][0]:.4f})")

    with Timer(log, "Refit scaler on full train+val"):
        scaler_full = StandardScaler().fit(F_all)
        F_all_s = scaler_full.transform(F_all).astype(np.float32)
        F_te_full = scaler_full.transform(F_te).astype(np.float32)

    with Timer(log, f"Refit {best_name} on full"):
        if best_name == "hgb":
            model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.08, l2_regularization=1.0, random_state=0
            )
            model.fit(F_all_s, y)
        elif best_name == "rf":
            model = RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2,
                max_features="sqrt", n_jobs=args.n_jobs, random_state=0
            )
            model.fit(F_all_s, y)
        elif best_name == "logreg":
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000)
            model.fit(F_all_s, y)
        elif best_name == "svc_rbf":
            model = SVC(kernel="rbf", C=10.0, gamma="scale", cache_size=2048)
            sub2 = np.random.RandomState(2).choice(len(F_all_s), min(args.svm_subset*2, len(F_all_s)), replace=False)
            model.fit(F_all_s[sub2], y[sub2])
        elif best_name == "voting":
            model = VotingClassifier(
                estimators=[
                    ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                           l2_regularization=1.0, random_state=0)),
                    ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                  max_features="sqrt", n_jobs=args.n_jobs, random_state=0)),
                    ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, cache_size=2048)),
                ],
                voting="soft",
            )
            sub2 = np.random.RandomState(2).choice(len(F_all_s), min(args.svm_subset*2, len(F_all_s)), replace=False)
            model.fit(F_all_s[sub2], y[sub2])
        else:
            raise ValueError(best_name)

    with Timer(log, "Predict test"):
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
            "feature_spec": {"hog": HOG_KW, "lbp_P": LBP_P, "lbp_R": LBP_R,
                             "lbp_method": LBP_METHOD, "hist_bins": HIST_BINS},
        },
        ckpt,
    )
    log.info(f"Wrote checkpoint to {ckpt}")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
