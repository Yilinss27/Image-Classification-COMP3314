"""Run 01 — Baseline: raw pixels + StandardScaler + PCA + (LogReg, KNN, SVC-RBF) + Voting.

Lectures used:
  L3: LogisticRegression, SVM (RBF), KNN
  L4: StandardScaler, stratified split
  L5: PCA dimensionality reduction
  L6: Pipeline, GridSearchCV, StratifiedKFold
  L7: VotingClassifier (soft voting ensemble)

Expected wall time on 20-core machine: ~30-60 min.
Expected accuracy: 0.45-0.55.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import (
    CKPT_DIR,
    LOG_DIR,
    SUB_DIR,
    Timer,
    get_logger,
    load_test_cached,
    load_train_cached,
    save_submission,
)

RUN_NAME = "run_01_baseline"


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny 1k-sample smoke test")
    ap.add_argument("--n-pca", type=int, default=200)
    ap.add_argument("--svm-subset", type=int, default=15000,
                    help="use up to this many samples for SVC(RBF) fit (n^2 cost)")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()

    log_file = LOG_DIR / f"{RUN_NAME}.log"
    log = get_logger(RUN_NAME, log_file)
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"args: {vars(args)}")

    # ---- Load ----
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
        args.n_pca = 64
        log.info("SMOKE MODE active")

    # ---- Flatten + split ----
    Xf = flatten(X)
    Xte_f = flatten(Xte)
    X_tr, X_val, y_tr, y_val = train_test_split(
        Xf, y, test_size=0.1, stratify=y, random_state=0
    )
    log.info(f"train {X_tr.shape}  val {X_val.shape}  test {Xte_f.shape}")

    # ---- Preprocess: scale + PCA fit on train only ----
    with Timer(log, "StandardScaler fit"):
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)
        Xte_s = scaler.transform(Xte_f).astype(np.float32)

    with Timer(log, f"PCA fit (n_components={args.n_pca})"):
        pca = PCA(n_components=args.n_pca, random_state=0)
        X_tr_p = pca.fit_transform(X_tr_s).astype(np.float32)
        X_val_p = pca.transform(X_val_s).astype(np.float32)
        Xte_p = pca.transform(Xte_s).astype(np.float32)
        log.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    results: dict[str, tuple[float, object]] = {}

    # ---- Model A: Logistic Regression ----
    with Timer(log, "LogReg fit"):
        logreg = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        logreg.fit(X_tr_p, y_tr)
        acc = accuracy_score(y_val, logreg.predict(X_val_p))
        log.info(f"LogReg val acc = {acc:.4f}")
        results["logreg"] = (acc, logreg)

    # ---- Model B: KNN ----
    with Timer(log, "KNN fit (k=10)"):
        knn = KNeighborsClassifier(n_neighbors=10, n_jobs=args.n_jobs, algorithm="auto")
        knn.fit(X_tr_p, y_tr)
        acc = accuracy_score(y_val, knn.predict(X_val_p))
        log.info(f"KNN val acc = {acc:.4f}")
        results["knn"] = (acc, knn)

    # ---- Model C: LinearSVC (fast, full data) ----
    with Timer(log, "LinearSVC fit"):
        lsvc = LinearSVC(C=0.1, dual="auto", max_iter=5000)
        lsvc.fit(X_tr_p, y_tr)
        acc = accuracy_score(y_val, lsvc.predict(X_val_p))
        log.info(f"LinearSVC val acc = {acc:.4f}")
        results["lsvc"] = (acc, lsvc)

    # ---- Model D: SVC RBF on subset (n^2 cost) ----
    n_sub = min(args.svm_subset, len(X_tr_p))
    rng = np.random.RandomState(1)
    sub = rng.choice(len(X_tr_p), n_sub, replace=False)
    with Timer(log, f"SVC-RBF fit ({n_sub} samples)"):
        svc = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, cache_size=1024)
        svc.fit(X_tr_p[sub], y_tr[sub])
        acc = accuracy_score(y_val, svc.predict(X_val_p))
        log.info(f"SVC-RBF val acc = {acc:.4f}")
        results["svc_rbf"] = (acc, svc)

    # ---- Validation summary ----
    log.info("=== Validation summary ===")
    for name, (acc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        log.info(f"  {name:12s}  {acc:.4f}")

    # ---- Ensemble: soft voting over probabilistic models (logreg + svc_rbf + knn) ----
    # Use logreg, knn and svc_rbf (all have predict_proba). LinearSVC doesn't.
    with Timer(log, "VotingClassifier (soft) fit"):
        voting = VotingClassifier(
            estimators=[
                ("logreg", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
                ("knn", KNeighborsClassifier(n_neighbors=10, n_jobs=args.n_jobs)),
                ("svc", SVC(kernel="rbf", C=10.0, gamma="scale",
                            probability=True, cache_size=1024)),
            ],
            voting="soft",
            n_jobs=1,  # fit sequentially to control memory
        )
        # Voting fits each on its own data. For SVC use subset, others full.
        # Simple approach: fit voting on the svm subset for all (fair comparison, faster).
        voting.fit(X_tr_p[sub], y_tr[sub])
        acc = accuracy_score(y_val, voting.predict(X_val_p))
        log.info(f"VotingClassifier val acc = {acc:.4f}")
        results["voting"] = (acc, voting)

    # ---- Pick best, refit on FULL training set (train+val), predict test ----
    best_name = max(results, key=lambda k: results[k][0])
    log.info(f"Best on val: {best_name} ({results[best_name][0]:.4f})")

    # Refit best on all of Xf (train+val combined) with same preprocessing
    with Timer(log, "Refit scaler+pca on full train+val"):
        Xall_s = scaler.fit_transform(Xf).astype(np.float32)
        Xte_full_s = scaler.transform(Xte_f).astype(np.float32)
        pca_full = PCA(n_components=args.n_pca, random_state=0)
        Xall_p = pca_full.fit_transform(Xall_s).astype(np.float32)
        Xte_full_p = pca_full.transform(Xte_full_s).astype(np.float32)

    with Timer(log, f"Refit {best_name} on full"):
        _, best_est = results[best_name]
        best_cls = type(best_est)
        # Rebuild with same hyperparams; for voting, clone full estimator
        if best_name == "voting":
            model = VotingClassifier(
                estimators=[
                    ("logreg", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
                    ("knn", KNeighborsClassifier(n_neighbors=10, n_jobs=args.n_jobs)),
                    ("svc", SVC(kernel="rbf", C=10.0, gamma="scale",
                                probability=True, cache_size=1024)),
                ],
                voting="soft",
            )
            # Refit on subset again to control time
            sub2 = np.random.RandomState(2).choice(len(Xall_p), min(args.svm_subset*2, len(Xall_p)), replace=False)
            model.fit(Xall_p[sub2], y[sub2])
        elif best_name == "svc_rbf":
            model = SVC(kernel="rbf", C=10.0, gamma="scale", cache_size=1024)
            sub2 = np.random.RandomState(2).choice(len(Xall_p), min(args.svm_subset*2, len(Xall_p)), replace=False)
            model.fit(Xall_p[sub2], y[sub2])
        elif best_name == "logreg":
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            model.fit(Xall_p, y)
        elif best_name == "knn":
            model = KNeighborsClassifier(n_neighbors=10, n_jobs=args.n_jobs)
            model.fit(Xall_p, y)
        elif best_name == "lsvc":
            model = LinearSVC(C=0.1, dual="auto", max_iter=5000)
            model.fit(Xall_p, y)
        else:
            raise ValueError(best_name)

    with Timer(log, "Predict test"):
        preds = model.predict(Xte_full_p)

    # ---- Save artifacts ----
    sub_path = SUB_DIR / f"sub_{RUN_NAME}.csv"
    save_submission(test_names, preds, sub_path)
    log.info(f"Wrote submission to {sub_path}")

    ckpt = CKPT_DIR / f"{RUN_NAME}.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca_full,
            "model": model,
            "best_name": best_name,
            "val_results": {k: v[0] for k, v in results.items()},
        },
        ckpt,
    )
    log.info(f"Wrote checkpoint to {ckpt}")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
