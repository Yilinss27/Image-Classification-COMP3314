"""Run 09: sklearn MKL refit for top cuML winners.

Uses the sklmkl conda env (MKL-backed numpy + sklearn 1.8) and re-encodes
features via cupy on GPU using the existing run_07 dict caches. Fits
sklearn LinearSVC with `dual=False` (primal coord descent — fast when
n >> d which is our regime).

Goal: recover the known ~0.3-0.5 pp solver gap between cuML L-BFGS and
sklearn liblinear at the same (P, K, C).

Expected runtime per config:
  - GPU encode: ~1-2 min
  - sklearn fit (dual=False, single-core liblinear): 1-10 min
  - full refit for submission: 2x val fit

Usage:
  conda activate sklmkl
  python runs/run_09_sklearn_refit.py --only P7K6000C0.002
  python runs/run_09_sklearn_refit.py   # run all targets
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LOG_DIR, SUB_DIR, Timer, get_logger, load_test_cached, load_train_cached, save_submission
from run_07_cuml_sweep import encode_images_gpu, get_or_fit_dict

RUN_NAME = "run_09_sklearn_refit"
N_PATCHES = 1_000_000
POOL = 2
STRIDE = 1
BATCH_SIZE = 128

# (P, K, C, tag, cuml_val) — tag is submission filename stub
TARGETS = [
    (7, 6000, 0.002, "P7K6000C0.002", 0.7876),   # new SOTA from Phase B
    (6, 8000, 0.003, "P6K8000C0.003", 0.7858),   # run_07 SOTA
    (6, 8000, 0.005, "P6K8000C0.005", 0.7842),
    (7, 4000, 0.003, "P7K4000C0.003", 0.7824),
    (8, 3200, 0.003, "P8K3200C0.003", 0.7816),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Run just one target by its tag (e.g. P7K6000C0.002)")
    ap.add_argument("--skip-refit", action="store_true",
                    help="Only fit on 90% train split, skip full-50k refit + submission")
    args = ap.parse_args()

    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")

    with Timer(log, "load train/test"):
        X, y, _ = load_train_cached(n_jobs=8)
        Xte, test_names = load_test_cached(n_jobs=8)

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0)
    y_tr, y_val = y[idx_tr], y[idx_val]

    rng = np.random.RandomState(0)
    results = []

    targets = [t for t in TARGETS if (args.only is None or t[3] == args.only)]
    if not targets:
        log.error(f"no target matches --only={args.only}")
        return
    log.info(f"will run {len(targets)} targets")

    for patch, K, C, tag, cuml_val in targets:
        log.info(f"\n### {tag} (cuML val={cuml_val:.4f}) ###")

        with Timer(log, f"dict K={K} P={patch}"):
            centroids, zca_mean, zca_W = get_or_fit_dict(
                X[idx_tr], K, patch, N_PATCHES, log, rng)

        with Timer(log, f"GPU encode train+test K={K} P={patch}"):
            F_full_tr = encode_images_gpu(
                X, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
            F_test = encode_images_gpu(
                Xte, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
        log.info(f"  F_full_tr {F_full_tr.shape}  F_test {F_test.shape}")

        F_tr = F_full_tr[idx_tr]
        F_val = F_full_tr[idx_val]
        scaler = StandardScaler().fit(F_tr)
        F_tr_s = scaler.transform(F_tr).astype(np.float32)
        F_val_s = scaler.transform(F_val).astype(np.float32)
        del F_tr, F_val
        gc.collect()
        log.info(f"  scaled train {F_tr_s.shape} val {F_val_s.shape}")

        # sklearn LinearSVC — primal (dual=False) since n >> d in most cells.
        # sklearn 1.8 default is dual="auto" but we force it explicitly.
        t0 = time.time()
        log.info(f"  fitting sklearn LinearSVC(C={C}, dual=False, loss='squared_hinge') ...")
        clf = LinearSVC(
            C=C,
            loss="squared_hinge",
            penalty="l2",
            dual=False,
            max_iter=3000,
            tol=1e-4,
        )
        clf.fit(F_tr_s, y_tr)
        fit_dt = time.time() - t0
        val_preds = clf.predict(F_val_s)
        val_acc = float(accuracy_score(y_val, val_preds))
        delta = val_acc - cuml_val
        mark = " ⬆" if val_acc > cuml_val else ""
        log.info(f"  sklearn val={val_acc:.4f} (cuML was {cuml_val:.4f}, "
                 f"Δ {delta:+.4f}{mark}, fit {fit_dt:.1f}s)")

        results.append(dict(
            P=patch, K=K, C=C, tag=tag,
            cuml_val=cuml_val, sklearn_val=val_acc, delta=delta,
            fit_seconds=fit_dt,
        ))

        if not args.skip_refit:
            with Timer(log, f"  full 50k refit {tag}"):
                scaler_full = StandardScaler().fit(F_full_tr)
                F_full_s = scaler_full.transform(F_full_tr).astype(np.float32)
                F_te_s = scaler_full.transform(F_test).astype(np.float32)
                final_clf = LinearSVC(
                    C=C,
                    loss="squared_hinge",
                    penalty="l2",
                    dual=False,
                    max_iter=3000,
                    tol=1e-5,
                )
                final_clf.fit(F_full_s, y)
                preds = final_clf.predict(F_te_s)
            sub_path = SUB_DIR / f"sub_run09_sklearn_{tag}.csv"
            save_submission(test_names, preds, sub_path)
            log.info(f"  wrote {sub_path.name}")
            del F_full_s, F_te_s, scaler_full, final_clf
            gc.collect()

        del F_full_tr, F_test, F_tr_s, F_val_s, scaler
        gc.collect()

    log.info("\n=== Summary ===")
    for r in results:
        log.info(f"  {r['tag']}: sklearn={r['sklearn_val']:.4f}  "
                 f"cuML={r['cuml_val']:.4f}  Δ={r['delta']:+.4f}  "
                 f"({r['fit_seconds']:.1f}s)")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
