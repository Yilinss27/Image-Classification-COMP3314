"""Phase B: lower-C extension for run_07 winners.

Re-encodes features into memory (no disk save) and runs cuML LinearSVC for
C in {0.0005, 0.001, 0.002} on the (P, K) cells where run_07 had C=0.003 as
the winner.

Dict caches from run_07 are reused (they're small, we kept all of them).
Full-50k refit is also done for each new best so we get a valid submission CSV.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LOG_DIR, SUB_DIR, Timer, get_logger, load_test_cached, load_train_cached, save_submission
from run_07_cuml_sweep import encode_images_gpu, get_or_fit_dict  # reuse helpers

from cuml.svm import LinearSVC

RUN_NAME = "run_08_phase_b"
N_PATCHES = 1_000_000  # match run_07 to reuse dicts
POOL = 2
STRIDE = 1
BATCH_SIZE = 128

# Phase B target cells: (P, K) pairs where C=0.003 won in run_07
TARGETS = [
    # (P, K)
    (4, 8000),
    (5, 4000), (5, 8000),
    (6, 2400), (6, 4000), (6, 6000), (6, 8000),
    (7, 1600), (7, 4000), (7, 6000), (7, 8000),
    (8, 2400), (8, 3200), (8, 4000), (8, 6000), (8, 8000),
]
NEW_C_LIST = [0.0005, 0.001, 0.002]
# Run_07 best vals at C=0.003 for these cells (for on-the-spot comparison)
RUN07_BASELINE = {
    (4, 8000): 0.7686,
    (5, 4000): 0.7728, (5, 8000): 0.7794,
    (6, 2400): 0.7752, (6, 4000): 0.7786, (6, 6000): 0.7780, (6, 8000): 0.7858,
    (7, 1600): 0.7702, (7, 4000): 0.7824, (7, 6000): 0.7810, (7, 8000): 0.7752,
    (8, 2400): 0.7742, (8, 3200): 0.7816, (8, 4000): 0.7702,
    (8, 6000): 0.7750, (8, 8000): 0.7502,
}


def main():
    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")
    log.info(f"{len(TARGETS)} (P,K) targets x {len(NEW_C_LIST)} C = "
             f"{len(TARGETS)*len(NEW_C_LIST)} fits")

    with Timer(log, "load train/test"):
        X, y, _ = load_train_cached(n_jobs=8)
        Xte, test_names = load_test_cached(n_jobs=8)
    log.info(f"train {X.shape}  test {Xte.shape}")

    idx_tr, idx_val = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=0)
    y_tr, y_val = y[idx_tr], y[idx_val]
    log.info(f"train sub {len(idx_tr)}  val {len(idx_val)}")

    rng = np.random.RandomState(0)
    results = []
    improvements = []

    for patch, K in TARGETS:
        baseline = RUN07_BASELINE[(patch, K)]
        log.info(f"\n### P={patch} K={K}  (run_07 C=0.003 baseline = {baseline:.4f}) ###")

        centroids, zca_mean, zca_W = get_or_fit_dict(
            X[idx_tr], K, patch, N_PATCHES, log, rng)

        with Timer(log, f"GPU encode train+test K={K} P={patch}"):
            F_full_tr = encode_images_gpu(
                X, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)
            F_test = encode_images_gpu(
                Xte, centroids, zca_mean, zca_W,
                patch=patch, stride=STRIDE, pool=POOL, batch_size=BATCH_SIZE)

        F_tr = F_full_tr[idx_tr]
        F_val = F_full_tr[idx_val]
        scaler = StandardScaler().fit(F_tr)
        F_tr_s = scaler.transform(F_tr).astype(np.float32)
        F_val_s = scaler.transform(F_val).astype(np.float32)
        del F_tr, F_val
        gc.collect()

        best_local = {"C": None, "acc": baseline, "source": "run07_baseline"}
        for C in NEW_C_LIST:
            t = time.time()
            clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                            max_iter=3000, tol=1e-5)
            clf.fit(F_tr_s, y_tr)
            preds = clf.predict(F_val_s)
            if hasattr(preds, "get"):
                preds = preds.get()
            acc = float(accuracy_score(y_val, preds))
            dt = time.time() - t
            delta = acc - baseline
            mark = " ⬆" if acc > baseline else ""
            log.info(f"  P={patch} K={K} C={C} val={acc:.4f}  "
                     f"(Δ vs C=0.003 baseline {delta:+.4f}{mark}, {dt:.1f}s)")
            results.append(dict(P=patch, K=K, C=C, val_acc=acc, delta=delta))
            if acc > best_local["acc"]:
                best_local = {"C": C, "acc": acc, "source": "phase_b"}

        if best_local["source"] == "phase_b":
            C_best = best_local["C"]
            log.info(f"  🎯 NEW best P={patch} K={K}: C={C_best} "
                     f"val={best_local['acc']:.4f} (beats baseline {baseline:.4f})")
            improvements.append((patch, K, C_best, best_local["acc"], baseline))

            # Refit on full 50k and save submission
            with Timer(log, f"refit full P={patch} K={K} C={C_best}"):
                scaler_full = StandardScaler().fit(F_full_tr)
                F_full_s = scaler_full.transform(F_full_tr).astype(np.float32)
                F_te_s = scaler_full.transform(F_test).astype(np.float32)
                final_clf = LinearSVC(C=C_best, loss="squared_hinge",
                                      penalty="l2", max_iter=3000, tol=1e-6)
                final_clf.fit(F_full_s, y)
                preds = final_clf.predict(F_te_s)
                if hasattr(preds, "get"):
                    preds = preds.get()
            sub_path = SUB_DIR / f"sub_run08_P{patch}_K{K}_C{C_best}.csv"
            save_submission(test_names, preds, sub_path)
            log.info(f"  wrote {sub_path.name}")
            del F_full_s, F_te_s, scaler_full, final_clf
        else:
            log.info(f"  no improvement; all new C values worse than baseline")

        del F_full_tr, F_test, F_tr_s, F_val_s, scaler
        gc.collect()

    # Summary
    log.info("\n=== Phase B complete ===")
    log.info(f"Improvements over run_07 C=0.003 baselines: "
             f"{len(improvements)} / {len(TARGETS)}")
    for patch, K, C, new_val, base in sorted(improvements, key=lambda r: -r[3]):
        log.info(f"  P={patch} K={K} C={C} val={new_val:.4f} "
                 f"(+{new_val-base:+.4f} vs {base:.4f})")

    summary_json = LOG_DIR / f"{RUN_NAME}_results.json"
    with open(summary_json, "w") as f:
        json.dump({"results": results, "improvements":
                   [dict(P=p, K=k, C=c, new_val=v, baseline=b)
                    for p, k, c, v, b in improvements]}, f, indent=2)
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
