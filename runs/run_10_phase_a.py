"""Phase A — refined grid for larger K, informed by Phase B s public-score
evidence that lower C generalizes better at the top cells.

Grids:
  P=6 (main): K in {10000,12000,14000,16000}, C in
              {0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004}  (32 cfgs)
  P=5 sanity: K in {10000,12000}, C in {0.001,0.002,0.003}          (6 cfgs)
  P=7 sanity: K in {8000,10000},  C in {0.0015,0.002,0.0025}         (6 cfgs)
  total: 44 configs

We encode features in memory (no disk cache) because K=16000 P=6 alone
weighs ~15 GB and the 50 GB /root/autodl-tmp disk cannot hold all four
K caches at once. Dict caches are still written to disk (small, kept so
future scripts can reuse them).

For each (P, K): we fit all Cs, pick the best val C, and write a submission
CSV refit on the full 50k train. We do NOT upload anything — the user
submits manually once results are in.
"""
from __future__ import annotations

import argparse
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
from run_07_cuml_sweep import encode_images_gpu, get_or_fit_dict

from cuml.svm import LinearSVC

RUN_NAME = "run_10_phase_a_P5P7"
N_PATCHES = 2_000_000
POOL = 2
STRIDE = 1
BATCH_SIZE = 64   # K=16000 encoding needs smaller batches

# (P, [K], [C]) — runs in order
# P=6 grids were stopped after K=10000/12000/14000; see
# logs/run_10_phase_a_P6_partial_results.json for partial results.
# Remaining sweep focuses on P=5 and P=7 only.
GRIDS = [
    (5, [10000, 12000],
        [0.001, 0.002, 0.003]),
    (7, [8000, 10000],
        [0.0015, 0.002, 0.0025]),
]

# Threshold for writing a submission CSV: if the best val for a (P, K) is
# at least this, we do a full-50k refit + save. Use 0.78 to be inclusive —
# disk cost is ~137 KB per sub, so keeping many is fine and lets us pick
# public-diverse candidates later.
SUB_VAL_THRESHOLD = 0.78


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel-c", type=int, default=1,
                    help="future hook; for now we run C values sequentially")
    ap.add_argument("--skip-refit", action="store_true")
    args = ap.parse_args()

    log = get_logger(RUN_NAME, LOG_DIR / f"{RUN_NAME}.log")
    log.info(f"=== {RUN_NAME} started ===")
    total = sum(len(Ks) * len(Cs) for _, Ks, Cs in GRIDS)
    n_cells = sum(len(Ks) for _, Ks, _ in GRIDS)
    log.info(f"GRIDS: {len(GRIDS)} P-groups, {n_cells} (P,K) cells, {total} total fits")
    log.info(f"settings: n_patches={N_PATCHES} pool={POOL} batch_size={BATCH_SIZE}")

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
    subs_written = []

    overall_t0 = time.time()

    for patch, K_list, C_list in GRIDS:
        log.info(f"\n{'=' * 60}")
        log.info(f"P={patch}  K_list={K_list}  C_list={C_list}")
        log.info(f"{'=' * 60}")

        for K in K_list:
            cell_t0 = time.time()
            log.info(f"\n### P={patch} K={K} ###")

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

            best_in_cell = {"C": None, "val": -1.0}
            for C in C_list:
                t = time.time()
                clf = LinearSVC(C=C, loss="squared_hinge", penalty="l2",
                                max_iter=5000, tol=1e-4)
                clf.fit(F_tr_s, y_tr)
                preds = clf.predict(F_val_s)
                if hasattr(preds, "get"):
                    preds = preds.get()
                acc = float(accuracy_score(y_val, preds))
                dt = time.time() - t
                log.info(f"  P={patch} K={K} C={C:.4f}  val={acc:.4f}  ({dt:.1f}s)")
                results.append(dict(P=patch, K=K, C=C, val_acc=acc, fit_sec=dt))
                if acc > best_in_cell["val"]:
                    best_in_cell = {"C": C, "val": acc}

            log.info(f"  -> best for P={patch} K={K}: C={best_in_cell['C']} "
                     f"val={best_in_cell['val']:.4f}  (cell took {time.time()-cell_t0:.1f}s)")

            # Write submission if cell is promising. Because we now know that
            # val-public gap can flip, we save a sub for EVERY cell whose best
            # val >= threshold, not just the new SOTA cells.
            if not args.skip_refit and best_in_cell["val"] >= SUB_VAL_THRESHOLD:
                C_best = best_in_cell["C"]
                with Timer(log, f"  refit full P={patch} K={K} C={C_best}"):
                    scaler_full = StandardScaler().fit(F_full_tr)
                    F_full_s = scaler_full.transform(F_full_tr).astype(np.float32)
                    F_te_s = scaler_full.transform(F_test).astype(np.float32)
                    final_clf = LinearSVC(C=C_best, loss="squared_hinge",
                                          penalty="l2", max_iter=5000, tol=1e-5)
                    final_clf.fit(F_full_s, y)
                    preds = final_clf.predict(F_te_s)
                    if hasattr(preds, "get"):
                        preds = preds.get()
                sub_path = SUB_DIR / f"sub_run10_P{patch}_K{K}_C{C_best}.csv"
                save_submission(test_names, preds, sub_path)
                log.info(f"  wrote {sub_path.name}")
                subs_written.append(dict(P=patch, K=K, C=C_best,
                                         val=best_in_cell["val"],
                                         path=str(sub_path.name)))
                del F_full_s, F_te_s, scaler_full, final_clf
                gc.collect()
            elif best_in_cell["val"] < SUB_VAL_THRESHOLD:
                log.info(f"  cell val {best_in_cell['val']:.4f} < threshold "
                         f"{SUB_VAL_THRESHOLD}, skipping refit")

            # Free all features before moving to next (P, K). Critical because
            # at K=16000 F_full_tr alone is ~12.8 GB.
            del F_full_tr, F_test, F_tr_s, F_val_s, scaler
            gc.collect()

            # Periodic json dump in case we crash mid-sweep
            dump = LOG_DIR / f"{RUN_NAME}_results.json"
            with open(dump, "w") as f:
                json.dump({"results": results, "subs_written": subs_written}, f, indent=2)

    log.info(f"\n=== Phase A done ({time.time()-overall_t0:.0f}s total) ===")
    log.info(f"configs fit: {len(results)}")
    log.info(f"submissions written: {len(subs_written)}")
    log.info(f"\nTop-10 by val:")
    for r in sorted(results, key=lambda r: -r["val_acc"])[:10]:
        log.info(f"  P={r['P']} K={r['K']} C={r['C']:.4f} val={r['val_acc']:.4f}")
    log.info(f"=== {RUN_NAME} done ===")


if __name__ == "__main__":
    main()
