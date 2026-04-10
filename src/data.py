"""Shared data loading, image I/O and submission utilities for COMP3314 Assignment 3."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
TRAIN_IMS = DATA_DIR / "train_ims"
TEST_IMS = DATA_DIR / "test_ims"
LOG_DIR = PROJECT_ROOT / "logs"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
SUB_DIR = PROJECT_ROOT / "submissions"
CACHE_DIR = PROJECT_ROOT / "cache"
for _d in (LOG_DIR, CKPT_DIR, SUB_DIR, CACHE_DIR):
    _d.mkdir(exist_ok=True)

IMG_H = IMG_W = 32
IMG_C = 3
NUM_CLASSES = 10


def _load_one(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_batch(paths: list[str]) -> np.ndarray:
    return np.stack([_load_one(p) for p in paths], axis=0)


def load_images(names: Iterable[str], ims_dir: Path, n_jobs: int = 8) -> np.ndarray:
    """Load a list of image names into a (N, H, W, 3) uint8 array using parallel workers."""
    names = list(names)
    paths = [str(ims_dir / n) for n in names]
    # chunk to amortize worker startup
    chunk = max(200, len(paths) // (n_jobs * 4) + 1)
    batches = [paths[i : i + chunk] for i in range(0, len(paths), chunk)]
    arrays = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_load_batch)(b) for b in batches
    )
    return np.concatenate(arrays, axis=0)


def load_train(subset: int | None = None, random_state: int = 0, n_jobs: int = 8):
    """Return (X, y, names) for training set.

    X shape: (N, 32, 32, 3) uint8
    y shape: (N,) int64
    """
    df = pd.read_csv(TRAIN_CSV)
    if subset is not None and subset < len(df):
        df = df.sample(n=subset, random_state=random_state).reset_index(drop=True)
    X = load_images(df["im_name"].tolist(), TRAIN_IMS, n_jobs=n_jobs)
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y, df["im_name"].tolist()


def load_test(subset: int | None = None, n_jobs: int = 8):
    """Return (X, names) for test set (test.csv label column is dummy zeros)."""
    df = pd.read_csv(TEST_CSV)
    if subset is not None and subset < len(df):
        df = df.iloc[:subset].reset_index(drop=True)
    X = load_images(df["im_name"].tolist(), TEST_IMS, n_jobs=n_jobs)
    return X, df["im_name"].tolist()


def load_train_cached(n_jobs: int = 8) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load full train set; cache as .npy after first call for ~5-10x speedup."""
    xcache = CACHE_DIR / "train_X.npy"
    ycache = CACHE_DIR / "train_y.npy"
    ncache = CACHE_DIR / "train_names.npy"
    if xcache.exists() and ycache.exists() and ncache.exists():
        X = np.load(xcache)
        y = np.load(ycache)
        names = np.load(ncache, allow_pickle=True).tolist()
        return X, y, names
    X, y, names = load_train(n_jobs=n_jobs)
    np.save(xcache, X)
    np.save(ycache, y)
    np.save(ncache, np.array(names, dtype=object))
    return X, y, names


def load_test_cached(n_jobs: int = 8) -> tuple[np.ndarray, list[str]]:
    xcache = CACHE_DIR / "test_X.npy"
    ncache = CACHE_DIR / "test_names.npy"
    if xcache.exists() and ncache.exists():
        X = np.load(xcache)
        names = np.load(ncache, allow_pickle=True).tolist()
        return X, names
    X, names = load_test(n_jobs=n_jobs)
    np.save(xcache, X)
    np.save(ncache, np.array(names, dtype=object))
    return X, names


def save_submission(names: list[str], preds: np.ndarray, path: str | Path) -> None:
    """Write Kaggle submission csv with columns im_name,label."""
    assert len(names) == len(preds), f"{len(names)} vs {len(preds)}"
    df = pd.DataFrame({"im_name": names, "label": preds.astype(int)})
    df.to_csv(path, index=False)


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Logger that prints to stdout and optionally a file, with timestamps."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class Timer:
    """Context manager that logs wall time of a block."""
    def __init__(self, logger: logging.Logger, label: str):
        self.logger = logger
        self.label = label
    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"[START] {self.label}")
        return self
    def __exit__(self, *args):
        dt = time.time() - self.t0
        self.logger.info(f"[END]   {self.label}  ({dt:.1f}s)")
