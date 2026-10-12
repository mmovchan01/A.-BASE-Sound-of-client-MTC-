"""Small shared utilities (numpy only, no heavy deps at import time)."""
from __future__ import annotations

import numpy as np

# canonical class order used everywhere (labels are stored as <U8 strings in the npz files)
CLASSES = ["angry", "neutral", "positive", "sad"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEED_DEFAULT = 42


def load_npz_embeddings(path, key="embeddings", mmap=True):
    """Open an npz file produced for the competition and return the embedding matrix."""
    data = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if key not in data:
        raise KeyError(f"{path} has no '{key}' key, found {list(data.keys())}")
    return data[key]


def encode(y) -> np.ndarray:
    """string labels -> int64 class indices following CLASSES order."""
    y = np.asarray(y)
    if y.dtype.kind in "iu":
        return y.astype(np.int64)
    y = y.astype(str)
    out = np.full(len(y), -1, dtype=np.int64)
    for c, i in CLASS_TO_IDX.items():
        out[y == c] = i
    if (out < 0).any():
        bad = np.unique(y[out < 0])[:5]
        raise ValueError(f"unknown labels: {bad}")
    return out


def decode(idx) -> np.ndarray:
    """int class indices -> string labels (dtype matching the provided npz files: <U8)."""
    return np.asarray(CLASSES, dtype="<U8")[np.asarray(idx, dtype=np.int64)]


def macro_f1(y_true, y_pred, n_classes: int = 4) -> float:
    """macro F1 computed with plain numpy (no sklearn dependency)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    f1s = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def confusion(y_true, y_pred, n_classes: int = 4) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        m[t, p] += 1
    return m


def stratified_split(y, val_frac: float, seed: int, groups=None):
    """Stratified train/val split. `groups` (optional) keeps grouped rows on one side."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(y), dtype=bool)
    if groups is None:
        for c in np.unique(y):
            idx = np.nonzero(y == c)[0]
            idx = idx[rng.permutation(len(idx))]
            n_val = int(round(val_frac * len(idx)))
            val_mask[idx[:n_val]] = True
    else:
        groups = np.asarray(groups)
        # assign whole groups to val until the requested fraction per class is reached
        order = rng.permutation(np.unique(groups))
        for c in np.unique(y):
            need = int(round(val_frac * int((y == c).sum())))
            got = 0
            for g in order:
                if got >= need:
                    break
                sel = groups == g
                if not (y[sel] == c).any() or val_mask[sel].any():
                    continue
                val_mask[sel] = True
                got += int((y[sel] == c).sum())
    return np.nonzero(~val_mask)[0], np.nonzero(val_mask)[0]
