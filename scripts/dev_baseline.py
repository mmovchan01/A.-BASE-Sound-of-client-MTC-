"""Local development harness: quick baselines on the 16k quantised dev subset.

Usage: python scripts/dev_baseline.py [--epochs N] [--model NAME] ...
This is *not* part of the delivered solution; it exists to iterate fast on ~16k rows
before launching full-data runs (131k rows) on a bigger machine / CI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from mtc_lib.common import CLASSES, encode, macro_f1, stratified_split  # noqa: E402
from mtc_lib.models import Standardizer, train_mlp, predict_proba, best_offsets  # noqa: E402


def load_dev(path="reports/devdata.npz", n=None, seed=0):
    d = np.load(path, allow_pickle=False)
    q, y, qmin, scale = d["q"], d["y"], d["qmin"], d["scale"]
    X = q.astype(np.float32) / np.where(scale > 0, scale, 1.0) + qmin
    y = encode(y)
    if n is not None:
        idx = np.random.default_rng(seed).choice(len(y), size=n, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def run_sklearn(Xtr, ytr, Xva, yva, kind="logreg", C=1.0, tag=""):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score

    t0 = time.time()
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, C=C, class_weight="balanced", n_jobs=-1)
    elif kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance", n_jobs=-1)
    else:
        raise ValueError(kind)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    f1 = f1_score(yva, pred, average="macro")
    print(f"[sk-{kind}{tag}] C={C} macroF1 {f1:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    return f1


def run_mlp(Xtr, ytr, Xva, yva, **kw):
    t0 = time.time()
    kw.setdefault("verbose", False)
    hidden = kw.pop("hidden", (512, 256))
    model, hist = train_mlp(Xtr, ytr, Xva, yva, hidden=hidden, **kw)
    P = predict_proba(model, Xva)
    f1 = macro_f1(yva, P.argmax(1))
    off, f1_off = best_offsets(P, yva)
    print(
        f"[mlp{hidden}] params={model.n_params():,} best_val {hist['best_f1']:.4f} "
        f"final {f1:.4f} +offsets {f1_off:.4f} offsets={np.round(off, 2)} ({time.time() - t0:.1f}s)",
        flush=True,
    )
    return {"f1": f1, "f1_offsets": f1_off, "best": hist["best_f1"], "params": model.n_params()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    X, y = load_dev(n=args.n, seed=args.seed)
    tr, va = stratified_split(y, args.val_frac, seed=42)
    print(f"dev: train {len(tr)} val {len(va)}  dim {X.shape[1]}")

    sc = Standardizer().fit(X[tr])
    Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
    ytr, yva = y[tr], y[va]

    results = {}
    if args.quick:
        results["logreg"] = run_sklearn(Xtr, ytr, Xva, yva, "logreg", C=1.0)
        results["mlp_512_256"] = run_mlp(Xtr, ytr, Xva, yva, epochs=args.epochs, lr=3e-3, seed=args.seed)
    else:
        for C in (0.03, 0.3, 3.0):
            results[f"logreg_C{C}"] = run_sklearn(Xtr, ytr, Xva, yva, "logreg", C=C)
        results["knn15"] = run_sklearn(Xtr, ytr, Xva, yva, "knn")
        results["mlp_512_256"] = run_mlp(Xtr, ytr, Xva, yva, epochs=args.epochs, lr=3e-3, seed=args.seed)
        results["mlp_1024"] = run_mlp(Xtr, ytr, Xva, yva, epochs=args.epochs, lr=3e-3, seed=args.seed, hidden=(1024,))
        results["mlp_lin"] = run_mlp(Xtr, ytr, Xva, yva, epochs=args.epochs, lr=3e-3, seed=args.seed, hidden=())
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
