"""Full-data experiment runner (runs on a machine with train.npz; used by CI matrix jobs).

Examples
--------
python scripts/experiment.py --config '{"name":"mlp_base"}' 
python scripts/experiment.py --config '{"name":"knn16","kind":"knn","k":16}'
python scripts/experiment.py --configs reports/configs_round1.json

Each config produces reports/jobs/<name>.json with validation macro-F1, the confusion
matrix and (optionally) the saved model weights under reports/jobs/<name>.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from mtc_lib.common import CLASSES, confusion, encode, macro_f1, stratified_split  # noqa: E402
from mtc_lib.models import Standardizer, best_offsets, predict_proba, softmax, train_mlp  # noqa: E402


def load_full(train_path="train.npz"):
    d = np.load(train_path, mmap_mode="r", allow_pickle=False)
    return d["embeddings"], encode(d["targets"])


def cosine_knn_scores(Xtr, ytr, Xva, k=16, temp=20.0, n_classes=4, chunk=1024):
    """Cosine kNN classifier returning softmax-ish scores (memory friendly)."""
    Tn = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    out = np.zeros((len(Xva), n_classes), dtype=np.float32)
    for i in range(0, len(Xva), chunk):
        Q = Xva[i : i + chunk]
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        S = Q @ Tn.T  # (chunk, n_train)
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        rows = np.arange(len(Q))[:, None]
        sims = S[rows, idx]
        labs = ytr[idx]
        w = np.exp((sims - sims.max(1, keepdims=True)) * temp)
        for c in range(n_classes):
            out[i : i + chunk, c] = (w * (labs == c)).sum(1)
        out[i : i + chunk] /= out[i : i + chunk].sum(1, keepdims=True)
    return out


def run_config(cfg: dict, X, y, tr, va, out_dir="reports/jobs"):
    os.makedirs(out_dir, exist_ok=True)
    name = cfg["name"]
    out_path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(out_path) and not cfg.get("force"):
        print(f"skip {name} (already done)")
        return json.load(open(out_path))

    kind = cfg.get("kind", "mlp")
    t0 = time.time()
    ytr, yva = y[tr], y[va]
    Xtr_raw, Xva_raw = np.asarray(X[tr], dtype=np.float32), np.asarray(X[va], dtype=np.float32)

    if kind == "knn":
        scores = cosine_knn_scores(Xtr_raw, ytr, Xva_raw, k=cfg.get("k", 16), temp=cfg.get("temp", 20.0))
        res = {
            "name": name, "kind": kind, "cfg": cfg,
            "val_f1": macro_f1(yva, scores.argmax(1)),
        }
        off, f1o = best_offsets(scores, yva)
        res["val_f1_offsets"] = f1o
        res["offsets"] = off.tolist()
        res["confusion"] = confusion(yva, scores.argmax(1)).tolist()
        res["sec"] = round(time.time() - t0, 1)
        json.dump(res, open(out_path, "w"), indent=2)
        print(json.dumps(res, indent=2), flush=True)
        return res

    sc = Standardizer()
    if cfg.get("raw", False):
        Xtr, Xva = Xtr_raw, Xva_raw
    else:
        sc.fit(Xtr_raw)
        Xtr, Xva = sc.transform(Xtr_raw), sc.transform(Xva_raw)

    train_kw = {k: v for k, v in cfg.items() if k not in ("name", "kind", "raw", "save_weights", "force", "matmul_knn")}
    if "hidden" in train_kw:
        train_kw["hidden"] = tuple(train_kw["hidden"])
    model, hist = train_mlp(Xtr, ytr, Xva, yva, verbose=True, log_prefix=f"[{name}] ", **train_kw)
    P = predict_proba(model, Xva)
    res = {
        "name": name, "kind": kind, "cfg": cfg,
        "val_f1": macro_f1(yva, P.argmax(1)),
        "val_f1_top_hist": hist["best_f1"],
        "best_epoch": hist["best_epoch"],
        "params": int(model.n_params()),
        "history": hist["history"],
    }
    off, f1o = best_offsets(P, yva)
    res["val_f1_offsets"] = f1o
    res["offsets"] = off.tolist()
    res["confusion"] = confusion(yva, P.argmax(1)).tolist()
    res["sec"] = round(time.time() - t0, 1)
    if cfg.get("save_weights"):
        import torch

        path = os.path.join(out_dir, f"{name}.pt")
        torch.save({"state_dict": model.state_dict(), "cfg": cfg, "scaler": sc.state_dict()}, path)
        res["weights"] = path
    json.dump(res, open(out_path, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="single config as JSON")
    ap.add_argument("--configs", type=str, default=None, help="path to JSON file with a list of configs")
    ap.add_argument("--train", default="train.npz")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--n", type=int, default=None, help="debug: subsample the training set")
    ap.add_argument("--out-dir", default="reports/jobs")
    args = ap.parse_args()

    if args.configs:
        cfgs = json.load(open(args.configs))
    else:
        cfgs = [json.loads(args.config)]

    X, y = load_full(args.train)
    print("data", X.shape, "classes", np.bincount(y), flush=True)
    idx = np.arange(len(y))
    if args.n:
        idx = np.random.default_rng(0).choice(len(y), size=args.n, replace=False)
    tr, va = stratified_split(y[idx], args.val_frac, seed=args.split_seed)
    tr, va = idx[tr], idx[va]
    print(f"split: train {len(tr)} val {len(va)}", flush=True)

    summary = []
    for cfg in cfgs:
        res = run_config(cfg, X, y, tr, va, out_dir=args.out_dir)
        summary.append({k: res.get(k) for k in ("name", "val_f1", "val_f1_offsets", "params", "sec")})
        json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
