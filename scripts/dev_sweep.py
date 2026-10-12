"""Hypothesis sweep on the local 16k dev subset (fast iteration, noisy but informative).

python scripts/dev_sweep.py --group arch|reg|aug|norm|all
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from mtc_lib.common import encode, macro_f1, stratified_split  # noqa: E402
from mtc_lib.models import Standardizer, train_mlp, predict_proba, best_offsets  # noqa: E402

torch.set_num_threads(2)

BASE = dict(epochs=25, batch_size=512, lr=3e-3, weight_decay=1e-4, dropout=0.3, hidden=(512, 256), seed=0)

GROUPS = {
    "arch": [
        dict(name="lin", hidden=()),
        dict(name="h256", hidden=(256,)),
        dict(name="h512", hidden=(512,)),
        dict(name="h512_256", hidden=(512, 256)),
        dict(name="h1024_512", hidden=(1024, 512)),
        dict(name="h1024_512_256", hidden=(1024, 512, 256)),
        dict(name="h2048", hidden=(2048,)),
        dict(name="h256_128", hidden=(256, 128)),
        dict(name="h512_256_128", hidden=(512, 256, 128)),
    ],
    "reg": [
        dict(name="base"),
        dict(name="lr1e-2", lr=1e-2),
        dict(name="lr1e-3", lr=1e-3),
        dict(name="wd1e-2", weight_decay=1e-2),
        dict(name="wd0", weight_decay=0.0),
        dict(name="dr0", dropout=0.0),
        dict(name="dr0.5", dropout=0.5),
        dict(name="ep60", epochs=60),
        dict(name="bs2048", batch_size=2048),
    ],
    "aug": [
        dict(name="base"),
        dict(name="ls0.05", label_smoothing=0.05),
        dict(name="ls0.1", label_smoothing=0.1),
        dict(name="mixup0.2", mixup_alpha=0.2),
        dict(name="mixup0.4", mixup_alpha=0.4),
        dict(name="noise0.05", input_noise=0.05),
        dict(name="noise0.1", input_noise=0.1),
        dict(name="ema0.999", ema_decay=0.999),
        dict(name="ema0.995", ema_decay=0.995),
    ],
    "cw": [
        dict(name="balanced", class_weight_mode="balanced"),
        dict(name="none", class_weight_mode="none"),
        dict(name="sqrt0.5", class_weight_mode="sqrt", class_weight_power=0.5),
        dict(name="bal+ls0.05", class_weight_mode="balanced", label_smoothing=0.05),
        dict(name="bal+mixup0.2", class_weight_mode="balanced", mixup_alpha=0.2),
    ],
    "norm": [
        dict(name="std", input_norm="none"),
        dict(name="std+layernorm", input_norm="layernorm"),
        dict(name="std+batchnorm", input_norm="batchnorm"),
        dict(name="l2only", input_norm="l2", raw=True),
        dict(name="raw", input_norm="none", raw=True),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="reports/dev_sweep.json")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = np.load("reports/devdata.npz", allow_pickle=False)
    q, y, qmin, scale = d["q"], d["y"], d["qmin"], d["scale"]
    X = q.astype(np.float32) / np.where(scale > 0, scale, 1.0) + qmin
    y = encode(y)
    if args.n:
        idx = np.random.default_rng(0).choice(len(y), size=args.n, replace=False)
        X, y = X[idx], y[idx]
    tr, va = stratified_split(y, args.val_frac, seed=42)
    sc = Standardizer().fit(X[tr])
    Xs = sc.transform(X)
    ytr, yva = y[tr], y[va]
    print(f"dev n={len(y)} train={len(tr)} val={len(va)}", flush=True)

    res = {}
    try:
        res = json.load(open(args.out))
    except Exception:
        pass

    groups = list(GROUPS) if args.group == "all" else args.group.split(",")
    for g in groups:
        for cfg in GROUPS[g]:
            name = f"{g}:{cfg['name']}"
            if name in res:
                continue
            kw = {k: v for k, v in cfg.items() if k != "name"}
            raw = kw.pop("raw", False)
            c = dict(BASE)
            c.update(kw)
            Xtr = Xs[tr] if not raw else X[tr]
            Xva = Xs[va] if not raw else X[va]
            t0 = time.time()
            model, hist = train_mlp(Xtr, ytr, Xva, yva, verbose=False, **c)
            P = predict_proba(model, Xva)
            f1 = macro_f1(yva, P.argmax(1))
            _, f1o = best_offsets(P, yva)
            res[name] = {
                "f1": f1, "f1_offsets": f1o, "best_hist": hist["best_f1"],
                "params": int(model.n_params()), "sec": round(time.time() - t0, 1),
                "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
            }
            print(f"{name:28s} f1 {f1:.4f} +off {f1o:.4f} params {model.n_params():>9,} ({res[name]['sec']}s)", flush=True)
            json.dump(res, open(args.out, "w"), indent=2)

    print("\n=== sorted ===")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["f1_offsets"]):
        print(f"{k:28s} {v['f1']:.4f} {v['f1_offsets']:.4f} {v['params']:>9,}")


if __name__ == "__main__":
    main()
