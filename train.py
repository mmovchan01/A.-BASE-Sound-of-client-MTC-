#!/usr/bin/env python3
"""MTC "Voice of the client" - full training pipeline (deliverable #1).

Trains the final model (a small MLP ensemble over frozen audio embeddings) from
``train.npz`` and stores everything needed for inference in a weight directory:

    <out>/model_0.pt ... model_{K-1}.pt   - PyTorch state dicts + scaler + config + offsets
    <out>/meta.json                       - architecture, class order, calibration, metrics

Reproduction: pass the recorded --seed and final_config.json. Without --seed a
new random seed is generated and saved in model metadata. The validation split is
used for checkpoint selection and decision calibration; reported F1 is a development
estimate, not an unbiased test score. predict.py performs inference only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtc_lib.common import CLASSES, confusion, encode, macro_f1, stratified_split
from mtc_lib.models import Standardizer, best_offsets, predict_proba, train_mlp

# ---------------------------------------------------------------------------------
# Final configuration, chosen by the offline experiments reported in README.md
# ---------------------------------------------------------------------------------
FINAL_CONFIG = dict(
    hidden=(512, 256),
    dropout=0.3,
    input_norm="none",
    epochs=30,
    batch_size=1024,
    lr=2e-3,
    weight_decay=1e-4,
    label_smoothing=0.05,
    class_weight_mode="balanced",
    mixup_alpha=0.0,
    ema_decay=0.0,
)
ENSEMBLE_SIZE = 2
HOLDOUT_FRAC = 0.05  # only used for calibration/reporting; the models train on the rest


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description="Train the final emotion-classification model")
    ap.add_argument("--data", default="train.npz", help="path to the training npz (embeddings, targets)")
    ap.add_argument("--out", default=None, help="output directory for weights (default models/final_seed<SEED>)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default="final_config.json")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ensemble", type=int, default=ENSEMBLE_SIZE)
    ap.add_argument("--holdout", type=float, default=HOLDOUT_FRAC)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**31)
    if not 0 < args.holdout < 1 or args.ensemble < 1:
        ap.error("holdout must be in (0,1) and ensemble >= 1")
    torch.use_deterministic_algorithms(True)

    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = args.out or f"models/final_seed{args.seed}"
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ValueError("Output directory must be empty to prevent mixing checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)

    t0 = time.time()
    d = np.load(args.data, allow_pickle=False)
    X_all = np.asarray(d["embeddings"], dtype=np.float32)
    y_all = encode(d["targets"])
    if X_all.ndim != 2 or len(X_all) != len(y_all) or not np.isfinite(X_all).all():
        raise ValueError("Invalid training embeddings")
    if set(y_all) != set(range(4)):
        raise ValueError("Training data must contain all four classes")
    print(f"[data] {X_all.shape} embeddings, classes: {dict(zip(*np.unique(d['targets'], return_counts=True)))}")

    tr_idx, ho_idx = stratified_split(y_all, args.holdout, seed=args.seed)
    X_tr_raw, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_ho_raw, y_ho = X_all[ho_idx], y_all[ho_idx]

    scaler = Standardizer().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_ho = scaler.transform(X_ho_raw)

    cfg = dict(FINAL_CONFIG)
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(json.load(f).get("training", {}))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    members, val_probas = [], []
    for k in range(args.ensemble):
        kwargs = dict(cfg)
        model, hist = train_mlp(
            X_tr, y_tr, X_ho, y_ho,
            seed=args.seed + 1000 * k,
            device=args.device,
            log_prefix=f"[member {k}] ",
            **kwargs,
        )
        P = predict_proba(model, X_ho, device=args.device)
        f1 = macro_f1(y_ho, P.argmax(1))
        print(f"[member {k}] params={model.n_params():,} holdout macro-F1={f1:.4f} (best epoch {hist['best_epoch']})")
        members.append(model)
        val_probas.append(P)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "cfg": cfg,
                "member": k,
                "seed": args.seed + 1000 * k,
                "val_f1": f1,
            },
            os.path.join(out_dir, f"model_{k}.pt"),
        )

    P_ens = np.mean(val_probas, axis=0)
    f1_ens = macro_f1(y_ho, P_ens.argmax(1))
    offsets, f1_cal = best_offsets(P_ens, y_ho)
    print(f"[ensemble] holdout macro-F1={f1_ens:.4f}  with class-offset calibration={f1_cal:.4f}")
    print("[confusion]\n" + str(confusion(y_ho, (np.log(np.clip(P_ens, 1e-12, 1)) + offsets).argmax(1))))

    meta = {
        "classes": CLASSES,
        "architecture": {
            "type": "EmbeddingMLP",
            "in_dim": int(X_all.shape[1]),
            "hidden": list(cfg["hidden"]),
            "dropout": cfg["dropout"],
            "input_norm": cfg["input_norm"],
            "n_params_per_member": int(members[0].n_params()),
        },
        "training": cfg,
        "seed": args.seed,
        "ensemble_size": args.ensemble,
        "holdout_frac": args.holdout,
        "holdout_macro_f1": f1_ens,
        "holdout_macro_f1_calibrated": f1_cal,
        "class_offsets": offsets.tolist(),
        "scaler": {k: v.tolist() for k, v in scaler.state_dict().items()},
        "train_seconds": round(time.time() - t0, 1),
        "library_versions": {"torch": torch.__version__, "numpy": np.__version__},
        "data": {"path": os.path.basename(args.data), "n_train": int(len(tr_idx)), "n_holdout": int(len(ho_idx))},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] weights written to {out_dir} in {meta['train_seconds']}s")


if __name__ == "__main__":
    main()
