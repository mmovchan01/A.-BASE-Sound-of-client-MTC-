#!/usr/bin/env python3
"""MTC "Voice of the client" - inference / submission builder (deliverable #2).

Loads the weights produced by ``train.py`` (no training happens here), runs the ensemble
on a test npz and writes the submission file:

    python predict.py --weights models/final_seed42 --test public_test.npz \
                      --out submission_seed_42.npz

The output npz mirrors the format of ``sample_submission.npz``: keys ``embeddings``
(the provided test embeddings) and ``targets`` (predicted emotion strings, dtype '<U8').
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtc_lib.common import CLASSES, decode
from mtc_lib.models import EmbeddingMLP, Standardizer, softmax


def load_ensemble(weights_dir: str, device: str = "cpu"):
    with open(os.path.join(weights_dir, "meta.json")) as f:
        meta = json.load(f)
    scaler = Standardizer.from_state_dict({k: np.asarray(v, dtype=np.float32) for k, v in meta["scaler"].items()})
    offsets = np.asarray(meta.get("class_offsets", [0.0] * len(meta["classes"])), dtype=np.float64)
    paths = sorted(glob.glob(os.path.join(weights_dir, "model_*.pt")))
    if not paths:
        raise FileNotFoundError(f"no model_*.pt found in {weights_dir}")
    models = []
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        model = EmbeddingMLP(
            in_dim=meta["architecture"]["in_dim"],
            n_classes=len(meta["classes"]),
            hidden=tuple(cfg["hidden"]),
            dropout=cfg["dropout"],
            input_norm=cfg["input_norm"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)
        models.append(model)
    return models, scaler, offsets, meta


@torch.no_grad()
def predict_proba_ensemble(models, X, batch_size=8192, device="cpu") -> np.ndarray:
    probs = []
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    for model in models:
        chunks = []
        for i in range(0, len(Xt), batch_size):
            logits = model(Xt[i : i + batch_size].to(device))
            chunks.append(softmax(logits.float().cpu().numpy()))
        probs.append(np.concatenate(chunks, axis=0))
    return np.mean(probs, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Predict emotions for a test npz and write a submission")
    ap.add_argument("--weights", default="models/final_seed42", help="directory with model_*.pt / meta.json")
    ap.add_argument("--test", default="public_test.npz", help="test npz containing 'embeddings'")
    ap.add_argument("--out", default=None, help="output submission path (default submission_seed_<SEED>.npz)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    t0 = time.time()
    models, scaler, offsets, meta = load_ensemble(args.weights)
    seed = meta.get("seed", 42)
    out_path = args.out or f"submission_seed_{seed}.npz"

    data = np.load(args.test, allow_pickle=False)
    X = np.asarray(data["embeddings"], dtype=np.float32)
    print(f"[test] {X.shape} embeddings from {args.test}; ensemble of {len(models)} members")

    proba = predict_proba_ensemble(models, scaler.transform(X), batch_size=args.batch_size, device=args.device)
    logits = np.log(np.clip(proba, 1e-12, 1.0)) + offsets
    pred = logits.argmax(1)
    labels = decode(pred)

    np.savez_compressed(out_path, embeddings=X, targets=labels)
    dist = {c: int((labels == c).sum()) for c in CLASSES}
    print(f"[out] {out_path}  label distribution: {dist}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
