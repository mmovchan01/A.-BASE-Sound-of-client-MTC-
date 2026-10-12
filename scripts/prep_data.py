"""Data preparation / reconnaissance for the MTC "Voice of the client" task.

Runs on a machine that has train.npz available (GitHub Actions runner uses Git LFS).
Produces:
  reports/stats.json        - dataset statistics, class counts, duplicate/leak analysis
  reports/devdata.npz       - small local development subset (uint8 quantised + float32 head)
  reports/leak_pairs.npz    - indices of exact train<->public_test matches (if any)
"""
import hashlib
import json
import os
import time

import numpy as np

t0 = time.time()
os.makedirs("reports", exist_ok=True)
rng = np.random.default_rng(12345)

d = np.load("train.npz", mmap_mode="r")
X = d["embeddings"]
y = d["targets"]
N, D = X.shape
print(f"train: {N} x {D} {X.dtype}, labels {y.dtype}", flush=True)

labels, counts = np.unique(y, return_counts=True)
classes = [str(c) for c in labels]
stats = {
    "n_train": int(N),
    "dim": int(D),
    "classes": classes,
    "class_counts": {str(c): int(n) for c, n in zip(labels, counts)},
    "n_public_test": int(np.load("public_test.npz")["embeddings"].shape[0]),
}

# ---- per-dim stats, chunked -------------------------------------------------
n_chunks = 40
idx = np.array_split(np.arange(N), n_chunks)
sum_ = np.zeros(D, dtype=np.float64)
sumsq = np.zeros(D, dtype=np.float64)
mn = np.full(D, np.inf)
mx = np.full(D, -np.inf)
for c in idx:
    xb = np.asarray(X[c[0]:c[-1] + 1], dtype=np.float32)
    sum_ += xb.sum(0, dtype=np.float64)
    sumsq += (xb.astype(np.float64) ** 2).sum(0)
    mn = np.minimum(mn, xb.min(0))
    mx = np.maximum(mx, xb.max(0))
mean = sum_ / N
std = np.sqrt(np.maximum(sumsq / N - mean ** 2, 0))
stats["global"] = {"mean_abs": float(np.abs(mean).mean()), "std_mean": float(std.mean())}
hn = []
for c in idx[:8]:
    xb = np.asarray(X[c[0]:c[-1] + 1], dtype=np.float32)
    hn.append(np.concatenate([np.linalg.norm(xb[:, :D // 2], axis=1), np.linalg.norm(xb[:, D // 2:], axis=1)]))
hn = np.concatenate(hn)
stats["global"]["half_norms_mean"] = float(hn.mean())
stats["global"]["half_norms_std"] = float(hn.std())
print("norms", stats["global"]["half_norms_mean"], stats["global"]["half_norms_std"], "| %.1fs" % (time.time() - t0), flush=True)


def hash_rows(arr, chunk=4000):
    h = np.empty(len(arr), dtype=np.uint64)
    for i in range(0, len(arr), chunk):
        sub = np.ascontiguousarray(np.asarray(arr[i:i + chunk], dtype=np.float32))
        for j, r in enumerate(sub):
            digest = hashlib.blake2b(r.tobytes(), digest_size=8).digest()
            h[i + j] = int.from_bytes(digest, "little")
    return h


t0 = time.time()
h_tr = hash_rows(X)
print("hashed train in %.1fs" % (time.time() - t0), flush=True)
uniq, inv, cnt = np.unique(h_tr, return_inverse=True, return_counts=True)
stats["exact_duplicate_rows"] = int(N - len(uniq))
stats["dup_group_sizes"] = {int(k): int(v) for k, v in zip(*np.unique(cnt, return_counts=True))}
dup_mask = np.isin(inv, np.nonzero(cnt > 1)[0])
inv_dup = inv[dup_mask]
lab_dup = y[dup_mask]
consistent = 0
groups = 0
for g in np.unique(inv_dup):
    sel = lab_dup[inv_dup == g]
    groups += 1
    if len(np.unique(sel)) == 1:
        consistent += 1
stats["dup_groups"] = int(groups)
stats["dup_groups_label_consistent"] = int(consistent)

Xt = np.load("public_test.npz", mmap_mode="r")["embeddings"]
h_te = hash_rows(Xt)
common, tr_i, te_i = np.intersect1d(h_tr, h_te, return_indices=True)
stats["train_test_exact_matches"] = int(len(common))
np.savez_compressed("reports/leak_pairs.npz", train_idx=tr_i.astype(np.int32), test_idx=te_i.astype(np.int32))
print("dup analysis done %.1fs" % (time.time() - t0), flush=True)

# ---- local dev subset -------------------------------------------------------
n_dev, n_head = 16000, 1500
per_class = {}
for c in classes:
    ci = np.nonzero(y == c)[0]
    per_class[c] = rng.choice(ci, size=int(round(n_dev * counts[classes.index(c)] / N)), replace=False)
dev_idx = np.concatenate([per_class[c] for c in classes])
rng.shuffle(dev_idx)
dev_idx = np.sort(dev_idx)

qmin, qmax = mn.astype(np.float32), mx.astype(np.float32)
scale = np.where(qmax > qmin, 255.0 / (qmax - qmin), 0.0).astype(np.float32)

Xd = np.asarray(X[dev_idx], dtype=np.float32)
Q = np.clip(np.rint((Xd - qmin) * scale), 0, 255).astype(np.uint8)
rec = Q.astype(np.float32) / np.where(scale > 0, scale, 1.0) + qmin
err = np.abs(rec - Xd)
print("quantisation max err %.5f mean err %.6f" % (err.max(), err.mean()), flush=True)
stats["quant"] = {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean())}

np.savez_compressed(
    "reports/devdata.npz",
    q=Q, y=y[dev_idx], qmin=qmin, scale=scale, idx=dev_idx.astype(np.int32),
    head=X[dev_idx[:n_head]].astype(np.float32), head_y=y[dev_idx[:n_head]],
)
del Xd, Q, rec, err
print("devdata written %.1fs" % (time.time() - t0), flush=True)

json.dump(stats, open("reports/stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2)[:4000], flush=True)
print("total %.1fs" % (time.time() - t0), flush=True)
