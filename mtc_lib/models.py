"""Model definitions and training / calibration helpers (PyTorch, CPU or CUDA)."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import macro_f1

# --------------------------------------------------------------------------------------
# Feature transforms
# --------------------------------------------------------------------------------------


class Standardizer:
    """Per-dimension standardization fitted on the training split (2048 floats saved with the model)."""

    def __init__(self, mean=None, std=None, eps: float = 1e-6):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.eps = eps

    def fit(self, X, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        if sample_weight is None:
            mean = X.mean(0)
            var = X.var(0)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            w = w / w.sum()
            mean = (X * w[:, None]).sum(0)
            var = (((X - mean) ** 2) * w[:, None]).sum(0)
        self.mean = mean.astype(np.float32)
        self.std = np.sqrt(np.maximum(var, 0) + self.eps).astype(np.float32)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X, sample_weight=None):
        return self.fit(X, sample_weight).transform(X)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state):
        return cls(mean=state["mean"], std=state["std"])


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class EmbeddingMLP(nn.Module):
    """MLP head on frozen audio embeddings.

    Architecture: optional input normalisation -> [Linear -> Act -> Dropout] * n_hidden -> Linear -> n_classes.
    Small on purpose (the competition rewards compact models at equal quality).
    """

    def __init__(
        self,
        in_dim: int = 2048,
        n_classes: int = 4,
        hidden=(512, 256),
        dropout: float = 0.3,
        input_norm: str = "none",  # one of none|layernorm|batchnorm|l2
        act: str = "gelu",
        dropout_last: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.input_norm = input_norm

        self.input_norm_layer = {
            "none": nn.Identity,
            "layernorm": lambda: nn.LayerNorm(in_dim),
            "batchnorm": lambda: nn.BatchNorm1d(in_dim),
        }[input_norm if input_norm in ("none", "layernorm", "batchnorm") else "none"]()

        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[act]
        layers, prev = [], in_dim
        for i, h in enumerate(self.hidden):
            layers += [nn.Linear(prev, h, bias=bias), act_cls()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        if dropout_last and dropout > 0:
            layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(prev, n_classes, bias=bias)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_norm == "l2":
            x = F.normalize(x, dim=-1)
        x = self.input_norm_layer(x)
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def _class_weights(y: np.ndarray, mode: str = "balanced", n_classes: int = 4, power: float = 1.0):
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "balanced":
        w = counts.sum() / (n_classes * counts)
    elif mode == "sqrt":
        w = (counts.sum() / (n_classes * counts)) ** power
    else:
        raise ValueError(mode)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    model: EmbeddingMLP | None = None,
    hidden=(512, 256),
    dropout: float = 0.3,
    input_norm: str = "none",
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    warmup_frac: float = 0.05,
    label_smoothing: float = 0.0,
    class_weight_mode: str = "balanced",
    class_weight_power: float = 1.0,
    mixup_alpha: float = 0.0,
    input_noise: float = 0.0,
    ema_decay: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    log_prefix: str = "",
    verbose: bool = True,
    eval_every: int = 1,
):
    """Train a classification head. Returns (model, history). `model`/`X_*` are torch-free inputs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_val = np.ascontiguousarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    dev = torch.device(device)
    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)
    Xva = torch.from_numpy(X_val).to(dev)
    yva = torch.from_numpy(y_val)

    if model is None:
        model = EmbeddingMLP(
            in_dim=X_train.shape[1], hidden=hidden, dropout=dropout, input_norm=input_norm
        )
    model = model.to(dev)

    cw = _class_weights(y_train, class_weight_mode, power=class_weight_power)
    if cw is not None:
        cw = cw.to(dev)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(y_train)
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ema_state = None
    if ema_decay > 0:
        ema_state = copy.deepcopy(model.state_dict())

    rng = np.random.default_rng(seed + 1)
    history = []
    best = {"f1": -1.0, "epoch": -1, "state": None}
    step = 0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        tot_loss = 0.0
        for i in range(steps_per_epoch):
            sel = perm[i * batch_size : (i + 1) * batch_size]
            xb = Xtr[sel].to(dev)
            yb = ytr[sel].to(dev)
            if input_noise > 0:
                xb = xb + input_noise * torch.randn_like(xb)
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(len(sel), device=dev)
                xb = lam * xb + (1 - lam) * xb[idx]
                out = model(xb)
                loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb[idx])
            else:
                out = model(xb)
                loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot_loss += float(loss) * len(sel)
            if ema_state is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
                        else:
                            ema_state[k].copy_(v)

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(Xva)
                va_f1 = macro_f1(y_val, logits.argmax(1).cpu().numpy())
            if ema_state is not None:
                ema_model = copy.deepcopy(model)
                ema_model.load_state_dict(ema_state)
                ema_model.eval()
                with torch.no_grad():
                    ema_f1 = macro_f1(y_val, ema_model(Xva).argmax(1).cpu().numpy())
            else:
                ema_f1 = float("nan")
            history.append({"epoch": epoch, "loss": tot_loss / n, "val_f1": va_f1, "val_f1_ema": ema_f1})
            score = max(va_f1, ema_f1 if ema_f1 == ema_f1 else -1)
            if score > best["f1"]:
                best = {
                    "f1": score,
                    "epoch": epoch,
                    "state": copy.deepcopy(ema_state if (ema_f1 == score and ema_state is not None) else model.state_dict()),
                    "which": "ema" if (ema_f1 == score and ema_state is not None) else "raw",
                }
            if verbose:
                print(
                    f"{log_prefix}epoch {epoch + 1:3d}/{epochs} loss {tot_loss / n:.4f} "
                    f"val_f1 {va_f1:.4f} ema {ema_f1:.4f}",
                    flush=True,
                )

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    return model, {"history": history, "best_f1": best["f1"], "best_epoch": best["epoch"]}


# --------------------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------------------


@torch.no_grad()
def predict_logits(model, X, batch_size: int = 8192, device: str = "cpu") -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size], dtype=np.float32)).to(dev)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_proba(model, X, **kw) -> np.ndarray:
    logits = predict_logits(model, X, **kw)
    return softmax(logits)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# Decision-rule calibration for macro-F1
# --------------------------------------------------------------------------------------


def best_offsets(proba: np.ndarray, y_true, grid=None, n_rounds: int = 3, seed: int = 0):
    """Coordinate search for per-class logit offsets maximising macro-F1 on held-out data.

    Because the metric is macro-F1 over imbalanced classes, shifting each class logit is a
    cheap, principled post-processing step (equivalent to tuning class priors).
    """
    proba = np.asarray(proba, dtype=np.float64)
    logp = np.log(np.clip(proba, 1e-12, 1.0))
    y_true = np.asarray(y_true)
    if grid is None:
        grid = np.linspace(-2.5, 2.5, 51)

    offsets = np.zeros(logp.shape[1])
    best = macro_f1(y_true, (logp + offsets).argmax(1))
    rng = np.random.default_rng(seed)
    for _ in range(n_rounds):
        improved = False
        for c in rng.permutation(logp.shape[1]):
            cur = offsets[c]
            cand_scores = []
            for g in grid:
                offsets[c] = g
                cand_scores.append(macro_f1(y_true, (logp + offsets).argmax(1)))
            j = int(np.argmax(cand_scores))
            if cand_scores[j] > best + 1e-9:
                best = float(cand_scores[j])
                offsets[c] = grid[j]
                improved = True
            else:
                offsets[c] = cur
        if not improved:
            break
    return offsets, float(best)
