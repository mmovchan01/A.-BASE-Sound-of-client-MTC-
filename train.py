#!/usr/bin/env python3
"""Train a compact MLP emotion classifier from the supplied embedding NPZ."""

from __future__ import annotations

import argparse
import copy
import secrets
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CLASSES = ("angry", "neutral", "positive", "sad")
FORMAT_VERSION = 1


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл обучающей выборки: {path}")
    if _is_lfs_pointer(path):
        raise RuntimeError(
            f"{path} содержит только Git LFS pointer, а не данные. "
            "Скачайте объект командой `git lfs pull` (или положите настоящий train.npz по этому пути)."
        )

    with np.load(path, allow_pickle=False) as data:
        missing = {"embeddings", "targets"}.difference(data.files)
        if missing:
            raise ValueError(f"В {path} отсутствуют поля: {sorted(missing)}")
        X = np.asarray(data["embeddings"], dtype=np.float32)
        raw_y = np.asarray(data["targets"])

    if X.ndim != 2:
        raise ValueError(f"Ожидался двумерный embeddings, получена форма {X.shape}")
    if raw_y.ndim != 1 or len(raw_y) != len(X):
        raise ValueError(f"targets должны иметь форму ({len(X)},), получена {raw_y.shape}")
    if not np.isfinite(X).all():
        raise ValueError("В embeddings обнаружены NaN или бесконечные значения")

    if raw_y.dtype.kind == "S":
        y = np.char.decode(raw_y, encoding="utf-8")
    elif raw_y.dtype.kind in "U":
        y = raw_y.astype(str, copy=False)
    else:
        raise ValueError(
            f"Ожидались строковые названия классов в targets, получен dtype {raw_y.dtype}"
        )
    y = np.char.strip(y)

    unknown = sorted(set(y.tolist()).difference(CLASSES))
    missing_classes = sorted(set(CLASSES).difference(y.tolist()))
    if unknown:
        raise ValueError(f"Неизвестные метки: {unknown}. Допустимые классы: {CLASSES}")
    if missing_classes:
        raise ValueError(f"В обучающей выборке отсутствуют классы: {missing_classes}")
    return X, y


def fit_scaler_in_batches(
    X: np.ndarray, rows: np.ndarray, batch_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-feature z-score statistics without copying the full dataset."""
    scaler = StandardScaler(copy=False)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        scaler.partial_fit(X[batch_rows])
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    # StandardScaler already replaces zero standard deviations by 1. Keep an
    # explicit floor to protect prediction on platform-dependent roundoff.
    scale = np.maximum(scale, np.float32(1e-8))
    return mean, scale


def standardize_in_place(
    X: np.ndarray, mean: np.ndarray, scale: np.ndarray, batch_size: int = 4096
) -> None:
    for start in range(0, len(X), batch_size):
        block = X[start : start + batch_size]
        block -= mean
        block /= scale


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def tune_class_biases(log_probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Tune four small additive log-probability offsets for macro-F1."""
    n_classes = len(CLASSES)
    biases = np.zeros(n_classes, dtype=np.float64)
    spread = float(np.median(np.ptp(log_probabilities, axis=1)))
    radius = max(0.5, min(8.0, spread / 3.0))

    # Coordinate search is deliberately low-dimensional: it adjusts only one
    # class intercept at a time and uses the held-out split, never test labels.
    for _ in range(3):
        for class_id in range(n_classes):
            candidates = np.linspace(biases[class_id] - radius, biases[class_id] + radius, 25)
            best_bias = biases[class_id]
            best_score = -1.0
            for candidate in candidates:
                trial = biases.copy()
                trial[class_id] = candidate
                prediction = np.argmax(log_probabilities + trial, axis=1)
                score = macro_f1(y_true, prediction)
                if score > best_score + 1e-12 or (
                    abs(score - best_score) <= 1e-12
                    and abs(candidate - biases[class_id]) < abs(best_bias - biases[class_id])
                ):
                    best_score = score
                    best_bias = float(candidate)
            biases[class_id] = best_bias
        # Logit offsets are identifiable only up to a common additive constant.
        biases -= biases.mean()
        radius *= 0.35
    return biases.astype(np.float32)


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Ожидается список размеров, например 512,128") from exc
    if not values or any(width <= 0 for width in values):
        raise argparse.ArgumentTypeError("Размеры скрытых слоёв должны быть положительными")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=Path("train.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/emotion_mlp.joblib"))
    parser.add_argument("--seed", type=int, default=None, help="Если не задан, выбирается случайно")
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-layers", type=parse_hidden_layers, default=(512, 128))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2-регуляризация MLP")
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.75,
        help="Степень обратной частоты класса; 1.0 даёт полностью сбалансированный вес",
    )
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.validation_size <= 0.5:
        raise SystemExit("--validation-size должен быть в диапазоне [0.05, 0.5]")
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise SystemExit("Число эпох, patience и batch-size должны быть положительными")
    if not 0.0 <= args.class_weight_power <= 1.5:
        raise SystemExit("--class-weight-power должен быть в диапазоне [0, 1.5]")

    seed = secrets.randbelow(2**32) if args.seed is None else int(args.seed)
    seed %= 2**32
    print(f"Seed: {seed}")
    started = time.time()

    X, y = load_training_data(args.train_path)
    if len(X) < 20:
        raise ValueError("Слишком мало обучающих примеров для stratified train/validation split")
    y_codes = np.asarray([CLASSES.index(label) for label in y], dtype=np.int64)
    counts = np.bincount(y_codes, minlength=len(CLASSES))
    if np.min(counts) < 2:
        raise ValueError(f"Для каждого класса нужно минимум 2 записи; получено: {dict(zip(CLASSES, counts))}")

    all_rows = np.arange(len(X), dtype=np.int64)
    train_rows, validation_rows = train_test_split(
        all_rows,
        test_size=args.validation_size,
        random_state=seed,
        stratify=y_codes,
    )
    train_counts = np.bincount(y_codes[train_rows], minlength=len(CLASSES)).astype(np.float64)
    class_weights = (len(train_rows) / (len(CLASSES) * train_counts)) ** args.class_weight_power
    class_weights /= np.average(class_weights, weights=train_counts)
    sample_weights = class_weights[y_codes[train_rows]].astype(np.float32)

    print(f"Эмбеддинги: {X.shape}; train={len(train_rows)}, validation={len(validation_rows)}")
    print("Частоты классов:", {name: int(count) for name, count in zip(CLASSES, counts)})
    print("Оценка mean/std признаков...")
    mean, scale = fit_scaler_in_batches(X, train_rows)
    standardize_in_place(X, mean, scale)

    X_train = np.ascontiguousarray(X[train_rows], dtype=np.float32)
    y_train = y_codes[train_rows]
    X_validation = np.ascontiguousarray(X[validation_rows], dtype=np.float32)
    y_validation = y_codes[validation_rows]
    del X, y

    model = MLPClassifier(
        hidden_layer_sizes=args.hidden_layers,
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate="constant",
        learning_rate_init=args.learning_rate,
        max_iter=1,
        shuffle=True,
        random_state=seed,
        tol=0.0,
        early_stopping=False,
    )

    best_model = None
    best_epoch = 0
    best_score = -1.0
    stale_epochs = 0
    classes = np.arange(len(CLASSES), dtype=np.int64)
    print(f"Training MLP {X_train.shape[1]} -> {args.hidden_layers} -> {len(CLASSES)}")
    with threadpool_limits(limits=max(1, args.threads)):
        for epoch in range(1, args.max_epochs + 1):
            model.partial_fit(
                X_train,
                y_train,
                classes=classes,
                sample_weight=sample_weights,
            )
            validation_probabilities = model.predict_proba(X_validation)
            score = macro_f1(y_validation, np.argmax(validation_probabilities, axis=1))
            print(f"epoch {epoch:02d}: validation macro-F1={score:.5f}")

            if score > best_score + 1e-5:
                best_score = score
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"Early stopping: нет улучшения {args.patience} эпох")
                    break

    if best_model is None:
        raise RuntimeError("Не удалось получить обученную модель")

    validation_probabilities = best_model.predict_proba(X_validation)
    log_probabilities = np.log(np.clip(validation_probabilities, 1e-12, 1.0))
    class_bias = tune_class_biases(log_probabilities, y_validation)
    calibrated_prediction = np.argmax(log_probabilities + class_bias, axis=1)
    calibrated_score = macro_f1(y_validation, calibrated_prediction)
    print(f"Лучший epoch: {best_epoch}")
    print(f"Validation macro-F1 (без смещений): {best_score:.5f}")
    print(f"Validation macro-F1 (с калибровкой): {calibrated_score:.5f}")
    print("Смещения классов:", {name: float(bias) for name, bias in zip(CLASSES, class_bias)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format_version": FORMAT_VERSION,
        "model": best_model,
        "classes": list(CLASSES),
        "mean": mean,
        "scale": scale,
        "class_bias": class_bias,
        "input_dim": int(X_train.shape[1]),
        "seed": seed,
        "selected_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "validation_calibrated_macro_f1": calibrated_score,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, args.output, compress=3)
    print(f"Модель сохранена: {args.output}")
    print(f"Время обучения: {(time.time() - started) / 60:.1f} мин")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
