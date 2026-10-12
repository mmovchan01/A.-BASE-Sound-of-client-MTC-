#!/usr/bin/env python3
"""Run a saved emotion classifier and create the contest submission NPZ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from train import CLASSES, FORMAT_VERSION, _is_lfs_pointer


def load_embeddings(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл с эмбеддингами: {path}")
    if _is_lfs_pointer(path):
        raise RuntimeError(
            f"{path} содержит Git LFS pointer вместо данных. Сначала скачайте настоящий файл."
        )
    with np.load(path, allow_pickle=False) as data:
        if "embeddings" not in data.files:
            raise ValueError(f"В {path} отсутствует поле embeddings")
        X = np.asarray(data["embeddings"], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Ожидался двумерный embeddings, получена форма {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("В embeddings обнаружены NaN или бесконечные значения")
    return X


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("public_test.npz"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/emotion_mlp.joblib"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size должен быть положительным")
    if not args.model.is_file():
        raise FileNotFoundError(f"Не найдены веса модели: {args.model}. Сначала запустите train.py")

    artifact = joblib.load(args.model)
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError("Неподдерживаемый формат файла модели")
    classes = tuple(artifact.get("classes", ()))
    if classes != CLASSES:
        raise ValueError(f"Веса содержат неожиданный порядок/набор классов: {classes}")

    X = load_embeddings(args.input)
    if X.shape[1] != int(artifact["input_dim"]):
        raise ValueError(
            f"Размер эмбеддинга модели {artifact['input_dim']}, "
            f"а в {args.input} получено {X.shape[1]} признаков"
        )
    mean = np.asarray(artifact["mean"], dtype=np.float32)
    scale = np.asarray(artifact["scale"], dtype=np.float32)
    class_bias = np.asarray(artifact["class_bias"], dtype=np.float32)
    if mean.shape != (X.shape[1],) or scale.shape != mean.shape:
        raise ValueError("Некорректные mean/scale в файле модели")
    if class_bias.shape != (len(classes),):
        raise ValueError("Некорректные class_bias в файле модели")

    model = artifact["model"]
    predictions = np.empty(len(X), dtype=np.int64)
    with threadpool_limits(limits=max(1, args.threads)):
        for start in range(0, len(X), args.batch_size):
            stop = min(start + args.batch_size, len(X))
            batch = np.array(X[start:stop], dtype=np.float32, copy=True)
            batch -= mean
            batch /= scale
            probabilities = model.predict_proba(batch)
            log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
            predictions[start:stop] = np.argmax(log_probabilities + class_bias, axis=1)

    targets = np.asarray(classes, dtype=f"<U{max(map(len, classes))}")[predictions]
    output = args.output or Path(f"submission_seed_{int(artifact['seed'])}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, embeddings=X, targets=targets)

    counts = {name: int(np.sum(targets == name)) for name in classes}
    print(f"Предсказано записей: {len(targets)}")
    print("Распределение классов:", counts)
    print(f"Submission сохранён: {output}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
