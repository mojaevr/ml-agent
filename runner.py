"""
runner.py — оркестратор запуска агентского кода.

Агент НЕ трогает этот файл.

Что делает runner:
  1. Загружает train/val/test данные → создаёт MLDataset
  2. Импортирует train_code.py агента
  3. Вызывает build_features(dataset) — feature engineering
  4. Вызывает build_model(dataset)    — обучение
  5. Считает ROC-AUC на val автоматически
  6. Сохраняет model + dataset (с pipeline) → для predict
  7. При --predict: загружает сохранённое, применяет те же трансформации

Использование:
    python runner.py --mode train  --run_dir runs/run_001
    python runner.py --mode predict --run_dir runs/run_001 --output preds.csv
"""

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml_dataset import MLDataset


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────

def load_module(path: str, module_name: str = "_agent_module"):
    """Загружает python файл как модуль (не через exec — для корректного pickle)."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_dataset(data_dir: str | Path, task_type: str = "binary_classification") -> MLDataset:
    """Загружает train/val/test из data_dir и создаёт MLDataset."""
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv") if (data_dir / "test.csv").exists() else None

    # Определяем target колонку (последняя или "target")
    target_col = _detect_target(train_df)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    X_test = test_df if test_df is not None else None

    return MLDataset(X_train, y_train, X_val, y_val, X_test, task_type=task_type)


def _detect_target(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    return df.columns[-1]


def save_artifacts(run_dir: Path, model, dataset: MLDataset, meta: dict):
    """Сохраняет модель и dataset (с pipeline) для последующего predict."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    dataset.save(run_dir / "dataset.pkl")
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_artifacts(run_dir: Path):
    """Загружает модель и dataset из run_dir."""
    with open(run_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    dataset = MLDataset.load(run_dir / "dataset.pkl")
    return model, dataset


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────

def run_train(run_dir: str, data_dir: str, task_type: str = "binary_classification") -> dict:
    """
    Полный цикл обучения.

    Returns:
        dict с roc_auc, duration_sec, и другой мета-информацией
    """
    run_dir = Path(run_dir)
    train_code_path = run_dir / "train_code.py"

    if not train_code_path.exists():
        raise FileNotFoundError(f"train_code.py не найден в {run_dir}")

    print(f"[runner] Загружаю данные из {data_dir}...")
    dataset = load_dataset(data_dir, task_type=task_type)
    print(f"[runner] Dataset: {dataset}")

    print(f"[runner] Загружаю train_code.py из {run_dir}...")
    agent_mod = load_module(str(train_code_path))

    # ── Feature engineering
    print("[runner] Вызываю build_features()...")
    t0 = time.time()
    try:
        dataset = agent_mod.build_features(dataset)
    except Exception as e:
        raise RuntimeError(f"build_features() упал: {e}") from e

    print(f"[runner] X_train после FE: {dataset.X_train_transformed.shape if hasattr(dataset.X_train_transformed, 'shape') else 'no shape'}")
    print(f"[runner] X_val  после FE: {dataset.X_val_transformed.shape if hasattr(dataset.X_val_transformed, 'shape') else 'no shape'}")

    # ── Обучение
    print("[runner] Вызываю build_model()...")
    try:
        model = agent_mod.build_model(dataset)
    except Exception as e:
        raise RuntimeError(f"build_model() упал: {e}") from e

    duration = time.time() - t0

    # ── Валидация
    print("[runner] Считаю ROC-AUC на val...")
    try:
        preds = model.predict_proba(dataset.X_val_transformed)[:, 1]
        roc_auc = float(roc_auc_score(dataset.y_val, preds))
    except Exception as e:
        raise RuntimeError(f"predict_proba() упал: {e}") from e

    print(f"[runner] ✅ ROC-AUC = {roc_auc:.5f}  ({duration:.1f}s)")

    # ── Сохраняем
    meta = {
        "roc_auc": roc_auc,
        "duration_sec": duration,
        "train_shape": list(dataset.X_train.shape),
        "val_shape": list(dataset.X_val.shape),
        "transformed_train_shape": list(np.array(dataset.X_train_transformed).shape),
        "pipeline_steps": [name for name, _ in dataset._transformers],
        "task_type": task_type,
    }
    save_artifacts(run_dir, model, dataset, meta)
    print(f"[runner] Сохранено в {run_dir}")

    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Predict (для test.csv → сабмит)
# ──────────────────────────────────────────────────────────────────────────────

def run_predict(run_dir: str, output_path: str):
    """
    Предсказание на test.csv с использованием сохранённого pipeline.

    Применяет ровно те же трансформации, что были при обучении.
    """
    run_dir = Path(run_dir)
    model, dataset = load_artifacts(run_dir)

    if dataset.X_test is None:
        raise ValueError("В сохранённом dataset нет X_test")

    print(f"[runner] Применяю pipeline к X_test: {dataset.X_test.shape}...")
    X_test_transformed = dataset.transform_new(dataset.X_test)
    print(f"[runner] X_test_transformed: {np.array(X_test_transformed).shape}")

    preds = model.predict_proba(X_test_transformed)[:, 1]
    submission = pd.DataFrame({"id": range(len(preds)), "target": preds})
    submission.to_csv(output_path, index=False)
    print(f"[runner] ✅ Сабмит сохранён: {output_path}  ({len(preds)} строк)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--run_dir", required=True, help="Директория с train_code.py / артефактами")
    parser.add_argument("--data_dir", default="data", help="Директория с train.csv, val.csv, test.csv")
    parser.add_argument("--output", default="submission.csv", help="Куда записать предсказания (для predict)")
    parser.add_argument("--task_type", default="binary_classification")
    args = parser.parse_args()

    if args.mode == "train":
        try:
            result = run_train(args.run_dir, args.data_dir, args.task_type)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"[runner] ❌ ОШИБКА: {e}")
            traceback.print_exc()
            sys.exit(1)

    elif args.mode == "predict":
        try:
            run_predict(args.run_dir, args.output)
        except Exception as e:
            print(f"[runner] ❌ ОШИБКА: {e}")
            traceback.print_exc()
            sys.exit(1)
