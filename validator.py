"""
validator.py — ЕДИНСТВЕННЫЙ способ считать ROC-AUC.
Агент не имеет доступа к val.csv напрямую.
Агент только сохраняет предикты в val_preds.npy — этот модуль считает метрику.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from typing import Optional
import argparse


def compute_roc_auc(val_preds_path: str, val_csv_path: str) -> Optional[float]:
    """
    Читает предикты агента и эталонные метки, считает ROC-AUC.
    
    val_preds_path: путь к .npy файлу с predict_proba (probability класса 1)
    val_csv_path:   путь к val.csv с колонкой 'target'
    """
    try:
        preds = np.load(val_preds_path)
        val_df = pd.read_csv(val_csv_path)

        if "target" not in val_df.columns:
            print(f"[VALIDATOR ERROR] Колонка 'target' не найдена в {val_csv_path}")
            return None

        y_true = val_df["target"].values

        if len(preds) != len(y_true):
            print(f"[VALIDATOR ERROR] Размер предиктов ({len(preds)}) != размер val ({len(y_true)})")
            return None

        # Если агент сохранил 2D массив (оба класса) — берём вероятность класса 1
        if preds.ndim == 2:
            preds = preds[:, 1]

        # Проверка на NaN/Inf
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            print(f"[VALIDATOR ERROR] Предикты содержат NaN или Inf")
            return None

        roc_auc = roc_auc_score(y_true, preds)
        return float(roc_auc)

    except FileNotFoundError as e:
        print(f"[VALIDATOR ERROR] Файл не найден: {e}")
        return None
    except Exception as e:
        print(f"[VALIDATOR ERROR] {e}")
        return None


def generate_submission(val_preds_path: str, test_csv_path: str, output_path: str) -> bool:
    """
    Генерирует файл сабмита для Kaggle на основе сохранённой модели и test.csv.
    Вызывается отдельно для любого run_id.
    """
    try:
        preds = np.load(val_preds_path)
        test_df = pd.read_csv(test_csv_path)

        if preds.ndim == 2:
            preds = preds[:, 1]

        submission = pd.DataFrame({
            "id": test_df.get("id", range(len(test_df))),
            "target": preds
        })
        submission.to_csv(output_path, index=False)
        print(f"[VALIDATOR] Сабмит сохранён: {output_path}")
        return True
    except Exception as e:
        print(f"[VALIDATOR ERROR] Не удалось создать сабмит: {e}")
        return False


if __name__ == "__main__":
    # Используется из orchestrator.py или вручную
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_preds", required=True, help="Путь к val_preds.npy")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--test_preds", help="Если указан — генерирует сабмит из test_preds.npy")
    parser.add_argument("--test_csv", default="data/test.csv")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    roc = compute_roc_auc(args.val_preds, args.val_csv)
    if roc is not None:
        print(f"ROC-AUC: {roc:.6f}")
    
    if args.test_preds:
        generate_submission(args.test_preds, args.test_csv, args.output)
