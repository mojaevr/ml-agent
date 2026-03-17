"""
train_code.py — шаблон для агента.

Агент должен реализовать:
  1. build_features(dataset) — feature engineering
  2. build_model(dataset)    — обучение модели

Агент НЕ должен трогать:
  - predict (runner вызывает его автоматически)
  - способ валидации (ROC-AUC считается снаружи)
  - сохранение модели (runner делает это сам)

Текущая реализация: BASELINE (предсказывает 0.5 для всех)
"""

# ─── Импорты ───────────────────────────────────────────────────────────────────
# Агент может добавлять любые импорты здесь
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


# ─── Классы ────────────────────────────────────────────────────────────────────
# Если нужны кастомные классы — определяй их ЗДЕСЬ (на верхнем уровне),
# НЕ внутри build_model(). Иначе pickle не сможет их сохранить.

class BaselineClassifier(BaseEstimator, ClassifierMixin):
    """Заглушка: всегда предсказывает 0.5. Агент должен заменить."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


# ─── Feature Engineering ───────────────────────────────────────────────────────

def build_features(dataset):
    """
    Применить feature engineering к dataset.

    Агент может:
      - Добавлять новые колонки в dataset.X_train и dataset.X_val / dataset.X_test
      - Удалять колонки
      - Использовать dataset.add_transformer(name, sklearn_transformer)
        для автоматического применения sklearn-трансформеров

    ВАЖНО: любые изменения должны применяться к X_train, X_val И X_test
           (или через add_transformer — тогда это происходит автоматически).

    Пример:
        # Добавить трансформер (применится ко всем сплитам автоматически):
        from sklearn.preprocessing import StandardScaler
        dataset.add_transformer("scaler", StandardScaler())

        # Или вручную для нестандартных трансформаций:
        for split in ["X_train", "X_val", "X_test"]:
            df = getattr(dataset, split)
            if df is not None:
                df["new_feature"] = df["col_a"] / (df["col_b"] + 1e-9)
                setattr(dataset, split, df)

    Args:
        dataset: MLDataset — объект с данными

    Returns:
        dataset: MLDataset — тот же объект с изменениями
    """
    # BASELINE: ничего не делаем
    # TODO: агент заменяет эту реализацию
    return dataset


# ─── Обучение модели ───────────────────────────────────────────────────────────

def build_model(dataset):
    """
    Обучить модель на dataset.

    Используй dataset.get_train_val() для получения данных:
        X_train, y_train, X_val, y_val = dataset.get_train_val()

    Или напрямую:
        dataset.X_train_transformed  — трансформированный X_train
        dataset.y_train              — target
        dataset.X_val_transformed    — трансформированный X_val
        dataset.y_val                — валидационный target

    Модель должна иметь метод predict_proba(X) → array (n, 2)
    (для бинарной классификации — вероятности класса 0 и 1).

    Args:
        dataset: MLDataset — объект после build_features()

    Returns:
        model — обученная модель с методом predict_proba()
    """
    X_train, y_train, X_val, y_val = dataset.get_train_val()

    # BASELINE: модель-заглушка
    # TODO: агент заменяет эту реализацию
    model = BaselineClassifier()
    model.fit(X_train, y_train)
    return model
