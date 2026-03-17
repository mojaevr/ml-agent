"""
MLDataset — единый интерфейс для агентов.

Хранит train/val/test данные и pipeline трансформаций.
Агент работает только через этот объект, predict применяет
те же трансформации автоматически.
"""

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class MLDataset:
    """
    Контейнер данных с автоматическим применением трансформаций.

    Агент может:
      1. Добавлять sklearn-трансформеры:
            dataset.add_transformer("scaler", StandardScaler())
         → трансформер fit на X_train, применяется на X_val и X_test автоматически

      2. Напрямую менять данные:
            dataset.X_train = my_custom_df
            dataset.X_val   = my_custom_df_val
         → при predict runner вызовет build_features(dataset) снова,
           поэтому трансформации воспроизводятся

    Атрибуты (read/write для агента):
        X_train : pd.DataFrame  — тренировочные фичи (без target)
        y_train : pd.Series     — тренировочный target
        X_val   : pd.DataFrame  — валидационные фичи
        y_val   : pd.Series     — валидационный target
        X_test  : pd.DataFrame  — тестовые фичи (без target, для сабмита)

    После add_transformer():
        X_train_transformed : np.ndarray / pd.DataFrame — трансформированный X_train
        X_val_transformed   : np.ndarray / pd.DataFrame — трансформированный X_val
        X_test_transformed  : np.ndarray / pd.DataFrame — трансформированный X_test

    Метаданные (read-only):
        feature_names : List[str]  — имена фич исходного датасета
        n_classes     : int        — число классов (2 для бинарной)
        task_type     : str        — "binary_classification" / "regression" / ...
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        task_type: str = "binary_classification",
    ):
        # Исходные данные — агент может читать и перезаписывать
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_val = X_val.copy()
        self.y_val = y_val.copy()
        self.X_test = X_test.copy() if X_test is not None else None

        # Метаданные
        self.feature_names: List[str] = list(X_train.columns)
        self.n_classes: int = int(y_train.nunique())
        self.task_type: str = task_type

        # Pipeline трансформаций (строится через add_transformer)
        self._transformers: List[Tuple[str, Any]] = []
        self._pipeline: Optional[Pipeline] = None
        self._pipeline_fitted: bool = False

        # Трансформированные данные (заполняются после fit_transform или add_transformer)
        self.X_train_transformed = self.X_train
        self.X_val_transformed = self.X_val
        self.X_test_transformed = self.X_test

    # ──────────────────────────────────────────────
    # Публичный интерфейс для агента
    # ──────────────────────────────────────────────

    def add_transformer(self, name: str, transformer: Any) -> "MLDataset":
        """
        Добавить sklearn-совместимый трансформер в pipeline.

        Трансформер будет fit на X_train и apply на X_val / X_test.
        Можно вызывать несколько раз — трансформеры применяются по порядку.

        Пример:
            dataset.add_transformer("scaler", StandardScaler())
            dataset.add_transformer("pca", PCA(n_components=10))
        """
        self._transformers.append((name, transformer))
        self._pipeline_fitted = False  # сбрасываем — надо перефитировать
        self._refit_pipeline()
        return self

    def info(self) -> str:
        """Краткая сводка о датасете — удобно для отладки."""
        lines = [
            f"Task:      {self.task_type}",
            f"X_train:   {self.X_train.shape}  (после feature engineering)",
            f"X_val:     {self.X_val.shape}",
            f"X_test:    {self.X_test.shape if self.X_test is not None else 'N/A'}",
            f"y_train:   {self.y_train.value_counts().to_dict()}",
            f"Pipeline:  {[name for name, _ in self._transformers] or 'empty'}",
            f"Transformed shapes:",
            f"  X_train_transformed: {np.array(self.X_train_transformed).shape}",
            f"  X_val_transformed:   {np.array(self.X_val_transformed).shape}",
        ]
        return "\n".join(lines)

    def get_train_val(self) -> Tuple[Any, pd.Series, Any, pd.Series]:
        """
        Удобный метод для передачи данных в модель.

        Возвращает: X_train_transformed, y_train, X_val_transformed, y_val
        """
        return (
            self.X_train_transformed,
            self.y_train,
            self.X_val_transformed,
            self.y_val,
        )

    # ──────────────────────────────────────────────
    # Внутренние методы (runner использует их)
    # ──────────────────────────────────────────────

    def _refit_pipeline(self):
        """Пересобирает и перефитирует pipeline на текущем X_train."""
        if not self._transformers:
            self.X_train_transformed = self.X_train
            self.X_val_transformed = self.X_val
            self.X_test_transformed = self.X_test
            self._pipeline_fitted = True
            return

        self._pipeline = Pipeline(steps=list(self._transformers))
        self.X_train_transformed = self._pipeline.fit_transform(
            self.X_train, self.y_train
        )
        self.X_val_transformed = self._pipeline.transform(self.X_val)
        if self.X_test is not None:
            self.X_test_transformed = self._pipeline.transform(self.X_test)
        self._pipeline_fitted = True

    def transform_new(self, X: pd.DataFrame) -> Any:
        """
        Применить тот же pipeline к новым данным (для predict).
        Используется runner-ом автоматически.
        """
        if not self._transformers or self._pipeline is None:
            return X
        return self._pipeline.transform(X)

    def save(self, path: str | Path):
        """Сохранить весь dataset (включая pipeline) на диск."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "MLDataset":
        """Загрузить dataset с диска."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return (
            f"MLDataset(task={self.task_type}, "
            f"train={self.X_train.shape}, val={self.X_val.shape}, "
            f"pipeline={[n for n, _ in self._transformers]})"
        )
