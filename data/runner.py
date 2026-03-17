"""
runner.py — НЕИЗМЕНЯЕМЫЙ скелет инфраструктуры.
Агент НЕ ТРОГАЕТ этот файл. Агент пишет только build_model() в train_code.py.

Использование в train_code.py:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace('/runs/' + os.environ.get('AGENT_RUN_DIR', '').split('/')[-1], ''))
    # ИЛИ проще:
    exec(open(os.environ['AGENT_RUNNER_PATH']).read())
    # После этого доступны: X_train, X_val, y_train, y_val, FEATURE_COLS
    # Нужно определить: build_model(X_train, y_train, X_val, y_val)
    # runner сам вызовет build_model и сохранит результаты

API функции build_model:
    Вход: X_train, X_val (np.ndarray float32), y_train, y_val (np.ndarray int)
    Выход: (model, extras_dict)
        model — обученная модель с методом predict_proba(X)
        extras_dict — любые дополнительные данные для predict (нормализаторы и т.д.)
                      НЕ ВКЛЮЧАЙ функции — только числа, массивы, sklearn-объекты
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

_MODE = os.environ.get("AGENT_MODE", "train")  # "train" или "predict"
RUN_DIR = os.environ["AGENT_RUN_DIR"]
FEATURE_COLS = ['id', 'return_1d', 'return_5d', 'return_10d', 'return_20d', 'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50', 'sma_ratio_200', 'ema_ratio_12', 'ema_ratio_26', 'macd_hist', 'rsi_14', 'bb_position', 'volatility_10d', 'volatility_20d', 'volatility_60d', 'volume_sma_ratio_10', 'volume_sma_ratio_20', 'daily_range', 'avg_range_10d', 'high_low_ratio', 'momentum_10d', 'momentum_20d', 'roc_5', 'roc_10', 'atr_14', 'stock_id']


def _encode_train(df):
    """Кодирует категориальные колонки, возвращает (df, encoders, freq_maps)."""
    encoders = {}
    freq_maps = {}
    HIGH_CARD_COLS = ['stock_id']
    for col in HIGH_CARD_COLS:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq).fillna(0.0)
            freq_maps[col] = freq
    return df, encoders, freq_maps


def _encode_predict(df, artifacts):
    """Применяет то же кодирование что и при обучении."""
    HIGH_CARD_COLS = ['stock_id']
    for col in HIGH_CARD_COLS:
        if col in df.columns:
            freq_map = artifacts.get('freq_maps', {}).get(col, {})
            df[col] = df[col].map(freq_map).fillna(0.0)
    return df


def _get_feature_matrix(df):
    """Возвращает numpy матрицу признаков."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[RUNNER] Предупреждение: отсутствуют колонки {missing}")
    X = df[available].copy()
    # Заполняем NaN медианой
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    return X.values.astype(np.float32), available


if _MODE == "train":
    # ── Загрузка ──────────────────────────────────────────────
    train_path = os.environ["AGENT_TRAIN_PATH"]
    print(f"[RUNNER] Загрузка {train_path}...")
    df = pd.read_csv(train_path)
    print(f"[RUNNER] Загружено {len(df):,} строк x {len(df.columns)} колонок")

    # ── Сплит (ЗАФИКСИРОВАНО — совпадает с внешним val.csv) ───
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )
    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)

    y_train = train_df["target"].values.astype(int)
    y_val = val_df["target"].values.astype(int)
    train_df = train_df.drop(columns=["target"])
    val_df = val_df.drop(columns=["target"])

    # ── Кодирование ───────────────────────────────────────────
    train_df, encoders, freq_maps = _encode_train(train_df)
    # Применяем то же кодирование к val
    val_df_enc = val_df.copy()
    _artifacts_tmp = {"encoders": encoders, "freq_maps": freq_maps}
    val_df_enc = _encode_predict(val_df_enc, _artifacts_tmp)

    # ── Матрицы признаков ─────────────────────────────────────
    X_train, used_cols = _get_feature_matrix(train_df)
    X_val, _ = _get_feature_matrix(val_df_enc)

    print(f"[RUNNER] X_train={X_train.shape}, X_val={X_val.shape}")
    print(f"[RUNNER] y_train positives={y_train.mean():.3f}, y_val={y_val.mean():.3f}")

    # ══════════════════════════════════════════════════════════
    # АГЕНТ ВЫЗЫВАЕТ СВОЮ ФУНКЦИЮ build_model ЗДЕСЬ
    # (определяется в train_code.py до вызова runner)
    # ══════════════════════════════════════════════════════════

elif _MODE == "predict":
    # В predict режиме runner ничего не делает сам.
    # predict_code.py вызывает train_code.py с AGENT_MODE=predict,
    # который сам определяет все кастомные классы и делает предикт.
    # Это единственный надёжный способ — train_code.py знает свои классы.
    pass
