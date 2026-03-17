"""
boilerplate.py — готовый шаблон загрузки данных для агентов.
СКОПИРУЙ ЭТОТ КОД В НАЧАЛО train_code.py и добавь свою модель.

Этот boilerplate уже правильно:
- загружает данные
- кодирует категориальные признаки
- делает правильный сплит (НЕ МЕНЯЙ random_state и test_size!)
- готовит X_train, X_val, y_train, y_val
- показывает как сохранить val_preds и модель
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── 1. Загрузка ПОЛНОГО тренировочного датасета ─────────────
# AGENT_TRAIN_PATH указывает на train_full.csv — весь датасет до сплита
df = pd.read_csv(os.environ["AGENT_TRAIN_PATH"])

# ── 2. ОБЯЗАТЕЛЬНЫЙ СПЛИТ ────────────────────────────────────
# ВАЖНО: эти параметры зафиксированы — внешний валидатор использует
# тот же сплит. Любое изменение приведёт к несовпадению размеров!
# НЕ МЕНЯЙ: test_size, random_state, stratify
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["target"]
)
# train_df: ~80% данных для обучения
# val_df:   ~20% данных — предикты на НИХ нужно сохранить в val_preds.npy
train_df = train_df.copy()
val_df = val_df.copy()

# ── 3. Целевая переменная ─────────────────────────────────────
y_train = train_df["target"].values
y_val = val_df["target"].values
train_df = train_df.drop(columns=["target"])
val_df = val_df.drop(columns=["target"])

# ── 4. Обработка признаков ────────────────────────────────────
    # ── Кодирование нечисловых колонок ──────────────────────────
    HIGH_CARD_COLS = ['stock_id']
    # Высококардинальные: target encoding или просто дроп
    # Ниже — простой вариант через частоту (freq encoding)
    for col in HIGH_CARD_COLS:
        freq = train_df[col].value_counts() / len(train_df)
        train_df[col] = train_df[col].map(freq).fillna(0)
        val_df[col] = val_df[col].map(freq).fillna(0)

# ── 5. Матрицы признаков ──────────────────────────────────────
# После всех преобразований:
FEATURE_COLS = [c for c in train_df.columns]
X_train = train_df[FEATURE_COLS].values.astype(np.float32)
X_val = val_df[FEATURE_COLS].values.astype(np.float32)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")
print(f"y_train positives: {y_train.mean():.3f}, y_val positives: {y_val.mean():.3f}")

# ═══════════════════════════════════════════════════════════════
# ДОБАВЬ СВОЮ МОДЕЛЬ ЗДЕСЬ
# ═══════════════════════════════════════════════════════════════
# Например:
# from lightgbm import LGBMClassifier
# model = LGBMClassifier(n_estimators=500, learning_rate=0.05)
# model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
# val_preds = model.predict_proba(X_val)[:, 1]
# ═══════════════════════════════════════════════════════════════

# ── 6. Сохранение результатов (ОБЯЗАТЕЛЬНО) ───────────────────
RUN_DIR = os.environ["AGENT_RUN_DIR"]

# Предикты на val (для валидации ROC-AUC)
# val_preds должен быть shape (N,) — вероятности класса 1, N = 20% от train
np.save(os.path.join(RUN_DIR, "val_preds.npy"), val_preds)

# Сохрани ВСЁ что нужно predict_code.py в один объект
# ВАЖНО: не используй lambda в pipeline — pickle их не сохраняет
pipeline = {
    "model": model,
    "feature_cols": FEATURE_COLS,
    # добавь сюда encoders и прочее если нужно
}
with open(os.path.join(RUN_DIR, "model.pkl"), "wb") as f:
    pickle.dump(pipeline, f)

print(f"Val ROC-AUC (примерный): {__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(y_val, val_preds):.6f}")
print("Done.")
