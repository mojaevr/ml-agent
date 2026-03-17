"""
data_prep.py — подготовка данных и генерация инфраструктуры для агентов.

Ключевое изменение: генерирует runner.py — неизменяемый исполняемый скелет.
Агент пишет ТОЛЬКО функцию build_model(X_train, y_train, X_val, y_val) → (model, extras).
Всё остальное (загрузка, сплит, кодирование, сохранение, predict) — в runner.py.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data")
SCHEMA_PATH = DATA_DIR / "data_schema.json"
RUNNER_PATH = DATA_DIR / "runner.py"


# ─────────────────────────────────────────────────────────────
# Сплит
# ─────────────────────────────────────────────────────────────

def prepare_split(
    train_full_path: str = "data/train_full.csv",
    train_out: str = "data/train.csv",
    val_out: str = "data/val.csv",
    val_size: float = 0.2,
    random_state: int = 42,
):
    if Path(train_out).exists() and Path(val_out).exists():
        print(f"[DATA] Сплит уже существует: {train_out}, {val_out}")
        return
    print(f"[DATA] Загрузка {train_full_path}...")
    df = pd.read_csv(train_full_path)
    print(f"[DATA] Загружено {len(df):,} строк, {len(df.columns)} колонок")
    strat = df["target"] if "target" in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=random_state, stratify=strat
    )
    DATA_DIR.mkdir(exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    print(f"[DATA] Train: {len(train_df):,} → {train_out}")
    print(f"[DATA] Val:   {len(val_df):,} → {val_out}")


# ─────────────────────────────────────────────────────────────
# Анализ схемы
# ─────────────────────────────────────────────────────────────

def analyze_schema(train_path: str = "data/train.csv") -> dict:
    print("[DATA] Анализ схемы данных...")
    df = pd.read_csv(train_path, nrows=50_000)
    full_size = sum(1 for _ in open(train_path)) - 1

    feature_cols = [c for c in df.columns if c != "target"]
    numeric_cols, categorical_cols, high_cardinality_cols, id_cols = [], [], [], []

    for col in feature_cols:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        col_lower = col.lower()
        is_numeric = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
        is_bool = pd.api.types.is_bool_dtype(dtype)
        is_string = pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype)

        if is_string and any(kw in col_lower for kw in ["id", "_id", "index", "idx"]) and n_unique > 1000:
            id_cols.append(col)
        elif is_numeric or is_bool:
            numeric_cols.append(col)
        elif is_string:
            (categorical_cols if n_unique <= 50 else high_cardinality_cols).append(col)
        else:
            try:
                df[col].astype(float)
                numeric_cols.append(col)
            except (ValueError, TypeError):
                high_cardinality_cols.append(col)

    def _safe_float(v):
        try:
            f = float(v)
            return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
        except Exception:
            return None

    numeric_stats = {}
    for col in numeric_cols[:30]:
        try:
            c = pd.to_numeric(df[col], errors="coerce").dropna()
            numeric_stats[col] = {
                "dtype": str(df[col].dtype),
                "mean": _safe_float(c.mean()) if len(c) else None,
                "std": _safe_float(c.std()) if len(c) else None,
                "min": _safe_float(c.min()) if len(c) else None,
                "max": _safe_float(c.max()) if len(c) else None,
                "null_pct": round(float(df[col].isnull().mean() * 100), 2),
                "zeros_pct": round(float((df[col] == 0).mean() * 100), 2),
            }
        except Exception as e:
            numeric_stats[col] = {"dtype": str(df[col].dtype), "error": str(e)}

    categorical_stats = {}
    for col in categorical_cols:
        categorical_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": df[col].value_counts().head(5).to_dict(),
            "null_pct": round(float(df[col].isnull().mean() * 100), 2),
        }

    target_info = {}
    if "target" in df.columns:
        vc = df["target"].value_counts()
        target_info = {
            "distribution": vc.to_dict(),
            "imbalance_ratio": round(float(vc.max() / vc.min()), 2) if len(vc) == 2 else None,
        }

    schema = {
        "total_rows_approx": full_size,
        "total_feature_cols": len(feature_cols),
        "target_info": target_info,
        "columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "high_cardinality": high_cardinality_cols,
            "id_like": id_cols,
        },
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "global_missing": {
            col: round(float(df[col].isnull().mean() * 100), 2)
            for col in feature_cols if df[col].isnull().any()
        },
    }

    SCHEMA_PATH.parent.mkdir(exist_ok=True)
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"[DATA] Схема: {len(numeric_cols)} числовых, {len(categorical_cols)} категориальных, "
          f"{len(high_cardinality_cols)} высокой кардинальности, {len(id_cols)} id-подобных")
    return schema


# ─────────────────────────────────────────────────────────────
# Генерация runner.py — неизменяемый скелет
# ─────────────────────────────────────────────────────────────

def generate_runner(schema: dict) -> str:
    """
    Генерирует data/runner.py — готовый исполняемый скелет.

    Агент импортирует runner и ТОЛЬКО пишет функцию build_model().
    runner.py берёт на себя всё остальное:
    - загрузка train_full.csv
    - кодирование категориальных колонок
    - правильный сплит (совпадает с val.csv)
    - сохранение val_preds.npy и model.pkl
    - генерация предиктов для test.csv
    """
    cols = schema["columns"]
    numeric_cols = cols["numeric"]
    categorical_cols = cols["categorical"]
    high_card_cols = cols["high_cardinality"]
    id_cols = cols["id_like"]

    all_feature_cols = [c for c in (numeric_cols + categorical_cols + high_card_cols)
                        if c not in id_cols]

    # Строим encoding блок
    enc_lines = []
    if categorical_cols:
        enc_lines += [
            f"    CAT_COLS = {categorical_cols}",
            "    for col in CAT_COLS:",
            "        if col in df.columns:",
            "            le = LabelEncoder()",
            "            combined = df[col].astype(str).fillna('__nan__')",
            "            le.fit(combined)",
            "            df[col] = le.transform(combined)",
            "            encoders[col] = le",
        ]
    if high_card_cols:
        enc_lines += [
            f"    HIGH_CARD_COLS = {high_card_cols}",
            "    for col in HIGH_CARD_COLS:",
            "        if col in df.columns:",
            "            freq = df[col].value_counts(normalize=True).to_dict()",
            "            df[col] = df[col].map(freq).fillna(0.0)",
            "            freq_maps[col] = freq",
        ]
    if id_cols:
        enc_lines += [
            f"    ID_COLS = {id_cols}",
            "    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])",
        ]

    encoding_code = "\n".join(enc_lines) if enc_lines else "    pass  # все колонки числовые"

    # predict encoding
    pred_enc_lines = []
    if categorical_cols:
        pred_enc_lines += [
            f"    CAT_COLS = {categorical_cols}",
            "    for col in CAT_COLS:",
            "        if col in df.columns and col in artifacts.get('encoders', {}):",
            "            le = artifacts['encoders'][col]",
            "            known = set(le.classes_)",
            "            df[col] = df[col].astype(str).fillna('__nan__')",
            "            df[col] = df[col].apply(lambda x: x if x in known else '__nan__')",
            "            df[col] = le.transform(df[col])",
        ]
    if high_card_cols:
        pred_enc_lines += [
            f"    HIGH_CARD_COLS = {high_card_cols}",
            "    for col in HIGH_CARD_COLS:",
            "        if col in df.columns:",
            "            freq_map = artifacts.get('freq_maps', {}).get(col, {})",
            "            df[col] = df[col].map(freq_map).fillna(0.0)",
        ]
    if id_cols:
        pred_enc_lines += [
            f"    ID_COLS = {id_cols}",
            "    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])",
        ]

    predict_encoding_code = "\n".join(pred_enc_lines) if pred_enc_lines else "    pass"

    runner_code = f'''"""
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
FEATURE_COLS = {all_feature_cols!r}


def _encode_train(df):
    """Кодирует категориальные колонки, возвращает (df, encoders, freq_maps)."""
    encoders = {{}}
    freq_maps = {{}}
{encoding_code}
    return df, encoders, freq_maps


def _encode_predict(df, artifacts):
    """Применяет то же кодирование что и при обучении."""
{predict_encoding_code}
    return df


def _get_feature_matrix(df):
    """Возвращает numpy матрицу признаков."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[RUNNER] Предупреждение: отсутствуют колонки {{missing}}")
    X = df[available].copy()
    # Заполняем NaN медианой
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    return X.values.astype(np.float32), available


if _MODE == "train":
    # ── Загрузка ──────────────────────────────────────────────
    train_path = os.environ["AGENT_TRAIN_PATH"]
    print(f"[RUNNER] Загрузка {{train_path}}...")
    df = pd.read_csv(train_path)
    print(f"[RUNNER] Загружено {{len(df):,}} строк x {{len(df.columns)}} колонок")

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
    _artifacts_tmp = {{"encoders": encoders, "freq_maps": freq_maps}}
    val_df_enc = _encode_predict(val_df_enc, _artifacts_tmp)

    # ── Матрицы признаков ─────────────────────────────────────
    X_train, used_cols = _get_feature_matrix(train_df)
    X_val, _ = _get_feature_matrix(val_df_enc)

    print(f"[RUNNER] X_train={{X_train.shape}}, X_val={{X_val.shape}}")
    print(f"[RUNNER] y_train positives={{y_train.mean():.3f}}, y_val={{y_val.mean():.3f}}")

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
'''

    RUNNER_PATH.parent.mkdir(exist_ok=True)
    with open(RUNNER_PATH, "w") as f:
        f.write(runner_code)
    print(f"[DATA] runner.py сохранён: {RUNNER_PATH}")
    return runner_code


# ─────────────────────────────────────────────────────────────
# Публичная функция
# ─────────────────────────────────────────────────────────────

def get_dataset_stats(train_path: str = "data/train.csv") -> tuple[str, str]:
    """Возвращает (dataset_description, runner_code)."""
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
    else:
        schema = analyze_schema(train_path)

    if RUNNER_PATH.exists():
        runner_code = RUNNER_PATH.read_text()
    else:
        runner_code = generate_runner(schema)

    cols = schema["columns"]
    target = schema.get("target_info", {})
    lines = [
        f"Размер: ~{schema['total_rows_approx']:,} строк × {schema['total_feature_cols']} фичей",
        f"Задача: бинарная классификация, target = 0/1",
        f"Баланс классов: {target.get('distribution', {})} (imbalance: {target.get('imbalance_ratio')})",
        "",
        "═══ ТИПЫ КОЛОНОК ═══",
        f"Числовые ({len(cols['numeric'])}): {', '.join(cols['numeric'][:20])}{'...' if len(cols['numeric']) > 20 else ''}",
    ]
    if cols["categorical"]:
        lines.append(f"Категориальные ({len(cols['categorical'])}): {', '.join(cols['categorical'])}")
    if cols["high_cardinality"]:
        lines.append(f"Высокая кардинальность ({len(cols['high_cardinality'])}): {', '.join(cols['high_cardinality'])}")
    if cols["id_like"]:
        lines.append(f"ID-колонки (дропаются): {', '.join(cols['id_like'])}")
    missing = schema.get("global_missing", {})
    lines.append(f"\nПропуски: {missing if missing else 'нет'}")
    lines.append("\n═══ СТАТИСТИКА (первые 15 числовых) ═══")
    for col, st in list(schema.get("numeric_stats", {}).items())[:15]:
        if "error" not in st:
            lines.append(f"  {col}: mean={st['mean']}, std={st['std']}, [{st['min']}, {st['max']}]")

    return "\n".join(lines), runner_code


def prepare_all(train_full_path: str = "data/train_full.csv"):
    prepare_split(train_full_path)
    schema = analyze_schema()
    generate_runner(schema)
    print("\n[DATA] ✅ Подготовка завершена")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_all()
    if args.stats:
        desc, _ = get_dataset_stats()
        print(desc)
