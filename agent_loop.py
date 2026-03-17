"""
agent_loop.py — агент пишет ТОЛЬКО функцию build_model().

Новая архитектура:
- runner.py (генерируется data_prep.py) берёт на себя: загрузку, сплит, кодирование, сохранение
- train_code.py = ТОЛЬКО build_model() + exec(runner)
- predict_code.py = ТОЛЬКО exec(runner с MODE=predict)
- Никаких KeyError, неправильных сплитов, pickle с функциями
"""
import ast
import os
import re
import time
import subprocess
import sys
import requests
from pathlib import Path
from typing import Optional

from budget_tracker import record_usage, check_budget

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_FIX_ATTEMPTS = 3

# Если настроен SOCKS-прокси но нет PySocks — requests падает.
# Снимаем SOCKS-прокси для API-запросов (оставляем только http/https прокси).
def _get_proxies() -> dict | None:
    """Возвращает настройки прокси для requests. Без SOCKS, http дублируется из https."""
    proxies = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        val = os.environ.get(key, "")
        if val and not val.lower().startswith("socks"):
            proxies[key.lower().split("_")[0]] = val
    # Если задан только https — дублируем в http (OpenRouter использует https,
    # но requests может потребовать оба для корректного проксирования)
    if "https" in proxies and "http" not in proxies:
        proxies["http"] = proxies["https"]
    if "http" in proxies and "https" not in proxies:
        proxies["https"] = proxies["http"]
    return proxies or None

# Проверяем есть ли SOCKS-прокси и PySocks
def _check_socks():
    import os
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        val = os.environ.get(key, "")
        if val.lower().startswith("socks"):
            try:
                import socks  # noqa
            except ImportError:
                print(f"⚠️  SOCKS-прокси ({val}) требует: pip install 'requests[socks]'")
                print("   Временно снимаем SOCKS для API-запросов...")
                # Сохраняем и убираем SOCKS-переменные
                os.environ.pop("ALL_PROXY", None); os.environ.pop("all_proxy", None)
                # Оставляем только если не socks
                for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
                    if os.environ.get(k,"").lower().startswith("socks"):
                        os.environ.pop(k, None)
            break

_check_socks()

# predict_code.py который всегда работает:
# загружает модель, импортирует build_features из train_code.py через importlib,
# применяет то же feature engineering что и при обучении
_DEFAULT_PREDICT_CODE = """# predict_code.py
import os, pickle, numpy as np, pandas as pd
RUN_DIR = os.environ["AGENT_RUN_DIR"]

# ШАГ 1: Загружаем модель
with open(os.path.join(RUN_DIR, "model.pkl"), "rb") as _f:
    _saved = pickle.load(_f)
_model = _saved["model"]
_feature_cols = _saved["feature_cols"]

# ШАГ 2: Восстанавливаем build_features из сохранённого исходника
_bfe_src = _saved.get("build_features_source", "")
if _bfe_src:
    _g = {"np": np, "numpy": np, "pd": pd}
    exec(_bfe_src, _g)
    _build_features = _g.get("build_features", lambda x: x)
else:
    _build_features = lambda x: x

# ШАГ 3: Получаем _encode_predict из runner.py (в режиме predict — только функции, без обучения)
os.environ["AGENT_MODE"] = "predict"
exec(open(os.environ["AGENT_RUNNER_PATH"]).read(), globals())

# ШАГ 4: Кодируем test и предсказываем
_df_test = pd.read_csv(os.environ["AGENT_TEST_PATH"])
_df_enc = _encode_predict(_df_test, {})
_avail = [c for c in _feature_cols if c in _df_enc.columns]
_X_raw = _df_enc[_avail].fillna(0).values.astype(np.float32)
_X_fe = _build_features(_X_raw)
if isinstance(_model, list):
    _preds = np.mean([m.predict_proba(_X_fe)[:, 1] for m in _model], axis=0)
else:
    _preds = _model.predict_proba(_X_fe)[:, 1]
np.save(os.path.join(RUN_DIR, "test_preds.npy"), _preds)
print(f"[PREDICT] test_preds: {_preds.shape}")
"""


# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────

def _call_openrouter(
    messages: list[dict],
    model: str,
    max_tokens: int = 4096,
    daily_limit_usd: float = 5.0,
    retries: int = 3,
    retry_delay: float = 8.0,
) -> tuple[str, int, int]:
    ok, spent = check_budget(daily_limit_usd)
    if not ok:
        raise RuntimeError(f"Дневной лимит ${daily_limit_usd} превышен! Потрачено: ${spent:.4f}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY не задан")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ml-agent",
        "X-Title": "ML AutoAgent",
    }
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}

    proxies = _get_proxies()
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=180, proxies=proxies)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            inp = usage.get("prompt_tokens", 0)
            out = usage.get("completion_tokens", 0)
            cost = record_usage(model, inp, out)
            print(f"   [LLM] {inp} in + {out} out = ${cost:.4f} ({model})")
            return text, inp, out
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            last_error = e
            wait = retry_delay * (attempt + 1)
            print(f"   [LLM] Сетевая ошибка (попытка {attempt+1}/{retries}), жду {wait:.0f}с...")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            last_error = Exception("Timeout")
            print(f"   [LLM] Timeout (попытка {attempt+1}/{retries}), жду {retry_delay:.0f}с...")
            time.sleep(retry_delay)
        except requests.exceptions.HTTPError as e:
            if resp.status_code >= 500:
                last_error = e
                print(f"   [LLM] HTTP {resp.status_code} (попытка {attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:200]}")

    raise RuntimeError(f"OpenRouter недоступен после {retries} попыток: {last_error}")


# ─────────────────────────────────────────────────────────────
# Вспомогательные
# ─────────────────────────────────────────────────────────────

def _extract_all_code_blocks(text: str) -> list[str]:
    blocks = re.findall(r"```python\n?(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\n?(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def _extract_build_model(text: str) -> Optional[str]:
    """Извлекает только функцию build_model из ответа LLM."""
    blocks = _extract_all_code_blocks(text)
    for block in blocks:
        if "def build_model" in block:
            return block
    # Fallback: весь первый блок
    return blocks[0] if blocks else None


def _extract_missing_module(stderr: str) -> Optional[str]:
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
    if not m:
        return None
    module = m.group(1).split(".")[0]
    pip_map = {
        "umap": "umap-learn", "sklearn": "scikit-learn", "cv2": "opencv-python",
        "PIL": "Pillow", "yaml": "pyyaml", "lightgbm": "lightgbm",
        "xgboost": "xgboost", "catboost": "catboost", "shap": "shap",
        "optuna": "optuna", "boruta": "boruta", "phik": "phik", "prince": "prince",
    }
    return pip_map.get(module, module)


def _try_pip_install(package: str) -> bool:
    print(f"   [AUTO-INSTALL] pip install {package}...")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            capture_output=True, text=True, timeout=120
        )
        ok = r.returncode == 0
        print(f"   [AUTO-INSTALL] {'✅' if ok else '❌'} {package}")
        return ok
    except Exception as e:
        print(f"   [AUTO-INSTALL] ❌ {e}")
        return False


def preflight_check(code: str) -> list[str]:
    """Проверяет train_code.py до запуска."""
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        issues.append(f"SyntaxError: {e}")
        return issues

    if "def build_model" not in code:
        issues.append("Нет функции build_model — агент должен определить её")
    if "AGENT_RUNNER_PATH" not in code:
        issues.append("Нет вызова runner.py — добавь: exec(open(os.environ['AGENT_RUNNER_PATH']).read(), globals())")
    if re.search(r"\blambda\b", code) and "pickle" in code:
        issues.append("lambda в pickle не сериализуется")

    # Кастомные классы не пикклятся между разными скриптами
    # Ни внутри build_model, ни на верхнем уровне
    top_level_classes = []
    nested_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it's top-level (parent is Module)
            top_level_classes.append(node.name)
        if isinstance(node, ast.FunctionDef) and node.name == "build_model":
            for child in ast.walk(node):
                if isinstance(child, ast.ClassDef):
                    nested_classes.append(child.name)

    # Any user-defined class (not from stdlib/sklearn) will break pickle
    known_ok = {"Pipeline", "ColumnTransformer"}  # sklearn classes are ok if imported
    bad_classes = [c for c in top_level_classes if c not in known_ok]
    if bad_classes:
        issues.append(
            f"Кастомные классы {bad_classes} нельзя сохранить в model.pkl — "
            f"pickle не найдёт их при загрузке в predict_code.py. "
            f"Используй VotingClassifier, StackingClassifier или усредняй предикты вручную."
        )
    for c in nested_classes:
        issues.append(
            f"Класс '{c}' внутри build_model — не пикклится. "
            f"Используй VotingClassifier вместо кастомного класса."
        )

    # Устаревшее LightGBM API
    lgb_checks = [
        ("verbose_eval=", "verbose_eval устарел - используй callbacks=[lgb.log_evaluation(N)]"),
        ("early_stopping_rounds=", "early_stopping_rounds в .fit() устарел - используй callbacks=[lgb.early_stopping(N)]"),
    ]
    for pattern, msg in lgb_checks:
        if re.search(pattern, code):
            issues.append(f"Устаревший LightGBM API: {msg}")

    return issues


# ─────────────────────────────────────────────────────────────
# Системный промпт
# ─────────────────────────────────────────────────────────────

def _build_system(runner_code: str) -> str:
    return f"""Ты — исследователь машинного обучения.

ЗАДАЧА: бинарная классификация табличных данных, максимизировать ROC-AUC.

══════════════════════════════════════════
КАК ПИСАТЬ КОД
══════════════════════════════════════════
Вся инфраструктура уже готова в runner.py. Тебе нужно написать ТОЛЬКО:
1. Функцию build_model(X_train, y_train, X_val, y_val) → (model, extras)
2. Шаблонный вызов runner (см. ниже)

train_code.py ВСЕГДА выглядит по этому шаблону (СКОПИРУЙ ДОСЛОВНО, меняй только отмеченные части):
```python
# train_code.py
import os, inspect, numpy as np, pickle
from sklearn.metrics import roc_auc_score

_MODE = os.environ.get("AGENT_MODE", "train")
RUN_DIR = os.environ["AGENT_RUN_DIR"]

# ── Кастомные классы НА ВЕРХНЕМ УРОВНЕ (не внутри build_model!) ──
# class MyEnsemble: ...  ← если нужен кастомный класс — определяй ЗДЕСЬ

def build_features(X_raw):
    # МЕНЯЙ ЭТУ ФУНКЦИЮ — feature engineering.
    Применяется одинаково к train, val и test.
    X_raw: numpy array (N, n_features)
    Возвращает: numpy array (N, m_features)
    #--
    # Пример: return np.hstack([X_raw, X_raw**2])
    return X_raw  # по умолчанию без изменений

def build_model(X_train, y_train, X_val, y_val):
    # МЕНЯЙ ЭТУ ФУНКЦИЮ — обучение модели.
    Возвращает обученную модель с методом predict_proba(X).
    #--
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    model = LGBMClassifier(n_estimators=500, learning_rate=0.05,
                           num_leaves=63, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    return model

# ══════════════════════════════════════════════════════════════
# ШАБЛОННЫЙ КОД — НЕ МЕНЯЙ НИЧЕГО НИЖЕ
# ══════════════════════════════════════════════════════════════
from sklearn.metrics import roc_auc_score as _roc_auc
import pickle as _pickle

if _MODE == "import_only":
    pass  # только импортируем классы/функции для predict_code.py
elif _MODE == "train":
    exec(open(os.environ["AGENT_RUNNER_PATH"]).read(), globals())
    _X_train = build_features(X_train)
    _X_val   = build_features(X_val)
    _model   = build_model(_X_train, y_train, _X_val, y_val)
    _val_preds = _model.predict_proba(_X_val)[:, 1]
    np.save(os.path.join(RUN_DIR, "val_preds.npy"), _val_preds)
    with open(os.path.join(RUN_DIR, "model.pkl"), "wb") as _f:
        _bfe_src = inspect.getsource(build_features)
        _pickle.dump({{"model": _model, "feature_cols": FEATURE_COLS,
                      "build_features_source": _bfe_src}}, _f)
    print(f"Val ROC-AUC: {{_roc_auc(y_val, _val_preds):.6f}}")
    print("Done.")


predict_code.py ВСЕГДА ОДИНАКОВЫЙ (не меняй, используй дословно):
```python
# predict_code.py
import os, pickle, numpy as np, pandas as pd
RUN_DIR = os.environ["AGENT_RUN_DIR"]
# Загружаем модель
with open(os.path.join(RUN_DIR, "model.pkl"), "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
feature_cols = saved["feature_cols"]
# Загружаем test и кодируем через runner
exec(open(os.environ["AGENT_RUNNER_PATH"]).read(), globals())
# Кодируем test вручную (runner экспортирует _encode_predict)
df_test = pd.read_csv(os.environ["AGENT_TEST_PATH"])
df_enc = _encode_predict(df_test, {{}})
avail = [c for c in feature_cols if c in df_enc.columns]
X_test_raw = df_enc[avail].fillna(0).values.astype(np.float32)
# Загружаем build_features из train_code.py
import importlib.util, sys
_spec = importlib.util.spec_from_file_location("_tc", os.path.join(RUN_DIR, "train_code.py"))
_tc = importlib.util.module_from_spec(_spec)
os.environ["AGENT_MODE"] = "import_only"
_spec.loader.exec_module(_tc)
X_test_fe = _tc.build_features(X_test_raw)
preds = model.predict_proba(X_test_fe)[:, 1]
np.save(os.path.join(RUN_DIR, "test_preds.npy"), preds)
print(f"[PREDICT] test_preds: {{preds.shape}}")
```

══════════════════════════════════════════
RUNNER.PY (уже готов, только для понимания)
══════════════════════════════════════════
{runner_code[:1500]}
... (runner берёт на себя загрузку, сплит, кодирование, сохранение)

══════════════════════════════════════════
ЛИМИТ ВРЕМЕНИ: 8 МИНУТ
══════════════════════════════════════════
Датасет ~500k строк. Запрещено (медленно):
- SMOTE → используй class_weight='balanced'
- GridSearchCV → используй Optuna n_trials≤15 или фиксированные параметры
- Нейросети без early stopping
ПРАВИЛЬНЫЙ API LightGBM (TypeError если использовать старый):
  callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(100)]  # ПРАВИЛЬНО
  verbose_eval=100  # ОШИБКА TypeError
  early_stopping_rounds=50 в .fit()  # ОШИБКА TypeError
РАЗРЕШЕНО: LightGBM/XGBoost/CatBoost с callbacks, VotingClassifier, feature engineering

══════════════════════════════════════════
ТВОЯ ЗОНА ТВОРЧЕСТВА (только внутри build_model)
══════════════════════════════════════════
- Feature engineering на X_train/X_val (numpy операции)
- Любой алгоритм: LightGBM, XGBoost, CatBoost, sklearn, ансамбли
- Нестандартные идеи: rank-трансформации, кластеризация как фичи, стекинг
- НЕ ОПРЕДЕЛЯЙ функции которые попадут в extras — только числа и sklearn-объекты
- КРИТИЧНО: НЕ СОЗДАВАЙ кастомные классы (EnsembleModel, Wrapper и т.п.) — они не пикклятся!
  Для ансамблей используй: sklearn VotingClassifier, StackingClassifier или усредняй предикты:
    p = (m1.predict_proba(X)[:,1] + m2.predict_proba(X)[:,1]) / 2
    np.save(..., p)  # и сохрани список моделей в pkl

══════════════════════════════════════════
ФОРМАТ ОТВЕТА
══════════════════════════════════════════
1. Описание подхода (2-3 предложения)
2. ```python
   # train_code.py
   <полный код>
   ```
3. ```python
   # predict_code.py
   import os
   os.environ["AGENT_MODE"] = "predict"
   _d = os.path.dirname(os.path.abspath(__file__))
   exec(open(os.path.join(_d, "train_code.py")).read())
   ```"""


_FIX_SYSTEM = """Ты — Python-разработчик. Код упал с ошибкой. Исправь ТОЛЬКО функцию build_model().
НЕ МЕНЯЙ шаблонный код после build_model (exec runner, сохранение).
Верни полный исправленный train_code.py в блоке ```python # train_code.py```"""

_TIMEOUT_FIX_SYSTEM = """Код превысил лимит 8 минут. Перепиши build_model() чтобы работало быстро.
- SMOTE → class_weight='balanced'
- GridSearchCV → фиксированные параметры или Optuna n_trials=5
- Нейросеть → LightGBM с n_estimators=200
Верни полный исправленный train_code.py в блоке ```python # train_code.py```"""


# ─────────────────────────────────────────────────────────────
# Основные функции
# ─────────────────────────────────────────────────────────────

def generate_training_code(
    dataset_stats: str,
    runner_code: str,
    memory_summary: str,
    run_id: str,
    model: str,
    daily_limit_usd: float = 5.0,
    agent_id: int = 0,
    discussion_context: str = "",
) -> tuple[Optional[str], Optional[str], str, float]:
    """Генерирует train_code.py (только build_model) и predict_code.py (шаблон)."""
    discussion_part = (
        f"\n=== ОБСУЖДЕНИЕ С ДРУГИМ АГЕНТОМ ===\n{discussion_context}\n"
        if discussion_context else ""
    )
    user_message = f"""=== СХЕМА ДАТАСЕТА ===
{dataset_stats}

=== ИСТОРИЯ ПОПЫТОК (не повторяй!) ===
{memory_summary}
{discussion_part}
Run ID: {run_id} | Агент #{agent_id}
Придумай НОВЫЙ нестандартный подход. Реализуй его в build_model()."""

    messages = [
        {"role": "system", "content": _build_system(runner_code)},
        {"role": "user", "content": user_message},
    ]

    # Jitter чтобы параллельные агенты не долбили API одновременно
    if agent_id > 0:
        time.sleep(agent_id * 2.0)
    response, inp, out = _call_openrouter(messages, model, max_tokens=5000, daily_limit_usd=daily_limit_usd)
    cost_estimate = (inp * 0.55 + out * 2.19) / 1_000_000

    blocks = _extract_all_code_blocks(response)
    train_code = None

    for block in blocks:
        if "train_code" in block[:50] or "build_model" in block:
            train_code = block
            break

    if not train_code and blocks:
        train_code = blocks[0]

    # predict_code ВСЕГДА фиксированный — агент не может его переписать
    # это единственный способ гарантировать правильную загрузку кастомных классов
    predict_code = _DEFAULT_PREDICT_CODE

    description = response.split("```")[0].strip()[:600]
    return train_code, predict_code, description, cost_estimate


def fix_code_after_error(
    train_code: str,
    predict_code: Optional[str],
    stdout: str,
    stderr: str,
    fix_attempt: int,
    model: str,
    daily_limit_usd: float = 5.0,
    is_timeout: bool = False,
) -> tuple[Optional[str], Optional[str]]:
    """Исправляет код после ошибки."""
    # Auto pip install
    missing = _extract_missing_module(stderr)
    if missing and not is_timeout:
        if _try_pip_install(missing):
            return train_code, predict_code

    system = _TIMEOUT_FIX_SYSTEM if is_timeout else _FIX_SYSTEM
    context = f"STDOUT: {stdout[-600:]}\n" if is_timeout else f"STDERR:\n{stderr[-1200:]}\nSTDOUT: {stdout[-300:]}\n"

    print(f"   [FIX] {'TIMEOUT' if is_timeout else 'ERROR'} fix (попытка #{fix_attempt})...")
    user_message = f"""{context}
КОД:
```python
# train_code.py
{train_code}
```"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    try:
        response, _, _ = _call_openrouter(messages, model, max_tokens=4000, daily_limit_usd=daily_limit_usd)
        blocks = _extract_all_code_blocks(response)
        fixed = next((b for b in blocks if "build_model" in b or "train_code" in b[:50]), None)
        if not fixed and blocks:
            fixed = blocks[0]
        return fixed, predict_code
    except Exception as e:
        print(f"   [FIX] Ошибка: {e}")
        return None, None


def fix_wrong_val_size(
    train_code: str,
    predict_code: Optional[str],
    actual_size: int,
    expected_size: int,
    model: str,
    daily_limit_usd: float = 5.0,
) -> tuple[Optional[str], Optional[str]]:
    """Исправляет неправильный размер val_preds."""
    # В новой архитектуре с runner.py размер всегда правильный
    # Если всё же случилось — значит агент не использовал runner
    print(f"   [SIZE-FIX] val_preds={actual_size} ≠ expected={expected_size}")
    print(f"   [SIZE-FIX] Скорее всего агент не использовал runner.py — это баг архитектуры")

    user_message = f"""val_preds.npy имеет {actual_size} элементов вместо {expected_size}.
Скорее всего ты не использовал runner.py для сплита.

Исправь train_code.py: убедись что используется:
    exec(open(os.environ['AGENT_RUNNER_PATH']).read())
и val_preds берётся из model.predict_proba(X_val)[:, 1] где X_val — из runner.

КОД:
```python
{train_code}
```"""

    messages = [
        {"role": "system", "content": _FIX_SYSTEM},
        {"role": "user", "content": user_message},
    ]
    try:
        response, _, _ = _call_openrouter(messages, model, max_tokens=4000, daily_limit_usd=daily_limit_usd)
        blocks = _extract_all_code_blocks(response)
        fixed = blocks[0] if blocks else None
        return fixed, predict_code
    except Exception as e:
        print(f"   [SIZE-FIX] Ошибка: {e}")
        return None, None


def propose_approach(
    agent_id: int,
    dataset_stats: str,
    memory_summary: str,
    model: str,
    daily_limit_usd: float = 5.0,
) -> str:
    user_message = f"""Датасет: {dataset_stats[:300]}
История (не повторяй): {memory_summary[:400]}
Агент #{agent_id}: одна нестандартная идея для build_model() (2-4 предложения, без кода):"""

    messages = [
        {"role": "system", "content": "ML исследователь. Предложи одну нестандартную идею. Конкретно и коротко."},
        {"role": "user", "content": user_message},
    ]
    try:
        response, _, _ = _call_openrouter(messages, model, max_tokens=400, daily_limit_usd=daily_limit_usd)
        return response.strip()
    except Exception as e:
        return f"[идея недоступна: {e}]"


def discuss_approaches(
    agent_id: int,
    my_proposal: str,
    other_proposals: list[str],
    memory_summary: str,
    model: str,
    daily_limit_usd: float = 5.0,
) -> str:
    others = "\n\n".join(f"Агент #{i}: {p[:300]}" for i, p in enumerate(other_proposals) if i != agent_id)
    user_message = f"""История: {memory_summary[:400]}
Коллеги: {others}
Моё предложение (агент #{agent_id}): {my_proposal}
Финальный план (3-5 предложений):"""

    messages = [
        {"role": "system", "content": "ML исследователь. Учти идеи коллег, дай конкретный финальный план для build_model(). Без воды."},
        {"role": "user", "content": user_message},
    ]
    try:
        response, _, _ = _call_openrouter(messages, model, max_tokens=500, daily_limit_usd=daily_limit_usd)
        return response.strip()
    except Exception as e:
        return f"[обсуждение недоступно: {e}]"


def analyze_results(
    run_id: str,
    approach_description: str,
    roc_auc: Optional[float],
    stdout: str,
    stderr: str,
    train_code: str,
    model: str,
    daily_limit_usd: float = 5.0,
) -> str:
    roc_str = f"{roc_auc:.6f}" if roc_auc else "НЕ ПОЛУЧЕН"
    user_message = f"""Run: {run_id} | ROC-AUC: {roc_str}
Подход: {approach_description[:400]}
STDOUT: {stdout[-1500:]}
STDERR: {stderr[-500:]}
Анализ (3-5 предложений, что конкретно пошло не так и что изменить):"""

    messages = [
        {"role": "system", "content": "Аналитик ML. Факты из stdout/stderr, конкретные рекомендации. Без воды."},
        {"role": "user", "content": user_message},
    ]
    try:
        response, _, _ = _call_openrouter(messages, model, max_tokens=500, daily_limit_usd=daily_limit_usd)
        return response.strip()
    except Exception as e:
        return f"Анализ недоступен: {e}"
