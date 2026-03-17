# ML Agent — автономная оптимизация ROC-AUC

Система запускает LLM-агента (DeepSeek/Qwen через OpenRouter) для автоматической
итеративной оптимизации бинарной классификации.

## Структура

```
ml_agent/
├── orchestrator.py      ← главный скрипт, запускать его
├── agent_loop.py        ← общение с LLM
├── sandbox_runner.py    ← изолированный запуск кода агента
├── validator.py         ← независимый подсчёт ROC-AUC (агент не трогает)
├── memory.py            ← история попыток
├── budget_tracker.py    ← учёт расходов, hard stop
├── data_prep.py         ← подготовка данных
├── requirements.txt
├── .env.example
│
├── data/
│   ├── train_full.csv   ← [вы кладёте сюда]
│   ├── test.csv         ← [вы кладёте сюда]
│   ├── train.csv        ← генерируется автоматически (80%)
│   └── val.csv          ← генерируется автоматически (20%), агент не видит
│
├── runs/
│   └── run_20240101_120000/
│       ├── train_code.py    ← код обучения этой итерации
│       ├── predict_code.py  ← код предикта на test.csv
│       ├── model.pkl        ← сохранённая модель
│       ├── val_preds.npy    ← предикты на val
│       └── meta.json        ← метрика, анализ, стоимость
│
├── memory/
│   └── attempts_log.json    ← история всех попыток
└── budget_log.json          ← расходы по дням
```

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка API ключа
```bash
cp .env.example .env
# Отредактируйте .env, вставьте OPENROUTER_API_KEY
export OPENROUTER_API_KEY=sk-or-v1-...
```
Получить ключ: https://openrouter.ai/keys

### 3. Подготовка данных
```bash
mkdir -p data
cp /путь/к/train.csv data/train_full.csv
cp /путь/к/test.csv data/test.csv

python orchestrator.py --prepare_data
```

### 4. Запуск агента
```bash
# 10 итераций, лимит $3/день, модель DeepSeek-R1
python orchestrator.py --iterations 10 --budget 3.0 --model deepseek/deepseek-r1

# Дешевле — Qwen3
python orchestrator.py --iterations 20 --budget 2.0 --model qwen/qwen3-235b-a22b

# Баланс цена/качество
python orchestrator.py --iterations 15 --budget 3.0 --model deepseek/deepseek-v3
```

### 5. Просмотр результатов
```bash
python orchestrator.py --list
```

### 6. Генерация сабмита для Kaggle
```bash
# Из лучшего run автоматически
python orchestrator.py --submit

# Из конкретного run
python orchestrator.py --submit --run_id run_20240101_143022 --output my_submission.csv
```

## Модели и цены (OpenRouter)

| Модель | Цена input | Цена output | Рекомендация |
|--------|-----------|-------------|--------------|
| `deepseek/deepseek-r1` | $0.55/M | $2.19/M | Лучший reasoning |
| `deepseek/deepseek-v3` | $0.27/M | $1.10/M | Хороший баланс |
| `qwen/qwen3-235b-a22b` | $0.13/M | $0.60/M | Максимум итераций |

При бюджете $3/день и контексте ~15k токенов:
- DeepSeek-R1: ~8-12 итераций
- DeepSeek-V3: ~15-20 итераций  
- Qwen3-235B: ~25-35 итераций

## Как работает агент

1. **Генерация кода**: LLM получает статистику датасета и историю попыток.
   Пишет `train_code.py` и `predict_code.py`. Каждый раз — новый контекст.
   
2. **Sandbox**: Код запускается в subprocess с таймаутом 10 минут.
   Агент не имеет доступа к `val.csv`.

3. **Валидация**: `validator.py` независимо считает ROC-AUC по `val_preds.npy`.
   Агент не может влиять на способ валидации.

4. **Анализ**: LLM видит stdout/stderr/метрику и пишет конкретный диагноз
   (переобучение? плохой feature engineering? ошибка данных?).

5. **Memory**: История сохраняется в `memory/attempts_log.json`.
   Агент читает её чтобы не повторять провальные подходы.

## Что делает агент с данными

Агент может:
- Любой feature engineering с `train.csv`
- Нормализация, трансформации, создание новых фичей
- Выбор алгоритма (LightGBM, XGBoost, нейросети, ансамбли)
- Настройка гиперпараметров
- Стекинг и блендинг моделей

Агент НЕ может:
- Читать `val.csv` (он его не знает)
- Менять способ подсчёта ROC-AUC
- Переписывать `validator.py`
