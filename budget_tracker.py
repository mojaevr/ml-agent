"""
budget_tracker.py — отслеживание расходов на токены через OpenRouter.
Hard stop при превышении дневного лимита.
"""
import json
import os
from datetime import date
from pathlib import Path

# Цены за 1M токенов (OpenRouter, актуальные)
MODEL_PRICES = {
    # DeepSeek
    "deepseek/deepseek-r1": {"input": 0.45, "output": 2.15},
    "deepseek/deepseek-chat": {"input": 0.32, "output": 0.89},       # DeepSeek V3
    "deepseek/deepseek-v3.2": {"input": 0.26, "output": 0.38},       # DeepSeek V3.2
    "deepseek/deepseek-v3.2-exp": {"input": 0.21, "output": 0.41},   # V3.2 Exp
    "deepseek/deepseek-chat-v3.1": {"input": 0.21, "output": 0.79},  # V3.1 Terminus
    # Qwen
    "qwen/qwen3-235b-a22b": {"input": 0.13, "output": 0.60},
    "qwen/qwen3-32b": {"input": 0.10, "output": 0.40},
}

BUDGET_FILE = Path("budget_log.json")


def _load() -> dict:
    if BUDGET_FILE.exists():
        with open(BUDGET_FILE) as f:
            return json.load(f)
    return {}


def _save(data: dict):
    with open(BUDGET_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_today_spent() -> float:
    data = _load()
    today = str(date.today())
    return data.get(today, {}).get("spent_usd", 0.0)


def record_usage(model: str, input_tokens: int, output_tokens: int) -> float:
    """Записывает использование токенов, возвращает стоимость этого вызова."""
    prices = MODEL_PRICES.get(model, {"input": 0.55, "output": 2.19})
    cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000

    data = _load()
    today = str(date.today())
    if today not in data:
        data[today] = {"spent_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}

    data[today]["spent_usd"] += cost
    data[today]["calls"] += 1
    data[today]["input_tokens"] += input_tokens
    data[today]["output_tokens"] += output_tokens
    _save(data)
    return cost


def check_budget(daily_limit_usd: float) -> tuple[bool, float]:
    """
    Проверяет, не превышен ли дневной лимит.
    Возвращает (можно_продолжать, потрачено_сегодня).
    """
    spent = get_today_spent()
    return spent < daily_limit_usd, spent


def print_budget_summary(daily_limit_usd: float):
    data = _load()
    today = str(date.today())
    today_data = data.get(today, {})
    spent = today_data.get("spent_usd", 0.0)
    calls = today_data.get("calls", 0)
    print(f"\n💰 Бюджет сегодня: ${spent:.4f} / ${daily_limit_usd:.2f} ({calls} вызовов API)")
    remaining = daily_limit_usd - spent
    print(f"   Остаток: ${remaining:.4f}")
