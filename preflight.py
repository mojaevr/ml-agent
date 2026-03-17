"""
preflight.py — статическая проверка кода агента перед запуском.

Ловит распространённые ошибки без запуска кода.
"""

import ast
import re
from typing import List


def check(code: str) -> List[str]:
    """
    Проверяет train_code.py агента.

    Returns:
        Список строк с описаниями проблем. Пустой список = всё OK.
    """
    issues = []

    # ── 1. Синтаксис ─────────────────────────────────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    # ── 2. Обязательные функции ───────────────────────────────────────────────
    top_level_funcs = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _is_top_level(node, tree)
    }

    if "build_features" not in top_level_funcs:
        issues.append("Нет функции build_features(dataset) на верхнем уровне")

    if "build_model" not in top_level_funcs:
        issues.append("Нет функции build_model(dataset) на верхнем уровне")

    # ── 3. Классы внутри функций (pickle не работает) ────────────────────────
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.ClassDef) and child is not node:
                    issues.append(
                        f"Класс '{child.name}' определён внутри функции '{node.name}' — "
                        f"перенеси его на верхний уровень файла (иначе pickle сломается)"
                    )

    # ── 4. Устаревший LightGBM API ────────────────────────────────────────────
    if re.search(r"\bverbose_eval\s*=", code):
        issues.append(
            "verbose_eval= устарел в LightGBM >= 4.0. "
            "Используй: callbacks=[lgb.log_evaluation(100)]"
        )
    if re.search(r"early_stopping_rounds\s*=\s*\d+", code) and ".fit(" in code:
        issues.append(
            "early_stopping_rounds= в .fit() устарел в LightGBM >= 4.0. "
            "Используй: callbacks=[lgb.early_stopping(50)]"
        )

    # ── 5. Прямой вызов exit/quit (убивает весь процесс) ─────────────────────
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("exit", "quit", "sys.exit"):
                issues.append(
                    f"Вызов {func.id}() убьёт весь runner. "
                    f"Используй raise вместо exit."
                )

    # ── 6. predict_proba должен возвращать 2D массив ──────────────────────────
    # (Эвристика: если агент переопределяет predict_proba)
    if "predict_proba" in code:
        if "[:, 1]" not in code and "column_stack" not in code and "hstack" not in code:
            # Слабая проверка: не блокируем, просто предупреждаем
            issues.append(
                "ПРЕДУПРЕЖДЕНИЕ: predict_proba() должна возвращать array shape (n, 2), "
                "где [:, 1] — вероятности положительного класса"
            )

    return issues


def _is_top_level(func_node: ast.FunctionDef, tree: ast.Module) -> bool:
    """Проверяет, что функция определена на верхнем уровне модуля."""
    return func_node in ast.walk(tree) and any(
        func_node is stmt for stmt in tree.body
    )
