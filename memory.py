"""
memory.py — хранит историю всех попыток агента.
Агент читает memory чтобы не повторяться и анализировать что не сработало.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

MEMORY_FILE = Path("memory/attempts_log.json")


def _load() -> list[dict]:
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return []


def _save(data: list[dict]):
    MEMORY_FILE.parent.mkdir(exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_attempt(
    run_id: str,
    roc_auc: Optional[float],
    approach_description: str,
    agent_analysis: str,
    error: Optional[str] = None,
    cost_usd: float = 0.0,
    duration_seconds: float = 0.0,
):
    """Добавляет результат итерации в память."""
    data = _load()
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "roc_auc": roc_auc,
        "approach_description": approach_description,
        "agent_analysis": agent_analysis,
        "error": error,
        "cost_usd": cost_usd,
        "duration_seconds": duration_seconds,
        "status": "error" if error else ("success" if roc_auc else "no_metric"),
    }
    data.append(entry)
    _save(data)
    return entry


def get_all_attempts() -> list[dict]:
    return _load()


def get_best_attempt() -> Optional[dict]:
    attempts = [a for a in _load() if a.get("roc_auc") is not None]
    if not attempts:
        return None
    return max(attempts, key=lambda x: x["roc_auc"])


def get_summary_for_agent(max_attempts: int = 15) -> str:
    """
    Возвращает краткое summary для передачи агенту в промпте.
    Только последние max_attempts попыток чтобы не раздувать контекст.
    """
    attempts = _load()
    if not attempts:
        return "Попыток ещё не было. Ты первый!"

    best = get_best_attempt()
    best_roc = best["roc_auc"] if best else None

    lines = [f"=== ИСТОРИЯ ПОПЫТОК (последние {min(len(attempts), max_attempts)}) ==="]
    if best_roc:
        lines.append(f"Лучший результат на данный момент: ROC-AUC = {best_roc:.6f} (run {best['run_id']})\n")

    # Последние попытки
    recent = attempts[-max_attempts:]
    for a in recent:
        roc_str = f"{a['roc_auc']:.6f}" if a.get("roc_auc") else "N/A (ошибка)"
        lines.append(f"[{a['run_id']}] ROC-AUC: {roc_str}")
        lines.append(f"  Подход: {a['approach_description'][:200]}")
        lines.append(f"  Анализ: {a['agent_analysis'][:300]}")
        if a.get("error"):
            lines.append(f"  Ошибка: {a['error'][:150]}")
        lines.append("")

    # Список подходов которые уже пробовали — агент должен избегать точных повторов
    tried = [a["approach_description"][:100] for a in attempts]
    lines.append("=== УЖЕ ПРОБОВАЛИ (не повторяй точно) ===")
    for i, t in enumerate(tried, 1):
        lines.append(f"{i}. {t}")

    return "\n".join(lines)


def get_failed_approaches() -> list[str]:
    """Возвращает описания подходов с плохим результатом."""
    attempts = _load()
    best = get_best_attempt()
    best_roc = best["roc_auc"] if best else 0.5
    failed = []
    for a in attempts:
        roc = a.get("roc_auc")
        if roc is None or roc < best_roc - 0.01:
            failed.append(a["approach_description"][:150])
    return failed
