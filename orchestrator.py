"""
orchestrator.py — главный цикл с параллельными агентами.

Порядок в одной итерации:
1. Обсуждение (proposal → discuss)
2. Генерация кода
3. Pre-flight проверка (синтаксис, паттерны) → LLM fix если нужно
4. Sandbox запуск + самолечение (auto pip install + LLM fix, до 3 раз)
5. Валидация ROC-AUC
6. Анализ результатов

Запуск:
    python orchestrator.py --iterations 5 --agents 2 --budget 3.0
    python orchestrator.py --submit
    python orchestrator.py --list
    python orchestrator.py --prepare_data
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent_loop import (
    MAX_FIX_ATTEMPTS,
    analyze_results,
    discuss_approaches,
    fix_code_after_error,
    fix_wrong_val_size,
    generate_training_code,
    preflight_check,
    propose_approach,
)
from budget_tracker import check_budget, print_budget_summary
from data_prep import get_dataset_stats, prepare_all, prepare_split
from memory import add_attempt, get_best_attempt, get_summary_for_agent
from sandbox_runner import run_predict_code, run_training_code
from validator import compute_roc_auc, generate_submission

RUNS_DIR = Path("runs")
DEFAULT_MODEL = "deepseek/deepseek-v3.2"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_run_id(agent_id: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_a{agent_id}"


def setup_run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_meta(run_dir: Path, meta: dict):
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_code(run_dir: Path, train_code: str, predict_code: Optional[str]):
    (run_dir / "train_code.py").write_text(train_code)
    if predict_code:
        (run_dir / "predict_code.py").write_text(predict_code)


# ─────────────────────────────────────────────────────────────
# Одна итерация одного агента
# ─────────────────────────────────────────────────────────────

def run_one_agent(
    agent_id: int,
    dataset_stats: str,
    boilerplate: str,
    model: str,
    daily_limit_usd: float,
    discussion_context: str = "",
    iteration_label: str = "",
) -> dict:
    run_id = make_run_id(agent_id)
    p = f"[A{agent_id}]"
    print(f"\n{'─'*55}")
    print(f"{p} 🚀 {run_id} {iteration_label}")
    print(f"{'─'*55}")

    run_dir = setup_run_dir(run_id)
    memory_summary = get_summary_for_agent()

    # ── 1. Генерация кода ──────────────────────────────────────
    print(f"{p} 🧠 Генерация кода...")
    try:
        train_code, predict_code, description, _ = generate_training_code(
            dataset_stats=dataset_stats,
            runner_code=boilerplate,
            memory_summary=memory_summary,
            run_id=run_id,
            model=model,
            daily_limit_usd=daily_limit_usd,
            agent_id=agent_id,
            discussion_context=discussion_context,
        )
    except RuntimeError as e:
        print(f"{p} ❌ LLM недоступен: {e}")
        return {"run_id": run_id, "roc_auc": None, "error": str(e), "agent_id": agent_id}

    if not train_code:
        msg = "LLM не вернул код"
        add_attempt(run_id, None, "Нет кода", msg, error=msg)
        return {"run_id": run_id, "roc_auc": None, "error": msg, "agent_id": agent_id}

    print(f"{p}    Подход: {description[:130]}...")

    # ── 2. Pre-flight проверка ─────────────────────────────────
    issues = preflight_check(train_code)
    if issues:
        print(f"{p} ⚠️  Pre-flight: {len(issues)} проблем(а)")
        for issue in issues:
            print(f"{p}    • {issue[:100]}")
        # Пробуем исправить через LLM только если код не слишком короткий
        ok, _ = check_budget(daily_limit_usd)
        if ok:
            issues_text = "\n".join(f"- {i}" for i in issues)
            fix_msg = f"""Исправь ТОЛЬКО указанные проблемы в train_code.py.
НЕ МЕНЯЙ логику build_model, только перенеси классы/функции на верхний уровень.

ПРОБЛЕМЫ:
{issues_text}

КОД:
```python
{train_code}
```
Верни полный исправленный код в блоке ```python # train_code.py```"""
            try:
                from agent_loop import _call_openrouter, _extract_all_code_blocks
                resp, _, _ = _call_openrouter(
                    [{"role": "user", "content": fix_msg}],
                    model, max_tokens=5000, daily_limit_usd=daily_limit_usd
                )
                blocks = _extract_all_code_blocks(resp)
                fixed = next((b for b in blocks if "build_model" in b), None) or (blocks[0] if blocks else None)
                if fixed and len(fixed) > len(train_code) * 0.7:
                    remaining = preflight_check(fixed)
                    if len(remaining) < len(issues):
                        train_code = fixed
                        print(f"{p} ✅ Pre-flight исправлен ({len(remaining)} осталось)")
                    else:
                        print(f"{p} ⚠️  Fix не помог, запускаем как есть")
            except Exception as e:
                print(f"{p} ⚠️  Pre-flight fix недоступен: {e}")
    else:
        print(f"{p} ✅ Pre-flight OK")

    save_code(run_dir, train_code, predict_code)

    # ── 3. Sandbox + самолечение ───────────────────────────────
    roc_auc = None
    sandbox_result = None
    fix_count = 0

    for attempt in range(MAX_FIX_ATTEMPTS + 1):
        label = "Запуск обучения" if attempt == 0 else f"Повторный запуск #{attempt}"
        print(f"{p} ⚙️  {label}...")

        sandbox_result = run_training_code(str(run_dir / "train_code.py"), str(run_dir))

        if sandbox_result["timeout"]:
            print(f"{p} ⏱️  TIMEOUT ({sandbox_result['duration_seconds']:.0f}с)")
            if attempt < MAX_FIX_ATTEMPTS:
                ok, _ = check_budget(daily_limit_usd)
                if ok:
                    fixed_train, fixed_predict = fix_code_after_error(
                        train_code=train_code,
                        predict_code=predict_code,
                        stdout=sandbox_result["stdout"],
                        stderr=sandbox_result["stderr"],
                        fix_attempt=attempt + 1,
                        model=model,
                        daily_limit_usd=daily_limit_usd,
                        is_timeout=True,
                    )
                    if fixed_train:
                        train_code = fixed_train
                        if fixed_predict:
                            predict_code = fixed_predict
                        fix_count += 1
                        save_code(run_dir, train_code, predict_code)
                        continue  # повторяем запуск с ускоренным кодом
            break

        if sandbox_result["success"]:
            dur = sandbox_result["duration_seconds"]
            print(f"{p} ✅ Обучение OK ({dur:.1f}с)")
            tail = sandbox_result["stdout"].strip().split("\n")[-3:]
            for line in tail:
                if line.strip():
                    print(f"{p}    {line}")
            break

        # Ошибка
        stderr_tail = sandbox_result["stderr"].strip().split("\n")[-3:]
        print(f"{p} ❌ Ошибка (exit {sandbox_result['exit_code']})")
        for line in stderr_tail:
            if line.strip():
                print(f"{p}    {line}")

        if attempt >= MAX_FIX_ATTEMPTS:
            print(f"{p} 🛑 Исчерпаны попытки ({MAX_FIX_ATTEMPTS})")
            break

        ok, _ = check_budget(daily_limit_usd)
        if not ok:
            print(f"{p} 💸 Бюджет исчерпан")
            break

        fixed_train, fixed_predict = fix_code_after_error(
            train_code=train_code,
            predict_code=predict_code,
            stdout=sandbox_result["stdout"],
            stderr=sandbox_result["stderr"],
            fix_attempt=attempt + 1,
            model=model,
            daily_limit_usd=daily_limit_usd,
        )

        if not fixed_train:
            print(f"{p} ❌ Fix не удался")
            break

        train_code = fixed_train
        if fixed_predict:
            predict_code = fixed_predict
        fix_count += 1

        # Pre-flight после каждого фикса
        issues = preflight_check(train_code)
        if issues:
            print(f"{p}    ⚠️  После фикса pre-flight: {[i[:60] for i in issues]}")

        save_code(run_dir, train_code, predict_code)

    # ── 4. Валидация ───────────────────────────────────────────
    print(f"{p} 📊 Валидация...")
    val_preds_path = run_dir / "val_preds.npy"

    if sandbox_result and sandbox_result["success"] and val_preds_path.exists():
        import numpy as np_check, pandas as pd_check
        # Проверяем размер ДО вычисления ROC-AUC
        try:
            preds_check = np_check.load(val_preds_path)
            if preds_check.ndim == 2:
                preds_check = preds_check[:, 1]
            val_expected = len(pd_check.read_csv("data/val.csv"))
            actual_size = len(preds_check)

            if actual_size != val_expected:
                print(f"{p} ⚠️  Размер val_preds={actual_size} ≠ val.csv={val_expected} — исправляю сплит...")
                ok, _ = check_budget(daily_limit_usd)
                if ok and fix_count < MAX_FIX_ATTEMPTS:
                    fixed_train, fixed_predict = fix_wrong_val_size(
                        train_code=train_code,
                        predict_code=predict_code,
                        actual_size=actual_size,
                        expected_size=val_expected,
                        model=model,
                        daily_limit_usd=daily_limit_usd,
                    )
                    if fixed_train:
                        train_code = fixed_train
                        if fixed_predict:
                            predict_code = fixed_predict
                        fix_count += 1
                        save_code(run_dir, train_code, predict_code)
                        # Повторный запуск
                        print(f"{p} ⚙️  Повторный запуск после исправления сплита...")
                        sandbox_result = run_training_code(str(run_dir / "train_code.py"), str(run_dir))
                        if not sandbox_result["success"] or not val_preds_path.exists():
                            print(f"{p} ❌ Повторный запуск не помог")
        except Exception as e:
            print(f"{p}    Ошибка проверки размера: {e}")

        roc_auc = compute_roc_auc(str(val_preds_path), "data/val.csv")
        if roc_auc is not None:
            best = get_best_attempt()
            best_roc = best["roc_auc"] if best else 0.0
            delta = roc_auc - best_roc if best_roc else 0
            delta_str = f" ({'+' if delta >= 0 else ''}{delta:.6f})" if best_roc else ""
            print(f"{p} 🎯 ROC-AUC = {roc_auc:.6f}{delta_str}")
        else:
            print(f"{p} ❌ Не удалось посчитать ROC-AUC")
    else:
        print(f"{p} ❌ val_preds.npy не создан")

    # ── 5. Анализ ──────────────────────────────────────────────
    print(f"{p} 🔍 Анализ...")
    ok, _ = check_budget(daily_limit_usd)
    if ok:
        analysis = analyze_results(
            run_id=run_id,
            approach_description=description,
            roc_auc=roc_auc,
            stdout=sandbox_result["stdout"] if sandbox_result else "",
            stderr=sandbox_result["stderr"] if sandbox_result else "",
            train_code=train_code,
            model=model,
            daily_limit_usd=daily_limit_usd,
        )
        print(f"{p}    {analysis[:200]}...")
    else:
        analysis = "Анализ пропущен (бюджет)"

    # ── Сохранение ─────────────────────────────────────────────
    error_msg = (sandbox_result["stderr"] if sandbox_result and not sandbox_result["success"] else None)
    meta = {
        "run_id": run_id, "agent_id": agent_id,
        "timestamp": datetime.now().isoformat(),
        "roc_auc": roc_auc, "approach": description, "analysis": analysis,
        "duration_seconds": sandbox_result["duration_seconds"] if sandbox_result else 0,
        "fix_attempts": fix_count, "model": model,
    }
    save_run_meta(run_dir, meta)
    add_attempt(
        run_id=run_id, roc_auc=roc_auc,
        approach_description=description, agent_analysis=analysis,
        error=error_msg,
        duration_seconds=meta["duration_seconds"],
    )

    best = get_best_attempt()
    if best and best["run_id"] == run_id and roc_auc:
        (RUNS_DIR / "best_run.txt").write_text(
            f"{run_id}\nROC-AUC: {roc_auc:.6f}\n{datetime.now().isoformat()}"
        )
        print(f"{p} 🏆 НОВЫЙ ЛУЧШИЙ: {roc_auc:.6f}")

    return {
        "run_id": run_id, "agent_id": agent_id, "roc_auc": roc_auc,
        "analysis": analysis, "description": description,
        "duration": meta["duration_seconds"],
    }


# ─────────────────────────────────────────────────────────────
# Параллельный раунд
# ─────────────────────────────────────────────────────────────

def run_parallel_round(
    n_agents: int,
    dataset_stats: str,
    boilerplate: str,
    model: str,
    daily_limit_usd: float,
    round_num: int,
    total_rounds: int,
    with_discussion: bool = True,
) -> list[dict]:
    print(f"\n{'═'*55}")
    print(f"🔄 РАУНД {round_num}/{total_rounds} | {n_agents} агентов")
    print(f"{'═'*55}")

    discussion_contexts = [""] * n_agents

    if with_discussion and n_agents > 1:
        ok, _ = check_budget(daily_limit_usd)
        if ok:
            print(f"\n💬 Фаза обсуждения...")
            memory_summary = get_summary_for_agent()
            proposals = {}

            with ThreadPoolExecutor(max_workers=n_agents) as pool:
                futures = {
                    pool.submit(propose_approach, i, dataset_stats, memory_summary, model, daily_limit_usd): i
                    for i in range(n_agents)
                }
                for f in as_completed(futures):
                    i = futures[f]
                    try:
                        proposals[i] = f.result()
                        print(f"   [A{i}] {proposals[i][:90]}...")
                    except Exception as e:
                        proposals[i] = f"[ошибка: {e}]"

            all_proposals = [proposals.get(i, "") for i in range(n_agents)]
            with ThreadPoolExecutor(max_workers=n_agents) as pool:
                futures = {
                    pool.submit(discuss_approaches, i, proposals.get(i, ""), all_proposals,
                                memory_summary, model, daily_limit_usd): i
                    for i in range(n_agents)
                }
                for f in as_completed(futures):
                    i = futures[f]
                    try:
                        discussion_contexts[i] = f.result()
                        print(f"   [A{i}] После обсуждения: {discussion_contexts[i][:80]}...")
                    except Exception as e:
                        discussion_contexts[i] = ""

    print(f"\n🚀 Параллельный запуск {n_agents} агентов...")
    results = []

    with ThreadPoolExecutor(max_workers=n_agents) as pool:
        futures = {
            pool.submit(
                run_one_agent,
                agent_id=i,
                dataset_stats=dataset_stats,
                boilerplate=boilerplate,
                model=model,
                daily_limit_usd=daily_limit_usd,
                discussion_context=discussion_contexts[i],
                iteration_label=f"[раунд {round_num}/{total_rounds}]",
            ): i
            for i in range(n_agents)
        }
        for f in as_completed(futures):
            agent_id = futures[f]
            try:
                results.append(f.result())
            except Exception as e:
                print(f"[A{agent_id}] ❌ Необработанная ошибка: {e}")
                results.append({"agent_id": agent_id, "roc_auc": None, "error": str(e)})

    successful = [r for r in results if r.get("roc_auc")]
    if successful:
        best = max(successful, key=lambda x: x["roc_auc"])
        print(f"\n📊 Итог раунда: {best['roc_auc']:.6f} (A{best['agent_id']})")
    else:
        print(f"\n📊 Раунд без успешных результатов")

    print_budget_summary(daily_limit_usd)
    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def generate_kaggle_submission(run_id: Optional[str] = None, output: Optional[str] = None):
    if run_id is None:
        best = get_best_attempt()
        if not best:
            print("❌ Нет успешных попыток")
            return
        run_id = best["run_id"]
        print(f"📦 Лучший run: {run_id} (ROC-AUC: {best['roc_auc']:.6f})")

    if output is None:
        output = f"{run_id}.csv"

    run_dir = RUNS_DIR / run_id
    predict_path = run_dir / "predict_code.py"
    if not predict_path.exists():
        print(f"❌ predict_code.py не найден в {run_dir}")
        return

    result = run_predict_code(str(predict_path), str(run_dir))
    if not result["success"]:
        print(f"❌ Predict failed:")
        print(result["stderr"][-800:])
        # Если AttributeError на pickle.load — runner нужно пересоздать (он устарел)
        if "AttributeError" in result["stderr"] and "Can't get attribute" in result["stderr"]:
            # Кастомный класс в pkl — старый run, нельзя починить без переобучения
            print()
            print("⚠️  Кастомный класс в model.pkl (старый run).")
            print("   Переобучить нельзя. Запустите новый раунд:")
            print("   python orchestrator.py --iterations 2 --agents 1 --budget 1.0")
            print("   Или выберите другой run: python orchestrator.py --list")
        # Пробуем заменить predict_code.py нашим стандартным
        from agent_loop import _DEFAULT_PREDICT_CODE
        predict_path.write_text(_DEFAULT_PREDICT_CODE)
        print("🔧 Заменяю predict_code.py стандартным, повторяю...")
        result = run_predict_code(str(predict_path), str(run_dir))
        if not result["success"]:
            print(f"❌ Не помогло: {result['stderr'][-300:]}")
            return

    test_preds = run_dir / "test_preds.npy"
    if not test_preds.exists():
        print(f"❌ test_preds.npy не создан")
        return

    if generate_submission(str(test_preds), "data/test.csv", output):
        print(f"✅ Сабмит: {output}")
        print(f"   kaggle competitions submit -f {output} -m '{run_id}'")


def list_runs():
    from memory import get_all_attempts
    attempts = get_all_attempts()
    if not attempts:
        print("Попыток ещё не было")
        return
    print(f"\n{'Run ID':<32} {'ROC-AUC':<12} {'Fix':<5} {'Время'}")
    print("─" * 58)
    for a in sorted(attempts, key=lambda x: x.get("roc_auc") or 0, reverse=True):
        roc = f"{a['roc_auc']:.6f}" if a.get("roc_auc") else "N/A   "
        status = "✅" if a.get("roc_auc") else "❌"
        dur = f"{a.get('duration_seconds', 0):.0f}s"
        print(f"{status} {a['run_id']:<30} {roc:<12} {dur}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-discussion", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="Показывать полный ответ LLM и код")
    args = parser.parse_args()

    if args.verbose:
        import agent_loop
        agent_loop.VERBOSE = True

    if args.list:
        list_runs(); return
    if args.submit:
        generate_kaggle_submission(run_id=args.run_id, output=args.output); return
    if args.prepare_data:
        prepare_all(); return

    # Проверка данных
    if not Path("data/train.csv").exists() or not Path("data/val.csv").exists():
        if Path("data/train_full.csv").exists():
            print("[!] Запускаю подготовку данных...")
            prepare_all()
        else:
            print("[!] Положите train_full.csv в data/ и запустите: python orchestrator.py --prepare_data")
            return

    from data_prep import SCHEMA_PATH, RUNNER_PATH, analyze_schema, generate_runner
    if not SCHEMA_PATH.exists():
        schema = analyze_schema()
        generate_runner(schema)
    elif not RUNNER_PATH.exists():
        with open(SCHEMA_PATH) as _f:
            import json as _json
            _schema = _json.load(_f)
        generate_runner(_schema)

    print(f"\n🤖 ML Agent | {args.model} | {args.agents} агентов | {args.iterations} раундов | ${args.budget}/день")
    print(f"📊 Загрузка схемы датасета...")

    dataset_stats, boilerplate = get_dataset_stats()
    RUNS_DIR.mkdir(exist_ok=True)

    all_results = []
    for rnd in range(1, args.iterations + 1):
        ok, spent = check_budget(args.budget)
        if not ok:
            print(f"\n💸 Лимит исчерпан (${spent:.4f}/${args.budget})")
            break

        results = run_parallel_round(
            n_agents=args.agents,
            dataset_stats=dataset_stats,
            boilerplate=boilerplate,
            model=args.model,
            daily_limit_usd=args.budget,
            round_num=rnd,
            total_rounds=args.iterations,
            with_discussion=not args.no_discussion,
        )
        all_results.extend(results)
        if rnd < args.iterations:
            time.sleep(2)

    # Итоги
    print(f"\n{'═'*55}\n📈 ИТОГИ\n{'═'*55}")
    successful = [r for r in all_results if r.get("roc_auc")]
    if successful:
        best_s = max(successful, key=lambda x: x["roc_auc"])
        print(f"Лучший: {best_s['roc_auc']:.6f} ({best_s['run_id']})")
    print(f"Успешных: {len(successful)}/{len(all_results)}")
    print_budget_summary(args.budget)

    best = get_best_attempt()
    if best:
        print(f"\n🏆 Лучший всего: {best['roc_auc']:.6f} ({best['run_id']})")
        print(f"   python orchestrator.py --submit")


if __name__ == "__main__":
    main()
