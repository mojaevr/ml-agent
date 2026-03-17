"""
sandbox_runner.py — запускает код агента в subprocess с таймаутом и изоляцией.
Агент не может модифицировать val.csv или validator.py.
"""
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Optional


TIMEOUT_SECONDS = 480  # 8 минут — соответствует ограничению в промпте агента

# Корень проекта = директория где лежит этот файл
PROJECT_ROOT = Path(__file__).parent.resolve()


def run_training_code(
    code_path: str,
    run_dir: str,
    timeout: int = TIMEOUT_SECONDS,
    extra_env: Optional[dict] = None,
) -> dict:
    """
    Запускает train_code.py агента в subprocess.

    Возвращает:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "duration_seconds": float,
            "exit_code": int,
            "timeout": bool,
        }
    """
    # Все пути — абсолютные, чтобы не зависеть от cwd
    abs_code_path = str(Path(code_path).resolve())
    abs_run_dir = str(Path(run_dir).resolve())
    abs_train_path = str((PROJECT_ROOT / "data" / "train_full.csv").resolve())

    env = os.environ.copy()
    env["AGENT_RUN_DIR"] = abs_run_dir
    env["AGENT_TRAIN_PATH"] = abs_train_path
    env["AGENT_RUNNER_PATH"] = str((PROJECT_ROOT / "data" / "runner.py").resolve())
    env["AGENT_MODE"] = "train"
    # Намеренно НЕ передаём val.csv — агент не знает путь к нему
    # Агент сохраняет предикты в AGENT_RUN_DIR/val_preds.npy

    if extra_env:
        env.update(extra_env)

    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, abs_code_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(PROJECT_ROOT),  # всегда корень проекта
        )
        duration = time.time() - start
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-5000:] if result.stdout else "",
            "stderr": result.stderr[-3000:] if result.stderr else "",
            "duration_seconds": duration,
            "exit_code": result.returncode,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return {
            "success": False,
            "stdout": "",
            "stderr": f"TIMEOUT: код выполнялся дольше {timeout}с и был остановлен",
            "duration_seconds": duration,
            "exit_code": -1,
            "timeout": True,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"SANDBOX ERROR: {e}",
            "duration_seconds": time.time() - start,
            "exit_code": -1,
            "timeout": False,
        }


def run_predict_code(
    code_path: str,
    run_dir: str,
    timeout: int = TIMEOUT_SECONDS,
) -> dict:
    """
    Запускает predict_code.py для генерации предиктов на test.csv.
    Используется при генерации сабмита для Kaggle.
    """
    abs_code_path = str(Path(code_path).resolve())
    abs_run_dir = str(Path(run_dir).resolve())
    abs_test_path = str((PROJECT_ROOT / "data" / "test.csv").resolve())

    env = os.environ.copy()
    env["AGENT_RUN_DIR"] = abs_run_dir
    env["AGENT_TEST_PATH"] = abs_test_path
    env["AGENT_RUNNER_PATH"] = str((PROJECT_ROOT / "data" / "runner.py").resolve())
    # AGENT_MODE set by predict_code.py itself

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, abs_code_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-5000:],
            "stderr": result.stderr[-3000:],
            "duration_seconds": time.time() - start,
            "exit_code": result.returncode,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "TIMEOUT при генерации предиктов",
            "duration_seconds": time.time() - start,
            "exit_code": -1,
            "timeout": True,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "duration_seconds": time.time() - start,
            "exit_code": -1,
            "timeout": False,
        }
