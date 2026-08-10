#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# Добавим корень проекта в пути, чтобы импортировать общие утилиты (опционально)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(benchmark_name):
    config_path = PROJECT_ROOT / "benchmarks" / "configs" / f"{benchmark_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_benchmark(config, model_dir, output_dir):
    model_dir = str(model_dir)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if config["type"] == "python_script":
        # Собираем аргументы командной строки из словаря args
        args = config.get("args", {})
        cmd = ["python", config["script_path"]]
        for key, value in args.items():
            # Если значение содержит плейсхолдеры, заменяем
            if isinstance(value, str):
                value = value.format(model_dir=model_dir, output_dir=output_dir)
            cmd.extend([f"--{key}", str(value)])
        working_dir = config.get("working_dir", None)
        
    elif config["type"] == "cli":
        # Команда уже содержит плейсхолдеры
        command = config["command"].format(model_dir=model_dir, output_dir=output_dir)
        cmd = command.split()  
        working_dir = config.get("working_dir", None)
    else:
        raise ValueError(f"Unknown benchmark type: {config['type']}")

    print(f"[{datetime.now()}] Запуск {config['name']}:")
    print(f"  Команда: {' '.join(cmd)}")
    print(f"  Рабочая папка: {working_dir or os.getcwd()}")
    
    # Запускаем процесс
    try:
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=False
        )
        # Сохраняем логи
        log_file = os.path.join(output_dir, f"{config['name']}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(log_file, "w") as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
        if result.returncode != 0:
            print(f"❌ Ошибка выполнения (код {result.returncode}). Лог сохранён в {log_file}")
            sys.exit(1)
        else:
            print(f"✅ Успешно завершено. Лог: {log_file}")
    except Exception as e:
        print(f"❌ Исключение: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Запуск бенчмарков из конфигов")
    parser.add_argument("benchmark", help="Имя бенчмарка (seceval, terminal_bench) или 'all'")
    parser.add_argument("--model_dir", default="./model/base", help="Путь к папке с моделью")
    parser.add_argument("--output_dir", default="./results", help="Папка для сохранения результатов")
    args = parser.parse_args()

    # Преобразуем относительные пути в абсолютные относительно корня проекта
    model_dir = (PROJECT_ROOT / args.model_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    
    if args.benchmark == "all":
        # Находим все .yaml в папке configs/benchmarks
        config_dir = PROJECT_ROOT / "configs" / "benchmarks"
        for config_file in config_dir.glob("*.yaml"):
            benchmark_name = config_file.stem
            print(f"\n===== Запуск {benchmark_name} =====")
            config = load_config(benchmark_name)
            run_benchmark(config, model_dir, output_dir)
    else:
        config = load_config(args.benchmark)
        run_benchmark(config, model_dir, output_dir)

if __name__ == "__main__":
    main()