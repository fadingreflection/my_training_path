#!/usr/bin/env python3
"""
Точка входа для SFT-пайплайна.
Запуск: python main.py [путь_к_config.yaml]
"""

import sys
from pathlib import Path

from pipeline.pipeline import SFTLPipeline


def main():
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not Path(config_path).exists():
        print(f"Ошибка: файл конфига не найден: {config_path}")
        sys.exit(1)

    pipeline = SFTLPipeline(config_path)
    pipeline.run()

if __name__ == "__main__":
    main()