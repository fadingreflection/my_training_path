#!/usr/bin/env python3
"""
Универсальный пайплайн сбора данных для архитектурного аналитика.
Использует OpenRouter API (бесплатные модели) через OpenAI-совместимый SDK.
"""

import os
import json
import subprocess
import tempfile
import shutil
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =========== КОНФИГУРАЦИЯ ===========
CONFIG = {
    "depth": 5,
    "ignore_dirs": {
        "venv", ".venv", "env", "node_modules", ".git", "__pycache__",
        ".idea", ".vscode", "dist", "build", "target",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "logs", "tmp",
        "coverage", ".coverage", "*.egg-info", "eggs"
    },
    "teacher_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "api_base_url": "https://openrouter.ai/api/v1",
    "api_key_env_var": "OPENROUTER_API_KEY",
    "max_tokens": 4096,
    "temperature": 0.2,
    "max_workers": 1,
    "rate_limit_sleep": 2.0,
    "log_file": "teacher_response.log",
}

# =========== ПРОМПТ ИЗ ФАЙЛА ===========
def load_system_prompt(filepath: str = "architect_prompt.txt") -> str:
    """Загружает системный промпт из текстового файла."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT = load_system_prompt("architect_prompt.txt")

# =========== КЛАСС 1: ИЗВЛЕКАТЕЛЬ СТРУКТУРЫ ===========

class RepositoryStructureExtractor:
    """
    Извлекает структуру репозитория (дерево папок/файлов) на заданную глубину,
    игнорируя служебные и шумные директории.
    """
    def __init__(self, depth: int = CONFIG["depth"], ignore_dirs: set = CONFIG["ignore_dirs"]):
        self.depth = depth
        self.ignore_dirs = ignore_dirs

    def get_repo_structure(self, repo_path: Path) -> Dict[str, Any]:
        """
        Рекурсивно обходит репозиторий и возвращает дерево.
        """
        def _walk(path: Path, current_depth: int):
            if current_depth == 0:
                return None
            result = {"directories": [], "files": []}
            try:
                for item in sorted(path.iterdir()):
                    name = item.name
                    if name in self.ignore_dirs or name.startswith('.'):
                        continue
                    if item.is_dir():
                        child = _walk(item, current_depth - 1)
                        if child and (child["directories"] or child["files"]):
                            result["directories"].append({"name": name, "children": child})
                    elif item.is_file():
                        if not name.endswith(('.pyc', '.pyo', '.so', '.dll')):
                            result["files"].append(name)
            except (PermissionError, OSError):
                pass
            return result

        root = _walk(repo_path, self.depth)
        return root if root else {"directories": [], "files": []}

    def flatten_structure(self, root, parent_path: str = "") -> Dict[str, Any]:
        """
        Преобразует дерево в плоский словарь вида:
        {
          "root_directories": [...],
          "root_files": [...],
          "src": {"directories": [...], "files": [...]},
          ...
        }
        """
        if root is None or isinstance(root, list):
            root = {"directories": [], "files": []}
        if not isinstance(root, dict):
            root = {"directories": [], "files": []}

        result = {}
        root_dirs = []
        root_files = []

        directories = root.get("directories", [])
        if not isinstance(directories, list):
            directories = []

        for d in directories:
            if not isinstance(d, dict):
                continue
            name = d.get("name", "unknown")
            root_dirs.append(name)
            child_path = f"{parent_path}/{name}" if parent_path else name

            children = d.get("children", {})
            if children is None or isinstance(children, list) or not isinstance(children, dict):
                children = {"directories": [], "files": []}
            child_flat = self.flatten_structure(children, child_path)
            result.update(child_flat)

        files = root.get("files", [])
        if not isinstance(files, list):
            files = []
        root_files.extend(files)

        current_key = parent_path if parent_path else "root"
        result[current_key] = {
            "directories": root_dirs if parent_path else root_dirs,
            "files": root_files if parent_path else root_files
        }

        if not parent_path:
            result["root_directories"] = root_dirs
            result["root_files"] = root_files
            if "root" in result:
                del result["root"]

        return result

    def build_repository_structure_json(self, repo_path: Path) -> Dict[str, Any]:
        """
        Основной метод: собирает структуру и возвращает JSON в формате,
        ожидаемом учителем и учеником.
        """
        tree = self.get_repo_structure(repo_path)
        if isinstance(tree, list):
            tree = {"directories": [], "files": []}
        flat = self.flatten_structure(tree, "")

        # Фильтруем пустые записи
        flat_filtered = {}
        for k, v in flat.items():
            if isinstance(v, dict) and (v.get("directories") or v.get("files")):
                flat_filtered[k] = v
            elif isinstance(v, list) and v:
                flat_filtered[k] = {"directories": [], "files": v}

        return {"repository_structure": flat_filtered}


# =========== КЛАСС 2: ГЕНЕРАТОР ОПИСАНИЯ ===========

class ArchitectureDescriptionGenerator:
    """
    Отправляет структуру репозитория в LLM-учитель и получает архитектурное описание.
    Сохраняет логи ответов и формирует обучающие примеры.
    """
    def __init__(
        self,
        api_key: str,
        base_url: str = CONFIG["api_base_url"],
        model: str = CONFIG["teacher_model"],
        max_tokens: int = CONFIG["max_tokens"],
        temperature: float = CONFIG["temperature"],
        system_prompt: str = SYSTEM_PROMPT,
        log_file: str = CONFIG["log_file"],
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Data Collector Pipeline",
            }
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.log_file = Path(log_file)

    def _call_api(self, structure_json: Dict[str, Any]) -> str:
        """Выполняет запрос к API и возвращает сырой ответ."""
        user_content = json.dumps(structure_json, indent=2)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": (
                    f"Repository structure (JSON):\n{user_content}\n\n"
                    f"Provide the architectural description as valid JSON. "
                    f"Return ONLY the JSON object, without any markdown formatting, "
                    f"explanations, or extra text."
                )}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _log_response(self, content: str) -> None:
        """Записывает сырой ответ в лог-файл."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"=== MODEL: {self.model} ===\n")
            f.write(f"=== FULL RESPONSE ===\n{content}\n")
            f.write("=" * 80 + "\n\n")
        print(f"📩 Full response logged to {self.log_file}")

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Парсит JSON из ответа, удаляя Markdown и лишний текст."""
        # Удаляем BOM и Markdown
        cleaned = content.replace('\ufeff', '')
        cleaned = re.sub(r'```json\s*', '', cleaned)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()

        # Пробуем распарсить напрямую
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Ищем JSON-объект
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error: {e}")
                print(f"🔍 Extracted JSON (first 500 chars):\n{json_str[:500]}")
                # Пробуем json_repair, если доступен
                try:
                    from json_repair import repair_json
                    repaired = repair_json(json_str)
                    return json.loads(repaired)
                except ImportError:
                    raise ValueError(f"Invalid JSON. First 500 chars: {json_str[:500]}")

        raise ValueError(f"No JSON object found. First 500 chars: {cleaned[:500]}")

    def generate_description(self, structure_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод: отправляет структуру, логирует ответ и возвращает
        распарсенный JSON с архитектурным описанием.
        """
        content = self._call_api(structure_json)
        self._log_response(content)
        return self._parse_json_response(content)

    @staticmethod
    def create_training_sample(
        structure_json: Dict[str, Any],
        description_json: Dict[str, Any],
        depth: int = CONFIG["depth"]
    ) -> Dict[str, Any]:
        """Формирует обучающий пример в формате messages для SFT."""
        system_content = (
            "You are an architecture analyst. Your task is to analyze the given "
            "repository structure and produce a high-level architectural description "
            "of the application."
        )

        user_content = (
            f"Repository structure (up to depth {depth}):\n"
            f"{json.dumps(structure_json, indent=2)}\n\n"
            f"Task:\n"
            f"1. Determine the type of application (web, CLI, library, desktop, etc.) "
            f"based on the directory names, file names, and any clues from the structure.\n"
            f"2. Identify the main components (modules) and their responsibilities. "
            f"Use the directory and file names to infer their roles.\n"
            f"3. Suggest which subdirectories (deeper than depth 2) are likely to contain "
            f"critical business logic or core functionality, and therefore should be "
            f"explored further to understand the application's behavior.\n\n"
            f"Return your answer ONLY in the following JSON format:\n"
            f"{{\n"
            f'  "app_type": "string",\n'
            f'  "components": [\n'
            f'    {{"name": "string", "path": "string", "responsibility": "string"}}\n'
            f'  ],\n'
            f'  "further_exploration": [\n'
            f'    {{"path": "string", "reasoning": "string"}}\n'
            f'  ]\n'
            f'}}'
        )

        assistant_content = json.dumps(description_json, indent=2)

        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }


# =========== ФУНКЦИИ ПАЙПЛАЙНА ===========

def process_repository(
    repo_url: str,
    temp_dir: Path,
    extractor: RepositoryStructureExtractor,
    generator: ArchitectureDescriptionGenerator
) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает один репозиторий: клонирует, собирает структуру,
    получает описание, формирует sample.
    """
    repo_name = repo_url.rstrip('/').split('/')[-1]
    if not repo_name:
        repo_name = f"repo_{int(time.time())}"

    clone_path = temp_dir / repo_name

    # Клонирование
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(clone_path)],
            check=True,
            capture_output=True,
            timeout=120
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone {repo_url}: {e.stderr.decode()[:200]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout cloning {repo_url}")
        return None

    # Проверка папки
    if not clone_path.exists() or not clone_path.is_dir():
        print(f"❌ Clone directory does not exist: {clone_path}")
        return None

    # Проверка содержимого (кроме .git)
    has_content = any(item.name != '.git' for item in clone_path.iterdir())
    if not has_content:
        print(f"❌ Cloned repository has no content: {clone_path}")
        return None

    try:
        # Сбор структуры
        structure = extractor.build_repository_structure_json(clone_path)

        # Проверка структуры
        repo_struct = structure.get("repository_structure", {})
        if not repo_struct.get("root_directories") and not repo_struct.get("root_files"):
            print(f"❌ Repository structure is empty for {repo_url}")
            return None

        # Генерация описания
        description = generator.generate_description(structure)

        # Валидация описания
        required_fields = {"app_type", "components", "further_exploration"}
        if not isinstance(description, dict) or not required_fields.issubset(description.keys()):
            print(f"❌ Teacher response missing required fields: {description.keys()}")
            return None

        # Формирование sample
        sample = generator.create_training_sample(structure, description, CONFIG["depth"])

        if not isinstance(sample, dict) or "messages" not in sample:
            print(f"❌ Sample is malformed for {repo_url}")
            return None

        print(f"✅ Processed: {repo_name}")
        return sample

    except Exception as e:
        print(f"❌ Error processing {repo_url}: {e}")
        return None
    finally:
        shutil.rmtree(clone_path, ignore_errors=True)


def run_pipeline(
    repo_list: List[str],
    output_file: str,
    extractor: RepositoryStructureExtractor,
    generator: ArchitectureDescriptionGenerator,
    max_workers: int = CONFIG["max_workers"],
    rate_limit_sleep: float = CONFIG["rate_limit_sleep"]
) -> None:
    """
    Запускает пайплайн для списка репозиториев.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output: {output_path}")
    print(f"🤖 Teacher model: {CONFIG['teacher_model']}")
    print(f"📊 Total repos: {len(repo_list)}")
    print("-" * 50)

    successful = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)

        with open(output_path, 'a', encoding='utf-8') as f:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_repository, repo, temp_path, extractor, generator): repo
                    for repo in repo_list
                }

                for future in as_completed(futures):
                    repo = futures[future]
                    try:
                        sample = future.result(timeout=300)
                        if sample:
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                            f.flush()
                            successful += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"❌ Error in future for {repo}: {e}")
                        failed += 1

                    time.sleep(rate_limit_sleep)

    print("-" * 50)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output saved to: {output_path}")


# =========== ТОЧКА ВХОДА ===========
if __name__ == "__main__":
    # 1. Проверяем наличие API-ключа
    api_key = os.environ.get(CONFIG["api_key_env_var"])
    if not api_key:
        raise ValueError(
            f"Environment variable {CONFIG['api_key_env_var']} not set. "
            f"Get your API key at: https://openrouter.ai/"
        )

    # 2. Создаём экземпляры классов
    extractor = RepositoryStructureExtractor(
        depth=CONFIG["depth"],
        ignore_dirs=CONFIG["ignore_dirs"]
    )

    generator = ArchitectureDescriptionGenerator(
        api_key=api_key,
        base_url=CONFIG["api_base_url"],
        model=CONFIG["teacher_model"],
        max_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"],
        system_prompt=SYSTEM_PROMPT,
        log_file=CONFIG["log_file"]
    )

    # 3. Список репозиториев (можно загрузить из файла)
    repos = [
        # # === ОРИГИНАЛЬНЫЙ СПИСОК (33 репозитория) ===
        # "https://github.com/sunblaze-ucb/cybergym",
        # "https://github.com/tensorflow/tensorflow",
        # "https://github.com/electron/electron",
        # "https://github.com/facebook/react-native",
        # "https://github.com/opencv/opencv",
        # "https://github.com/bitcoin/bitcoin",
        # "https://github.com/microsoft/terminal",
        # "https://github.com/tesseract-ocr/tesseract",
        # "https://github.com/ggml-org/llama.cpp",
        # "https://github.com/ggml-org/whisper.cpp",
        # "https://github.com/nomic-ai/gpt4all",
        # "https://github.com/facebook/folly",
        # "https://github.com/pocoproject/poco",
        # "https://github.com/abseil/abseil-cpp",
        # "https://github.com/nlohmann/json",
        # "https://github.com/raysan5/raylib",
        # "https://github.com/libhv/libhv",
        # "https://github.com/oatpp/oatpp",
        # "https://github.com/ggml-org/ggml",
        # "https://github.com/IJackDaniel/DoodleJump",
        # "https://github.com/redis/redis",
        # "https://github.com/sqlite/sqlite",
        # "https://github.com/libuv/libuv",
        # "https://github.com/ggsurya/C-Projects",
        # "https://github.com/SonawaneAshwini/CProgrammingProjects",
        # "https://github.com/mehtadeven/C-Projects",
        # "https://github.com/BlackCat1503/Sh21",
        # "https://github.com/mesinkasir/cax",
        # "https://github.com/ChoiWheatley/cpp-algorithms",
        # "https://github.com/SMPesnya/c-cpp-embedded-katas",
        # "https://github.com/hugorbarbosa/cpp-project-template",
        # "https://github.com/thejohnfreeman/project-template-cpp",
        # "https://github.com/jgoossen851/cpp-project-template",

        # === НОВЫЕ РЕПОЗИТОРИИ (начиная отсюда) ===
        "https://github.com/TonyHuangYJ/libigl",
        "https://github.com/ramizouari/CPLibrary",
        "https://github.com/San7o/tenno-tl",
        "https://github.com/krulis-martin/bpplib",
        "https://github.com/MangaD/jsocketpp",
        "https://github.com/vinniefalco/beast2",
        "https://github.com/study-game-engines/sparcle",
        "https://github.com/dragosdragan03/Communication-Protocols",
        "https://github.com/Kabir-Narula/Socket-IO_CPP",
        "https://github.com/lizining1231/Evolutionary-Cpp-Network-Library",
        "https://github.com/softadastra/PulseGrid",
        "https://github.com/ITx-prash/securebank-cpp",
        "https://github.com/thefxng/Pulse",
        "https://github.com/TareqAlKushari/CPP",
        "https://github.com/MuthoniGathiithi/C-_Plus_Plus_Projects-",
        "https://github.com/AgvanGrigoryan/CPP-Modules",
        "https://github.com/cristibercea/OOP_in_CPP",
        "https://github.com/VSCRM/Project_zero",
        "https://github.com/YashJ-1171/Task-Scheduler-Optimization-System",
        "https://github.com/KwabenaSarkodie/Group-12---OOP-Project---Library-management-system",
        "https://github.com/Muril0EN/assistive-tech-prototypes",
        "https://github.com/MangaD/cpp-project-template",
    ]

    # 4. Запуск пайплайна
    run_pipeline(
        repo_list=repos,
        output_file="architecture_samples.jsonl",
        extractor=extractor,
        generator=generator,
        max_workers=CONFIG["max_workers"],
        rate_limit_sleep=CONFIG["rate_limit_sleep"]
    )