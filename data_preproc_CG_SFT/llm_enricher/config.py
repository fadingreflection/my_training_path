# llm_enricher/config.py

# --- OpenRouter API Configuration ---
TEACHER_MODEL = "deepseek/deepseek-chat"#"nvidia/nemotron-3-ultra-550b-a55b:free"
API_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"  # переменная окружения с ключом

# --- Generation parameters ---
MAX_TOKENS = 4096
TEMPERATURE = 0.2

# --- Concurrency and logging ---
MAX_WORKERS = 10          # число параллельных запросов
RATE_LIMIT_SLEEP = 0.5   # пауза между запросами (сек)
LOG_FILE = "./logs/teacher_response.log"

# --- File paths ---
RAW_DATA_PATH = "../preprocessor/output/raw_data_parsed_shuffled_balanced.jsonl"
OUTPUT_PATH = "./output/sft_data_enriched.jsonl"