# --- Paths for Judge ---
JUDGE_INPUT_PATH = "../llm_enricher/output/sft_data_enriched.jsonl"
JUDGE_OUTPUT_CLEAN = "./output/sft_data_clean.jsonl"
JUDGE_OUTPUT_REJECTED = "./output/sft_data_rejected.jsonl"
JUDGE_OUTPUT_CORRECTED = "./output/sft_data_corrected.jsonl"

# --- OpenRouter API Configuration ---
# Список моделей для голосования (3 модели)
JUDGE_MODELS = [
    "deepseek/deepseek-chat",                     # ваша текущая модель
    "anthropic/claude-opus-5-fast",    
    "google/gemini-3.7-flash"              
]
# Оставляем для обратной совместимости (если где-то используется)
TEACHER_MODEL = JUDGE_MODELS[0]   # по умолчанию первая

API_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"

# --- Generation parameters ---
MAX_TOKENS = 4096
TEMPERATURE = 0.2

# --- Concurrency and logging ---
MAX_WORKERS = 5
RATE_LIMIT_SLEEP = 2.0
LOG_FILE = "./logs/teacher_response.log"

# --- Judge sampling (для smoke test) ---
JUDGE_SAMPLE_SIZE = None   # если 0 или None – обрабатывать все записи

# --- Resume support ---
RESUME_FILE = "./judge_progress.txt"
SAVE_EVERY = 50   # сохранять состояние каждые N записей