"""
Пилот teacher-цепочек (шаг 1): 30 функций, проверки качества, смета.

Настройка (env):  OPENROUTER_API_KEY, OPENROUTER_BASE_URL (если прокси), PILOT_MODEL,
                  REASONING_EFFORT (low|medium|high|max, по умолчанию low),
                  REASONING_MAX_TOKENS (опционально, жёсткий бюджет на reasoning)
Запуск:           python pilot_chains.py
Выход:            data/chains_pilot.jsonl, data/raw/<idx>.txt, data/pilot.log
"""
import json
import logging
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd


def _load_dotenv_fallback(path):
    """Мини-парсер .env без зависимостей: KEY=VALUE, # комментарии, кавычки снимаются."""
    f = Path(path)
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


try:
    from dotenv import load_dotenv as _ld
    _ld()
    _ld(Path(__file__).parent / ".env")
except ImportError:
    _load_dotenv_fallback(".env")
    _load_dotenv_fallback(Path(__file__).parent / ".env")


# ---------- логирование ----------
Path("data").mkdir(exist_ok=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/pilot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("pilot")


MODEL = os.environ.get("PILOT_MODEL", "deepseek/deepseek-v4.1-flash")
PRICE_IN  = float(os.environ.get("PRICE_IN_PER_1M", "0"))   # $ за 1M входных токенов; 0 = не считать
PRICE_OUT = float(os.environ.get("PRICE_OUT_PER_1M", "0"))  # $ за 1M выходных токенов

# Ограничение reasoning: 'low' | 'medium' | 'high' | 'max'
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low").strip().lower()
# Жёсткий бюджет на reasoning-токены. Если задан — добавляется в запрос.
REASONING_MAX_TOKENS = os.environ.get("REASONING_MAX_TOKENS")
REASONING_MAX_TOKENS = int(REASONING_MAX_TOKENS) if REASONING_MAX_TOKENS else None

# Стриминг ответа и вывод в консоль
STREAM = os.environ.get("STREAM", "1") not in {"0", "false", "False", "no"}
STREAM_SHOW_REASONING = os.environ.get("STREAM_SHOW_REASONING", "0") in {"1", "true", "True", "yes"}

N_PILOT = 30
FULL_N  = 4063


def _build_reasoning_payload():
    """Собирает поле reasoning для OpenRouter. Пустой dict — не отправлять."""
    r = {}
    if REASONING_EFFORT in {"low", "medium", "high", "max"}:
        r["effort"] = REASONING_EFFORT
    if REASONING_MAX_TOKENS:
        r["max_tokens"] = REASONING_MAX_TOKENS
    return r


def _open_stream(req):
    """Открывает SSE-соединение. Возвращает file-like объект."""
    return urllib.request.urlopen(req, timeout=300)


def chat_stream(messages, tag=""):
    """Стриминговый чат: печатает content в stdout, собирает usage из последнего чанка."""
    key = os.environ["OPENROUTER_API_KEY"]
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "Accept": "text/event-stream",
    }
    if os.environ.get("OR_REFERER"):
        headers["HTTP-Referer"] = os.environ["OR_REFERER"]
        headers["X-Title"] = "bigvul-bf-distillation"

    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": messages,
        "stream": True,
        # просим OpenRouter вернуть usage в последнем чанке
        "stream_options": {"include_usage": True},
    }
    reasoning = _build_reasoning_payload()
    if reasoning:
        payload["reasoning"] = reasoning

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body, headers=headers,
    )

    for attempt in range(3):
        t0 = time.time()
        log.info(
            f"[{tag}] стрим к {MODEL} (попытка {attempt+1}/3, reasoning={reasoning or 'off'})…"
        )
        content_parts = []
        usage = {}
        try:
            with _open_stream(req) as resp:
                print(f"\n----- [{tag}] STREAM START -----", flush=True)
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # usage может прийти в отдельном чанке (include_usage)
                    if chunk.get("usage"):
                        usage = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    # reasoning-текст (если провайдер его отдаёт)
                    r_text = delta.get("reasoning") or delta.get("reasoning_content")
                    if r_text and STREAM_SHOW_REASONING:
                        sys.stdout.write(f"\033[2m{r_text}\033[0m")
                        sys.stdout.flush()

                    piece = delta.get("content")
                    if piece:
                        content_parts.append(piece)
                        sys.stdout.write(piece)
                        sys.stdout.flush()

                print(f"\n----- [{tag}] STREAM END -----", flush=True)

            dt = time.time() - t0
            u = usage or {}
            details = u.get("completion_tokens_details") or {}
            r_tokens = details.get("reasoning_tokens")
            log.info(
                f"[{tag}] OK за {dt:.1f}s | in={u.get('prompt_tokens')} "
                f"out={u.get('completion_tokens')} "
                f"reasoning={r_tokens if r_tokens is not None else 'n/a'}"
            )
            return "".join(content_parts), usage
        except Exception as e:
            dt = time.time() - t0
            log.warning(f"[{tag}] ошибка стрима за {dt:.1f}s: {type(e).__name__}: {e}")
            if attempt == 2:
                log.error(f"[{tag}] все 3 попытки провалились")
                raise
            backoff = 5 * (attempt + 1)
            log.info(f"[{tag}] повтор через {backoff}s")
            time.sleep(backoff)


def chat_blocking(messages, tag=""):
    """Не-стриминговый фолбэк (если STREAM=0)."""
    key = os.environ["OPENROUTER_API_KEY"]
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    if os.environ.get("OR_REFERER"):
        headers["HTTP-Referer"] = os.environ["OR_REFERER"]
        headers["X-Title"] = "bigvul-bf-distillation"

    payload = {"model": MODEL, "temperature": 0, "messages": messages}
    reasoning = _build_reasoning_payload()
    if reasoning:
        payload["reasoning"] = reasoning

    body = json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, headers=headers)

    for attempt in range(3):
        t0 = time.time()
        log.info(f"[{tag}] запрос к {MODEL} (попытка {attempt+1}/3, reasoning={reasoning or 'off'})…")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode())
            dt = time.time() - t0
            u = d.get("usage", {}) or {}
            details = u.get("completion_tokens_details") or {}
            r_tokens = details.get("reasoning_tokens")
            log.info(
                f"[{tag}] OK за {dt:.1f}s | in={u.get('prompt_tokens')} "
                f"out={u.get('completion_tokens')} "
                f"reasoning={r_tokens if r_tokens is not None else 'n/a'}"
            )
            return d["choices"][0]["message"]["content"], d["usage"]
        except Exception as e:
            dt = time.time() - t0
            log.warning(f"[{tag}] ошибка за {dt:.1f}s: {type(e).__name__}: {e}")
            if attempt == 2:
                log.error(f"[{tag}] все 3 попытки провалились")
                raise
            backoff = 5 * (attempt + 1)
            log.info(f"[{tag}] повтор через {backoff}s")
            time.sleep(backoff)


def chat(messages, tag=""):
    if STREAM:
        return chat_stream(messages, tag=tag)
    return chat_blocking(messages, tag=tag)


# ---------- разбор и проверки ----------
SEC = ["EXPLANATION", "FRAME", "TRIPLET", "BF_CHAIN", "CWE"]
ALLOWED_CHAIN = {"MAL", "MUS", "MDL", "DVL", "DVR", "TCV", "TCM", "IEX", "ACE", "TPR", "DOS"}
FAILURES = {"IEX", "ACE", "TPR", "DOS"}


def parse_chain(text):
    parts = re.split(r"^### ", text, flags=re.M)
    out = {}
    for p in parts[1:]:
        name, _, body = p.partition("\n")
        out[name.strip()] = body.strip()
    return out


def checks(sec):
    r = {}
    r["all_sections"] = all(k in sec for k in SEC)
    if not r["all_sections"]:
        return r
    if sec["EXPLANATION"].startswith("NO_VULNERABILITY"):
        r["no_vuln"] = True
        return r
    try:
        frame = json.loads(sec["FRAME"])
        r["frame_json"] = True
        r["frame_keys"] = set(frame) == {"source", "operation", "missing_check", "sink", "consequence"}
    except Exception:
        r["frame_json"] = False
        r["frame_keys"] = False
        frame = {}
    cons_f = str(frame.get("consequence", ""))
    trip = sec.get("TRIPLET", "")
    r["consistency_consequence"] = bool(trip) and (
        trip.rstrip(")").split("->")[-1].strip().rstrip(")") == cons_f or cons_f in trip
    )
    sink_core = str(frame.get("sink", "")).split("(")[0].strip()
    r["consistency_sink"] = bool(sink_core) and sink_core in sec["EXPLANATION"]
    toks = [t.strip() for t in sec.get("BF_CHAIN", "").split("->") if t.strip()]
    r["chain_vocab"] = bool(toks) and all(t in ALLOWED_CHAIN for t in toks)
    r["chain_ends_failure"] = bool(toks) and toks[-1] in FAILURES
    m = re.search(r"CWE-(\d+)", sec.get("CWE", ""))
    r["cwe_id"] = int(m.group(1)) if m else None
    return r


# ---------- выборка ----------
df = pd.read_csv("../../lora_research_to_aij/eda/data/processed/dedup_final.csv")
top6 = df["cwe"].value_counts().head(6).index.tolist()
pilot = (
    df[df["cwe"].isin(top6)]
    .groupby("cwe", group_keys=False)
    .apply(lambda g: g.sample(5, random_state=0))
)
log.info(f"классы пилота: {top6} | строк: {len(pilot)}")

# ---------- промпт: вырезаем части из prompt_template.md ----------
md = Path("prompt_template.md").read_text()
system = md.split("## SYSTEM", 1)[1].split("## USER", 1)[0]
user_tpl = md.split("## USER (шаблон вызова)", 1)[1].split("## FEW-SHOT", 1)[0]
shots = md.split("## FEW-SHOT ПРИМЕР 1", 1)[1]
shot1 = "### ПРИМЕР 1 " + shots.split("## FEW-SHOT ПРИМЕР 2")[0].lstrip()
shot2 = "### ПРИМЕР 2 " + shots.split("## FEW-SHOT ПРИМЕР 2")[1].split("---")[0].lstrip()

# ---------- основной цикл ----------
rows, tot_in, tot_out, tot_reason = [], 0, 0, 0
n = len(pilot)
t_start = time.time()
log.info(
    f"старт пилота: {n} запросов, модель={MODEL}, "
    f"reasoning_effort={REASONING_EFFORT}, "
    f"reasoning_max_tokens={REASONING_MAX_TOKENS}, stream={STREAM}"
)

out_path = Path("data/chains_pilot.jsonl")

for i, (_, r) in enumerate(pilot.iterrows(), start=1):
    idx = int(r.name)
    tag = f"{i}/{n} idx={idx} label={r['cwe_id']}"
    log.info(f"[{tag}] собираю промпт, длина функции={len(str(r['func_before']))} симв.")

    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": shot1 + "\n\n" + shot2 + "\n\n" +
             user_tpl.replace("{CODE}", f"### CODE\n```c\n{r['func_before']}\n```")},
    ]

    text, usage = chat(messages, tag=tag)
    usage = usage or {}
    tot_in += usage.get("prompt_tokens", 0)
    tot_out += usage.get("completion_tokens", 0)
    details = usage.get("completion_tokens_details") or {}
    tot_reason += details.get("reasoning_tokens", 0) or 0

    # 2. после КАЖДОГО сэмпла сохраняем сырой выход
    (Path("data/raw") / f"{idx:06d}.txt").write_text(text, encoding="utf-8")

    sec = parse_chain(text)
    c = checks(sec)
    c.update({"idx": idx, "cwe_label": int(r["cwe_id"]), "raw": text})
    rows.append(c)

    ok = bool(c.get("all_sections")) and bool(c.get("frame_json"))
    n_sec = sum(k in sec for k in SEC)
    log.info(
        f"[{tag}] -> cwe={c.get('cwe_id')} ok={ok} | секций={n_sec}/5 "
        f"| накоплено in={tot_in} out={tot_out} reason={tot_reason}"
    )

    # 3. построчная дозапись в jsonl (append), чтобы ничего не терялось
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

    time.sleep(1)

log.info(f"цикл завершён за {time.time() - t_start:.1f}s")

# ---------- финальная сводка ----------
res = pd.read_json(out_path, lines=True)

log.info("=== СВОДКА ПИЛОТА ===")
for col in [
    "all_sections", "frame_json", "frame_keys", "consistency_consequence",
    "consistency_sink", "chain_vocab", "chain_ends_failure",
]:
    if col in res:
        log.info(f"{col}: {res[col].sum()}/{len(res)}")

nv = res.get("no_vuln", pd.Series(False, index=res.index)).sum()
log.info(f"NO_VULNERABILITY: {nv}/{len(res)}")

mm = res[(res["cwe_id"].notna()) & (res["cwe_id"] != res["cwe_label"])]
log.info(f"CWE mismatch (teacher != BigVul): {len(mm)}/{len(res)}")
if len(mm):
    log.info("\n" + mm[["cwe_label", "cwe_id"]].to_string(index=False))

log.info(f"токены пилота: in={tot_in} out={tot_out} reasoning={tot_reason}")
if PRICE_IN and PRICE_OUT:
    cost = (tot_in * PRICE_IN + tot_out * PRICE_OUT) / 1e6
    log.info(
        f"стоимость пилота: ${cost:.2f} | прогноз полного прогона ({FULL_N}): "
        f"${cost / len(res) * FULL_N:.0f}"
    )
else:
    log.info("цены не заданы (PRICE_IN_PER_1M/PRICE_OUT_PER_1M) — возьмите на странице модели в OpenRouter")