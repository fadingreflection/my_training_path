"""Two-step prompts: vulnerability explanation, then CWE hierarchy mapping."""

REVIEW = (
    "You are a security expert. Given a code snippet your task is to review it "
    "with all attention and generate a comprehensive step by step explanation "
    "whether the code is vulnerable or not. If the code is vulnerable - give "
    "concise reasoning in terms of algorithmic flows why it is vulnerable. "
    "If the code is safe just answer \"SAFE\"."
)

MAP_CWE = (
    "Now use your explanation to map this vulnerability to CWE hierarchy. "
    "Go from top pillar level to the bottom ones."
)


def review_user_message(code: str) -> str:
    return f"{REVIEW}\n\n{code.strip()}\n"


def is_safe_verdict(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    head = t.splitlines()[0].strip().strip('"').strip("'").upper()
    if head in {"SAFE", "SAFE."}:
        return True
    # whole-answer SAFE with optional punctuation / whitespace
    compact = t.replace(".", "").replace('"', "").strip().upper()
    return compact == "SAFE"
