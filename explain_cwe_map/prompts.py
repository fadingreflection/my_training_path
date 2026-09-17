"""Two-step prompts: vulnerability explanation, then CWE hierarchy mapping."""

REVIEW = """You are a security engineer reviewing one C/C++ function body. Produce only an explanation of whether it is vulnerable. Do not name CWE IDs, CVE IDs, or a CWE hierarchy. Do not write a fix.

Reasoning method:
1. Read the function once.
2. Form at least two competing hypotheses about what could be wrong.
   Typical families: memory bounds, resource lifetime, integer arithmetic,
   input validation, privilege/access control, information disclosure,
   re-entrancy, TOCTOU.
3. Select the single most security-relevant path:
   attacker input → missing check → sink.
   Ignore accesses that are not on this path.
4. Write the explanation around that path only.

Output rules:
- First line must be exactly one of: VULNERABLE. / INSUFFICIENT_INFO. / SAFE
- Then briefly state the competing hypotheses and why the others were ruled out.
- Then explain the chosen path: which input is attacker-controlled, which
  check is missing or wrong, which statement is the sink.
- Quote the expressions on that path. Do not paste the whole function.
- Do not enumerate every line or every field access.
- Do not describe accesses that are not on the critical path.
- Keep the explanation under 1000 words. Finish it. Do not truncate.
- Stay inside the shown function. Do not invent callers, globals, or callees
  that are not in the snippet.

When to say VULNERABLE vs INSUFFICIENT_INFO vs SAFE:
- If any of these patterns is present in the shown code, the function is
  VULNERABLE even if a callee is not shown: deletion during iteration,
  double-free, use-after-free, re-entrancy through callbacks, resource
  freed in callee and used in the caller, missing privilege or access-control
  check, integer overflow/underflow feeding a length or index, uninitialized
  read exposed to the attacker, out-of-bounds read reachable from attacker
  input.
- INSUFFICIENT_INFO only if none of those patterns appear in this body and
  the sink is solely in an unseen callee. One sentence why. Not SAFE.
- SAFE only if this function itself clearly cannot be exploited from the
  shown code.
"""

MAP_CWE = (
    "Now use your explanation to map this vulnerability to CWE hierarchy. "
    "Go from top pillar level to the bottom ones."
)


def review_user_message(code: str) -> str:
    return f"{REVIEW.rstrip()}\n\n{code.strip()}\n"


def _head_token(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    head = t.splitlines()[0].strip().strip('"').strip("'")
    return head.split()[0].upper().rstrip(".:") if head else ""


def is_safe_verdict(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _head_token(t) == "SAFE":
        return True
    compact = t.replace(".", "").replace('"', "").strip().upper()
    return compact == "SAFE"


def is_insufficient_verdict(text: str) -> bool:
    return _head_token(text) == "INSUFFICIENT_INFO"


def skip_cwe_mapping(text: str) -> bool:
    return is_safe_verdict(text) or is_insufficient_verdict(text)
