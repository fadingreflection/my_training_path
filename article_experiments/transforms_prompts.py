# """Prompts for meaning-preserving transformation generation.

# Each prompt contains a single format placeholder:
#   - PARAPHRASE_PROMPT, STYLE_FORMAL_PROMPT, STYLE_INFORMAL_PROMPT -> {explanation}
#   - CODE_VARIATION_PROMPT, EXPLAIN_PROMPT -> {code}
# """

# PARAPHRASE_PROMPT = """You are rewriting a vulnerability explanation while preserving its exact meaning.

# ORIGINAL EXPLANATION:
# {explanation}

# TASK:
# Rewrite this explanation using different words, but preserve:
# 1. The same vulnerability mechanism
# 2. The same source (where data comes from)
# 3. The same operation (what is done with the data)
# 4. The same missing check (what guard is absent)
# 5. The same sink (where the dangerous action occurs)
# 6. The same consequence (what happens as a result)

# Requirements:
# - Change vocabulary and sentence structure
# - Do not add new information
# - Do not remove any of the five mechanism roles
# - Do not change the vulnerability type
# - Do not include any preamble or meta-commentary
# - Output only the rewritten explanation text

# REWRITTEN EXPLANATION:
# """


# STYLE_FORMAL_PROMPT = """Rewrite the following vulnerability explanation in a formal, technical, academic register.

# Use precise terminology. Avoid contractions. Use passive voice where appropriate.
# Write as if for a security audit report.

# Do NOT change the vulnerability mechanism: keep the same source, operation,
# missing check, sink, and consequence.

# Do NOT include any preamble. Output only the rewritten text.

# ORIGINAL:
# {explanation}

# FORMAL VERSION:
# """


# STYLE_INFORMAL_PROMPT = """Rewrite the following vulnerability explanation in an informal, conversational register.

# Use plain language. Use contractions. Write as if explaining the issue to a
# colleague in a code review chat.

# Do NOT change the vulnerability mechanism: keep the same source, operation,
# missing check, sink, and consequence.

# Do NOT include any preamble. Output only the rewritten text.

# ORIGINAL:
# {explanation}

# INFORMAL VERSION:

"""Prompts for meaning-preserving transformation generation.

Each prompt contains a single format placeholder:
  - PARAPHRASE_PROMPT, STYLE_FORMAL_PROMPT, STYLE_INFORMAL_PROMPT -> {explanation}
"""

PARAPHRASE_PROMPT = """You are rewriting a vulnerability explanation while preserving its exact meaning.

ORIGINAL EXPLANATION:
{explanation}

TASK:
Rewrite this explanation using different words, but preserve all five elements of the vulnerability mechanism:
1. source         — where attacker-controlled data comes from
2. operation      — what is done with the data
3. missing check  — which guard is absent
4. sink           — where the dangerous action occurs
5. consequence    — what results from the failure

Requirements:
- Change vocabulary and sentence structure
- Keep approximately the same length and level of detail
- Do not add new information
- Do not remove any of the five mechanism elements
- Do not change the vulnerability type or consequence
- Do not include any preamble or meta-commentary
- Output only the rewritten explanation text

REWRITTEN EXPLANATION:
"""


STYLE_FORMAL_PROMPT = """Rewrite the following vulnerability explanation in a formal, technical register.

Write as if for a security audit report. Use precise terminology. Avoid contractions.

TASK:
Preserve the vulnerability mechanism exactly: source, operation, missing check, sink, consequence.
Preserve completeness: every element in the original must still be present.
Do not add new information beyond what is stated in the original.
Keep approximately the same level of detail as the original.

Do not include any preamble. Output only the rewritten text.

ORIGINAL:
{explanation}

FORMAL VERSION:
"""


STYLE_INFORMAL_PROMPT = """Rewrite the following vulnerability explanation in an informal, conversational register.

Write as if explaining the issue to a colleague in a code review chat. Use plain language and contractions.

TASK:
Preserve the vulnerability mechanism exactly: source, operation, missing check, sink, consequence.
Preserve completeness: every element in the original must still be present.
Do not add new information beyond what is stated in the original.
Keep approximately the same level of detail as the original.

Do not include any preamble. Output only the rewritten text.

ORIGINAL:
{explanation}

INFORMAL VERSION:
"""