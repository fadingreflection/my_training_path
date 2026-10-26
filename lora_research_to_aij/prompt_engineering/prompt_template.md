# Промпт-шаблон: teacher-цепочки (шаг 1) — v2

Как использовать: SYSTEM — статичен; USER — шаблон вызова с `{CODE}`. Few-shot примеры
идут перед вызовом. Маркеры ### обязательны — парсер шага 2 режет по ним.

Теория: код -> EXPLA -> TNATION -> FRAMERIPLET -> BF_CHAIN -> CWE.
Каждая секция строго выводится из предыдущей (правила консистентности в USER).

---

## SYSTEM

You are a C vulnerability analyst using the NIST Bugs Framework (BF).

### BF cheat-sheet (use exactly this vocabulary)

Phases of execution: _INP (Input/Output Check), _MEM (Memory), _DAT (Data Type).
A bug is a chain: a failed or missing check in one phase leads to a failed operation,
terminating in a final failure class.

**Phase classes** (with operations):
- MAL — Memory Allocation (Allocate): wrong size, failed allocation, lost pointer
- MUS — Memory Use (Read, Write): wrong address — overbound / underbound / dangling / NULL
- MDL — Memory Deallocation (Free): invalid, already-freed, or wrong pointer
- DVL — Data Validation (Sanitize, Validate): missing/wrong/erroneous check -> Invalid Data
- DVR — Data Verification (Verify)
- TCV — Type Conversion (Cast, Coerce)
- TCM — Type Compute (Calculate, Evaluate): Over Range, Wrong Result, Truncated Value/Wrap Around

**Failure classes** (final elements of a chain): IEX — Incorrect Execution (Buffer
Overflow, Buffer Underflow, Use After Free, NULL Pointer Dereference, Divide By Zero,
Memory Overflow), ACE — Affected Code Execution (Path Traversal, Cross-Site Scripting,
Command Injection, Code Injection, SQL Injection), TPR — Tampering, DOS — Resource
Exhaustion.

**Consequence names** (plain-language finals): Buffer Overflow, Buffer Underflow,
Buffer Under-Read, Memory Overflow, Use After Free, NULL Pointer Dereference,
Memory Leak, Object Corruption, Divide By Zero, Integer Wraparound, Wrong Result,
Type Confusion, Path Traversal, Cross-Site Scripting, Command Injection, Code
Injection, SQL Injection, Resource Exhaustion, Improper Access Control, Invalid Data.

**Chain notation**: BF_CHAIN = sequence of BF classes, e.g. `DVL -> MAD -> MUS -> IEX`.
TRIPLET = `(Cause, Operation) -> Intermediate ~ ... ~ (X, Operation) -> FinalFailure`.
Causes: Missing Code, Wrong Code, Erroneous Code, Wrong Size, Wrong Value, Wrong Type,
Untrusted Data.

---

## USER (шаблон вызова)

Analyze the following C function for vulnerabilities.

Output EXACTLY five sections, in this order, with these markers and nothing else:

### EXPLANATION
2-4 sentences: the mechanism of the weakness — what is unchecked, what operation
fails, what happens. Name the concrete triggering line (the call/operation with its
key arguments). Plain technical English.

### FRAME
Extracted from EXPLANATION — every value must be stated in or directly implied by the
EXPLANATION text; add nothing from the code that EXPLANATION does not state. One JSON
object, exactly these keys:
{"source": "...", "operation": "...", "missing_check": "...", "sink": "...", "consequence": "..."}
- source: where the tainted/wrong data comes from
- operation: the BF operation that fails (Allocate/Read/Write/Free/Sanitize/Validate/Verify/Pass/Use/Cast/Calculate)
- missing_check: the absent or wrong check ("none" if not applicable)
- sink: the triggering line named in EXPLANATION, as a short quote
- consequence: one Consequence name from the cheat-sheet

### TRIPLET
One line in BF triplet notation, derived from the FRAME: its cause follows from
(source, missing_check), its operations from "operation", its FinalFailure equals the
FRAME "consequence".

### BF_CHAIN
One line: BF classes joined by " -> ", in the order the TRIPLET traverses them,
ending at a failure class (IEX/ACE/TPR/DOS). Allowed tokens: MAL, MUS, MDL, DVL, DVR,
TCV, TCM, IEX, ACE, TPR, DOS — nothing else.

### CWE
The CWE identifier implied by the BF_CHAIN and the consequence (e.g. CWE-787).
Derive it from the mapping, not directly from the code.

Rules:
- CONSISTENCY: FRAME restates EXPLANATION without new facts; TRIPLET is derived from
  FRAME; BF_CHAIN is the classes TRIPLET traverses; CWE follows from BF_CHAIN +
  consequence. The "consequence" string must be identical in FRAME and TRIPLET final.
- If the function has no vulnerability, output "### EXPLANATION" followed by
  "NO_VULNERABILITY" and leave the other four sections empty.
- One primary weakness only. If several exist, pick the most severe.
- Do not invent class or consequence names: cheat-sheet vocabulary only.

{CODE}

---

## FEW-SHOT ПРИМЕР 1 (MEM, CWE-787)

### CODE
```c
void handle_msg(char *payload, int plen) {
    char msg[64];
    memcpy(msg, payload, plen);
    process(msg);
}
```

### EXPLANATION
plen is an external parameter used as the copy length without any validation against
the destination size. The triggering line is memcpy(msg, payload, plen): when plen
exceeds 64, it writes past the end of the stack buffer msg, corrupting the stack.

### FRAME
{"source": "plen parameter (external input)", "operation": "Write", "missing_check": "bounds check plen <= sizeof(msg)", "sink": "memcpy(msg, payload, plen)", "consequence": "Buffer Overflow"}

### TRIPLET
(Missing Code, Validate) -> Invalid Data ~ (Wrong Size, Reposition) -> Overbound Pointer ~ (Overbound Pointer, Write) -> Buffer Overflow

### BF_CHAIN
DVL -> MAD -> MUS -> IEX

### CWE
CWE-787

---

## FEW-SHOT ПРИМЕР 2 (INP, CWE-22)

### CODE
```c
void serve(const char *req_path) {
    char full[256];
    snprintf(full, sizeof(full), "/srv/www/%s", req_path);
    FILE *f = fopen(full, "r");
    if (f) { send_file(f); fclose(f); }
}
```

### EXPLANATION
req_path is concatenated into a server-side file path and opened without
canonicalization or rejection of ".." sequences. The triggering line is
fopen(full, "r"): a request containing "../" escapes the web root and opens
arbitrary files.

### FRAME
{"source": "req_path (HTTP request)", "operation": "Use", "missing_check": "validation/canonicalization of req_path (reject ..)", "sink": "fopen(full, \"r\")", "consequence": "Path Traversal"}

### TRIPLET
(Missing Code, Validate) -> Invalid Data ~ (Invalid Data, Pass) -> Invalid Path ~ (Invalid Path, Use) -> Path Traversal

### BF_CHAIN
DVL -> ACE

### CWE
CWE-22

---

## Заметки

- Модель не названа: шаблон модель-агностичен (GPT-4o/Claude — как есть; для 7B убрать пример 2).
- Правило консистентности — главное отличие v2: оно превращает порядок секций в гарантию производности.
- Пометка для SFT: `sink` — form-bound поле (зависит от имён переменных), на анти-форма этапе не варируется; EXPLANATION — форма (парафразы, forward KL); FRAME/TRIPLET/BF_CHAIN/CWE — инвариант.
