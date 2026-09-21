# Identifier-precision audit (not a gate-2 verdict)

Gate 2 candidate: **`identifier_precision`** (code-shaped vs body AST) in `v0_baseline.json`. **`identifier_precision_raw`** is the legacy all-words metric (prose-inflated).

## What 0.20 actually is

- Raw tokens vs function-body AST: **0.1975** (10191/51602).
- Of raw false tokens, **15841** are English prose (`attacker`, `path`, `sink`…).
- Code-like tokens vs the same AST gold: **0.8716** (12264/14070).
- Code-like vs expanded snippet gold: **0.9023** (12696/14070).

## Category precision (code-like tokens)

| category | n | vs body AST | vs expanded snippet | vs role gold |
|---|---:|---:|---:|---:|
| local_or_param | 3582 | 0.9453 | 0.9701 | 0.6401 |
| field | 4276 | 0.989 | 0.989 | 0.9885 |
| callee | 4594 | 0.84 | 0.8548 | 0.8424 |
| type | 78 | 1.0 | 1.0 | 1.0 |
| macro | 1288 | 0.479 | 0.6444 | 0.6444 |
| other | 252 | 0.377 | 0.623 | 0.623 |

## Manual-rule audit of 50 code-like false tokens (vs body AST)

Labels: {'invented': 8, 'prose_leak': 10, 'ast_limitation': 32}

If `ast_limitation` dominates, the gold is the function body, not the model.
Expanded gold still cannot see declarations in other files / headers; that is Phase 1 if needed.

| token | category | label | in snippet | excerpt |
|---|---|---|---|---|
| `ChangeCipherSpec` | other | invented | False | The client controls the flow of the handshake (ClientHello, whether resumption i |
| `NULL` | macro | prose_leak | True | 3. Invalid key payload reaching the dereference after the key's payload is torn  |
| `field` | callee | ast_limitation | False | This hands PostScript a string object whose size field (`code + devlen`) exceeds |
| `clamping` | callee | ast_limitation | False | is correct, but the NULL-check omission at allocation remains the single securit |
| `continues` | callee | ast_limitation | False | The condition `if (used + offset < skb->len)` dereferences `skb->len` **after**  |
| `copy_uaddr` | local_or_param | ast_limitation | True | 2. **Address/name-length leak in `copy_uaddr`**: the `memcpy(uaddr, llc_ui_skb_c |
| `nd_trunc_longjmp` | local_or_param | invented | False | - Missing check: the function casts `dat` to `const uint32_t *ptr = (const uint3 |
| `UID` | macro | ast_limitation | False | 1. *Missing privilege/access-control check on the UID change* — ruled out as the |
| `checked` | callee | prose_leak | False | The critical path analysis: the attacker controls the stream contents (`in`). Th |
| `ACLs` | other | invented | False | which recurses into it, so an attacker can build arbitrary nested scaffolding an |
| `ServiceWorker` | other | invented | False | - `if (!enabled_)` — a global domain toggle for the DevTools ServiceWorker domai |
| `contents` | callee | ast_limitation | False | - Attacker input: the file contents, specifically the RLE control byte `count=(u |
| `SPS` | macro | ast_limitation | False | 2. Uninitialized memory read from the SPS structure when the PPS-level scaling m |
| `PostScript` | other | invented | False | - Attacker-controlled input: a PostScript program invoking this operator (`setst |
| `trunc` | local_or_param | ast_limitation | True | 2. **Unbounded loop driven by attacker-controlled counts (`ngroups`, `njoin`, `n |
| `ID` | local_or_param | ast_limitation | True | Everything else on the attacker-reachable surface is properly guarded: the tag b |
| `dependencies` | callee | ast_limitation | False | There is no path in this function where attacker-controlled input reaches a miss |
| `body` | callee | prose_leak | False | 3. **Unvalidated `blk_protocol` reaching `BUG()`.** Ruled out: it is a backend-s |
| `LC_CTYPE` | macro | ast_limitation | True | 1. Input validation / memory bounds: the locale string comes from an environment |
| `IPv6` | other | invented | False | 2. *Input validation of `flags`:* The flag bits (`pcrIpChecksum`, `pcrTcpV4Check |
| `arenas_` | local_or_param | ast_limitation | True | - `type` is an internal enum consumed by a switch with `NOTREACHED()` in the def |
| `MimeHandlerViewGuest` | other | ast_limitation | True | The missing check: nothing verifies that `plugin_rfh` is a frame within the actu |
| `functional` | callee | ast_limitation | False | Hypothesis 2 was ruled out: after `kmalloc` fails the function simply returns `- |
| `boundary` | callee | ast_limitation | False | - Missing check: the function casts `dat` to `const uint32_t *ptr = (const uint3 |
| `loop` | callee | prose_leak | False | 1. *Integer truncation in the length accumulation.* `remaining` is declared `uns |
| `stages` | callee | ast_limitation | False | Everything else on the attacker-reachable surface is properly guarded: the tag b |
| `perf_event_exit_task` | callee | ast_limitation | True | 2. Race with `perf_event_exit_task()` clearing `event->owner`, causing `list_del |
| `DoS` | other | invented | False | 2. *Resource exhaustion / DoS via a huge `vmalloc`* — the `totalram_pages` guard |
| `num_attributes` | field | ast_limitation | False | - Missing check: the function validates the object state and looks up `type_prop |
| `task` | callee | ast_limitation | False | - The page remains locked forever. Any other operation that needs that page — a  |
| `check` | callee | prose_leak | False | Hypothesis 4 is ruled out immediately — there is no ownership or callback in the |
| `formed` | callee | ast_limitation | False | This is a missing input-validation/access-control check on the resource-selectio |
| `GetParameter` | local_or_param | invented | False | - **Attacker input:** the structure pointed to by `portDefn`, delivered through  |
| `state` | callee | ast_limitation | False | Competing hypotheses: (1) an out-of-bounds access on `uipc_main` channel arrays  |
| `written` | callee | prose_leak | False | The `nameBuffer` path is safe because `nameLen` is bounded to 255 by its `XMP_Un |
| `Hypothesis` | field | prose_leak | False | Hypothesis (1) was ruled out because the sink `printk(KERN_INFO "%s value is cha |
| `IPC` | macro | ast_limitation | False | 3. *Uninitialized output*: if the callee fails its lookup and never writes `*can |
| `new_size` | local_or_param | ast_limitation | True | then compares an extent end that may itself be near the 32-bit boundary against  |
| `enumerated` | callee | ast_limitation | False | - Attacker-controlled input: the value `code` returned by `iodev->procs.enumerat |
| `sink` | callee | prose_leak | False | So on the first iteration, whatever bytes happen to be in the caller-provided `t |
| `discard_and_relse` | local_or_param | ast_limitation | True | 3. Use-after-free / refcount error on `sk` during the TIME_WAIT and NEW_SYN_RECV |
| `GPU` | macro | ast_limitation | False | If the stream contains a PPS with `pic_scaling_matrix_present_flag == 0` while t |
| `URB_FREE_BUFFER` | macro | ast_limitation | False | 1. **Conditional double-free**: the explicit `kfree(mixer->urb->transfer_buffer) |
| `caller` | callee | prose_leak | False | Attacker-controlled input: `int nsems = params->u.nsems;` — a signed integer sup |
| `VLA` | macro | ast_limitation | False | Competing hypotheses: (1) the variable-length array `fragment_table_index[indexe |
| `wrap` | callee | ast_limitation | False | Here `int nvec` is converted to the unsigned `size_t` of `sizeof`. On ABIs where |
| `so` | local_or_param | prose_leak | False | This check is skipped entirely when `m_size < 0` (unknown segment size, so `segm |
| `registration` | callee | ast_limitation | False | In short: the sole guard on the failure of `zswapcolors` — `if (code < 0)` — gua |
| `VMA` | macro | ast_limitation | False | Competing hypotheses: (1) an out-of-bounds / NULL dereference on `page` when the |
| `VMA` | callee | ast_limitation | False | Attacker-controlled input: `address`, the fault address from userspace that the  |
