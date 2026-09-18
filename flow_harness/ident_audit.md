# Identifier-precision audit (not a gate-2 verdict)

Raw `identifier_precision` in `v0_baseline.json` stays **0.2027**. Gate 2 is **not set**.

## What 0.20 actually is

- Raw tokens vs function-body AST: **0.2102** (9112/43345).
- Of raw false tokens, **13701** are English prose (`attacker`, `path`, `sink`…).
- Code-like tokens vs the same AST gold: **0.9082** (9948/10953).
- Code-like vs expanded snippet gold: **0.9342** (10232/10953).

## Category precision (code-like tokens)

| category | n | vs body AST | vs expanded snippet | vs role gold |
|---|---:|---:|---:|---:|
| local_or_param | 3292 | 0.9438 | 0.9845 | 0.6753 |
| field | 3442 | 0.9919 | 0.9924 | 0.9916 |
| callee | 3307 | 0.8706 | 0.8969 | 0.8812 |
| type | 72 | 1.0 | 1.0 | 1.0 |
| macro | 840 | 0.5667 | 0.6393 | 0.6393 |
| other | 0 | None | None | None |

## Manual-rule audit of 50 code-like false tokens (vs body AST)

Labels: {'ast_limitation': 48, 'invented': 2}

If `ast_limitation` dominates, the gold is the function body, not the model.
Expanded gold still cannot see declarations in other files / headers; that is Phase 1 if needed.

| token | category | label | in snippet | excerpt |
|---|---|---|---|---|
| `INT64_MAX` | macro | ast_limitation | False | - Missing/wrong check: the multiplications are performed without any overflow gu |
| `enough` | callee | ast_limitation | False | This compares the declared length only against the destination buffer size, not  |
| `contents` | callee | ast_limitation | False | The attacker controls the packet contents (`pkt->data`, `pkt->size`), and theref |
| `entered` | callee | ast_limitation | False | Additionally, `efree(source)` is executed before `zend_shared_alloc_register_xla |
| `plaintext` | callee | ast_limitation | True | the replay-protected packet ID is read **from the IV**, i.e. from data that is t |
| `ICC` | macro | ast_limitation | False | - Attacker-controlled input: the bytes of the ICC profile stream `in`. An attack |
| `systems` | callee | ast_limitation | False | The attacker-controlled input is `nvec`, which comes from the interrupt-count va |
| `fail` | callee | ast_limitation | False | - `native_handle_create(numFds, numInts)` is called with the raw attacker values |
| `size_t` | local_or_param | ast_limitation | True | - `size_t new_size = gc_info_table_size_ ? 2 * gc_info_table_size_ : kInitialSiz |
| `TOCTOU` | macro | ast_limitation | False | 3. Missing lock/TOCTOU on `slot->fd`. |
| `root` | callee | ast_limitation | True | When `lp_strict_modes(module)` is true, the code unconditionally sets `ok = 0` a |
| `SPS` | macro | ast_limitation | False | 1. The SPS lookup can fail silently and hand an invalid/uninitialized SPS to the |
| `payload` | callee | ast_limitation | True | Competing hypotheses: (1) out-of-bounds indexing of `m_clusters` via `pCurr->m_i |
| `RPC` | macro | ast_limitation | False | 3. Uninitialized `pages[]` array used as RPC buffers when `buf_to_pages()` fails |
| `GPU` | macro | ast_limitation | False | then copies that uninitialized stack/heap residue into `iq_matrix_buf`, which is |
| `uint32_t` | local_or_param | ast_limitation | True | - Sink: `EXTRACT_32BITS(ptr)` — the cast `const uint32_t *ptr = (const uint32_t  |
| `VMA` | macro | ast_limitation | False | This check is wrong in direction: it does not verify that `address` lies within  |
| `safely` | callee | ast_limitation | False | Hypothesis 2 is ruled out: `find_swevent_head_rcu` is an unseen callee that perf |
| `FIFO` | callee | ast_limitation | True | Competing hypotheses: (1) integer overflow in the size computations `SVGA_BITMAP |
| `OOB` | macro | ast_limitation | False | 2. Memory bounds / lifetime issue in `asn1_read_uint8` or the tag helpers — rule |
| `CFB` | macro | ast_limitation | True | 4. *Replay-check logic* — the packet ID handling itself (`packet_id_test`/`packe |
| `period` | callee | ast_limitation | False | The protection against freeing during this window is solely the RCU read-side cr |
| `repeatedly` | callee | ast_limitation | False | 2. **Integer underflow of `len` feeding an index or loop bound.** `len` is decre |
| `dimensions` | callee | ast_limitation | False | The sink is `vmsvga_fifo_read_raw(s)`, executed up to `SVGA_BITMAP_SIZE(x, y) +  |
| `parser` | callee | ast_limitation | False | and, by contract of this parser family, `parse_value` records the error position |
| `unvalidated` | callee | ast_limitation | False | Even where the multiplication is done in size_t width, `delay` up to 2^31−1 time |
| `file_util` | local_or_param | ast_limitation | True | Chosen path: attacker-controlled input is `options.name` (a shared-memory name,  |
| `item` | callee | ast_limitation | False | Chosen path: attacker-controlled input is the folder name, `folder_item_->name() |
| `NAL` | callee | ast_limitation | False | - Attacker input: a SEI NAL (`i1_nal_type == NAL_PREFIX_SEI`) with `u4_payload_t |
| `MD5` | macro | ast_limitation | False | 4. **Missing access-control / policy check**: `xfrm4_policy_check` is invoked on |
| `INT_MAX` | callee | ast_limitation | False | 1. Truncation: `int copy = min(bytes, iov->iov_len - base);` computes the min in |
| `NONDMA` | macro | ast_limitation | False | When the NONDMA flag is clear (DMA/transfer mode), no modulo or bounds check is  |
| `ssize_t` | local_or_param | ast_limitation | True | 1. *Heap buffer overflow in the run-length decode buffer* — the `pixels` scratch |
| `err3` | local_or_param | ast_limitation | True | 3. Error-path lifetime handling — the `err2`/`err3` cleanup does remove the idr  |
| `parameter` | callee | ast_limitation | True | 2. **Missing input validation on `pname`**: The switch enumerates the accepted p |
| `qedi_iscsi_io_setup` | local_or_param | invented | False | The copy always reads exactly 31 bytes from `func`, regardless of how long the s |
| `patterns` | callee | ast_limitation | False | INSUFFICIENT_INFO. The only attacker-reachable sink is `perf_swevent_event(bp, 1 |
| `st_mode` | field | ast_limitation | False | When `lp_strict_modes(module)` is true, the code unconditionally sets `ok = 0` a |
| `wrapped` | callee | ast_limitation | False | The addition `when + sgi_clock_period - 1` can wrap in `unsigned long`, and the  |
| `refcount_inc` | local_or_param | invented | False | An attacker who can repeatedly splice or get a reference to the same pipe buffer |
| `TCP` | macro | ast_limitation | True | 3. *Predictable TCP initial sequence number on reconnect (chosen path).* This is |
| `idiom` | callee | ast_limitation | False | 1. **Iterator invalidation / erase-during-iteration**: Both loops erase from the |
| `id` | local_or_param | ast_limitation | True | The attacker-controlled input is the set of attributes on the newly created node |
| `out_free` | local_or_param | ast_limitation | True | 2. Server-controlled length (`res.acl_len`) overflowing the copy into `buf`. Thi |
| `uint8_t` | local_or_param | ast_limitation | True | - `memset(reinterpret_cast<uint8_t*>(g_gc_info_table) + gc_info_table_size_ * si |
| `XFS_ACL_MAX_ENTRIES` | macro | ast_limitation | False | There is no check that `count` is non-negative or bounded by `XFS_ACL_MAX_ENTRIE |
| `USB` | macro | ast_limitation | False | 3. Uninitialized stack data from a short USB read — this is the chosen path. |
| `secret` | callee | ast_limitation | False | The chosen path: the function's purpose is to gate access to a secret (`challeng |
| `JNI` | field | ast_limitation | False | 1. Missing access-control check: the method is exposed via JNI and can be invoke |
| `maybe_get_net` | callee | ast_limitation | False | - *Missing check*: the function never validates that the namespace referenced by |
