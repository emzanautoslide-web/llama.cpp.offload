# MoE Offload Known Issues

Status as of the `llama-cli` chat correctness fix on 2026-06-09.

## Open

### `llama-completion` Conversation Mode Can Produce Bad Interactive Turns

Observed with `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf`,
`llama-completion.exe`, `--moe-offload`, and an interactive conversation. A
simple user turn such as:

```text
what is the capital of France
```

can produce an incorrect transcript that echoes user text or role labels
instead of answering `Paris`.

Current diagnosis:

- This is reproducible in `llama-completion` conversation mode, but raw
  completion mode (`-no-cnv`) with the same offloaded model can answer the
  same factual prompt correctly.
- `llama-completion` auto-enables conversation mode when the model has a chat
  template. Passing `-p "Hello"` in that mode pre-starts the conversation; it
  is not treated as a system prompt. The tool prints a warning for this case.
- The Qwen3.5 chat template in this GGUF is Jinja-based. A deterministic
  single-turn test without `--jinja` echoed the question and ended; enabling
  `--jinja` produced an answer containing `Paris`.
- Default MoE D-3/D-4 traces are now gated behind explicit debug environment
  variables, so the old terminal interleaving is no longer expected in normal
  chat runs.
- This has not yet been proven to be a slot-cache/logit correctness bug. The
  current evidence points first at chat-template invocation and noisy MoE
  diagnostics in `llama-completion` conversation mode.

Workarounds while this is open:

- For interactive chat, use `llama-cli` with the Jinja chat-template path:

  ```powershell
  .\build-moe\bin\Release\llama-cli.exe `
    --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
    --moe-offload `
    --moe-cache-vram-mb 8000 `
    --moe-predictor lru `
    --jinja `
    --reasoning off `
    -sys "You are a helpful assistant."
  ```

- For one-shot factual checks, use raw completion mode:

  ```powershell
  .\build-moe\bin\Release\llama-completion.exe `
    --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
    --moe-offload `
    --moe-cache-vram-mb 8000 `
    --moe-predictor lru `
    -no-cnv `
    -p "Q: what is the capital of France?`nA:" `
    -n 32
  ```

- For chat mode with this model, prefer `--jinja` and use `-sys` for system
  instructions instead of seeding the session with `-p "Hello"`.

Next fix:

- Keep `llama-completion` scoped to raw completion and logit diagnostics unless
  its conversation-mode behavior is intentionally revisited.
- Re-run a golden-logit check on the formatted chat prompt before treating this
  as fully closed.

### Batched Streaming Prefill Can Corrupt Chat Logits

`llama-cli --moe-offload` produced wrong visible answers for simple prompts
such as `who are you?`, `what is the capital of France?`, and `what is 1 + 1?`
when streaming prefill used multi-token ubatches. The same model answered
cleanly without MoE offload, and `LLAMA_MOE_STREAMING_UBATCH=1` made the
offloaded path answer cleanly.

Current diagnosis:

- This is a MoE streaming prefill correctness issue, not a Qwen model issue.
- `LLAMA_MOE_STREAMING_UBATCH=1` is clean on the tested prompts.
- `LLAMA_MOE_STREAMING_UBATCH=4` can leak extra `</think>` text.
- `LLAMA_MOE_STREAMING_UBATCH=8` can stream reasoning text instead of the final
  answer.
- Async H2D was not the cause; `LLAMA_MOE_DEBUG_NO_ASYNC=1` did not fix it.
- Forced no-hit reloads, identity slot-table fill, and compute sync diagnostics
  did not fix it.

Mitigation:

- `llama-cli --moe-offload` now requests `n_ubatch=1` by default unless
  `LLAMA_MOE_STREAMING_UBATCH` is explicitly set.
- Bench and diagnostic tools can still force larger streaming ubatches to
  continue investigating the batched slot path.

### `llama-cli` Reasoning-Budget Chat Output

During the 2026-06-09 chat smoke, `llama-cli` with:

```text
--reasoning-budget 0
```

could still display reasoning text instead of concise final answers for this
Qwen3.5 GGUF. The supported chat invocation now uses `--reasoning off`.

### Streaming Capacity Diagnostics

Streaming mode (`n_slots < n_experts`) no longer has a fixed `n_ubatch <= 8`
rewrite. `llama_context` now auto-sizes the effective ubatch from the slot
budget and model top-k, and the slot pool reserves at least the one-token
top-k working set per layer. The graph still has this internal constraint:

```text
unique experts selected by one MoE callback <= n_slots
```

Default auto mode keeps the worst-case bound within that capacity. The callback
still aborts with a clear `n_uniq exceeds n_slots` diagnostic if diagnostics
force an unsafe effective ubatch or force the minimum slot count below top-k.

Impact:

- Larger VRAM budgets can use larger streaming prefill ubatches.
- Very small caches shrink to effective ubatch 1 and may reserve slightly more
  than the requested cache budget to keep one token runnable.
- Explicit diagnostics can use `LLAMA_MOE_STREAMING_UBATCH=off|N`,
  `LLAMA_MOE_UBATCH_SAFETY=F`, and `LLAMA_MOE_MIN_SLOTS=N`.

### CUDA Top-k MoE Fusion Disabled

CUDA top-k MoE fusion is disabled in `LLAMA_MOE_OFFLOAD` builds. During
closeout, full-cache forced-streaming diagnostics showed the same logit drift
with and without remapped slot ids, while `GGML_CUDA_DISABLE_FUSION=1` matched
golden logits exactly. Per-fusion isolation showed that disabling top-k MoE
fusion alone was sufficient.

Impact:

- Correctness is preserved for the MVP.
- Throughput may be lower until the fused path is made safe for the MoE
  offload callback/remap flow.

### Specialized `.slot` `MUL_MAT_ID` Paths Bypassed

For MVP correctness, CUDA `MUL_MAT_ID` on `.slot` tensors bypasses the
specialized MMVQ/MMQ/MMF paths and uses the generic sorted CUDA path.

Impact:

- This avoids the suspect remapped-slot kernel path while golden logits are
  enforced.
- A safe specialized path can be restored after targeted CUDA validation.

### Profiler `stall_us` Is Approximate

`h2d_us` and `compute_us` are event-based where CUDA events are available.
`stall_us` is still host wall time around miss completion and compute-stream
wait insertion, so it is an approximation of overlap loss rather than a pure
GPU timeline metric.

## Deferred

The following remain outside MVP scope:

- `--moe-oracle`: parser plumbing may exist, but passing it fails fast with a
  clear post-MVP error.
- Direct I/O.
- Multi-GPU.
- CPU DRAM expert tier.
- KV offload.
- Learned predictors.
- Speculative prefetch/decoding.
- FineMoE-style splitting.
- True sub-token expert-rank chunking for running with fewer than top-k slots
  per layer without reserving the minimum working set.

## Closed

### Default D-3/D-4 Trace Spam

The old slot-pool `[moe-d3]` and `[moe-d4]` diagnostic prints are quiet by
default. Set `LLAMA_MOE_DEBUG_D4=1`, `LLAMA_MOE_DEBUG_LOADS=1`, or
`LLAMA_MOE_DEBUG_TRACE=1` to recover the detailed tensor/load traces.

### Fixed Streaming UBatch Cap

The old unconditional streaming cap of `n_ubatch <= 8` has been replaced by
adaptive sizing. The recommendation is derived from `n_slots / top_k` and is
bounded by the requested ubatch, with a minimum effective ubatch of 1. The slot
pool reads top-k from GGUF metadata and reserves that many slots per layer as
the minimum viable cache. The miss-loading path now submits and drains I/O in
chunks so larger prefill ubatches do not require a pinned buffer for every blob
in a layer at once.

### Streaming Golden-Logit Drift

Historical Phase J measurement on `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf` reported
`max|d| = 4.64e-01` over 8 decode steps at `--temp 0 --seed 42`.

The June 5 closeout build passes the forced-eviction golden-logit gate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 8
```

Observed result:

- `n_steps=8`
- `n_vocab=248320`
- `max|d|=0`
- `mean|d|=0`

The repacker was not the cause. Original vs repacked/offload-disabled logits
match, full-residency offload matches, and streaming matches full residency
after the runtime/CUDA correctness guards.

### Repacker Layout and Table Validation

The implemented layout is documented as `fused-tensors-page-aligned-v1`, with
fused expert tensors retained and `moe_offload.expert_blob.table` providing
per-expert byte ranges.

`test-repack-slices` compares every table entry against the exact byte slice in
the original GGUF when `LLAMA_MOE_TEST_ORIGINAL_GGUF` and `LLAMA_MOE_TEST_GGUF`
are provided.

### EAMC Sidecar Persistence

`--moe-eamc-path PATH` is wired through common args, `llama-bench`, and
`llama-moe-bench`. EAMC saves/loads a binary sidecar with shape/version checks
and ignores incompatible sidecars with a warning. `test-eamc-cosine` covers
sidecar round-trip.

### EAMC Predictor Overhead

EAMC scoring now caches the nearest-neighbor ranking within a callback, so
eviction candidate scans do not recompute the full cosine ranking repeatedly.
The 1024 prefill + 256 decode bench gate completed three repeats successfully.

### Oracle Ambiguity

`test-moe-oracle-failfast` verifies `--moe-oracle` exits as post-MVP instead of
silently enabling an unsupported path.

### Earlier Lifecycle Issues

IO worker shutdown, pinned-buffer starvation, multi-token prompt hangs, and
same-callback eviction of just-reserved experts were closed before the final
June 5 validation pass.
