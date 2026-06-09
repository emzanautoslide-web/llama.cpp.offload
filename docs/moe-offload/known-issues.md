# MoE Offload Known Issues

Status as of the adaptive ubatch and interactive-mode diagnosis work on
2026-06-09.

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
- The MoE offload path still has unconditional `[moe-d4]` `stderr` debug
  prints, including periodic callback counters. In an interactive terminal
  these lines interleave with generated tokens and make the visible transcript
  look corrupted even when they are not model tokens.
- This has not yet been proven to be a slot-cache/logit correctness bug. The
  current evidence points first at chat-template invocation and noisy MoE
  diagnostics in interactive mode.

Workarounds while this is open:

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
- Redirect `stderr` when checking answer text, because current `[moe-d4]`
  diagnostics are not quiet enough for interactive use.

Next fix:

- Gate or remove the unconditional `[moe-d4]` prints.
- Add a chat-template smoke test for `llama-completion -cnv --jinja -st`.
- Re-run a golden-logit check on the formatted chat prompt before treating this
  as fully closed.

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
