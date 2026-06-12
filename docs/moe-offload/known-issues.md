# MoE Offload Known Issues

Status as of the guarded Phase I validation on 2026-06-12.

## Open

### Batched Streaming Chat Prefill Is Not Yet Safe

`llama-cli --moe-offload` produced wrong visible answers for simple chat
prompts such as `who are you?`, `what is the capital of France?`, and
`what is 1 + 1?` when streaming chat prefill used multi-token ubatches. The
same model answered cleanly without MoE offload, and the offloaded path
answered cleanly when streaming prefill was reduced to single-token ubatches.

Current status:

- `llama-cli --moe-offload` defaults to correctness-first `n_ubatch=1` unless
  `LLAMA_MOE_STREAMING_UBATCH` is explicitly set.
- The validated chat invocation is `llama-cli --jinja --reasoning off`.
- `LLAMA_MOE_STREAMING_UBATCH=4` has leaked extra `</think>` text.
- `LLAMA_MOE_STREAMING_UBATCH=8` has streamed reasoning text instead of the
  final answer.
- Async-H2D diagnostics, forced no-hit reloads, identity slot-table fill, and
  compute synchronization diagnostics did not fix the chat issue.
- Raw-completion golden-logit gates have passed at larger ubatches, including
  the Phase I `LLAMA_MOE_SLOT_MMVQ=1` + `LLAMA_MOE_PREFILL_MMVQ=1` +
  `LLAMA_MOE_SLOT_GRAPHS=1` + `LLAMA_MOE_SLOT_GLU_FUSION=1` gate. The Phase I
  formatted chat smoke also passed at the default ubatch and forced
  `LLAMA_MOE_STREAMING_UBATCH=8`, but the historical failures were prompt- and
  frontend-sensitive, so the interactive default remains ubatch 1 until the
  broader Phase K matrix is rerun.

Mitigation:

- Do not raise the `llama-cli --moe-offload` default above ubatch 1 until the
  formatted-chat smoke and golden-logit matrix pass at larger streaming
  ubatches.
- Benchmark and diagnostic tools may still force larger ubatches while this is
  investigated.

## Current Limits And Caveats

### `llama-completion` Conversation Mode Is Not The Supported Chat Frontend

`llama-completion.exe` can still produce bad interactive conversation turns
with this Qwen3.5 GGUF when it auto-enables conversation mode from the model's
chat template. This has not been proven to be a slot-cache/logit correctness
bug; the evidence points first at chat-template invocation and frontend
semantics.

Current guidance:

- Use `llama-cli --moe-offload --jinja --reasoning off` for interactive chat.
- Use `llama-completion -no-cnv` for raw completion and logit diagnostics.
- Avoid using `-p "Hello"` as a system prompt in conversation mode; it seeds
  user-visible conversation text.
- If `llama-completion` conversation mode is revisited, add a dedicated
  formatted-chat transcript smoke and logit gate.

### `--reasoning-budget 0` Is Not The Validated Qwen Chat Mode

For this Qwen3.5 GGUF, `llama-cli --reasoning-budget 0` can still display
reasoning text instead of concise final answers. The supported MoE chat command
uses `--reasoning off`.

### Streaming Capacity Diagnostics Are Expected For Unsafe Overrides

Streaming mode (`n_slots < n_experts`) no longer has a fixed `n_ubatch <= 8`
rewrite. The runtime auto-sizes the effective ubatch from the slot budget and
model top-k, and reserves at least one token's top-k working set per layer.

The graph still has this internal invariant:

```text
unique experts selected by one MoE callback <= n_slots
```

Default auto mode keeps the worst-case bound within that capacity. If
diagnostic overrides force an unsafe effective ubatch or set
`LLAMA_MOE_MIN_SLOTS` below top-k, the callback can still abort with a clear
`n_uniq exceeds n_slots` diagnostic.

Relevant diagnostics:

- `LLAMA_MOE_STREAMING_UBATCH=N`
- `LLAMA_MOE_STREAMING_UBATCH=off`
- `LLAMA_MOE_UBATCH_SAFETY=F`
- `LLAMA_MOE_MIN_SLOTS=N`

### CUDA Top-k MoE Fusion Is Disabled By Default

CUDA top-k MoE fusion is still disabled for normal `LLAMA_MOE_OFFLOAD` runs.
`LLAMA_MOE_TOPK_FUSION=1` is ignored in offload builds.

Phase H found and fixed the earlier single-token decode correctness failure
behind an explicit diagnostic gate, `LLAMA_MOE_TOPK_FUSION_DIAG=1`. The root
cause was the callback/fusion boundary: streaming mode stopped graph execution
at `ffn_moe_topk`, but the fused CUDA router needs the graph segment through
final routing weights to execute as one unit. Phase H registers final routing
weights as the callback point while still reading expert IDs from the original
top-k tensor.

Current status:

- Diagnostic decode top-k fusion passed the synthetic routing test.
- Golden logits passed with top-k fusion alone, top-k + GLU fusion, and top-k +
  GLU fusion + guarded graphs: `max|d| = 0`.
- `llama-cli` chat smoke passed with the full diagnostic stack.
- The same-build 8000 MiB EAMC benchmark did not show a material decode gain:
  TPOT was flat at 31.91 -> 31.88 ms/token, `topk_d2h_us` stayed flat at
  1.58 ms/token, and callback wall time was slightly worse.
- The large Phase H prefill speedup is not evidence for top-k fusion. It is
  present in the Phase H baseline with diagnostic fusion disabled, because the
  callback now reads strided top-k views correctly with
  `ggml_backend_tensor_get_2d()`.

Impact:

- Correctness is preserved by keeping the normal path default-off.
- The diagnostic gate exists for focused decode experiments only.
- Do not promote top-k fusion until a broader benchmark shows consistent TPOT
  or callback/top-k-D2H improvement without regressing H2D, stall, predictor,
  SSD, hit rate, or misses/token.

### Specialized `.slot` `MUL_MAT_ID` Paths Are Mostly Bypassed

By default, CUDA `MUL_MAT_ID` on `.slot` tensors still uses the generic sorted
CUDA path. Phase E added guarded quantized single-token `.slot` MMVQ decode
with `LLAMA_MOE_SLOT_MMVQ=1`; Phase F added optional CUDA graph capture for
that same decode shape with `LLAMA_MOE_SLOT_GRAPHS=1`; Phase G added optional
decode-only `.slot` quantized `MUL_MAT_ID + GLU` fusion with
`LLAMA_MOE_SLOT_GLU_FUSION=1`.

Current status:

- The default remains correctness-first.
- The guarded MMVQ + graph decode path passed the synthetic CUDA graph replay
  test, golden-logit gate, chat smoke, and the 8000 MiB EAMC benchmark on the
  2026-06-12 dev-box run.
- The Phase E benchmark improved TTFT from 19704.8 ms to 18481.0 ms and TPOT
  from 62.13 ms/token to 47.20 ms/token versus Phase D; H2D, stall, predictor
  time, SSD read time, callback wall time, and hit rates were flat or better.
- The Phase F benchmark improved TPOT from 47.20 ms/token to 30.63 ms/token
  and decode `compute_us` from 24.34 ms/token to 10.46 ms/token versus Phase E.
  This benchmark used the static validation build because the shared CUDA DLL
  was blocked by Windows Smart App Control.
- The Phase G benchmark improved TPOT from 31.56 ms/token to 30.67 ms/token
  versus the Phase F guard stack in the same static build and reduced decode
  callback wall time from 17.78 ms/token to 17.06 ms/token. Decode
  `gpu_compute` was effectively flat at 10.44 to 10.37 ms/token.
- Phase H fixed strided top-k callback reads on the default path. On the 8000
  MiB EAMC benchmark this improved prefill from 14021.9 ms to 5070.4 ms versus
  the Phase G GLU run, mostly by reducing false expert demand in prefill
  callbacks.
- Phase H also fixed top-k fusion correctness under
  `LLAMA_MOE_TOPK_FUSION_DIAG=1`, but kept it default-off because decode TPOT
  and `topk_d2h_us` were flat in the same-build top-k-fusion benchmark.
- Phase I added guarded multi-token `.slot` MMVQ prefill behind
  `LLAMA_MOE_PREFILL_MMVQ=1`, used together with `LLAMA_MOE_SLOT_MMVQ=1`.
  Synthetic CUDA coverage passed with changing slot IDs and changed slot tensor
  contents, raw golden logits passed with `max|d|=0`, and `llama-cli --jinja
  --reasoning off` chat smoke passed at default ubatch and forced
  `LLAMA_MOE_STREAMING_UBATCH=8`.
- The Phase I same-build 8000 MiB EAMC benchmark improved TTFT from 6505.2 ms
  to 6202.5 ms, but decode TPOT regressed from 36.72 ms/token to
  40.85 ms/token and decode callback wall time rose from 22.40 ms/token to
  26.00 ms/token. Keep `LLAMA_MOE_PREFILL_MMVQ=1` experimental/default-off.
- `.slot` MMQ/MMF, prefill graphs/fusion, generic sorted graph capture, normal
  top-k fusion, and non-GLU fusion remain bypassed or disabled until
  separately validated.

### EAMC Row Caps Are Diagnostic Only

Default EAMC scoring uses the full sparse corpus. `LLAMA_MOE_EAMC_ROWS=N`
exists for experiments, but row caps or approximate EAMC scoring must not
become defaults until a clean same-start-sidecar benchmark preserves hit rates.
The recorded Phase C comparison had a non-identical starting sidecar, so the
decode hit-rate delta from that run is not a clean default-change gate.

### Profiler `stall_us` Is Approximate

`h2d_us` and `compute_us` are event-based where CUDA events are available.
After Perf-D, the normal async path no longer synchronizes the host on every
H2D event before recycling pinned memory. `stall_us` is still host wall time
around miss-loop no-progress/yield cases and fallback synchronization, so it is
an approximation of overlap loss rather than a pure GPU timeline metric.

### Windows Application Control Can Block Fresh CUDA Builds

On the 2026-06-12 Perf-D dev-box run, Windows Application Control blocked the
freshly rebuilt compressed `ggml-cuda.dll` with status `0xc0e90002`.
Configuring `build-moe` with `-DGGML_CUDA_COMPRESSION_MODE=none` produced a
loadable CUDA backend for benchmarking.

During Phase F, Smart App Control later blocked the rebuilt shared
`build-moe\bin\Release\ggml-cuda.dll` even with compression disabled. The
symptom is `llama-cli.exe` or `llama-moe-bench.exe` exiting immediately with
`0xc0e90002` while Code Integrity logs that the process attempted to load
`ggml-cuda.dll` and it did not meet Enterprise signing requirements. The local
dev-box workaround was to use a static CUDA validation build and copy the
static `llama-cli.exe`, `llama-moe-bench.exe`, and `llama-completion.exe` into
`build-moe\bin\Release`; the original shared launchers were backed up under
`build-moe\bin\Release\shared-launcher-backup-phase-f-*`.

The same policy can block freshly rebuilt unsigned test executables, which
affected `test-cuda-stream.exe` during CTest validation.

The fix is to disable Windows Application Control in Settings--Privacy Settings--Security Settings

## Deferred

The following remain outside MVP/current performance scope:

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

### Fixed Streaming UBatch Cap And Small-Cache Warmup Abort

The old unconditional streaming cap of `n_ubatch <= 8` has been replaced by
adaptive sizing. The recommendation is derived from `n_slots / top_k` and is
bounded by the requested ubatch, with a minimum effective ubatch of 1.

The 2026-06-09 small-cache warmup abort (`n_uniq=256 exceeds n_slots=96`) was
also fixed by keeping MoE warmup routing at the model top-k instead of expanding
to all experts in streaming mode.

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

The Phase E guarded MMVQ decode path also passed golden logits with
`LLAMA_MOE_SLOT_MMVQ=1`, cache=8000 MiB, `-n 32`, `-ub 8`, and `max|d|=0`.

The repacker was not the cause. Original vs repacked/offload-disabled logits
match, full-residency offload matches, and streaming matches full residency
after the runtime/CUDA correctness guards.

### Repacker Layout And Table Validation

The implemented layout is documented as `fused-tensors-page-aligned-v1`, with
fused expert tensors retained and `moe_offload.expert_blob.table` providing
per-expert byte ranges.

`test-repack-slices` compares every table entry against the exact byte slice in
the original GGUF when `LLAMA_MOE_TEST_ORIGINAL_GGUF` and
`LLAMA_MOE_TEST_GGUF` are provided.

### EAMC Sidecar Persistence

`--moe-eamc-path PATH` is wired through common args, `llama-bench`, and
`llama-moe-bench`. EAMC loads/saves a binary sidecar with shape/version checks
and ignores incompatible sidecars with a warning.

During inference, EAMC stays online and appends rows to the in-memory corpus,
but the sidecar is not written after each internal `llama_decode()` batch.
Persistence is deferred to explicit benchmark end or context/session teardown.
`test-eamc-cosine` covers sidecar round-trip and bounded FIFO/ring replacement.

### EAMC Predictor Overhead

Phase B removed the hidden per-token sidecar save and quadratic full-corpus
redundancy pruning from the decode hot path. Phase C then moved EAMC scoring to
sparse in-memory rows, an inverted `(layer, expert)` index for uncapped
dense-equivalent cosine scoring, and lazy per-callback score-vector
materialization.

The 8000 MiB, 256 prefill + 256 decode, 3-repeat Phase C run reduced decode
predictor scoring from the Phase B 90.54 ms/token measurement to about
2.52 ms/token. The Phase E guarded-MMVQ benchmark reported 1.32 ms/token
predictor time.

### Per-Completion H2D Host Synchronization

Phase D removed the normal per-completion `cudaEventSynchronize()` from pinned
H2D buffer recycling. The compute stream still waits on each H2D completion
event for correctness, but pinned buffers are now returned to the pool after
nonblocking event polling reports completion. Decode `stall_us` fell from about
16.8 ms/token to 0.41 ms/token in Phase D, 0.12 ms/token in Phase E, and about
0.13 ms/token in the Phase F graph run.

### Guarded `.slot` MMVQ Decode Validation

Phase E restored quantized single-token `.slot` MMVQ decode behind
`LLAMA_MOE_SLOT_MMVQ=1`. Phase F then enabled CUDA graph capture for the same
decode-only path behind `LLAMA_MOE_SLOT_GRAPHS=1`. The default remains generic
sorted CUDA, but the guarded paths passed:

- synthetic CUDA `test-slot-mmvq`,
- repeated graph replay with changing slot ids and slot contents,
- golden-logit gate with `max|d|=0`,
- `llama-cli` chat smoke,
- 8000 MiB EAMC benchmark with faster TTFT and TPOT.

### Oracle Ambiguity

`test-moe-oracle-failfast` verifies `--moe-oracle` exits as post-MVP instead of
silently enabling an unsupported path.

### Earlier Lifecycle Issues

IO worker shutdown, pinned-buffer starvation, multi-token prompt hangs,
queue-full accounting, failed I/O completion reporting, and same-callback
eviction of just-reserved experts were closed before the final June 5 and
June 9 validation passes.
