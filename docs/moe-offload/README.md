# MoE Offload MVP

This fork adds a guarded MoE offload overlay behind `LLAMA_MOE_OFFLOAD`. The default build path remains off.

## Build

```bash
cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON
cmake --build build-moe --config Release --target llama-moe-bench llama-completion llama-moe-repack
cmake --build build-moe\tools\cli --config Release --target llama-cli
```

## Repack

```powershell
.\build-moe\bin\Release\llama-moe-repack.exe `
  --input C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.gguf `
  --output C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --manifest C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf.json
```

The current repacker writes a stock-loadable GGUF with page-aligned tensor data and `moe_offload.*` metadata. It records the layout as `fused-tensors-page-aligned-v1`: the original fused expert tensors remain in the file, and `moe_offload.expert_blob.table` records each `(logical_layer, expert, kind)` byte range relative to the GGUF data section.

## Chat with `llama-cli`

Use `llama-cli` as the supported interactive chat frontend for MoE-offloaded
models:

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

For this Qwen3.5 GGUF, keep `--jinja` enabled and use `-sys` for system
instructions. Use `--reasoning off` for normal chat; `--reasoning-budget 0`
can still stream reasoning text for this model and is not the recommended chat
setting.

`llama-cli --moe-offload` defaults to correctness-first `n_ubatch=1` because
the current batched streaming prefill path can corrupt Qwen MoE chat logits.
Set `LLAMA_MOE_STREAMING_UBATCH=N` only for diagnostics while the batched path
remains under investigation.

Use `llama-completion -no-cnv` for raw completion and logit diagnostics. Avoid
using `-p "Hello"` as a system prompt in conversation mode; it seeds the
conversation as user-visible text.

Default D-3/D-4 slot-pool traces are quiet in chat. Set
`LLAMA_MOE_DEBUG_D4=1`, `LLAMA_MOE_DEBUG_LOADS=1`, or
`LLAMA_MOE_DEBUG_TRACE=1` when those diagnostics are needed.

## Raw Completion

```powershell
.\build-moe\bin\Release\llama-completion.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 8000 `
  --moe-predictor eamc `
  -p "Hello" -n 32
```

`--moe-predictor` accepts `lru` or `eamc`. `--moe-eamc-path PATH` selects the
EAMC predictor sidecar; when omitted with `--moe-predictor eamc`, the runtime
uses the model path with a `.eamc` suffix. EAMC updates remain online during
inference, but sidecar persistence is deferred: the hot path updates the
in-memory corpus and saves the sidecar only at explicit benchmark end or
context/session teardown. `--moe-profile-csv PATH` and
`--moe-profile-summary PATH` write profiling data.

In streaming mode (`n_slots < n_experts`), the runtime auto-sizes the effective
`n_ubatch` from the cache VRAM budget and model top-k instead of applying the
old fixed cap of 8. Very small cache budgets still reserve at least one
token's top-k working set per layer; below that point the runtime warns that it
is using the minimum viable slot count. Set `LLAMA_MOE_STREAMING_UBATCH=N` for
a fixed diagnostic value, `LLAMA_MOE_STREAMING_UBATCH=off` to disable
auto-sizing, or `LLAMA_MOE_UBATCH_SAFETY=F` to tune the slot-budget safety
factor.

## Bench

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 1024 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 12000 --moe-predictor eamc `
  --moe-eamc-path C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.eamc `
  --moe-profile-csv moe-profile.csv `
  --moe-profile-summary moe-summary.txt `
  -ub 16
```

The bench tool enables MoE offload automatically, runs prefill + decode loops,
and always prints the summary report to stdout. It accepts `-ub`, `--ubatch`,
and `--ubatch-size` for explicit prefill-ubatch measurements. If `--moe-profile-summary` is
set, `llama-moe-bench` writes the same §4.7 report to that path after all
repeats complete. During long runs, it also checkpoints the same summary file
after prefill and after each completed repeat, so the file appears before the
full benchmark finishes. It intentionally does not use the runtime
`end_request()` summary writer, which emits the older aggregate-only format. If
`--moe-profile-csv` is set, a per-layer profile CSV is written to that path.

Example summary:
```
model: Qwen3.5-35B-A3B Q4_K_M
predictor: eamc      cache: 12000 MB   ssd: C:
n_prompt: 1024  n_gen: 256  repeats: 3

ubatch: requested=16 effective=16  slots=145/256  mode=streaming

phase     tokens   total_ms   per_token_ms   tok/s
prefill     1024     1320.4          1.29     774
decode       256     9874.2         38.57      26

cache hit rate (prefill): 0.0%
cache hit rate (decode): 78.3%
SSD bytes read (decode): 3.42 GB  (avg 13.4 MB/token)
TTFT: 1320.4 ms
TPOT: 38.57 ms
total: 11194.6 ms

I/O breakdown (decode, mean per token):
  ssd_read          21.50 ms
  h2d                7.80 ms
  gpu_compute        0.00 ms
  stall (overlap loss)     29.30 ms
  predictor          0.04 ms

VRAM peak: 14.80 GB / 15.92 GB  (experts budget: 7.81 GB, other model/kv/compute: 6.99 GB)
DRAM peak (process): 1.40 GB
SSD reads: 18234 (avg 0.19 MB each, avg latency 1.18 ms)
profile rows: prefill=40 decode=10240
```

CSV columns:

| Column | Meaning |
| --- | --- |
| `row_type` | `layer` for per-layer callback rows, `request` for one row per internal `llama_decode()` batch. |
| `request_idx` | Monotonic internal decode-batch id. |
| `repeat_idx` | Benchmark repeat id when set by `llama-moe-bench`, otherwise `-1`. |
| `batch_idx` | Benchmark batch id when set by `llama-moe-bench`; prefill is `0`, decode tokens start at `1`. |
| `token_idx` | First token index covered by this callback row. Prefill rows cover a ubatch; decode rows cover one token. |
| `phase` | `prefill` or `decode`. |
| `layer` | Logical MoE layer index. |
| `k_required` | Distinct experts required by the router for this layer/callback. |
| `k_hit` | Required experts already resident in the slot cache. |
| `k_miss` | Required experts loaded from the repacked GGUF. |
| `ssd_read_us` | Wall microseconds spent in `fread` for misses (worker-thread-measured). |
| `h2d_us` | Microseconds spent in host→device copy. On the CUDA async path this is the sum of `cudaEventElapsedTime` between the per-blob begin/end events recorded on the dedicated MoE H2D stream; on the sync fallback it is host-measured around `ggml_backend_tensor_set`. Mixed contribution per row when both paths fire. |
| `compute_us` | CUDA-event-timed compute interval for the layer, derived from begin/end events recorded on the compute stream and queried at `slot_pool_end_request()`. Each row's interval brackets the slot_table write of layer L through the topk callback of layer L+1 (or `end_request` for the final layer); this is an over-approximation that includes layer L's MoE compute plus the following attention block. |
| `stall_us` | Host wall-clock around miss completion and compute-stream wait insertion for miss events. This is still an approximation of overlap loss, not a pure CUDA event duration. |
| `pred_us` | Host microseconds spent in predictor `observe()` plus eviction `score()` calls for this row. |
| `pred_observe_us` | Predictor observation time. |
| `pred_score_us` | Predictor eviction-score time. |
| `callback_wall_us` | Host wall-clock time for the MoE eval callback. |
| `topk_d2h_us` | Host time spent reading top-k expert ids from the backend tensor. |
| `slot_ids_h2d_us` | Host time spent writing remapped slot ids. |
| `slot_table_h2d_us` | Host time spent writing the legacy slot-table path when used. |
| `cache_resident_experts` | Number of experts resident for the layer after miss handling. |
| `predictor` | `lru` or `eamc`. |
| `request_wall_us` | Request-row wall time for the internal `llama_decode()` batch. |
| `request_end_us` | Request-row time spent finalizing MoE profiler/predictor state. |
| `predictor_end_us` | Request-row time spent appending online predictor state. For Phase B EAMC this is in-memory only. |
| `predictor_save_us` | Request-row sidecar save time. This should be zero during per-token decode for Phase B EAMC. |
| `profile_flush_us` | Request-row CSV flush time. |
| `sidecar_write_bytes` | Request-row sidecar bytes written. This should be zero during per-token decode for Phase B EAMC. |

## Current State

Implemented in this slice:

- `LLAMA_MOE_OFFLOAD` build option.
- Guarded CLI/model parameters: `--moe-offload`, cache budget knobs, predictor choice, EAMC sidecar path, profile paths, and oracle fail-fast.
- Loader metadata validation for `moe_offload.version`.
- LRU and EAMC predictor implementations, including binary EAMC sidecar load/save with online in-memory updates and deferred persistence.
- Slot-cache model and profiling CSV/summary plumbing, including predictor observe/score/end/save timing.
- Guarded hooks at model load, decode request boundaries, and MoE top-k graph construction.
- `llama-moe-repack` and `llama-moe-bench` targets.
- **D-4**: Pre-allocated slot_table tensors via `create_unfiled_tensor`; zero-init of all slot tensors; adaptive streaming ubatch sizing from the slot/VRAM budget.
- **D-5**: Threaded I/O worker (`src/moe-offload/io.{h,cpp}`).
- **E**: Per-layer CSV profiling rows; snapshot summary; LRU/EAMC predictor wired into eval-callback eviction.
- **F**: `llama-moe-bench` emits the §4.7-style stdout/file summary and writes nonempty CSV rows when requested.
- **G**: CUDA stream accessor (`moe_io_cuda_*` in `ggml/src/ggml-cuda/moe_offload_io.cu`), pinned-host staging, dedicated MoE H2D stream, event pool.
- **H**: Eval-callback rewired to async H2D via `io_h2d_async_timed` + `io_compute_wait` (`cudaStreamWaitEvent` on the compute stream). Synchronous double-read of the miss blob removed.
- **I**: Real `compute_us` via CUDA events. Profile rows are buffered per `llama_decode` batch and flushed at `slot_pool_end_request()` after `cudaEventElapsedTime` queries. Per-miss `h2d_us` now comes from event timing on the async path.
- **Perf-B**: EAMC remains online in DRAM, but per-token sidecar saves were removed. Full-capacity EAMC uses FIFO/ring replacement instead of quadratic redundancy pruning; sidecars are saved at benchmark end or context/session teardown.
- **J**: `--logit-dump PATH` on `llama-completion`, `tests/moe-offload/compare_logits.py`, and PowerShell harness `tests/moe-offload/test-golden-logits.ps1`. See "Correctness" below.
- **K**: C++ unit tests registered with CTest under label `moe-offload` (`test-eamc-cosine`, `test-lru-eviction`, `test-manifest-roundtrip`, `test-repack-slices`) alongside Phase G's `test-cuda-stream` smoke and the `test-moe-oracle-failfast` CLI test. Run with `ctest --test-dir build-moe -C Release -L moe-offload`.
- **MVP closeout**: Streaming slot tensors now default to the actual cache slot count; set `LLAMA_MOE_FULL_EXPERT_AXIS=1` only as a guarded fallback. CUDA `MUL_MAT_ID` on `.slot` tensors bypasses the specialized MMVQ/MMQ/MMF paths and uses the generic sorted path for correctness-first validation.
- **L**: IO worker shutdown wired into `llama_context::~llama_context()`; small-cache and multi-token-prompt streaming deadlocks fixed by excluding just-reserved experts from LRU/predictor eviction within the same callback. Larger-ubatch miss loading now submits and drains I/O in chunks instead of requiring the pinned-buffer pool to cover every blob for the layer.

## Correctness

Streaming numerical correctness is gated by a per-decode logit dump and a
full-residency vs. streaming comparison. The harness lives at
`tests/moe-offload/test-golden-logits.ps1` and is *not* registered with
CTest (the reference model is multi-GB).

How to run:

```powershell
powershell -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
    -Tol 1e-3 -NPredict 8 -StreamCacheMb 12000
```

The harness runs `llama-completion --logit-dump` twice on the same model
and seed: once at `--moe-cache-vram-mb 99999` (full residency, reference)
and once at `--moe-cache-vram-mb 12000` (streaming, n_slots=145 vs.
n_experts=256 — forces eviction every layer). `tests/moe-offload/compare_logits.py`
then mmaps both binary dumps and reports `max|Δlogit|` / `mean|Δlogit|`.
Exit 0 iff `max|Δlogit| < --tol`.

Observed on the dev box (2026-05-28, Qwen3.5-35B-A3B-Q4_K_M, prompt
`"Hello"`, `--temp 0 --seed 42 -n 8`, 12000 MiB streaming cache,
n_slots=145, hits=630 misses=970):

| Metric | Value |
|---|---|
| `n_steps` | 8 |
| `n_vocab` | 248320 |
| `max|Δlogit|` | **4.64e-01** |
| `mean|Δlogit|` | 3.67e-02 |
| `tol` | 1e-3 |
| Result | **FAIL** (drift first observed at step 3) |

The drift is real — the gate is doing its job. Streaming-mode top-1 tokens
still match full-residency for the first 8 decodes at temp=0 (per Phase H/I
text-equivalence smoke), but the full softmax row is not bit-equivalent.
Root cause is not yet pinned down; Phase J anticipated a stale
`topk → slot_table` registration on a rebuild path
(`reset_graph_state()` now also fires in `llama_context::graph_reserve()`),
but that alone did not move the measured drift. Remaining hypotheses:

- A slot whose contents are replaced between the eval callback finishing
  H2D and the mul_mat_id consuming the row (eviction/wait ordering bug
  in `slot_pool::moe_eval_callback`).
- An expert whose slot_table entry is set by a later layer's callback
  before the current layer's mul_mat_id runs (callback ordering vs.
  scheduler topology).

The June 5 closeout changes address the two leading runtime suspects from this
section: slot tensors now use the actual cache slot count by default, and CUDA
`MUL_MAT_ID` on `.slot` tensors bypasses the specialized MMVQ/MMQ/MMF paths.
The historical drift number above remains in the document until the dev-box
golden-logit harness is rerun against the new build.

Deviations from the source plan (`implementation_plan_mvp_20260529.md` §3):

- `--moe-cache-vram-mb 4000` previously deadlocked during prefill on this
  dev box; Phase L raised the IO worker's pinned-buffer pool floor so the
  miss-loop can submit a layer's worst-case unique-expert demand without
  starving on staging buffers. Re-tested at cache=4000 with `-n 4 -p
  "Hello"`: runs to completion. The harness still defaults to 12000 MiB
  because the dev-box reference run was captured at that size; lower
  caches can now be used safely.
- The plan called for prompt `"The quick brown fox"`. Multi-token-prompt
  streaming hangs (same buffer-pool starvation) are also fixed by the
  Phase L bump. Re-tested with prompt `"The quick brown fox"` at
  cache=12000, `-n 4`: prefill + decode complete cleanly.
- Phase J's measured drift number (`max|Δlogit|=4.64e-01`) was captured
  on the build that produced the first `"Hello, I am a 20 year"` smoke.
  Subsequent regressions on the same HEAD have made full-residency
  itself crash with `GGML_ASSERT(buf != NULL && "tensor buffer not set")`
  inside `prefetch_all_experts`; this regression is pre-existing
  relative to Phase L (confirmed by `git stash` baseline) and prevents
  the harness from regenerating the reference dump. Root-causing this
  regression is on the post-MVP list; the Phase J drift number stays in
  the table above as the last good measurement.

## Test surface

CTest targets under label `moe-offload`:

| Test | Phase | Purpose |
|---|---|---|
| `test-cuda-stream` | G | Pinned alloc + async H2D + event ordering smoke. |
| `test-eamc-cosine` | K / Perf-B | EAMC cosine ordering, sidecar round-trip, and bounded FIFO/ring replacement on full insert. |
| `test-lru-eviction` | K | Hand-computed LRU score values, victim ordering, per-layer isolation. |
| `test-manifest-roundtrip` | K | GGUF manifest sanity (version=2, table size, per-record file ranges). Self-skips with exit 0 when `LLAMA_MOE_TEST_GGUF` is unset. |
| `test-repack-slices` | MVP closeout | Byte-level original-vs-repacked expert slice verification. Self-skips unless `LLAMA_MOE_TEST_ORIGINAL_GGUF` and `LLAMA_MOE_TEST_GGUF` are set. |
| `test-moe-oracle-failfast` | MVP closeout | Verifies `--moe-oracle` exits with a post-MVP error. |

```powershell
ctest --test-dir build-moe -C Release -L moe-offload --output-on-failure
```

Dev-box-only PowerShell harnesses under `tests/moe-offload/` (not
registered with CTest because they need the multi-GB model):

- `test-golden-logits.ps1` — Phase J streaming-vs-full-residency logit
  comparison gate.
- `compare_logits.py` — binary-dump comparator used by the harness.

- `test-llama-cli-chat.ps1` - `llama-cli --moe-offload --jinja`
  chat smoke; asserts identity, France, and arithmetic answers are coherent
  and default D-3/D-4 traces are quiet.

See [`tests/moe-offload/README.md`](../../tests/moe-offload/README.md)
for runtime prereqs and invocation details.

## Troubleshooting

### "n_uniq exceeds n_slots" error
```
moe_eval_callback: n_uniq=96 exceeds n_slots=48
```
Default auto-sizing should avoid this during normal streaming runs by reducing
the effective ubatch as far as 1 and by reserving the model top-k slots as the
minimum viable cache. If this appears, check for diagnostic overrides such as
`LLAMA_MOE_STREAMING_UBATCH=off`, a forced `LLAMA_MOE_STREAMING_UBATCH=N` that
is too large for the cache, or a forced `LLAMA_MOE_MIN_SLOTS` below model
top-k.

### Adaptive streaming ubatch
Streaming mode auto-sizes `n_ubatch` from `n_slots / top_k` and rounds to a
common graph size. Larger `--moe-cache-vram-mb` values therefore allow larger
prefill microbatches. The startup log prints the requested and effective
ubatch. If the budget fits fewer than top-k slots, startup warns and reserves
the minimum one-token working set so inference still runs, just with effective
ubatch 1.

For diagnostics:

```powershell
$env:LLAMA_MOE_STREAMING_UBATCH="16"   # force a fixed effective ubatch
$env:LLAMA_MOE_STREAMING_UBATCH="off"  # disable auto-sizing
$env:LLAMA_MOE_UBATCH_SAFETY="0.5"     # choose a more conservative auto value
$env:LLAMA_MOE_MIN_SLOTS="8"           # override the minimum working set
```

### CUDA illegal memory access at launch_mul_mat_q
The old fixed-cap workaround targeted an older remap path. Current streaming
uses callback-filled slot-id tensors and generic `.slot` `MUL_MAT_ID`.
Large-ubatch failures should now surface as `n_uniq exceeds n_slots` or an I/O
progress error; a CUDA illegal access is a regression and should be captured
with the cache size, effective ubatch, and prompt.
