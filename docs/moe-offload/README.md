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
  --moe-cache-vram-mb 12000 `
  --moe-predictor lru `
  --moe-fast-paths `
  --jinja `
  --reasoning off `
  -sys "You are a helpful assistant."
```

For this Qwen3.5 GGUF, keep `--jinja` enabled and use `-sys` for system
instructions. Use `--reasoning off` for normal chat; `--reasoning-budget 0`
can still stream reasoning text for this model and is not the recommended chat
setting.

`llama-cli --moe-offload` defaults to correctness-first `n_ubatch=1` unless
`--moe-fast-paths` or `LLAMA_MOE_FAST_PATHS=1` is set. The fast profile uses
the accepted Phase K guard stack, lets the runtime auto-size streaming ubatch,
and prints the effective `n_ubatch` after model load. Keep
`LLAMA_MOE_STREAMING_UBATCH=N` only for diagnostics or fallback testing.

Set `LLAMA_MOE_SLOT_MMVQ=1` to enable the guarded CUDA `.slot` quantized
single-token MMVQ decode fast path. Set `LLAMA_MOE_SLOT_GRAPHS=1` as well to
enable CUDA graph capture for that same guarded decode shape. Set
`LLAMA_MOE_SLOT_GLU_FUSION=1` to additionally enable the guarded decode-only
`.slot` `MUL_MAT_ID + GLU` fusion path. Set `LLAMA_MOE_PREFILL_MMVQ=1`
together with `LLAMA_MOE_SLOT_MMVQ=1` to enable the guarded multi-token
`.slot` MMVQ prefill path. These guards are off by default; MMQ/MMF, prefill
graphs/fusion, generic sorted graph capture, and CUDA top-k MoE fusion remain
disabled for normal `.slot`/offload runs. Phase H adds
`LLAMA_MOE_TOPK_FUSION_DIAG=1` for focused single-token decode diagnostics
only; it is default-off because the same-build benchmark did not show a
material TPOT or `topk_d2h_us` gain. Phase H's large prefill win came from the
default strided top-k callback read fix, not from enabling top-k fusion.

`--moe-fast-paths` is the documented human-facing profile for interactive
chat. It enables the accepted guards above, leaves prefill MMVQ disabled, and
does not turn on diagnostic top-k fusion.

To profile the same interactive path:

```powershell
.\build-moe\bin\Release\llama-cli.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 12000 `
  --moe-predictor lru `
  --moe-fast-paths `
  --moe-profile-csv tests\moe-offload\_out\cli-profile.csv `
  --moe-profile-summary tests\moe-offload\_out\cli-profile.summary.txt `
  --jinja `
  --reasoning off `
  -sys "You are a helpful assistant."
```

The fallback path is the same command without `--moe-fast-paths`; it reports
effective `n_ubatch=1`. You can also set `LLAMA_MOE_STREAMING_UBATCH=1` to
force the conservative path during diagnostics.

Short interactive turns are not valid TPOT comparisons against
`llama-moe-bench`: bench runs a synthetic direct-decode loop with a fixed prompt
and generation length, while `llama-cli` adds chat-template, server-task,
streaming, sampling, and console work and may route to a different expert set.
Use the same cache budget, predictor, fast-path stack, cache reset/warm policy,
and enough generated tokens to average out cold-start noise. In the final
Phase G closeout check, matched 12000 MiB runs with the accepted guard stack
measured bench-like `llama-cli` decode speed: LRU was 28.36 ms/token in CLI
versus 27.26 ms/token in bench, and EAMC was 34.59 ms/token in CLI versus
31.27 ms/token in bench.

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
context/session teardown. EAMC scoring uses sparse in-memory activation rows,
an indexed cosine pass, and lazy per-callback score materialization. The default
path scores the full corpus; `LLAMA_MOE_EAMC_ROWS=N` is diagnostic-only and
must be hit-rate gated before becoming a default. `--moe-profile-csv PATH` and
`--moe-profile-summary PATH` write profiling data. CLI profile CSV is
per-layer/request; the summary now follows the bench-style MoE report shape.

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
and always prints the summary report to stdout. It is a synthetic direct-decode
measurement; use matched settings before comparing its TPOT to interactive
`llama-cli`. It accepts `-ub`, `--ubatch`, and `--ubatch-size` for explicit
prefill-ubatch measurements. Use
`--moe-reset-cache-between-repeats` to measure cold-cache prefill for every
repeat, `--moe-warm-cache` to run one unmeasured cache warmup before timing,
and `--moe-hot-start` for benchmark-only EAMC-sidecar preloading. Hot-start is
experimental and default-off; it is useful for calibration, not a recommended
default. If `--moe-profile-summary` is
set, `llama-moe-bench` writes the same §4.7 report to that path after all
repeats complete. During long runs, it also checkpoints the same summary file
after prefill and after each completed repeat, so the file appears before the
full benchmark finishes. It intentionally does not use the runtime
`end_request()` summary writer, which emits the older aggregate-only format. If
`--moe-profile-csv` is set, a per-layer profile CSV is written to that path.

Set `LLAMA_MOE_SLOT_MMVQ=1` to enable the guarded CUDA `.slot` quantized
single-token MMVQ decode fast path. Set `LLAMA_MOE_SLOT_GRAPHS=1` as well to
enable CUDA graph capture for that same guarded decode shape. Set
`LLAMA_MOE_SLOT_GLU_FUSION=1` to additionally enable the guarded decode-only
`.slot` `MUL_MAT_ID + GLU` fusion path. Set `LLAMA_MOE_PREFILL_MMVQ=1`
together with `LLAMA_MOE_SLOT_MMVQ=1` to enable the guarded multi-token
`.slot` MMVQ prefill path. These guards are off by default; MMQ/MMF, prefill
graphs/fusion, generic sorted graph capture, and CUDA top-k MoE fusion remain
disabled for normal `.slot`/offload runs. Phase H's
`LLAMA_MOE_TOPK_FUSION_DIAG=1` path is a diagnostic single-token decode path,
not a recommended benchmark default. The Phase H prefill speedup is present in
the baseline run with top-k fusion disabled because callback reads now handle
strided top-k views correctly.

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
TTFT cold: 1320.4 ms  (n=1)
TTFT warm: 0.0 ms  (n=0)
TPOT: 38.57 ms
total: 11194.6 ms

I/O breakdown (prefill, mean per token):
  ssd_read           6.82 ms
  h2d                4.87 ms
  gpu_compute        7.49 ms
  stall (overlap loss)      0.09 ms
  predictor          0.35 ms

Profiler breakdown (prefill, mean per token):
  predictor_observe     0.00 ms
  predictor_score       0.35 ms
  callback_wall         8.48 ms
  topk_d2h              0.11 ms
  slot_ids_h2d          0.03 ms
  slot_table_h2d        0.00 ms
  eamc_cosine           0.34 ms
  eamc_materialize      0.00 ms
  eamc_rows_scored    152.0 rows/token
  eamc_cache_hits     134.85 hits/token
  eamc_cache_misses     0.15 misses/token
  request_end           0.07 ms
  predictor_end         0.01 ms
  predictor_save        0.00 ms
  profile_flush         0.00 ms
  sidecar_written       0.00 MB/token

I/O breakdown (decode, mean per token):
  ssd_read          21.50 ms
  h2d                7.80 ms
  gpu_compute        0.00 ms
  stall (overlap loss)     29.30 ms
  predictor          0.04 ms

Profiler breakdown (decode, mean per token):
  predictor_observe     0.00 ms
  predictor_score       0.04 ms
  callback_wall        31.00 ms
  topk_d2h              1.80 ms
  slot_ids_h2d          0.30 ms
  slot_table_h2d        0.00 ms
  eamc_cosine           0.03 ms
  eamc_materialize      0.00 ms
  eamc_rows_scored    256.0 rows/token
  eamc_cache_hits       8.00 hits/token
  eamc_cache_misses     1.00 misses/token
  request_end           0.10 ms
  predictor_end         0.00 ms
  predictor_save        0.00 ms
  profile_flush         0.02 ms
  sidecar_written       0.00 MB/token

Wall/profile reconciliation (decode):
  wall_decode_us        29622600
  profiled_decode_us    30012000
  unattributed_decode_us -389400

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
| `stall_us` | Host wall-clock spent waiting/yielding for miss progress after async H2D copies are submitted. The normal CUDA path inserts a compute-stream event wait and recycles pinned buffers later via event polling, so this no longer includes a host synchronize for every H2D copy. It is still an approximation of overlap loss, not a pure CUDA event duration. |
| `pred_us` | Host microseconds spent in predictor `observe()` plus eviction `score()` calls for this row. |
| `pred_observe_us` | Predictor observation time. |
| `pred_score_us` | Predictor eviction-score time. |
| `callback_wall_us` | Host wall-clock time for the MoE eval callback. |
| `topk_d2h_us` | Host time spent reading top-k expert ids from the backend tensor. |
| `slot_ids_h2d_us` | Host time spent writing remapped slot ids. |
| `slot_table_h2d_us` | Host time spent writing the legacy slot-table path when used. |
| `eamc_rows_scored` | EAMC corpus rows considered while computing nearest-neighbor scores for this callback. Zero for non-EAMC predictors or callbacks that do not score. |
| `eamc_cosine_us` | Host microseconds spent computing EAMC nearest-neighbor cosine scores. |
| `eamc_score_materialize_us` | Host microseconds spent materializing the per-layer EAMC score vector after nearest-neighbor scoring. |
| `eamc_score_cache_hits` | Count of EAMC score-vector cache hits within eviction candidate scoring. |
| `eamc_score_cache_misses` | Count of EAMC score-vector materializations within eviction candidate scoring. |
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
- **Perf-C**: EAMC scoring uses sparse corpus rows, an inverted `(layer, expert)` index for dense-equivalent uncapped cosine scoring, lazy per-callback score-vector materialization, and profiler counters for cosine/materialization/cache activity. Corpus row caps remain diagnostic-only through `LLAMA_MOE_EAMC_ROWS=N`.
- **Perf-D**: Async H2D pinned buffers are no longer recycled by synchronizing the host on every copy event. The compute stream still waits on each H2D end event for correctness; pinned buffers are tracked in an inflight list and returned to the pool after `io_event_query()` reports completion. Miss submissions are sorted by file offset. On the 8000 MiB EAMC benchmark this cut decode `stall_us` from about 16.8 ms/token to 0.41 ms/token, but TPOT did not improve because SSD/callback time rose in that run.
- **Perf-E**: `LLAMA_MOE_SLOT_MMVQ=1` restores only the guarded quantized single-token `.slot` MMVQ decode path. On the 8000 MiB EAMC benchmark this improved TTFT from 19704.8 ms to 18481.0 ms and TPOT from 62.13 ms/token to 47.20 ms/token versus Phase D; H2D, stall, predictor time, SSD read time, callback wall time, and hit rates were flat or better.
- **Perf-F**: `LLAMA_MOE_SLOT_GRAPHS=1` enables CUDA graph capture only for the guarded `.slot` MMVQ decode path. On the 8000 MiB EAMC benchmark this improved TPOT from 47.20 ms/token to 30.63 ms/token versus Phase E and reduced decode `compute_us` from 24.34 ms/token to 10.46 ms/token; H2D, stall, predictor time, SSD read time, callback wall time, and hit rates were flat or better.
- **Perf-G**: `LLAMA_MOE_SLOT_GLU_FUSION=1` enables the guarded decode-only `.slot` quantized `MUL_MAT_ID + GLU` fusion path on top of `LLAMA_MOE_SLOT_MMVQ=1`. On the 8000 MiB EAMC benchmark this improved TPOT from 31.56 ms/token to 30.67 ms/token versus the Phase F guard stack in the same build and reduced callback wall time from 17.78 ms/token to 17.06 ms/token; `gpu_compute` was effectively flat at 10.44 to 10.37 ms/token and H2D, hit rate, misses/token, and top-k D2H stayed flat.
- **Perf-H**: Phase H fixed callback reads of strided top-k views by using `ggml_backend_tensor_get_2d()` when the top-k tensor is a non-contiguous view. This is on the default path and explains the large prefill jump: versus the Phase G GLU run, Phase H baseline prefill improved from 14021.9 ms to 5070.4 ms, average experts observed per prefill callback fell from 64.0 to about 20.7, prefill misses fell from 44581 to 11610, and prefill callback wall time fell from 23467 ms to 6498 ms. Separately, `LLAMA_MOE_TOPK_FUSION_DIAG=1` enables an opt-in diagnostic single-token decode top-k MoE fusion path. Phase H fixed the earlier top-k golden-logit failure by moving the streaming callback point from `ffn_moe_topk` to final routing weights while still reading IDs from the original top-k tensor. Synthetic routing, golden logits, and chat smoke passed, including the full MMVQ + graphs + GLU stack, but the 8000 MiB same-build EAMC top-k-fusion benchmark was flat: TPOT 31.91 to 31.88 ms/token and `topk_d2h_us` 1.58 to 1.58 ms/token. Keep top-k fusion default-off.
- **Perf-I**: `LLAMA_MOE_PREFILL_MMVQ=1`, used together with `LLAMA_MOE_SLOT_MMVQ=1`, enables the guarded multi-token `.slot` MMVQ prefill path. Synthetic CUDA coverage now checks the multi-token path with changing slot IDs and changed slot tensor contents, and the guarded stack passed raw golden logits (`max|d|=0`) plus `llama-cli --jinja --reasoning off` chat smoke at the default ubatch and forced `LLAMA_MOE_STREAMING_UBATCH=8`. In the 8000 MiB EAMC same-build benchmark it improved TTFT from 6505.2 ms to 6202.5 ms, but decode TPOT regressed from 36.72 ms/token to 40.85 ms/token, so the guard remains experimental and default-off.
- **Perf-J**: `llama-moe-bench` now reports cold and warm TTFT separately, emits prefill I/O/profiler breakdowns alongside decode, and supports `--moe-reset-cache-between-repeats`, `--moe-warm-cache`, and benchmark-only `--moe-hot-start`. On the 16 GiB dev box, the cache matrix showed 8000 MiB reaches effective ubatch 8, 12000 MiB reaches effective ubatch 16 with a clear TTFT/TPOT improvement, 14000 MiB is slightly faster but nearly fills VRAM, and 16000 MiB over-pressures VRAM and badly regresses decode. Hot-start is mechanically wired but remains experimental/default-off because the smoke run worsened TTFT.
- **Perf-K closeout**: Static CUDA validation passed `ctest --test-dir build-moe-static -C Release -L moe-offload --output-on-failure` with 8/8 tests passing. The accepted guard stack (`LLAMA_MOE_SLOT_MMVQ=1`, `LLAMA_MOE_SLOT_GRAPHS=1`, `LLAMA_MOE_SLOT_GLU_FUSION=1`, `LLAMA_MOE_PREFILL_MMVQ=0`, `LLAMA_MOE_TOPK_FUSION_DIAG=0`) passed the golden-logit gate at cache 4000 MiB, `-ub 8`, `max|d|=0`, and the full cache/ubatch matrix for 4000/8000/12000 MiB x 8/16/32/64 also passed with `max|d|=0`. `llama-cli --jinja --reasoning off` chat smoke passed at the default ubatch and forced `LLAMA_MOE_STREAMING_UBATCH=8`. The final 12000 MiB EAMC benchmark measured TTFT cold 4398.2 ms, TPOT 26.78 ms/token, prefill `gpu_compute` 7.49 ms/token, decode `gpu_compute` 11.16 ms/token, and peak VRAM 14.08 / 15.92 GB.
- **J**: `--logit-dump PATH` on `llama-completion`, `tests/moe-offload/compare_logits.py`, and PowerShell harness `tests/moe-offload/test-golden-logits.ps1`. See "Correctness" below.
- **K**: C++ unit tests registered with CTest under label `moe-offload` (`test-eamc-cosine`, `test-lru-eviction`, `test-manifest-roundtrip`, `test-repack-slices`) alongside Phase G's `test-cuda-stream` smoke, the Phase E/F `test-slot-mmvq` CUDA micro-test, and the `test-moe-oracle-failfast` CLI test. Run with `ctest --test-dir build-moe -C Release -L moe-offload`.
- **MVP closeout**: Streaming slot tensors now default to the actual cache slot count; set `LLAMA_MOE_FULL_EXPERT_AXIS=1` only as a guarded fallback. By default CUDA `MUL_MAT_ID` on `.slot` tensors uses the generic sorted path for correctness-first validation. The guarded decode-only MMVQ path is available through `LLAMA_MOE_SLOT_MMVQ=1`, with decode-only CUDA graphs additionally gated by `LLAMA_MOE_SLOT_GRAPHS=1`, decode-only GLU fusion gated by `LLAMA_MOE_SLOT_GLU_FUSION=1`, guarded multi-token MMVQ prefill gated by `LLAMA_MOE_PREFILL_MMVQ=1`, and diagnostic-only decode top-k fusion gated by `LLAMA_MOE_TOPK_FUSION_DIAG=1`; MMQ/MMF, prefill graphs/fusion, generic sorted graph capture, and normal top-k MoE fusion remain disabled for `.slot`/offload.
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

Observed in the Phase K closeout on the dev box (2026-06-12,
Qwen3.5-35B-A3B-Q4_K_M, prompt `"Hello"`, `--temp 0 --seed 42 -n 8`,
accepted Perf guard stack, 4000 MiB streaming cache, `-ub 8`):

| Metric | Value |
|---|---|
| `n_steps` | 8 |
| `n_vocab` | 248320 |
| `max|Δlogit|` | **0** |
| `mean|Δlogit|` | 0 |
| `tol` | 1e-3 |
| Result | **PASS** |

The Phase K ubatch matrix also passed 12/12 cases with `max|Δlogit| = 0`:
streaming cache 4000, 8000, and 12000 MiB crossed with `-ub` 8, 16, 32, and
64. The matrix artifact is
`tests/moe-offload/_out/phase-k-ubatch-matrix/summary.csv`.

The June 5 closeout changes addressed the two leading runtime suspects from
this section: slot tensors now use the actual cache slot count by default, and
CUDA `MUL_MAT_ID` on `.slot` tensors uses the generic sorted path by default.
The Phase F guarded MMVQ + graph decode path passed the golden-logit harness on
2026-06-12 with `LLAMA_MOE_SLOT_MMVQ=1`, `LLAMA_MOE_SLOT_GRAPHS=1`,
cache=8000 MiB, `-n 32`, `-ub 8`, and `max|d|=0`. Phase G then passed the
same gate with `LLAMA_MOE_SLOT_GLU_FUSION=1` added, cache=4000 MiB,
`-n 8`, `-ub 8`, and `max|d|=0`. Phase H fixed the diagnostic top-k fusion
callback boundary and passed golden logits with top-k alone, top-k + GLU, and
top-k + GLU + graphs, all with `max|d|=0`. Normal
`LLAMA_MOE_TOPK_FUSION=1` remains ignored in `LLAMA_MOE_OFFLOAD` builds;
use `LLAMA_MOE_TOPK_FUSION_DIAG=1` only for focused diagnostics.

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
- Historical Phase J drift (`max|Δlogit|=4.64e-01`) is closed by the later
  runtime/CUDA correctness fixes and Phase K revalidation.

## Test surface

CTest targets under label `moe-offload`:

| Test | Phase | Purpose |
|---|---|---|
| `test-cuda-stream` | G | Pinned alloc + async H2D + event ordering smoke. |
| `test-slot-mmvq` | Perf-E/F/G/I | Synthetic CUDA `.slot` Q4_0 `MUL_MAT_ID` check for generic sorted decode, guarded MMVQ decode, guarded MMVQ graph replay with changing slot ids/contents, guarded prefill MMVQ with changing slot ids/contents, guarded decode-only GLU fusion replay, and guarded prefill graph/fusion fallback. Registered only when the CUDA backend target is available. |
| `test-topk-moe-fusion` | Perf-H | Synthetic CUDA Qwen-style router check for unfused vs diagnostic fused `softmax -> argsort_top_k -> get_rows -> norm -> scale`, including single-token decode and multi-token fallback. Registered only when the CUDA backend target is available. |
| `test-eamc-cosine` | K / Perf-B / Perf-C | EAMC dense-equivalent scoring, sidecar round-trip, prefill-like repeated layer waves, and bounded FIFO/ring replacement on full insert. |
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
uses callback-filled slot-id tensors and generic `.slot` `MUL_MAT_ID` by
default. The guarded `LLAMA_MOE_SLOT_MMVQ=1` path restores only quantized
single-token MMVQ decode unless `LLAMA_MOE_PREFILL_MMVQ=1` is also set for the
guarded multi-token prefill path.
Large-ubatch failures should now surface as `n_uniq exceeds n_slots` or an I/O
progress error; a CUDA illegal access is a regression and should be captured
with the cache size, effective ubatch, and prompt.

### Windows Application Control blocks rebuilt CUDA binaries

On the dev box used for Perf-D, a freshly rebuilt compressed CUDA backend failed
to load with `0xc0e90002` / "Application Control policy has blocked this file".
Reconfiguring with uncompressed CUDA fatbins produced a loadable backend:

```powershell
cmake -S . -B build-moe -DGGML_CUDA_COMPRESSION_MODE=none
cmake --build build-moe --config Release --target llama-moe-bench
```

This is a local Windows policy issue, not a MoE runtime setting. It can also
block freshly rebuilt unsigned test executables such as `test-cuda-stream.exe`.
