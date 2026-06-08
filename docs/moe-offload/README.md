# MoE Offload MVP

This fork adds a guarded MoE offload overlay behind `LLAMA_MOE_OFFLOAD`. The default build path remains off.

## Build

```bash
cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON
cmake --build build-moe --config Release --target llama-moe-bench llama-completion llama-moe-repack
```

## Repack

```powershell
.\build-moe\bin\Release\llama-moe-repack.exe `
  --input C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.gguf `
  --output C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --manifest C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf.json
```

The current repacker writes a stock-loadable GGUF with page-aligned tensor data and `moe_offload.*` metadata. It records the layout as `fused-tensors-page-aligned-v1`: the original fused expert tensors remain in the file, and `moe_offload.expert_blob.table` records each `(logical_layer, expert, kind)` byte range relative to the GGUF data section.

## Run

```powershell
.\build-moe\bin\Release\llama-completion.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 8000 `
  --moe-predictor lru `
  -p "Hello" -n 32
```

`--moe-predictor` accepts `lru` or `eamc`. `--moe-eamc-path PATH` selects the EAMC predictor sidecar; when omitted with `--moe-predictor eamc`, the runtime uses the model path with a `.eamc` suffix. `--moe-profile-csv PATH` and `--moe-profile-summary PATH` write profiling data.

## Bench

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 1024 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 --moe-predictor eamc `
  --moe-eamc-path C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.eamc `
  --moe-profile-csv moe-profile.csv `
  --moe-profile-summary moe-summary.txt
```

The bench tool enables MoE offload automatically, runs prefill + decode loops,
and always prints the summary report to stdout. If `--moe-profile-summary` is
set, `llama-moe-bench` writes the same §4.7 report to that path after all
repeats complete. During long runs, it also checkpoints the same summary file
after prefill and after each completed repeat, so the file appears before the
full benchmark finishes. It intentionally does not use the runtime
`end_request()` summary writer, which emits the older aggregate-only format. If
`--moe-profile-csv` is set, a per-layer profile CSV is written to that path.

Example summary:
```
model: Qwen3.5-35B-A3B Q4_K_M
predictor: eamc      cache: 8000 MB   ssd: C:
n_prompt: 1024  n_gen: 256  repeats: 3

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
| `cache_resident_experts` | Number of experts resident for the layer after miss handling. |
| `predictor` | `lru` or `eamc`. |

## Current State

Implemented in this slice:

- `LLAMA_MOE_OFFLOAD` build option.
- Guarded CLI/model parameters: `--moe-offload`, cache budget knobs, predictor choice, EAMC sidecar path, profile paths, and oracle fail-fast.
- Loader metadata validation for `moe_offload.version`.
- LRU and EAMC predictor implementations, including binary EAMC sidecar save/load.
- Slot-cache model and profiling CSV/summary plumbing, including `pred_us` predictor overhead.
- Guarded hooks at model load, decode request boundaries, and MoE top-k graph construction.
- `llama-moe-repack` and `llama-moe-bench` targets.
- **D-4**: Pre-allocated slot_table tensors via `create_unfiled_tensor`; zero-init of all slot tensors; ubatch auto-cap to 8 in streaming mode.
- **D-5**: Threaded I/O worker (`src/moe-offload/io.{h,cpp}`).
- **E**: Per-layer CSV profiling rows; snapshot summary; LRU/EAMC predictor wired into eval-callback eviction.
- **F**: `llama-moe-bench` emits the §4.7-style stdout/file summary and writes nonempty CSV rows when requested.
- **G**: CUDA stream accessor (`moe_io_cuda_*` in `ggml/src/ggml-cuda/moe_offload_io.cu`), pinned-host staging, dedicated MoE H2D stream, event pool.
- **H**: Eval-callback rewired to async H2D via `io_h2d_async_timed` + `io_compute_wait` (`cudaStreamWaitEvent` on the compute stream). Synchronous double-read of the miss blob removed.
- **I**: Real `compute_us` via CUDA events. Profile rows are buffered per `llama_decode` batch and flushed at `slot_pool_end_request()` after `cudaEventElapsedTime` queries. Per-miss `h2d_us` now comes from event timing on the async path.
- **J**: `--logit-dump PATH` on `llama-completion`, `tests/moe-offload/compare_logits.py`, and PowerShell harness `tests/moe-offload/test-golden-logits.ps1`. See "Correctness" below.
- **K**: C++ unit tests registered with CTest under label `moe-offload` (`test-eamc-cosine`, `test-lru-eviction`, `test-manifest-roundtrip`, `test-repack-slices`) alongside Phase G's `test-cuda-stream` smoke and the `test-moe-oracle-failfast` CLI test. Run with `ctest --test-dir build-moe -C Release -L moe-offload`.
- **MVP closeout**: Streaming slot tensors now default to the actual cache slot count; set `LLAMA_MOE_FULL_EXPERT_AXIS=1` only as a guarded fallback. CUDA `MUL_MAT_ID` on `.slot` tensors bypasses the specialized MMVQ/MMQ/MMF paths and uses the generic sorted path for correctness-first validation.
- **L**: IO worker shutdown wired into `llama_context::~llama_context()`; small-cache (≤8000 MiB) and multi-token-prompt streaming deadlocks fixed by enlarging the IO buffer pool to cover worst-case in-flight misses and by excluding just-reserved experts from LRU/predictor eviction within the same callback.

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
| `test-eamc-cosine` | K | EAMC cosine ordering, sidecar round-trip, and redundancy replacement on full insert. |
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

See [`tests/moe-offload/README.md`](../../tests/moe-offload/README.md)
for runtime prereqs and invocation details.

## Troubleshooting

### "n_uniq exceeds n_slots" error
```
moe_eval_callback: n_uniq=96 exceeds n_slots=48
```
Increase `--moe-cache-vram-mb` so `n_slots >= worst-case unique experts per batch`,
or reduce batch size with `-ub`.

### Streaming mode caps ubatch to 8
When `--moe-cache-vram-mb` is small enough that `n_slots < n_expert` (streaming mode),
the ubatch is automatically capped to **8** to avoid a known `ggml_get_rows`→`mm_ids_helper`
kernel crash. This reduces prefill throughput (~2-5× slowdown). To restore full throughput,
use full residency: `--moe-cache-vram-mb 99999`.

### CUDA illegal memory access at launch_mul_mat_q
This is the ubatch-capping issue above. Ensure you are using the latest build
which auto-caps ubatch to 8 in streaming mode. If you need larger ubatch,
use full residency or investigate `ggml/src/ggml-cuda/mmq.cu` + `mmid.cu`.
