# MoE Offload MVP

This fork adds a guarded MoE offload overlay behind `LLAMA_MOE_OFFLOAD`. The
default build path remains off.

## Build

```bash
cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON
cmake --build build-moe --config Release --target llama-completion llama-moe-bench
```

The non-MoE build gate is kept separate. On 2026-06-05,
`build-vanilla-check` with `LLAMA_MOE_OFFLOAD=OFF` and `GGML_CUDA=OFF` built
`llama-completion` successfully.

## Repack

```bash
build-moe/bin/llama-moe-repack \
  --input C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --output C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf \
  --manifest C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf.json
```

The repacker writes a stock-loadable GGUF with page-aligned tensor data and
`moe_offload.*` metadata. The implemented layout is
`fused-tensors-page-aligned-v1`: the original fused expert tensors remain in
the file, and `moe_offload.expert_blob.table` records each
`(logical_layer, expert, kind)` byte range relative to the GGUF data section.

Closeout validation on Qwen3.5-35B-A3B Q4_K_M:

- Manifest/table size/range checks pass.
- `test-repack-slices` byte-compares every table entry against the exact slice
  in the original GGUF.
- Original GGUF vs repacked `.moe.gguf` with offload disabled matched golden
  logits exactly.

## Run

```powershell
.\build-moe\bin\Release\llama-completion.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 8000 `
  --moe-predictor lru `
  -p "Hello" -n 32
```

`--moe-predictor` accepts `lru` or `eamc`. `--moe-eamc-path PATH` selects the
EAMC predictor sidecar. When omitted with `--moe-predictor eamc`, the runtime
uses the model path with a `.eamc` suffix. `--moe-profile-csv PATH` and
`--moe-profile-summary PATH` write profiling data.

`--moe-oracle` is not part of the MVP. The parser field exists, but passing the
flag fails fast with a clear post-MVP error.

## Bench

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 1024 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 --moe-predictor eamc `
  --moe-eamc-path tests/moe-offload/_out/moe-bench-1024x256.eamc `
  --moe-profile-csv tests/moe-offload/_out/moe-bench-1024x256.csv `
  --moe-profile-summary tests/moe-offload/_out/moe-bench-1024x256.summary.txt `
  -ngl 99 -c 4096
```

The bench tool enables MoE offload automatically, runs prefill and decode loops,
prints the summary to stdout, and writes CSV/summary files when requested.

Observed dev-box closeout result on 2026-06-05:

- Exit code: 0.
- Repeats: 3.
- TTFT: 68463.7 ms.
- TPOT: 115.54 ms.
- Total: 98041.7 ms.
- Prefill hit rate: 77.8%.
- Decode hit rate: 90.2%.
- Decode SSD reads: 42.63 GB, average 56.85 MB/token.
- Predictor overhead: 40.63 ms/token.
- VRAM peak: 10.15 GB / 15.92 GB.
- DRAM peak: 1.56 GB.
- Profile rows: prefill=15360, decode=30720.

There is no MVP performance floor. This number is a stability and measurement
gate, not a throughput target.

## Profile CSV

| Column | Meaning |
| --- | --- |
| `token_idx` | First token index covered by this callback row. Prefill rows cover a ubatch; decode rows cover one token. |
| `phase` | `prefill` or `decode`. |
| `layer` | Logical MoE layer index. |
| `k_required` | Distinct experts required by the router for this layer/callback. |
| `k_hit` | Required experts already resident in the slot cache. |
| `k_miss` | Required experts loaded from the repacked GGUF. |
| `ssd_read_us` | Worker-thread wall microseconds spent reading missed expert blobs from the GGUF. |
| `h2d_us` | Host-to-device copy time. On the CUDA async path this is event-based timing on the dedicated MoE H2D stream; on the sync fallback it is host-measured around `ggml_backend_tensor_set`. |
| `compute_us` | CUDA-event-timed compute interval for the layer. The interval still over-approximates pure MoE compute because it brackets from the previous slot update to the next MoE top-k callback. |
| `stall_us` | Host wall time spent waiting for miss completion and inserting compute-stream waits. This is an approximation of overlap loss, not a pure CUDA event duration. |
| `pred_us` | Host microseconds spent in predictor `observe()` plus eviction `score()` calls for this row. |
| `cache_resident_experts` | Number of experts resident for the layer after miss handling. |
| `predictor` | `lru` or `eamc`. |

## Current State

Implemented in the MVP slice:

- `LLAMA_MOE_OFFLOAD` build option.
- Guarded CLI/model parameters: `--moe-offload`, cache budget knobs, predictor
  choice, EAMC sidecar path, profile paths, and oracle fail-fast.
- Loader metadata validation for `moe_offload.version`.
- `llama-moe-repack` and `llama-moe-bench` targets.
- Repacker byte-slice verifier for original-vs-repacked expert ranges.
- LRU and EAMC predictor implementations, including binary EAMC sidecar
  save/load with shape/version checks.
- EAMC nearest-neighbor score caching so repeated eviction scoring in the same
  callback does not recompute the full ranking for each candidate.
- Slot-cache model and profiling CSV/summary plumbing, including `pred_us`.
- Guarded hooks at model load, decode request boundaries, and MoE top-k graph
  construction.
- Async SSD-to-VRAM expert loading with pinned staging buffers, a dedicated
  MoE H2D stream, event pool, and compute-stream waits for misses.
- Event-based `h2d_us` and `compute_us`.
- Streaming slot tensors default to the actual cache slot count.
- Streaming remap uses callback-filled `moe.slot_ids.<layer>` tensors consumed
  directly by `MUL_MAT_ID`, avoiding same-forward-pass slot-table corruption.
- CUDA `.slot` `MUL_MAT_ID` bypasses the specialized MMVQ/MMQ/MMF paths and
  uses the generic sorted path for correctness-first validation.
- CUDA top-k MoE fusion is disabled in `LLAMA_MOE_OFFLOAD` builds because it was
  isolated as the source of the full-vs-streaming golden-logit mismatch.
- Fingerprint diagnostics are debug-only via `LLAMA_MOE_DEBUG_SLOT_FP`.
- Streaming `n_ubatch` is still capped to 8.

## Correctness

Golden-logit validation is done with `llama-completion --logit-dump` and
`tests/moe-offload/compare_logits.py`. The PowerShell harness runs a
full-residency reference and a forced-eviction streaming run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
  -Model "C:\AI\models\qwen\Qwen3.5-35B-A3B-Q4_K_M.moe.gguf" `
  -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000 `
  -Prompt "Hello" -Seed 42 -Context 4096 -UBatch 8
```

Observed dev-box closeout result on 2026-06-05:

| Metric | Value |
| --- | --- |
| `n_steps` | 8 |
| `n_vocab` | 248320 |
| `max|d|` | 0 |
| `mean|d|` | 0 |
| Result | PASS |

Additional deterministic correctness checks:

- Full-residency offload matched repacked/offload-disabled golden logits.
- Streaming offload at `--moe-cache-vram-mb 4000` matched full residency.
- LRU and EAMC produced identical logit dumps at `--temp 0 --seed 42`.

The earlier Phase J drift (`max|d| = 4.64e-01`) is closed. The root cause was
not the repacker. It was isolated to runtime CUDA graph behavior: the
correctness gate passed when CUDA fusion was disabled, and disabling top-k MoE
fusion under `LLAMA_MOE_OFFLOAD` was sufficient to keep the gate passing.

## Test Surface

CTest targets under label `moe-offload`:

| Test | Purpose |
| --- | --- |
| `test-cuda-stream` | Pinned alloc, async H2D, and event ordering smoke. |
| `test-eamc-cosine` | EAMC cosine ordering, sidecar round-trip, and redundancy replacement on full insert. |
| `test-lru-eviction` | Hand-computed LRU score values, victim ordering, and per-layer isolation. |
| `test-manifest-roundtrip` | GGUF manifest sanity: version, table size, and per-record file ranges. Self-skips when `LLAMA_MOE_TEST_GGUF` is unset. |
| `test-repack-slices` | Byte-level original-vs-repacked expert slice verification. Self-skips unless `LLAMA_MOE_TEST_ORIGINAL_GGUF` and `LLAMA_MOE_TEST_GGUF` are set. |
| `test-moe-oracle-failfast` | Verifies `--moe-oracle` exits with a post-MVP error. |

```powershell
ctest --test-dir build-moe -C Release -L moe-offload --output-on-failure
```

Closeout result: 6/6 tests passed.

Dev-box-only harnesses under `tests/moe-offload/` are not registered with CTest
because they need the multi-GB model:

- `test-golden-logits.ps1`: streaming-vs-full-residency logit gate.
- `compare_logits.py`: binary-dump comparator used by the harness.

See [`tests/moe-offload/README.md`](../../tests/moe-offload/README.md) for
runtime prerequisites and invocation details.

## Limitations

- Streaming mode caps `n_ubatch` to 8 when `n_slots < n_experts`.
- CUDA top-k MoE fusion remains disabled in `LLAMA_MOE_OFFLOAD` builds.
- CUDA `.slot` `MUL_MAT_ID` uses a correctness-first generic path instead of
  the specialized MMVQ/MMQ/MMF path.
- `stall_us` is still an approximation of overlap loss.
- Direct I/O, multi-GPU, CPU DRAM expert tier, KV offload, learned predictors,
  speculative prefetch/decoding, FineMoE-style splitting, and ubatch-cap
  removal are post-MVP.

## Troubleshooting

### `n_uniq exceeds n_slots`

```text
moe_eval_callback: n_uniq=96 exceeds n_slots=48
```

All unique experts selected in one callback must fit in the per-layer cache at
the same time. Increase `--moe-cache-vram-mb` so
`n_slots >= worst-case unique experts per batch`, or reduce batch size.

### Streaming mode caps ubatch to 8

When `--moe-cache-vram-mb` is small enough that `n_slots < n_experts`,
`n_ubatch` is automatically capped to 8. This avoids the known large-ubatch
streaming crash but reduces prefill throughput. Use full residency with a large
cache budget if you need uncapped ubatch behavior.

### `--moe-oracle` fails

This is expected. Oracle mode is deferred to post-MVP and fails fast by design.
