# MoE Offload MVP

This fork adds a guarded MoE offload overlay behind `LLAMA_MOE_OFFLOAD`. The default build path remains off.

## Build

```bash
cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON
cmake --build build-moe --config Release --target llama-moe-bench
```

## Repack

```bash
build-moe/bin/llama-moe-repack \
  --input C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --output C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf \
  --manifest C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf.json
```

The current repacker writes a stock-loadable GGUF with page-aligned tensor data and `moe_offload.*` metadata. It records the layout as `fused-tensors-page-aligned-v0`; fused expert tensors are not split into per-expert contiguous blobs yet.

## Run

```powershell
.\build-moe\bin\Release\llama-completion.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --moe-offload `
  --moe-cache-vram-mb 8000 `
  --moe-predictor lru `
  -p "Hello" -n 32
```

`--moe-predictor` accepts `lru` or `eamc`. `--moe-profile-csv PATH` and `--moe-profile-summary PATH` write profiling data.

## Bench

```powershell
.\build-moe\bin\Release\llama-moe-bench.exe `
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf `
  --pp 1024 --tg 256 --repeat 3 `
  --moe-cache-vram-mb 8000 --moe-predictor eamc `
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
| `ssd_read_us` | Wall microseconds spent in file seek/read for misses. |
| `h2d_us` | Wall microseconds spent in `ggml_backend_tensor_set` for slot uploads. |
| `compute_us` | Reserved for CUDA event timing around MoE compute; currently `0` in the sync MVP path. |
| `stall_us` | Current sync wait cost for miss handling (`ssd_read_us + h2d_us`). |
| `cache_resident_experts` | Number of experts resident for the layer after miss handling. |
| `predictor` | `lru` or `eamc`. |

## Current State

Implemented in this slice:

- `LLAMA_MOE_OFFLOAD` build option.
- Guarded CLI/model parameters: `--moe-offload`, cache budget knobs, predictor choice, profile paths, oracle flag.
- Loader metadata validation for `moe_offload.version`.
- LRU and EAMC predictor implementations.
- Slot-cache model and profiling CSV/summary plumbing.
- Guarded hooks at model load, decode request boundaries, and MoE top-k graph construction.
- `llama-moe-repack` and `llama-moe-bench` targets.
- **D-4**: Pre-allocated slot_table tensors via `create_unfiled_tensor`; zero-init of all slot tensors; ubatch auto-cap to 8 in streaming mode.
- **D-5**: Threaded I/O worker (`src/moe-offload/io.{h,cpp}`) exists, but the active miss path uses measured synchronous `fread` + `ggml_backend_tensor_set` so slot contents are correct. Full async H2D/event overlap remains post-MVP.
- **E**: Per-layer CSV profiling rows; snapshot summary; LRU/EAMC predictor wired into eval-callback eviction.
- **F**: `llama-moe-bench` emits the §4.7-style stdout/file summary and writes nonempty CSV rows when requested.

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