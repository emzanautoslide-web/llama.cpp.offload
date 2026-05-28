# MoE Offload MVP

This fork adds a guarded MoE offload overlay behind `LLAMA_MOE_OFFLOAD`. The default build path remains off.

## Build

```bash
cmake -B build-moe -DLLAMA_MOE_OFFLOAD=ON -DGGML_CUDA=ON
cmake --build build-moe --config Release
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

```bash
build-moe/bin/llama-cli \
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf \
  --moe-offload \
  --moe-cache-vram-mb 8000 \
  --moe-predictor lru \
  --moe-profile-csv moe-profile.csv \
  -p "The quick brown fox" -n 32
```

`--moe-predictor` accepts `lru` or `eamc`. `--moe-profile-summary PATH` writes the current summary output at request end.

## Bench

```bash
build-moe/bin/llama-moe-bench \
  --model C:/AI/models/qwen/Qwen3.5-35B-A3B-Q4_K_M.moe.gguf \
  --pp 1024 --tg 256 --repeat 3 \
  --moe-offload --moe-cache-vram-mb 8000 --moe-predictor eamc
```

The wrapper forwards to `llama-bench` and translates `--pp`, `--tg`, and `--repeat` to the upstream bench flags.

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
- **D-5**: Threaded async I/O worker (`src/moe-offload/io.{h,cpp}`) with SPSC ring queue and staging buffer pool.
- **E**: Per-layer CSV profiling rows; end-request summary; LRU/EAMC predictor wired into eval-callback eviction.

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