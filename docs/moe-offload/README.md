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

Still pending for the full SSD-backed MVP:

- Split fused expert tensors into per-expert contiguous blobs.
- Allocate physical VRAM expert slots and remap `mul_mat_id` IDs to those slots.
- Implement CUDA H2D stream/event synchronization and SSD demand fetch.
- Emit real per-layer hit/miss/stall rows from the graph execution path.