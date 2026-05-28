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

Tracking this drift to zero is on the Phase L / post-MVP hygiene list.

Deviations from the source plan (`implementation_plan_mvp_20260529.md` §3):

- `--moe-cache-vram-mb 4000` deadlocks on this dev box (n_slots=48 falls
  below the n_ubatch×topk=64 lower bound and the slot pool stalls before
  generation starts). The harness defaults to 12000 MiB, which still
  forces ~60% misses and remains a representative streaming workload.
- The plan called for prompt `"The quick brown fox"`. On this dev box
  any prompt longer than ~2 tokens hangs `llama-completion` during
  prefill in streaming mode (no diagnostic output beyond
  `generate: n_predict=...`). The harness defaults to `"Hello"` to keep
  the gate runnable; the same prompt-length sensitivity is reproducible
  without `--logit-dump`, so the gate itself is not the cause. Both
  deadlocks are filed against Phase L.

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