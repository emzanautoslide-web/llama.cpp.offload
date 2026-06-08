# MoE Offload Known Issues

Status as of the MVP closeout work on 2026-06-05.

## Open

### Streaming `n_ubatch` Cap

In streaming mode (`n_slots < n_experts`), `llama_context` still caps
`n_ubatch` at 8. The cap is acceptable for MVP because the forced-eviction
golden-logit gate passes with `-UBatch 8`, but removing it remains post-MVP
CUDA/runtime work.

Impact:

- Prefill throughput is lower than an uncapped run.
- Full-residency mode is the workaround when large ubatches are required.

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
- Removing the streaming ubatch cap.

## Closed

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
