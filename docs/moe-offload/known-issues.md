# MoE Offload — Known Issues

Tracking list for known issues and deferred items as of the Phase L closeout
(see [`docs/implementation_plan_mvp_20260529.md`](../../../paper/docs/implementation_plan_mvp_20260529.md)
in the sibling `paper/` workspace and the per-phase
[`implementation_plan_mvp_20260529_extend.md`](../../../paper/docs/implementation_plan_mvp_20260529_extend.md)
log).

## Open

### Streaming numerical drift (`max|Δlogit| ≥ 1e-3`)

Last measured at Phase J: `max|Δlogit| = 4.64e-01`,
`mean|Δlogit| = 3.67e-02` over 8 decode steps at `--moe-cache-vram-mb 12000`
on `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf`, `--temp 0 --seed 42 -p "Hello"`.

Hypotheses (none yet confirmed):
- Slot contents replaced between eval-callback completion and
  `mul_mat_id` consumer in `slot_pool::moe_eval_callback`.
- Late-layer callback writing a slot_table entry before the prior layer's
  `mul_mat_id` reads it.

Gate harness: `tests/moe-offload/test-golden-logits.ps1`
(+ `compare_logits.py`). Currently blocked from re-running by the
full-residency crash below.

### `prefetch_all_experts` aborts at full-residency

`GGML_ASSERT(buf != NULL && "tensor buffer not set")` fires inside
`ggml_backend_tensor_set` during the dummy-byte sync in
`prefetch_all_experts` on `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf`.
Observed on the current HEAD with and without the Phase L edits
(confirmed via `git stash` baseline). Pre-existing regression relative
to the Phase H/I/J/K smoke runs documented in the extend log.

Effect: the Phase J logit harness cannot regenerate the full-residency
reference dump until this is fixed. Streaming runs still produce output
(though the drift makes it diverge quickly).

### Streaming `n_ubatch` capped to 8

In streaming mode (`n_slots < n_experts`), `llama_context` auto-caps
`cparams.n_ubatch` at 8 (see `kMaxStreamingUbatch` in
[`src/llama-context.cpp`](../../src/llama-context.cpp)). Removing the cap
trips a `ggml_get_rows` → `mm_ids_helper` interaction in the CUDA
backend that is post-MVP work per source plan §6. Prefill throughput in
streaming mode is bounded by this cap.

### EAMC predictor sidecar persistence

The predictor surface in
[`src/moe-offload/predictor.{h,cpp}`](../../src/moe-offload/predictor.h)
does not implement load/save. The `// Phase E-3: let predictor finalize
(e.g. EAMC sidecar dump).` hook in `runtime.cpp` is a stub. Source plan
§9 item 1 sidecar round-trip remains uncovered by tests; documented in
the Phase K extend section.

## Deferred (per source plan §6)

These are explicitly out of MVP scope and will not be implemented in
this milestone:

- `--moe-oracle` mode (source plan §8).
- Windows `FILE_FLAG_NO_BUFFERING` / direct I/O — buffered `fread` is
  sufficient on the dev-box NVMe.
- Multi-GPU. The MoE H2D stream and event pool are single-device.
- Pinned-buffer pool tuning beyond the Phase L floor, multi-thread I/O
  worker, NUMA pinning.
- CPU DRAM tier, KV-cache offload, FineMoE-style sub-expert splits,
  learned predictors, speculative prefetch — all post-MVP.

## Closed in Phase L

- IO worker thread not joined at process exit
  (`STATUS_STACK_BUFFER_OVERRUN` / `-1073740791`). Wired
  `slot_pool_shutdown_io()` into `llama_context::~llama_context()`.
- Small-cache (`--moe-cache-vram-mb ≤ 8000`) deadlock before generation.
  Root cause: IO pinned-buffer pool capped at 64, while a single layer's
  worst-case demand is `n_ubatch × top_k × EXPERT_KIND_COUNT` ≈ 192
  buffers. Fix: floor the pool at 192 (cap 256) so a layer's misses
  never starve.
- Multi-token-prompt (`>~2 tokens`) hang during streaming prefill — same
  root cause as the small-cache deadlock; same fix.
- LRU/predictor eviction picking a just-reserved expert as victim within
  the same callback. Fix: track reserved-this-call experts and skip them
  in the eviction candidate scan; hard-abort with a diagnostic when no
  valid victim remains.
