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

Phase M instrumentation (this turn) localized the failure mode but did
not yet isolate the root cause. Findings on `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf`,
`-p "Hello" -n 4 --temp 0 --seed 42`, `--moe-cache-vram-mb 12000`
(`n_slots = 145 < n_expert = 256`):

- The drift is **deterministic** (bit-identical across reruns,
  md5 `1e86cc87…`). Not a CUDA stream race.
- **Step 0 (prompt forward) drift = 0**. Drift only accumulates from
  step 1 onward (per-step max: `0, 3.29e-1, 3.02e-1, 4.64e-1`). So the
  dynamic slot_table mapping computed by the eval-callback is
  computationally equivalent to full-residency's identity mapping for
  the **first** forward pass.
- The cache holds **145 slots/layer with 0 evictions** across all 160
  callbacks for this prompt+generation (max `uniq` per callback = 8 ≤ 145
  free slots). Drift cannot be eviction-driven for this test.
- Negative results from gated diagnostics (all produced bit-identical
  drift, leaving md5 unchanged):
  - `LLAMA_MOE_DEBUG_SYNC` — compute_backend synchronize after the
    slot_table `ggml_backend_tensor_set`; not slot_table stream ordering.
  - `LLAMA_MOE_DEBUG_NO_ASYNC` — force `req.h2d = false`, sync H2D
    fallback for weights; not the async cudaMemcpyAsync path.
  - `LLAMA_MOE_DEBUG_NO_HIT` — clear `lc.exp2slot/lru/slot_to_expert`
    at start of every callback so every uniq expert is a fresh MISS;
    not cross-step cache state.
  - `LLAMA_MOE_DEBUG_ID_FILL` — fill non-uniq `slot_table_host`
    positions with identity instead of 0; mmq does not read non-top-k
    slot_table entries.
  - Tightening `reserved_this_call` to include all of `uniq` (hits +
    misses) — defensive but no observable effect (no evictions occur).
- Positive control: `LLAMA_MOE_DEBUG_CB_NOP` (early-return the entire
  callback) raises drift to `1.62e+01` (md5 `70456e…`). Confirms the
  callback's writes are consumed by mul_mat_id.
- At `--moe-cache-vram-mb 24000`, `streaming_mode()` returns false
  (n_slots ≥ n_expert) → identity slot_table everywhere → drift = 0.
  This is the full-residency path, not "streaming with a larger cache",
  so it does not localize the streaming bug.

Open hypotheses, ranked:
1. **Cross-step slot-tensor content corruption.** Something writes into
   the slot weight tensor's memory between step 0's callback and step
   1's `mul_mat_id`. The pre-existing comment in
   `prefetch_all_experts` (`src/moe-offload/slot_pool.cpp` ~L405-414)
   explicitly states the mmq kernel "can access unused / excess slots
   when processing large batches" — extended-access behavior in mmq is
   plausible. If mmq also *writes* to scratch in the slot tensor's
   buffer (or if a scheduler-allocated scratch tensor aliases the slot
   tensor's memory), this would corrupt cached HIT data after step 0.
2. **mmq kernel reads quantized scales from a per-expert layout
   keyed by the slot index passed in.** Reading `slot_tensor[S]` may
   resolve scales from an offset table indexed by S; if step 1 routes
   the same expert to a different `S` than the contiguous step-0 layout,
   the kernel may compute slightly different rounding. Step 0 = 0 drift
   would then be a coincidence of cold-cache slot order. (Less likely.)
3. **Graph topology divergence at step 1+.** Some downstream node
   reads slot data differently across forward passes. Unlikely because
   full residency uses the same graph wiring.

Next concrete diagnostic (not yet implemented): at end of each callback,
read back via `ggml_backend_tensor_get` a content hash of the slot
tensor's first 1 KiB at slot 0/1/2/3, and the host
`slot_table_host[:32]`. Run for 4 tokens; compare per-(token,layer)
records to see whether slot 0 content changes between callbacks
without being written by `tensor_set`/`io_h2d`. If yes → hypothesis 1
confirmed.

Gate harness: `tests/moe-offload/test-golden-logits.ps1`
(+ `compare_logits.py`). Now usable after the prefetch crash fix below.

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

## Closed in Phase M

- **`prefetch_all_experts` aborts at full-residency.** Root cause: when a
  layer's MoE block is not part of the built compute graph, the
  scheduler never binds a backend buffer to its slot tensors, leaving
  `slot->buffer == nullptr`. The per-expert load loop blindly called
  `ggml_backend_tensor_set` which asserts on the null buffer.
  Fix: skip with `if (!slot->buffer) continue;`, mirroring the existing
  guard in `populate_slot_tables_identity`
  (`src/moe-offload/slot_pool.cpp` ~L470). Verified by full-residency
  golden-logits run completing cleanly and producing
  `Hello, I am a` as the top-1 generation.

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
