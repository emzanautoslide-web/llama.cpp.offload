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

#### Phase M diagnostics (May 29)

`LLAMA_MOE_DEBUG_SLOT_HASH` confirmed cross-step GPU buffer corruption:
slot weight tensor data changed between consecutive eval-callback visits
without a matching eviction.  Slots 1-3 at layer 1 were read back as
all-zeros (hash `0x51d88627df287325`) at token 1 callback START, even
though they held valid expert data at token 0 callback END and there
were zero evictions (`n_slots=145`, `uniq=8` per layer).

Layer 0's slots 1-3 survived intact.  This suggests a scheduler
temporary or mmq kernel write aliases L1's slot-tensor GPU memory during
L0's MoE compute, but the exact mechanism within `ggml_backend_sched` /
`ggml_gallocr` has not been isolated.

#### Phase M.7 fix: fingerprint-based integrity + auto-reload

`src/moe-offload/slot_pool.cpp` now maintains a per-layer, per-cached-expert
FNV1a-64 fingerprint of the first 1024 bytes of each expert's gate-kind
slot data (computed from the pinned I/O buffer at load time).

At the start of each eval-callback, before counting HIT experts, the code
reads back the first 1 KiB of each candidate HIT expert's slot from the
GPU, hashes it, and compares against the stored fingerprint.  If the
hash mismatches, the expert is evicted from the cache (marking its slot
free and removing all bookkeeping entries), and the miss-loop below
reloads it from SSD.

Fingerprints are cleared on normal eviction and recomputed on every
successful load.

#### Phase M.8 results (May 29)

Streaming run (`--moe-cache-vram-mb 12000`, `-n 8`) with
`LLAMA_MOE_DEBUG_EVICT=1 LLAMA_MOE_DEBUG_SLOT_HASH=1`:

- **10 fp-corrupt events** detected across 8 tokens × 40 layers = 320
  callbacks. The fingerprint verification is working and catches real
  GPU buffer corruption.
- **Repeated corruption on same experts**: L12 expert=0 corrupted at
  tokens 2,3,4,5; L0 expert=238 corrupted at tokens 2,3. After
  reloading, the expert's data is corrupted AGAIN during the current
  forward pass' graph execution — the `mul_mat_id` consumer reads
  corrupted data before the next callback can detect and fix it.
- **Generated output is `Hello,///////`** — identical garbage to the
  full-residency run. The fingerprint fix detects corruption but cannot
  prevent it from affecting the current forward pass.

**Conclusion**: The corruption happens WITHIN a single forward pass
(during graph execution, between callback return and `mul_mat_id`
consumption). The fingerprint + auto-reload approach can only fix data
for the NEXT token, not the current one. A full fix requires preventing
the corruption at the source, which is suspected to be either:
1. Cross-layer MMQ kernel buffer overflow (L11 corrupts L12)
2. Scheduler temporary aliasing within the same GPU buffer

#### Mitigation status

- ✅ Corruption **detected** (fingerprint verification catches it)
- ✅ Corrupted data **reloaded** (eviction + SSD reload on next callback)
- ❌ Current token **not protected** (corruption happens after reload,
  during same forward pass)
- Full fix requires dedicated GPU buffers per layer or MMQ kernel
  analysis — **deferred to post-MVP**.

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
