# MoE Offload Known Issues

Status as of the MVP closeout work on 2026-06-05.

## Open

### Streaming Golden-Logit Validation

Historical Phase J measurement on `Qwen3.5-35B-A3B-Q4_K_M.moe.gguf` reported
`max|Delta logit| = 4.64e-01` over 8 decode steps at `--temp 0 --seed 42`.
Phase M diagnostics showed slot data corruption between callback return and
the same forward pass' `mul_mat_id` consumption.

The June 5 implementation addresses the two concrete runtime suspects:

- Slot tensors now default to the actual cache slot count instead of the full
  original expert count. `LLAMA_MOE_FULL_EXPERT_AXIS=1` keeps the previous
  full-axis allocation only as a guarded fallback.
- CUDA `MUL_MAT_ID` for `.slot` tensors bypasses the specialized MMVQ/MMQ/MMF
  paths and uses the generic sorted path for correctness-first validation.

The dev-box golden-logit harness still needs to be rerun against the new build:

```powershell
powershell -ExecutionPolicy Bypass -File tests\moe-offload\test-golden-logits.ps1 `
    -Tol 1e-3 -NPredict 8 -StreamCacheMb 4000
```

### Streaming `n_ubatch` Cap

In streaming mode (`n_slots < n_experts`), `llama_context` still caps
`n_ubatch` at 8. The cap is acceptable for MVP if the golden-logit gate passes
and remains documented. Removing it is post-MVP work.

## Deferred

The following remain outside MVP scope:

- `--moe-oracle`: parser plumbing may exist, but passing it fails fast with a
  clear post-MVP error.
- Direct I/O, multi-GPU, CPU DRAM expert tier, KV offload, learned predictors,
  speculative prefetch/decoding, FineMoE-style splitting, and removing the
  streaming ubatch cap.

## Closed

- EAMC predictor sidecar persistence: `--moe-eamc-path PATH` is wired through
  common args, `llama-bench`, and `llama-moe-bench`; EAMC now saves/loads a
  binary sidecar with shape/version checks and ignores incompatible sidecars
  with a warning. `test-eamc-cosine` covers sidecar round-trip.
- Repacker layout naming mismatch: the implemented layout is documented as
  `fused-tensors-page-aligned-v1`, with fused expert tensors retained and
  `moe_offload.expert_blob.table` providing per-expert byte ranges.
- Manifest-only validation gap: `test-repack-slices` compares every table entry
  against the exact byte slice in the original GGUF when dev-box model paths
  are provided.
- Oracle ambiguity: `test-moe-oracle-failfast` verifies `--moe-oracle` exits as
  post-MVP instead of silently enabling an unsupported path.
- IO worker shutdown, pinned-buffer starvation, multi-token prompt hangs, and
  same-callback eviction of just-reserved experts were closed in the earlier
  Phase L/M implementation.
