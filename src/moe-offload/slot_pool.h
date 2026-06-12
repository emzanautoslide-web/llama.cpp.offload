#pragma once

#include "loader.h"
#include "llama-model-loader.h"

#include <cstdint>
#include <initializer_list>
#include <string>

struct ggml_tensor;
struct llama_hparams;
struct LLM_TN_IMPL;

namespace llama_moe {

// Called from configure_from_params once the manifest is known. Computes the
// uniform slot count per MoE layer using the configured cache budget (Phase B
// defaults to all-resident: n_slots == n_experts_per_layer).
void configure_slot_pool();

// Reset per-load bookkeeping (slot tensor registry). Tensors themselves live in
// the loader's ctx_map and outlive this reset.
void reset_slot_pool();

// Reset only streaming cache residency and counters. Slot tensors and graph
// registrations remain intact. Intended for benchmark calibration.
LLAMA_API void slot_pool_reset_cache();

// Returns the uniform slot count per MoE layer.
LLAMA_API uint32_t n_slots_per_layer();

// Returns the routed expert count recorded in the MoE manifest.
LLAMA_API uint32_t n_experts_per_layer();

// Recommended effective ubatch for streaming mode. The recommendation is
// derived from the current slot budget and model top-k. `safety` defaults to a
// worst-case unique-expert bound when set to 1.0; lower values are more
// conservative and values above 1.0 are diagnostic/optimistic.
uint32_t recommended_ubatch(uint32_t requested, uint32_t n_expert_used, float safety);

// Classifies the tensor name. Returns true if `tn` is a routed-expert weight on
// a transformer layer listed in moe_offload.layer_ids. Fills logical index and
// kind on success.
bool should_intercept(const LLM_TN_IMPL & tn, int * out_logical_layer, expert_kind * out_kind);

// Intercepts the model-side create_tensor call for a routed-expert weight.
//   - Reads the source tensor's dtype from the loader's GGUF meta.
//   - Allocates a slot tensor [d_in, d_out, n_slots] on the layer's preferred
//     backend buffer via llama_model_loader::create_unfiled_tensor.
//   - Marks the original GGUF weight as unloaded so the loader's accounting
//     stays consistent and load_all_data skips it.
//   - Registers the resulting tensor in the per-layer slot table.
ggml_tensor * intercept_expert_tensor(
        llama_model_loader & ml,
        const llama_hparams & hparams,
        const buft_list_t * buft_list_layer,
        const LLM_TN_IMPL & tn,
        const std::initializer_list<int64_t> & ne,
        int flags);

// Look up a previously registered slot tensor.
ggml_tensor * get_slot_tensor(int logical_layer, expert_kind kind);

// Phase D-4: look up the per-layer slot_table tensor (shape [1, n_expert] I32)
// pre-allocated by `intercept_expert_tensor`. Used as the source operand of
// `ggml_get_rows` inside `remap_selected_experts` and written by the
// eval-callback via `ggml_backend_tensor_set`.
ggml_tensor * get_slot_table_tensor(int logical_layer);

// Phase C: synchronously fill every slot tensor with its corresponding expert blob
// from disk (one slot == one expert). Called after the loader has allocated backing
// buffers via ggml_backend_alloc_ctx_tensors and finished load_all_data. Returns
// false on any I/O or shape mismatch. No-op when n_slots < n_experts_per_layer
// (non-resident mode handled by the eval-callback in Phase D).
bool prefetch_all_experts();

// Phase D-1: remap tensor registry. `remap_selected_experts` registers the
// top-k tensor so the eval-callback can stop immediately after it is computed.
// In streaming mode it also registers a [n_expert_used, n_tokens] I32 tensor
// that the callback fills directly with slot indices for downstream
// MUL_MAT_ID. In full-residency mode the original expert IDs are used.
void register_slot_table_for_topk(int logical_layer, ggml_tensor * topk, ggml_tensor * slot_table);
void register_slot_ids_for_topk(int logical_layer, ggml_tensor * topk, ggml_tensor * slot_ids);
void register_weights_for_topk(int logical_layer, ggml_tensor * topk, ggml_tensor * weights);
void populate_slot_tables_identity();

// Clear per-graph-build slot_table bookkeeping (topk->slot_table maps and the
// flat list of registered slot_tables). Called by llama-context.cpp immediately
// before rebuilding a compute graph so that only the new graph's tensor
// pointers are visible. The persistent slot_tensors registry (model weights)
// and the per-layer LRU cache are NOT touched.
void reset_graph_state();

// Phase D-2: streaming mode (n_slots < n_experts). When true, the loader does
// NOT prefetch all experts; instead a per-layer LRU cache fills slots on demand
// via the eval-callback below.
LLAMA_API bool streaming_mode();

// Phase D-5: initialize the async I/O subsystem. Must be called once after
// slot tensors have backing buffers. Idempotent (safe to call multiple times).
void slot_pool_init_io(const std::string & source_path);

// Phase D-5: shutdown the async I/O worker.
void slot_pool_shutdown_io();

// Phase H: tell the slot pool which CUDA backend owns the compute stream
// so the eval-callback can stall it on async H2D events via
// cudaStreamWaitEvent. Pass nullptr to disable async H2D (callback falls
// back to synchronous ggml_backend_tensor_set).
void slot_pool_set_compute_backend(ggml_backend_t backend);

// Reset per-request predictor observations before an internal decode batch.
void slot_pool_begin_request();

// End the current internal decode batch (calls predictor.end_request, but does
// not persist predictor sidecars).
void slot_pool_end_request();

// Persist predictor state at a logical request/session boundary.
bool slot_pool_flush_predictor();

// Phase D-2: scheduler eval-callback. On post-eval of `ffn_moe_topk-<il>` it
// reads selected expert IDs, ensures their blobs are resident in slots (loading
// misses synchronously from disk), and writes the slot_table tensor for that
// layer so the downstream ggml_get_rows yields slot indices instead of expert
// indices. Always returns true so compute continues. Signature matches
// `ggml_backend_sched_eval_callback`.
bool moe_eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

} // namespace llama_moe
