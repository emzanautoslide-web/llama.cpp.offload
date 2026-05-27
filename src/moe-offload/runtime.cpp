#include "runtime.h"

#include "profiler.h"
#include "slot_pool.h"

#include "ggml.h"
#include "llama-impl.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>

namespace llama_moe {

namespace {

struct runtime_state {
    std::mutex mutex;
    runtime_options options;
    manifest mf;
    uint64_t request_idx = 0;
    std::unique_ptr<profiler> prof;
};

runtime_state & state() {
    static runtime_state instance;
    return instance;
}

} // namespace

void configure_runtime(const runtime_options & options, const manifest & mf) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.options = options;
    s.mf = mf;
    s.request_idx = 0;
    s.prof.reset();

    if (s.options.enabled && !s.options.profile_csv.empty()) {
        s.prof.reset(new profiler());
        if (!s.prof->open(s.options.profile_csv)) {
            LLAMA_LOG_WARN("%s: failed to open MoE profile CSV: %s\n", __func__, s.options.profile_csv.c_str());
            s.prof.reset();
        }
    }
}

bool runtime_enabled() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.options.enabled;
}

const runtime_options & get_options() {
    return state().options;
}

const manifest & get_manifest() {
    return state().mf;
}

void begin_request() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.options.enabled) {
        ++s.request_idx;
    }
}

void end_request() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.options.enabled || s.options.profile_summary.empty() || !s.prof) {
        return;
    }

    std::ofstream out(s.options.profile_summary, std::ios::out | std::ios::trunc);
    if (out) {
        out << s.prof->summary();
    }
}

ggml_tensor * remap_selected_experts(
        ggml_context * ctx,
        ggml_tensor * selected_experts,
        int layer,
        int64_t n_expert,
        int64_t n_expert_used) {
    if (!runtime_enabled() || n_slots_per_layer() == 0) {
        return selected_experts;
    }
    if (!selected_experts || !ctx) {
        return selected_experts;
    }

    // selected_experts: [n_expert_used, n_tokens] I32 (ne = [neu, nt, 1, 1])
    const int64_t neu = selected_experts->ne[0];
    const int64_t nt  = selected_experts->ne[1];
    if (neu != n_expert_used || nt <= 0) {
        return selected_experts;
    }

    // Per-layer slot_table tensor: shape [1, n_expert] I32.
    //
    // IMPORTANT: we do NOT mark this as `ggml_set_input`. Input tensors are
    // allocated in the scheduler's CPU input buffer and copied to the GPU copy
    // at the start of each compute split. Any mid-graph host-side write would
    // never reach the GPU. By leaving the INPUT flag unset, the scheduler
    // allocates this leaf tensor in the GPU compute buffer directly, so the
    // eval-callback's `ggml_backend_tensor_set` writes straight to GPU memory
    // that the immediately-following `ggml_get_rows` kernel reads.
    ggml_tensor * slot_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
    char name[64];
    std::snprintf(name, sizeof(name), "moe.slot_table.%d", layer);
    ggml_set_name(slot_table, name);

    // Register the (topk -> slot_table) association for THIS graph build so
    // the eval-callback can look up the right slot_table tensor by the topk
    // tensor it fires on. selected_experts here IS the topk tensor (cb name
    // "ffn_moe_topk-<il>" was already set by the caller).
    register_slot_table_for_topk(layer, selected_experts, slot_table);

    ggml_tensor * sel_flat = ggml_reshape_1d(ctx, ggml_cont(ctx, selected_experts), neu * nt);
    ggml_tensor * mapped   = ggml_get_rows(ctx, slot_table, sel_flat); // [1, neu*nt, 1, 1] I32
    return ggml_reshape_2d(ctx, mapped, neu, nt);
}

} // namespace llama_moe