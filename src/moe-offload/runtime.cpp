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

    if (s.options.enabled) {
        s.prof.reset(new profiler());
        if (!s.options.profile_csv.empty() && !s.prof->open(s.options.profile_csv)) {
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
    // Phase E-3: let predictor finalize (e.g. EAMC sidecar dump).
    slot_pool_end_request();

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

profiler * get_profiler() {
    auto & s = state();
    return s.prof.get();
}

profile_snapshot get_profile_snapshot() {
    profile_snapshot snap;
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.prof) {
        s.prof->flush();
        snap = s.prof->snapshot();
    }
    return snap;
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

    const int64_t neu = selected_experts->ne[0];
    const int64_t nt  = selected_experts->ne[1];
    if (neu != n_expert_used || nt <= 0) {
        return selected_experts;
    }
    (void) n_expert;

    // Strategy (D-4): use the per-layer persistent slot_table tensor that was
    // pre-allocated by `intercept_expert_tensor` at model-load time. Because
    // it lives in the same backend buffer as the slot weight tensors (NOT a
    // scheduler-managed temporary), its storage is never recycled, so the
    // eval-callback's mid-graph `ggml_backend_tensor_set` is read correctly
    // by this `ggml_get_rows`.
    ggml_tensor * slot_table = get_slot_table_tensor(layer);
    static int remap_cnt = 0;
    if (remap_cnt < 4) {
        fprintf(stderr, "[moe-d4] remap_selected_experts #%d L=%d slot_table=%p has_buf=%d\n",
                remap_cnt, layer, (void*)slot_table,
                slot_table && slot_table->buffer ? 1 : 0);
    }
    ++remap_cnt;
    if (!slot_table) {
        return selected_experts;
    }

    register_slot_table_for_topk(layer, selected_experts, slot_table);

    ggml_tensor * sel_flat = ggml_reshape_1d(ctx, ggml_cont(ctx, selected_experts), neu * nt);
    ggml_tensor * mapped   = ggml_get_rows(ctx, slot_table, sel_flat);
    return ggml_reshape_2d(ctx, mapped, neu, nt);
}

} // namespace llama_moe