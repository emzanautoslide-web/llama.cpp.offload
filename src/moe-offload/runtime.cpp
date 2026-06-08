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

    if (!streaming_mode()) {
        return selected_experts;
    }

    ggml_tensor * slot_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, neu, nt);
    ggml_format_name(slot_ids, "moe.slot_ids.%d", layer);
    ggml_set_output(slot_ids);
    register_slot_ids_for_topk(layer, selected_experts, slot_ids);
    return slot_ids;
}

} // namespace llama_moe
