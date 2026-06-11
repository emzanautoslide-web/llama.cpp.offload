#include "runtime.h"

#include "profiler.h"
#include "slot_pool.h"

#include "ggml.h"
#include "llama-impl.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <chrono>

namespace llama_moe {

namespace {

struct runtime_state {
    std::mutex mutex;
    runtime_options options;
    manifest mf;
    uint64_t request_idx = 0;
    int repeat_idx = -1;
    int batch_idx = -1;
    std::string request_phase = "unknown";
    profile_request_row active_request;
    std::chrono::steady_clock::time_point request_start;
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
    s.repeat_idx = -1;
    s.batch_idx = -1;
    s.request_phase = "unknown";
    s.active_request = {};
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
        s.active_request = {};
        s.active_request.request_idx = s.request_idx;
        s.active_request.repeat_idx = s.repeat_idx;
        s.active_request.batch_idx = s.batch_idx;
        s.active_request.phase = s.request_phase;
        s.request_start = std::chrono::steady_clock::now();
    }
}

void end_request() {
    auto request_end_start = std::chrono::steady_clock::now();

    // Phase E-3: let predictor finalize (e.g. EAMC sidecar dump).
    slot_pool_end_request();

    auto request_end_done = std::chrono::steady_clock::now();

    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.options.enabled) {
        return;
    }

    s.active_request.request_wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
            request_end_done - s.request_start).count();
    s.active_request.request_end_us = std::chrono::duration_cast<std::chrono::microseconds>(
            request_end_done - request_end_start).count();

    if (s.prof) {
        s.prof->record_request(s.active_request);
    }

    if (s.options.profile_summary.empty() || !s.prof) {
        return;
    }

    std::ofstream out(s.options.profile_summary, std::ios::out | std::ios::trunc);
    if (out) {
        out << s.prof->summary();
    }
}

void set_profile_request_context(int repeat_idx, int batch_idx, const char * phase) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.repeat_idx = repeat_idx;
    s.batch_idx = batch_idx;
    s.request_phase = phase && phase[0] ? phase : "unknown";
}

uint64_t current_request_idx() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.active_request.request_idx;
}

profile_request_row current_profile_request_row() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.active_request;
}

void add_current_request_timing(const char * observed_phase, int64_t predictor_end_us, int64_t predictor_save_us, int64_t profile_flush_us, uint64_t sidecar_write_bytes) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.active_request.phase == "unknown" && observed_phase && observed_phase[0]) {
        s.active_request.phase = observed_phase;
    }
    s.active_request.predictor_end_us += predictor_end_us;
    s.active_request.predictor_save_us += predictor_save_us;
    s.active_request.profile_flush_us += profile_flush_us;
    s.active_request.sidecar_write_bytes += sidecar_write_bytes;
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

void reset_profile() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.prof) {
        s.prof->reset(s.options.profile_csv);
    }
    s.request_idx = 0;
    s.active_request = {};
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
