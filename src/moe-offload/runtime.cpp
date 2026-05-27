#include "runtime.h"

#include "profiler.h"

#include "ggml.h"
#include "llama-impl.h"

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
    (void) ctx;
    (void) layer;
    (void) n_expert;
    (void) n_expert_used;
    return selected_experts;
}

} // namespace llama_moe