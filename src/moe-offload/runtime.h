#pragma once

#include "loader.h"
#include "profiler.h"

#include <cstdint>
#include <string>

struct ggml_context;
struct ggml_tensor;

namespace llama_moe {

struct runtime_options {
    bool enabled = false;
    std::string model_path;
    uint64_t cache_vram_mb = 0;
    float cache_vram_frac = 0.0f;
    std::string predictor = "lru";
    std::string eamc_path;
    std::string profile_csv;
    std::string profile_summary;
    bool oracle = false;
};

void configure_runtime(const runtime_options & options, const manifest & mf);
bool runtime_enabled();

const runtime_options & get_options();
const manifest &        get_manifest();

void begin_request();
void end_request();
LLAMA_API bool flush_predictor();

LLAMA_API void set_profile_request_context(int repeat_idx, int batch_idx, const char * phase);
uint64_t current_request_idx();
profile_request_row current_profile_request_row();
void add_current_request_timing(const char * observed_phase, int64_t predictor_end_us, int64_t predictor_save_us, int64_t profile_flush_us, uint64_t sidecar_write_bytes);

// Phase E: access the global profiler for recording per-layer rows from
// the eval-callback. Returns nullptr if profiling is not enabled.
profiler * get_profiler();

LLAMA_API profile_snapshot get_profile_snapshot();
LLAMA_API void reset_profile();

ggml_tensor * remap_selected_experts(
        ggml_context * ctx,
        ggml_tensor * selected_experts,
        int layer,
        int64_t n_expert,
        int64_t n_expert_used);

} // namespace llama_moe
