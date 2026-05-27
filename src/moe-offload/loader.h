#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct llama_model_loader;

namespace llama_moe {

struct expert_record {
    uint64_t file_offset = 0;
    uint64_t size = 0;
    uint16_t layer = 0;
    uint16_t expert = 0;
};

struct manifest {
    bool present = false;
    uint32_t version = 0;
    uint32_t n_layers = 0;
    uint32_t n_experts_per_layer = 0;
    uint64_t expert_blob_size_max = 0;
    std::string layout;
    std::vector<expert_record> experts;
};

manifest inspect_manifest(const gguf_context * ctx);

bool configure_from_params(
        const llama_model_params & params,
        const std::string & model_path,
        const llama_model_loader & ml);

} // namespace llama_moe