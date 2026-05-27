#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct llama_model_loader;

namespace llama_moe {

enum expert_kind {
    EXPERT_GATE = 0,
    EXPERT_UP   = 1,
    EXPERT_DOWN = 2,
    EXPERT_KIND_COUNT = 3,
};

const char * expert_kind_name(expert_kind k);

struct expert_record {
    uint64_t rel_offset = 0; // relative to GGUF data section start
    uint64_t size = 0;
};

struct manifest {
    bool present = false;
    uint32_t version = 0;
    uint32_t n_layers = 0;             // logical MoE layer count (== layer_ids.size())
    uint32_t n_experts_per_layer = 0;
    uint64_t expert_blob_size_max = 0;
    uint64_t data_offset = 0;          // absolute file byte offset of tensor data section
    std::string layout;
    std::string source_path;
    std::vector<uint32_t> layer_ids;   // transformer layer index per logical MoE layer
    // experts[((logical * n_experts_per_layer + e) * EXPERT_KIND_COUNT + kind)]
    std::vector<expert_record> experts;

    // Returns the (file_offset, size) absolute record for a logical MoE layer.
    const expert_record & at(uint32_t logical_layer, uint32_t expert, expert_kind kind) const;
    int logical_layer_of(uint32_t transformer_layer) const; // -1 if not MoE
};

manifest inspect_manifest(const gguf_context * ctx, const std::string & source_path);

bool configure_from_params(
        const llama_model_params & params,
        const std::string & model_path,
        const llama_model_loader & ml);

} // namespace llama_moe