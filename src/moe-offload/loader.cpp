#include "loader.h"

#include "predictor.h"
#include "runtime.h"
#include "slot_pool.h"

#include "gguf.h"
#include "llama-impl.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace llama_moe {

namespace {

std::string default_eamc_path(const std::string & model_path) {
    const size_t slash = model_path.find_last_of("/\\");
    const size_t dot = model_path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        return model_path.substr(0, dot) + ".eamc";
    }
    return model_path + ".eamc";
}

bool read_u32_kv(const gguf_context * ctx, int64_t kid, uint32_t & value) {
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(ctx, kid); return true;
        case GGUF_TYPE_UINT16: value = gguf_get_val_u16(ctx, kid); return true;
        case GGUF_TYPE_UINT8:  value = gguf_get_val_u8(ctx, kid);  return true;
        case GGUF_TYPE_INT32:  value = (uint32_t) gguf_get_val_i32(ctx, kid); return true;
        default: return false;
    }
}

bool read_u32(const gguf_context * ctx, const char * key, uint32_t & value) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        return false;
    }
    return read_u32_kv(ctx, kid, value);
}

bool has_suffix(const char * value, const char * suffix) {
    if (!value || !suffix) {
        return false;
    }
    const size_t value_len = std::strlen(value);
    const size_t suffix_len = std::strlen(suffix);
    return value_len >= suffix_len && std::strcmp(value + value_len - suffix_len, suffix) == 0;
}

bool read_suffix_u32(const gguf_context * ctx, const char * suffix, uint32_t & value) {
    const int n_kv = gguf_get_n_kv(ctx);
    for (int i = 0; i < n_kv; ++i) {
        if (has_suffix(gguf_get_key(ctx, i), suffix) && read_u32_kv(ctx, i, value)) {
            return true;
        }
    }
    return false;
}

bool read_u64(const gguf_context * ctx, const char * key, uint64_t & value) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        return false;
    }
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_UINT64: value = gguf_get_val_u64(ctx, kid); return true;
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(ctx, kid); return true;
        case GGUF_TYPE_INT64:  value = (uint64_t) gguf_get_val_i64(ctx, kid); return true;
        case GGUF_TYPE_INT32:  value = (uint64_t) gguf_get_val_i32(ctx, kid); return true;
        default: return false;
    }
}

std::string read_string(const gguf_context * ctx, const char * key) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_STRING) {
        return "";
    }
    return gguf_get_val_str(ctx, kid);
}

} // namespace

manifest inspect_manifest(const gguf_context * ctx, const std::string & source_path) {
    manifest result;
    if (!ctx) {
        return result;
    }

    if (!read_u32(ctx, "moe_offload.version", result.version)) {
        return result;
    }

    result.present = true;
    result.source_path = source_path;
    result.data_offset = (uint64_t) gguf_get_data_offset(ctx);
    read_u32(ctx, "moe_offload.n_moe_layers", result.n_layers);
    read_u32(ctx, "moe_offload.n_experts_per_layer", result.n_experts_per_layer);
    read_suffix_u32(ctx, ".expert_used_count", result.n_expert_used);
    read_u64(ctx, "moe_offload.expert_blob_size_max", result.expert_blob_size_max);
    result.layout = read_string(ctx, "moe_offload.layout");

    const int64_t layer_ids_kid = gguf_find_key(ctx, "moe_offload.layer_ids");
    if (layer_ids_kid >= 0 && gguf_get_kv_type(ctx, layer_ids_kid) == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx, layer_ids_kid) == GGUF_TYPE_UINT32) {
        const size_t n = gguf_get_arr_n(ctx, layer_ids_kid);
        const auto * data = static_cast<const uint32_t *>(gguf_get_arr_data(ctx, layer_ids_kid));
        result.layer_ids.assign(data, data + n);
    }
    if (result.layer_ids.empty() && result.n_layers > 0) {
        // legacy / v1 layout: assume contiguous transformer layers 0..n-1
        result.layer_ids.resize(result.n_layers);
        for (uint32_t i = 0; i < result.n_layers; ++i) result.layer_ids[i] = i;
    }
    if (!result.layer_ids.empty()) {
        result.n_layers = (uint32_t) result.layer_ids.size();
    }

    const int64_t table_kid = gguf_find_key(ctx, "moe_offload.expert_blob.table");
    if (table_kid >= 0 && gguf_get_kv_type(ctx, table_kid) == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx, table_kid) == GGUF_TYPE_UINT64) {
        const size_t n = gguf_get_arr_n(ctx, table_kid);
        const auto * data = static_cast<const uint64_t *>(gguf_get_arr_data(ctx, table_kid));
        const size_t expected = (size_t) result.n_layers * (size_t) result.n_experts_per_layer * (size_t) EXPERT_KIND_COUNT * 2;
        if (data && n == expected && expected > 0) {
            result.experts.resize(expected / 2);
            for (size_t i = 0; i < result.experts.size(); ++i) {
                result.experts[i].rel_offset = data[2 * i + 0];
                result.experts[i].size = data[2 * i + 1];
            }
        }
    }

    return result;
}

const char * expert_kind_name(expert_kind k) {
    switch (k) {
        case EXPERT_GATE: return "gate";
        case EXPERT_UP:   return "up";
        case EXPERT_DOWN: return "down";
        default:          return "?";
    }
}

const expert_record & manifest::at(uint32_t logical_layer, uint32_t expert, expert_kind kind) const {
    static const expert_record empty{};
    const size_t idx = ((size_t) logical_layer * n_experts_per_layer + expert) * EXPERT_KIND_COUNT + (size_t) kind;
    if (idx >= experts.size()) {
        return empty;
    }
    return experts[idx];
}

int manifest::logical_layer_of(uint32_t transformer_layer) const {
    for (size_t i = 0; i < layer_ids.size(); ++i) {
        if (layer_ids[i] == transformer_layer) return (int) i;
    }
    return -1;
}

bool configure_from_params(
        const llama_model_params & params,
        const std::string & model_path,
        const llama_model_loader & ml) {
    manifest mf = inspect_manifest(ml.metadata, model_path);

    if (params.moe_oracle) {
        LLAMA_LOG_ERROR("%s: --moe-oracle is deferred to post-MVP and is not implemented in this build\n", __func__);
        return false;
    }

    if (!params.moe_offload) {
        if (mf.present) {
            LLAMA_LOG_INFO("%s: MoE offload metadata detected; runtime flag is disabled\n", __func__);
        }
        configure_runtime({}, mf);
        reset_slot_pool();
        return true;
    }

    if (!mf.present) {
        LLAMA_LOG_ERROR("%s: --moe-offload requires a repacked GGUF with moe_offload.version metadata; run llama-moe-repack first\n", __func__);
        return false;
    }

    runtime_options opts;
    opts.enabled = true;
    opts.model_path = model_path;
    opts.cache_vram_mb = params.moe_cache_vram_mb;
    opts.cache_vram_frac = params.moe_cache_vram_frac;
    opts.predictor = params.moe_predictor ? params.moe_predictor : "lru";
    opts.eamc_path = params.moe_eamc_path ? params.moe_eamc_path : "";
    if (opts.eamc_path.empty() && opts.predictor == "eamc") {
        opts.eamc_path = default_eamc_path(model_path);
    }
    opts.profile_csv = params.moe_profile_csv ? params.moe_profile_csv : "";
    opts.profile_summary = params.moe_profile_summary ? params.moe_profile_summary : "";
    opts.oracle = params.moe_oracle;

    try {
        parse_predictor_kind(opts.predictor);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }

    configure_runtime(opts, mf);
    configure_slot_pool();
        LLAMA_LOG_INFO("%s: enabled MoE offload metadata v%u, layout=%s, predictor=%s, cache=%llu MiB\n",
            __func__, mf.version, mf.layout.empty() ? "unknown" : mf.layout.c_str(), opts.predictor.c_str(), (unsigned long long) opts.cache_vram_mb);
    return true;
}

} // namespace llama_moe
