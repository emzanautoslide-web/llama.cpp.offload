#include "loader.h"

#include "predictor.h"
#include "runtime.h"

#include "gguf.h"
#include "llama-impl.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <stdexcept>

namespace llama_moe {

namespace {

bool read_u32(const gguf_context * ctx, const char * key, uint32_t & value) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        return false;
    }
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(ctx, kid); return true;
        case GGUF_TYPE_UINT16: value = gguf_get_val_u16(ctx, kid); return true;
        case GGUF_TYPE_UINT8:  value = gguf_get_val_u8(ctx, kid);  return true;
        case GGUF_TYPE_INT32:  value = (uint32_t) gguf_get_val_i32(ctx, kid); return true;
        default: return false;
    }
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

manifest inspect_manifest(const gguf_context * ctx) {
    manifest result;
    if (!ctx) {
        return result;
    }

    if (!read_u32(ctx, "moe_offload.version", result.version)) {
        return result;
    }

    result.present = true;
    read_u32(ctx, "moe_offload.n_moe_layers", result.n_layers);
    read_u32(ctx, "moe_offload.n_experts_per_layer", result.n_experts_per_layer);
    read_u64(ctx, "moe_offload.expert_blob_size_max", result.expert_blob_size_max);
    result.layout = read_string(ctx, "moe_offload.layout");

    const int64_t table_kid = gguf_find_key(ctx, "moe_offload.expert_blob.table");
    if (table_kid >= 0 && gguf_get_kv_type(ctx, table_kid) == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx, table_kid) == GGUF_TYPE_UINT64) {
        const size_t n = gguf_get_arr_n(ctx, table_kid);
        const auto * data = static_cast<const uint64_t *>(gguf_get_arr_data(ctx, table_kid));
        if (data && n % 2 == 0 && result.n_layers > 0 && result.n_experts_per_layer > 0) {
            const size_t n_records = std::min(n / 2, (size_t) result.n_layers * (size_t) result.n_experts_per_layer);
            result.experts.reserve(n_records);
            for (size_t i = 0; i < n_records; ++i) {
                expert_record rec;
                rec.file_offset = data[2 * i + 0];
                rec.size = data[2 * i + 1];
                rec.layer = (uint16_t) (i / result.n_experts_per_layer);
                rec.expert = (uint16_t) (i % result.n_experts_per_layer);
                result.experts.push_back(rec);
            }
        }
    }

    return result;
}

bool configure_from_params(
        const llama_model_params & params,
        const std::string & model_path,
        const llama_model_loader & ml) {
    manifest mf = inspect_manifest(ml.metadata);

    if (!params.moe_offload) {
        if (mf.present) {
            LLAMA_LOG_INFO("%s: MoE offload metadata detected; runtime flag is disabled\n", __func__);
        }
        configure_runtime({}, mf);
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
        LLAMA_LOG_INFO("%s: enabled MoE offload metadata v%u, layout=%s, predictor=%s, cache=%llu MiB\n",
            __func__, mf.version, mf.layout.empty() ? "unknown" : mf.layout.c_str(), opts.predictor.c_str(), (unsigned long long) opts.cache_vram_mb);
    return true;
}

} // namespace llama_moe