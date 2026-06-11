#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

// Windows ftell/fseek are 32-bit; we routinely operate on > 2 GiB GGUF files.
#if defined(_WIN32)
#  define moe_ftell(f)       _ftelli64(f)
#  define moe_fseek(f, o, w) _fseeki64((f), (int64_t)(o), (w))
#else
#  define moe_ftell(f)       ftello(f)
#  define moe_fseek(f, o, w) fseeko((f), (off_t)(o), (w))
#endif

namespace {

struct args {
    std::string input;
    std::string output;
    std::string manifest;
    uint32_t alignment = 4096;
};

struct moe_tensor_info {
    int64_t dst_id = -1;
    uint64_t size = 0;
    int64_t  n_expert = 0;
};

struct moe_layer_info {
    int layer = -1;
    int64_t n_expert = 0;
    moe_tensor_info gate;
    moe_tensor_info up;
    moe_tensor_info down;

    bool complete() const {
        return n_expert > 0 && gate.dst_id >= 0 && up.dst_id >= 0 && down.dst_id >= 0;
    }
};

enum moe_expert_kind {
    MOE_EXPERT_GATE = 0,
    MOE_EXPERT_UP   = 1,
    MOE_EXPERT_DOWN = 2,
    MOE_EXPERT_KIND_COUNT = 3,
};

void usage(const char * exe) {
    std::cerr << "usage: " << exe << " --input MODEL.gguf --output MODEL.moe.gguf [--manifest out.json] [--alignment 4096]\n";
}

bool parse_args(int argc, char ** argv, args & out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (++i >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[i];
        };

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--input" || arg == "-i") {
            out.input = need_value(arg.c_str());
        } else if (arg == "--output" || arg == "-o") {
            out.output = need_value(arg.c_str());
        } else if (arg == "--manifest") {
            out.manifest = need_value(arg.c_str());
        } else if (arg == "--alignment") {
            out.alignment = (uint32_t) std::stoul(need_value(arg.c_str()));
        } else if (out.input.empty()) {
            out.input = arg;
        } else if (out.output.empty()) {
            out.output = arg;
        } else {
            throw std::invalid_argument("unexpected argument: " + arg);
        }
    }

    if (out.input.empty() || out.output.empty()) {
        return false;
    }
    if (out.input == out.output) {
        throw std::invalid_argument("input and output paths must differ");
    }
    if (out.alignment == 0 || (out.alignment & (out.alignment - 1)) != 0) {
        throw std::invalid_argument("alignment must be a non-zero power of two");
    }
    if (out.manifest.empty()) {
        out.manifest = out.output + ".json";
    }
    return true;
}

bool copy_exact(std::ifstream & in, FILE * out, uint64_t offset, uint64_t size) {
    static constexpr size_t buf_size = 8 * 1024 * 1024;
    std::vector<char> buffer(buf_size);
    in.clear();
    in.seekg((std::streamoff) offset, std::ios::beg);
    if (!in) {
        return false;
    }

    uint64_t remaining = size;
    while (remaining > 0) {
        const size_t chunk = (size_t) std::min<uint64_t>(remaining, buffer.size());
        in.read(buffer.data(), (std::streamsize) chunk);
        if ((size_t) in.gcount() != chunk) {
            return false;
        }
        if (std::fwrite(buffer.data(), 1, chunk, out) != chunk) {
            return false;
        }
        remaining -= chunk;
    }
    return true;
}

bool pad_to(FILE * out, uint64_t absolute_offset) {
    const int64_t pos = moe_ftell(out);
    if (pos < 0) {
        return false;
    }
    if ((uint64_t) pos > absolute_offset) {
        return false;
    }

    static const char zeros[4096] = {};
    uint64_t remaining = absolute_offset - (uint64_t) pos;
    while (remaining > 0) {
        const size_t chunk = (size_t) std::min<uint64_t>(remaining, sizeof(zeros));
        if (std::fwrite(zeros, 1, chunk, out) != chunk) {
            return false;
        }
        remaining -= chunk;
    }
    return true;
}

bool parse_moe_tensor_name(const std::string & name, int & layer, std::string & kind) {
    int parsed_layer = -1;
    char parsed_kind[32] = {};
    if (std::sscanf(name.c_str(), "blk.%d.ffn_%31[^.]", &parsed_layer, parsed_kind) != 2) {
        return false;
    }
    std::string k = parsed_kind;
    if (k != "gate_exps" && k != "up_exps" && k != "down_exps") {
        return false;
    }
    layer = parsed_layer;
    kind = k;
    return true;
}

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    args par;
    try {
        if (!parse_args(argc, argv, par)) {
            usage(argv[0]);
            return 1;
        }
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    ggml_context * meta_ctx = nullptr;
    gguf_init_params init_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };

    gguf_context * src = gguf_init_from_file(par.input.c_str(), init_params);
    if (!src || !meta_ctx) {
        std::cerr << "error: failed to read GGUF metadata from " << par.input << "\n";
        if (src) {
            gguf_free(src);
        }
        return 1;
    }

    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    // NOTE: gguf_set_val_u32(GGUF_KEY_GENERAL_ALIGNMENT, X) only writes the KV;
    // it does NOT update ctx->alignment, so gguf_add_tensor would compute offsets
    // at the default 32-byte alignment while the loader would expect X-aligned
    // offsets. Until gguf exposes an alignment setter, we keep the source
    // file's alignment (copied via gguf_set_kv) and rely on the data plane
    // to handle sub-page alignment internally.
    (void) par.alignment;

    std::map<int, moe_layer_info> moe_layers;
    int64_t n_tensors = gguf_get_n_tensors(src);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(src, i);
        ggml_tensor * tensor = ggml_get_tensor(meta_ctx, name);
        if (!tensor) {
            std::cerr << "error: tensor metadata missing from ggml context: " << name << "\n";
            gguf_free(dst);
            gguf_free(src);
            ggml_free(meta_ctx);
            return 1;
        }
        gguf_add_tensor(dst, tensor);

        int layer = -1;
        std::string kind;
        if (parse_moe_tensor_name(name, layer, kind)) {
            auto & info = moe_layers[layer];
            info.layer = layer;
            info.n_expert = tensor->ne[2];
            const int64_t dst_id = gguf_find_tensor(dst, name);
            const uint64_t tensor_size = (uint64_t) gguf_get_tensor_size(src, i);
            moe_tensor_info ti;
            ti.dst_id = dst_id;
            ti.size = tensor_size;
            ti.n_expert = tensor->ne[2];
            if (kind == "gate_exps") {
                info.gate = ti;
            } else if (kind == "up_exps") {
                info.up = ti;
            } else if (kind == "down_exps") {
                info.down = ti;
            }
        }
    }

    std::vector<uint32_t> layer_ids;
    uint32_t n_experts_per_layer = 0;
    uint64_t expert_blob_size_max = 0;
    for (const auto & kv : moe_layers) {
        const auto & layer = kv.second;
        if (!layer.complete()) {
            continue;
        }
        layer_ids.push_back((uint32_t) layer.layer);
        if (n_experts_per_layer == 0) {
            n_experts_per_layer = (uint32_t) layer.n_expert;
        } else if ((uint32_t) layer.n_expert != n_experts_per_layer) {
            std::cerr << "error: layer " << layer.layer << " has " << layer.n_expert
                      << " experts but expected " << n_experts_per_layer << "\n";
            gguf_free(dst);
            gguf_free(src);
            ggml_free(meta_ctx);
            return 1;
        }
        const uint64_t per_gate = layer.gate.size / (uint64_t) layer.n_expert;
        const uint64_t per_up   = layer.up.size   / (uint64_t) layer.n_expert;
        const uint64_t per_down = layer.down.size / (uint64_t) layer.n_expert;
        expert_blob_size_max = std::max({expert_blob_size_max, per_gate, per_up, per_down});
    }
    const uint32_t n_moe_layers = (uint32_t) layer_ids.size();

    // Build the per (logical_layer, expert, kind) -> (relative_offset, size) table.
    // Relative offsets are within the GGUF data section; the loader adds gguf_get_data_offset(dst)
    // at runtime to produce absolute file offsets. Records are laid out as:
    //   idx = ((logical_layer * n_experts_per_layer + expert) * 3 + kind) * 2 + field
    //   kind: 0=gate, 1=up, 2=down ;  field: 0=offset, 1=size
    std::vector<uint64_t> table;
    table.resize((size_t) n_moe_layers * n_experts_per_layer * MOE_EXPERT_KIND_COUNT * 2, 0);
    auto record = [&](uint32_t li, uint32_t e, moe_expert_kind k, uint64_t off, uint64_t sz) {
        const size_t idx = (((size_t) li * n_experts_per_layer + e) * MOE_EXPERT_KIND_COUNT + (size_t) k) * 2;
        table[idx + 0] = off;
        table[idx + 1] = sz;
    };
    for (uint32_t li = 0; li < n_moe_layers; ++li) {
        const auto & layer = moe_layers[(int) layer_ids[li]];
        const auto pack = [&](moe_expert_kind kind, const moe_tensor_info & ti) {
            const uint64_t base = (uint64_t) gguf_get_tensor_offset(dst, ti.dst_id);
            const uint64_t per_expert = ti.size / (uint64_t) layer.n_expert;
            for (int64_t e = 0; e < layer.n_expert; ++e) {
                record(li, (uint32_t) e, kind, base + (uint64_t) e * per_expert, per_expert);
            }
        };
        pack(MOE_EXPERT_GATE, layer.gate);
        pack(MOE_EXPERT_UP,   layer.up);
        pack(MOE_EXPERT_DOWN, layer.down);
    }

    gguf_set_val_u32(dst, "moe_offload.version", 2);
    gguf_set_val_str(dst, "moe_offload.layout", "fused-tensors-page-aligned-v1");
    gguf_set_val_u32(dst, "moe_offload.n_moe_layers", n_moe_layers);
    gguf_set_val_u32(dst, "moe_offload.n_experts_per_layer", n_experts_per_layer);
    gguf_set_val_u64(dst, "moe_offload.expert_blob_size_max", expert_blob_size_max);
    gguf_set_arr_data(dst, "moe_offload.layer_ids", GGUF_TYPE_UINT32,
                      layer_ids.data(), layer_ids.size());
    gguf_set_arr_data(dst, "moe_offload.expert_blob.table", GGUF_TYPE_UINT64,
                      table.data(), table.size());

    if (!gguf_write_to_file(dst, par.output.c_str(), true)) {
        std::cerr << "error: failed to write GGUF metadata to " << par.output << "\n";
        gguf_free(dst);
        gguf_free(src);
        ggml_free(meta_ctx);
        return 1;
    }

    std::ifstream in(par.input, std::ios::binary);
    FILE * out = std::fopen(par.output.c_str(), "rb+");
    if (!in || !out) {
        std::cerr << "error: failed to open input/output data streams\n";
        if (out) {
            std::fclose(out);
        }
        gguf_free(dst);
        gguf_free(src);
        ggml_free(meta_ctx);
        return 1;
    }

    const uint64_t src_data_offset = (uint64_t) gguf_get_data_offset(src);
    // gguf_init_empty does not populate ctx->offset, so gguf_get_data_offset(dst) is 0.
    // The true destination data section starts at the end of the metadata-only write.
    moe_fseek(out, 0, SEEK_END);
    const uint64_t dst_data_offset = (uint64_t) moe_ftell(out);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(src, i);
        const int64_t dst_id = gguf_find_tensor(dst, name);
        if (dst_id < 0) {
            std::cerr << "error: output tensor metadata missing: " << name << "\n";
            std::fclose(out);
            return 1;
        }

        const uint64_t src_offset = src_data_offset + (uint64_t) gguf_get_tensor_offset(src, i);
        const uint64_t dst_offset = dst_data_offset + (uint64_t) gguf_get_tensor_offset(dst, dst_id);
        const uint64_t size = (uint64_t) gguf_get_tensor_size(src, i);

        if (!pad_to(out, dst_offset) || !copy_exact(in, out, src_offset, size)) {
            std::cerr << "error: failed while copying tensor data: " << name << "\n";
            std::fclose(out);
            return 1;
        }
    }

    std::fclose(out);

    std::ofstream manifest(par.manifest, std::ios::out | std::ios::trunc);
    if (manifest) {
        manifest << "{\n";
        manifest << "  \"input\": \"" << json_escape(par.input) << "\",\n";
        manifest << "  \"output\": \"" << json_escape(par.output) << "\",\n";
        manifest << "  \"alignment\": " << par.alignment << ",\n";
        manifest << "  \"layout\": \"fused-tensors-page-aligned-v1\",\n";
        manifest << "  \"version\": 2,\n";
        manifest << "  \"n_moe_layers\": " << n_moe_layers << ",\n";
        manifest << "  \"n_experts_per_layer\": " << n_experts_per_layer << ",\n";
        manifest << "  \"expert_blob_size_max\": " << expert_blob_size_max << ",\n";
        manifest << "  \"note\": \"v1 layout: fused expert tensors page-aligned; per-expert byte ranges recorded in moe_offload.expert_blob.table (relative to data section start). Order: (logical_layer, expert, kind={gate,up,down}, field={offset,size}).\"\n";
        manifest << "}\n";
    }

    std::cout << "wrote " << par.output << "\n";
    std::cout << "wrote " << par.manifest << "\n";
    std::cout << "MoE layers: " << n_moe_layers << ", experts/layer: " << n_experts_per_layer << "\n";

    gguf_free(dst);
    gguf_free(src);
    ggml_free(meta_ctx);
    return 0;
}