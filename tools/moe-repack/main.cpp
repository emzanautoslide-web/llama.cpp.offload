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

namespace {

struct args {
    std::string input;
    std::string output;
    std::string manifest;
    uint32_t alignment = 4096;
};

struct moe_layer_info {
    int layer = -1;
    int64_t n_expert = 0;
    uint64_t gate_size = 0;
    uint64_t up_size = 0;
    uint64_t down_size = 0;
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
    const long long pos = std::ftell(out);
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
    gguf_set_val_u32(dst, GGUF_KEY_GENERAL_ALIGNMENT, par.alignment);

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
            const uint64_t tensor_size = (uint64_t) gguf_get_tensor_size(src, i);
            if (kind == "gate_exps") {
                info.gate_size = tensor_size;
            } else if (kind == "up_exps") {
                info.up_size = tensor_size;
            } else if (kind == "down_exps") {
                info.down_size = tensor_size;
            }
        }
    }

    uint32_t n_moe_layers = 0;
    uint32_t n_experts_per_layer = 0;
    uint64_t expert_blob_size_max = 0;
    for (const auto & kv : moe_layers) {
        const auto & layer = kv.second;
        if (layer.n_expert <= 0 || layer.gate_size == 0 || layer.up_size == 0 || layer.down_size == 0) {
            continue;
        }
        ++n_moe_layers;
        if (n_experts_per_layer == 0) {
            n_experts_per_layer = (uint32_t) layer.n_expert;
        }
        const uint64_t per_expert = (layer.gate_size + layer.up_size + layer.down_size) / (uint64_t) layer.n_expert;
        expert_blob_size_max = std::max(expert_blob_size_max, per_expert);
    }

    gguf_set_val_u32(dst, "moe_offload.version", 1);
    gguf_set_val_str(dst, "moe_offload.layout", "fused-tensors-page-aligned-v0");
    gguf_set_val_u32(dst, "moe_offload.n_moe_layers", n_moe_layers);
    gguf_set_val_u32(dst, "moe_offload.n_experts_per_layer", n_experts_per_layer);
    gguf_set_val_u64(dst, "moe_offload.expert_blob_size_max", expert_blob_size_max);

    if (!gguf_write_to_file(dst, par.output.c_str(), true)) {
        std::cerr << "error: failed to write GGUF metadata to " << par.output << "\n";
        gguf_free(dst);
        gguf_free(src);
        ggml_free(meta_ctx);
        return 1;
    }

    std::ifstream in(par.input, std::ios::binary);
    FILE * out = std::fopen(par.output.c_str(), "ab");
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
    const uint64_t dst_data_offset = (uint64_t) gguf_get_data_offset(dst);

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
        manifest << "  \"layout\": \"fused-tensors-page-aligned-v0\",\n";
        manifest << "  \"n_moe_layers\": " << n_moe_layers << ",\n";
        manifest << "  \"n_experts_per_layer\": " << n_experts_per_layer << ",\n";
        manifest << "  \"expert_blob_size_max\": " << expert_blob_size_max << ",\n";
        manifest << "  \"note\": \"MVP metadata repack: tensors are page-aligned, fused expert tensors are not split yet.\"\n";
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