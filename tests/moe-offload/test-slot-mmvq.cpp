// ===========================================================================
// Phase E micro-test: guarded CUDA MMVQ for MoE slot-backed MUL_MAT_ID.
//
// The test builds a synthetic `.slot` MUL_MAT_ID graph and compares the CUDA
// output against a CPU reference computed from the same quantized slot tensor.
// It runs the default generic sorted path, the guarded LLAMA_MOE_SLOT_MMVQ=1
// path, the LLAMA_MOE_SLOT_GRAPHS=1 replay path, and Phase G's guarded
// LLAMA_MOE_SLOT_GLU_FUSION=1 decode-only GLU fusion. A prefill-shaped batch
// is also checked with the guards enabled to verify it still falls back
// cleanly.
// ===========================================================================

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct tensor_data {
    std::vector<uint8_t> as_quant;
    std::vector<float>   as_float;
    std::vector<float>   b;
    std::vector<int32_t> ids;
    std::vector<float>   reference;
};

struct graph_case {
    ggml_context * ctx = nullptr;
    ggml_tensor  * as  = nullptr;
    ggml_tensor  * b   = nullptr;
    ggml_tensor  * ids = nullptr;
    ggml_tensor  * out = nullptr;
    ggml_cgraph  * gf  = nullptr;
};

struct glu_tensor_data {
    std::vector<uint8_t> up_quant;
    std::vector<uint8_t> gate_quant;
    std::vector<float>   up_float;
    std::vector<float>   gate_float;
    std::vector<float>   b;
    std::vector<int32_t> ids;
};

struct glu_graph_case {
    ggml_context * ctx  = nullptr;
    ggml_tensor  * up   = nullptr;
    ggml_tensor  * gate = nullptr;
    ggml_tensor  * b    = nullptr;
    ggml_tensor  * ids  = nullptr;
    ggml_tensor  * out  = nullptr;
    ggml_cgraph  * gf   = nullptr;
};

int fail(const char * msg) {
    fprintf(stderr, "[test-slot-mmvq] FAIL: %s\n", msg);
    return 1;
}

void set_slot_mmvq_env(bool enabled) {
#if defined(_WIN32)
    _putenv(enabled ? "LLAMA_MOE_SLOT_MMVQ=1" : "LLAMA_MOE_SLOT_MMVQ=0");
#else
    setenv("LLAMA_MOE_SLOT_MMVQ", enabled ? "1" : "0", 1);
#endif
}

void set_slot_graphs_env(bool enabled) {
#if defined(_WIN32)
    _putenv(enabled ? "LLAMA_MOE_SLOT_GRAPHS=1" : "LLAMA_MOE_SLOT_GRAPHS=0");
#else
    setenv("LLAMA_MOE_SLOT_GRAPHS", enabled ? "1" : "0", 1);
#endif
}

void set_slot_glu_fusion_env(bool enabled) {
#if defined(_WIN32)
    _putenv(enabled ? "LLAMA_MOE_SLOT_GLU_FUSION=1" : "LLAMA_MOE_SLOT_GLU_FUSION=0");
#else
    setenv("LLAMA_MOE_SLOT_GLU_FUSION", enabled ? "1" : "0", 1);
#endif
}

graph_case build_case(ggml_type type, int64_t k, int64_t m, int64_t n_slots, int64_t n_used, int64_t n_tokens) {
    const size_t tensor_count = 8;
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*tensor_count + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    graph_case gc;
    gc.ctx = ggml_init(params);
    if (!gc.ctx) {
        return gc;
    }

    gc.as = ggml_new_tensor_3d(gc.ctx, type, k, m, n_slots);
    ggml_set_name(gc.as, "phase_e.synthetic.slot");

    gc.b = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    ggml_set_name(gc.b, "phase_e.b");

    gc.ids = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_name(gc.ids, "phase_e.ids");

    gc.out = ggml_mul_mat_id(gc.ctx, gc.as, gc.b, gc.ids);
    ggml_set_name(gc.out, "phase_e.out");

    gc.gf = ggml_new_graph(gc.ctx);
    ggml_build_forward_expand(gc.gf, gc.out);

    return gc;
}

glu_graph_case build_glu_case(ggml_type type, int64_t k, int64_t m, int64_t n_slots, int64_t n_used, int64_t n_tokens) {
    const size_t tensor_count = 12;
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*tensor_count + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    glu_graph_case gc;
    gc.ctx = ggml_init(params);
    if (!gc.ctx) {
        return gc;
    }

    gc.up = ggml_new_tensor_3d(gc.ctx, type, k, m, n_slots);
    ggml_set_name(gc.up, "phase_g.up.slot");

    gc.gate = ggml_new_tensor_3d(gc.ctx, type, k, m, n_slots);
    ggml_set_name(gc.gate, "phase_g.gate.slot");

    gc.b = ggml_new_tensor_3d(gc.ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    ggml_set_name(gc.b, "phase_g.b");

    gc.ids = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_name(gc.ids, "phase_g.ids");

    ggml_tensor * up_mm = ggml_mul_mat_id(gc.ctx, gc.up, gc.b, gc.ids);
    ggml_set_name(up_mm, "phase_g.up_mm");

    ggml_tensor * gate_mm = ggml_mul_mat_id(gc.ctx, gc.gate, gc.b, gc.ids);
    ggml_set_name(gate_mm, "phase_g.gate_mm");

    gc.out = ggml_swiglu_split(gc.ctx, gate_mm, up_mm);
    ggml_set_name(gc.out, "phase_g.out");

    gc.gf = ggml_new_graph(gc.ctx);
    ggml_build_forward_expand(gc.gf, gc.out);

    return gc;
}

void fill_data(tensor_data & data, ggml_type type, int64_t k, int64_t m, int64_t n_slots, int64_t n_used, int64_t n_tokens, int variant = 0) {
    data.as_float.resize((size_t) k*m*n_slots);
    data.b.resize((size_t) k*n_used*n_tokens);
    data.ids.resize((size_t) n_used*n_tokens);
    data.reference.assign((size_t) m*n_used*n_tokens, 0.0f);

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                const size_t idx = (size_t) slot*k*m + (size_t) row*k + (size_t) col;
                const int v = (int) ((idx*17 + 11*slot + 5*row + col + 23*variant) % 127) - 63;
                data.as_float[idx] = (float) v / 64.0f;
            }
        }
    }

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t used = 0; used < n_used; ++used) {
            data.ids[(size_t) token*n_used + (size_t) used] = (int32_t) ((token*3 + used*5 + 7*variant) % n_slots);
            for (int64_t col = 0; col < k; ++col) {
                const size_t idx = (size_t) token*n_used*k + (size_t) used*k + (size_t) col;
                const int v = (int) ((idx*13 + 7*token + 3*used + col + 19*variant) % 97) - 48;
                data.b[idx] = (float) v / 48.0f;
            }
        }
    }

    data.as_quant.resize(ggml_row_size(type, k) * (size_t) m * (size_t) n_slots);
    std::vector<float> row((size_t) k);
    const size_t row_size = ggml_row_size(type, k);
    for (int64_t slot = 0; slot < n_slots; ++slot) {
        for (int64_t row_idx = 0; row_idx < m; ++row_idx) {
            const size_t src = (size_t) slot*k*m + (size_t) row_idx*k;
            std::copy(data.as_float.begin() + src, data.as_float.begin() + src + k, row.begin());
            ggml_quantize_chunk(type, row.data(), data.as_quant.data() + ((size_t) slot*m + (size_t) row_idx)*row_size,
                    0, 1, k, nullptr);
        }
    }
}

void quantize_slots(ggml_type type, int64_t k, int64_t m, int64_t n_slots,
        const std::vector<float> & src, std::vector<uint8_t> & dst) {
    dst.resize(ggml_row_size(type, k) * (size_t) m * (size_t) n_slots);
    std::vector<float> row((size_t) k);
    const size_t row_size = ggml_row_size(type, k);

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        for (int64_t row_idx = 0; row_idx < m; ++row_idx) {
            const size_t src_idx = (size_t) slot*k*m + (size_t) row_idx*k;
            std::copy(src.begin() + src_idx, src.begin() + src_idx + k, row.begin());
            ggml_quantize_chunk(type, row.data(), dst.data() + ((size_t) slot*m + (size_t) row_idx)*row_size,
                    0, 1, k, nullptr);
        }
    }
}

void fill_glu_data(glu_tensor_data & data, ggml_type type, int64_t k, int64_t m, int64_t n_slots,
        int64_t n_used, int64_t n_tokens, int variant = 0) {
    data.up_float.resize((size_t) k*m*n_slots);
    data.gate_float.resize((size_t) k*m*n_slots);
    data.b.resize((size_t) k*n_used*n_tokens);
    data.ids.resize((size_t) n_used*n_tokens);

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                const size_t idx = (size_t) slot*k*m + (size_t) row*k + (size_t) col;
                const int up_v = (int) ((idx*19 + 13*slot + 3*row + col + 29*variant) % 127) - 63;
                const int gate_v = (int) ((idx*23 + 5*slot + 11*row + 7*col + 31*variant) % 127) - 63;
                data.up_float[idx] = (float) up_v / 64.0f;
                data.gate_float[idx] = (float) gate_v / 64.0f;
            }
        }
    }

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t used = 0; used < n_used; ++used) {
            data.ids[(size_t) token*n_used + (size_t) used] = (int32_t) ((token*7 + used*3 + 5*variant) % n_slots);
            for (int64_t col = 0; col < k; ++col) {
                const size_t idx = (size_t) token*n_used*k + (size_t) used*k + (size_t) col;
                const int v = (int) ((idx*17 + 5*token + 9*used + col + 37*variant) % 97) - 48;
                data.b[idx] = (float) v / 48.0f;
            }
        }
    }

    quantize_slots(type, k, m, n_slots, data.up_float, data.up_quant);
    quantize_slots(type, k, m, n_slots, data.gate_float, data.gate_quant);
}

bool compute_reference(tensor_data & data, ggml_type type, int64_t k, int64_t m, int64_t n_used, int64_t n_tokens) {
    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) {
        return false;
    }

    const size_t row_size = ggml_row_size(type, k);
    std::vector<float> row((size_t) k);

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t used = 0; used < n_used; ++used) {
            const int64_t slot = data.ids[(size_t) token*n_used + (size_t) used];
            for (int64_t row_idx = 0; row_idx < m; ++row_idx) {
                const uint8_t * qrow = data.as_quant.data() + ((size_t) slot*m + (size_t) row_idx)*row_size;
                traits->to_float(qrow, row.data(), k);

                float sum = 0.0f;
                for (int64_t col = 0; col < k; ++col) {
                    const float bv = data.b[(size_t) token*n_used*k + (size_t) used*k + (size_t) col];
                    sum += row[(size_t) col] * bv;
                }
                data.reference[(size_t) token*n_used*m + (size_t) used*m + (size_t) row_idx] = sum;
            }
        }
    }

    return true;
}

bool check_output(const char * name, const std::vector<float> & out, const std::vector<float> & reference) {
    if (out.size() != reference.size()) {
        fprintf(stderr, "[test-slot-mmvq] %s: output size mismatch got=%zu want=%zu\n",
                name, out.size(), reference.size());
        return false;
    }

    double max_abs = 0.0;
    double rms = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double diff = (double) out[i] - (double) reference[i];
        max_abs = std::max(max_abs, std::abs(diff));
        rms += diff*diff;
    }
    rms = std::sqrt(rms / std::max<size_t>(out.size(), 1));

    // Q4_0 accumulation differences are expected; this gate catches indexing
    // and layout mistakes without requiring bit-identical CUDA/CPU arithmetic.
    if (max_abs > 0.08 || rms > 0.02) {
        fprintf(stderr, "[test-slot-mmvq] %s: mismatch max_abs=%.8f rms=%.8f\n", name, max_abs, rms);
        return false;
    }

    printf("[test-slot-mmvq] %s OK max_abs=%.8f rms=%.8f\n", name, max_abs, rms);
    return true;
}

bool run_cuda_case(const char * name, ggml_backend_t backend, ggml_type type,
        int64_t k, int64_t m, int64_t n_slots, int64_t n_used, int64_t n_tokens, bool enable_mmvq, bool enable_graphs = false) {
    set_slot_mmvq_env(enable_mmvq);
    set_slot_graphs_env(enable_graphs);

    graph_case gc = build_case(type, k, m, n_slots, n_used, n_tokens);
    if (!gc.ctx) {
        fprintf(stderr, "[test-slot-mmvq] %s: ggml_init failed\n", name);
        return false;
    }

    tensor_data data;
    fill_data(data, type, k, m, n_slots, n_used, n_tokens);
    if (!compute_reference(data, type, k, m, n_used, n_tokens)) {
        ggml_free(gc.ctx);
        fprintf(stderr, "[test-slot-mmvq] %s: reference conversion unavailable\n", name);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(gc.ctx, backend);
    if (!buffer) {
        ggml_free(gc.ctx);
        fprintf(stderr, "[test-slot-mmvq] %s: tensor allocation failed\n", name);
        return false;
    }

    ggml_backend_tensor_set(gc.as, data.as_quant.data(), 0, data.as_quant.size());
    ggml_backend_tensor_set(gc.b, data.b.data(), 0, data.b.size()*sizeof(float));
    ggml_backend_tensor_set(gc.ids, data.ids.data(), 0, data.ids.size()*sizeof(int32_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gc.gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(gc.ctx);
        fprintf(stderr, "[test-slot-mmvq] %s: graph compute failed: %s\n", name, ggml_status_to_string(status));
        return false;
    }

    std::vector<float> out((size_t) ggml_nelements(gc.out));
    ggml_backend_tensor_get(gc.out, out.data(), 0, out.size()*sizeof(float));

    const bool ok = check_output(name, out, data.reference);

    ggml_backend_buffer_free(buffer);
    ggml_free(gc.ctx);
    return ok;
}

bool run_cuda_graph_replay_case(const char * name, ggml_backend_t backend, ggml_type type,
        int64_t k, int64_t m, int64_t n_slots, int64_t n_used) {
    set_slot_mmvq_env(true);
    set_slot_graphs_env(true);

    graph_case gc = build_case(type, k, m, n_slots, n_used, 1);
    if (!gc.ctx) {
        fprintf(stderr, "[test-slot-mmvq] %s: ggml_init failed\n", name);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(gc.ctx, backend);
    if (!buffer) {
        ggml_free(gc.ctx);
        fprintf(stderr, "[test-slot-mmvq] %s: tensor allocation failed\n", name);
        return false;
    }

    bool ok = true;
    for (int iter = 0; iter < 5; ++iter) {
        tensor_data data;
        fill_data(data, type, k, m, n_slots, n_used, 1, iter);
        if (!compute_reference(data, type, k, m, n_used, 1)) {
            fprintf(stderr, "[test-slot-mmvq] %s: reference conversion unavailable\n", name);
            ok = false;
            break;
        }

        ggml_backend_tensor_set(gc.as, data.as_quant.data(), 0, data.as_quant.size());
        ggml_backend_tensor_set(gc.b, data.b.data(), 0, data.b.size()*sizeof(float));
        ggml_backend_tensor_set(gc.ids, data.ids.data(), 0, data.ids.size()*sizeof(int32_t));

        const ggml_status status = ggml_backend_graph_compute(backend, gc.gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[test-slot-mmvq] %s iter=%d: graph compute failed: %s\n",
                    name, iter, ggml_status_to_string(status));
            ok = false;
            break;
        }

        std::vector<float> out((size_t) ggml_nelements(gc.out));
        ggml_backend_tensor_get(gc.out, out.data(), 0, out.size()*sizeof(float));

        char iter_name[128];
        snprintf(iter_name, sizeof(iter_name), "%s iter=%d", name, iter);
        if (!check_output(iter_name, out, data.reference)) {
            ok = false;
            break;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(gc.ctx);
    return ok;
}

bool set_glu_case_tensors(const char * name, const glu_graph_case & gc, const glu_tensor_data & data) {
    if (!gc.up || !gc.gate || !gc.b || !gc.ids) {
        fprintf(stderr, "[test-slot-mmvq] %s: incomplete GLU graph\n", name);
        return false;
    }

    ggml_backend_tensor_set(gc.up, data.up_quant.data(), 0, data.up_quant.size());
    ggml_backend_tensor_set(gc.gate, data.gate_quant.data(), 0, data.gate_quant.size());
    ggml_backend_tensor_set(gc.b, data.b.data(), 0, data.b.size()*sizeof(float));
    ggml_backend_tensor_set(gc.ids, data.ids.data(), 0, data.ids.size()*sizeof(int32_t));
    return true;
}

bool compute_glu_output(const char * name, ggml_backend_t backend, const glu_graph_case & gc,
        const glu_tensor_data & data, bool enable_glu_fusion, bool enable_graphs, std::vector<float> & out) {
    set_slot_mmvq_env(true);
    set_slot_glu_fusion_env(enable_glu_fusion);
    set_slot_graphs_env(enable_graphs);

    if (!set_glu_case_tensors(name, gc, data)) {
        return false;
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gc.gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[test-slot-mmvq] %s: graph compute failed: %s\n", name, ggml_status_to_string(status));
        return false;
    }

    out.resize((size_t) ggml_nelements(gc.out));
    ggml_backend_tensor_get(gc.out, out.data(), 0, out.size()*sizeof(float));
    return true;
}

bool run_cuda_glu_case(const char * name, ggml_backend_t backend, ggml_type type,
        int64_t k, int64_t m, int64_t n_slots, int64_t n_used, int64_t n_tokens,
        bool enable_glu_fusion, bool enable_graphs = false) {
    set_slot_mmvq_env(true);
    set_slot_glu_fusion_env(false);
    set_slot_graphs_env(false);

    glu_graph_case baseline_gc = build_glu_case(type, k, m, n_slots, n_used, n_tokens);
    if (!baseline_gc.ctx) {
        fprintf(stderr, "[test-slot-mmvq] %s baseline: ggml_init failed\n", name);
        return false;
    }

    glu_graph_case gc = build_glu_case(type, k, m, n_slots, n_used, n_tokens);
    if (!gc.ctx) {
        ggml_free(baseline_gc.ctx);
        fprintf(stderr, "[test-slot-mmvq] %s: ggml_init failed\n", name);
        return false;
    }

    glu_tensor_data data;
    fill_glu_data(data, type, k, m, n_slots, n_used, n_tokens);

    ggml_backend_buffer_t baseline_buffer = ggml_backend_alloc_ctx_tensors(baseline_gc.ctx, backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(gc.ctx, backend);
    if (!baseline_buffer || !buffer) {
        if (baseline_buffer) {
            ggml_backend_buffer_free(baseline_buffer);
        }
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(baseline_gc.ctx);
        ggml_free(gc.ctx);
        fprintf(stderr, "[test-slot-mmvq] %s: tensor allocation failed\n", name);
        return false;
    }

    std::vector<float> baseline;
    std::vector<float> out;
    bool ok = compute_glu_output("GLU baseline", backend, baseline_gc, data, false, false, baseline);
    ok = compute_glu_output(name, backend, gc, data, enable_glu_fusion, enable_graphs, out) && ok;
    ok = ok && check_output(name, out, baseline);

    ggml_backend_buffer_free(baseline_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(baseline_gc.ctx);
    ggml_free(gc.ctx);
    return ok;
}

bool run_cuda_glu_graph_replay_case(const char * name, ggml_backend_t backend, ggml_type type,
        int64_t k, int64_t m, int64_t n_slots, int64_t n_used) {
    set_slot_mmvq_env(true);
    set_slot_glu_fusion_env(true);
    set_slot_graphs_env(true);

    glu_graph_case baseline_gc = build_glu_case(type, k, m, n_slots, n_used, 1);
    if (!baseline_gc.ctx) {
        fprintf(stderr, "[test-slot-mmvq] %s baseline: ggml_init failed\n", name);
        return false;
    }

    glu_graph_case gc = build_glu_case(type, k, m, n_slots, n_used, 1);
    if (!gc.ctx) {
        ggml_free(baseline_gc.ctx);
        fprintf(stderr, "[test-slot-mmvq] %s: ggml_init failed\n", name);
        return false;
    }

    ggml_backend_buffer_t baseline_buffer = ggml_backend_alloc_ctx_tensors(baseline_gc.ctx, backend);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(gc.ctx, backend);
    if (!baseline_buffer || !buffer) {
        if (baseline_buffer) {
            ggml_backend_buffer_free(baseline_buffer);
        }
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(baseline_gc.ctx);
        ggml_free(gc.ctx);
        fprintf(stderr, "[test-slot-mmvq] %s: tensor allocation failed\n", name);
        return false;
    }

    bool ok = true;
    for (int iter = 0; iter < 5; ++iter) {
        glu_tensor_data data;
        fill_glu_data(data, type, k, m, n_slots, n_used, 1, iter);

        std::vector<float> baseline;
        std::vector<float> out;
        ok = compute_glu_output("GLU graph replay baseline", backend, baseline_gc, data, false, false, baseline);
        ok = compute_glu_output(name, backend, gc, data, true, true, out) && ok;

        char iter_name[128];
        snprintf(iter_name, sizeof(iter_name), "%s iter=%d", name, iter);
        if (!ok || !check_output(iter_name, out, baseline)) {
            ok = false;
            break;
        }
    }

    ggml_backend_buffer_free(baseline_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(baseline_gc.ctx);
    ggml_free(gc.ctx);
    return ok;
}

} // namespace

int main() {
#if !defined(GGML_USE_CUDA)
    fprintf(stderr, "[test-slot-mmvq] CUDA disabled at build time - skipping.\n");
    return 0;
#else
    ggml_backend_load_all();

    if (ggml_backend_cuda_get_device_count() <= 0) {
        fprintf(stderr, "[test-slot-mmvq] No CUDA device available - skipping.\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        return fail("ggml_backend_cuda_init failed");
    }

    constexpr ggml_type type = GGML_TYPE_Q4_0;
    constexpr int64_t k = 128;
    constexpr int64_t m = 64;
    constexpr int64_t n_slots = 16;
    constexpr int64_t n_used = 4;

    bool ok = true;
    ok = run_cuda_case("decode generic sorted", backend, type, k, m, n_slots, n_used, 1, false) && ok;
    ok = run_cuda_case("decode guarded MMVQ", backend, type, k, m, n_slots, n_used, 1, true) && ok;
    ok = run_cuda_graph_replay_case("decode guarded MMVQ graphs", backend, type, k, m, n_slots, n_used) && ok;
    ok = run_cuda_case("prefill guard fallback", backend, type, k, m, n_slots, n_used, 4, true, true) && ok;
    ok = run_cuda_glu_case("decode guarded GLU unfused", backend, type, k, m, n_slots, n_used, 1, false) && ok;
    ok = run_cuda_glu_case("decode guarded GLU fusion", backend, type, k, m, n_slots, n_used, 1, true) && ok;
    ok = run_cuda_glu_graph_replay_case("decode guarded GLU fusion graphs", backend, type, k, m, n_slots, n_used) && ok;
    ok = run_cuda_glu_case("prefill GLU fusion fallback", backend, type, k, m, n_slots, n_used, 4, true, true) && ok;

    ggml_backend_free(backend);
    set_slot_mmvq_env(false);
    set_slot_graphs_env(false);
    set_slot_glu_fusion_env(false);

    return ok ? 0 : 1;
#endif
}
