// Phase H diagnostic: ordered CUDA top-k MoE fusion check.
//
// Builds the decode-shaped MoE router subgraph used by Qwen-style softmax
// routing, then compares unfused CUDA output against the Phase H diagnostic
// fused path. Unlike the broad backend op test, this preserves expert order
// and validates the strided top-k view directly.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct route_case {
    ggml_context * ctx = nullptr;
    ggml_tensor  * logits = nullptr;
    ggml_tensor  * selected = nullptr;
    ggml_tensor  * weights = nullptr;
    ggml_cgraph  * gf = nullptr;
};

void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int fail(const char * msg) {
    fprintf(stderr, "[test-topk-moe-fusion] FAIL: %s\n", msg);
    return 1;
}

route_case build_route_case(int64_t n_expert, int64_t n_used, int64_t n_tokens) {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*16 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };

    route_case rc;
    rc.ctx = ggml_init(params);
    if (!rc.ctx) {
        return rc;
    }

    rc.logits = ggml_new_tensor_2d(rc.ctx, GGML_TYPE_F32, n_expert, n_tokens);
    ggml_set_name(rc.logits, "phase_h.logits");

    ggml_tensor * probs = ggml_soft_max(rc.ctx, rc.logits);
    ggml_set_name(probs, "phase_h.probs");

    rc.selected = ggml_argsort_top_k(rc.ctx, probs, (int) n_used);
    ggml_set_name(rc.selected, "phase_h.topk");

    ggml_tensor * probs_3d = ggml_reshape_3d(rc.ctx, probs, 1, n_expert, n_tokens);
    ggml_set_name(probs_3d, "phase_h.probs_3d");

    rc.weights = ggml_get_rows(rc.ctx, probs_3d, rc.selected);
    ggml_set_name(rc.weights, "phase_h.weights");

    rc.weights = ggml_reshape_2d(rc.ctx, rc.weights, n_used, n_tokens);
    ggml_tensor * weights_sum = ggml_sum_rows(rc.ctx, rc.weights);
    ggml_set_name(weights_sum, "phase_h.weights_sum");

    weights_sum = ggml_clamp(rc.ctx, weights_sum, 6.103515625e-5f, INFINITY);
    ggml_set_name(weights_sum, "phase_h.weights_sum_clamped");

    rc.weights = ggml_div(rc.ctx, rc.weights, weights_sum);
    ggml_set_name(rc.weights, "phase_h.weights_norm");

    rc.weights = ggml_reshape_3d(rc.ctx, rc.weights, 1, n_used, n_tokens);
    ggml_set_name(rc.weights, "phase_h.weights_norm_3d");

    rc.weights = ggml_scale(rc.ctx, rc.weights, 1.25f);
    ggml_set_name(rc.weights, "phase_h.weights_scaled");

    rc.gf = ggml_new_graph(rc.ctx);
    ggml_build_forward_expand(rc.gf, rc.weights);
    return rc;
}

void fill_logits(std::vector<float> & logits, int64_t n_expert, int64_t n_tokens, int variant) {
    logits.resize((size_t) n_expert * (size_t) n_tokens);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const int v = (int) ((expert*37 + token*19 + variant*23) % 211) - 105;
            logits[(size_t) token*(size_t) n_expert + (size_t) expert] =
                (float) v / 19.0f + 0.0001f * (float) expert;
        }
    }
}

bool read_i32_tensor_compact(const ggml_tensor * t, std::vector<int32_t> & out) {
    out.assign((size_t) ggml_nelements(t), 0);
    if (t->type != GGML_TYPE_I32 || t->nb[0] != (int64_t) sizeof(int32_t)) {
        return false;
    }
    const int64_t n_tokens = t->ne[1] * t->ne[2] * t->ne[3];
    if (t->nb[1] == t->ne[0] * (int64_t) sizeof(int32_t)) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size()*sizeof(int32_t));
    } else {
        ggml_backend_tensor_get_2d(t, out.data(), 0,
                (size_t) t->ne[0] * sizeof(int32_t),
                (size_t) n_tokens,
                (size_t) t->nb[1],
                (size_t) t->ne[0] * sizeof(int32_t));
    }
    return true;
}

bool run_route(ggml_backend_t backend, const char * name, bool fuse,
        int64_t n_expert, int64_t n_used, int64_t n_tokens, int variant,
        std::vector<int32_t> & ids, std::vector<float> & weights) {
    set_env_var("LLAMA_MOE_TOPK_FUSION_DIAG", fuse ? "1" : "0");

    route_case rc = build_route_case(n_expert, n_used, n_tokens);
    if (!rc.ctx) {
        fprintf(stderr, "[test-topk-moe-fusion] %s: ggml_init failed\n", name);
        return false;
    }

    std::vector<float> logits;
    fill_logits(logits, n_expert, n_tokens, variant);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(rc.ctx, backend);
    if (!buffer) {
        ggml_free(rc.ctx);
        fprintf(stderr, "[test-topk-moe-fusion] %s: tensor allocation failed\n", name);
        return false;
    }

    ggml_backend_tensor_set(rc.logits, logits.data(), 0, logits.size()*sizeof(float));
    ggml_backend_tensor_memset(rc.selected->view_src ? rc.selected->view_src : rc.selected, 0, 0,
            ggml_nbytes(rc.selected->view_src ? rc.selected->view_src : rc.selected));

    const ggml_status status = ggml_backend_graph_compute(backend, rc.gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[test-topk-moe-fusion] %s: graph compute failed: %s\n",
                name, ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(rc.ctx);
        return false;
    }

    if (!read_i32_tensor_compact(rc.selected, ids)) {
        fprintf(stderr, "[test-topk-moe-fusion] %s: failed to read ids\n", name);
        ggml_backend_buffer_free(buffer);
        ggml_free(rc.ctx);
        return false;
    }

    weights.assign((size_t) ggml_nelements(rc.weights), 0.0f);
    ggml_backend_tensor_get(rc.weights, weights.data(), 0, weights.size()*sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(rc.ctx);
    return true;
}

bool compare_case(const char * name, ggml_backend_t backend,
        int64_t n_expert, int64_t n_used, int64_t n_tokens, int variant) {
    std::vector<int32_t> ids_ref;
    std::vector<int32_t> ids_fused;
    std::vector<float> weights_ref;
    std::vector<float> weights_fused;

    if (!run_route(backend, name, false, n_expert, n_used, n_tokens, variant, ids_ref, weights_ref) ||
        !run_route(backend, name, true,  n_expert, n_used, n_tokens, variant, ids_fused, weights_fused)) {
        return false;
    }

    if (ids_ref != ids_fused) {
        fprintf(stderr, "[test-topk-moe-fusion] %s: id mismatch\n", name);
        for (size_t i = 0; i < std::min<size_t>(ids_ref.size(), 16); ++i) {
            fprintf(stderr, "  id[%zu] ref=%d fused=%d\n", i, ids_ref[i], ids_fused[i]);
        }
        return false;
    }

    double max_abs = 0.0;
    double rms = 0.0;
    for (size_t i = 0; i < weights_ref.size(); ++i) {
        const double diff = (double) weights_ref[i] - (double) weights_fused[i];
        max_abs = std::max(max_abs, std::abs(diff));
        rms += diff*diff;
    }
    rms = std::sqrt(rms / std::max<size_t>(weights_ref.size(), 1));

    if (max_abs > 5e-6 || rms > 1e-6) {
        fprintf(stderr, "[test-topk-moe-fusion] %s: weight mismatch max_abs=%.9g rms=%.9g\n",
                name, max_abs, rms);
        return false;
    }

    printf("[test-topk-moe-fusion] %s OK max_abs=%.9g rms=%.9g\n", name, max_abs, rms);
    return true;
}

} // namespace

int main() {
#if !defined(GGML_USE_CUDA)
    fprintf(stderr, "[test-topk-moe-fusion] CUDA disabled at build time - skipping.\n");
    return 0;
#else
    ggml_backend_load_all();

    if (ggml_backend_cuda_get_device_count() <= 0) {
        fprintf(stderr, "[test-topk-moe-fusion] No CUDA device available - skipping.\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        return fail("ggml_backend_cuda_init failed");
    }

    bool ok = true;
    ok = compare_case("decode qwen softmax norm scale", backend, 128, 8, 1, 0) && ok;
    ok = compare_case("prefill diagnostic fallback", backend, 128, 8, 4, 1) && ok;

    ggml_backend_free(backend);
    set_env_var("LLAMA_MOE_TOPK_FUSION_DIAG", "0");

    return ok ? 0 : 1;
#endif
}
