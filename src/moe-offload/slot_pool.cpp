#include "slot_pool.h"

#include "io.h"
#include "loader.h"
#include "predictor.h"
#include "profiler.h"
#include "runtime.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-impl.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#  define moe_fseek(f, off, whence) _fseeki64((f), (off), (whence))
#  define moe_ftell(f)              _ftelli64(f)
#else
#  define moe_fseek(f, off, whence) fseeko((f), (off), (whence))
#  define moe_ftell(f)              ftello(f)
#endif

namespace llama_moe {

namespace {

struct slot_pool_state {
    std::mutex mutex;
    bool configured = false;
    uint32_t n_slots = 0;
    // slot_tensors[logical_layer][kind] -> tensor
    std::vector<std::array<ggml_tensor *, EXPERT_KIND_COUNT>> slot_tensors;
    // Phase D-4: per-logical-layer slot_table tensor [1, n_expert] I32,
    // pre-allocated on the same backend buffer as the slot weight tensors
    // (via `ml.create_unfiled_tensor`). Persistent storage means the
    // scheduler does NOT recycle the buffer for graph temporaries, so the
    // eval-callback's mid-graph `ggml_backend_tensor_set` is read correctly
    // by the immediately-following `ggml_get_rows` consumer.
    std::vector<ggml_tensor *> slot_table_tensors;
    // Per-graph top-k registries. Streaming mode uses topk_to_slot_ids so the
    // callback can fill the exact ids consumed by MUL_MAT_ID. The older
    // slot-table path is kept for guarded fallback diagnostics.
    std::unordered_map<ggml_tensor *, ggml_tensor *> topk_to_slot_table;
    std::unordered_map<ggml_tensor *, ggml_tensor *> topk_to_slot_ids;
    std::unordered_map<ggml_tensor *, int> topk_to_logical;
    // Flat list of all registered slot_table tensors across graph builds.
    // Used by populate_slot_tables_identity (non-streaming mode) which writes
    // the identity mapping to all of them.
    std::vector<ggml_tensor *> all_slot_tables;

    // Phase D-2: per-logical-layer LRU cache (only used when streaming).
    struct layer_cache {
        std::vector<int32_t> slot_to_expert;          // size = n_slots, -1 = free
        std::unordered_map<int32_t, int32_t> exp2slot;
        std::list<int32_t> lru;                       // front = most-recent expert
        std::unordered_map<int32_t, std::list<int32_t>::iterator> lru_it;
        // Optional debug fingerprint of the first 1024 bytes of each cached
        // expert's slot data (gate kind).
        std::unordered_map<int32_t, uint64_t> fingerprints;
    };
    std::vector<layer_cache> cache;
    // Reusable host buffer for slot_table writes (size n_expert).
    std::vector<int32_t> slot_table_host;
    // Reusable scratch for the largest expert blob (one kind).
    std::vector<uint8_t> io_scratch;
    // Lazily opened FILE handle on mf.source_path.
    FILE * io_fp = nullptr;
    // Stats
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t topk_calls = 0;

    // Phase E: predictor
    std::unique_ptr<predictor> pred;
    bool pred_dirty = false;
    uint64_t token_idx = 0;
    uint64_t current_token_idx = 0;

    // Phase H: CUDA backend whose compute stream is stalled on async H2D
    // events via cudaStreamWaitEvent. Set by slot_pool_set_compute_backend.
    // When null, the eval-callback falls back to synchronous H2D.
    ggml_backend_t compute_backend = nullptr;

    // Phase I: per-batch buffered profile rows. We record compute_us / h2d_us
    // via CUDA timing events whose elapsed time can only be queried after the
    // compute stream catches up. Rows accumulate here during the batch and
    // are patched + flushed to the profiler at slot_pool_end_request().
    struct pending_profile_row {
        profile_row row;
        int  logical = -1;
        void * compute_begin_event = nullptr;
        void * compute_end_event   = nullptr;
        // (begin, end) timing-capable events per missed expert blob.
        std::vector<std::pair<void *, void *>> h2d_events;
    };
    std::vector<pending_profile_row> pending_rows;
    int last_pending_idx = -1;
    std::string current_request_phase = "unknown";

    // Async H2D buffer lifetime tracking. Events are borrowed from the
    // pending profile rows above; only pinned_buf ownership lives here.
    struct inflight_h2d_buffer {
        void * pinned_buf = nullptr;
        void * h2d_begin_event = nullptr;
        void * h2d_event = nullptr;
        int layer = -1;
        int expert = -1;
        int kind = -1;
        int slot = -1;
    };
    std::vector<inflight_h2d_buffer> inflight_h2d;
};

slot_pool_state & state() {
    static slot_pool_state instance;
    return instance;
}

int release_completed_h2d_buffers(slot_pool_state & s);
void wait_all_h2d_buffers(slot_pool_state & s);
void discard_pending_profile_rows(slot_pool_state & s);

uint32_t min_viable_slots(const manifest & mf) {
    const uint32_t n_experts = mf.n_experts_per_layer;
    if (n_experts == 0) {
        return 0;
    }

    uint32_t n_min = mf.n_expert_used > 0 ? mf.n_expert_used : 8;
    if (const char * env = std::getenv("LLAMA_MOE_MIN_SLOTS")) {
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(env, &end, 10);
        if (end != env && parsed > 0) {
            n_min = (uint32_t) parsed;
        }
    }

    return std::min(n_min, n_experts);
}

uint32_t compute_n_slots(const runtime_options & opts, const manifest & mf) {
    if (mf.n_experts_per_layer == 0) {
        return 0;
    }
    // Phase B default: full residency. Real budget-based slot computation lands in Phase C.
    uint32_t n = mf.n_experts_per_layer;

    if (opts.cache_vram_mb > 0 && mf.n_layers > 0 && mf.expert_blob_size_max > 0) {
        const uint64_t budget = opts.cache_vram_mb * 1024ull * 1024ull;
        const uint64_t per_slot = (uint64_t) EXPERT_KIND_COUNT * mf.expert_blob_size_max;
        const uint64_t per_layer_for_all = per_slot * mf.n_experts_per_layer;
        const uint64_t total_full = per_layer_for_all * mf.n_layers;
        if (total_full > budget) {
            const uint64_t per_layer_budget = budget / mf.n_layers;
            const uint64_t fits = per_layer_budget / per_slot;
            if (fits == 0) {
                n = 1;
            } else if (fits < mf.n_experts_per_layer) {
                n = (uint32_t) fits;
            }
        }
    }
    const uint32_t n_min = min_viable_slots(mf);
    if (n < n_min) {
        LLAMA_LOG_WARN("%s: MoE cache budget fits only %u slots/layer; reserving minimum viable %u slots/layer (top_k=%u)\n",
                __func__, n, n_min, mf.n_expert_used);
        n = n_min;
    }
    if (n < 1) n = 1;
    if (n > mf.n_experts_per_layer) n = mf.n_experts_per_layer;
    return n;
}

llm_tensor base_tensor_of(llm_tensor t) {
    return t;
}

bool debug_slot_fingerprint() {
    static const bool enabled = std::getenv("LLAMA_MOE_DEBUG_SLOT_FP") != nullptr;
    return enabled;
}

bool debug_d4_trace() {
    static const bool enabled =
        std::getenv("LLAMA_MOE_DEBUG_D4") != nullptr ||
        std::getenv("LLAMA_MOE_DEBUG_TRACE") != nullptr;
    return enabled;
}

bool debug_load_trace() {
    static const bool enabled =
        std::getenv("LLAMA_MOE_DEBUG_LOADS") != nullptr ||
        debug_d4_trace();
    return enabled;
}

} // namespace

void configure_slot_pool() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);

    wait_all_h2d_buffers(s);
    discard_pending_profile_rows(s);

    s.configured = false;
    s.n_slots = 0;
    s.slot_tensors.clear();

    if (!runtime_enabled()) {
        return;
    }
    const manifest & mf = get_manifest();
    if (!mf.present || mf.n_layers == 0 || mf.n_experts_per_layer == 0) {
        return;
    }

    s.n_slots = compute_n_slots(get_options(), mf);
    s.slot_tensors.assign(mf.n_layers, std::array<ggml_tensor *, EXPERT_KIND_COUNT>{nullptr, nullptr, nullptr});
    s.slot_table_tensors.assign(mf.n_layers, nullptr);
    s.topk_to_slot_table.clear();
    s.topk_to_slot_ids.clear();
    s.topk_to_logical.clear();
    s.all_slot_tables.clear();
    s.cache.assign(mf.n_layers, slot_pool_state::layer_cache{});
    for (auto & lc : s.cache) {
        lc.slot_to_expert.assign(s.n_slots, -1);
    }
    s.slot_table_host.assign(mf.n_experts_per_layer, 0);
    s.io_scratch.clear();
    if (s.io_fp) { fclose(s.io_fp); s.io_fp = nullptr; }
    s.cache_hits = 0;
    s.cache_misses = 0;
    s.topk_calls = 0;
    s.token_idx = 0;
    s.current_token_idx = 0;

    s.pending_rows.clear();
    s.last_pending_idx = -1;
    s.pred_dirty = false;

    // Phase E: init predictor from runtime options
    const runtime_options & opts = get_options();
    predictor_kind pk = predictor_kind::lru;
    try { pk = parse_predictor_kind(opts.predictor); } catch (...) {}
    s.pred = make_predictor(pk, (int) mf.n_layers, (int) mf.n_experts_per_layer);
    if (pk == predictor_kind::eamc && !opts.eamc_path.empty()) {
        s.pred->load(opts.eamc_path);
    }
    s.pred->begin_request();

    s.configured = true;

    LLAMA_LOG_INFO("%s: slot pool configured: %u logical MoE layers x %u experts -> %u slots/layer\n",
            __func__, mf.n_layers, mf.n_experts_per_layer, s.n_slots);
}

void reset_slot_pool() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    wait_all_h2d_buffers(s);
    discard_pending_profile_rows(s);
    s.configured = false;
    s.n_slots = 0;
    s.slot_tensors.clear();
    s.slot_table_tensors.clear();
    s.topk_to_slot_table.clear();
    s.topk_to_slot_ids.clear();
    s.topk_to_logical.clear();
    s.all_slot_tables.clear();
    s.cache.clear();
    s.slot_table_host.clear();
    s.io_scratch.clear();
    if (s.io_fp) { fclose(s.io_fp); s.io_fp = nullptr; }
    s.pred.reset();
    s.pred_dirty = false;
}

void slot_pool_reset_cache() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    wait_all_h2d_buffers(s);
    for (auto & lc : s.cache) {
        std::fill(lc.slot_to_expert.begin(), lc.slot_to_expert.end(), -1);
        lc.exp2slot.clear();
        lc.lru.clear();
        lc.lru_it.clear();
        lc.fingerprints.clear();
    }
    s.cache_hits = 0;
    s.cache_misses = 0;
    s.topk_calls = 0;
    s.token_idx = 0;
    s.current_token_idx = 0;
}

uint32_t n_slots_per_layer() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.n_slots;
}

uint32_t n_experts_per_layer() {
    const manifest & mf = get_manifest();
    return mf.n_experts_per_layer;
}

uint32_t recommended_ubatch(uint32_t requested, uint32_t n_expert_used, float safety) {
    const uint32_t n_slots = n_slots_per_layer();
    if (n_slots == 0 || n_expert_used == 0) {
        return requested;
    }
    if (!std::isfinite(safety) || safety <= 0.0f) {
        safety = 1.0f;
    }

    const double raw = std::floor((double) n_slots * (double) safety / (double) n_expert_used);
    uint32_t target = raw > 0.0 ? (uint32_t) raw : 1u;
    target = std::max<uint32_t>(1, target);
    target = std::min(target, requested);

    // Keep the auto setting on common graph/cache shapes. Explicit diagnostic
    // values can still be forced with LLAMA_MOE_STREAMING_UBATCH=N.
    static constexpr uint32_t allowed[] = {
        2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1,
    };
    for (uint32_t value : allowed) {
        if (target >= value) {
            return value;
        }
    }
    return target;
}

bool should_intercept(const LLM_TN_IMPL & tn, int * out_logical_layer, expert_kind * out_kind) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.configured || s.n_slots == 0) {
        return false;
    }
    if (tn.bid < 0) {
        return false;
    }
    expert_kind kind;
    switch (tn.tensor) {
        case LLM_TENSOR_FFN_GATE_EXPS:    kind = EXPERT_GATE; break;
        case LLM_TENSOR_FFN_UP_EXPS:      kind = EXPERT_UP;   break;
        case LLM_TENSOR_FFN_DOWN_EXPS:    kind = EXPERT_DOWN; break;
        // Fused gate+up: treat as gate for the registry (Phase B does not split it).
        case LLM_TENSOR_FFN_GATE_UP_EXPS: kind = EXPERT_GATE; break;
        default: return false;
    }
    const manifest & mf = get_manifest();
    const int logical = mf.logical_layer_of((uint32_t) tn.bid);
    if (logical < 0) {
        return false;
    }
    if (out_logical_layer) *out_logical_layer = logical;
    if (out_kind)          *out_kind          = kind;
    return true;
}

ggml_tensor * intercept_expert_tensor(
        llama_model_loader & ml,
        const llama_hparams & hparams,
        const buft_list_t * buft_list_layer,
        const LLM_TN_IMPL & tn,
        const std::initializer_list<int64_t> & ne,
        int flags) {
    (void) flags;

    int logical_layer = -1;
    expert_kind kind = EXPERT_GATE;
    if (!should_intercept(tn, &logical_layer, &kind)) {
        return nullptr;
    }

    static int intercept_cnt = 0;
    if (debug_d4_trace() && intercept_cnt < 4) {
        fprintf(stderr, "[moe-d4] intercept_expert_tensor ENTER #%d L%d k%d tn=%s\n",
                intercept_cnt, logical_layer, (int)kind, tn.str().c_str());
    }
    ++intercept_cnt;

    const std::string src_name = tn.str();
    ggml_tensor * src_meta = ml.get_tensor_meta(src_name.c_str());
    if (!src_meta) {
        // Weight is not in the GGUF; let the normal create_tensor path handle it.
        return nullptr;
    }

    const uint32_t n_slots = n_slots_per_layer();
    if (n_slots == 0) {
        return nullptr;
    }

    // ne provided by the model code is [d_in, d_out, n_expert]; substitute the
    // expert axis with the slot count.
    if (ne.size() < 3) {
        throw std::runtime_error("intercept_expert_tensor: expected 3D expert tensor shape");
    }
    const int64_t d_in  = ne.begin()[0];
    const int64_t d_out = ne.begin()[1];

    const int64_t n_expert = ne.begin()[2];
    const bool full_axis_guard = std::getenv("LLAMA_MOE_FULL_EXPERT_AXIS") != nullptr;
    const int64_t n_slot_axis = full_axis_guard ? n_expert : (int64_t) n_slots;
    if (full_axis_guard) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_FULL_EXPERT_AXIS is set; slot tensors use original expert axis as a guarded fallback\n",
                    __func__);
            warned = true;
        }
    }

    const std::string slot_name = src_name + ".slot";

    ggml_tensor * slot = ml.create_unfiled_tensor(
            hparams, buft_list_layer, slot_name, src_meta->type,
            GGML_OP_MUL_MAT_ID, { d_in, d_out, n_slot_axis });

    if (!slot) {
        throw std::runtime_error("intercept_expert_tensor: create_unfiled_tensor returned null");
    }

    // D-4 DEBUG: verify slot tensor strides match source metadata strides
    if (debug_d4_trace() && logical_layer == 0 && kind == EXPERT_GATE) {
        fprintf(stderr, "[moe-d4] slot tensor %s: type=%s ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu]\n",
                slot_name.c_str(), ggml_type_name(slot->type),
                (long long)slot->ne[0], (long long)slot->ne[1], (long long)slot->ne[2],
                slot->nb[0], slot->nb[1], slot->nb[2]);
        // Also dump source meta for comparison
        fprintf(stderr, "[moe-d4] src_meta %s: type=%s ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu]\n",
                src_name.c_str(), ggml_type_name(src_meta->type),
                (long long)src_meta->ne[0], (long long)src_meta->ne[1], (long long)src_meta->ne[2],
                src_meta->nb[0], src_meta->nb[1], src_meta->nb[2]);
    }

    // The original GGUF weight is no longer materialized via the loader; account for it.
    ml.mark_tensor_unloaded(src_name);

    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (logical_layer >= 0 && (size_t) logical_layer < s.slot_tensors.size()) {
        s.slot_tensors[logical_layer][kind] = slot;
    }

    // Create the per-layer slot_table tensor on the FIRST kind we see for this
    // layer. Shape [1, n_expert] I32 — same backend buffer as the slot weights.
    {
        size_t stvsz = s.slot_table_tensors.size();
        if (logical_layer >= 0 && (size_t) logical_layer < stvsz &&
            s.slot_table_tensors[logical_layer] == nullptr) {
            const int64_t n_expert = (int64_t) get_manifest().n_experts_per_layer;
            char st_name[64];
            std::snprintf(st_name, sizeof(st_name), "moe.slot_table.%d", logical_layer);
            ggml_tensor * stt = ml.create_unfiled_tensor(
                    hparams, buft_list_layer, st_name, GGML_TYPE_I32,
                    GGML_OP_GET_ROWS, { (int64_t) 1, n_expert });
            if (stt) {
                s.slot_table_tensors[logical_layer] = stt;
                if (debug_d4_trace()) {
                    fprintf(stderr, "[moe-d4] created %s L%d ne=[%lld,%lld] buf=%s has_buf=%d\n",
                            st_name, logical_layer,
                            (long long) stt->ne[0], (long long) stt->ne[1],
                            stt->buffer ? ggml_backend_buffer_name(stt->buffer) : "NULL",
                            stt->buffer ? 1 : 0);
                }
            } else {
                LLAMA_LOG_WARN("%s: failed to create %s for logical layer %d\n",
                        __func__, st_name, logical_layer);
            }
        } else if (logical_layer >= 0) {
            if (debug_d4_trace() && (size_t) logical_layer >= stvsz) {
                fprintf(stderr, "[moe-d4] slot_table_tensors OOB: L%d >= size=%zu\n",
                        logical_layer, stvsz);
            }
            // else: already created for this layer, silent skip
        }
    }

    static bool logged_first = false;
    if (!logged_first) {
        LLAMA_LOG_INFO("%s: intercepted %s -> %s [%lld, %lld, %u] dtype=%s\n",
                __func__, src_name.c_str(), slot_name.c_str(),
                (long long) d_in, (long long) d_out, n_slots, ggml_type_name(src_meta->type));
        logged_first = true;
    }
    return slot;
}

ggml_tensor * get_slot_tensor(int logical_layer, expert_kind kind) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (logical_layer < 0 || (size_t) logical_layer >= s.slot_tensors.size()) {
        return nullptr;
    }
    return s.slot_tensors[logical_layer][(int) kind];
}

ggml_tensor * get_slot_table_tensor(int logical_layer) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (logical_layer < 0 || (size_t) logical_layer >= s.slot_table_tensors.size()) {
        return nullptr;
    }
    return s.slot_table_tensors[logical_layer];
}

bool prefetch_all_experts() {
    if (debug_d4_trace()) {
        fprintf(stderr, "[moe-d4] prefetch_all_experts ENTER\n");
    }
    if (!runtime_enabled()) {
        if (debug_d4_trace()) {
            fprintf(stderr, "[moe-d4] prefetch_all_experts: runtime disabled, skip\n");
        }
        return true;
    }
    const manifest & mf = get_manifest();
    if (!mf.present || mf.experts.empty()) {
        return true;
    }

    const uint32_t n_slots = n_slots_per_layer();
    if (n_slots == 0) {
        return true;
    }

    // D-4: Zero-initialize all slot tensor buffers so that unused / excess
    // slots contain valid zero data instead of GPU garbage. The mmq kernel
    // (used by mul_mat_id) can access these slots when processing large
    // batches, and garbage data causes illegal memory accesses during
    // weight quantization (confirmed via compute-sanitizer). Zeroed slots
    // produce valid all-zero Q4_K blocks which the kernel handles safely.
    {
        const int64_t t_z = ggml_time_us();
        size_t n_zeroed = 0;
        ggml_tensor * last_slot = nullptr;
        if (debug_d4_trace()) {
            fprintf(stderr, "[moe-d4] prefetch_all_experts: starting zero-init of slot tensors...\n");
        }
        for (uint32_t li = 0; li < mf.n_layers; ++li) {
            for (int k = 0; k < EXPERT_KIND_COUNT; ++k) {
                ggml_tensor * slot = get_slot_tensor((int) li, (expert_kind) k);
                if (!slot || !slot->buffer) {
                    continue;
                }
                ggml_backend_tensor_memset(slot, 0, 0, ggml_nbytes(slot));
                last_slot = slot;
                ++n_zeroed;
            }
        }
        // The CUDA backend's memset_tensor uses cudaMemsetAsync on
        // cudaStreamPerThread WITHOUT syncing.  Force a sync by doing a
        // tiny tensor_set (which calls cudaStreamSynchronize) so that
        // subsequent compute-stream kernels see the zeroed data.
        if (last_slot) {
            uint8_t dummy = 0;
            ggml_backend_tensor_set(last_slot, &dummy, 0, 1);
        }
        if (debug_d4_trace()) {
            fprintf(stderr, "[moe-d4] prefetch_all_experts: zeroed %zu slot tensors in %.2f ms (incl sync)\n",
                    n_zeroed, (ggml_time_us() - t_z) / 1000.0);
        }
    }

    if (n_slots < mf.n_experts_per_layer) {
        if (debug_d4_trace()) {
            fprintf(stderr, "[moe-d4] prefetch_all_experts: skipping prefetch (n_slots=%u < n_experts=%u; streaming mode)\n",
                    n_slots, mf.n_experts_per_layer);
        }
        return true;
    }

    FILE * fp = fopen(mf.source_path.c_str(), "rb");
    if (!fp) {
        LLAMA_LOG_ERROR("%s: failed to open %s for expert prefetch\n", __func__, mf.source_path.c_str());
        return false;
    }

    const int64_t t0 = ggml_time_us();
    uint64_t bytes_read = 0;
    size_t   n_reads    = 0;
    std::vector<uint8_t> buf;

    bool ok = true;
    for (uint32_t li = 0; li < mf.n_layers && ok; ++li) {
        for (int k = 0; k < EXPERT_KIND_COUNT && ok; ++k) {
            ggml_tensor * slot = get_slot_tensor((int) li, (expert_kind) k);
            if (!slot) {
                continue; // kind not present (e.g. fused gate_up uses only one slot)
            }
            if (!slot->buffer) {
                // Phase M.1: layer's MoE block was not part of any built
                // compute graph, so the scheduler never bound a backend
                // buffer. Skip silently — the slot tensor cannot be a
                // mul_mat_id consumer either.
                continue;
            }
            const size_t stride = slot->nb[2];
            for (uint32_t e = 0; e < mf.n_experts_per_layer; ++e) {
                const expert_record & rec = mf.at(li, e, (expert_kind) k);
                if (rec.size == 0) {
                    continue;
                }
                if (rec.size > stride) {
                    LLAMA_LOG_ERROR("%s: expert size %llu > slot stride %zu (li=%u e=%u kind=%s)\n",
                            __func__, (unsigned long long) rec.size, stride, li, e,
                            expert_kind_name((expert_kind) k));
                    ok = false;
                    break;
                }
                const uint64_t file_offset = mf.data_offset + rec.rel_offset;
                if (moe_fseek(fp, (int64_t) file_offset, SEEK_SET) != 0) {
                    LLAMA_LOG_ERROR("%s: seek to %llu failed\n", __func__, (unsigned long long) file_offset);
                    ok = false;
                    break;
                }
                if (buf.size() < rec.size) {
                    buf.resize(rec.size);
                }
                const size_t got = fread(buf.data(), 1, rec.size, fp);
                if (got != rec.size) {
                    LLAMA_LOG_ERROR("%s: short read at li=%u e=%u kind=%s: got %zu of %llu\n",
                            __func__, li, e, expert_kind_name((expert_kind) k),
                            got, (unsigned long long) rec.size);
                    ok = false;
                    break;
                }
                ggml_backend_tensor_set(slot, buf.data(), (size_t) e * stride, rec.size);
                bytes_read += rec.size;
                ++n_reads;
            }
        }
        if (ok && (li + 1) % 8 == 0) {
            LLAMA_LOG_INFO("%s: prefetched layer %u/%u (%.2f GiB so far)\n",
                    __func__, li + 1, mf.n_layers, bytes_read / 1024.0 / 1024.0 / 1024.0);
        }
    }

    fclose(fp);

    if (!ok) {
        return false;
    }

    const double elapsed_s = (ggml_time_us() - t0) / 1e6;
    const double gib       = bytes_read / 1024.0 / 1024.0 / 1024.0;
    LLAMA_LOG_INFO("%s: prefetched %zu expert slices, %.2f GiB in %.2f s (%.2f GiB/s)\n",
            __func__, n_reads, gib, elapsed_s, elapsed_s > 0 ? gib / elapsed_s : 0.0);
    return true;
}

void register_slot_table_for_topk(int logical_layer, ggml_tensor * topk, ggml_tensor * slot_table) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.configured || !topk || !slot_table) {
        if (debug_d4_trace()) {
            fprintf(stderr, "[moe-d4] register_slot_table_for_topk SKIP: cfg=%d topk=%p st=%p L=%d\n",
                    s.configured ? 1 : 0, (void*)topk, (void*)slot_table, logical_layer);
        }
        return;
    }
    s.topk_to_slot_table[topk]   = slot_table;
    s.topk_to_logical[topk]      = logical_layer;
    s.all_slot_tables.push_back(slot_table);
    static int reg_cnt = 0;
    if (debug_d4_trace() && reg_cnt < 4) {
        fprintf(stderr, "[moe-d4] register_slot_table_for_topk #%d L=%d topk=%p st=%p st->buf=%s\n",
                reg_cnt, logical_layer, (void*)topk, (void*)slot_table,
                slot_table->buffer ? ggml_backend_buffer_name(slot_table->buffer) : "NULL");
    }
    ++reg_cnt;
}

void register_slot_ids_for_topk(int logical_layer, ggml_tensor * topk, ggml_tensor * slot_ids) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.configured || !topk || !slot_ids) {
        if (std::getenv("LLAMA_MOE_DEBUG_SLOT_IDS")) {
            fprintf(stderr, "[moe-d4] register_slot_ids_for_topk SKIP: cfg=%d topk=%p ids=%p L=%d\n",
                    s.configured ? 1 : 0, (void*) topk, (void*) slot_ids, logical_layer);
        }
        return;
    }
    s.topk_to_slot_ids[topk] = slot_ids;
    s.topk_to_logical[topk]  = logical_layer;
    static int reg_cnt = 0;
    if (std::getenv("LLAMA_MOE_DEBUG_SLOT_IDS") && reg_cnt < 4) {
        fprintf(stderr, "[moe-d4] register_slot_ids_for_topk #%d L=%d topk=%p ids=%p ids->buf=%s\n",
                reg_cnt, logical_layer, (void*) topk, (void*) slot_ids,
                slot_ids->buffer ? ggml_backend_buffer_name(slot_ids->buffer) : "NULL");
    }
    ++reg_cnt;
}

void reset_graph_state() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.topk_to_slot_table.clear();
    s.topk_to_slot_ids.clear();
    s.topk_to_logical.clear();
    s.all_slot_tables.clear();
}

void populate_slot_tables_identity() {
    auto & s = state();
    std::vector<ggml_tensor *> tensors;
    uint32_t n_experts = 0;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.configured) {
            return;
        }
        tensors = s.all_slot_tables;
    }
    n_experts = get_manifest().n_experts_per_layer;
    if (n_experts == 0) {
        return;
    }
    std::vector<int32_t> identity(n_experts);
    for (uint32_t i = 0; i < n_experts; ++i) {
        identity[i] = (int32_t) i;
    }
    const size_t nbytes = identity.size() * sizeof(int32_t);
    size_t n_written = 0;
    for (ggml_tensor * t : tensors) {
        if (!t) {
            continue;
        }
        // Tensor may not be allocated yet if graph_reuse path hasn't run alloc.
        if (!t->buffer) {
            continue;
        }
        ggml_backend_tensor_set(t, identity.data(), 0, nbytes);
        ++n_written;
    }
    static bool logged = false;
    if (!logged && n_written > 0) {
        LLAMA_LOG_INFO("%s: wrote identity slot_table to %zu MoE layers (n_expert=%u)\n",
                __func__, n_written, n_experts);
        logged = true;
    }
}

bool streaming_mode() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.configured) return false;
    return s.n_slots < get_manifest().n_experts_per_layer;
}

namespace {

struct load_stats {
    int64_t ssd_read_us = 0;
    int64_t h2d_us = 0;
    uint64_t ssd_bytes = 0;
    uint64_t ssd_reads = 0;
};

static int64_t elapsed_us(std::chrono::steady_clock::time_point start,
                          std::chrono::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

int release_completed_h2d_buffers(slot_pool_state & s) {
    int released = 0;
    auto it = s.inflight_h2d.begin();
    while (it != s.inflight_h2d.end()) {
        if (!it->h2d_event || io_event_query(it->h2d_event)) {
            if (it->pinned_buf) {
                io_release_buffer(it->pinned_buf);
            }
            it = s.inflight_h2d.erase(it);
            ++released;
        } else {
            ++it;
        }
    }
    return released;
}

void wait_all_h2d_buffers(slot_pool_state & s) {
    for (auto & b : s.inflight_h2d) {
        if (b.h2d_event) {
            io_event_sync(b.h2d_event);
        }
        if (b.pinned_buf) {
            io_release_buffer(b.pinned_buf);
            b.pinned_buf = nullptr;
        }
    }
    s.inflight_h2d.clear();
}

void discard_pending_profile_rows(slot_pool_state & s) {
    for (auto & p : s.pending_rows) {
        if (p.compute_begin_event) io_event_release(p.compute_begin_event);
        if (p.compute_end_event)   io_event_release(p.compute_end_event);
        for (auto & ev_pair : p.h2d_events) {
            if (ev_pair.first)  io_event_release(ev_pair.first);
            if (ev_pair.second) io_event_release(ev_pair.second);
        }
    }
    s.pending_rows.clear();
    s.last_pending_idx = -1;
}

void set_tensor_for_compute(slot_pool_state & s, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (s.compute_backend && tensor && tensor->buffer &&
            ggml_backend_supports_buft(s.compute_backend, ggml_backend_buffer_get_type(tensor->buffer))) {
        ggml_backend_tensor_set_async(s.compute_backend, tensor, data, offset, size);
    } else {
        ggml_backend_tensor_set(tensor, data, offset, size);
    }
}

// Load one expert's three blobs (gate, up, down) from disk into the chosen
// slot index. Caller holds s.mutex.
bool load_expert_into_slot(slot_pool_state & s, const manifest & mf,
                           uint32_t logical_layer, int32_t expert, int32_t slot,
                           load_stats * stats = nullptr) {
    if (!s.io_fp) {
        s.io_fp = fopen(mf.source_path.c_str(), "rb");
        if (!s.io_fp) {
            LLAMA_LOG_ERROR("moe_eval_callback: fopen failed for %s\n", mf.source_path.c_str());
            return false;
        }
    }
    for (int k = 0; k < EXPERT_KIND_COUNT; ++k) {
        ggml_tensor * slot_tensor = s.slot_tensors[logical_layer][k];
        if (!slot_tensor) continue;
        const expert_record & rec = mf.at(logical_layer, (uint32_t) expert, (expert_kind) k);
        if (s.io_scratch.size() < rec.size) s.io_scratch.resize(rec.size);
        const auto read_start = std::chrono::steady_clock::now();
        if (moe_fseek(s.io_fp, (long long)(mf.data_offset + rec.rel_offset), SEEK_SET) != 0) {
            LLAMA_LOG_ERROR("moe_eval_callback: seek failed L%u e%d k%d\n", logical_layer, expert, k);
            return false;
        }
        size_t got = fread(s.io_scratch.data(), 1, rec.size, s.io_fp);
        const auto read_end = std::chrono::steady_clock::now();
        if (got != rec.size) {
            LLAMA_LOG_ERROR("moe_eval_callback: short read L%u e%d k%d got=%zu want=%zu\n",
                    logical_layer, expert, k, got, (size_t) rec.size);
            return false;
        }
        if (stats) {
            stats->ssd_read_us += elapsed_us(read_start, read_end);
            stats->ssd_bytes += rec.size;
            ++stats->ssd_reads;
        }
        const size_t stride = slot_tensor->nb[2];
        if (rec.size > stride) {
            LLAMA_LOG_ERROR("load_expert: rec.size=%llu > stride=%zu (L%u e%d k%d)\n",
                    (unsigned long long) rec.size, stride, logical_layer, expert, k);
            return false;
        }
        const size_t buf_size = ggml_backend_buffer_get_size(slot_tensor->buffer);
        const size_t write_off = (size_t) slot * stride;
        const size_t tensor_off = (size_t)((char *)slot_tensor->data - (char *)ggml_backend_buffer_get_base(slot_tensor->buffer));
        if (tensor_off + write_off + rec.size > buf_size) {
            LLAMA_LOG_ERROR("load_expert: OOB write tensor_off=%zu slot_off=%zu rec=%llu buf_size=%zu (L%u e%d k%d slot=%d)\n",
                    tensor_off, write_off, (unsigned long long) rec.size, buf_size, logical_layer, expert, k, slot);
            return false;
        }
        // D-3: dump all loads (capped) so we can verify slot_off math at slots beyond ~16.
        static int dump = 0;
        if (debug_load_trace() && dump < 3000) {
            ++dump;
            const char * bname = slot_tensor->buffer ? ggml_backend_buffer_name(slot_tensor->buffer) : "<null>";
            fprintf(stderr, "[moe-d3] load#%d L%u e%d k%d slot=%d rec.size=%llu stride=%zu write_off=%zu tensor_off=%zu buf_size=%zu buf=%s data=%p\n",
                    dump, logical_layer, expert, k, slot,
                    (unsigned long long) rec.size, stride, write_off, tensor_off, buf_size, bname, slot_tensor->data);
            fflush(stderr);
        }
        const auto h2d_start = std::chrono::steady_clock::now();
        set_tensor_for_compute(s, slot_tensor, s.io_scratch.data(), write_off, rec.size);
        const auto h2d_end = std::chrono::steady_clock::now();
        if (stats) {
            stats->h2d_us += elapsed_us(h2d_start, h2d_end);
        }
    }
    return true;
}

// Mark an expert as most-recent in the LRU. Caller holds s.mutex.
void lru_touch(slot_pool_state::layer_cache & lc, int32_t expert) {
    auto it = lc.lru_it.find(expert);
    if (it != lc.lru_it.end()) {
        lc.lru.erase(it->second);
    }
    lc.lru.push_front(expert);
    lc.lru_it[expert] = lc.lru.begin();
}

} // namespace

bool moe_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    const auto callback_start = std::chrono::steady_clock::now();
    (void) user_data;
    if (!t || !t->name[0]) {
        return true;
    }
    if (std::getenv("LLAMA_MOE_DEBUG_CB_PROOF")) {
        static int cb_proof_n = 0;
        if (cb_proof_n < 5) {
            fprintf(stderr, "[cb-proof] cb#%d ask=%d t=%s\n", cb_proof_n++, (int)ask, t->name);
        }
    }
    if (std::getenv("LLAMA_MOE_DEBUG_CB_NOP")) {
        return true;
    }

    auto & s = state();

    if (ask) {
        std::lock_guard<std::mutex> lock(s.mutex);
        return s.configured && s.topk_to_logical.find(t) != s.topk_to_logical.end();
    }

    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.configured) return true;
    release_completed_h2d_buffers(s);

    // Fast lookup: only react to topk tensors we previously registered.
    auto stt_it = s.topk_to_slot_table.find(t);
    ggml_tensor * stt = stt_it == s.topk_to_slot_table.end() ? nullptr : stt_it->second;
    auto slot_ids_it = s.topk_to_slot_ids.find(t);
    ggml_tensor * slot_ids_tensor = slot_ids_it == s.topk_to_slot_ids.end() ? nullptr : slot_ids_it->second;
    auto log_it = s.topk_to_logical.find(t);
    if (log_it == s.topk_to_logical.end()) return true;
    const int logical = log_it->second;

    // Phase I: the scheduler has just computed and synchronized the graph
    // segment ending at this top-k node. Close the previous MoE layer's
    // compute interval here, before we start staging the current layer's
    // experts.
    if (s.compute_backend && s.last_pending_idx >= 0 &&
        (size_t) s.last_pending_idx < s.pending_rows.size()) {
        auto & p = s.pending_rows[s.last_pending_idx];
        if (!p.compute_end_event) {
            void * ev = io_event_acquire();
            if (ev && io_record_on_compute(s.compute_backend, ev)) {
                p.compute_end_event = ev;
            } else if (ev) {
                io_event_release(ev);
            }
        }
    }

    const manifest & mf = get_manifest();
    if (logical < 0 || (size_t) logical >= s.cache.size()) return true;
    auto & lc = s.cache[logical];

    // Phase M.6: read slot hashes at callback START (before any modification).
    // When set, compare with the END hash saved from the previous visit for
    // this (layer,slot).  A CHANGED-BEFORE-WRITE tag means the GPU buffer was
    // corrupted between callbacks (not by us).
    static const bool diag_slot_start = std::getenv("LLAMA_MOE_DEBUG_SLOT_START") != nullptr;
    if (diag_slot_start && logical < 3) {
        constexpr int kMonSlots[] = {0, 1, 2, 3, 8};
        static uint64_t prev_end_hash[3][5];
        static bool     prev_end_valid[3][5] = {};
        ggml_tensor * st = nullptr;
        for (int k = 0; k < EXPERT_KIND_COUNT; ++k) {
            if (s.slot_tensors[logical][k]) { st = s.slot_tensors[logical][k]; break; }
        }
        if (st && st->buffer) {
            const size_t buf_sz = ggml_backend_buffer_get_size(st->buffer);
            const size_t stride = (size_t) st->nb[2];
            uint8_t buf[1024];
            for (int si = 0; si < 5; ++si) {
                int s_idx = kMonSlots[si];
                if ((uint32_t) s_idx >= s.n_slots) continue;
                size_t off = (size_t) s_idx * stride;
                size_t sz  = stride < sizeof(buf) ? stride : sizeof(buf);
                if (off + sz > buf_sz) continue;
                ggml_backend_tensor_get(st, buf, off, sz);
                uint64_t h = 0xcbf29ce484222325ULL;
                for (size_t i = 0; i < sz; ++i) { h ^= buf[i]; h *= 0x100000001b3ULL; }
                const char * tag = "";
                if (prev_end_valid[logical][si] && h != prev_end_hash[logical][si]) {
                    tag = " CORRUPTED-SINCE-LAST-END";
                }
                fprintf(stderr, "[slot-hash-start] tok=%llu L%d slot=%d hash=0x%016llx%s\n",
                        (unsigned long long) s.token_idx, logical, s_idx,
                        (unsigned long long) h, tag);
            }
        }
    }

    // Read top-k IDs (D2H). Shape: [n_expert_used, n_tokens] I32.
    const int64_t n_elem = ggml_nelements(t);
    const int64_t n_tokens = t->ne[1];
    std::vector<int32_t> ids((size_t) n_elem);
    const auto topk_d2h_start = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
    const auto topk_d2h_end = std::chrono::steady_clock::now();
    const int64_t topk_d2h_us = elapsed_us(topk_d2h_start, topk_d2h_end);

    const char * phase = n_tokens > 1 ? "prefill" : "decode";
    if (logical == 0) {
        s.current_token_idx = s.token_idx;
    }
    const uint64_t row_token_idx = s.current_token_idx;

    // Collect unique experts.
    std::unordered_set<int32_t> uniq;
    uniq.reserve(ids.size());
    for (int32_t e : ids) {
        if (e >= 0 && (uint32_t) e < mf.n_experts_per_layer) uniq.insert(e);
    }

    int64_t pred_observe_us = 0;
    int64_t pred_score_us = 0;

    // Phase E-3: observe expert usage for predictor.
    {
        std::vector<int> obs(uniq.begin(), uniq.end());
        const auto pred_start = std::chrono::steady_clock::now();
        s.pred->observe(logical, obs);
        const auto pred_end = std::chrono::steady_clock::now();
        pred_observe_us += elapsed_us(pred_start, pred_end);
    }

    // Architectural constraint: all unique experts selected this batch must fit
    // resident in the per-layer cache simultaneously. Otherwise LRU would evict
    // experts that are still needed by downstream mul_mat_id reads in the same
    // forward pass, producing corrupted weights / CUDA OOB. Surface this as a
    // clear hard error rather than crashing inside CUDA.
    if (uniq.size() > s.n_slots) {
        static int err_logged = 0;
        if (err_logged < 3) {
            LLAMA_LOG_ERROR("moe_eval_callback: n_uniq=%zu exceeds n_slots=%u "
                    "(layer=%d, n_tokens*n_expert_used=%lld). "
                    "Increase --moe-cache-vram-mb so n_slots >= worst-case unique experts per batch, "
                    "or reduce batch size.\n",
                    uniq.size(), s.n_slots, logical, (long long) n_elem);
            ++err_logged;
        }
        GGML_ABORT("MoE-offload: per-batch unique experts exceeds slot count");
    }

    // Ensure residency. Process hits first (just touch), then misses (evict + load).
    int dbg_hits = 0, dbg_misses = 0, dbg_evictions = 0, dbg_free_alloc = 0;

    // Phase M.4 diagnostic: optionally invalidate the cache before this
    // callback so every uniq expert is treated as a MISS and freshly
    // re-loaded. If this collapses streaming drift to ~0, the bug is in
    // how HIT slot data is consumed across callbacks (likely a missing
    // cross-step H2D-event wait on the compute stream).
    {
        static const bool diag_no_hit = std::getenv("LLAMA_MOE_DEBUG_NO_HIT") != nullptr;
        if (diag_no_hit) {
            lc.exp2slot.clear();
            lc.lru.clear();
            lc.lru_it.clear();
            std::fill(lc.slot_to_expert.begin(), lc.slot_to_expert.end(), -1);
        }
    }

    // Debug-only slot integrity detector. This does not repair cache state;
    // production correctness must come from stable slot storage and remapping.
    if (debug_slot_fingerprint()) {
        ggml_tensor * fp_tensor = s.slot_tensors[logical][EXPERT_GATE];
        if (fp_tensor && fp_tensor->buffer) {
            const size_t stride   = (size_t) fp_tensor->nb[2];
            const size_t buf_size = ggml_backend_buffer_get_size(fp_tensor->buffer);
            uint8_t fpbuf[1024];
            size_t sz = stride < sizeof(fpbuf) ? stride : sizeof(fpbuf);
            std::vector<int32_t> corrupted;
            for (int32_t e : uniq) {
                auto it = lc.exp2slot.find(e);
                if (it == lc.exp2slot.end()) continue; // MISS, no fingerprint to check
                auto fpit = lc.fingerprints.find(e);
                if (fpit == lc.fingerprints.end()) continue; // no fingerprint stored yet

                int32_t sl = it->second;
                size_t off = (size_t) sl * stride;
                if (off + sz > buf_size) continue;
                ggml_backend_tensor_get(fp_tensor, fpbuf, off, sz);
                uint64_t h = 0xcbf29ce484222325ULL;
                for (size_t i = 0; i < sz; ++i) { h ^= fpbuf[i]; h *= 0x100000001b3ULL; }
                if (h != fpit->second) {
                    corrupted.push_back(e);
                }
            }
            if (!corrupted.empty()) {
                static int fp_corrupt_log = 0;
                for (int32_t e : corrupted) {
                    if (fp_corrupt_log < 10) {
                        fprintf(stderr, "[slot-fp] tok=%llu L%d expert=%d fingerprint mismatch\n",
                                (unsigned long long) row_token_idx, logical, e);
                        ++fp_corrupt_log;
                    }
                }
            }
        }
    }

    for (int32_t e : uniq) {
        auto it = lc.exp2slot.find(e);
        if (it != lc.exp2slot.end()) {
            lru_touch(lc, e);
            ++s.cache_hits;
            ++dbg_hits;
        }
    }

    load_stats layer_load_stats;
    int misses_loaded = 0;
    predictor_score_stats pred_stats;

    auto add_pred_stats = [](predictor_score_stats & dst, const predictor_score_stats & src) {
        dst.eamc_rows_scored += src.eamc_rows_scored;
        dst.eamc_cosine_us += src.eamc_cosine_us;
        dst.eamc_score_materialize_us += src.eamc_score_materialize_us;
        dst.eamc_score_cache_hits += src.eamc_score_cache_hits;
        dst.eamc_score_cache_misses += src.eamc_score_cache_misses;
    };

    // Phase H: per-miss bookkeeping needed when the worker completion comes
    // back out of submission order. We carry the slot tensor pointer (so the
    // fallback sync H2D path can call ggml_backend_tensor_set) and the
    // write_off, keyed by pinned_buf because that's the unique handle the
    // worker echoes back to us.
    struct miss_meta {
        ggml_tensor * slot_tensor;
        size_t        write_off;
    };
    struct miss_blob {
        int32_t       expert;
        int           kind;
        int32_t       slot;
        ggml_tensor * slot_tensor;
        size_t        write_off;
        size_t        blob_size;
        uint64_t      file_offset;
        char *        gpu_dst;
    };
    std::unordered_map<void *, miss_meta> miss_lookup;
    std::vector<miss_blob> miss_blobs;
    int submitted_misses = 0;
    int completed_misses = 0;
    int64_t stall_us = 0;

    // Phase I: per-miss h2d timing events stashed for elapsed-time query at
    // end_request. Owned by the pending_profile_row we create below.
    std::vector<std::pair<void *, void *>> h2d_events_for_row;

    // Phase L.2: track experts reserved in THIS callback so the eviction
    // path cannot pick them as victims. Without this, the LRU/EAMC score
    // function may return a lower score for a just-reserved expert than
    // for an established resident, causing us to evict a slot whose H2D
    // is still in flight (corrupting weights for the consumer matmul of
    // the very same forward pass).
    std::unordered_set<int32_t> reserved_this_call;

    // Phase M.3: every expert in `uniq` is a top-k consumer of this very
    // forward pass — it MUST NOT be evicted by a later miss in the same
    // callback. Previously only just-loaded misses were protected, which
    // meant a HIT expert could have its slot stolen; the slot_table fill
    // below would then read `lc.exp2slot[hit]` (now erased) and silently
    // default-construct to 0, sending mul_mat_id to the wrong slot. This
    // produced a deterministic ~0.46 max|Δlogit| drift versus full
    // residency whenever the cache was small enough to force evictions.
    for (int32_t e : uniq) {
        reserved_this_call.insert(e);
    }

    for (int32_t e : uniq) {
        if (lc.exp2slot.count(e)) continue;

        // Pick a slot. Free slots first, else LRU victim.
        int32_t slot = -1;
        for (uint32_t si = 0; si < s.n_slots; ++si) {
            if (lc.slot_to_expert[si] < 0) { slot = (int32_t) si; break; }
        }
        if (slot >= 0) ++dbg_free_alloc;
        if (slot < 0) {
            // Phase E: use predictor.score for eviction (lower = evict).
            // Phase L.2: skip experts reserved earlier in this same
            // callback — their slots hold in-flight H2D data.
            float best_score = 1e30f;
            int32_t best_victim = -1;
            const auto pred_start = std::chrono::steady_clock::now();
            for (const auto & [exp, sl] : lc.exp2slot) {
                if (reserved_this_call.count(exp)) continue;
                float sc = s.pred->score(logical, exp);
                if (sc < best_score) { best_score = sc; best_victim = exp; }
            }
            const auto pred_end = std::chrono::steady_clock::now();
            pred_score_us += elapsed_us(pred_start, pred_end);
            add_pred_stats(pred_stats, s.pred->take_score_stats());
            if (best_victim < 0) {
                // Fall back to LRU tail, but still skip reserved-this-call.
                for (auto it = lc.lru.rbegin(); it != lc.lru.rend(); ++it) {
                    if (!reserved_this_call.count(*it)) { best_victim = *it; break; }
                }
            }
            if (best_victim < 0) {
                LLAMA_LOG_ERROR("moe_eval_callback: no evictable victim at L%d "
                        "(n_slots=%u, uniq=%zu, reserved_this_call=%zu). "
                        "Increase --moe-cache-vram-mb.\n",
                        logical, s.n_slots, uniq.size(), reserved_this_call.size());
                GGML_ABORT("MoE-offload: slot pool exhausted by in-flight reservations");
            }
            lc.lru.erase(lc.lru_it[best_victim]);
            lc.lru_it.erase(best_victim);
            slot = lc.exp2slot[best_victim];
            lc.exp2slot.erase(best_victim);
            lc.fingerprints.erase(best_victim);
            lc.slot_to_expert[slot] = -1;
            ++dbg_evictions;
        }

        // Queue one blob load per expert kind. Submission happens below in a
        // chunked submit/drain loop, so larger ubatches do not require enough
        // pinned buffers for every miss in this layer at once.
        for (int k = 0; k < EXPERT_KIND_COUNT; ++k) {
            ggml_tensor * slot_tensor = s.slot_tensors[logical][k];
            if (!slot_tensor) continue;
            const expert_record & rec = mf.at((uint32_t) logical, (uint32_t) e, (expert_kind) k);

            const size_t stride = slot_tensor->nb[2];
            if (rec.size > stride) {
                LLAMA_LOG_ERROR("moe_eval_callback: rec.size=%llu > stride=%zu (L%d e%d k%d)\n",
                        (unsigned long long) rec.size, stride, logical, e, k);
                GGML_ABORT("MoE-offload: expert blob too large for slot stride");
            }
            const size_t buf_size = ggml_backend_buffer_get_size(slot_tensor->buffer);
            const size_t write_off = (size_t) slot * stride;
            const size_t tensor_off = (size_t)((char *)slot_tensor->data -
                                               (char *)ggml_backend_buffer_get_base(slot_tensor->buffer));
            if (tensor_off + write_off + rec.size > buf_size) {
                LLAMA_LOG_ERROR("moe_eval_callback: OOB write tensor_off=%zu slot_off=%zu rec=%llu buf_size=%zu (L%d e%d k%d slot=%d)\n",
                        tensor_off, write_off, (unsigned long long) rec.size, buf_size, logical, e, k, (int) slot);
                GGML_ABORT("MoE-offload: slot OOB");
            }

            miss_blobs.push_back(miss_blob{
                e,
                k,
                slot,
                slot_tensor,
                write_off,
                (size_t) rec.size,
                mf.data_offset + rec.rel_offset,
                (char *) slot_tensor->data + write_off,
            });
        }

        // Reserve the slot for this expert immediately so a later miss in the
        // same batch does not pick it as a victim. The actual bytes land
        // asynchronously below.
        lc.slot_to_expert[slot] = e;
        lc.exp2slot[e] = slot;
        lru_touch(lc, e);
        reserved_this_call.insert(e);
        ++s.cache_misses;
        ++misses_loaded;
        ++dbg_misses;
    }

    if (std::getenv("LLAMA_MOE_DEBUG_EVICT")) {
        fprintf(stderr, "[evict] tok=%llu L%d uniq=%zu hits=%d misses=%d free_alloc=%d evictions=%d n_slots=%u\n",
                (unsigned long long) row_token_idx, logical, uniq.size(),
                dbg_hits, dbg_misses, dbg_free_alloc, dbg_evictions, s.n_slots);
    }

    std::sort(miss_blobs.begin(), miss_blobs.end(),
            [](const miss_blob & a, const miss_blob & b) {
                return a.file_offset < b.file_offset;
            });

    auto drain_completed = [&]() -> int {
        auto completions = io_drain_completed();
        const int n = (int) completions.size();
        for (auto & c : completions) {
            bool release_pinned_now = true;
            if (!c.ok) {
                LLAMA_LOG_ERROR("moe_eval_callback: I/O failed loading L%d e%d k%d slot=%d "
                        "offset=%llu bytes=%zu got=%zu errno=%d\n",
                        c.layer, c.expert, c.kind, c.slot,
                        (unsigned long long) c.file_offset,
                        c.blob_size, c.bytes_read, c.io_error);
                io_release_buffer(c.pinned_buf);
                miss_lookup.erase(c.pinned_buf);
                ++completed_misses;
                GGML_ABORT("MoE-offload: expert blob I/O failed");
            }

            layer_load_stats.ssd_read_us += c.ssd_read_us;
            layer_load_stats.ssd_bytes   += c.blob_size;
            ++layer_load_stats.ssd_reads;

            if (c.h2d_event) {
                // Async path: tell the compute stream to wait on the H2D end
                // event before consuming slot weights. Recycle the pinned
                // staging buffer later, once the end event has completed.
                const bool compute_wait_ok = io_compute_wait(s.compute_backend, c.h2d_event);
                h2d_events_for_row.emplace_back(c.h2d_begin_event, c.h2d_event);
                if (compute_wait_ok) {
                    s.inflight_h2d.push_back(slot_pool_state::inflight_h2d_buffer{
                        c.pinned_buf,
                        c.h2d_begin_event,
                        c.h2d_event,
                        c.layer,
                        c.expert,
                        c.kind,
                        c.slot,
                    });
                    release_pinned_now = false;
                } else {
                    const auto h2d_wait_start = std::chrono::steady_clock::now();
                    io_event_sync(c.h2d_event);
                    const auto h2d_wait_end = std::chrono::steady_clock::now();
                    stall_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            h2d_wait_end - h2d_wait_start).count();
                }
            } else {
                // Fallback: CUDA unavailable or io_h2d_async failed. Look up
                // the slot tensor by the pinned-buffer handle and do a sync
                // ggml_backend_tensor_set.
                auto it = miss_lookup.find(c.pinned_buf);
                if (it == miss_lookup.end()) {
                    LLAMA_LOG_ERROR("moe_eval_callback: completion without lookup entry (L%d)\n", logical);
                } else {
                    const auto h2d_start = std::chrono::steady_clock::now();
                    set_tensor_for_compute(s, it->second.slot_tensor, c.pinned_buf,
                                           it->second.write_off, c.blob_size);
                    const auto h2d_end = std::chrono::steady_clock::now();
                    layer_load_stats.h2d_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            h2d_end - h2d_start).count();
                }
            }

            if (debug_slot_fingerprint() && c.kind == EXPERT_GATE && c.blob_size > 0) {
                size_t fp_sz = c.blob_size < 1024 ? (size_t) c.blob_size : 1024;
                uint64_t fp = 0xcbf29ce484222325ULL;
                const uint8_t * src = (const uint8_t *) c.pinned_buf;
                for (size_t i = 0; i < fp_sz; ++i) { fp ^= src[i]; fp *= 0x100000001b3ULL; }
                lc.fingerprints[c.expert] = fp;
            }

            if (release_pinned_now) {
                io_release_buffer(c.pinned_buf);
            }
            miss_lookup.erase(c.pinned_buf);
            ++completed_misses;
        }
        release_completed_h2d_buffers(s);
        return n;
    };

    if (!miss_blobs.empty()) {
        static const bool diag_no_async = std::getenv("LLAMA_MOE_DEBUG_NO_ASYNC") != nullptr;
        size_t next_miss = 0;

        while (next_miss < miss_blobs.size() || completed_misses < submitted_misses) {
            bool made_progress = false;

            while (next_miss < miss_blobs.size()) {
                void * pinned = io_try_acquire_buffer();
                if (!pinned) {
                    if (release_completed_h2d_buffers(s) > 0) {
                        made_progress = true;
                        continue;
                    }
                    break;
                }

                const miss_blob & blob = miss_blobs[next_miss];
                io_request req{};
                req.layer       = logical;
                req.expert      = blob.expert;
                req.kind        = blob.kind;
                req.slot        = blob.slot;
                req.pinned_buf  = pinned;
                req.blob_size   = blob.blob_size;
                req.file_offset = blob.file_offset;
                req.gpu_dst     = blob.gpu_dst;
                req.h2d         = !diag_no_async && (s.compute_backend != nullptr);
                req.h2d_event   = nullptr;
                req.h2d_begin_event = nullptr;
                req.ssd_read_us = 0;

                if (!io_submit(req)) {
                    io_release_buffer(pinned);
                    break;
                }

                miss_lookup[pinned] = miss_meta{ blob.slot_tensor, blob.write_off };
                ++submitted_misses;
                ++next_miss;
                made_progress = true;
            }

            if (drain_completed() > 0) {
                made_progress = true;
            }
            if (release_completed_h2d_buffers(s) > 0) {
                made_progress = true;
            }

            if (!made_progress) {
                const bool waiting_for_h2d_buffer =
                    next_miss < miss_blobs.size() && !s.inflight_h2d.empty();
                if (io_outstanding() == 0 && !waiting_for_h2d_buffer) {
                    if (drain_completed() > 0 || release_completed_h2d_buffers(s) > 0) {
                        continue;
                    }
                    LLAMA_LOG_ERROR("moe_eval_callback: no I/O progress while loading misses "
                            "(submitted=%d completed=%d total=%zu)\n",
                            submitted_misses, completed_misses, miss_blobs.size());
                    GGML_ABORT("MoE-offload: async I/O made no progress");
                }

                const auto wait_start = std::chrono::steady_clock::now();
                std::this_thread::yield();
                drain_completed();
                release_completed_h2d_buffers(s);
                const auto wait_end = std::chrono::steady_clock::now();
                stall_us += std::chrono::duration_cast<std::chrono::microseconds>(
                        wait_end - wait_start).count();
            }
        }
    }

    int64_t slot_ids_h2d_us = 0;
    if (slot_ids_tensor && slot_ids_tensor->buffer) {
        std::vector<int32_t> slot_ids_host(ids.size(), 0);
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t e = ids[i];
            auto it = lc.exp2slot.find(e);
            if (it == lc.exp2slot.end()) {
                LLAMA_LOG_ERROR("moe_eval_callback: missing slot mapping after residency L%d e%d\n",
                        logical, e);
                GGML_ABORT("MoE-offload: missing slot mapping");
            }
            slot_ids_host[i] = it->second;
        }
        const auto slot_ids_h2d_start = std::chrono::steady_clock::now();
        set_tensor_for_compute(s, slot_ids_tensor, slot_ids_host.data(), 0,
                               slot_ids_host.size() * sizeof(int32_t));
        const auto slot_ids_h2d_end = std::chrono::steady_clock::now();
        slot_ids_h2d_us = elapsed_us(slot_ids_h2d_start, slot_ids_h2d_end);
    }

    // Strategy (D-4): build slot_table host buffer (init 0, set hot entries).
    // Phase M.5 diagnostic: optionally init to IDENTITY instead of 0 for
    // non-uniq positions. If the mmq kernel reads slot weights at non-top-k
    // indices, identity (slot_table[E]=E) routes those reads to slot E, which
    // is either the cached expert E (good) or zero (initial). Compare with
    // the default zero-init (which routes all non-uniq reads to slot 0 →
    // whichever expert is cached there → non-zero garbage contribution).
    {
        static const bool diag_id = std::getenv("LLAMA_MOE_DEBUG_ID_FILL") != nullptr;
        if (diag_id) {
            for (size_t i = 0; i < s.slot_table_host.size(); ++i) {
                s.slot_table_host[i] = (int32_t) i;
            }
        } else {
            std::fill(s.slot_table_host.begin(), s.slot_table_host.end(), 0);
        }
    }
    for (int32_t e : uniq) {
        s.slot_table_host[(size_t) e] = lc.exp2slot[e];
    }

    // Write slot_table to the persistent GPU-resident slot_table tensor. The
    // tensor was pre-allocated by `intercept_expert_tensor` in the same buffer
    // as the slot weight tensors, so its storage is never aliased by the
    // scheduler's temporaries — the immediately-following `ggml_get_rows`
    // consumer will read what we just wrote.
    int64_t slot_table_h2d_us = 0;
    if (stt && stt->buffer) {
        const auto slot_table_h2d_start = std::chrono::steady_clock::now();
        set_tensor_for_compute(s, stt, s.slot_table_host.data(), 0,
                               s.slot_table_host.size() * sizeof(int32_t));
        const auto slot_table_h2d_end = std::chrono::steady_clock::now();
        slot_table_h2d_us = elapsed_us(slot_table_h2d_start, slot_table_h2d_end);

        // Phase M.2 diagnostic: force a full compute-stream drain after the
        // slot_table write. If this collapses streaming drift to ~0, the bug
        // is stream-ordering between the slot_table H2D (cudaStreamPerThread)
        // and the get_rows/mul_mat_id consumers on the compute stream. Gated
        // on env var so it costs nothing in production.
        if (s.compute_backend) {
            static const bool diag_sync = std::getenv("LLAMA_MOE_DEBUG_SYNC") != nullptr;
            if (diag_sync) {
                ggml_backend_synchronize(s.compute_backend);
            }
        }

        if (debug_d4_trace() && s.topk_calls < 8) {
            // Log first few callbacks: buffer name, uniq count, first few entries
            const char * bn = ggml_backend_buffer_name(stt->buffer);
            size_t nz = 0;
            for (auto v : s.slot_table_host) if (v != 0) ++nz;
            fprintf(stderr, "[moe-d4] cb#%llu layer=%d uniq=%zu nz=%zu buf=%s stt=%p slot_ids=%p\n",
                    (unsigned long long) s.topk_calls, logical,
                    uniq.size(), nz, bn ? bn : "?", (void *) stt, (void *) slot_ids_tensor);
        }
    } else if (!slot_ids_tensor) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("%s: slot_table for layer %d has no buffer (stt=%p)\n",
                    __func__, logical, (void *) stt);
        }
    }
    ++s.topk_calls;

    // Phase E-1 / Phase I: buffer the per-layer profile row. compute_us
    // and h2d_us are patched in at slot_pool_end_request() once the CUDA
    // timing events have signalled.
    {
        int k_req = (int) uniq.size();
        slot_pool_state::pending_profile_row p;
        p.logical = logical;
        const profile_request_row request_row = current_profile_request_row();
        p.row.request_idx = request_row.request_idx;
        p.row.repeat_idx = request_row.repeat_idx;
        p.row.batch_idx = request_row.batch_idx;
        p.row.token_idx = row_token_idx;
        p.row.phase = phase;
        p.row.layer = logical;
        p.row.k_required = k_req;
        p.row.k_hit = k_req - misses_loaded;
        p.row.k_miss = misses_loaded;
        p.row.ssd_read_us = layer_load_stats.ssd_read_us;
        // Phase I: h2d_us comes from CUDA events on the async path. The
        // sync-fallback contribution (already host-measured) is preserved
        // here; event-elapsed times are added at end_request.
        p.row.h2d_us = layer_load_stats.h2d_us;
        p.row.compute_us = 0;  // filled in at end_request
        // Phase H: stall_us is the host-measured wall-clock around the
        // io_compute_wait drain.
        p.row.stall_us = stall_us;
        p.row.pred_observe_us = pred_observe_us;
        p.row.pred_score_us = pred_score_us;
        p.row.pred_us = pred_observe_us + pred_score_us;
        p.row.eamc_rows_scored = pred_stats.eamc_rows_scored;
        p.row.eamc_cosine_us = pred_stats.eamc_cosine_us;
        p.row.eamc_score_materialize_us = pred_stats.eamc_score_materialize_us;
        p.row.eamc_score_cache_hits = pred_stats.eamc_score_cache_hits;
        p.row.eamc_score_cache_misses = pred_stats.eamc_score_cache_misses;
        p.row.topk_d2h_us = topk_d2h_us;
        p.row.slot_ids_h2d_us = slot_ids_h2d_us;
        p.row.slot_table_h2d_us = slot_table_h2d_us;
        p.row.ssd_bytes = layer_load_stats.ssd_bytes;
        p.row.ssd_reads = layer_load_stats.ssd_reads;
        p.row.cache_resident_experts = (int) lc.exp2slot.size();
        p.row.predictor = s.pred->name();
        p.h2d_events = std::move(h2d_events_for_row);

        // Phase I: record compute_begin on the compute stream right after
        // the slot_table write. compute_end is recorded when the next MoE
        // layer's topk callback fires (ask=true branch) or at
        // slot_pool_end_request for the final layer.
        if (s.compute_backend) {
            void * ev = io_event_acquire();
            if (ev && io_record_on_compute(s.compute_backend, ev)) {
                p.compute_begin_event = ev;
            } else if (ev) {
                io_event_release(ev);
            }
        }

        s.pending_rows.push_back(std::move(p));
        s.last_pending_idx = (int) s.pending_rows.size() - 1;
    }

    if (s.last_pending_idx >= 0 && (size_t) s.last_pending_idx < s.pending_rows.size()) {
        const auto callback_end = std::chrono::steady_clock::now();
        s.pending_rows[(size_t) s.last_pending_idx].row.callback_wall_us = elapsed_us(callback_start, callback_end);
    }

    // Phase M.6 diagnostic: content-hash slots after the callback has finished
    // all writes.  Compares against the saved END hash from the *previous*
    // callback visit for the same (layer, slot).  If a slot changed WITHOUT an
    // eviction, the GPU buffer is being aliased or overwritten between callbacks.
    static const bool diag_slot_hash = std::getenv("LLAMA_MOE_DEBUG_SLOT_HASH") != nullptr;
    if (diag_slot_hash && logical < 3) {
        constexpr int kMonSlots[] = {0, 1, 2, 3, 8};
        constexpr int kMonCount   = 5;
        // saved END hash from the previous time this (layer,slot) was visited
        static uint64_t prev_hash[3][kMonCount];
        static bool     prev_hash_valid[3][kMonCount] = {};

        ggml_tensor * st_for_read = nullptr;
        for (int k = 0; k < EXPERT_KIND_COUNT; ++k) {
            if (s.slot_tensors[logical][k]) { st_for_read = s.slot_tensors[logical][k]; break; }
        }
        if (st_for_read && st_for_read->buffer) {
            const size_t buf_size = ggml_backend_buffer_get_size(st_for_read->buffer);
            const size_t stride   = (size_t) st_for_read->nb[2];
            uint8_t buf[1024];
            for (int si = 0; si < kMonCount; ++si) {
                int s_idx = kMonSlots[si];
                if ((uint32_t) s_idx >= s.n_slots) continue;
                size_t off = (size_t) s_idx * stride;
                size_t sz  = stride < sizeof(buf) ? stride : sizeof(buf);
                if (off + sz > buf_size) continue;
                ggml_backend_tensor_get(st_for_read, buf, off, sz);
                uint64_t hash = 0xcbf29ce484222325ULL;
                for (size_t i = 0; i < sz; ++i) { hash ^= buf[i]; hash *= 0x100000001b3ULL; }
                const char * tag = "";
                if (prev_hash_valid[logical][si] && hash != prev_hash[logical][si]) {
                    tag = " CHANGED-SINCE-LAST";
                }
                fprintf(stderr, "[slot-hash-end] tok=%llu L%d slot=%d hash=0x%016llx%s\n",
                        (unsigned long long) row_token_idx, logical, s_idx,
                        (unsigned long long) hash, tag);
                prev_hash[logical][si]       = hash;
                prev_hash_valid[logical][si] = true;
            }
        }
    }

    if ((uint32_t) logical + 1 == mf.n_layers) {
        s.token_idx += n_tokens > 0 ? (uint64_t) n_tokens : 1;
    }

    if (debug_d4_trace() && s.topk_calls % 200 == 0) {
        fprintf(stderr, "[moe-d4] moe_eval_callback: %llu topk calls, hits=%llu misses=%llu\n",
                (unsigned long long) s.topk_calls,
                (unsigned long long) s.cache_hits,
                (unsigned long long) s.cache_misses);
    }
    return true;
}

// ── Phase D-5: async I/O lifecycle ──────────────────────────────────────

void slot_pool_init_io(const std::string & source_path) {
    // Lazy-init: called on first eval-callback or externally. If the worker
    // is already running, this is a no-op.
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    const manifest & mf = get_manifest();

    // The eval callback submits misses in chunks and drains completions as
    // buffers/queue capacity becomes tight, so the pool no longer has to cover
    // an entire large-ubatch layer miss set. Keep enough buffers for decode and
    // moderate prefill overlap while bounding pinned memory cost.
    constexpr int kMinIoBuffers = 32;
    constexpr int kMaxIoBuffers = 256;
    int n_buffers = (int)(2 * n_slots_per_layer());
    if (n_buffers < kMinIoBuffers) n_buffers = kMinIoBuffers;
    if (n_buffers > kMaxIoBuffers) n_buffers = kMaxIoBuffers;
    size_t blob_max = mf.expert_blob_size_max > 0 ? mf.expert_blob_size_max : (1024*1024);

    LLAMA_LOG_INFO("%s: initializing async I/O worker (n_buffers=%d, blob_max=%zu)\n",
            __func__, n_buffers, blob_max);

    if (!io_init(source_path.c_str(), blob_max, n_buffers)) {
        LLAMA_LOG_WARN("%s: io_init failed; will use sync I/O\n", __func__);
    }
}

void slot_pool_shutdown_io() {
    auto & s = state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        wait_all_h2d_buffers(s);
    }
    io_shutdown();
    LLAMA_LOG_INFO("%s: io worker stopped; events_in_use=%zu\n",
            __func__, io_events_in_use());
}

void slot_pool_set_compute_backend(ggml_backend_t backend) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.compute_backend = backend;
    LLAMA_LOG_INFO("%s: compute backend = %s\n", __func__,
            backend ? ggml_backend_name(backend) : "<none>");
}

void slot_pool_begin_request() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.current_request_phase = "unknown";
    if (s.pred) {
        s.pred->begin_request();
    }
}

void slot_pool_end_request() {
    const auto end_request_start = std::chrono::steady_clock::now();
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);

    // Phase I: close out the final layer's compute interval. The compute
    // stream is by now (or shortly will be) idle, so cudaEventRecord here
    // captures roughly the end of the last layer's MoE compute.
    if (s.compute_backend && s.last_pending_idx >= 0 &&
        (size_t) s.last_pending_idx < s.pending_rows.size()) {
        auto & p = s.pending_rows[s.last_pending_idx];
        if (!p.compute_end_event) {
            void * ev = io_event_acquire();
            if (ev && io_record_on_compute(s.compute_backend, ev)) {
                p.compute_end_event = ev;
            } else if (ev) {
                io_event_release(ev);
            }
        }
    }

    // Ensure borrowed H2D events are complete before profile rows query and
    // release them. This can still block at request end if copies from the
    // final callback are in flight.
    wait_all_h2d_buffers(s);

    // Phase I: drain buffered rows. Query CUDA elapsed times for compute
    // and h2d intervals, patch into the row, hand off to the profiler,
    // then release all events back to the pool.
    profiler * prof = get_profiler();
    int64_t profile_flush_us = 0;
    for (auto & p : s.pending_rows) {
        if (s.current_request_phase == "unknown") {
            s.current_request_phase = p.row.phase;
        } else if (s.current_request_phase != p.row.phase) {
            s.current_request_phase = "mixed";
        }
        int64_t compute_us = 0;
        if (p.compute_begin_event && p.compute_end_event) {
            int64_t us = io_event_elapsed_us(p.compute_begin_event, p.compute_end_event);
            if (us > 0) compute_us = us;
        }
        p.row.compute_us = compute_us;

        int64_t h2d_us_events = 0;
        for (auto & ev_pair : p.h2d_events) {
            if (ev_pair.first && ev_pair.second) {
                int64_t us = io_event_elapsed_us(ev_pair.first, ev_pair.second);
                if (us > 0) h2d_us_events += us;
            }
        }
        p.row.h2d_us += h2d_us_events;

        if (prof) {
            prof->record(p.row);
        }

        if (p.compute_begin_event) io_event_release(p.compute_begin_event);
        if (p.compute_end_event)   io_event_release(p.compute_end_event);
        for (auto & ev_pair : p.h2d_events) {
            if (ev_pair.first)  io_event_release(ev_pair.first);
            if (ev_pair.second) io_event_release(ev_pair.second);
        }
    }
    s.pending_rows.clear();
    s.last_pending_idx = -1;

    int64_t predictor_end_us = 0;
    int64_t predictor_save_us = 0;
    uint64_t sidecar_write_bytes = 0;
    if (s.pred) {
        const auto pred_end_start = std::chrono::steady_clock::now();
        s.pred->end_request();
        const auto pred_end_done = std::chrono::steady_clock::now();
        predictor_end_us = elapsed_us(pred_end_start, pred_end_done);
        if (std::strcmp(s.pred->name(), "eamc") == 0) {
            s.pred_dirty = true;
        }
    }
    if (prof) {
        const auto flush_start = std::chrono::steady_clock::now();
        prof->flush();
        const auto flush_done = std::chrono::steady_clock::now();
        profile_flush_us = elapsed_us(flush_start, flush_done);
    }

    (void) end_request_start;
    add_current_request_timing(s.current_request_phase.c_str(), predictor_end_us, predictor_save_us, profile_flush_us, sidecar_write_bytes);
    s.current_request_phase = "unknown";
}

bool slot_pool_flush_predictor() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.pred || std::strcmp(s.pred->name(), "eamc") != 0) {
        return true;
    }

    const runtime_options & opts = get_options();
    if (opts.eamc_path.empty() || !s.pred_dirty) {
        return true;
    }

    const bool ok = s.pred->save(opts.eamc_path);
    if (ok) {
        s.pred_dirty = false;
        std::ifstream in(opts.eamc_path, std::ios::binary | std::ios::ate);
        if (in) {
            const std::streamoff size = in.tellg();
            LLAMA_LOG_INFO("%s: saved EAMC sidecar: %s (%lld bytes)\n",
                    __func__, opts.eamc_path.c_str(), (long long) size);
        } else {
            LLAMA_LOG_INFO("%s: saved EAMC sidecar: %s\n",
                    __func__, opts.eamc_path.c_str());
        }
    }
    return ok;
}

} // namespace llama_moe
