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

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
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
    // Phase D-1/D-2: per-(topk tensor) slot_table input tensor. Because the
    // compute graph can be rebuilt multiple times (graph_reserve passes,
    // different ubatch shapes) we record the (topk -> slot_table) association
    // per graph build so the eval-callback always writes to the slot_table
    // tensor of the SAME graph being executed.
    std::unordered_map<ggml_tensor *, ggml_tensor *> topk_to_slot_table;
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
        // Phase M.7: FNV1a-64 hash of the first 1024 bytes of each cached
        // expert's slot data (gate kind).  Used to detect cross-step GPU
        // buffer corruption so we can force a reload.
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
};

slot_pool_state & state() {
    static slot_pool_state instance;
    return instance;
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
    if (n < 1) n = 1;
    if (n > mf.n_experts_per_layer) n = mf.n_experts_per_layer;
    return n;
}

llm_tensor base_tensor_of(llm_tensor t) {
    return t;
}

} // namespace

void configure_slot_pool() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);

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

    // Phase E: init predictor from runtime options
    const runtime_options & opts = get_options();
    predictor_kind pk = predictor_kind::lru;
    try { pk = parse_predictor_kind(opts.predictor); } catch (...) {}
    s.pred = make_predictor(pk, (int) mf.n_layers, (int) mf.n_experts_per_layer);
    s.pred->begin_request();

    s.configured = true;

    LLAMA_LOG_INFO("%s: slot pool configured: %u logical MoE layers x %u experts -> %u slots/layer\n",
            __func__, mf.n_layers, mf.n_experts_per_layer, s.n_slots);
}

void reset_slot_pool() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.configured = false;
    s.n_slots = 0;
    s.slot_tensors.clear();
    s.slot_table_tensors.clear();
    s.topk_to_slot_table.clear();
    s.topk_to_logical.clear();
    s.all_slot_tables.clear();
    s.cache.clear();
    s.slot_table_host.clear();
    s.io_scratch.clear();
    if (s.io_fp) { fclose(s.io_fp); s.io_fp = nullptr; }
}

uint32_t n_slots_per_layer() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.n_slots;
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
    if (intercept_cnt < 4) {
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

    // Always allocate the slot tensor with the original expert count (n_expert)
    // as ne[2]. The mmq kernel used by mul_mat_id can crash with non-standard
    // expert counts. The cache still manages only n_slots experts;
    // unused slots [n_slots, n_expert-1] are never accessed because the
    // remap maps expert IDs strictly into [0, n_slots-1].
    const int64_t n_expert = ne.begin()[2];
    (void) n_slots; // usable cache size; expert axis must stay at n_expert for mmq

    const std::string slot_name = src_name + ".slot";

    ggml_tensor * slot = ml.create_unfiled_tensor(
            hparams, buft_list_layer, slot_name, src_meta->type,
            GGML_OP_MUL_MAT_ID, { d_in, d_out, n_expert });

    if (!slot) {
        throw std::runtime_error("intercept_expert_tensor: create_unfiled_tensor returned null");
    }

    // D-4 DEBUG: verify slot tensor strides match source metadata strides
    if (logical_layer == 0 && kind == EXPERT_GATE) {
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
                fprintf(stderr, "[moe-d4] created %s L%d ne=[%lld,%lld] buf=%s has_buf=%d\n",
                        st_name, logical_layer,
                        (long long) stt->ne[0], (long long) stt->ne[1],
                        stt->buffer ? ggml_backend_buffer_name(stt->buffer) : "NULL",
                        stt->buffer ? 1 : 0);
            } else {
                fprintf(stderr, "[moe-d4] FAILED create_unfiled_tensor for %s L%d\n",
                        st_name, logical_layer);
            }
        } else if (logical_layer >= 0) {
            if ((size_t) logical_layer >= stvsz) {
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
    fprintf(stderr, "[moe-d4] prefetch_all_experts ENTER\n");
    if (!runtime_enabled()) {
        fprintf(stderr, "[moe-d4] prefetch_all_experts: runtime disabled, skip\n");
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
        fprintf(stderr, "[moe-d4] prefetch_all_experts: starting zero-init of slot tensors...\n");
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
        fprintf(stderr, "[moe-d4] prefetch_all_experts: zeroed %zu slot tensors in %.2f ms (incl sync)\n",
                n_zeroed, (ggml_time_us() - t_z) / 1000.0);
    }

    if (n_slots < mf.n_experts_per_layer) {
        fprintf(stderr, "[moe-d4] prefetch_all_experts: skipping prefetch (n_slots=%u < n_experts=%u; streaming mode)\n",
                n_slots, mf.n_experts_per_layer);
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
        fprintf(stderr, "[moe-d4] register_slot_table_for_topk SKIP: cfg=%d topk=%p st=%p L=%d\n",
                s.configured ? 1 : 0, (void*)topk, (void*)slot_table, logical_layer);
        return;
    }
    s.topk_to_slot_table[topk]   = slot_table;
    s.topk_to_logical[topk]      = logical_layer;
    s.all_slot_tables.push_back(slot_table);
    static int reg_cnt = 0;
    if (reg_cnt < 4) {
        fprintf(stderr, "[moe-d4] register_slot_table_for_topk #%d L=%d topk=%p st=%p st->buf=%s\n",
                reg_cnt, logical_layer, (void*)topk, (void*)slot_table,
                slot_table->buffer ? ggml_backend_buffer_name(slot_table->buffer) : "NULL");
    }
    ++reg_cnt;
}

void reset_graph_state() {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.topk_to_slot_table.clear();
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
        if (dump < 3000) {
            ++dump;
            const char * bname = slot_tensor->buffer ? ggml_backend_buffer_name(slot_tensor->buffer) : "<null>";
            fprintf(stderr, "[moe-d3] load#%d L%u e%d k%d slot=%d rec.size=%llu stride=%zu write_off=%zu tensor_off=%zu buf_size=%zu buf=%s data=%p\n",
                    dump, logical_layer, expert, k, slot,
                    (unsigned long long) rec.size, stride, write_off, tensor_off, buf_size, bname, slot_tensor->data);
            fflush(stderr);
        }
        const auto h2d_start = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(slot_tensor, s.io_scratch.data(), write_off, rec.size);
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
        // Phase I: when the *next* MoE layer's topk is about to be computed,
        // close the previous layer's compute interval by recording
        // compute_end on the compute stream. We take the lock only when
        // there is plausibly something to do (cheap atomic-like check on
        // the previously-set index).
        if (s.last_pending_idx < 0) {
            return true;
        }
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.configured || !s.compute_backend) return true;
        if (s.last_pending_idx < 0 ||
            (size_t) s.last_pending_idx >= s.pending_rows.size()) return true;
        auto stt_it = s.topk_to_slot_table.find(t);
        if (stt_it == s.topk_to_slot_table.end()) return true;
        auto & p = s.pending_rows[s.last_pending_idx];
        if (!p.compute_end_event) {
            void * ev = io_event_acquire();
            if (ev && io_record_on_compute(s.compute_backend, ev)) {
                p.compute_end_event = ev;
            } else if (ev) {
                io_event_release(ev);
            }
        }
        return true;
    }

    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.configured) return true;

    // Fast lookup: only react to topk tensors we previously registered.
    auto stt_it = s.topk_to_slot_table.find(t);
    if (stt_it == s.topk_to_slot_table.end()) return true;
    ggml_tensor * stt = stt_it->second;
    auto log_it = s.topk_to_logical.find(t);
    if (log_it == s.topk_to_logical.end()) return true;
    const int logical = log_it->second;

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
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));

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

    // Phase E-3: observe expert usage for predictor.
    {
        std::vector<int> obs(uniq.begin(), uniq.end());
        s.pred->observe(logical, obs);
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

    // Phase M.7: verify HIT expert slot data integrity.  The GPU buffer that
    // backs slot tensors can be aliased by scheduler temporaries or mmq
    // kernel scratch writes during graph execution.  We read back and hash
    // the first 1 KiB of each HIT expert's gate-kind slot data and compare
    // against the fingerprint stored after the last successful load.  A
    // mismatch means the slot was corrupted; we evict the stale entry so
    // the miss-loop below reloads it from SSD.
    {
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
                    auto it = lc.exp2slot.find(e);
                    if (it != lc.exp2slot.end()) {
                        lc.slot_to_expert[it->second] = -1;
                        auto lru_it = lc.lru_it.find(e);
                        if (lru_it != lc.lru_it.end()) { lc.lru.erase(lru_it->second); lc.lru_it.erase(lru_it); }
                        lc.exp2slot.erase(it);
                        lc.fingerprints.erase(e);
                    }
                    if (fp_corrupt_log < 10) {
                        fprintf(stderr, "[fp-corrupt] tok=%llu L%d expert=%d fingerprint mismatch — forcing reload\n",
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

    // Phase H: per-miss bookkeeping needed when the worker completion comes
    // back out of submission order. We carry the slot tensor pointer (so the
    // fallback sync H2D path can call ggml_backend_tensor_set) and the
    // write_off, keyed by pinned_buf because that's the unique handle the
    // worker echoes back to us.
    struct miss_meta {
        ggml_tensor * slot_tensor;
        size_t        write_off;
    };
    std::unordered_map<void *, miss_meta> miss_lookup;
    int submitted_misses = 0;

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
            for (const auto & [exp, sl] : lc.exp2slot) {
                if (reserved_this_call.count(exp)) continue;
                float sc = s.pred->score(logical, exp);
                if (sc < best_score) { best_score = sc; best_victim = exp; }
            }
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

        // Phase H: submit one async read per (kind) into the slot's GPU
        // address. The worker freads into a pinned staging buffer and then
        // issues cudaMemcpyAsync on the dedicated MoE H2D stream, recording
        // an event. The callback waits on those events below.
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

            void * pinned = io_acquire_buffer();
            io_request req{};
            req.layer       = logical;
            req.expert      = e;
            req.kind        = k;
            req.slot        = slot;
            req.pinned_buf  = pinned;
            req.blob_size   = rec.size;
            req.file_offset = mf.data_offset + rec.rel_offset;
            req.gpu_dst     = (char *) slot_tensor->data + write_off;
            // Phase M.2 diagnostic: optionally disable async H2D so the
            // worker leaves h2d_event=null and the callback falls back to
            // sync ggml_backend_tensor_set. If this collapses drift, the
            // async write path (worker → io_h2d_async_timed) is at fault.
            {
                static const bool diag_no_async = std::getenv("LLAMA_MOE_DEBUG_NO_ASYNC") != nullptr;
                req.h2d         = !diag_no_async && (s.compute_backend != nullptr);
            }
            req.h2d_event   = nullptr;
            req.h2d_begin_event = nullptr;
            req.ssd_read_us = 0;

            miss_lookup[pinned] = miss_meta{ slot_tensor, write_off };

            if (!io_submit(req)) {
                LLAMA_LOG_ERROR("moe_eval_callback: io_submit failed (queue full) L%d e%d k%d\n",
                        logical, e, k);
                GGML_ABORT("MoE-offload: io_submit queue overflow");
            }
            ++submitted_misses;
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

    int64_t stall_us = 0;
    // Phase I: per-miss h2d timing events stashed for elapsed-time query at
    // end_request. Owned by the pending_profile_row we create below.
    std::vector<std::pair<void *, void *>> h2d_events_for_row;
    if (submitted_misses > 0) {
        const auto wait_start = std::chrono::steady_clock::now();

        // Drain the worker: spin until all submitted reads (and their queued
        // async H2D copies) have been issued.
        while (io_outstanding() > 0) {
            std::this_thread::yield();
        }
        auto completions = io_drain_completed();

        for (auto & c : completions) {
            layer_load_stats.ssd_read_us += c.ssd_read_us;
            layer_load_stats.ssd_bytes   += c.blob_size;
            ++layer_load_stats.ssd_reads;

            if (c.h2d_event) {
                // Async path: tell the compute stream to wait on the H2D
                // end event before consuming the slot weights. Stash both
                // events (begin, end) so end_request can query their
                // elapsed time for real h2d_us. They are released back to
                // the pool at end_request after the query.
                io_compute_wait(s.compute_backend, c.h2d_event);
                h2d_events_for_row.emplace_back(c.h2d_begin_event, c.h2d_event);
            } else {
                // Fallback: CUDA unavailable or io_h2d_async failed. Look up
                // the slot tensor by the pinned-buffer handle and do a sync
                // ggml_backend_tensor_set.
                auto it = miss_lookup.find(c.pinned_buf);
                if (it == miss_lookup.end()) {
                    LLAMA_LOG_ERROR("moe_eval_callback: completion without lookup entry (L%d)\n", logical);
                } else {
                    const auto h2d_start = std::chrono::steady_clock::now();
                    ggml_backend_tensor_set(it->second.slot_tensor, c.pinned_buf,
                                            it->second.write_off, c.blob_size);
                    const auto h2d_end = std::chrono::steady_clock::now();
                    layer_load_stats.h2d_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            h2d_end - h2d_start).count();
                }
            }

            // Phase M.7: fingerprint the loaded data for cross-step
            // integrity verification.  We hash the first 1 KiB of the
            // pinned buffer (which holds the source expert blob).  Only
            // fingerprint the GATE kind so the verification read-back
            // touches just one tensor per layer.
            if (c.kind == EXPERT_GATE && c.blob_size > 0) {
                size_t fp_sz = c.blob_size < 1024 ? (size_t) c.blob_size : 1024;
                uint64_t fp = 0xcbf29ce484222325ULL;
                const uint8_t * src = (const uint8_t *) c.pinned_buf;
                for (size_t i = 0; i < fp_sz; ++i) { fp ^= src[i]; fp *= 0x100000001b3ULL; }
                lc.fingerprints[c.expert] = fp;
            }

            io_release_buffer(c.pinned_buf);
        }

        const auto wait_end = std::chrono::steady_clock::now();
        stall_us = std::chrono::duration_cast<std::chrono::microseconds>(
                wait_end - wait_start).count();
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
    if (stt && stt->buffer) {
        ggml_backend_tensor_set(stt, s.slot_table_host.data(), 0,
                                s.slot_table_host.size() * sizeof(int32_t));

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

        if (s.topk_calls < 8) {
            // Log first few callbacks: buffer name, uniq count, first few entries
            const char * bn = ggml_backend_buffer_name(stt->buffer);
            size_t nz = 0;
            for (auto v : s.slot_table_host) if (v != 0) ++nz;
            fprintf(stderr, "[moe-d4] cb#%llu layer=%d uniq=%zu nz=%zu buf=%s stt=%p\n",
                    (unsigned long long) s.topk_calls, logical,
                    uniq.size(), nz, bn ? bn : "?", (void *) stt);
        }
    } else {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[moe-d4] moe_eval_callback: slot_table for layer %d has no buffer (stt=%p)\n",
                    logical, (void *) stt);
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

    if (s.topk_calls % 200 == 0) {
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

    // Phase L.2: the buffer pool must accommodate the worst-case per-layer
    // miss demand WITHOUT requiring an intermediate drain, because the
    // miss-loop currently submits all misses for a layer before draining
    // any. Worst case = (kMaxStreamingUbatch * top_k_max) unique experts
    // per layer, each contributing EXPERT_KIND_COUNT (=3) in-flight
    // requests. Conservative bound is 8 (ubatch cap) * 8 (typical top_k
    // upper-bound) * 3 = 192. We also keep at least 2x slot count so the
    // single-token decode path overlaps freely. Cap at 256 to bound pinned
    // memory cost (~256 * blob_max bytes).
    constexpr int kMinIoBuffers = 192;
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

void slot_pool_end_request() {
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

    // Phase I: drain buffered rows. Query CUDA elapsed times for compute
    // and h2d intervals, patch into the row, hand off to the profiler,
    // then release all events back to the pool.
    profiler * prof = get_profiler();
    for (auto & p : s.pending_rows) {
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

    if (s.pred) {
        s.pred->end_request();
    }
}

} // namespace llama_moe
