#include "slot_pool.h"

#include "loader.h"
#include "runtime.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-impl.h"

#include <array>
#include <cstdint>
#include <cstdio>
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

    const std::string slot_name = src_name + ".slot";

    ggml_tensor * slot = ml.create_unfiled_tensor(
            hparams, buft_list_layer, slot_name, src_meta->type,
            GGML_OP_MUL_MAT_ID, { d_in, d_out, (int64_t) n_slots });

    if (!slot) {
        throw std::runtime_error("intercept_expert_tensor: create_unfiled_tensor returned null");
    }

    // The original GGUF weight is no longer materialized via the loader; account for it.
    ml.mark_tensor_unloaded(src_name);

    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (logical_layer >= 0 && (size_t) logical_layer < s.slot_tensors.size()) {
        s.slot_tensors[logical_layer][kind] = slot;
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

bool prefetch_all_experts() {
    if (!runtime_enabled()) {
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
    if (n_slots < mf.n_experts_per_layer) {
        LLAMA_LOG_INFO("%s: skipping prefetch (n_slots=%u < n_experts=%u; eval-callback streaming mode)\n",
                __func__, n_slots, mf.n_experts_per_layer);
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
        return;
    }
    s.topk_to_slot_table[topk]   = slot_table;
    s.topk_to_logical[topk]      = logical_layer;
    s.all_slot_tables.push_back(slot_table);
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

// Load one expert's three blobs (gate, up, down) from disk into the chosen
// slot index. Caller holds s.mutex.
bool load_expert_into_slot(slot_pool_state & s, const manifest & mf,
                           uint32_t logical_layer, int32_t expert, int32_t slot) {
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
        if (moe_fseek(s.io_fp, (long long)(mf.data_offset + rec.rel_offset), SEEK_SET) != 0) {
            LLAMA_LOG_ERROR("moe_eval_callback: seek failed L%u e%d k%d\n", logical_layer, expert, k);
            return false;
        }
        size_t got = fread(s.io_scratch.data(), 1, rec.size, s.io_fp);
        if (got != rec.size) {
            LLAMA_LOG_ERROR("moe_eval_callback: short read L%u e%d k%d got=%zu want=%zu\n",
                    logical_layer, expert, k, got, (size_t) rec.size);
            return false;
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
        static int dump = 0;
        if (dump < 6) {
            ++dump;
            LLAMA_LOG_INFO("load_expert dump: L%u e%d k%d slot=%d rec.size=%llu stride=%zu tensor_off=%zu buf_size=%zu data=%p\n",
                    logical_layer, expert, k, slot,
                    (unsigned long long) rec.size, stride, tensor_off, buf_size, slot_tensor->data);
        }
        ggml_backend_tensor_set(slot_tensor, s.io_scratch.data(), write_off, rec.size);
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
    if (ask) return true;       // run for every node
    if (!t || !t->name[0]) return true;

    auto & s = state();
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

    // Read top-k IDs (D2H). Shape: [n_expert_used, n_tokens] I32.
    const int64_t n_elem = ggml_nelements(t);
    std::vector<int32_t> ids((size_t) n_elem);
    ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));

    // Collect unique experts.
    std::unordered_set<int32_t> uniq;
    uniq.reserve(ids.size());
    for (int32_t e : ids) {
        if (e >= 0 && (uint32_t) e < mf.n_experts_per_layer) uniq.insert(e);
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
    for (int32_t e : uniq) {
        auto it = lc.exp2slot.find(e);
        if (it != lc.exp2slot.end()) {
            lru_touch(lc, e);
            ++s.cache_hits;
        }
    }
    for (int32_t e : uniq) {
        if (lc.exp2slot.count(e)) continue;
        // Miss: pick a slot. Free slots first, else LRU victim.
        int32_t slot = -1;
        for (uint32_t si = 0; si < s.n_slots; ++si) {
            if (lc.slot_to_expert[si] < 0) { slot = (int32_t) si; break; }
        }
        if (slot < 0) {
            // Evict the LRU expert (back of list).
            int32_t victim = lc.lru.back();
            lc.lru.pop_back();
            lc.lru_it.erase(victim);
            slot = lc.exp2slot[victim];
            lc.exp2slot.erase(victim);
            lc.slot_to_expert[slot] = -1;
        }
        if (!load_expert_into_slot(s, mf, (uint32_t) logical, e, slot)) {
            return true;  // I/O failed; can't proceed but don't crash compute
        }
        lc.slot_to_expert[slot] = e;
        lc.exp2slot[e] = slot;
        lru_touch(lc, e);
        ++s.cache_misses;
    }

    // Build slot_table host buffer. Only entries for selected experts matter.
    // Initialize all to 0 (safe sentinel; will be overwritten for hot ones).
    std::fill(s.slot_table_host.begin(), s.slot_table_host.end(), 0);
    for (int32_t e : uniq) {
        s.slot_table_host[(size_t) e] = lc.exp2slot[e];
    }
    ggml_tensor * stt_ok = stt;
    static bool first_call_logged = false;
    if (!first_call_logged) {
        first_call_logged = true;
        const char * buf_name = (stt_ok && stt_ok->buffer) ? ggml_backend_buffer_name(stt_ok->buffer) : "<null>";
        LLAMA_LOG_INFO("moe_eval_callback: FIRST CALL t=%p name=%s logical=%d stt=%p stt_buf=%p stt_buf_name=%s n_uniq=%zu n_slots=%u\n",
                (void*)t, t->name, logical, (void*)stt_ok, stt_ok ? (void*)stt_ok->buffer : nullptr,
                buf_name, uniq.size(), s.n_slots);
        // also log first slot tensor buffer name to be sure
        if (s.slot_tensors[logical][0]) {
            const char * sn = s.slot_tensors[logical][0]->buffer ? ggml_backend_buffer_name(s.slot_tensors[logical][0]->buffer) : "<null>";
            LLAMA_LOG_INFO("  slot_tensor[0] buf=%s nb2=%zu\n", sn, s.slot_tensors[logical][0]->nb[2]);
        }
        for (int32_t e : uniq) {
            int32_t sl = lc.exp2slot.count(e) ? lc.exp2slot[e] : -1;
            LLAMA_LOG_INFO("  expert %d -> slot %d\n", e, sl);
        }
    }
    // Bounds check: ensure all written slot indices are valid.
    {
        static int bad_count = 0;
        for (size_t i = 0; i < s.slot_table_host.size(); ++i) {
            int32_t v = s.slot_table_host[i];
            if (v < 0 || (uint32_t) v >= s.n_slots) {
                if (bad_count < 10) {
                    LLAMA_LOG_ERROR("moe_eval_callback: BAD slot %d at expert %zu (n_slots=%u) logical=%d topk=%llu\n",
                            v, i, s.n_slots, logical, (unsigned long long) s.topk_calls);
                    ++bad_count;
                }
                break;
            }
        }
    }
    // Per-layer first-call trace
    static std::vector<bool> per_layer_logged;
    if (per_layer_logged.size() != s.cache.size()) per_layer_logged.assign(s.cache.size(), false);
    if (!per_layer_logged[logical]) {
        per_layer_logged[logical] = true;
        const char * buf_name = (stt_ok && stt_ok->buffer) ? ggml_backend_buffer_name(stt_ok->buffer) : "<null>";
        LLAMA_LOG_INFO("moe_eval_callback: layer %d first call n_uniq=%zu stt_buf=%s misses=%llu\n",
                logical, uniq.size(), buf_name, (unsigned long long) s.cache_misses);
    }
    // Per-call trace (first batch only)
    static uint64_t call_counter = 0;
    if (call_counter < 200) {
        LLAMA_LOG_INFO("moe_eval_callback: call #%llu layer=%d n_uniq=%zu\n",
                (unsigned long long) call_counter, logical, uniq.size());
        ++call_counter;
    }
    if (stt_ok && stt_ok->buffer) {
        ggml_backend_tensor_set(stt_ok, s.slot_table_host.data(), 0,
                                s.slot_table_host.size() * sizeof(int32_t));
        // Verify GPU write took effect by reading back (debug only, first layer)
        if (logical == 0 && s.topk_calls == 0) {
            std::vector<int32_t> back(s.slot_table_host.size());
            ggml_backend_tensor_get(stt_ok, back.data(), 0, back.size() * sizeof(int32_t));
            int mismatches = 0;
            for (size_t i = 0; i < back.size(); ++i) {
                if (back[i] != s.slot_table_host[i]) ++mismatches;
            }
            LLAMA_LOG_INFO("moe_eval_callback: GPU readback verify layer 0 mismatches=%d back[16]=%d back[160]=%d (sent 0 and 1)\n",
                    mismatches, back[16], back[160]);
        }
    } else {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_ERROR("moe_eval_callback: stt has no buffer! stt=%p logical=%d\n",
                    (void*)stt_ok, logical);
        }
    }
    ++s.topk_calls;
    if (s.topk_calls % 200 == 0) {
        LLAMA_LOG_INFO("moe_eval_callback: %llu topk calls, hits=%llu misses=%llu\n",
                (unsigned long long) s.topk_calls,
                (unsigned long long) s.cache_hits,
                (unsigned long long) s.cache_misses);
    }
    return true;
}

} // namespace llama_moe
