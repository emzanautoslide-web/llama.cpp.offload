#include "predictor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace llama_moe {

predictor_kind parse_predictor_kind(const std::string & name) {
    if (name == "lru") {
        return predictor_kind::lru;
    }
    if (name == "eamc") {
        return predictor_kind::eamc;
    }
    throw std::invalid_argument("unknown MoE predictor: " + name);
}

const char * predictor_kind_name(predictor_kind kind) {
    switch (kind) {
        case predictor_kind::lru:  return "lru";
        case predictor_kind::eamc: return "eamc";
    }
    return "unknown";
}

bool predictor::load(const std::string &) {
    return false;
}

bool predictor::save(const std::string &) const {
    return false;
}

predictor_score_stats predictor::take_score_stats() const {
    return {};
}

class lru_predictor final : public predictor {
public:
    lru_predictor(int n_layers, int n_experts) :
        n_layers(n_layers),
        n_experts(n_experts),
        last_use((size_t) n_layers * (size_t) n_experts, 0) {
    }

    const char * name() const override { return "lru"; }

    void begin_request() override {}

    void observe(int layer, const std::vector<int> & experts_used) override {
        ++step;
        if (layer < 0 || layer >= n_layers) {
            return;
        }
        for (int expert : experts_used) {
            if (expert >= 0 && expert < n_experts) {
                last_use[index(layer, expert)] = step;
            }
        }
    }

    float score(int layer, int expert) const override {
        if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) {
            return 0.0f;
        }
        return (float) last_use[index(layer, expert)];
    }

    void end_request() override {}

private:
    size_t index(int layer, int expert) const {
        return (size_t) layer * (size_t) n_experts + (size_t) expert;
    }

    int n_layers;
    int n_experts;
    uint64_t step = 0;
    std::vector<uint64_t> last_use;
};

class eamc_predictor final : public predictor {
public:
    eamc_predictor(int n_layers, int n_experts, size_t capacity, size_t top_k) :
        n_layers(n_layers),
        n_experts(n_experts),
        capacity(capacity),
        top_k(top_k),
        effective_rows(parse_effective_rows()),
        current_layers((size_t) n_layers),
        corpus_index((size_t) n_layers * (size_t) n_experts),
        last_use((size_t) n_layers * (size_t) n_experts, 0) {
    }

    const char * name() const override { return "eamc"; }

    void begin_request() override {
        for (auto & layer : current_layers) {
            layer.clear();
        }
        current_layer = -1;
        current_norm2 = 0.0;
        invalidate_score_cache();
    }

    void observe(int layer, const std::vector<int> & experts_used) override {
        ++step;
        invalidate_score_cache();
        if (layer < 0 || layer >= n_layers) {
            return;
        }
        current_layer = std::max(current_layer, layer);
        for (int expert : experts_used) {
            if (expert >= 0 && expert < n_experts) {
                auto & values = current_layers[(size_t) layer];
                auto it = std::lower_bound(values.begin(), values.end(), expert,
                    [](const auto & kv, int needle) { return kv.first < needle; });
                const float old = it != values.end() && it->first == expert ? it->second : 0.0f;
                const float updated = old + 1.0f;
                current_norm2 += (double) updated * (double) updated - (double) old * (double) old;
                if (it != values.end() && it->first == expert) {
                    it->second = updated;
                } else {
                    values.insert(it, {expert, updated});
                }
                last_use[index(layer, expert)] = step;
            }
        }
    }

    float score(int layer, int expert) const override {
        if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) {
            return 0.0f;
        }
        if (corpus.empty() || current_layer < 0) {
            return (float) last_use[index(layer, expert)];
        }

        materialize_scores(layer);
        if (score_vector_valid && score_vector_layer == layer &&
                expert >= 0 && (size_t) expert < score_vector.size()) {
            return score_vector[(size_t) expert];
        }

        const auto & ranked = nearest_neighbors();
        const size_t k = ranked.size();

        double weighted = 0.0;
        double weights = 0.0;
        for (size_t i = 0; i < k; ++i) {
            const float sim = std::max(0.0f, ranked[i].first);
            const float value = layer_value(corpus[ranked[i].second], layer, expert);
            const float proximity = layer > current_layer
                ? std::max(0.0f, 1.0f - (float) (layer - current_layer) / (float) std::max(1, n_layers))
                : 1.0f;
            const double w = (double) sim * (double) proximity;
            weighted += (double) value * w;
            weights += w;
        }

        if (weights == 0.0) {
            return (float) last_use[index(layer, expert)];
        }
        return (float) (weighted / weights);
    }

    void end_request() override {
        if (current_norm2 == 0.0f || capacity == 0) {
            return;
        }
        sparse_row row = make_current_row();
        if (corpus.size() < capacity) {
            corpus.push_back(std::move(row));
            add_index_entries(corpus.size() - 1, corpus.back());
            next_replace = corpus.size() % capacity;
        } else {
            remove_index_entries(next_replace, corpus[next_replace]);
            corpus[next_replace] = std::move(row);
            add_index_entries(next_replace, corpus[next_replace]);
            next_replace = (next_replace + 1) % capacity;
        }
        invalidate_score_cache();
    }

    bool load(const std::string & path) override {
        if (path.empty()) {
            return false;
        }

        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return false;
        }

        char magic[5] = {};
        uint32_t file_n_layers = 0;
        uint32_t file_n_experts = 0;
        uint64_t file_capacity = 0;
        uint64_t file_top_k = 0;
        uint64_t rows = 0;

        in.read(magic, 4);
        in.read(reinterpret_cast<char *>(&file_n_layers), sizeof(file_n_layers));
        in.read(reinterpret_cast<char *>(&file_n_experts), sizeof(file_n_experts));
        in.read(reinterpret_cast<char *>(&file_capacity), sizeof(file_capacity));
        in.read(reinterpret_cast<char *>(&file_top_k), sizeof(file_top_k));
        in.read(reinterpret_cast<char *>(&rows), sizeof(rows));

        const size_t row_size = (size_t) n_layers * (size_t) n_experts;
        if (!in || std::string(magic, 4) != "EAM1" ||
            file_n_layers != (uint32_t) n_layers ||
            file_n_experts != (uint32_t) n_experts ||
            file_capacity == 0 ||
            rows > file_capacity ||
            row_size == 0) {
            std::fprintf(stderr, "[moe-eamc] ignoring incompatible sidecar: %s\n", path.c_str());
            return false;
        }

        std::vector<sparse_row> loaded;
        loaded.reserve((size_t) rows);
        for (uint64_t r = 0; r < rows; ++r) {
            std::vector<float> row(row_size);
            in.read(reinterpret_cast<char *>(row.data()), (std::streamsize) (row.size() * sizeof(float)));
            if (!in) {
                std::fprintf(stderr, "[moe-eamc] ignoring truncated sidecar: %s\n", path.c_str());
                return false;
            }
            loaded.push_back(dense_to_sparse(row));
        }

        corpus = std::move(loaded);
        capacity = (size_t) file_capacity;
        top_k = (size_t) file_top_k;
        next_replace = capacity == 0 ? 0 : corpus.size() % capacity;
        rebuild_corpus_index();
        invalidate_score_cache();
        std::fprintf(stderr, "[moe-eamc] loaded %zu EAMC row(s) from %s\n", corpus.size(), path.c_str());
        return true;
    }

    bool save(const std::string & path) const override {
        if (path.empty() || corpus.empty()) {
            return false;
        }

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "[moe-eamc] failed to open sidecar for write: %s\n", path.c_str());
            return false;
        }

        const char magic[4] = {'E', 'A', 'M', '1'};
        const uint32_t file_n_layers = (uint32_t) n_layers;
        const uint32_t file_n_experts = (uint32_t) n_experts;
        const uint64_t file_capacity = (uint64_t) capacity;
        const uint64_t file_top_k = (uint64_t) top_k;
        const uint64_t rows = (uint64_t) corpus.size();

        out.write(magic, sizeof(magic));
        out.write(reinterpret_cast<const char *>(&file_n_layers), sizeof(file_n_layers));
        out.write(reinterpret_cast<const char *>(&file_n_experts), sizeof(file_n_experts));
        out.write(reinterpret_cast<const char *>(&file_capacity), sizeof(file_capacity));
        out.write(reinterpret_cast<const char *>(&file_top_k), sizeof(file_top_k));
        out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        std::vector<float> dense;
        for (const auto & row : corpus) {
            dense_row(row, dense);
            out.write(reinterpret_cast<const char *>(dense.data()), (std::streamsize) (dense.size() * sizeof(float)));
        }

        if (!out) {
            std::fprintf(stderr, "[moe-eamc] failed while writing sidecar: %s\n", path.c_str());
            return false;
        }
        return true;
    }

    predictor_score_stats take_score_stats() const override {
        predictor_score_stats result = stats;
        stats = {};
        return result;
    }

private:
    struct sparse_row {
        std::vector<std::vector<std::pair<int, float>>> layers;
        std::vector<double> prefix_norm2;
        double norm2 = 0.0;
    };

    size_t index(int layer, int expert) const {
        return (size_t) layer * (size_t) n_experts + (size_t) expert;
    }

    int64_t elapsed_us(
            const std::chrono::steady_clock::time_point & start,
            const std::chrono::steady_clock::time_point & end) const {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    size_t parse_effective_rows() const {
        const char * env = std::getenv("LLAMA_MOE_EAMC_ROWS");
        if (!env || !env[0]) {
            return 0;
        }
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(env, &end, 10);
        if (end == env || parsed == 0) {
            return 0;
        }
        return (size_t) parsed;
    }

    sparse_row dense_to_sparse(const std::vector<float> & dense) const {
        sparse_row row;
        row.layers.resize((size_t) n_layers);
        row.prefix_norm2.resize((size_t) n_layers, 0.0);
        double prefix = 0.0;
        for (int layer = 0; layer < n_layers; ++layer) {
            auto & out = row.layers[(size_t) layer];
            for (int expert = 0; expert < n_experts; ++expert) {
                const float v = dense[index(layer, expert)];
                if (v != 0.0f) {
                    out.push_back({expert, v});
                    prefix += (double) v * (double) v;
                }
            }
            row.prefix_norm2[(size_t) layer] = prefix;
        }
        row.norm2 = prefix;
        return row;
    }

    sparse_row make_current_row() const {
        sparse_row row;
        row.layers.resize((size_t) n_layers);
        row.prefix_norm2.resize((size_t) n_layers, 0.0);
        double prefix = 0.0;
        for (int layer = 0; layer < n_layers; ++layer) {
            auto & out = row.layers[(size_t) layer];
            out = current_layers[(size_t) layer];
            for (const auto & kv : out) {
                prefix += (double) kv.second * (double) kv.second;
            }
            row.prefix_norm2[(size_t) layer] = prefix;
        }
        row.norm2 = prefix;
        return row;
    }

    void add_index_entries(size_t row_idx, const sparse_row & row) {
        for (int layer = 0; layer < n_layers; ++layer) {
            for (const auto & kv : row.layers[(size_t) layer]) {
                corpus_index[index(layer, kv.first)].push_back({row_idx, kv.second});
            }
        }
    }

    void remove_index_entries(size_t row_idx, const sparse_row & row) {
        for (int layer = 0; layer < n_layers; ++layer) {
            for (const auto & kv : row.layers[(size_t) layer]) {
                auto & entries = corpus_index[index(layer, kv.first)];
                entries.erase(std::remove_if(entries.begin(), entries.end(),
                    [row_idx](const auto & entry) { return entry.first == row_idx; }), entries.end());
            }
        }
    }

    void rebuild_corpus_index() {
        corpus_index.assign((size_t) n_layers * (size_t) n_experts, {});
        for (size_t row_idx = 0; row_idx < corpus.size(); ++row_idx) {
            add_index_entries(row_idx, corpus[row_idx]);
        }
    }

    void dense_row(const sparse_row & row, std::vector<float> & dense) const {
        dense.assign((size_t) n_layers * (size_t) n_experts, 0.0f);
        for (int layer = 0; layer < n_layers; ++layer) {
            for (const auto & kv : row.layers[(size_t) layer]) {
                dense[index(layer, kv.first)] = kv.second;
            }
        }
    }

    float layer_value(const sparse_row & row, int layer, int expert) const {
        if (layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) {
            return 0.0f;
        }
        const auto & values = row.layers[(size_t) layer];
        const auto it = std::lower_bound(values.begin(), values.end(), expert,
            [](const auto & kv, int needle) { return kv.first < needle; });
        return it != values.end() && it->first == expert ? it->second : 0.0f;
    }

    float cosine_partial(const sparse_row & row) const {
        double dot = 0.0;
        const int layer_limit = std::min(current_layer + 1, n_layers);
        for (int layer = 0; layer < layer_limit; ++layer) {
            const auto & cur = current_layers[(size_t) layer];
            const auto & corpus_layer = row.layers[(size_t) layer];
            for (const auto & kv : cur) {
                const int expert = kv.first;
                const auto it = std::lower_bound(corpus_layer.begin(), corpus_layer.end(), expert,
                    [](const auto & row_kv, int needle) { return row_kv.first < needle; });
                if (it != corpus_layer.end() && it->first == expert) {
                    dot += (double) kv.second * (double) it->second;
                }
            }
        }
        const double norm_a = current_norm2;
        const double norm_b = layer_limit > 0 ? row.prefix_norm2[(size_t) layer_limit - 1] : 0.0;
        if (norm_a == 0.0 || norm_b == 0.0) {
            return 0.0f;
        }
        return (float) (dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    const std::vector<std::pair<float, size_t>> & nearest_neighbors() const {
        if (score_cache_valid && score_cache_step == step && score_cache_layer == current_layer) {
            return score_cache;
        }

        score_cache.clear();
        if (corpus.empty() || current_layer < 0 || top_k == 0) {
            score_cache_valid = true;
            score_cache_step = step;
            score_cache_layer = current_layer;
            return score_cache;
        }

        const auto cosine_start = std::chrono::steady_clock::now();
        std::vector<std::pair<float, size_t>> ranked;
        const size_t n_scored = corpus_rows_to_score();
        if (n_scored != corpus.size()) {
            ranked.reserve(n_scored);
            for (size_t i : corpus_row_indices()) {
                ranked.push_back({cosine_partial(corpus[i]), i});
            }
        } else {
            ensure_dot_capacity();
            begin_dot_epoch();
            const int layer_limit = std::min(current_layer + 1, n_layers);
            for (int layer = 0; layer < layer_limit; ++layer) {
                for (const auto & cur : current_layers[(size_t) layer]) {
                    for (const auto & entry : corpus_index[index(layer, cur.first)]) {
                        const size_t row_idx = entry.first;
                        if (dot_mark[row_idx] != dot_epoch) {
                            dot_mark[row_idx] = dot_epoch;
                            dot_accum[row_idx] = 0.0;
                        }
                        dot_accum[row_idx] += (double) cur.second * (double) entry.second;
                    }
                }
            }
            ranked.reserve(corpus.size());
            const double norm_a = current_norm2;
            for (size_t row_idx = 0; row_idx < corpus.size(); ++row_idx) {
                float sim = 0.0f;
                if (dot_mark[row_idx] == dot_epoch) {
                    const double norm_b = layer_limit > 0 ? corpus[row_idx].prefix_norm2[(size_t) layer_limit - 1] : 0.0;
                    if (norm_a != 0.0 && norm_b != 0.0) {
                        sim = (float) (dot_accum[row_idx] / (std::sqrt(norm_a) * std::sqrt(norm_b)));
                    }
                }
                ranked.push_back({sim, row_idx});
            }
        }
        const auto cosine_done = std::chrono::steady_clock::now();
        stats.eamc_rows_scored += (uint64_t) n_scored;
        stats.eamc_cosine_us += elapsed_us(cosine_start, cosine_done);
        const size_t k = std::min(top_k, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + (ptrdiff_t) k, ranked.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });
        score_cache.assign(ranked.begin(), ranked.begin() + (ptrdiff_t) k);
        score_cache_valid = true;
        score_cache_step = step;
        score_cache_layer = current_layer;
        return score_cache;
    }

    size_t corpus_rows_to_score() const {
        if (effective_rows == 0 || effective_rows >= corpus.size()) {
            return corpus.size();
        }
        return effective_rows;
    }

    std::vector<size_t> corpus_row_indices() const {
        const size_t n = corpus_rows_to_score();
        std::vector<size_t> result;
        result.reserve(n);
        if (n == corpus.size()) {
            for (size_t i = 0; i < corpus.size(); ++i) {
                result.push_back(i);
            }
            return result;
        }

        const size_t start = next_replace >= n ? next_replace - n : corpus.size() + next_replace - n;
        for (size_t i = 0; i < n; ++i) {
            result.push_back((start + i) % corpus.size());
        }
        return result;
    }

    void ensure_dot_capacity() const {
        if (dot_accum.size() < corpus.size()) {
            dot_accum.resize(corpus.size(), 0.0);
            dot_mark.resize(corpus.size(), 0);
        }
    }

    void begin_dot_epoch() const {
        ++dot_epoch;
        if (dot_epoch == 0) {
            std::fill(dot_mark.begin(), dot_mark.end(), 0);
            dot_epoch = 1;
        }
    }

    void materialize_scores(int layer) const {
        if (score_vector_valid && score_vector_step == step &&
                score_vector_current_layer == current_layer && score_vector_layer == layer) {
            ++stats.eamc_score_cache_hits;
            return;
        }

        ++stats.eamc_score_cache_misses;
        const auto & ranked = nearest_neighbors();
        const auto materialize_start = std::chrono::steady_clock::now();
        score_vector.assign((size_t) n_experts, 0.0f);
        double weights = 0.0;
        std::vector<double> weighted((size_t) n_experts, 0.0);
        for (const auto & entry : ranked) {
            const float sim = std::max(0.0f, entry.first);
            const float proximity = layer > current_layer
                ? std::max(0.0f, 1.0f - (float) (layer - current_layer) / (float) std::max(1, n_layers))
                : 1.0f;
            const double w = (double) sim * (double) proximity;
            if (w == 0.0) {
                continue;
            }
            weights += w;
            for (const auto & kv : corpus[entry.second].layers[(size_t) layer]) {
                weighted[(size_t) kv.first] += (double) kv.second * w;
            }
        }

        if (weights == 0.0) {
            for (int expert = 0; expert < n_experts; ++expert) {
                score_vector[(size_t) expert] = (float) last_use[index(layer, expert)];
            }
        } else {
            for (int expert = 0; expert < n_experts; ++expert) {
                score_vector[(size_t) expert] = (float) (weighted[(size_t) expert] / weights);
            }
        }

        score_vector_valid = true;
        score_vector_step = step;
        score_vector_current_layer = current_layer;
        score_vector_layer = layer;
        const auto materialize_done = std::chrono::steady_clock::now();
        stats.eamc_score_materialize_us += elapsed_us(materialize_start, materialize_done);
    }

    void invalidate_score_cache() const {
        score_cache_valid = false;
        score_vector_valid = false;
    }

    int n_layers;
    int n_experts;
    size_t capacity;
    size_t top_k;
    size_t effective_rows;
    size_t next_replace = 0;
    int current_layer = -1;
    uint64_t step = 0;
    std::vector<std::vector<std::pair<int, float>>> current_layers;
    double current_norm2 = 0.0;
    std::vector<uint64_t> last_use;
    std::vector<sparse_row> corpus;
    std::vector<std::vector<std::pair<size_t, float>>> corpus_index;
    mutable predictor_score_stats stats;
    mutable bool score_cache_valid = false;
    mutable uint64_t score_cache_step = 0;
    mutable int score_cache_layer = -1;
    mutable std::vector<std::pair<float, size_t>> score_cache;
    mutable std::vector<double> dot_accum;
    mutable std::vector<uint32_t> dot_mark;
    mutable uint32_t dot_epoch = 0;
    mutable bool score_vector_valid = false;
    mutable uint64_t score_vector_step = 0;
    mutable int score_vector_current_layer = -1;
    mutable int score_vector_layer = -1;
    mutable std::vector<float> score_vector;
};

std::unique_ptr<predictor> make_predictor(
        predictor_kind kind,
        int n_layers,
        int n_experts,
        size_t eamc_capacity,
        size_t eamc_top_k) {
    switch (kind) {
        case predictor_kind::lru:
            return std::unique_ptr<predictor>(new lru_predictor(n_layers, n_experts));
        case predictor_kind::eamc:
            return std::unique_ptr<predictor>(new eamc_predictor(n_layers, n_experts, eamc_capacity, eamc_top_k));
    }
    throw std::invalid_argument("unknown MoE predictor kind");
}

} // namespace llama_moe
