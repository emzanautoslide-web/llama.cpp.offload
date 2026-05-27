#include "predictor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
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
        current((size_t) n_layers * (size_t) n_experts, 0.0f),
        last_use((size_t) n_layers * (size_t) n_experts, 0) {
    }

    const char * name() const override { return "eamc"; }

    void begin_request() override {
        std::fill(current.begin(), current.end(), 0.0f);
        current_layer = -1;
    }

    void observe(int layer, const std::vector<int> & experts_used) override {
        ++step;
        if (layer < 0 || layer >= n_layers) {
            return;
        }
        current_layer = std::max(current_layer, layer);
        for (int expert : experts_used) {
            if (expert >= 0 && expert < n_experts) {
                current[index(layer, expert)] += 1.0f;
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

        std::vector<std::pair<float, size_t>> ranked;
        ranked.reserve(corpus.size());
        for (size_t i = 0; i < corpus.size(); ++i) {
            ranked.push_back({cosine_partial(corpus[i]), i});
        }
        const size_t k = std::min(top_k, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + (ptrdiff_t) k, ranked.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });

        double weighted = 0.0;
        double weights = 0.0;
        for (size_t i = 0; i < k; ++i) {
            const float sim = std::max(0.0f, ranked[i].first);
            const float value = corpus[ranked[i].second][index(layer, expert)];
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
        const bool has_activation = std::any_of(current.begin(), current.end(), [](float v) { return v != 0.0f; });
        if (!has_activation || capacity == 0) {
            return;
        }
        corpus.push_back(current);
        if (corpus.size() > capacity) {
            evict_redundant();
        }
    }

private:
    size_t index(int layer, int expert) const {
        return (size_t) layer * (size_t) n_experts + (size_t) expert;
    }

    float cosine_partial(const std::vector<float> & row) const {
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        const int layer_limit = std::min(current_layer + 1, n_layers);
        for (int layer = 0; layer < layer_limit; ++layer) {
            for (int expert = 0; expert < n_experts; ++expert) {
                const size_t i = index(layer, expert);
                dot += (double) current[i] * (double) row[i];
                norm_a += (double) current[i] * (double) current[i];
                norm_b += (double) row[i] * (double) row[i];
            }
        }
        if (norm_a == 0.0 || norm_b == 0.0) {
            return 0.0f;
        }
        return (float) (dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    static float cosine_full(const std::vector<float> & a, const std::vector<float> & b) {
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += (double) a[i] * (double) b[i];
            norm_a += (double) a[i] * (double) a[i];
            norm_b += (double) b[i] * (double) b[i];
        }
        if (norm_a == 0.0 || norm_b == 0.0) {
            return 0.0f;
        }
        return (float) (dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    void evict_redundant() {
        size_t best = 0;
        float best_mean = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < corpus.size(); ++i) {
            double mean = 0.0;
            for (size_t j = 0; j < corpus.size(); ++j) {
                if (i != j) {
                    mean += cosine_full(corpus[i], corpus[j]);
                }
            }
            mean /= (double) std::max<size_t>(1, corpus.size() - 1);
            if ((float) mean > best_mean) {
                best_mean = (float) mean;
                best = i;
            }
        }
        corpus.erase(corpus.begin() + (ptrdiff_t) best);
    }

    int n_layers;
    int n_experts;
    size_t capacity;
    size_t top_k;
    int current_layer = -1;
    uint64_t step = 0;
    std::vector<float> current;
    std::vector<uint64_t> last_use;
    std::vector<std::vector<float>> corpus;
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