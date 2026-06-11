#include "cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llama_moe {

expert_cache::expert_cache(int n_layers, int n_experts, int slots_per_layer, std::unique_ptr<predictor> pred) :
    n_layers(n_layers),
    n_experts(n_experts),
    slots_per_layer(slots_per_layer),
    layers((size_t) n_layers),
    pred(std::move(pred)) {
    if (n_layers <= 0 || n_experts <= 0 || slots_per_layer <= 0) {
        throw std::invalid_argument("invalid MoE cache shape");
    }

    for (auto & layer : layers) {
        layer.slot_to_expert.assign((size_t) slots_per_layer, -1);
        layer.last_use.assign((size_t) slots_per_layer, 0);
        layer.pinned.assign((size_t) slots_per_layer, false);
    }
}

void expert_cache::begin_request() {
    if (pred) {
        pred->begin_request();
    }
}

layer_cache_stats expert_cache::begin_layer(int layer, const std::vector<int> & required_experts) {
    if (layer < 0 || layer >= n_layers) {
        throw std::out_of_range("MoE layer out of range");
    }

    ++step;
    auto & state = layers[(size_t) layer];
    std::fill(state.pinned.begin(), state.pinned.end(), false);

    layer_cache_stats stats;
    stats.layer = layer;

    std::vector<int> unique_required = required_experts;
    std::sort(unique_required.begin(), unique_required.end());
    unique_required.erase(std::unique(unique_required.begin(), unique_required.end()), unique_required.end());

    for (int expert : unique_required) {
        if (expert < 0 || expert >= n_experts) {
            continue;
        }
        ++stats.k_required;
        auto found = state.expert_to_slot.find(expert);
        if (found != state.expert_to_slot.end()) {
            const int slot = found->second;
            state.last_use[(size_t) slot] = step;
            state.pinned[(size_t) slot] = true;
            stats.expert_to_slot[expert] = slot;
            ++stats.k_hit;
            continue;
        }

        const int slot = choose_slot(state, layer);
        const int evicted = state.slot_to_expert[(size_t) slot];
        if (evicted >= 0) {
            state.expert_to_slot.erase(evicted);
        }
        state.slot_to_expert[(size_t) slot] = expert;
        state.expert_to_slot[expert] = slot;
        state.last_use[(size_t) slot] = step;
        state.pinned[(size_t) slot] = true;

        stats.expert_to_slot[expert] = slot;
        stats.missed_experts.push_back(expert);
        ++stats.k_miss;
    }

    stats.resident = resident_count(layer);
    return stats;
}

void expert_cache::end_layer(int layer, const std::vector<int> & used_experts) {
    if (layer < 0 || layer >= n_layers) {
        return;
    }
    std::fill(layers[(size_t) layer].pinned.begin(), layers[(size_t) layer].pinned.end(), false);
    if (pred) {
        pred->observe(layer, used_experts);
    }
}

void expert_cache::end_request() {
    if (pred) {
        pred->end_request();
    }
}

int expert_cache::resident_count(int layer) const {
    if (layer < 0 || layer >= n_layers) {
        return 0;
    }
    return (int) layers[(size_t) layer].expert_to_slot.size();
}

const predictor * expert_cache::get_predictor() const {
    return pred.get();
}

int expert_cache::choose_slot(layer_state & state, int layer) {
    for (int slot = 0; slot < slots_per_layer; ++slot) {
        if (state.slot_to_expert[(size_t) slot] < 0) {
            return slot;
        }
    }

    int best_slot = -1;
    float best_score = std::numeric_limits<float>::infinity();
    uint64_t best_last_use = std::numeric_limits<uint64_t>::max();

    for (int slot = 0; slot < slots_per_layer; ++slot) {
        if (state.pinned[(size_t) slot]) {
            continue;
        }
        const int expert = state.slot_to_expert[(size_t) slot];
        const float pred_score = pred ? pred->score(layer, expert) : 0.0f;
        const uint64_t lru_score = state.last_use[(size_t) slot];
        if (pred_score < best_score || (pred_score == best_score && lru_score < best_last_use)) {
            best_score = pred_score;
            best_last_use = lru_score;
            best_slot = slot;
        }
    }

    if (best_slot < 0) {
        best_slot = 0;
    }
    return best_slot;
}

} // namespace llama_moe