#pragma once

#include "predictor.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llama_moe {

struct layer_cache_stats {
    int layer = -1;
    int k_required = 0;
    int k_hit = 0;
    int k_miss = 0;
    int resident = 0;
    std::unordered_map<int, int> expert_to_slot;
    std::vector<int> missed_experts;
};

class expert_cache {
public:
    expert_cache(int n_layers, int n_experts, int slots_per_layer, std::unique_ptr<predictor> pred);

    void begin_request();
    layer_cache_stats begin_layer(int layer, const std::vector<int> & required_experts);
    void end_layer(int layer, const std::vector<int> & used_experts);
    void end_request();

    int resident_count(int layer) const;
    const predictor * get_predictor() const;

private:
    struct layer_state {
        std::vector<int> slot_to_expert;
        std::unordered_map<int, int> expert_to_slot;
        std::vector<uint64_t> last_use;
        std::vector<bool> pinned;
    };

    int choose_slot(layer_state & state, int layer);

    int n_layers;
    int n_experts;
    int slots_per_layer;
    uint64_t step = 0;
    std::vector<layer_state> layers;
    std::unique_ptr<predictor> pred;
};

} // namespace llama_moe