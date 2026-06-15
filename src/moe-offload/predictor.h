#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llama_moe {

enum class predictor_kind {
    lru,
    eamc,
};

struct predictor_score_stats {
    uint64_t eamc_rows_scored = 0;
    int64_t  eamc_cosine_us = 0;
    int64_t  eamc_score_materialize_us = 0;
    uint64_t eamc_score_cache_hits = 0;
    uint64_t eamc_score_cache_misses = 0;
};

predictor_kind parse_predictor_kind(const std::string & name);
const char * predictor_kind_name(predictor_kind kind);

class predictor {
public:
    virtual ~predictor() = default;

    virtual const char * name() const = 0;
    virtual void begin_request() = 0;
    virtual void observe(int layer, const std::vector<int> & experts_used) = 0;
    virtual float score(int layer, int expert) const = 0;
    virtual void end_request() = 0;
    virtual bool load(const std::string & path);
    virtual bool save(const std::string & path) const;
    virtual predictor_score_stats take_score_stats() const;
};

std::unique_ptr<predictor> make_predictor(
        predictor_kind kind,
        int n_layers,
        int n_experts,
        size_t eamc_capacity = 1024,
        size_t eamc_top_k = 8);

} // namespace llama_moe
