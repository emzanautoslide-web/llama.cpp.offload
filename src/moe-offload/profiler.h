#pragma once

#include "llama.h"

#include <cstdint>
#include <fstream>
#include <string>

namespace llama_moe {

struct profile_row {
    uint64_t token_idx = 0;
    std::string phase = "decode";
    int layer = -1;
    int k_required = 0;
    int k_hit = 0;
    int k_miss = 0;
    int64_t ssd_read_us = 0;
    int64_t h2d_us = 0;
    int64_t compute_us = 0;
    int64_t stall_us = 0;
    uint64_t ssd_bytes = 0;
    uint64_t ssd_reads = 0;
    int cache_resident_experts = 0;
    std::string predictor = "lru";
};

struct profile_phase_stats {
    uint64_t rows = 0;
    uint64_t required = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t ssd_bytes = 0;
    uint64_t ssd_reads = 0;
    int64_t ssd_read_us = 0;
    int64_t h2d_us = 0;
    int64_t compute_us = 0;
    int64_t stall_us = 0;
    int cache_resident_peak = 0;
};

struct profile_snapshot {
    profile_phase_stats prefill;
    profile_phase_stats decode;
};

struct profile_summary_context {
    std::string model;
    std::string predictor;
    std::string storage;
    uint64_t cache_mb = 0;
    int n_prompt = 0;
    int n_gen = 0;
    int n_repeat = 1;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    double total_ms = 0.0;
    uint64_t vram_peak_bytes = 0;
    uint64_t vram_total_bytes = 0;
    uint64_t dram_peak_bytes = 0;
};

class profiler {
public:
    profiler() = default;
    explicit profiler(const std::string & csv_path);
    ~profiler();

    bool open(const std::string & csv_path);
    void record(const profile_row & row);
    void flush();
    profile_snapshot snapshot() const;
    std::string summary() const;

private:
    void write_header();
    profile_phase_stats total() const;

    std::ofstream csv;
    profile_phase_stats prefill_stats;
    profile_phase_stats decode_stats;
};

LLAMA_API std::string format_summary(
    const profile_summary_context & ctx,
    const profile_snapshot & profile);

} // namespace llama_moe