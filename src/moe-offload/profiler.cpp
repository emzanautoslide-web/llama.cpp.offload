#include "profiler.h"

#include <iomanip>
#include <sstream>

namespace llama_moe {

profiler::profiler(const std::string & csv_path) {
    open(csv_path);
}

profiler::~profiler() {
    flush();
}

bool profiler::open(const std::string & csv_path) {
    if (csv_path.empty()) {
        return true;
    }
    csv.open(csv_path, std::ios::out | std::ios::trunc);
    if (!csv) {
        return false;
    }
    write_header();
    return true;
}

void profiler::record(const profile_row & row) {
    profile_phase_stats & stats = row.phase == "prefill" ? prefill_stats : decode_stats;
    ++stats.rows;
    stats.required += (uint64_t) row.k_required;
    stats.hits += (uint64_t) row.k_hit;
    stats.misses += (uint64_t) row.k_miss;
    stats.ssd_bytes += row.ssd_bytes;
    stats.ssd_reads += row.ssd_reads;
    stats.ssd_read_us += row.ssd_read_us;
    stats.h2d_us += row.h2d_us;
    stats.compute_us += row.compute_us;
    stats.stall_us += row.stall_us;
    if (row.cache_resident_experts > stats.cache_resident_peak) {
        stats.cache_resident_peak = row.cache_resident_experts;
    }

    if (csv.is_open()) {
        csv << row.token_idx << ','
            << row.phase << ','
            << row.layer << ','
            << row.k_required << ','
            << row.k_hit << ','
            << row.k_miss << ','
            << row.ssd_read_us << ','
            << row.h2d_us << ','
            << row.compute_us << ','
            << row.stall_us << ','
            << row.cache_resident_experts << ','
            << row.predictor << '\n';
    }
}

void profiler::flush() {
    if (csv.is_open()) {
        csv.flush();
    }
}

profile_snapshot profiler::snapshot() const {
    profile_snapshot snap;
    snap.prefill = prefill_stats;
    snap.decode = decode_stats;
    return snap;
}

std::string profiler::summary() const {
    std::ostringstream out;
    const profile_phase_stats stats = total();
    const double hit_rate = stats.required == 0 ? 0.0 : 100.0 * (double) stats.hits / (double) stats.required;
    out << "MoE offload summary\n";
    out << "rows: " << stats.rows << '\n';
    out << "experts required: " << stats.required << '\n';
    out << "cache hits: " << stats.hits << '\n';
    out << "cache misses: " << stats.misses << '\n';
    out << "cache hit rate: " << std::fixed << std::setprecision(2) << hit_rate << "%\n";
    out << "ssd_bytes: " << stats.ssd_bytes << '\n';
    out << "ssd_reads: " << stats.ssd_reads << '\n';
    out << "ssd_read_us: " << stats.ssd_read_us << '\n';
    out << "h2d_us: " << stats.h2d_us << '\n';
    out << "compute_us: " << stats.compute_us << '\n';
    out << "stall_us: " << stats.stall_us << '\n';
    return out.str();
}

profile_phase_stats profiler::total() const {
    profile_phase_stats stats;
    stats.rows = prefill_stats.rows + decode_stats.rows;
    stats.required = prefill_stats.required + decode_stats.required;
    stats.hits = prefill_stats.hits + decode_stats.hits;
    stats.misses = prefill_stats.misses + decode_stats.misses;
    stats.ssd_bytes = prefill_stats.ssd_bytes + decode_stats.ssd_bytes;
    stats.ssd_reads = prefill_stats.ssd_reads + decode_stats.ssd_reads;
    stats.ssd_read_us = prefill_stats.ssd_read_us + decode_stats.ssd_read_us;
    stats.h2d_us = prefill_stats.h2d_us + decode_stats.h2d_us;
    stats.compute_us = prefill_stats.compute_us + decode_stats.compute_us;
    stats.stall_us = prefill_stats.stall_us + decode_stats.stall_us;
    stats.cache_resident_peak = prefill_stats.cache_resident_peak > decode_stats.cache_resident_peak
        ? prefill_stats.cache_resident_peak
        : decode_stats.cache_resident_peak;
    return stats;
}

void profiler::write_header() {
    csv << "token_idx,phase,layer,k_required,k_hit,k_miss,ssd_read_us,h2d_us,compute_us,stall_us,cache_resident_experts,predictor\n";
}

} // namespace llama_moe