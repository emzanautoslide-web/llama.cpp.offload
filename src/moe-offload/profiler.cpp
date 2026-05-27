#include "profiler.h"

#include <iomanip>
#include <sstream>

namespace llama_moe {

profiler::profiler(const std::string & csv_path) {
    open(csv_path);
}

profiler::~profiler() {
    if (csv.is_open()) {
        csv.flush();
    }
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
    ++rows;
    required += (uint64_t) row.k_required;
    hits += (uint64_t) row.k_hit;
    misses += (uint64_t) row.k_miss;
    ssd_read_us += row.ssd_read_us;
    h2d_us += row.h2d_us;
    compute_us += row.compute_us;
    stall_us += row.stall_us;

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

std::string profiler::summary() const {
    std::ostringstream out;
    const double hit_rate = required == 0 ? 0.0 : 100.0 * (double) hits / (double) required;
    out << "MoE offload summary\n";
    out << "rows: " << rows << '\n';
    out << "experts required: " << required << '\n';
    out << "cache hits: " << hits << '\n';
    out << "cache misses: " << misses << '\n';
    out << "cache hit rate: " << std::fixed << std::setprecision(2) << hit_rate << "%\n";
    out << "ssd_read_us: " << ssd_read_us << '\n';
    out << "h2d_us: " << h2d_us << '\n';
    out << "compute_us: " << compute_us << '\n';
    out << "stall_us: " << stall_us << '\n';
    return out.str();
}

void profiler::write_header() {
    csv << "token_idx,phase,layer,k_required,k_hit,k_miss,ssd_read_us,h2d_us,compute_us,stall_us,cache_resident_experts,predictor\n";
}

} // namespace llama_moe