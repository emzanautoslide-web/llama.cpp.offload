#include "profiler.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace llama_moe {

namespace {

double bytes_to_gib(uint64_t bytes) {
    return (double) bytes / (1024.0 * 1024.0 * 1024.0);
}

double bytes_to_mib(double bytes) {
    return bytes / (1024.0 * 1024.0);
}

double hit_rate_percent(const profile_phase_stats & stats) {
    return stats.required == 0 ? 0.0 : 100.0 * (double) stats.hits / (double) stats.required;
}

double us_per_token_ms(int64_t usec, int tokens, int repeats) {
    const int denom = std::max(1, tokens * repeats);
    return (double) usec / 1000.0 / (double) denom;
}

double safe_div(double numerator, uint64_t denominator) {
    return denominator == 0 ? 0.0 : numerator / (double) denominator;
}

void append_io_breakdown(std::ostringstream & out, const char * label, const profile_phase_stats & stats, int tokens, int repeats) {
    out << "I/O breakdown (" << label << ", mean per token):\n";
    out << "  ssd_read       " << std::setw(8) << std::setprecision(2)
        << us_per_token_ms(stats.ssd_read_us, tokens, repeats) << " ms\n";
    out << "  h2d            " << std::setw(8)
        << us_per_token_ms(stats.h2d_us, tokens, repeats) << " ms\n";
    out << "  gpu_compute    " << std::setw(8)
        << us_per_token_ms(stats.compute_us, tokens, repeats) << " ms\n";
    out << "  stall (overlap loss) " << std::setw(8)
        << us_per_token_ms(stats.stall_us, tokens, repeats) << " ms\n";
    out << "  predictor      " << std::setw(8)
        << us_per_token_ms(stats.pred_us, tokens, repeats) << " ms\n\n";
}

void append_profiler_breakdown(std::ostringstream & out, const char * label,
        const profile_phase_stats & stats, int tokens, int repeats) {
    const uint64_t token_total = (uint64_t) std::max(1, tokens) * (uint64_t) std::max(1, repeats);

    out << "Profiler breakdown (" << label << ", mean per token):\n";
    out << "  predictor_observe " << std::setw(8)
        << us_per_token_ms(stats.pred_observe_us, tokens, repeats) << " ms\n";
    out << "  predictor_score   " << std::setw(8)
        << us_per_token_ms(stats.pred_score_us, tokens, repeats) << " ms\n";
    out << "  callback_wall     " << std::setw(8)
        << us_per_token_ms(stats.callback_wall_us, tokens, repeats) << " ms\n";
    out << "  topk_d2h          " << std::setw(8)
        << us_per_token_ms(stats.topk_d2h_us, tokens, repeats) << " ms\n";
    out << "  slot_ids_h2d      " << std::setw(8)
        << us_per_token_ms(stats.slot_ids_h2d_us, tokens, repeats) << " ms\n";
    out << "  slot_table_h2d    " << std::setw(8)
        << us_per_token_ms(stats.slot_table_h2d_us, tokens, repeats) << " ms\n";
    out << "  eamc_cosine       " << std::setw(8)
        << us_per_token_ms(stats.eamc_cosine_us, tokens, repeats) << " ms\n";
    out << "  eamc_materialize  " << std::setw(8)
        << us_per_token_ms(stats.eamc_score_materialize_us, tokens, repeats) << " ms\n";
    out << "  eamc_rows_scored  " << std::setw(8) << std::setprecision(1)
        << safe_div((double) stats.eamc_rows_scored, token_total) << " rows/token\n";
    out << std::setprecision(2);
    out << "  eamc_cache_hits   " << std::setw(8)
        << safe_div((double) stats.eamc_score_cache_hits, token_total) << " hits/token\n";
    out << "  eamc_cache_misses " << std::setw(8)
        << safe_div((double) stats.eamc_score_cache_misses, token_total) << " misses/token\n";
    out << "  request_end       " << std::setw(8)
        << us_per_token_ms(stats.request_end_us, tokens, repeats) << " ms\n";
    out << "  predictor_end     " << std::setw(8)
        << us_per_token_ms(stats.predictor_end_us, tokens, repeats) << " ms\n";
    out << "  predictor_save    " << std::setw(8)
        << us_per_token_ms(stats.predictor_save_us, tokens, repeats) << " ms\n";
    out << "  profile_flush     " << std::setw(8)
        << us_per_token_ms(stats.profile_flush_us, tokens, repeats) << " ms\n";
    out << "  sidecar_written   " << std::setw(8)
        << bytes_to_mib(safe_div((double) stats.sidecar_write_bytes, token_total))
        << " MB/token\n\n";
}

} // namespace

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

void profiler::reset(const std::string & csv_path) {
    flush();
    if (csv.is_open()) {
        csv.close();
    }
    prefill_stats = {};
    decode_stats = {};
    open(csv_path);
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
    stats.pred_us += row.pred_us;
    stats.pred_observe_us += row.pred_observe_us;
    stats.pred_score_us += row.pred_score_us;
    stats.callback_wall_us += row.callback_wall_us;
    stats.topk_d2h_us += row.topk_d2h_us;
    stats.slot_ids_h2d_us += row.slot_ids_h2d_us;
    stats.slot_table_h2d_us += row.slot_table_h2d_us;
    stats.eamc_rows_scored += row.eamc_rows_scored;
    stats.eamc_cosine_us += row.eamc_cosine_us;
    stats.eamc_score_materialize_us += row.eamc_score_materialize_us;
    stats.eamc_score_cache_hits += row.eamc_score_cache_hits;
    stats.eamc_score_cache_misses += row.eamc_score_cache_misses;
    if (row.cache_resident_experts > stats.cache_resident_peak) {
        stats.cache_resident_peak = row.cache_resident_experts;
    }

    if (csv.is_open()) {
        csv << "layer" << ','
            << row.request_idx << ','
            << row.repeat_idx << ','
            << row.batch_idx << ','
            << row.token_idx << ','
            << row.phase << ','
            << row.layer << ','
            << row.k_required << ','
            << row.k_hit << ','
            << row.k_miss << ','
            << row.ssd_read_us << ','
            << row.h2d_us << ','
            << row.compute_us << ','
            << row.stall_us << ','
            << row.pred_us << ','
            << row.pred_observe_us << ','
            << row.pred_score_us << ','
            << row.callback_wall_us << ','
            << row.topk_d2h_us << ','
            << row.slot_ids_h2d_us << ','
            << row.slot_table_h2d_us << ','
            << row.eamc_rows_scored << ','
            << row.eamc_cosine_us << ','
            << row.eamc_score_materialize_us << ','
            << row.eamc_score_cache_hits << ','
            << row.eamc_score_cache_misses << ','
            << row.cache_resident_experts << ','
            << row.predictor << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << '\n';
    }
}

void profiler::record_request(const profile_request_row & row) {
    profile_phase_stats & stats = row.phase == "prefill" ? prefill_stats : decode_stats;
    ++stats.requests;
    stats.request_wall_us += row.request_wall_us;
    stats.request_end_us += row.request_end_us;
    stats.predictor_end_us += row.predictor_end_us;
    stats.predictor_save_us += row.predictor_save_us;
    stats.profile_flush_us += row.profile_flush_us;
    stats.sidecar_write_bytes += row.sidecar_write_bytes;

    if (csv.is_open()) {
        csv << "request" << ','
            << row.request_idx << ','
            << row.repeat_idx << ','
            << row.batch_idx << ','
            << 0 << ','
            << row.phase << ','
            << -1 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << 0 << ','
            << "" << ','
            << row.request_wall_us << ','
            << row.request_end_us << ','
            << row.predictor_end_us << ','
            << row.predictor_save_us << ','
            << row.profile_flush_us << ','
            << row.sidecar_write_bytes << '\n';
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

// Legacy aggregate summary used by runtime end_request() for non-bench callers.
// The §4.7 llama-bench-style report is generated by format_summary() below.
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
    out << "pred_us: " << stats.pred_us << '\n';
    out << "pred_observe_us: " << stats.pred_observe_us << '\n';
    out << "pred_score_us: " << stats.pred_score_us << '\n';
    out << "callback_wall_us: " << stats.callback_wall_us << '\n';
    out << "topk_d2h_us: " << stats.topk_d2h_us << '\n';
    out << "slot_ids_h2d_us: " << stats.slot_ids_h2d_us << '\n';
    out << "slot_table_h2d_us: " << stats.slot_table_h2d_us << '\n';
    out << "eamc_rows_scored: " << stats.eamc_rows_scored << '\n';
    out << "eamc_cosine_us: " << stats.eamc_cosine_us << '\n';
    out << "eamc_score_materialize_us: " << stats.eamc_score_materialize_us << '\n';
    out << "eamc_score_cache_hits: " << stats.eamc_score_cache_hits << '\n';
    out << "eamc_score_cache_misses: " << stats.eamc_score_cache_misses << '\n';
    out << "request_wall_us: " << stats.request_wall_us << '\n';
    out << "request_end_us: " << stats.request_end_us << '\n';
    out << "predictor_end_us: " << stats.predictor_end_us << '\n';
    out << "predictor_save_us: " << stats.predictor_save_us << '\n';
    out << "profile_flush_us: " << stats.profile_flush_us << '\n';
    out << "sidecar_write_bytes: " << stats.sidecar_write_bytes << '\n';
    return out.str();
}

std::string format_summary(
        const profile_summary_context & ctx,
        const profile_snapshot & profile) {
    const double prefill_tok_s = ctx.n_prompt > 0 && ctx.ttft_ms > 0.0
        ? ctx.n_prompt / (ctx.ttft_ms / 1000.0)
        : 0.0;
    const double decode_total_ms = ctx.tpot_ms * ctx.n_gen;
    const double decode_tok_s = ctx.tpot_ms > 0.0 ? 1000.0 / ctx.tpot_ms : 0.0;
    const uint64_t decode_tokens_total = (uint64_t) std::max(1, ctx.n_gen) * (uint64_t) std::max(1, ctx.n_repeat);
    const double decode_bytes_per_token = (double) profile.decode.ssd_bytes / (double) decode_tokens_total;
    const double avg_read_mib = profile.decode.ssd_reads == 0 ? 0.0
        : bytes_to_mib((double) profile.decode.ssd_bytes / (double) profile.decode.ssd_reads);
    const double avg_read_latency_ms = profile.decode.ssd_reads == 0 ? 0.0
        : (double) profile.decode.ssd_read_us / 1000.0 / (double) profile.decode.ssd_reads;
    const double expert_cache_gib = (double) ctx.cache_mb / 1024.0;
    const double vram_peak_gib = bytes_to_gib(ctx.vram_peak_bytes);
    const double vram_other_gib = std::max(0.0, vram_peak_gib - expert_cache_gib);
    const int64_t profiled_decode_us =
        profile.decode.ssd_read_us +
        profile.decode.h2d_us +
        profile.decode.compute_us +
        profile.decode.stall_us +
        profile.decode.pred_us +
        profile.decode.request_end_us;
    const int64_t wall_decode_us = (int64_t) (ctx.tpot_ms * 1000.0 * (double) ctx.n_gen * (double) ctx.n_repeat);
    const int64_t unattributed_decode_us = wall_decode_us - profiled_decode_us;

    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "model: " << ctx.model << '\n';
    out << "predictor: " << std::left << std::setw(8) << ctx.predictor << std::right
        << "  cache: " << ctx.cache_mb << " MB"
        << "   ssd: " << ctx.storage << '\n';
    out << "n_prompt: " << ctx.n_prompt << "  n_gen: " << ctx.n_gen << "  repeats: " << ctx.n_repeat << "\n\n";
    if (ctx.n_ubatch > 0) {
        out << "ubatch: requested=" << ctx.n_ubatch_requested
            << " effective=" << ctx.n_ubatch;
        if (ctx.n_slots > 0 || ctx.n_experts > 0) {
            out << "  slots=" << ctx.n_slots << "/" << ctx.n_experts
                << "  mode=" << (ctx.streaming ? "streaming" : "full-residency");
        }
        out << "\n\n";
    }
    if (ctx.cache_reset_between_repeats || ctx.warm_cache || ctx.hot_start) {
        out << "calibration: "
            << "cache_reset_between_repeats=" << (ctx.cache_reset_between_repeats ? "true" : "false")
            << "  warm_cache=" << (ctx.warm_cache ? "true" : "false")
            << "  hot_start=" << (ctx.hot_start ? "true" : "false")
            << "\n\n";
    }

    out << "phase     tokens   total_ms   per_token_ms   tok/s\n";
    out << "prefill   " << std::setw(6) << ctx.n_prompt
        << "   " << std::setw(8) << std::setprecision(1) << ctx.ttft_ms
        << "        " << std::setw(6) << std::setprecision(2) << (ctx.n_prompt > 0 ? ctx.ttft_ms / ctx.n_prompt : 0.0)
        << "   " << std::setw(6) << std::setprecision(0) << prefill_tok_s << '\n';
    out << std::fixed << std::setprecision(1);
    out << "decode    " << std::setw(6) << ctx.n_gen
        << "   " << std::setw(8) << decode_total_ms
        << "        " << std::setw(6) << std::setprecision(2) << ctx.tpot_ms
        << "   " << std::setw(6) << std::setprecision(0) << decode_tok_s << "\n\n";

    out << std::fixed << std::setprecision(1);
    out << "cache hit rate (prefill): " << hit_rate_percent(profile.prefill) << "%\n";
    out << "cache hit rate (decode): " << hit_rate_percent(profile.decode) << "%\n";
    out << "SSD bytes read (decode): " << std::setprecision(2) << bytes_to_gib(profile.decode.ssd_bytes)
        << " GB  (avg " << bytes_to_mib(decode_bytes_per_token) << " MB/token)\n";
    out << "TTFT: " << std::setprecision(1) << ctx.ttft_ms << " ms\n";
    if (ctx.cold_prefill_count > 0 || ctx.warm_prefill_count > 0) {
        out << "TTFT cold: " << std::setprecision(1) << ctx.cold_ttft_ms
            << " ms  (n=" << ctx.cold_prefill_count << ")\n";
        out << "TTFT warm: " << std::setprecision(1) << ctx.warm_ttft_ms
            << " ms  (n=" << ctx.warm_prefill_count << ")\n";
    }
    out << "TPOT: " << std::setprecision(2) << ctx.tpot_ms << " ms\n";
    out << "total: " << std::setprecision(1) << ctx.total_ms << " ms\n\n";

    append_io_breakdown(out, "prefill", profile.prefill, ctx.n_prompt, ctx.n_repeat);
    append_profiler_breakdown(out, "prefill", profile.prefill, ctx.n_prompt, ctx.n_repeat);
    append_io_breakdown(out, "decode", profile.decode, ctx.n_gen, ctx.n_repeat);
    append_profiler_breakdown(out, "decode", profile.decode, ctx.n_gen, ctx.n_repeat);

    out << "Wall/profile reconciliation (decode):\n";
    out << "  wall_decode_us        " << wall_decode_us << '\n';
    out << "  profiled_decode_us    " << profiled_decode_us << '\n';
    out << "  unattributed_decode_us " << unattributed_decode_us << "\n\n";

    if (ctx.vram_peak_bytes > 0 || ctx.vram_total_bytes > 0) {
        out << "VRAM peak: " << std::setprecision(2) << vram_peak_gib << " GB";
        if (ctx.vram_total_bytes > 0) {
            out << " / " << bytes_to_gib(ctx.vram_total_bytes) << " GB";
        }
        out << "  (experts budget: " << expert_cache_gib
            << " GB, other model/kv/compute: " << vram_other_gib << " GB)\n";
    } else {
        out << "VRAM peak: unavailable"
            << "  (experts budget: " << expert_cache_gib << " GB)\n";
    }
    if (ctx.dram_peak_bytes > 0) {
        out << "DRAM peak (process): " << bytes_to_gib(ctx.dram_peak_bytes) << " GB\n";
    }
    out << "SSD reads: " << profile.decode.ssd_reads
        << " (avg " << avg_read_mib << " MB each, avg latency " << avg_read_latency_ms << " ms)\n";
    out << "profile rows: prefill=" << profile.prefill.rows << " decode=" << profile.decode.rows << '\n';
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
    stats.pred_us = prefill_stats.pred_us + decode_stats.pred_us;
    stats.pred_observe_us = prefill_stats.pred_observe_us + decode_stats.pred_observe_us;
    stats.pred_score_us = prefill_stats.pred_score_us + decode_stats.pred_score_us;
    stats.callback_wall_us = prefill_stats.callback_wall_us + decode_stats.callback_wall_us;
    stats.topk_d2h_us = prefill_stats.topk_d2h_us + decode_stats.topk_d2h_us;
    stats.slot_ids_h2d_us = prefill_stats.slot_ids_h2d_us + decode_stats.slot_ids_h2d_us;
    stats.slot_table_h2d_us = prefill_stats.slot_table_h2d_us + decode_stats.slot_table_h2d_us;
    stats.eamc_rows_scored = prefill_stats.eamc_rows_scored + decode_stats.eamc_rows_scored;
    stats.eamc_cosine_us = prefill_stats.eamc_cosine_us + decode_stats.eamc_cosine_us;
    stats.eamc_score_materialize_us = prefill_stats.eamc_score_materialize_us + decode_stats.eamc_score_materialize_us;
    stats.eamc_score_cache_hits = prefill_stats.eamc_score_cache_hits + decode_stats.eamc_score_cache_hits;
    stats.eamc_score_cache_misses = prefill_stats.eamc_score_cache_misses + decode_stats.eamc_score_cache_misses;
    stats.request_wall_us = prefill_stats.request_wall_us + decode_stats.request_wall_us;
    stats.request_end_us = prefill_stats.request_end_us + decode_stats.request_end_us;
    stats.predictor_end_us = prefill_stats.predictor_end_us + decode_stats.predictor_end_us;
    stats.predictor_save_us = prefill_stats.predictor_save_us + decode_stats.predictor_save_us;
    stats.profile_flush_us = prefill_stats.profile_flush_us + decode_stats.profile_flush_us;
    stats.sidecar_write_bytes = prefill_stats.sidecar_write_bytes + decode_stats.sidecar_write_bytes;
    stats.requests = prefill_stats.requests + decode_stats.requests;
    stats.cache_resident_peak = prefill_stats.cache_resident_peak > decode_stats.cache_resident_peak
        ? prefill_stats.cache_resident_peak
        : decode_stats.cache_resident_peak;
    return stats;
}

void profiler::write_header() {
    csv << "row_type,request_idx,repeat_idx,batch_idx,token_idx,phase,layer,k_required,k_hit,k_miss,ssd_read_us,h2d_us,compute_us,stall_us,pred_us,pred_observe_us,pred_score_us,callback_wall_us,topk_d2h_us,slot_ids_h2d_us,slot_table_h2d_us,eamc_rows_scored,eamc_cosine_us,eamc_score_materialize_us,eamc_score_cache_hits,eamc_score_cache_misses,cache_resident_experts,predictor,request_wall_us,request_end_us,predictor_end_us,predictor_save_us,profile_flush_us,sidecar_write_bytes\n";
}

} // namespace llama_moe
