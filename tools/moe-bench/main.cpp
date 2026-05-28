// MoE offload benchmark tool.
// Runs prefill + decode loops and prints the MVP profiling summary.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "moe-offload/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <psapi.h>
#else
#  include <sys/resource.h>
#endif

#ifndef LLAMA_MOE_OFFLOAD
#  error "llama-moe-bench requires a build configured with LLAMA_MOE_OFFLOAD=ON"
#endif

struct bench_params {
    std::string model;
    std::string prompt;
    int n_prompt = 1024;
    int n_gen = 256;
    int n_repeat = 3;
    int moe_cache_mb = 8000;
    std::string moe_predictor = "eamc";
    std::string moe_profile_csv;
    std::string moe_profile_summary;
    int n_gpu_layers = 99;
    int n_ctx = 4096;
};

static bool parse_args(int argc, char ** argv, bench_params & p) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value_for = [&](const char * name, std::string & value) -> bool {
            const std::string key(name);
            if (arg == key && i + 1 < argc) {
                value = argv[++i];
                return true;
            }
            if (arg.rfind(key + "=", 0) == 0) {
                value = arg.substr(key.size() + 1);
                return true;
            }
            return false;
        };
        auto int_for = [&](const char * name, int & value) -> bool {
            std::string text;
            if (!value_for(name, text)) {
                return false;
            }
            value = std::atoi(text.c_str());
            return true;
        };

        if (value_for("--model", p.model)) {}
        else if (int_for("--pp", p.n_prompt)) {}
        else if (int_for("--tg", p.n_gen)) {}
        else if (int_for("--repeat", p.n_repeat)) {}
        else if (int_for("--moe-cache-vram-mb", p.moe_cache_mb)) {}
        else if (value_for("--moe-predictor", p.moe_predictor)) {}
        else if (value_for("--moe-profile-csv", p.moe_profile_csv)) {}
        else if (value_for("--moe-profile-summary", p.moe_profile_summary)) {}
        else if (int_for("-ngl", p.n_gpu_layers)) {}
        else if (int_for("-c", p.n_ctx)) {}
        else if (value_for("-p", p.prompt)) {}
    }
    if (p.n_repeat < 1) p.n_repeat = 1;
    if (p.n_prompt < 1) p.n_prompt = 1;
    if (p.n_gen < 1) p.n_gen = 1;
    return !p.model.empty();
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double bytes_to_gib(uint64_t bytes) {
    return (double) bytes / (1024.0 * 1024.0 * 1024.0);
}

static double bytes_to_mib(uint64_t bytes) {
    return (double) bytes / (1024.0 * 1024.0);
}

static double bytes_to_mib(double bytes) {
    return bytes / (1024.0 * 1024.0);
}

static std::string basename_of(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

static std::string storage_label_of(const std::string & path) {
    if (path.size() >= 2 && path[1] == ':') {
        return path.substr(0, 2);
    }
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static uint64_t process_dram_peak_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &counters, sizeof(counters))) {
        return (uint64_t) counters.PeakWorkingSetSize;
    }
    return 0;
#else
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#  if defined(__APPLE__)
    return (uint64_t) usage.ru_maxrss;
#  else
    return (uint64_t) usage.ru_maxrss * 1024ull;
#  endif
#endif
}

static uint64_t sample_vram_used_bytes(uint64_t * total_out = nullptr) {
    uint64_t used_sum = 0;
    uint64_t total_sum = 0;
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        if (total_bytes >= free_bytes) {
            used_sum += (uint64_t) (total_bytes - free_bytes);
            total_sum += (uint64_t) total_bytes;
        }
    }
    if (total_out) {
        *total_out = total_sum;
    }
    return used_sum;
}

static double hit_rate_percent(const llama_moe::profile_phase_stats & stats) {
    return stats.required == 0 ? 0.0 : 100.0 * (double) stats.hits / (double) stats.required;
}

static double us_per_token_ms(int64_t usec, int tokens, int repeat) {
    const int denom = std::max(1, tokens * repeat);
    return (double) usec / 1000.0 / (double) denom;
}

static std::string build_summary(
        const bench_params & p,
        const std::string & model_desc,
        int n_prompt_tokens,
        double avg_ttft,
        double avg_tpot,
        double avg_total,
        const llama_moe::profile_snapshot & profile,
        uint64_t vram_peak_bytes,
        uint64_t vram_total_bytes,
        uint64_t dram_peak_bytes) {
    const double prefill_tok_s = n_prompt_tokens > 0 ? n_prompt_tokens / (avg_ttft / 1000.0) : 0.0;
    const double decode_total_ms = avg_tpot * p.n_gen;
    const double decode_tok_s = avg_tpot > 0.0 ? 1000.0 / avg_tpot : 0.0;
    const uint64_t decode_tokens_total = (uint64_t) p.n_gen * (uint64_t) p.n_repeat;
    const uint64_t decode_ssd_bytes = profile.decode.ssd_bytes;
    const double decode_bytes_per_token = decode_tokens_total == 0 ? 0.0 : (double) decode_ssd_bytes / (double) decode_tokens_total;
    const double avg_read_mib = profile.decode.ssd_reads == 0 ? 0.0 : bytes_to_mib((double) profile.decode.ssd_bytes / (double) profile.decode.ssd_reads);
    const double avg_read_latency_ms = profile.decode.ssd_reads == 0 ? 0.0 : (double) profile.decode.ssd_read_us / 1000.0 / (double) profile.decode.ssd_reads;
    const double expert_cache_gib = (double) p.moe_cache_mb / 1024.0;
    const double vram_peak_gib = bytes_to_gib(vram_peak_bytes);
    const double vram_other_gib = std::max(0.0, vram_peak_gib - expert_cache_gib);

    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "model: " << (model_desc.empty() ? basename_of(p.model) : model_desc) << '\n';
    out << "predictor: " << std::left << std::setw(8) << p.moe_predictor << std::right
        << "  cache: " << p.moe_cache_mb << " MB"
        << "   ssd: " << storage_label_of(p.model) << '\n';
    out << "n_prompt: " << n_prompt_tokens << "  n_gen: " << p.n_gen << "  repeats: " << p.n_repeat << "\n\n";

    out << "phase     tokens   total_ms   per_token_ms   tok/s\n";
    out << "prefill   " << std::setw(6) << n_prompt_tokens
        << "   " << std::setw(8) << std::setprecision(1) << avg_ttft
        << "        " << std::setw(6) << std::setprecision(2) << (n_prompt_tokens > 0 ? avg_ttft / n_prompt_tokens : 0.0)
        << "   " << std::setw(6) << std::setprecision(0) << prefill_tok_s << '\n';
    out << std::fixed << std::setprecision(1);
    out << "decode    " << std::setw(6) << p.n_gen
        << "   " << std::setw(8) << decode_total_ms
        << "        " << std::setw(6) << std::setprecision(2) << avg_tpot
        << "   " << std::setw(6) << std::setprecision(0) << decode_tok_s << "\n\n";

    out << std::fixed << std::setprecision(1);
    out << "cache hit rate (prefill): " << hit_rate_percent(profile.prefill) << "%\n";
    out << "cache hit rate (decode): " << hit_rate_percent(profile.decode) << "%\n";
    out << "SSD bytes read (decode): " << std::setprecision(2) << bytes_to_gib(decode_ssd_bytes)
        << " GB  (avg " << bytes_to_mib(decode_bytes_per_token) << " MB/token)\n";
    out << "TTFT: " << std::setprecision(1) << avg_ttft << " ms\n";
    out << "TPOT: " << std::setprecision(2) << avg_tpot << " ms\n";
    out << "total: " << std::setprecision(1) << avg_total << " ms\n\n";

    out << "I/O breakdown (decode, mean per token):\n";
    out << "  ssd_read       " << std::setw(8) << std::setprecision(2) << us_per_token_ms(profile.decode.ssd_read_us, p.n_gen, p.n_repeat) << " ms\n";
    out << "  h2d            " << std::setw(8) << us_per_token_ms(profile.decode.h2d_us, p.n_gen, p.n_repeat) << " ms\n";
    out << "  gpu_compute    " << std::setw(8) << us_per_token_ms(profile.decode.compute_us, p.n_gen, p.n_repeat) << " ms\n";
    out << "  stall (overlap loss) " << std::setw(8) << us_per_token_ms(profile.decode.stall_us, p.n_gen, p.n_repeat) << " ms\n\n";

    out << "VRAM peak: " << std::setprecision(2) << vram_peak_gib << " GB";
    if (vram_total_bytes > 0) {
        out << " / " << bytes_to_gib(vram_total_bytes) << " GB";
    }
    out << "  (experts budget: " << expert_cache_gib << " GB, other model/kv/compute: " << vram_other_gib << " GB)\n";
    out << "DRAM peak (process): " << bytes_to_gib(dram_peak_bytes) << " GB\n";
    out << "SSD reads: " << profile.decode.ssd_reads
        << " (avg " << avg_read_mib << " MB each, avg latency " << avg_read_latency_ms << " ms)\n";
    out << "profile rows: prefill=" << profile.prefill.rows << " decode=" << profile.decode.rows << '\n';
    return out.str();
}

static llama_token greedy_token(llama_context * ctx, int vocab_size) {
    float * logits = llama_get_logits_ith(ctx, 0);
    if (!logits) {
        return 0;
    }
    llama_token token = 0;
    float max_logit = logits[0];
    for (int v = 1; v < vocab_size; ++v) {
        if (logits[v] > max_logit) {
            max_logit = logits[v];
            token = v;
        }
    }
    return token;
}

int main(int argc, char ** argv) {
    bench_params p;
    if (!parse_args(argc, argv, p)) {
        fprintf(stderr, "Usage: llama-moe-bench --model <path> --pp N --tg N [--repeat N] [--moe-cache-vram-mb MB] [--moe-predictor lru|eamc] [--moe-profile-csv PATH] [--moe-profile-summary PATH]\n");
        return 1;
    }

    std::string prompt_text = p.prompt;
    if (prompt_text.empty()) {
        prompt_text.reserve((size_t) p.n_prompt * 8);
        for (int i = 0; i < p.n_prompt; ++i) {
            prompt_text += "Hello. ";
        }
    }

    llama_backend_init();
    uint64_t vram_total_bytes = 0;
    uint64_t vram_peak_bytes = sample_vram_used_bytes(&vram_total_bytes);
    uint64_t dram_peak_bytes = process_dram_peak_bytes();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = p.n_gpu_layers;
    model_params.use_mmap = false;
    model_params.moe_offload = true;
    model_params.moe_cache_vram_mb = (uint64_t) p.moe_cache_mb;
    model_params.moe_predictor = p.moe_predictor.c_str();
    model_params.moe_profile_csv = p.moe_profile_csv.empty() ? nullptr : p.moe_profile_csv.c_str();
    // The runtime end_request() summary writer emits the legacy aggregate
    // format. llama-moe-bench owns --moe-profile-summary so it can write the
    // full §4.7 benchmark report after all repeats are complete.
    model_params.moe_profile_summary = nullptr;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = p.n_ctx;
    ctx_params.n_batch = std::max(2048, p.n_prompt);

    llama_model * model = llama_model_load_from_file(p.model.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        llama_backend_free();
        return 1;
    }
    vram_peak_bytes = std::max(vram_peak_bytes, sample_vram_used_bytes(&vram_total_bytes));
    dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    vram_peak_bytes = std::max(vram_peak_bytes, sample_vram_used_bytes(&vram_total_bytes));
    dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());

    char model_desc_buf[512] = {};
    std::string model_desc;
    if (llama_model_desc(model, model_desc_buf, sizeof(model_desc_buf)) > 0) {
        model_desc = model_desc_buf;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    const int n_tokens_max = p.n_prompt + 128;
    std::vector<llama_token> prompt_tokens((size_t) n_tokens_max);
    int n_prompt_tokens = llama_tokenize(vocab,
            prompt_text.c_str(), (int) prompt_text.size(),
            prompt_tokens.data(), n_tokens_max, true, true);
    if (n_prompt_tokens < 0) {
        n_prompt_tokens = p.n_prompt;
        for (int i = 0; i < n_prompt_tokens; ++i) {
            prompt_tokens[i] = (i % 32000) + 1;
        }
    }
    if (n_prompt_tokens > p.n_prompt) {
        n_prompt_tokens = p.n_prompt;
    }

    std::vector<double> ttft_ms;
    std::vector<double> tpot_ms;
    std::vector<double> total_ms;
    ttft_ms.reserve((size_t) p.n_repeat);
    tpot_ms.reserve((size_t) p.n_repeat);
    total_ms.reserve((size_t) p.n_repeat);

    int exit_code = 0;

    auto average_or_zero = [](const std::vector<double> & values) -> double {
        if (values.empty()) {
            return 0.0;
        }
        double total = 0.0;
        for (double value : values) {
            total += value;
        }
        return total / (double) values.size();
    };

    auto write_summary = [&](bool print_stdout) -> llama_moe::profile_snapshot {
        llama_moe::profile_summary_context summary_ctx;
        summary_ctx.model = model_desc.empty() ? basename_of(p.model) : model_desc;
        summary_ctx.predictor = p.moe_predictor;
        summary_ctx.storage = storage_label_of(p.model);
        summary_ctx.cache_mb = (uint64_t) p.moe_cache_mb;
        summary_ctx.n_prompt = n_prompt_tokens;
        summary_ctx.n_gen = p.n_gen;
        summary_ctx.n_repeat = (int) std::max<size_t>(1, std::max(ttft_ms.size(), total_ms.size()));
        summary_ctx.ttft_ms = average_or_zero(ttft_ms);
        summary_ctx.tpot_ms = average_or_zero(tpot_ms);
        summary_ctx.total_ms = total_ms.empty() ? summary_ctx.ttft_ms : average_or_zero(total_ms);
        summary_ctx.vram_peak_bytes = vram_peak_bytes;
        summary_ctx.vram_total_bytes = vram_total_bytes;
        summary_ctx.dram_peak_bytes = dram_peak_bytes;

        const llama_moe::profile_snapshot profile = llama_moe::get_profile_snapshot();
        const std::string summary = llama_moe::format_summary(summary_ctx, profile);

        if (print_stdout) {
            fputc('\n', stdout);
            fputs(summary.c_str(), stdout);
            fflush(stdout);
        }

        if (!p.moe_profile_summary.empty()) {
            fprintf(stderr, "[moe-bench] writing summary to: %s\n", p.moe_profile_summary.c_str());
            std::ofstream out(p.moe_profile_summary, std::ios::out | std::ios::trunc);
            if (!out) {
                fprintf(stderr, "[moe-bench] ERROR: failed to open summary file: %s\n", p.moe_profile_summary.c_str());
            } else {
                out << summary;
                out.close();
                if (out.fail()) {
                    fprintf(stderr, "[moe-bench] ERROR: write failed for summary file: %s\n", p.moe_profile_summary.c_str());
                } else if (print_stdout) {
                    fprintf(stderr, "[moe-bench] summary written successfully to: %s\n", p.moe_profile_summary.c_str());
                }
            }
        }

        return profile;
    };

    for (int rep = 0; rep < p.n_repeat; ++rep) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_perf_context_reset(ctx);

        const double t0 = now_ms();
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt_tokens);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill decode failed (rep %d)\n", rep);
            exit_code = 1;
            break;
        }
        const double t1 = now_ms();
        ttft_ms.push_back(t1 - t0);
        write_summary(false);

        llama_token token = greedy_token(ctx, vocab_size);
        for (int gen = 0; gen < p.n_gen; ++gen) {
            batch = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at gen %d (rep %d)\n", gen, rep);
                exit_code = 1;
                break;
            }
            token = greedy_token(ctx, vocab_size);
            vram_peak_bytes = std::max(vram_peak_bytes, sample_vram_used_bytes(&vram_total_bytes));
            dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
        }

        const double t2 = now_ms();
        tpot_ms.push_back((t2 - t1) / p.n_gen);
        total_ms.push_back(t2 - t0);
        vram_peak_bytes = std::max(vram_peak_bytes, sample_vram_used_bytes(&vram_total_bytes));
        dram_peak_bytes = std::max(dram_peak_bytes, process_dram_peak_bytes());
        write_summary(false);
    }

    fprintf(stderr, "[moe-bench] computing summary...\n");
    const llama_moe::profile_snapshot profile = write_summary(true);
    if (profile.prefill.rows + profile.decode.rows == 0) {
        fprintf(stderr, "warning: no MoE profile rows were recorded; check that the model is a repacked *.moe.gguf and that the run entered streaming mode\n");
    }

    fprintf(stderr, "[moe-bench] done. prefill_rows=%llu decode_rows=%llu\n",
            (unsigned long long) profile.prefill.rows,
            (unsigned long long) profile.decode.rows);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}