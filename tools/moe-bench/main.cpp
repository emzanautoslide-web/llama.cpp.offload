// MoE offload benchmark tool.
// Runs prefill + decode loops and prints a formatted summary.

#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

struct bench_params {
    std::string model;
    std::string prompt;
    int n_prompt = 1024;
    int n_gen    = 256;
    int n_repeat = 3;
    int  moe_cache_mb = 8000;
    std::string moe_predictor = "eamc";
    std::string moe_profile_csv;
    std::string moe_profile_summary;
    int n_gpu_layers = 99;
    int n_ctx = 4096;
};

static bool parse_args(int argc, char ** argv, bench_params & p) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc)      { p.model = argv[++i]; }
        else if (arg == "--pp" && i+1 < argc)     { p.n_prompt = atoi(argv[++i]); }
        else if (arg == "--tg" && i+1 < argc)     { p.n_gen = atoi(argv[++i]); }
        else if (arg == "--repeat" && i+1 < argc) { p.n_repeat = atoi(argv[++i]); }
        else if (arg == "--moe-cache-vram-mb" && i+1 < argc) { p.moe_cache_mb = atoi(argv[++i]); }
        else if (arg == "--moe-predictor" && i+1 < argc)    { p.moe_predictor = argv[++i]; }
        else if (arg == "--moe-profile-csv" && i+1 < argc)  { p.moe_profile_csv = argv[++i]; }
        else if (arg == "--moe-profile-summary" && i+1 < argc) { p.moe_profile_summary = argv[++i]; }
        else if (arg == "-ngl" && i+1 < argc)     { p.n_gpu_layers = atoi(argv[++i]); }
        else if (arg == "-c" && i+1 < argc)       { p.n_ctx = atoi(argv[++i]); }
        else if (arg == "-p" && i+1 < argc)       { p.prompt = argv[++i]; }
    }
    return !p.model.empty();
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    bench_params p;
    if (!parse_args(argc, argv, p)) {
        fprintf(stderr, "Usage: llama-moe-bench --model <path> --pp N --tg N [--repeat N] [--moe-cache-vram-mb MB] [--moe-predictor lru|eamc] [--moe-profile-csv PATH] [--moe-profile-summary PATH]\n");
        return 1;
    }

    // Generate a prompt of the requested length
    std::string prompt_text = p.prompt;
    if (prompt_text.empty() && p.n_prompt > 0) {
        prompt_text.reserve((size_t)p.n_prompt * 8);
        for (int i = 0; i < p.n_prompt; ++i) prompt_text += "Hello. ";
    }

    // ── Init ──────────────────────────────────────────────────────────
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = p.n_gpu_layers;
    model_params.use_mmap = false;

#ifdef LLAMA_MOE_OFFLOAD
    model_params.moe_cache_vram_mb = (uint64_t) p.moe_cache_mb;
    model_params.moe_predictor     = p.moe_predictor.c_str();
    if (!p.moe_profile_csv.empty())
        model_params.moe_profile_csv = p.moe_profile_csv.c_str();
    if (!p.moe_profile_summary.empty())
        model_params.moe_profile_summary = p.moe_profile_summary.c_str();
#endif

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = p.n_ctx;
    ctx_params.n_batch = 2048;

    llama_model * model = llama_model_load_from_file(p.model.c_str(), model_params);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    // ── Tokenize prompt ───────────────────────────────────────────────
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int vocab_size = llama_vocab_n_tokens(vocab);

    int n_tokens_max = p.n_prompt + 128;
    std::vector<llama_token> prompt_tokens((size_t)n_tokens_max);
    int n_prompt_tokens = llama_tokenize(vocab,
        prompt_text.c_str(), (int)prompt_text.size(),
        prompt_tokens.data(), n_tokens_max, true, true);
    if (n_prompt_tokens < 0) {
        // Fallback: use dummy tokens
        n_prompt_tokens = p.n_prompt;
        for (int i = 0; i < n_prompt_tokens; ++i) prompt_tokens[i] = (i % 32000) + 1;
    }
    if (n_prompt_tokens > p.n_prompt) n_prompt_tokens = p.n_prompt;

    // ── Warmup ────────────────────────────────────────────────────────
    {
        int n_warmup = std::min(n_prompt_tokens, 16);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_warmup);
        llama_decode(ctx, batch);
    }

    // ── Benchmark loop ────────────────────────────────────────────────
    std::vector<double> ttft_ms, tpot_ms, total_ms;

    for (int rep = 0; rep < p.n_repeat; ++rep) {
        double t0 = now_ms();

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt_tokens);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill decode failed\n"); return 1;
        }
        double t1 = now_ms();
        ttft_ms.push_back(t1 - t0);

        // Decode loop
        float * logits = llama_get_logits_ith(ctx, 0);
        llama_token token = 0;
        float max_logit = logits[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (logits[v] > max_logit) { max_logit = logits[v]; token = v; }
        }

        for (int g = 0; g < p.n_gen; ++g) {
            batch = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at gen %d\n", g); break;
            }
            logits = llama_get_logits_ith(ctx, 0);
            token = 0; max_logit = logits[0];
            for (int v = 1; v < vocab_size; ++v) {
                if (logits[v] > max_logit) { max_logit = logits[v]; token = v; }
            }
        }
        double t2 = now_ms();
        tpot_ms.push_back((t2 - t1) / p.n_gen);
        total_ms.push_back(t2 - t0);
    }

    // ── Summary ───────────────────────────────────────────────────────
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    double avg_ttft = 0, avg_tpot = 0, avg_total = 0;
    for (int i = 0; i < p.n_repeat; ++i) {
        avg_ttft += ttft_ms[i]; avg_tpot += tpot_ms[i]; avg_total += total_ms[i];
    }
    avg_ttft /= p.n_repeat; avg_tpot /= p.n_repeat; avg_total /= p.n_repeat;

    fprintf(stdout, "\n======== MoE Offload Benchmark ========\n");
    fprintf(stdout, "model: %s\n", p.model.c_str());
    fprintf(stdout, "predictor: %-8s  cache: %d MB\n", p.moe_predictor.c_str(), p.moe_cache_mb);
    fprintf(stdout, "n_prompt: %d  n_gen: %d  repeats: %d\n\n", n_prompt_tokens, p.n_gen, p.n_repeat);

    fprintf(stdout, "phase     tokens   total_ms   per_token_ms   tok/s\n");
    fprintf(stdout, "prefill   %6d   %8.1f         %5.2f    %6.0f\n",
            n_prompt_tokens, avg_ttft, n_prompt_tokens > 0 ? avg_ttft / n_prompt_tokens : 0,
            n_prompt_tokens > 0 ? n_prompt_tokens / (avg_ttft / 1000.0) : 0);
    fprintf(stdout, "decode    %6d   %8.1f         %5.2f    %6.0f\n",
            p.n_gen, avg_tpot * p.n_gen, avg_tpot,
            avg_tpot > 0 ? 1.0 / (avg_tpot / 1000.0) : 0);
    fprintf(stdout, "\nTTFT: %.1f ms\n", avg_ttft);
    fprintf(stdout, "TPOT: %.2f ms\n", avg_tpot);
    fprintf(stdout, "total: %.1f ms\n", avg_total);

    if (!p.moe_profile_summary.empty()) {
        fprintf(stdout, "\nProfile summary written to: %s\n", p.moe_profile_summary.c_str());
    }
    fprintf(stdout, "========================================\n");

    return 0;
}
