#include "io.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

// No CUDA headers in this translation unit. The worker freads into pinned
// host buffers and (when CUDA is available) forwards the H2D copy to the
// extern "C" shims in ggml/src/ggml-cuda/moe_offload_io.cu via the
// io_h2d_async / io_pinned_alloc surface declared in io.h.

namespace llama_moe {
namespace {

// ── Staging buffer pool (pinned when CUDA is available, malloc otherwise) ──

struct buffer_pool {
    std::vector<void *> free_list;
    size_t buf_size = 0;
    bool   pinned   = false;   // true when buffers came from io_pinned_alloc
    std::mutex mutex;
    std::condition_variable cv;

    void init(size_t size, int count) {
        buf_size = size;
        // Try pinned host memory first (overlaps with CUDA async copies).
        // Fall back to malloc on CPU-only builds or if pinned alloc fails.
        for (int i = 0; i < count; ++i) {
            void * p = io_pinned_alloc(size);
            if (p) {
                pinned = true;
            } else {
                p = malloc(size);
            }
            free_list.push_back(p);
        }
    }

    void * acquire() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !free_list.empty(); });
        void * p = free_list.back();
        free_list.pop_back();
        return p;
    }

    void release(void * p) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            free_list.push_back(p);
        }
        cv.notify_one();
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex);
        for (void * p : free_list) {
            if (pinned) io_pinned_free(p);
            else        free(p);
        }
        free_list.clear();
    }
};

// ── SPSC work queue ────────────────────────────────────────────────────

struct work_queue {
    static constexpr int kCapacity = 256;
    io_request items[kCapacity];
    std::atomic<int> head{0};
    std::atomic<int> tail{0};
    std::mutex mutex;
    std::condition_variable cv;

    bool push(const io_request & req) {
        int h = head.load(std::memory_order_relaxed);
        int t = tail.load(std::memory_order_acquire);
        int next = (h + 1) % kCapacity;
        if (next == t) return false;
        items[h] = req;
        head.store(next, std::memory_order_release);
        cv.notify_one();
        return true;
    }

    bool pop(io_request & req) {
        int t = tail.load(std::memory_order_relaxed);
        int h = head.load(std::memory_order_acquire);
        if (t == h) return false;
        req = items[t];
        tail.store((t + 1) % kCapacity, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return tail.load(std::memory_order_acquire) == head.load(std::memory_order_acquire);
    }
};

// ── Completed request list ─────────────────────────────────────────────

struct completed_list {
    std::vector<io_request> items;
    std::mutex mutex;

    void push(const io_request & req) {
        std::lock_guard<std::mutex> lock(mutex);
        items.push_back(req);
    }

    std::vector<io_request> drain() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<io_request> result;
        result.swap(items);
        return result;
    }
};

// ── I/O worker (fread only — no CUDA) ──────────────────────────────────

struct io_worker {
    std::thread   thread;
    std::atomic<bool> stop_flag{false};
    work_queue    queue;
    completed_list done;
    buffer_pool   pool;
    FILE *        fp = nullptr;
    std::atomic<int> outstanding{0};

    void run() {
        io_request req;
        while (!stop_flag.load(std::memory_order_relaxed)) {
            {
                std::unique_lock<std::mutex> lock(queue.mutex);
                queue.cv.wait(lock, [this] {
                    return !queue.empty() || stop_flag.load(std::memory_order_relaxed);
                });
            }
            if (stop_flag.load(std::memory_order_relaxed) && queue.empty()) break;

            while (queue.pop(req)) {
                const auto read_start = std::chrono::steady_clock::now();
                int rc = _fseeki64(fp, (int64_t) req.file_offset, SEEK_SET);
                if (rc != 0) {
                    fprintf(stderr, "[moe-io] seek to %llu failed\n",
                            (unsigned long long) req.file_offset);
                    pool.release(req.pinned_buf);
                    outstanding.fetch_sub(1, std::memory_order_relaxed);
                    continue;
                }
                size_t got = fread(req.pinned_buf, 1, req.blob_size, fp);
                if (got != req.blob_size) {
                    fprintf(stderr, "[moe-io] short read: got %zu of %zu\n", got, req.blob_size);
                    pool.release(req.pinned_buf);
                    outstanding.fetch_sub(1, std::memory_order_relaxed);
                    continue;
                }
                const auto read_end = std::chrono::steady_clock::now();
                req.ssd_read_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        read_end - read_start).count();

                // Phase H: issue async H2D into the slot's GPU address. When
                // CUDA is unavailable or the call fails, leave h2d_event null
                // and let the caller fall back to ggml_backend_tensor_set.
                // Phase I: also record a begin event so the eval-callback can
                // compute real h2d_us via cudaEventElapsedTime.
                req.h2d_event       = nullptr;
                req.h2d_begin_event = nullptr;
                if (req.h2d && req.gpu_dst) {
                    void * ev_begin = nullptr;
                    void * ev_end   = nullptr;
                    if (io_h2d_async_timed(req.gpu_dst, req.pinned_buf, req.blob_size,
                                           &ev_begin, &ev_end)) {
                        req.h2d_begin_event = ev_begin;
                        req.h2d_event       = ev_end;
                    } else {
                        static bool warned = false;
                        if (!warned) {
                            warned = true;
                            fprintf(stderr, "[moe-io] io_h2d_async_timed failed; "
                                            "falling back to sync H2D\n");
                        }
                    }
                }

                // Push to completed list — caller drains, performs sync H2D
                // fallback if needed, waits on the event, then recycles.
                done.push(req);
                outstanding.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    void start(const char * source_path) {
        fp = fopen(source_path, "rb");
        if (!fp) {
            fprintf(stderr, "[moe-io] failed to open %s\n", source_path);
            return;
        }
        stop_flag.store(false, std::memory_order_relaxed);
        thread = std::thread(&io_worker::run, this);
    }

    void stop() {
        stop_flag.store(true, std::memory_order_relaxed);
        queue.cv.notify_all();
        if (thread.joinable()) thread.join();
        if (fp) { fclose(fp); fp = nullptr; }
        pool.shutdown();
    }
};

io_worker g_worker;

} // namespace

// ── Public API ─────────────────────────────────────────────────────────

bool io_init(const char * source_path, size_t blob_size_max, int n_buffers) {
    g_worker.pool.init(blob_size_max, n_buffers);
    g_worker.start(source_path);
    return (g_worker.fp != nullptr);
}

void io_shutdown() { g_worker.stop(); }

void * io_acquire_buffer() { return g_worker.pool.acquire(); }

void io_release_buffer(void * buf) { g_worker.pool.release(buf); }

bool io_submit(struct io_request req) {
    g_worker.outstanding.fetch_add(1, std::memory_order_relaxed);
    return g_worker.queue.push(req);
}

int io_wait_all() {
    // Spin until all outstanding requests complete
    while (g_worker.outstanding.load(std::memory_order_relaxed) > 0) {
        std::this_thread::yield();
    }
    // Drain completed items — caller does H2D and releases buffers
    auto done = g_worker.done.drain();
    return (int) done.size();
}

std::vector<io_request> io_drain_completed() {
    return g_worker.done.drain();
}

int io_outstanding() {
    return g_worker.outstanding.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// CUDA-backed pinned staging / async H2D / event helpers (Phase G).
//
// When the build links against the ggml-cuda backend (GGML_USE_CUDA is
// defined for ggml.dll consumers), forward to the extern "C" symbols
// implemented in `ggml/src/ggml-cuda/moe_offload_io.cu`. Otherwise compile
// no-op stubs so callers can probe at runtime via the nullptr/false return.
// ---------------------------------------------------------------------------

#if defined(GGML_USE_CUDA)

extern "C" {
    GGML_BACKEND_API void * moe_io_cuda_pinned_alloc (size_t bytes);
    GGML_BACKEND_API void   moe_io_cuda_pinned_free  (void * p);
    GGML_BACKEND_API bool   moe_io_cuda_h2d_async    (void * dst_dev, const void * src_pinned,
                                                      size_t bytes, void ** out_ev);
    GGML_BACKEND_API void   moe_io_cuda_event_release(void * ev);
    GGML_BACKEND_API bool   moe_io_cuda_compute_wait (ggml_backend_t backend, void * ev);
    GGML_BACKEND_API bool   moe_io_cuda_event_sync   (void * ev);
    GGML_BACKEND_API size_t moe_io_cuda_events_in_use(void);
    GGML_BACKEND_API void * moe_io_cuda_event_acquire (void);
    GGML_BACKEND_API bool   moe_io_cuda_record_on_h2d (void * ev);
    GGML_BACKEND_API bool   moe_io_cuda_record_on_compute(ggml_backend_t backend, void * ev);
    GGML_BACKEND_API int64_t moe_io_cuda_elapsed_us   (void * begin, void * end);
    GGML_BACKEND_API bool   moe_io_cuda_h2d_async_timed(void * dst_dev, const void * src_pinned,
                                                        size_t bytes, void ** out_ev_begin,
                                                        void ** out_ev_end);
}

void * io_pinned_alloc(size_t bytes)                       { return moe_io_cuda_pinned_alloc(bytes); }
void   io_pinned_free (void * p)                           { moe_io_cuda_pinned_free(p); }
bool   io_h2d_async   (void * d, const void * s, size_t n,
                       void ** out_ev)                     { return moe_io_cuda_h2d_async(d, s, n, out_ev); }
void   io_event_release(void * ev)                         { moe_io_cuda_event_release(ev); }
bool   io_compute_wait(ggml_backend_t b, void * ev)        { return moe_io_cuda_compute_wait(b, ev); }
bool   io_event_sync  (void * ev)                          { return moe_io_cuda_event_sync(ev); }
size_t io_events_in_use()                                  { return moe_io_cuda_events_in_use(); }
bool   io_h2d_async_timed(void * d, const void * s, size_t n,
                          void ** evb, void ** eve)        { return moe_io_cuda_h2d_async_timed(d, s, n, evb, eve); }
void * io_event_acquire ()                                 { return moe_io_cuda_event_acquire(); }
bool   io_record_on_h2d (void * ev)                        { return moe_io_cuda_record_on_h2d(ev); }
bool   io_record_on_compute(ggml_backend_t b, void * ev)   { return moe_io_cuda_record_on_compute(b, ev); }
int64_t io_event_elapsed_us(void * b, void * e)            { return moe_io_cuda_elapsed_us(b, e); }

#else // !GGML_USE_CUDA — stubs for CPU-only build

void * io_pinned_alloc(size_t)                             { return nullptr; }
void   io_pinned_free (void *)                             { }
bool   io_h2d_async   (void *, const void *, size_t, void **) { return false; }
void   io_event_release(void *)                            { }
bool   io_compute_wait(ggml_backend_t, void *)             { return false; }
bool   io_event_sync  (void *)                             { return false; }
size_t io_events_in_use()                                  { return 0; }
bool   io_h2d_async_timed(void *, const void *, size_t, void **, void **) { return false; }
void * io_event_acquire ()                                 { return nullptr; }
bool   io_record_on_h2d (void *)                           { return false; }
bool   io_record_on_compute(ggml_backend_t, void *)        { return false; }
int64_t io_event_elapsed_us(void *, void *)                { return -1; }

#endif // GGML_USE_CUDA

} // namespace llama_moe
