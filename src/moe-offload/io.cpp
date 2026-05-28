#include "io.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

// No CUDA headers — this is a plain C++ file.  The worker only does fread
// into host buffers; H2D is handled by the callback via ggml_backend_tensor_set.

namespace llama_moe {
namespace {

// ── Staging buffer pool (plain host memory — no CUDA dependency) ───────

struct buffer_pool {
    std::vector<void *> free_list;
    size_t buf_size = 0;
    std::mutex mutex;
    std::condition_variable cv;

    void init(size_t size, int count) {
        buf_size = size;
        for (int i = 0; i < count; ++i) {
            void * p = malloc(size);
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
        for (void * p : free_list) free(p);
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
                // Push to completed list — caller will do H2D
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

int io_outstanding() {
    return g_worker.outstanding.load(std::memory_order_relaxed);
}

} // namespace llama_moe
