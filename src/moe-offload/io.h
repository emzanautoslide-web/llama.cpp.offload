#pragma once

#include "llama.h"

#include <cstdint>
#include <cstddef>
#include <vector>

// Forward declaration of the ggml backend handle so we don't pull
// ggml-backend.h into every translation unit that includes this header.
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace llama_moe {

// ---------------------------------------------------------------------------
// Threaded I/O pipeline for MoE expert offloading.
//
// Architecture:
//   eval-callback  ──(enqueue)──▶  SPSC ring  ──▶  I/O worker thread
//                                                      │
//   ggml_backend_tensor_set ◀─── pinned buffers ◀── fread from disk
//
// Flow per missed expert:
//   1. callback acquires a pinned staging buffer from the pool
//   2. callback pushes an io_request onto the ring (fread only — no H2D)
//   3. worker pops the request, freads expert blob into the pinned buffer
//   4. callback calls io_wait_all() to block until all freads complete
//   5. callback does ggml_backend_tensor_set from pinned buffers to GPU
//   6. callback releases pinned buffers back to pool
//
// This avoids CUDA stream/event complexity while still overlapping SSD
// reads across multiple experts via the worker thread.
// ---------------------------------------------------------------------------

struct io_request {
    int      layer;        // logical MoE layer index
    int32_t  expert;       // expert id (0..n_expert-1)
    int      kind;         // EXPERT_GATE / EXPERT_UP / EXPERT_DOWN
    int32_t  slot;         // destination slot index
    void *   pinned_buf;   // pinned host buffer (owned by pool)
    size_t   blob_size;    // bytes to read
    uint64_t file_offset;  // absolute byte offset in .moe.gguf
    char *   gpu_dst;      // GPU destination address (slot_tensor->data + slot * nb[2])

    // Phase H: async H2D plumbing.
    bool     h2d;          // when true, worker issues io_h2d_async after fread
    void *   h2d_event;    // opaque CUDA event handle filled by the worker on success
    // Phase I: paired begin event so the eval-callback can later query
    // cudaEventElapsedTime(begin, h2d_event) for real h2d_us.
    void *   h2d_begin_event;
    int64_t  ssd_read_us;  // worker-measured fread duration in microseconds
    bool     ok = true;     // false when seek/read failed; caller still owns pinned_buf
    size_t   bytes_read = 0;
    int      io_error = 0;
};

// One-time init.
// - source_path: path to the .moe.gguf (opened once by the worker)
// - blob_size_max: largest expert blob in bytes (for pinned buffer sizing)
// - n_buffers: number of pinned staging buffers in the ring pool
bool io_init(const char * source_path, size_t blob_size_max, int n_buffers);

// Shut down the worker thread and free all resources.
void io_shutdown();

// Acquire a pinned staging buffer from the pool. Blocks if none available.
void * io_acquire_buffer();

// Non-blocking variant. Returns nullptr if no staging buffer is currently free.
void * io_try_acquire_buffer();

// Release a pinned buffer back to the free pool.
void io_release_buffer(void * buf);

// Enqueue a read request. Returns false if the queue is full.
bool io_submit(struct io_request req);

// Block until all previously submitted reads are complete.
// Returns the number of requests drained.
int io_wait_all();

// Phase H: drain completed requests from the worker (does not block).
// Caller takes ownership of the returned records: must release any
// `pinned_buf`/`h2d_event` they contain.
std::vector<io_request> io_drain_completed();

// Number of outstanding (submitted but not yet waited) requests.
int io_outstanding();

// ---------------------------------------------------------------------------
// CUDA-backed pinned staging + async H2D + event plumbing (Phase G).
//
// These thin wrappers forward to symbols defined in
// `ggml/src/ggml-cuda/moe_offload_io.cu` when the build has CUDA enabled.
// On a CPU-only build they all return null/false; callers fall back to
// plain malloc + synchronous ggml_backend_tensor_set.
//
// Events are returned as opaque `void *` so this header does not depend on
// <cuda_runtime.h>.
// ---------------------------------------------------------------------------

// Allocate `bytes` of pinned host memory. Returns nullptr on failure or on a
// CPU-only build.
LLAMA_API void * io_pinned_alloc(size_t bytes);

// Free memory previously returned by io_pinned_alloc.
LLAMA_API void io_pinned_free(void * p);

// Queue an async host->device copy on the dedicated MoE H2D stream and
// record an event when it completes. *out_ev is set to an opaque handle.
// Returns false on error or if CUDA is unavailable.
LLAMA_API bool io_h2d_async(void * dst_dev, const void * src_pinned, size_t bytes, void ** out_ev);

// Return an opaque event handle back to the internal pool.
LLAMA_API void io_event_release(void * ev);

// Make the given CUDA backend's compute stream wait until *ev signals.
// Returns false if `backend` is not a CUDA backend or CUDA is unavailable.
LLAMA_API bool io_compute_wait(ggml_backend_t backend, void * ev);

// Block the calling thread until *ev signals (test/diagnostic helper).
// Returns false if CUDA is unavailable.
LLAMA_API bool io_event_sync(void * ev);

// Diagnostic: number of events currently held by callers (not in the pool).
LLAMA_API size_t io_events_in_use();

// ---------------------------------------------------------------------------
// Phase I: timing-event helpers used to measure compute_us / h2d_us.
// ---------------------------------------------------------------------------

// Variant of io_h2d_async that also records a begin event on the H2D stream
// before the memcpy. Both events come from the same pool; both are
// timing-capable so they can be used with io_event_elapsed_us. The end event
// (*out_ev_end) is also the one the compute stream should wait on via
// io_compute_wait. Returns false if CUDA is unavailable or any CUDA call
// fails.
LLAMA_API bool io_h2d_async_timed(void * dst_dev, const void * src_pinned, size_t bytes,
                                  void ** out_ev_begin, void ** out_ev_end);

// Acquire a single timing-capable event from the pool. Returns nullptr on
// failure or on a CPU-only build.
LLAMA_API void * io_event_acquire();

// Record `ev` on the dedicated MoE H2D stream.
LLAMA_API bool io_record_on_h2d(void * ev);

// Record `ev` on the backend's compute stream.
LLAMA_API bool io_record_on_compute(ggml_backend_t backend, void * ev);

// Synchronize on `end` then return elapsed time from `begin` to `end` in
// microseconds. Returns -1 on error or on a CPU-only build.
LLAMA_API int64_t io_event_elapsed_us(void * begin, void * end);

} // namespace llama_moe
