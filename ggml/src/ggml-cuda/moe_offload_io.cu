// ===========================================================================
// MoE expert-offloading CUDA plumbing.
//
// This file is logically owned by the moe-offload subsystem
// (src/moe-offload/) but lives inside the ggml-cuda backend so it can
// touch CUDA runtime APIs and the internal ggml_backend_cuda_context.
// It is auto-globbed into the ggml-cuda backend library by
// ggml/src/ggml-cuda/CMakeLists.txt.
//
// When LLAMA_MOE_OFFLOAD is not defined, this file compiles to nothing.
//
// Provides (Phase G of implementation_plan_mvp_20260529.md):
//   - pinned host staging alloc/free       (cudaHostAlloc/cudaFreeHost)
//   - async H2D + event record             (moe_h2d_stream + event pool)
//   - compute-stream wait on event         (cudaStreamWaitEvent)
//
// Symbols are exported as `extern "C"` so src/moe-offload/io.cpp (compiled
// by the C++ host compiler, not nvcc) can forward to them.
// ===========================================================================

#ifdef LLAMA_MOE_OFFLOAD

#include "common.cuh"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <vector>

namespace {

// One H2D stream per active device, created lazily on first use.
// MVP assumption: single CUDA device. Multi-device is post-MVP.
struct moe_io_state {
    std::mutex                mutex;
    cudaStream_t              h2d_stream = nullptr;
    int                       stream_device = -1;
    std::vector<cudaEvent_t>  event_pool;        // free events
    size_t                    events_in_use = 0; // for diagnostics

    static constexpr size_t kEventPoolHardCap = 4096;
};

moe_io_state g_state;

cudaStream_t ensure_h2d_stream() {
    // Caller must hold g_state.mutex.
    if (g_state.h2d_stream) return g_state.h2d_stream;
    int dev = -1;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaGetDevice failed: %s\n", cudaGetErrorString(e));
        return nullptr;
    }
    e = cudaStreamCreateWithFlags(&g_state.h2d_stream, cudaStreamNonBlocking);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaStreamCreate failed: %s\n", cudaGetErrorString(e));
        g_state.h2d_stream = nullptr;
        return nullptr;
    }
    g_state.stream_device = dev;
    return g_state.h2d_stream;
}

cudaEvent_t acquire_event() {
    // Caller must hold g_state.mutex.
    // Phase I: events are timing-capable so they can be used for both
    // cudaStreamWaitEvent (sync ordering) and cudaEventElapsedTime
    // (per-layer compute_us, per-miss h2d_us).
    if (!g_state.event_pool.empty()) {
        cudaEvent_t ev = g_state.event_pool.back();
        g_state.event_pool.pop_back();
        g_state.events_in_use++;
        return ev;
    }
    cudaEvent_t ev = nullptr;
    cudaError_t e = cudaEventCreateWithFlags(&ev, cudaEventDefault);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaEventCreate failed: %s\n", cudaGetErrorString(e));
        return nullptr;
    }
    g_state.events_in_use++;
    return ev;
}

void release_event_locked(cudaEvent_t ev) {
    // Caller must hold g_state.mutex.
    if (!ev) return;
    g_state.events_in_use--;
    if (g_state.event_pool.size() < moe_io_state::kEventPoolHardCap) {
        g_state.event_pool.push_back(ev);
    } else {
        cudaEventDestroy(ev);
    }
}

} // namespace

extern "C" {

// Pinned host allocation (cudaHostAllocPortable). Returns nullptr on failure.
GGML_BACKEND_API void * moe_io_cuda_pinned_alloc(size_t bytes) {
    void * p = nullptr;
    cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocPortable);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaHostAlloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(e));
        return nullptr;
    }
    return p;
}

GGML_BACKEND_API void moe_io_cuda_pinned_free(void * p) {
    if (!p) return;
    cudaError_t e = cudaFreeHost(p);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaFreeHost failed: %s\n", cudaGetErrorString(e));
    }
}

// Queue an async H2D copy on moe_h2d_stream and record an event on completion.
// *out_ev is set to a `cudaEvent_t` cast to `void *`. Returns false on error.
GGML_BACKEND_API bool moe_io_cuda_h2d_async(void * dst_dev, const void * src_pinned, size_t bytes, void ** out_ev) {
    if (!dst_dev || !src_pinned || bytes == 0 || !out_ev) return false;
    std::lock_guard<std::mutex> lock(g_state.mutex);
    cudaStream_t s = ensure_h2d_stream();
    if (!s) return false;
    cudaError_t e = cudaMemcpyAsync(dst_dev, src_pinned, bytes, cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaMemcpyAsync failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    cudaEvent_t ev = acquire_event();
    if (!ev) return false;
    e = cudaEventRecord(ev, s);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaEventRecord failed: %s\n", cudaGetErrorString(e));
        release_event_locked(ev);
        return false;
    }
    *out_ev = (void *) ev;
    return true;
}

GGML_BACKEND_API void moe_io_cuda_event_release(void * ev) {
    if (!ev) return;
    std::lock_guard<std::mutex> lock(g_state.mutex);
    release_event_locked((cudaEvent_t) ev);
}

// Make the backend's compute stream wait until *ev has signalled.
// `backend` must be a CUDA backend (cast to ggml_backend_cuda_context).
GGML_BACKEND_API bool moe_io_cuda_compute_wait(ggml_backend_t backend, void * ev) {
    if (!backend || !ev) return false;
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    cudaStream_t compute_stream = cuda_ctx->stream();
    cudaError_t e = cudaStreamWaitEvent(compute_stream, (cudaEvent_t) ev, 0);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaStreamWaitEvent failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// Block the calling host thread until *ev has signalled (test helper).
GGML_BACKEND_API bool moe_io_cuda_event_sync(void * ev) {
    if (!ev) return false;
    cudaError_t e = cudaEventSynchronize((cudaEvent_t) ev);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaEventSynchronize failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// Non-blocking completion query for an event.
GGML_BACKEND_API bool moe_io_cuda_event_query(void * ev) {
    if (!ev) return false;
    cudaError_t e = cudaEventQuery((cudaEvent_t) ev);
    if (e == cudaSuccess) {
        return true;
    }
    if (e == cudaErrorNotReady) {
        return false;
    }
    fprintf(stderr, "[moe-io-cuda] cudaEventQuery failed: %s\n", cudaGetErrorString(e));
    return false;
}

// Diagnostic: number of events currently checked out of the pool.
GGML_BACKEND_API size_t moe_io_cuda_events_in_use() {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    return g_state.events_in_use;
}

// ---------------------------------------------------------------------------
// Phase I: timing-event helpers.
//
// Events returned by the existing pool are timing-capable; the helpers below
// expose explicit "record on h2d / compute stream" + "query elapsed" so the
// eval-callback can fill profile_row.compute_us and profile_row.h2d_us with
// real CUDA-measured numbers (no host-side timing).
// ---------------------------------------------------------------------------

// Acquire one event from the pool. Returns nullptr on failure.
GGML_BACKEND_API void * moe_io_cuda_event_acquire() {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    cudaEvent_t ev = acquire_event();
    return (void *) ev;
}

// Record event on the dedicated MoE H2D stream.
GGML_BACKEND_API bool moe_io_cuda_record_on_h2d(void * ev) {
    if (!ev) return false;
    std::lock_guard<std::mutex> lock(g_state.mutex);
    cudaStream_t s = ensure_h2d_stream();
    if (!s) return false;
    cudaError_t e = cudaEventRecord((cudaEvent_t) ev, s);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] record_on_h2d cudaEventRecord failed: %s\n",
                cudaGetErrorString(e));
        return false;
    }
    return true;
}

// Record event on the backend's compute stream.
GGML_BACKEND_API bool moe_io_cuda_record_on_compute(ggml_backend_t backend, void * ev) {
    if (!backend || !ev) return false;
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    cudaStream_t compute_stream = cuda_ctx->stream();
    cudaError_t e = cudaEventRecord((cudaEvent_t) ev, compute_stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] record_on_compute cudaEventRecord failed: %s\n",
                cudaGetErrorString(e));
        return false;
    }
    return true;
}

// Synchronize on `end` and return elapsed time (begin -> end) in microseconds.
// Returns -1 on error.
GGML_BACKEND_API int64_t moe_io_cuda_elapsed_us(void * begin, void * end) {
    if (!begin || !end) return -1;
    cudaError_t e = cudaEventSynchronize((cudaEvent_t) end);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] elapsed_us cudaEventSynchronize failed: %s\n",
                cudaGetErrorString(e));
        return -1;
    }
    float ms = 0.0f;
    e = cudaEventElapsedTime(&ms, (cudaEvent_t) begin, (cudaEvent_t) end);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] cudaEventElapsedTime failed: %s\n",
                cudaGetErrorString(e));
        return -1;
    }
    if (ms < 0.0f) ms = 0.0f;
    return (int64_t) (ms * 1000.0f);
}

// Variant of moe_io_cuda_h2d_async that also records a begin event before the
// memcpy. Both events come from the same pool and are timing-capable, so they
// can be used both for cudaStreamWaitEvent (compute ordering on *ev_end) and
// for cudaEventElapsedTime (per-miss h2d_us).
GGML_BACKEND_API bool moe_io_cuda_h2d_async_timed(void * dst_dev, const void * src_pinned,
                                                  size_t bytes, void ** out_ev_begin,
                                                  void ** out_ev_end) {
    if (!dst_dev || !src_pinned || bytes == 0 || !out_ev_begin || !out_ev_end) return false;
    std::lock_guard<std::mutex> lock(g_state.mutex);
    cudaStream_t s = ensure_h2d_stream();
    if (!s) return false;

    cudaEvent_t ev_begin = acquire_event();
    if (!ev_begin) return false;
    cudaError_t e = cudaEventRecord(ev_begin, s);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] h2d_async_timed begin record failed: %s\n",
                cudaGetErrorString(e));
        release_event_locked(ev_begin);
        return false;
    }

    e = cudaMemcpyAsync(dst_dev, src_pinned, bytes, cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] h2d_async_timed cudaMemcpyAsync failed: %s\n",
                cudaGetErrorString(e));
        release_event_locked(ev_begin);
        return false;
    }

    cudaEvent_t ev_end = acquire_event();
    if (!ev_end) {
        release_event_locked(ev_begin);
        return false;
    }
    e = cudaEventRecord(ev_end, s);
    if (e != cudaSuccess) {
        fprintf(stderr, "[moe-io-cuda] h2d_async_timed end record failed: %s\n",
                cudaGetErrorString(e));
        release_event_locked(ev_begin);
        release_event_locked(ev_end);
        return false;
    }

    *out_ev_begin = (void *) ev_begin;
    *out_ev_end   = (void *) ev_end;
    return true;
}

} // extern "C"

#endif // LLAMA_MOE_OFFLOAD
