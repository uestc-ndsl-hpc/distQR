#pragma once
// Column-partitioned distributed blocked QR implementation.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include "panel_process.cuh"
#include "utils/cublas_gemm_traits.cuh"

namespace distributed_qr_col {

constexpr int kPanelWidth = 32;

template <typename T>
constexpr ncclDataType_t NcclType();

template <>
constexpr ncclDataType_t NcclType<float>() {
    return ncclFloat;
}

template <>
constexpr ncclDataType_t NcclType<double>() {
    return ncclDouble;
}

inline void AssertCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        spdlog::error("{}: {}", context, cudaGetErrorString(status));
        std::exit(1);
    }
}

inline void AssertCublas(cublasStatus_t status, const char* context) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        spdlog::error("{}: cublas error {}", context, static_cast<int>(status));
        std::exit(1);
    }
}

inline void AssertNccl(ncclResult_t status, const char* context) {
    if (status != ncclSuccess) {
        spdlog::error("{}: {}", context, ncclGetErrorString(status));
        std::exit(1);
    }
}

struct ColPartition {
    int n_global = 0;
    int world_size = 1;
    int rank = 0;
    int col_start = 0;
    int col_end = 0;
    int local_cols = 0;
};

inline ColPartition MakeColPartition(int n_global, int world_size, int rank) {
    ColPartition part{};
    part.n_global = n_global;
    part.world_size = world_size;
    part.rank = rank;

    const int total_panels = n_global / kPanelWidth;
    const int base = total_panels / world_size;
    const int rem = total_panels % world_size;
    const int local_panels = base + ((rank < rem) ? 1 : 0);
    const int panel_start = rank * base + std::min(rank, rem);

    part.col_start = panel_start * kPanelWidth;
    part.local_cols = local_panels * kPanelWidth;
    part.col_end = part.col_start + part.local_cols;
    return part;
}

inline int OwnerOfPanel(int panel_idx, int total_panels, int world_size) {
    const int base = total_panels / world_size;
    const int rem = total_panels % world_size;
    const int threshold = (base + 1) * rem;
    if (panel_idx < threshold) {
        return panel_idx / (base + 1);
    }
    if (base == 0) {
        return rem;
    }
    return rem + (panel_idx - threshold) / base;
}

template <typename T>
__global__ void write_upper_r_to_panel_kernel(
    int panel_row_begin, const T* R, int ldr, T* A, int lda) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= kPanelWidth || col >= kPanelWidth) {
        return;
    }

    const int dst_row = panel_row_begin + row;
    A[dst_row + static_cast<size_t>(col) * lda] =
        (row <= col) ? R[row + static_cast<size_t>(col) * ldr] : static_cast<T>(0);
}

template <typename T>
struct DistributedQrColWorkspace {
    T* d_r_panel = nullptr;  // [kPanelWidth x kPanelWidth]
    T* d_tsqr_work_panel = nullptr;
    size_t tsqr_work_panel_elems = 0;
    T* d_panel_w = nullptr;  // [m x kPanelWidth]
    T* d_panel_y = nullptr;  // [m x kPanelWidth]
    T* d_tmp0 = nullptr;     // [kPanelWidth x tile_cols]
    T* d_tmp1 = nullptr;     // [kPanelWidth x tile_cols]
    size_t tmp_elems = 0;
};

template <typename T>
void panel_update_tile_pipeline(cublasHandle_t cublas_handle,
                                cudaStream_t compute_stream,
                                cudaEvent_t w_ready,
                                cudaEvent_t y_ready,
                                int m,
                                int panel_col,
                                int panel_rows,
                                int cols_local,
                                int tile_cols,
                                const T* W,
                                const T* Y,
                                T* A_trail,
                                int lda_local,
                                T* d_tmp0,
                                T* d_tmp1) {
    if (cols_local <= 0 || panel_rows <= 0 || tile_cols <= 0) {
        return;
    }

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const T minus_one = static_cast<T>(-1);

    struct PersistentEvents {
        bool initialized = false;
        cudaEvent_t gemm1_done[2] = {};
        cudaEvent_t apply_done[2] = {};
    };
    static PersistentEvents events;
    if (!events.initialized) {
        AssertCuda(cudaEventCreateWithFlags(&events.gemm1_done[0], cudaEventDisableTiming),
                   "cudaEventCreate gemm1_done[0]");
        AssertCuda(cudaEventCreateWithFlags(&events.gemm1_done[1], cudaEventDisableTiming),
                   "cudaEventCreate gemm1_done[1]");
        AssertCuda(cudaEventCreateWithFlags(&events.apply_done[0], cudaEventDisableTiming),
                   "cudaEventCreate apply_done[0]");
        AssertCuda(cudaEventCreateWithFlags(&events.apply_done[1], cudaEventDisableTiming),
                   "cudaEventCreate apply_done[1]");
        events.initialized = true;
    }

    T* tmps[2] = {d_tmp0, d_tmp1};
    T* a_tiles[2] = {nullptr, nullptr};
    int widths[2] = {0, 0};

    AssertCuda(cudaStreamWaitEvent(compute_stream, w_ready, 0),
               "cudaStreamWaitEvent compute_stream <- w_ready");
    AssertCuda(cudaEventRecord(events.apply_done[0], compute_stream),
               "cudaEventRecord apply_done[0]");
    AssertCuda(cudaEventRecord(events.apply_done[1], compute_stream),
               "cudaEventRecord apply_done[1]");

    const int tile_count = (cols_local + tile_cols - 1) / tile_cols;
    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        const int idx = tile_idx & 1;
        const int col0 = tile_idx * tile_cols;
        const int width = std::min(tile_cols, cols_local - col0);
        T* a_tile = A_trail + static_cast<size_t>(col0) * lda_local;
        T* a_tile_sub = a_tile + panel_col;

        AssertCuda(cudaStreamWaitEvent(compute_stream, events.apply_done[idx], 0),
                   "cudaStreamWaitEvent compute_stream <- apply_done[idx]");
        AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kPanelWidth,
                                               width, panel_rows, &one, W, m, a_tile_sub, lda_local,
                                               &zero, tmps[idx], kPanelWidth),
                     "tmp = W^T * A_tile");
        AssertCuda(cudaEventRecord(events.gemm1_done[idx], compute_stream),
                   "cudaEventRecord gemm1_done[idx]");

        a_tiles[idx] = a_tile;
        widths[idx] = width;

        if (tile_idx > 0) {
            const int prev = idx ^ 1;
            T* a_prev_sub = a_tiles[prev] + panel_col;

            AssertCuda(cudaStreamWaitEvent(compute_stream, y_ready, 0),
                       "cudaStreamWaitEvent compute_stream <- y_ready");
            AssertCuda(cudaStreamWaitEvent(compute_stream, events.gemm1_done[prev], 0),
                       "cudaStreamWaitEvent compute_stream <- gemm1_done[prev]");
            AssertCublas(
                CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_rows,
                                          widths[prev], kPanelWidth, &minus_one, Y, m, tmps[prev],
                                          kPanelWidth, &one, a_prev_sub, lda_local),
                "A_tile -= Y * tmp");
            AssertCuda(cudaEventRecord(events.apply_done[prev], compute_stream),
                       "cudaEventRecord apply_done[prev]");
        }
    }

    if (tile_count > 0) {
        const int last = (tile_count - 1) & 1;
        T* a_last_sub = a_tiles[last] + panel_col;

        AssertCuda(cudaStreamWaitEvent(compute_stream, y_ready, 0),
                   "cudaStreamWaitEvent compute_stream <- y_ready(last)");
        AssertCuda(cudaStreamWaitEvent(compute_stream, events.gemm1_done[last], 0),
                   "cudaStreamWaitEvent compute_stream <- gemm1_done[last]");
        AssertCublas(
            CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_rows,
                                      widths[last], kPanelWidth, &minus_one, Y, m, tmps[last],
                                      kPanelWidth, &one, a_last_sub, lda_local),
            "A_last -= Y * tmp");
        AssertCuda(cudaEventRecord(events.apply_done[last], compute_stream),
                   "cudaEventRecord apply_done[last]");
    }
}

template <typename T>
void distributed_blocked_qr_factorize_col(cublasHandle_t cublas_handle,
                                          ncclComm_t nccl_comm,
                                          const ColPartition& part,
                                          int m,
                                          int n,
                                          int nb,
                                          T* d_A_local,
                                          int lda_local,
                                          T* d_W_local,
                                          T* d_Y_local,
                                          DistributedQrColWorkspace<T>* ws,
                                          cudaStream_t compute_stream,
                                          cudaStream_t comm_stream,
                                          int overlap_tile_cols = 0) {
    if (!ws) {
        spdlog::error("distributed_blocked_qr_factorize_col got null workspace.");
        std::exit(1);
    }
    if (n != part.n_global) {
        spdlog::error("ColPartition.n_global ({}) mismatches n ({}).", part.n_global, n);
        std::exit(1);
    }
    if (n % kPanelWidth != 0 || nb % kPanelWidth != 0) {
        spdlog::error("Require n and nb to be multiples of {} (got n={} nb={}).", kPanelWidth, n,
                      nb);
        std::exit(1);
    }

    const int tile_target = (overlap_tile_cols <= 0) ? nb : overlap_tile_cols;
    const int tile_cols = std::max(kPanelWidth, std::min(tile_target, nb));
    const size_t tmp_need = static_cast<size_t>(kPanelWidth) * static_cast<size_t>(tile_cols);
    if (ws->tmp_elems < tmp_need) {
        spdlog::error("Col workspace tmp is too small (need {} elems, got {}).", tmp_need,
                      ws->tmp_elems);
        std::exit(1);
    }
    if (!ws->d_tmp0 || !ws->d_tmp1) {
        spdlog::error("Col workspace tmp buffers are null.");
        std::exit(1);
    }

    AssertCublas(cublasSetStream(cublas_handle, compute_stream), "cublasSetStream(compute_stream)");
    const ncclDataType_t nccl_type = NcclType<T>();

    struct PersistentEvents {
        bool initialized = false;
        cudaEvent_t panel_ready = {};
        cudaEvent_t w_ready = {};
        cudaEvent_t y_ready = {};
        cudaEvent_t panel_update_done = {};
    };
    static PersistentEvents events;
    if (!events.initialized) {
        AssertCuda(cudaEventCreateWithFlags(&events.panel_ready, cudaEventDisableTiming),
                   "cudaEventCreate panel_ready");
        AssertCuda(cudaEventCreateWithFlags(&events.w_ready, cudaEventDisableTiming),
                   "cudaEventCreate w_ready");
        AssertCuda(cudaEventCreateWithFlags(&events.y_ready, cudaEventDisableTiming),
                   "cudaEventCreate y_ready");
        AssertCuda(cudaEventCreateWithFlags(&events.panel_update_done, cudaEventDisableTiming),
                   "cudaEventCreate panel_update_done");
        events.initialized = true;
    }

    const int total_panels = n / kPanelWidth;
    for (int panel_col = 0; panel_col < n; panel_col += kPanelWidth) {
        const int panel_idx = panel_col / kPanelWidth;
        const int owner = OwnerOfPanel(panel_idx, total_panels, part.world_size);
        const int local_panel_col = panel_col - part.col_start;
        const bool owner_has_panel = (owner == part.rank) && (local_panel_col >= 0) &&
                                     (local_panel_col + kPanelWidth <= part.local_cols);
        const int panel_rows = m - panel_col;

        if (owner_has_panel) {
            T* panel_A = d_A_local + static_cast<size_t>(local_panel_col) * lda_local;
            T* panel_A_sub = panel_A + panel_col;
            tsqr<T>(cublas_handle, panel_rows, panel_A_sub, lda_local, ws->d_r_panel, kPanelWidth,
                    ws->d_tsqr_work_panel, ws->tsqr_work_panel_elems, compute_stream);

            generate_wy<T>(panel_rows, kPanelWidth, panel_A_sub, lda_local,
                           ws->d_panel_y + panel_col, m, ws->d_panel_w + panel_col, m,
                           compute_stream);

            if (d_Y_local) {
                AssertCuda(
                    cudaMemcpy2DAsync(
                        d_Y_local + static_cast<size_t>(local_panel_col) * lda_local + panel_col,
                        static_cast<size_t>(lda_local) * sizeof(T), ws->d_panel_y + panel_col,
                        static_cast<size_t>(m) * sizeof(T),
                        static_cast<size_t>(panel_rows) * sizeof(T), kPanelWidth,
                        cudaMemcpyDeviceToDevice, compute_stream),
                    "cudaMemcpy2DAsync panel_y -> d_Y_local");
            }
            if (d_W_local) {
                AssertCuda(
                    cudaMemcpy2DAsync(
                        d_W_local + static_cast<size_t>(local_panel_col) * lda_local + panel_col,
                        static_cast<size_t>(lda_local) * sizeof(T), ws->d_panel_w + panel_col,
                        static_cast<size_t>(m) * sizeof(T),
                        static_cast<size_t>(panel_rows) * sizeof(T), kPanelWidth,
                        cudaMemcpyDeviceToDevice, compute_stream),
                    "cudaMemcpy2DAsync panel_w -> d_W_local");
            }

            const dim3 block_dim(16, 16);
            const dim3 grid_dim((kPanelWidth + block_dim.x - 1) / block_dim.x,
                                (kPanelWidth + block_dim.y - 1) / block_dim.y);
            write_upper_r_to_panel_kernel<<<grid_dim, block_dim, 0, compute_stream>>>(
                panel_col, ws->d_r_panel, kPanelWidth, panel_A, lda_local);
            AssertCuda(cudaGetLastError(), "write_upper_r_to_panel_kernel launch");

            AssertCuda(cudaEventRecord(events.panel_ready, compute_stream),
                       "cudaEventRecord panel_ready");
        }

        if (owner_has_panel) {
            AssertCuda(cudaStreamWaitEvent(comm_stream, events.panel_ready, 0),
                       "cudaStreamWaitEvent comm_stream <- panel_ready");
        }

        AssertNccl(ncclBroadcast(ws->d_panel_w + panel_col, ws->d_panel_w + panel_col,
                                 static_cast<size_t>(panel_rows) * kPanelWidth, nccl_type, owner,
                                 nccl_comm, comm_stream),
                   "ncclBroadcast panel W");
        AssertCuda(cudaEventRecord(events.w_ready, comm_stream), "cudaEventRecord w_ready");
        AssertNccl(ncclBroadcast(ws->d_panel_y + panel_col, ws->d_panel_y + panel_col,
                                 static_cast<size_t>(panel_rows) * kPanelWidth, nccl_type, owner,
                                 nccl_comm, comm_stream),
                   "ncclBroadcast panel Y");
        AssertCuda(cudaEventRecord(events.y_ready, comm_stream), "cudaEventRecord y_ready");

        const int trail_begin = panel_col + kPanelWidth;
        const int local_begin_global = std::max(trail_begin, part.col_start);
        if (local_begin_global >= part.col_end) {
            continue;
        }
        const int local_begin = local_begin_global - part.col_start;
        const int cols_local = part.col_end - local_begin_global;
        T* A_trail = d_A_local + static_cast<size_t>(local_begin) * lda_local;
        panel_update_tile_pipeline(cublas_handle, compute_stream, events.w_ready, events.y_ready, m,
                                   panel_col, panel_rows, cols_local, tile_cols,
                                   ws->d_panel_w + panel_col, ws->d_panel_y + panel_col, A_trail,
                                   lda_local, ws->d_tmp0, ws->d_tmp1);

        // d_panel_w/d_panel_y are shared scratch buffers across panels. Make comm stream wait
        // until this panel's update has finished before reusing them in the next panel.
        AssertCuda(cudaEventRecord(events.panel_update_done, compute_stream),
                   "cudaEventRecord panel_update_done");
        AssertCuda(cudaStreamWaitEvent(comm_stream, events.panel_update_done, 0),
                   "cudaStreamWaitEvent comm_stream <- panel_update_done");
    }
}

}  // namespace distributed_qr_col
