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
#include <vector>

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
    // Packed (contiguous) W/Y for a single panel: [panel_rows x kPanelWidth].
    // Used for communication and in-block updates.
    T* d_pack_w = nullptr;
    T* d_pack_y = nullptr;
    size_t pack_elems = 0;

    // Block-level WY buffers for a single outer block (nb columns).
    // Layout: column-major [m x nb], with zeros in rows [block_begin, inner) for panel columns.
    // Used for the block trailing update (single large GEMM like the single-GPU algorithm).
    T* d_block_w = nullptr;
    T* d_block_y = nullptr;
    size_t block_storage_elems = 0;

    // Work buffers for GEMM: [k x tile_cols], k up to nb.
    T* d_tmp0 = nullptr;
    T* d_tmp1 = nullptr;
    size_t tmp_elems = 0;
};

struct CommProfile {
    size_t bytes = 0;
};

template <typename T>
void panel_update_tile_pipeline(cublasHandle_t cublas_handle,
                                cudaStream_t compute_stream,
                                cudaEvent_t w_ready,
                                cudaEvent_t y_ready,
                                int panel_col,
                                int panel_rows,
                                int cols_local,
                                int tile_cols,
                                const T* W,
                                const T* Y,
                                int lda_wy,
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
                                               width, panel_rows, &one, W, lda_wy, a_tile_sub,
                                               lda_local, &zero, tmps[idx], kPanelWidth),
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
                                          widths[prev], kPanelWidth, &minus_one, Y, lda_wy,
                                          tmps[prev], kPanelWidth, &one, a_prev_sub, lda_local),
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
                                      widths[last], kPanelWidth, &minus_one, Y, lda_wy, tmps[last],
                                      kPanelWidth, &one, a_last_sub, lda_local),
            "A_last -= Y * tmp");
        AssertCuda(cudaEventRecord(events.apply_done[last], compute_stream),
                   "cudaEventRecord apply_done[last]");
    }
}

template <typename T>
void panel_update_one_shot(cublasHandle_t cublas_handle,
                           int panel_col,
                           int panel_rows,
                           int cols_local,
                           const T* W,
                           const T* Y,
                           int lda_wy,
                           T* A_trail,
                           int lda_local,
                           T* work,
                           size_t work_elems) {
    if (panel_rows <= 0 || cols_local <= 0) {
        return;
    }

    const size_t need = static_cast<size_t>(kPanelWidth) * static_cast<size_t>(cols_local);
    if (work_elems < need) {
        spdlog::error("panel_update_one_shot work too small (need {} elems, got {}).", need,
                      work_elems);
        std::exit(1);
    }

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const T minus_one = static_cast<T>(-1);

    T* a_sub = A_trail + panel_col;
    AssertCublas(
        CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kPanelWidth, cols_local,
                                  panel_rows, &one, W, lda_wy, a_sub, lda_local, &zero, work,
                                  kPanelWidth),
        "in-block work = W^T * A");
    AssertCublas(
        CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_rows, cols_local,
                                  kPanelWidth, &minus_one, Y, lda_wy, work, kPanelWidth, &one,
                                  a_sub, lda_local),
        "in-block A -= Y * work");
}

template <typename T>
void block_update_tile_pipeline(cublasHandle_t cublas_handle,
                                cudaStream_t compute_stream,
                                int row_offset,
                                int rows,
                                int k,
                                int cols_local,
                                int tile_cols,
                                const T* W,
                                const T* Y,
                                int lda_wy,
                                T* A_trail,
                                int lda_local,
                                T* d_tmp0,
                                T* d_tmp1) {
    if (k <= 0 || cols_local <= 0 || rows <= 0 || tile_cols <= 0) {
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
        T* a_tile_sub = a_tile + row_offset;

        AssertCuda(cudaStreamWaitEvent(compute_stream, events.apply_done[idx], 0),
                   "cudaStreamWaitEvent compute_stream <- apply_done[idx]");
        AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, width,
                                               rows, &one, W, lda_wy, a_tile_sub, lda_local,
                                               &zero, tmps[idx], k),
                     "tmp = W^T * A_tile");
        AssertCuda(cudaEventRecord(events.gemm1_done[idx], compute_stream),
                   "cudaEventRecord gemm1_done[idx]");

        a_tiles[idx] = a_tile;
        widths[idx] = width;

        if (tile_idx > 0) {
            const int prev = idx ^ 1;
            T* a_prev_sub = a_tiles[prev] + row_offset;
            AssertCuda(cudaStreamWaitEvent(compute_stream, events.gemm1_done[prev], 0),
                       "cudaStreamWaitEvent compute_stream <- gemm1_done[prev]");
            AssertCublas(
                CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, rows,
                                          widths[prev], k, &minus_one, Y, lda_wy, tmps[prev], k,
                                          &one, a_prev_sub, lda_local),
                "A_tile -= Y * tmp");
            AssertCuda(cudaEventRecord(events.apply_done[prev], compute_stream),
                       "cudaEventRecord apply_done[prev]");
        }
    }

    if (tile_count > 0) {
        const int last = (tile_count - 1) & 1;
        T* a_last_sub = a_tiles[last] + row_offset;
        AssertCuda(cudaStreamWaitEvent(compute_stream, events.gemm1_done[last], 0),
                   "cudaStreamWaitEvent compute_stream <- gemm1_done[last]");
        AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, rows,
                                              widths[last], k, &minus_one, Y, lda_wy, tmps[last], k,
                                              &one, a_last_sub, lda_local),
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
                                          int overlap_tile_cols = 0,
                                          CommProfile* comm_profile = nullptr) {
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
    const int panel_block_cols = nb;
    const size_t tmp_need = static_cast<size_t>(nb) * static_cast<size_t>(tile_cols);
    if (ws->tmp_elems < tmp_need) {
        spdlog::error("Col workspace tmp is too small (need {} elems, got {}).", tmp_need,
                      ws->tmp_elems);
        std::exit(1);
    }
    if (!ws->d_tmp0 || !ws->d_tmp1) {
        spdlog::error("Col workspace tmp buffers are null.");
        std::exit(1);
    }
    if (comm_profile) {
        comm_profile->bytes = 0;
    }
    const size_t pack_need = static_cast<size_t>(m) * static_cast<size_t>(kPanelWidth);
    if (!ws->d_pack_w || !ws->d_pack_y || ws->pack_elems < pack_need) {
        spdlog::error("Col workspace pack buffers too small (need {} elems, got {}).", pack_need,
                      ws->pack_elems);
        std::exit(1);
    }
    const size_t block_storage_need =
        static_cast<size_t>(m) * static_cast<size_t>(std::max(panel_block_cols, kPanelWidth));
    if (!ws->d_block_w || !ws->d_block_y || ws->block_storage_elems < block_storage_need) {
        spdlog::error("Col workspace block storage too small (need {} elems, got {}).",
                      block_storage_need, ws->block_storage_elems);
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
    for (int block_begin = 0; block_begin < n; block_begin += panel_block_cols) {
        const int block_end = std::min(block_begin + panel_block_cols, n);
        const int kb = block_end - block_begin;

        // Even if this rank has no local columns in/after this block, it must still
        // participate in panel broadcasts to match send/recv.

        // Build block-level WY factors in ws->d_block_{w,y} for this nb-block.
        // Only the rows [block_begin, m) are used by the block update.
        for (int inner = block_begin; inner < block_end; inner += kPanelWidth) {
            const int panel_idx = inner / kPanelWidth;
            const int owner = OwnerOfPanel(panel_idx, total_panels, part.world_size);
            const int panel_rows = m - inner;
            const int k_prev = inner - block_begin;
            const int block_col_off = inner - block_begin;

            if (panel_rows <= 0) {
                continue;
            }

            // Owner computes the panel WY in packed form, then updates W_i using the current
            // block WY (single-GPU style), then broadcasts packed W/Y to all ranks.
            bool owner_prepared = false;
            if (part.rank == owner) {
                const int local_panel_col = inner - part.col_start;
                const bool owner_has_panel =
                    (local_panel_col >= 0) && (local_panel_col + kPanelWidth <= part.local_cols);
                if (owner_has_panel) {
                    T* panel_A = d_A_local + static_cast<size_t>(local_panel_col) * lda_local;
                    T* panel_A_sub = panel_A + inner;

                    tsqr<T>(cublas_handle, panel_rows, panel_A_sub, lda_local, ws->d_r_panel,
                            kPanelWidth, ws->d_tsqr_work_panel, ws->tsqr_work_panel_elems,
                            compute_stream);

                    generate_wy<T>(panel_rows, kPanelWidth, panel_A_sub, lda_local, ws->d_pack_y,
                                   panel_rows, ws->d_pack_w, panel_rows, compute_stream);

                    // Update W_i in packed buffer so that stacking within the block forms a
                    // single compact WY transform (same recurrence as row-partition/single GPU).
                    if (k_prev > 0) {
                        const T one = static_cast<T>(1);
                        const T zero = static_cast<T>(0);
                        const T minus_one = static_cast<T>(-1);

                        const T* y_prev_sub =
                            ws->d_block_y + static_cast<size_t>(inner) +
                            static_cast<size_t>(0) * static_cast<size_t>(m);
                        const T* w_prev_sub =
                            ws->d_block_w + static_cast<size_t>(inner) +
                            static_cast<size_t>(0) * static_cast<size_t>(m);

                        // tmp = Y_prev^T * W_i  (k_prev x kPanelWidth)
                        AssertCublas(CublasGemmTraits<T>::Gemm(
                                         cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k_prev,
                                         kPanelWidth, panel_rows, &one, y_prev_sub, m,
                                         ws->d_pack_w, panel_rows, &zero, ws->d_tmp0, k_prev),
                                     "tmp = Y_prev^T * W_i");
                        // W_i -= W_prev * tmp  (panel_rows x kPanelWidth)
                        AssertCublas(CublasGemmTraits<T>::Gemm(
                                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_rows,
                                         kPanelWidth, k_prev, &minus_one, w_prev_sub, m, ws->d_tmp0,
                                         k_prev, &one, ws->d_pack_w, panel_rows),
                                     "W_i -= W_prev * tmp");
                    }

                    if (d_Y_local) {
                        AssertCuda(
                            cudaMemcpy2DAsync(
                                d_Y_local + static_cast<size_t>(local_panel_col) * lda_local + inner,
                                static_cast<size_t>(lda_local) * sizeof(T), ws->d_pack_y,
                                static_cast<size_t>(panel_rows) * sizeof(T),
                                static_cast<size_t>(panel_rows) * sizeof(T), kPanelWidth,
                                cudaMemcpyDeviceToDevice, compute_stream),
                            "cudaMemcpy2DAsync pack_y -> d_Y_local");
                        const int zero_rows = inner - block_begin;
                        if (zero_rows > 0) {
                            T* z_y = d_Y_local + static_cast<size_t>(local_panel_col) * lda_local +
                                     block_begin;
                            AssertCuda(cudaMemset2DAsync(z_y, static_cast<size_t>(lda_local) * sizeof(T),
                                                         0, static_cast<size_t>(zero_rows) * sizeof(T),
                                                         kPanelWidth, compute_stream),
                                       "cudaMemset2DAsync zero d_Y_local top");
                        }
                    }
                    if (d_W_local) {
                        AssertCuda(
                            cudaMemcpy2DAsync(
                                d_W_local + static_cast<size_t>(local_panel_col) * lda_local + inner,
                                static_cast<size_t>(lda_local) * sizeof(T), ws->d_pack_w,
                                static_cast<size_t>(panel_rows) * sizeof(T),
                                static_cast<size_t>(panel_rows) * sizeof(T), kPanelWidth,
                                cudaMemcpyDeviceToDevice, compute_stream),
                            "cudaMemcpy2DAsync pack_w -> d_W_local");
                        const int zero_rows = inner - block_begin;
                        if (zero_rows > 0) {
                            T* z_w = d_W_local + static_cast<size_t>(local_panel_col) * lda_local +
                                     block_begin;
                            AssertCuda(cudaMemset2DAsync(z_w, static_cast<size_t>(lda_local) * sizeof(T),
                                                         0, static_cast<size_t>(zero_rows) * sizeof(T),
                                                         kPanelWidth, compute_stream),
                                       "cudaMemset2DAsync zero d_W_local top");
                        }
                    }

                    const dim3 block_dim(16, 16);
                    const dim3 grid_dim((kPanelWidth + block_dim.x - 1) / block_dim.x,
                                        (kPanelWidth + block_dim.y - 1) / block_dim.y);
                    write_upper_r_to_panel_kernel<<<grid_dim, block_dim, 0, compute_stream>>>(
                        inner, ws->d_r_panel, kPanelWidth, panel_A, lda_local);
                    AssertCuda(cudaGetLastError(), "write_upper_r_to_panel_kernel launch");

                    owner_prepared = true;
                }

                if (owner_prepared) {
                    AssertCuda(cudaEventRecord(events.panel_ready, compute_stream),
                               "cudaEventRecord panel_ready");
                    AssertCuda(cudaStreamWaitEvent(comm_stream, events.panel_ready, 0),
                               "cudaStreamWaitEvent comm_stream <- panel_ready");
                }
            }

            const bool needs_comm = (part.world_size > 1);
            if (needs_comm) {
                AssertNccl(ncclGroupStart(), "ncclGroupStart panel W/Y");
                if (part.rank == owner) {
                    for (int r = 0; r < part.world_size; ++r) {
                        if (r == owner) {
                            continue;
                        }
                        AssertNccl(ncclSend(ws->d_pack_w, static_cast<size_t>(panel_rows) * kPanelWidth,
                                            nccl_type, r, nccl_comm, comm_stream),
                                   "ncclSend panel W");
                        AssertNccl(ncclSend(ws->d_pack_y, static_cast<size_t>(panel_rows) * kPanelWidth,
                                            nccl_type, r, nccl_comm, comm_stream),
                                   "ncclSend panel Y");
                        if (comm_profile) {
                            comm_profile->bytes +=
                                2ULL * static_cast<size_t>(panel_rows) * kPanelWidth * sizeof(T);
                        }
                    }
                } else {
                    AssertNccl(ncclRecv(ws->d_pack_w, static_cast<size_t>(panel_rows) * kPanelWidth,
                                        nccl_type, owner, nccl_comm, comm_stream),
                               "ncclRecv panel W");
                    AssertNccl(ncclRecv(ws->d_pack_y, static_cast<size_t>(panel_rows) * kPanelWidth,
                                        nccl_type, owner, nccl_comm, comm_stream),
                               "ncclRecv panel Y");
                    if (comm_profile) {
                        comm_profile->bytes +=
                            2ULL * static_cast<size_t>(panel_rows) * kPanelWidth * sizeof(T);
                    }
                }
                AssertNccl(ncclGroupEnd(), "ncclGroupEnd panel W/Y");
                AssertCuda(cudaEventRecord(events.w_ready, comm_stream), "cudaEventRecord w_ready");
                AssertCuda(cudaEventRecord(events.y_ready, comm_stream), "cudaEventRecord y_ready");
                AssertCuda(cudaStreamWaitEvent(compute_stream, events.w_ready, 0),
                           "cudaStreamWaitEvent compute_stream <- w_ready");
                AssertCuda(cudaStreamWaitEvent(compute_stream, events.y_ready, 0),
                           "cudaStreamWaitEvent compute_stream <- y_ready");
            } else {
                // Ensure the in-block update pipeline can safely wait on these events.
                AssertCuda(cudaEventRecord(events.w_ready, compute_stream),
                           "cudaEventRecord w_ready(np1)");
                AssertCuda(cudaEventRecord(events.y_ready, compute_stream),
                           "cudaEventRecord y_ready(np1)");
            }

            // Scatter packed W/Y into block buffers at (row=inner, col=block_col_off).
            // Also zero the [block_begin, inner) region of these panel columns.
            {
                T* dst_w = ws->d_block_w + static_cast<size_t>(inner) +
                           static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
                T* dst_y = ws->d_block_y + static_cast<size_t>(inner) +
                           static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
                AssertCuda(cudaMemcpy2DAsync(dst_w, static_cast<size_t>(m) * sizeof(T), ws->d_pack_w,
                                             static_cast<size_t>(panel_rows) * sizeof(T),
                                             static_cast<size_t>(panel_rows) * sizeof(T),
                                             kPanelWidth, cudaMemcpyDeviceToDevice, compute_stream),
                           "cudaMemcpy2DAsync pack_w -> block_w");
                AssertCuda(cudaMemcpy2DAsync(dst_y, static_cast<size_t>(m) * sizeof(T), ws->d_pack_y,
                                             static_cast<size_t>(panel_rows) * sizeof(T),
                                             static_cast<size_t>(panel_rows) * sizeof(T),
                                             kPanelWidth, cudaMemcpyDeviceToDevice, compute_stream),
                           "cudaMemcpy2DAsync pack_y -> block_y");

                const int zero_rows = inner - block_begin;
                if (zero_rows > 0) {
                    T* z_w = ws->d_block_w + static_cast<size_t>(block_begin) +
                             static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
                    T* z_y = ws->d_block_y + static_cast<size_t>(block_begin) +
                             static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
                    AssertCuda(cudaMemset2DAsync(z_w, static_cast<size_t>(m) * sizeof(T), 0,
                                                 static_cast<size_t>(zero_rows) * sizeof(T),
                                                 kPanelWidth, compute_stream),
                               "cudaMemset2DAsync zero block_w top");
                    AssertCuda(cudaMemset2DAsync(z_y, static_cast<size_t>(m) * sizeof(T), 0,
                                                 static_cast<size_t>(zero_rows) * sizeof(T),
                                                 kPanelWidth, compute_stream),
                               "cudaMemset2DAsync zero block_y top");
                }
            }

            // Apply this panel transform to local columns inside the current block only
            // (prepare later panels), like the single-GPU algorithm.
            const int inblock_begin_global = std::max(inner + kPanelWidth, part.col_start);
            const int inblock_end_global = std::min(block_end, part.col_end);
            if (inblock_begin_global < inblock_end_global) {
                const int local_begin = inblock_begin_global - part.col_start;
                const int cols_local = inblock_end_global - inblock_begin_global;
                T* a_trail = d_A_local + static_cast<size_t>(local_begin) * lda_local;

                panel_update_one_shot(cublas_handle, inner, panel_rows, cols_local, ws->d_pack_w,
                                      ws->d_pack_y, panel_rows, a_trail, lda_local, ws->d_tmp0,
                                      ws->tmp_elems);
            }
        }

        // Apply the full block WY to trailing columns after the block.
        const int trail_begin_global = std::max(block_end, part.col_start);
        if (trail_begin_global < part.col_end) {
            const int local_begin = trail_begin_global - part.col_start;
            const int cols_local = part.col_end - trail_begin_global;

            const int rows = m - block_begin;
            const T* w_big = ws->d_block_w + static_cast<size_t>(block_begin);
            const T* y_big = ws->d_block_y + static_cast<size_t>(block_begin);
            T* a_trail = d_A_local + static_cast<size_t>(local_begin) * lda_local;

            block_update_tile_pipeline(cublas_handle, compute_stream, block_begin, rows, kb,
                                       cols_local, tile_cols, w_big, y_big, m, a_trail, lda_local,
                                       ws->d_tmp0, ws->d_tmp1);
        }
    }
}

}  // namespace distributed_qr_col
