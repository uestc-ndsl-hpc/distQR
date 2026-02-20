#pragma once

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

namespace distributed_qr {

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

struct RowPartition {
    int m_global = 0;
    int world_size = 1;
    int rank = 0;
    int row_start = 0;
    int row_end = 0;
    int local_rows = 0;
};

inline RowPartition MakeRowPartition(int m_global, int world_size, int rank) {
    RowPartition part{};
    part.m_global = m_global;
    part.world_size = world_size;
    part.rank = rank;

    const int base = m_global / world_size;
    const int rem = m_global % world_size;
    part.local_rows = base + ((rank < rem) ? 1 : 0);
    part.row_start = rank * base + std::min(rank, rem);
    part.row_end = part.row_start + part.local_rows;
    return part;
}

inline void LocalSubRows(int global_row_begin,
                         int global_row_end,
                         const RowPartition& part,
                         int* local_row_offset,
                         int* local_rows) {
    const int begin = std::max(global_row_begin, part.row_start);
    const int end = std::min(global_row_end, part.row_end);
    if (end <= begin) {
        *local_row_offset = 0;
        *local_rows = 0;
        return;
    }
    *local_row_offset = begin - part.row_start;
    *local_rows = end - begin;
}

template <typename T>
__global__ void pack_rank_blocks_to_stack_kernel(int world_size,
                                                 int block_rows,
                                                 const T* gathered_blocks,
                                                 T* stacked_matrix) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_elems = block_rows * block_rows;
    const int total = world_size * block_elems;
    if (idx >= total) {
        return;
    }

    const int rank_id = idx / block_elems;
    const int inner = idx % block_elems;
    const int col = inner / block_rows;
    const int row = inner % block_rows;

    const int stack_row = rank_id * block_rows + row;
    const int ld_stack = world_size * block_rows;
    stacked_matrix[stack_row + col * ld_stack] = gathered_blocks[idx];
}

template <typename T>
__global__ void write_r_panel_to_local_a_kernel(int panel_col_begin,
                                                int panel_width,
                                                int row_start,
                                                int row_end,
                                                const T* R,
                                                int ldr,
                                                T* A_local,
                                                int lda_local) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= panel_width || col >= panel_width) {
        return;
    }

    const int global_row = panel_col_begin + row;
    if (global_row < row_start || global_row >= row_end) {
        return;
    }

    const int local_row = global_row - row_start;
    const int global_col = panel_col_begin + col;
    const T value = (row <= col) ? R[row + col * ldr] : static_cast<T>(0);
    A_local[local_row + global_col * lda_local] = value;
}

template <typename T>
__global__ void unpack_padded_allgather_panel_kernel(int world_size,
                                                     int max_rows_per_rank,
                                                     int panel_cols,
                                                     int panel_rows_global,
                                                     const T* gathered_padded_panel,
                                                     const int* rows_per_rank,
                                                     const int* row_offsets,
                                                     T* panel_global) {
    const int rank_id = blockIdx.z;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (rank_id >= world_size || col >= panel_cols) {
        return;
    }

    const int rows_this_rank = rows_per_rank[rank_id];
    if (row >= rows_this_rank) {
        return;
    }

    const size_t block_elems =
        static_cast<size_t>(max_rows_per_rank) * static_cast<size_t>(panel_cols);
    const size_t src_base = static_cast<size_t>(rank_id) * block_elems;
    const size_t src =
        src_base + static_cast<size_t>(row) + static_cast<size_t>(col) * max_rows_per_rank;

    const int dst_row = row_offsets[rank_id] + row;
    panel_global[static_cast<size_t>(dst_row) + static_cast<size_t>(col) * panel_rows_global] =
        gathered_padded_panel[src];
}

template <typename T>
struct DistributedQrWorkspace {
    T* d_r_local = nullptr;     // [kPanelWidth x kPanelWidth]
    T* d_r_gathered = nullptr;  // [world_size][kPanelWidth x kPanelWidth]
    T* d_r_stack = nullptr;     // [(world_size*kPanelWidth) x kPanelWidth]
    T* d_r_global = nullptr;    // [kPanelWidth x kPanelWidth]
    T* d_tsqr_work_local = nullptr;
    size_t tsqr_work_local_elems = 0;
    T* d_tsqr_work_stack = nullptr;
    size_t tsqr_work_stack_elems = 0;
    T* d_panel_tmp = nullptr;  // [local_panel_rows x kPanelWidth]
    T* d_work0 = nullptr;      // [nb x nb]
    T* d_work1 = nullptr;      // [nb x nb]
};

template <typename T>
void allreduce_panel_update_overlap(cublasHandle_t cublas_handle,
                                    ncclComm_t nccl_comm,
                                    cudaStream_t compute_stream,
                                    cudaStream_t comm_stream,
                                    int local_rows,
                                    int k,
                                    int cols,
                                    int tile_cols,
                                    const T* W,
                                    int ldw,
                                    const T* Y,
                                    int ldy,
                                    T* A,
                                    int lda,
                                    T* d_work0,
                                    T* d_work1) {
    if (k <= 0 || cols <= 0 || tile_cols <= 0) {
        return;
    }

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const T minus_one = static_cast<T>(-1);
    const ncclDataType_t nccl_type = NcclType<T>();

    T* work_buffers[2] = {d_work0, d_work1};
    T* a_tiles[2] = {nullptr, nullptr};
    int tile_widths[2] = {0, 0};

    struct PersistentEvents {
        bool initialized = false;
        cudaEvent_t gemm_done[2] = {};
        cudaEvent_t allreduce_done[2] = {};
    };
    static PersistentEvents events;
    if (!events.initialized) {
        AssertCuda(cudaEventCreateWithFlags(&events.gemm_done[0], cudaEventDisableTiming),
                   "cudaEventCreate gemm_done[0]");
        AssertCuda(cudaEventCreateWithFlags(&events.gemm_done[1], cudaEventDisableTiming),
                   "cudaEventCreate gemm_done[1]");
        AssertCuda(cudaEventCreateWithFlags(&events.allreduce_done[0], cudaEventDisableTiming),
                   "cudaEventCreate allreduce_done[0]");
        AssertCuda(cudaEventCreateWithFlags(&events.allreduce_done[1], cudaEventDisableTiming),
                   "cudaEventCreate allreduce_done[1]");
        events.initialized = true;
    }

    const int tile_count = (cols + tile_cols - 1) / tile_cols;
    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        const int idx = tile_idx & 1;
        const int col0 = tile_idx * tile_cols;
        const int width = std::min(tile_cols, cols - col0);
        T* a_tile = A + static_cast<size_t>(col0) * static_cast<size_t>(lda);

        a_tiles[idx] = a_tile;
        tile_widths[idx] = width;

        if (local_rows > 0) {
            AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k,
                                                   width, local_rows, &one, W, ldw, a_tile, lda,
                                                   &zero, work_buffers[idx], k),
                         "work = W^T * A_tile(local)");
        } else {
            AssertCuda(cudaMemsetAsync(work_buffers[idx], 0,
                                       static_cast<size_t>(k) * width * sizeof(T), compute_stream),
                       "cudaMemsetAsync work_buffers[idx]");
        }

        AssertCuda(cudaEventRecord(events.gemm_done[idx], compute_stream),
                   "cudaEventRecord gemm_done[idx]");
        AssertCuda(cudaStreamWaitEvent(comm_stream, events.gemm_done[idx], 0),
                   "cudaStreamWaitEvent comm_stream <- gemm_done[idx]");
        AssertNccl(ncclAllReduce(work_buffers[idx], work_buffers[idx],
                                 static_cast<size_t>(k) * static_cast<size_t>(width), nccl_type,
                                 ncclSum, nccl_comm, comm_stream),
                   "ncclAllReduce work_buffers[idx]");
        AssertCuda(cudaEventRecord(events.allreduce_done[idx], comm_stream),
                   "cudaEventRecord allreduce_done[idx]");

        if (tile_idx > 0) {
            const int prev = idx ^ 1;
            AssertCuda(cudaStreamWaitEvent(compute_stream, events.allreduce_done[prev], 0),
                       "cudaStreamWaitEvent compute_stream <- allreduce_done[prev]");
            if (local_rows > 0) {
                AssertCublas(
                    CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, local_rows,
                                              tile_widths[prev], k, &minus_one, Y, ldy,
                                              work_buffers[prev], k, &one, a_tiles[prev], lda),
                    "A_tile -= Y * work(global)");
            }
        }
    }

    if (tile_count > 0) {
        const int last = (tile_count - 1) & 1;
        AssertCuda(cudaStreamWaitEvent(compute_stream, events.allreduce_done[last], 0),
                   "cudaStreamWaitEvent compute_stream <- allreduce_done[last]");
        if (local_rows > 0) {
            AssertCublas(
                CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, local_rows,
                                          tile_widths[last], k, &minus_one, Y, ldy,
                                          work_buffers[last], k, &one, a_tiles[last], lda),
                "A_last -= Y * work(global)");
        }
    }
}

template <typename T>
void distributed_blocked_qr_factorize(cublasHandle_t cublas_handle,
                                      ncclComm_t nccl_comm,
                                      const RowPartition& part,
                                      int m,
                                      int n,
                                      int nb,
                                      T* d_A_local,
                                      int lda_local,
                                      T* d_W_local,
                                      T* d_Y_local,
                                      DistributedQrWorkspace<T>* ws,
                                      cudaStream_t compute_stream,
                                      cudaStream_t comm_stream,
                                      int overlap_tile_cols = 0) {
    if (!ws) {
        spdlog::error("distributed_blocked_qr_factorize got null workspace.");
        std::exit(1);
    }
    if (n % kPanelWidth != 0 || nb % kPanelWidth != 0) {
        spdlog::error("Require n and nb to be multiples of {} (got n={} nb={}).", kPanelWidth, n,
                      nb);
        std::exit(1);
    }
    if (m != part.m_global) {
        spdlog::error("RowPartition.m_global ({}) mismatches m ({}).", part.m_global, m);
        std::exit(1);
    }

    const int stack_rows = part.world_size * kPanelWidth;
    const int tile_target = (overlap_tile_cols <= 0) ? nb : overlap_tile_cols;
    const int tile_cols = std::min(std::max(kPanelWidth, tile_target), nb);
    const ncclDataType_t nccl_type = NcclType<T>();

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const T minus_one = static_cast<T>(-1);

    AssertCublas(cublasSetStream(cublas_handle, compute_stream), "cublasSetStream(compute_stream)");

    std::vector<RowPartition> rank_parts(static_cast<size_t>(part.world_size));
    int max_rows_per_rank = 0;
    for (int r = 0; r < part.world_size; ++r) {
        rank_parts[static_cast<size_t>(r)] = MakeRowPartition(m, part.world_size, r);
        max_rows_per_rank =
            std::max(max_rows_per_rank, rank_parts[static_cast<size_t>(r)].local_rows);
    }

    struct PersistentScratch {
        bool events_initialized = false;
        int world_size = -1;
        int m_global = -1;
        int max_rows_per_rank = -1;

        T* d_panel_padded = nullptr;
        T* d_panel_allgather = nullptr;
        T* d_panel_global = nullptr;
        T* d_panel_w_global = nullptr;
        T* d_panel_y_global = nullptr;
        int* d_rows_per_rank = nullptr;
        int* d_row_offsets = nullptr;

        cudaEvent_t tsqr_done = {};
        cudaEvent_t allgather_done = {};
        cudaEvent_t stage2_done = {};
        cudaEvent_t bcast_done = {};
        cudaEvent_t tmp_local_done = {};
        cudaEvent_t tmp_allreduce_done = {};
        cudaEvent_t panel_pack_done = {};
        cudaEvent_t panel_allgather_done = {};
    };
    static PersistentScratch scratch;

    if (!scratch.events_initialized) {
        AssertCuda(cudaEventCreateWithFlags(&scratch.tsqr_done, cudaEventDisableTiming),
                   "cudaEventCreate tsqr_done");
        AssertCuda(cudaEventCreateWithFlags(&scratch.allgather_done, cudaEventDisableTiming),
                   "cudaEventCreate allgather_done");
        AssertCuda(cudaEventCreateWithFlags(&scratch.stage2_done, cudaEventDisableTiming),
                   "cudaEventCreate stage2_done");
        AssertCuda(cudaEventCreateWithFlags(&scratch.bcast_done, cudaEventDisableTiming),
                   "cudaEventCreate bcast_done");
        AssertCuda(cudaEventCreateWithFlags(&scratch.tmp_local_done, cudaEventDisableTiming),
                   "cudaEventCreate tmp_local_done");
        AssertCuda(cudaEventCreateWithFlags(&scratch.tmp_allreduce_done, cudaEventDisableTiming),
                   "cudaEventCreate tmp_allreduce_done");
        AssertCuda(cudaEventCreateWithFlags(&scratch.panel_pack_done, cudaEventDisableTiming),
                   "cudaEventCreate panel_pack_done");
        AssertCuda(cudaEventCreateWithFlags(&scratch.panel_allgather_done, cudaEventDisableTiming),
                   "cudaEventCreate panel_allgather_done");
        scratch.events_initialized = true;
    }

    if (scratch.world_size != part.world_size || scratch.m_global != m ||
        scratch.max_rows_per_rank < max_rows_per_rank) {
        if (scratch.d_panel_padded) {
            AssertCuda(cudaFree(scratch.d_panel_padded), "cudaFree scratch.d_panel_padded");
            AssertCuda(cudaFree(scratch.d_panel_allgather), "cudaFree scratch.d_panel_allgather");
            AssertCuda(cudaFree(scratch.d_panel_global), "cudaFree scratch.d_panel_global");
            AssertCuda(cudaFree(scratch.d_panel_w_global), "cudaFree scratch.d_panel_w_global");
            AssertCuda(cudaFree(scratch.d_panel_y_global), "cudaFree scratch.d_panel_y_global");
            AssertCuda(cudaFree(scratch.d_rows_per_rank), "cudaFree scratch.d_rows_per_rank");
            AssertCuda(cudaFree(scratch.d_row_offsets), "cudaFree scratch.d_row_offsets");
        }

        const size_t padded_panel_elems =
            static_cast<size_t>(std::max(max_rows_per_rank, 1)) * kPanelWidth;
        const size_t gathered_panel_elems =
            static_cast<size_t>(part.world_size) * padded_panel_elems;
        const size_t global_panel_elems = static_cast<size_t>(m) * kPanelWidth;

        AssertCuda(cudaMalloc(&scratch.d_panel_padded, padded_panel_elems * sizeof(T)),
                   "cudaMalloc scratch.d_panel_padded");
        AssertCuda(cudaMalloc(&scratch.d_panel_allgather, gathered_panel_elems * sizeof(T)),
                   "cudaMalloc scratch.d_panel_allgather");
        AssertCuda(cudaMalloc(&scratch.d_panel_global, global_panel_elems * sizeof(T)),
                   "cudaMalloc scratch.d_panel_global");
        AssertCuda(cudaMalloc(&scratch.d_panel_w_global, global_panel_elems * sizeof(T)),
                   "cudaMalloc scratch.d_panel_w_global");
        AssertCuda(cudaMalloc(&scratch.d_panel_y_global, global_panel_elems * sizeof(T)),
                   "cudaMalloc scratch.d_panel_y_global");
        AssertCuda(cudaMalloc(&scratch.d_rows_per_rank,
                              static_cast<size_t>(part.world_size) * sizeof(int)),
                   "cudaMalloc scratch.d_rows_per_rank");
        AssertCuda(
            cudaMalloc(&scratch.d_row_offsets, static_cast<size_t>(part.world_size) * sizeof(int)),
            "cudaMalloc scratch.d_row_offsets");

        scratch.world_size = part.world_size;
        scratch.m_global = m;
        scratch.max_rows_per_rank = max_rows_per_rank;
    }

    T* d_panel_padded = scratch.d_panel_padded;
    T* d_panel_allgather = scratch.d_panel_allgather;
    T* d_panel_global = scratch.d_panel_global;
    T* d_panel_w_global = scratch.d_panel_w_global;
    T* d_panel_y_global = scratch.d_panel_y_global;
    int* d_rows_per_rank = scratch.d_rows_per_rank;
    int* d_row_offsets = scratch.d_row_offsets;
    const int padded_rows = std::max(scratch.max_rows_per_rank, 1);
    const size_t padded_panel_elems = static_cast<size_t>(padded_rows) * kPanelWidth;

    cudaEvent_t tsqr_done = scratch.tsqr_done;
    cudaEvent_t allgather_done = scratch.allgather_done;
    cudaEvent_t stage2_done = scratch.stage2_done;
    cudaEvent_t bcast_done = scratch.bcast_done;
    cudaEvent_t tmp_local_done = scratch.tmp_local_done;
    cudaEvent_t tmp_allreduce_done = scratch.tmp_allreduce_done;
    cudaEvent_t panel_pack_done = scratch.panel_pack_done;
    cudaEvent_t panel_allgather_done = scratch.panel_allgather_done;

    for (int outer = 0; outer < n; outer += nb) {
        const int end = std::min(outer + nb, n);
        const int kb = end - outer;

        int outer_row_off = 0;
        int m_sub_local = 0;
        LocalSubRows(outer, m, part, &outer_row_off, &m_sub_local);

        T* w_big = d_W_local + static_cast<size_t>(outer_row_off) +
                   static_cast<size_t>(outer) * static_cast<size_t>(lda_local);
        T* y_big = d_Y_local + static_cast<size_t>(outer_row_off) +
                   static_cast<size_t>(outer) * static_cast<size_t>(lda_local);

        for (int inner = outer; inner < end; inner += kPanelWidth) {
            int panel_row_off = 0;
            int panel_rows_local = 0;
            LocalSubRows(inner, m, part, &panel_row_off, &panel_rows_local);

            T* panel_A = d_A_local + static_cast<size_t>(panel_row_off) +
                         static_cast<size_t>(inner) * static_cast<size_t>(lda_local);
            T* panel_W = d_W_local + static_cast<size_t>(panel_row_off) +
                         static_cast<size_t>(inner) * static_cast<size_t>(lda_local);
            T* panel_Y = d_Y_local + static_cast<size_t>(panel_row_off) +
                         static_cast<size_t>(inner) * static_cast<size_t>(lda_local);

            if (panel_rows_local > 0) {
                tsqr<T>(cublas_handle, panel_rows_local, panel_A, lda_local, ws->d_r_local,
                        kPanelWidth, ws->d_tsqr_work_local, ws->tsqr_work_local_elems,
                        compute_stream);
            } else {
                AssertCuda(
                    cudaMemsetAsync(ws->d_r_local, 0,
                                    static_cast<size_t>(kPanelWidth) * kPanelWidth * sizeof(T),
                                    compute_stream),
                    "cudaMemsetAsync d_r_local");
            }

            AssertCuda(cudaEventRecord(tsqr_done, compute_stream), "cudaEventRecord tsqr_done");
            AssertCuda(cudaStreamWaitEvent(comm_stream, tsqr_done, 0),
                       "cudaStreamWaitEvent comm_stream <- tsqr_done");

            AssertNccl(ncclAllGather(ws->d_r_local, ws->d_r_gathered,
                                     static_cast<size_t>(kPanelWidth) * kPanelWidth, nccl_type,
                                     nccl_comm, comm_stream),
                       "ncclAllGather R_local");
            AssertCuda(cudaEventRecord(allgather_done, comm_stream),
                       "cudaEventRecord allgather_done");

            AssertCuda(cudaStreamWaitEvent(compute_stream, allgather_done, 0),
                       "cudaStreamWaitEvent compute_stream <- allgather_done");

            const int gather_elems = part.world_size * kPanelWidth * kPanelWidth;
            const int threads = 256;
            const int blocks = (gather_elems + threads - 1) / threads;
            pack_rank_blocks_to_stack_kernel<<<blocks, threads, 0, compute_stream>>>(
                part.world_size, kPanelWidth, ws->d_r_gathered, ws->d_r_stack);
            AssertCuda(cudaGetLastError(), "pack_rank_blocks_to_stack_kernel launch");

            tsqr<T>(cublas_handle, stack_rows, ws->d_r_stack, stack_rows, ws->d_r_global,
                    kPanelWidth, ws->d_tsqr_work_stack, ws->tsqr_work_stack_elems, compute_stream);

            AssertCuda(cudaEventRecord(stage2_done, compute_stream), "cudaEventRecord stage2_done");
            AssertCuda(cudaStreamWaitEvent(comm_stream, stage2_done, 0),
                       "cudaStreamWaitEvent comm_stream <- stage2_done");

            AssertNccl(ncclBroadcast(ws->d_r_global, ws->d_r_global,
                                     static_cast<size_t>(kPanelWidth) * kPanelWidth, nccl_type, 0,
                                     nccl_comm, comm_stream),
                       "ncclBroadcast R_global");
            AssertCuda(cudaEventRecord(bcast_done, comm_stream), "cudaEventRecord bcast_done");
            AssertCuda(cudaStreamWaitEvent(compute_stream, bcast_done, 0),
                       "cudaStreamWaitEvent compute_stream <- bcast_done");

            if (panel_rows_local > 0) {
                T* q2_block = ws->d_r_stack + static_cast<size_t>(part.rank) * kPanelWidth;
                AssertCublas(CublasGemmTraits<T>::Gemm(
                                 cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_rows_local,
                                 kPanelWidth, kPanelWidth, &one, panel_A, lda_local, q2_block,
                                 stack_rows, &zero, ws->d_panel_tmp, panel_rows_local),
                             "panel_A = panel_A * q2_block");
                AssertCuda(cudaMemcpy2DAsync(panel_A, static_cast<size_t>(lda_local) * sizeof(T),
                                             ws->d_panel_tmp,
                                             static_cast<size_t>(panel_rows_local) * sizeof(T),
                                             static_cast<size_t>(panel_rows_local) * sizeof(T),
                                             kPanelWidth, cudaMemcpyDeviceToDevice, compute_stream),
                           "cudaMemcpy2DAsync panel tmp -> panel_A");
            }

            std::vector<int> h_rows_per_rank(static_cast<size_t>(part.world_size), 0);
            std::vector<int> h_row_offsets(static_cast<size_t>(part.world_size), 0);
            int panel_rows_global = 0;
            for (int r = 0; r < part.world_size; ++r) {
                int unused_row_offset = 0;
                int rows_this_rank = 0;
                LocalSubRows(inner, m, rank_parts[static_cast<size_t>(r)], &unused_row_offset,
                             &rows_this_rank);
                h_rows_per_rank[static_cast<size_t>(r)] = rows_this_rank;
                h_row_offsets[static_cast<size_t>(r)] = panel_rows_global;
                panel_rows_global += rows_this_rank;
            }

            AssertCuda(cudaMemcpyAsync(d_rows_per_rank, h_rows_per_rank.data(),
                                       static_cast<size_t>(part.world_size) * sizeof(int),
                                       cudaMemcpyHostToDevice, compute_stream),
                       "cudaMemcpyAsync h_rows_per_rank -> d_rows_per_rank");
            AssertCuda(cudaMemcpyAsync(d_row_offsets, h_row_offsets.data(),
                                       static_cast<size_t>(part.world_size) * sizeof(int),
                                       cudaMemcpyHostToDevice, compute_stream),
                       "cudaMemcpyAsync h_row_offsets -> d_row_offsets");

            AssertCuda(
                cudaMemsetAsync(d_panel_padded, 0, padded_panel_elems * sizeof(T), compute_stream),
                "cudaMemsetAsync d_panel_padded");
            if (panel_rows_local > 0) {
                AssertCuda(
                    cudaMemcpy2DAsync(d_panel_padded, static_cast<size_t>(padded_rows) * sizeof(T),
                                      panel_A, static_cast<size_t>(lda_local) * sizeof(T),
                                      static_cast<size_t>(panel_rows_local) * sizeof(T),
                                      kPanelWidth, cudaMemcpyDeviceToDevice, compute_stream),
                    "cudaMemcpy2DAsync panel_A -> d_panel_padded");
            }

            AssertCuda(cudaEventRecord(panel_pack_done, compute_stream),
                       "cudaEventRecord panel_pack_done");
            AssertCuda(cudaStreamWaitEvent(comm_stream, panel_pack_done, 0),
                       "cudaStreamWaitEvent comm_stream <- panel_pack_done");

            AssertNccl(ncclAllGather(d_panel_padded, d_panel_allgather, padded_panel_elems,
                                     nccl_type, nccl_comm, comm_stream),
                       "ncclAllGather panel padded");
            AssertCuda(cudaEventRecord(panel_allgather_done, comm_stream),
                       "cudaEventRecord panel_allgather_done");
            AssertCuda(cudaStreamWaitEvent(compute_stream, panel_allgather_done, 0),
                       "cudaStreamWaitEvent compute_stream <- panel_allgather_done");

            {
                const dim3 block_dim(16, 8, 1);
                const dim3 grid_dim((padded_rows + block_dim.x - 1) / block_dim.x,
                                    (kPanelWidth + block_dim.y - 1) / block_dim.y, part.world_size);
                unpack_padded_allgather_panel_kernel<<<grid_dim, block_dim, 0, compute_stream>>>(
                    part.world_size, padded_rows, kPanelWidth, panel_rows_global, d_panel_allgather,
                    d_rows_per_rank, d_row_offsets, d_panel_global);
                AssertCuda(cudaGetLastError(), "unpack_padded_allgather_panel_kernel launch");
            }

            generate_wy(panel_rows_global, kPanelWidth, d_panel_global, panel_rows_global,
                        d_panel_y_global, panel_rows_global, d_panel_w_global, panel_rows_global,
                        compute_stream);

            if (panel_rows_local > 0) {
                const int panel_rank_row_offset = h_row_offsets[static_cast<size_t>(part.rank)];
                AssertCuda(cudaMemcpy2DAsync(panel_Y, static_cast<size_t>(lda_local) * sizeof(T),
                                             d_panel_y_global + panel_rank_row_offset,
                                             static_cast<size_t>(panel_rows_global) * sizeof(T),
                                             static_cast<size_t>(panel_rows_local) * sizeof(T),
                                             kPanelWidth, cudaMemcpyDeviceToDevice, compute_stream),
                           "cudaMemcpy2DAsync d_panel_y_global -> panel_Y");
                AssertCuda(cudaMemcpy2DAsync(panel_W, static_cast<size_t>(lda_local) * sizeof(T),
                                             d_panel_w_global + panel_rank_row_offset,
                                             static_cast<size_t>(panel_rows_global) * sizeof(T),
                                             static_cast<size_t>(panel_rows_local) * sizeof(T),
                                             kPanelWidth, cudaMemcpyDeviceToDevice, compute_stream),
                           "cudaMemcpy2DAsync d_panel_w_global -> panel_W");
            }

            {
                dim3 block_dim(16, 16);
                dim3 grid_dim((kPanelWidth + block_dim.x - 1) / block_dim.x,
                              (kPanelWidth + block_dim.y - 1) / block_dim.y);
                write_r_panel_to_local_a_kernel<<<grid_dim, block_dim, 0, compute_stream>>>(
                    inner, kPanelWidth, part.row_start, part.row_end, ws->d_r_global, kPanelWidth,
                    d_A_local, lda_local);
                AssertCuda(cudaGetLastError(), "write_r_panel_to_local_a_kernel launch");
            }

            const int n_remain_in_block = end - (inner + kPanelWidth);
            if (n_remain_in_block > 0) {
                T* a_remain = panel_A + static_cast<size_t>(kPanelWidth) * lda_local;
                allreduce_panel_update_overlap(
                    cublas_handle, nccl_comm, compute_stream, comm_stream, panel_rows_local,
                    kPanelWidth, n_remain_in_block, tile_cols, panel_W, lda_local, panel_Y,
                    lda_local, a_remain, lda_local, ws->d_work0, ws->d_work1);
            }

            if (inner > outer) {
                const int k_prev = inner - outer;
                T* w_prev = w_big;
                T* y_prev = y_big;
                T* w_i_sub = d_W_local + static_cast<size_t>(outer_row_off) +
                             static_cast<size_t>(inner) * static_cast<size_t>(lda_local);

                if (m_sub_local > 0) {
                    AssertCublas(
                        CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k_prev,
                                                  kPanelWidth, m_sub_local, &one, y_prev, lda_local,
                                                  w_i_sub, lda_local, &zero, ws->d_work0, k_prev),
                        "tmp = Y_prev^T * W_i(local)");
                } else {
                    AssertCuda(
                        cudaMemsetAsync(ws->d_work0, 0,
                                        static_cast<size_t>(k_prev) * kPanelWidth * sizeof(T),
                                        compute_stream),
                        "cudaMemsetAsync tmp buffer");
                }

                AssertCuda(cudaEventRecord(tmp_local_done, compute_stream),
                           "cudaEventRecord tmp_local_done");
                AssertCuda(cudaStreamWaitEvent(comm_stream, tmp_local_done, 0),
                           "cudaStreamWaitEvent comm_stream <- tmp_local_done");
                AssertNccl(ncclAllReduce(ws->d_work0, ws->d_work0,
                                         static_cast<size_t>(k_prev) * kPanelWidth, nccl_type,
                                         ncclSum, nccl_comm, comm_stream),
                           "ncclAllReduce tmp = Y_prev^T * W_i");
                AssertCuda(cudaEventRecord(tmp_allreduce_done, comm_stream),
                           "cudaEventRecord tmp_allreduce_done");
                AssertCuda(cudaStreamWaitEvent(compute_stream, tmp_allreduce_done, 0),
                           "cudaStreamWaitEvent compute_stream <- tmp_allreduce_done");

                if (m_sub_local > 0) {
                    AssertCublas(CublasGemmTraits<T>::Gemm(
                                     cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_sub_local,
                                     kPanelWidth, k_prev, &minus_one, w_prev, lda_local,
                                     ws->d_work0, k_prev, &one, w_i_sub, lda_local),
                                 "W_i -= W_prev * tmp(global)");
                }
            }
        }

        const int n_trail = n - end;
        if (n_trail > 0) {
            T* a_trail = d_A_local + static_cast<size_t>(outer_row_off) +
                         static_cast<size_t>(end) * static_cast<size_t>(lda_local);
            allreduce_panel_update_overlap(cublas_handle, nccl_comm, compute_stream, comm_stream,
                                           m_sub_local, kb, n_trail, tile_cols, w_big, lda_local,
                                           y_big, lda_local, a_trail, lda_local, ws->d_work0,
                                           ws->d_work1);
        }
    }
}

}  // namespace distributed_qr
