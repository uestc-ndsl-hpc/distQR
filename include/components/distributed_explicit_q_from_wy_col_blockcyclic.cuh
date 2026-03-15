#pragma once
// Distributed explicit-Q generation from block-WY factors for 1D col-blockcyclic layout.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "distributed_blocked_qr_col_blockcyclic.cuh"
#include "utils/nvtx_range.cuh"

namespace distributed_qr_col_blockcyclic {

template <typename T>
__global__ void set_local_identity_diag_kernel(int m,
                                               int global_col_begin,
                                               int cols,
                                               T* A_local,
                                               int lda_local) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }

    const int global_col = global_col_begin + col;
    if (global_col < m) {
        A_local[global_col + static_cast<size_t>(col) * lda_local] = static_cast<T>(1);
    }
}

template <typename T>
void set_explicit_q_identity_col_blockcyclic(const ColBlockCyclicPartition& part,
                                             int m,
                                             T* d_Q_local,
                                             int lda_local,
                                             cudaStream_t compute_stream) {
    if (part.local_cols <= 0) {
        return;
    }
    if (!d_Q_local) {
        spdlog::error("set_explicit_q_identity_col_blockcyclic got null d_Q_local.");
        std::exit(1);
    }

    const size_t local_elems =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(part.local_cols, 1));
    AssertCuda(cudaMemsetAsync(d_Q_local, 0, local_elems * sizeof(T), compute_stream),
               "cudaMemsetAsync explicit Q local zero");

    constexpr int kThreads = 256;
    for (size_t i = 0; i < part.block_starts.size(); ++i) {
        const int global_begin = part.block_starts[i];
        const int cols = part.block_ends[i] - global_begin;
        const int local_begin = part.block_local_offsets[i];
        T* dst = d_Q_local + static_cast<size_t>(local_begin) * lda_local;
        const int blocks = (cols + kThreads - 1) / kThreads;
        set_local_identity_diag_kernel<T><<<blocks, kThreads, 0, compute_stream>>>(
            m, global_begin, cols, dst, lda_local);
        AssertCuda(cudaGetLastError(), "set_local_identity_diag_kernel launch");
    }
}

template <typename T>
void apply_block_q_one_shot(cublasHandle_t cublas_handle,
                            int row_offset,
                            int rows,
                            int k,
                            int cols_local,
                            const T* W,
                            const T* Y,
                            int lda_wy,
                            T* d_Q_local,
                            int lda_local,
                            T* d_work,
                            size_t work_elems) {
    if (k <= 0 || cols_local <= 0 || rows <= 0) {
        return;
    }

    const size_t need = static_cast<size_t>(k) * static_cast<size_t>(cols_local);
    if (work_elems < need) {
        spdlog::error("apply_block_q_one_shot work too small (need {} elems, got {}).", need,
                      work_elems);
        std::exit(1);
    }

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const T minus_one = static_cast<T>(-1);

    T* q_sub = d_Q_local + row_offset;
    AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, cols_local,
                                           rows, &one, Y, lda_wy, q_sub, lda_local, &zero,
                                           d_work, k),
                 "explicit Q work = Y^T * Q");
    AssertCublas(
        CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols_local, k,
                                  &minus_one, W, lda_wy, d_work, k, &one, q_sub, lda_local),
        "explicit Q Q -= W * work");
}

template <typename T>
void apply_block_q_tile_pipeline(cublasHandle_t cublas_handle,
                                 cudaStream_t compute_stream,
                                 int row_offset,
                                 int rows,
                                 int k,
                                 int cols_local,
                                 int tile_cols,
                                 const T* W,
                                 const T* Y,
                                 int lda_wy,
                                 T* d_Q_local,
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
                   "cudaEventCreate explicitQ gemm1_done[0]");
        AssertCuda(cudaEventCreateWithFlags(&events.gemm1_done[1], cudaEventDisableTiming),
                   "cudaEventCreate explicitQ gemm1_done[1]");
        AssertCuda(cudaEventCreateWithFlags(&events.apply_done[0], cudaEventDisableTiming),
                   "cudaEventCreate explicitQ apply_done[0]");
        AssertCuda(cudaEventCreateWithFlags(&events.apply_done[1], cudaEventDisableTiming),
                   "cudaEventCreate explicitQ apply_done[1]");
        events.initialized = true;
    }

    T* tmps[2] = {d_tmp0, d_tmp1};
    T* q_tiles[2] = {nullptr, nullptr};
    int widths[2] = {0, 0};

    AssertCuda(cudaEventRecord(events.apply_done[0], compute_stream),
               "cudaEventRecord explicitQ apply_done[0]");
    AssertCuda(cudaEventRecord(events.apply_done[1], compute_stream),
               "cudaEventRecord explicitQ apply_done[1]");

    const int tile_count = (cols_local + tile_cols - 1) / tile_cols;
    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        const int idx = tile_idx & 1;
        const int col0 = tile_idx * tile_cols;
        const int width = std::min(tile_cols, cols_local - col0);
        T* q_tile = d_Q_local + static_cast<size_t>(col0) * lda_local;
        T* q_tile_sub = q_tile + row_offset;

        AssertCuda(cudaStreamWaitEvent(compute_stream, events.apply_done[idx], 0),
                   "cudaStreamWaitEvent explicitQ <- apply_done[idx]");
        AssertCublas(
            CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, width, rows,
                                      &one, Y, lda_wy, q_tile_sub, lda_local, &zero, tmps[idx],
                                      k),
            "explicit Q tmp = Y^T * Q_tile");
        AssertCuda(cudaEventRecord(events.gemm1_done[idx], compute_stream),
                   "cudaEventRecord explicitQ gemm1_done[idx]");

        q_tiles[idx] = q_tile;
        widths[idx] = width;

        if (tile_idx > 0) {
            const int prev = idx ^ 1;
            T* q_prev_sub = q_tiles[prev] + row_offset;
            AssertCuda(cudaStreamWaitEvent(compute_stream, events.gemm1_done[prev], 0),
                       "cudaStreamWaitEvent explicitQ <- gemm1_done[prev]");
            AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, rows,
                                                   widths[prev], k, &minus_one, W, lda_wy,
                                                   tmps[prev], k, &one, q_prev_sub, lda_local),
                         "explicit Q Q_tile -= W * tmp");
            AssertCuda(cudaEventRecord(events.apply_done[prev], compute_stream),
                       "cudaEventRecord explicitQ apply_done[prev]");
        }
    }

    if (tile_count > 0) {
        const int last = (tile_count - 1) & 1;
        T* q_last_sub = q_tiles[last] + row_offset;
        AssertCuda(cudaStreamWaitEvent(compute_stream, events.gemm1_done[last], 0),
                   "cudaStreamWaitEvent explicitQ <- gemm1_done[last]");
        AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, rows,
                                               widths[last], k, &minus_one, W, lda_wy,
                                               tmps[last], k, &one, q_last_sub, lda_local),
                     "explicit Q Q_last -= W * tmp");
        AssertCuda(cudaEventRecord(events.apply_done[last], compute_stream),
                   "cudaEventRecord explicitQ apply_done[last]");
    }
}

template <typename T>
void generate_explicit_q_from_wy_col_blockcyclic(
    cublasHandle_t cublas_handle,
    ncclComm_t nccl_comm,
    const ColBlockCyclicPartition& part,
    int m,
    int n,
    int nb,
    const T* d_W_local,
    const T* d_Y_local,
    T* d_Q_local,
    int lda_local,
    DistributedQrColBlockCyclicWorkspace<T>* ws,
    cudaStream_t compute_stream,
    cudaStream_t comm_stream,
    int tile_cols = 0,
    PanelCommMode panel_comm_mode = PanelCommMode::SendRecv) {
    // This mirrors single-GPU GenerateExplicitQFromWY():
    // walk outer blocks right-to-left and apply Q_k = I - W_k Y_k^T to the local trailing
    // columns of the distributed identity.
    static_assert(kSupportedQrType<T>,
                  "generate_explicit_q_from_wy_col_blockcyclic only supports float and double.");

    if (!ws) {
        spdlog::error("generate_explicit_q_from_wy_col_blockcyclic got null workspace.");
        std::exit(1);
    }
    if (m < n || nb <= 0 || n != part.n_global) {
        spdlog::error(
            "generate_explicit_q_from_wy_col_blockcyclic invalid args (m={} n={} nb={} "
            "part.n_global={}).",
            m, n, nb, part.n_global);
        std::exit(1);
    }
    if (n % kPanelWidth != 0 || nb % kPanelWidth != 0) {
        spdlog::error(
            "generate_explicit_q_from_wy_col_blockcyclic requires n and nb multiples of {} "
            "(got n={} nb={}).",
            kPanelWidth, n, nb);
        std::exit(1);
    }
    if (part.block_cols <= 0 || part.block_cols % nb != 0 || part.block_cols % kPanelWidth != 0) {
        spdlog::error(
            "generate_explicit_q_from_wy_col_blockcyclic requires block_cols to be positive and "
            "multiples of nb and {} (got {}).",
            kPanelWidth, part.block_cols);
        std::exit(1);
    }
    if (!d_W_local || !d_Y_local) {
        spdlog::error("generate_explicit_q_from_wy_col_blockcyclic requires non-null W/Y.");
        std::exit(1);
    }
    if (part.local_cols > 0 && !d_Q_local) {
        spdlog::error(
            "generate_explicit_q_from_wy_col_blockcyclic requires non-null d_Q_local when "
            "local_cols > 0.");
        std::exit(1);
    }
    if (panel_comm_mode != PanelCommMode::SendRecv && panel_comm_mode != PanelCommMode::Broadcast) {
        spdlog::error("Unsupported panel_comm_mode value {}.", static_cast<int>(panel_comm_mode));
        std::exit(1);
    }

    const size_t block_storage_need = static_cast<size_t>(m) * static_cast<size_t>(nb);
    if (!ws->d_block_w || !ws->d_block_y || ws->block_storage_elems < block_storage_need) {
        spdlog::error("Explicit-Q block storage too small (need {} elems, got {}).",
                      block_storage_need, ws->block_storage_elems);
        std::exit(1);
    }

    const bool use_one_shot = (tile_cols <= 0);
    const int work_tile_cols = use_one_shot ? std::max(part.local_cols, 1)
                                            : std::max(kPanelWidth, std::min(tile_cols, n));
    const size_t tmp_need = static_cast<size_t>(nb) * static_cast<size_t>(work_tile_cols);
    if (!ws->d_tmp0 || ws->tmp_elems < tmp_need || (!use_one_shot && !ws->d_tmp1)) {
        spdlog::error(
            "Explicit-Q tmp buffers too small (need {} elems, got {}; d_tmp0={} d_tmp1={}).",
            tmp_need, ws->tmp_elems, static_cast<const void*>(ws->d_tmp0),
            static_cast<const void*>(ws->d_tmp1));
        std::exit(1);
    }

    if (!ws->d_block_w_compact || !ws->d_block_y_compact ||
        ws->block_compact_elems < block_storage_need) {
        spdlog::error(
            "Explicit-Q ping-pong requires a second full block buffer pair in "
            "d_block_w_compact/d_block_y_compact (need {} elems, got {}).",
            block_storage_need, ws->block_compact_elems);
        std::exit(1);
    }

    AssertCublas(cublasSetStream(cublas_handle, compute_stream), "cublasSetStream(compute_stream)");
    const ncclDataType_t nccl_type = NcclType<T>();

    set_explicit_q_identity_col_blockcyclic(part, m, d_Q_local, lda_local, compute_stream);

    T* block_w_slots[2] = {ws->d_block_w, ws->d_block_w_compact};
    T* block_y_slots[2] = {ws->d_block_y, ws->d_block_y_compact};

    struct BlockTask {
        bool valid = false;
        int block_begin = 0;
        int block_end = 0;
        int block_rows = 0;
        int kb = 0;
        int owner = 0;
        size_t msg_elems = 0;
        bool rank_needs_block = false;
    };

    struct PersistentEvents {
        bool initialized = false;
        cudaEvent_t comm_done[2] = {};
        cudaEvent_t compute_done[2] = {};
    };
    static PersistentEvents events;
    if (!events.initialized) {
        AssertCuda(cudaEventCreateWithFlags(&events.comm_done[0], cudaEventDisableTiming),
                   "cudaEventCreate explicitQ comm_done[0]");
        AssertCuda(cudaEventCreateWithFlags(&events.comm_done[1], cudaEventDisableTiming),
                   "cudaEventCreate explicitQ comm_done[1]");
        AssertCuda(cudaEventCreateWithFlags(&events.compute_done[0], cudaEventDisableTiming),
                   "cudaEventCreate explicitQ compute_done[0]");
        AssertCuda(cudaEventCreateWithFlags(&events.compute_done[1], cudaEventDisableTiming),
                   "cudaEventCreate explicitQ compute_done[1]");
        events.initialized = true;
    }
    for (int slot = 0; slot < 2; ++slot) {
        AssertCuda(cudaEventRecord(events.comm_done[slot], comm_stream),
                   "cudaEventRecord explicitQ comm_done[slot]");
        AssertCuda(cudaEventRecord(events.compute_done[slot], comm_stream),
                   "cudaEventRecord explicitQ compute_done[slot]");
    }

    auto enqueue_block = [&](int block_begin, int slot) {
        BlockTask task{};
        if (block_begin < 0) {
            return task;
        }

        task.valid = true;
        task.block_begin = block_begin;
        task.block_end = std::min(block_begin + nb, n);
        task.kb = task.block_end - task.block_begin;
        task.block_rows = m - task.block_begin;
        task.owner = OwnerOfPanel(task.block_begin, part);
        task.msg_elems =
            static_cast<size_t>(task.block_rows) * static_cast<size_t>(task.kb);
        task.rank_needs_block = RankHasColsAfter(part, part.rank, task.block_begin);

        AssertCuda(cudaStreamWaitEvent(comm_stream, events.compute_done[slot], 0),
                   "cudaStreamWaitEvent explicitQ comm <- compute_done[slot]");

        if (part.rank == task.owner) {
            const int local_block_col = LocalColOffset(part, task.block_begin);
            if (local_block_col < 0) {
                spdlog::error("Explicit-Q owner rank {} missing block begin col {}.", task.owner,
                              task.block_begin);
                std::exit(1);
            }

            auto pack_range = distqr::nvtx::MakeScopedRangef("explicit_q_pack r=%d b=%d:%d s=%d",
                                                             part.rank, task.block_begin,
                                                             task.block_end, slot);
            AssertCuda(
                cudaMemcpy2DAsync(block_w_slots[slot],
                                  static_cast<size_t>(task.block_rows) * sizeof(T),
                                  d_W_local + static_cast<size_t>(local_block_col) * lda_local +
                                      task.block_begin,
                                  static_cast<size_t>(lda_local) * sizeof(T),
                                  static_cast<size_t>(task.block_rows) * sizeof(T), task.kb,
                                  cudaMemcpyDeviceToDevice, comm_stream),
                "cudaMemcpy2DAsync explicitQ block W pack");
            AssertCuda(
                cudaMemcpy2DAsync(block_y_slots[slot],
                                  static_cast<size_t>(task.block_rows) * sizeof(T),
                                  d_Y_local + static_cast<size_t>(local_block_col) * lda_local +
                                      task.block_begin,
                                  static_cast<size_t>(lda_local) * sizeof(T),
                                  static_cast<size_t>(task.block_rows) * sizeof(T), task.kb,
                                  cudaMemcpyDeviceToDevice, comm_stream),
                "cudaMemcpy2DAsync explicitQ block Y pack");
        }

        if (part.world_size > 1) {
            if (panel_comm_mode == PanelCommMode::Broadcast) {
                {
                    auto bcast_w_range =
                        distqr::nvtx::MakeScopedRangef("explicit_q_bcast_w r=%d b=%d:%d s=%d",
                                                       part.rank, task.block_begin, task.block_end,
                                                       slot);
                    AssertNccl(ncclBroadcast(block_w_slots[slot], block_w_slots[slot],
                                             task.msg_elems, nccl_type, task.owner, nccl_comm,
                                             comm_stream),
                               "ncclBroadcast explicitQ block W");
                }
                {
                    auto bcast_y_range =
                        distqr::nvtx::MakeScopedRangef("explicit_q_bcast_y r=%d b=%d:%d s=%d",
                                                       part.rank, task.block_begin, task.block_end,
                                                       slot);
                    AssertNccl(ncclBroadcast(block_y_slots[slot], block_y_slots[slot],
                                             task.msg_elems, nccl_type, task.owner, nccl_comm,
                                             comm_stream),
                               "ncclBroadcast explicitQ block Y");
                }
            } else if (part.rank == task.owner) {
                int active_receivers = 0;
                for (int r = 0; r < part.world_size; ++r) {
                    if (r == task.owner) {
                        continue;
                    }
                    if (!RankHasColsAfter(part, r, task.block_begin)) {
                        continue;
                    }
                    ++active_receivers;
                }
                if (active_receivers > 0) {
                    AssertNccl(ncclGroupStart(), "ncclGroupStart explicitQ block W/Y");
                    for (int r = 0; r < part.world_size; ++r) {
                        if (r == task.owner || !RankHasColsAfter(part, r, task.block_begin)) {
                            continue;
                        }
                        AssertNccl(
                            ncclSend(block_w_slots[slot], task.msg_elems, nccl_type, r, nccl_comm,
                                     comm_stream),
                            "ncclSend explicitQ block W");
                        AssertNccl(
                            ncclSend(block_y_slots[slot], task.msg_elems, nccl_type, r, nccl_comm,
                                     comm_stream),
                            "ncclSend explicitQ block Y");
                    }
                    AssertNccl(ncclGroupEnd(), "ncclGroupEnd explicitQ block W/Y");
                }
            } else if (task.rank_needs_block) {
                AssertNccl(ncclGroupStart(), "ncclGroupStart explicitQ recv block W/Y");
                AssertNccl(
                    ncclRecv(block_w_slots[slot], task.msg_elems, nccl_type, task.owner, nccl_comm,
                             comm_stream),
                    "ncclRecv explicitQ block W");
                AssertNccl(
                    ncclRecv(block_y_slots[slot], task.msg_elems, nccl_type, task.owner, nccl_comm,
                             comm_stream),
                    "ncclRecv explicitQ block Y");
                AssertNccl(ncclGroupEnd(), "ncclGroupEnd explicitQ recv block W/Y");
            }
        }

        AssertCuda(cudaEventRecord(events.comm_done[slot], comm_stream),
                   "cudaEventRecord explicitQ comm_done[slot]");
        return task;
    };

    const int last_block = ((n + nb - 1) / nb - 1) * nb;
    int current_slot = 0;
    BlockTask current = enqueue_block(last_block, current_slot);
    int next_slot = 1;
    BlockTask next = enqueue_block(last_block - nb, next_slot);

    while (current.valid) {
        auto block_range = distqr::nvtx::MakeScopedRangef("explicit_q_block r=%d b=%d:%d s=%d",
                                                          part.rank, current.block_begin,
                                                          current.block_end, current_slot);

        AssertCuda(cudaStreamWaitEvent(compute_stream, events.comm_done[current_slot], 0),
                   "cudaStreamWaitEvent explicitQ compute <- comm_done[current_slot]");

        const int local_begin = LocalColPrefix(part, current.block_begin);
        const int cols_local = part.local_cols - local_begin;
        if (current.rank_needs_block && cols_local > 0) {
            T* q_local_trail = d_Q_local + static_cast<size_t>(local_begin) * lda_local;
            auto apply_range =
                distqr::nvtx::MakeScopedRangef("explicit_q_apply r=%d b=%d:%d s=%d", part.rank,
                                               current.block_begin, current.block_end,
                                               current_slot);
            if (use_one_shot) {
                apply_block_q_one_shot(cublas_handle, current.block_begin, current.block_rows,
                                       current.kb, cols_local, block_w_slots[current_slot],
                                       block_y_slots[current_slot], current.block_rows,
                                       q_local_trail, lda_local, ws->d_tmp0, ws->tmp_elems);
            } else {
                apply_block_q_tile_pipeline(cublas_handle, compute_stream, current.block_begin,
                                            current.block_rows, current.kb, cols_local,
                                            work_tile_cols, block_w_slots[current_slot],
                                            block_y_slots[current_slot], current.block_rows,
                                            q_local_trail, lda_local, ws->d_tmp0, ws->d_tmp1);
            }
        }
        AssertCuda(cudaEventRecord(events.compute_done[current_slot], compute_stream),
                   "cudaEventRecord explicitQ compute_done[current_slot]");

        current = next;
        current_slot = next_slot;
        next_slot ^= 1;
        next = enqueue_block(current.valid ? (current.block_begin - nb) : -1, next_slot);
    }

}

}  // namespace distributed_qr_col_blockcyclic
