#pragma once
// Column-partitioned distributed blocked QR with 1D block-cyclic column layout.

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
#include "utils/nvtx_range.cuh"

namespace distributed_qr_col_blockcyclic {

constexpr int kPanelWidth = 32;

template <typename T>
constexpr bool kSupportedQrType = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
constexpr ncclDataType_t NcclType() {
    static_assert(kSupportedQrType<T>,
                  "distributed_qr_col_blockcyclic only supports float and double.");
    if constexpr (std::is_same_v<T, float>) {
        return ncclFloat;
    } else {
        return ncclDouble;
    }
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

struct ColBlockCyclicPartition {
    int n_global = 0;
    int world_size = 1;
    int rank = 0;
    int block_cols = 0;
    std::vector<int> block_starts;
    std::vector<int> block_ends;
    std::vector<int> block_local_offsets;
    int local_cols = 0;
};

inline ColBlockCyclicPartition MakeColBlockCyclicPartition(int n_global,
                                                           int block_cols,
                                                           int world_size,
                                                           int rank) {
    ColBlockCyclicPartition part{};
    part.n_global = n_global;
    part.block_cols = block_cols;
    part.world_size = world_size;
    part.rank = rank;

    if (block_cols <= 0) {
        return part;
    }

    int local_cols = 0;
    int block_idx = 0;
    for (int col = 0; col < n_global; col += block_cols, ++block_idx) {
        const int end = std::min(col + block_cols, n_global);
        if (block_idx % world_size == rank) {
            part.block_starts.push_back(col);
            part.block_ends.push_back(end);
            part.block_local_offsets.push_back(local_cols);
            local_cols += end - col;
        }
    }
    part.local_cols = local_cols;
    return part;
}

inline int OwnerOfPanel(int panel_col, const ColBlockCyclicPartition& part) {
    if (part.block_cols <= 0) {
        return 0;
    }
    const int block_idx = panel_col / part.block_cols;
    return block_idx % part.world_size;
}

inline bool RankHasColsAfter(const ColBlockCyclicPartition& part, int rank, int col) {
    if (col >= part.n_global) {
        return false;
    }
    if (part.block_cols <= 0 || part.world_size <= 0) {
        return (rank == 0) && (col < part.n_global);
    }

    const int num_blocks = (part.n_global + part.block_cols - 1) / part.block_cols;
    const int first_block_idx = col / part.block_cols;
    int idx = rank;
    if (idx < first_block_idx) {
        const int delta = first_block_idx - idx;
        const int steps = (delta + part.world_size - 1) / part.world_size;
        idx += steps * part.world_size;
    }
    return idx < num_blocks;
}

inline int LocalColOffset(const ColBlockCyclicPartition& part, int global_col) {
    for (size_t i = 0; i < part.block_starts.size(); ++i) {
        if (global_col >= part.block_starts[i] && global_col < part.block_ends[i]) {
            return part.block_local_offsets[i] + (global_col - part.block_starts[i]);
        }
    }
    return -1;
}

inline int LocalColPrefix(const ColBlockCyclicPartition& part, int global_col) {
    if (global_col <= 0) {
        return 0;
    }
    if (global_col >= part.n_global) {
        return part.local_cols;
    }

    int prefix = 0;
    for (size_t i = 0; i < part.block_starts.size(); ++i) {
        const int block_begin = part.block_starts[i];
        const int block_end = part.block_ends[i];
        if (global_col <= block_begin) {
            break;
        }
        prefix += std::min(global_col, block_end) - block_begin;
    }
    return prefix;
}

template <typename Func>
inline void ForEachLocalSegment(const ColBlockCyclicPartition& part,
                                int begin,
                                int end,
                                Func&& fn) {
    if (end <= begin) {
        return;
    }
    for (size_t i = 0; i < part.block_starts.size(); ++i) {
        const int seg_begin = std::max(begin, part.block_starts[i]);
        const int seg_end = std::min(end, part.block_ends[i]);
        if (seg_begin >= seg_end) {
            continue;
        }
        const int local_begin = part.block_local_offsets[i] + (seg_begin - part.block_starts[i]);
        fn(seg_begin, seg_end, local_begin);
    }
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
struct DistributedQrColBlockCyclicWorkspace {
    T* d_r_panel = nullptr;  // [kPanelWidth x kPanelWidth]
    T* d_tsqr_work_panel = nullptr;
    size_t tsqr_work_panel_elems = 0;

    // Packed (contiguous) W/Y for a single panel: [panel_rows x kPanelWidth].
    // Double-buffered so that panel k comm can overlap panel k+1 compute.
    T* d_pack_w[2] = {nullptr, nullptr};
    T* d_pack_y[2] = {nullptr, nullptr};
    size_t pack_elems = 0;

    // Block-level WY buffers for a single outer block (nb columns).
    T* d_block_w = nullptr;
    T* d_block_y = nullptr;
    size_t block_storage_elems = 0;

    // Work buffers for GEMM: [k x tile_cols], k up to nb.
    T* d_tmp0 = nullptr;
    T* d_tmp1 = nullptr;
    size_t tmp_elems = 0;

    // Compact block WY buffers: [(nb + kPanelWidth) x nb]
    // Used to build a global block WY without scattering into [m x nb].
    T* d_block_w_compact = nullptr;
    T* d_block_y_compact = nullptr;
    size_t block_compact_elems = 0;
};

struct CommProfile {
    size_t bytes = 0;
};

enum class PanelCommMode {
    SendRecv = 0,
    Broadcast = 1,
};

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
    AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kPanelWidth,
                                           cols_local, panel_rows, &one, W, lda_wy, a_sub,
                                           lda_local, &zero, work, kPanelWidth),
                 "in-block work = W^T * A");
    AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_rows,
                                           cols_local, kPanelWidth, &minus_one, Y, lda_wy, work,
                                           kPanelWidth, &one, a_sub, lda_local),
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
        AssertCublas(
            CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, width, rows, &one,
                                      W, lda_wy, a_tile_sub, lda_local, &zero, tmps[idx], k),
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
            AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, rows,
                                                   widths[prev], k, &minus_one, Y, lda_wy,
                                                   tmps[prev], k, &one, a_prev_sub, lda_local),
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
                                               widths[last], k, &minus_one, Y, lda_wy, tmps[last],
                                               k, &one, a_last_sub, lda_local),
                     "A_last -= Y * tmp");
        AssertCuda(cudaEventRecord(events.apply_done[last], compute_stream),
                   "cudaEventRecord apply_done[last]");
    }
}

template <typename T>
void block_update_one_shot(cublasHandle_t cublas_handle,
                           int row_offset,
                           int rows,
                           int k,
                           int cols_local,
                           const T* W,
                           const T* Y,
                           int lda_wy,
                           T* A_trail,
                           int lda_local,
                           T* work,
                           size_t work_elems) {
    if (k <= 0 || cols_local <= 0 || rows <= 0) {
        return;
    }

    const size_t need = static_cast<size_t>(k) * static_cast<size_t>(cols_local);
    if (work_elems < need) {
        spdlog::error("block_update_one_shot work too small (need {} elems, got {}).", need,
                      work_elems);
        std::exit(1);
    }

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const T minus_one = static_cast<T>(-1);

    T* a_sub = A_trail + row_offset;
    AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, cols_local,
                                           rows, &one, W, lda_wy, a_sub, lda_local, &zero, work, k),
                 "trail one-shot work = W^T * A");
    AssertCublas(
        CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols_local, k,
                                  &minus_one, Y, lda_wy, work, k, &one, a_sub, lda_local),
        "trail one-shot A -= Y * work");
}

template <typename T>
void distributed_blocked_qr_factorize_col_blockcyclic(
    cublasHandle_t cublas_handle,
    ncclComm_t nccl_comm,
    const ColBlockCyclicPartition& part,
    int m,
    int n,
    int nb,
    T* d_A_local,
    int lda_local,
    T* d_W_local,
    T* d_Y_local,
    DistributedQrColBlockCyclicWorkspace<T>* ws,
    cudaStream_t compute_stream,
    cudaStream_t comm_stream,
    int overlap_tile_cols = 0,
    CommProfile* comm_profile = nullptr,
    PanelCommMode panel_comm_mode = PanelCommMode::SendRecv,
    bool use_compact_local_gemm = false) {
    static_assert(kSupportedQrType<T>,
                  "distributed_qr_col_blockcyclic only supports float and double.");

    if (!ws) {
        spdlog::error("distributed_blocked_qr_factorize_col_blockcyclic got null workspace.");
        std::exit(1);
    }
    if (!ws->d_r_panel) {
        spdlog::error("Col blockcyclic workspace d_r_panel is null.");
        std::exit(1);
    }
    const size_t tsqr_work_need = tsqr_work_elems<T>(m);
    if (tsqr_work_need > 0 &&
        (!ws->d_tsqr_work_panel || ws->tsqr_work_panel_elems < tsqr_work_need)) {
        spdlog::error("Col blockcyclic TSQR workspace too small (need {} elems, got {}).",
                      tsqr_work_need, ws->tsqr_work_panel_elems);
        std::exit(1);
    }
    if (n != part.n_global) {
        spdlog::error("ColBlockCyclicPartition.n_global ({}) mismatches n ({}).", part.n_global, n);
        std::exit(1);
    }
    if (n % kPanelWidth != 0 || nb % kPanelWidth != 0) {
        spdlog::error("Require n and nb to be multiples of {} (got n={} nb={}).", kPanelWidth, n,
                      nb);
        std::exit(1);
    }
    if (part.block_cols <= 0 || part.block_cols % nb != 0 || part.block_cols % kPanelWidth != 0) {
        spdlog::error("Require block_cols to be positive and multiple of nb and {} (got {}).",
                      kPanelWidth, part.block_cols);
        std::exit(1);
    }
    if (panel_comm_mode != PanelCommMode::SendRecv && panel_comm_mode != PanelCommMode::Broadcast) {
        spdlog::error("Unsupported panel_comm_mode value {}.", static_cast<int>(panel_comm_mode));
        std::exit(1);
    }

    const bool trail_one_shot = overlap_tile_cols <= 0;
    const int tile_cols = trail_one_shot ? std::max(part.local_cols, 1)
                                         : std::max(kPanelWidth, std::min(overlap_tile_cols, nb));
    const size_t tmp_need = static_cast<size_t>(nb) * static_cast<size_t>(tile_cols);
    if (ws->tmp_elems < tmp_need) {
        spdlog::error("Col blockcyclic tmp too small (need {} elems, got {}).", tmp_need,
                      ws->tmp_elems);
        std::exit(1);
    }
    if (!ws->d_tmp0 || !ws->d_tmp1) {
        spdlog::error("Col blockcyclic tmp buffers are null.");
        std::exit(1);
    }
    if (comm_profile) {
        comm_profile->bytes = 0;
    }
    const size_t pack_need = static_cast<size_t>(m) * static_cast<size_t>(kPanelWidth);
    if (!ws->d_pack_w[0] || !ws->d_pack_w[1] || !ws->d_pack_y[0] || !ws->d_pack_y[1] ||
        ws->pack_elems < pack_need) {
        spdlog::error(
            "Col blockcyclic pack buffers too small or null (need {} elems per buffer, got {}).",
            pack_need, ws->pack_elems);
        std::exit(1);
    }
    const size_t block_storage_need = static_cast<size_t>(m) * static_cast<size_t>(nb);
    if (!ws->d_block_w || !ws->d_block_y || ws->block_storage_elems < block_storage_need) {
        spdlog::error("Col blockcyclic block storage too small (need {} elems, got {}).",
                      block_storage_need, ws->block_storage_elems);
        std::exit(1);
    }
    const int compact_rows = nb + kPanelWidth;
    const size_t compact_need = static_cast<size_t>(compact_rows) * static_cast<size_t>(nb);
    if (!ws->d_block_w_compact || !ws->d_block_y_compact ||
        ws->block_compact_elems < compact_need) {
        spdlog::error("Col blockcyclic compact storage too small (need {} elems, got {}).",
                      compact_need, ws->block_compact_elems);
        std::exit(1);
    }

    AssertCublas(cublasSetStream(cublas_handle, compute_stream), "cublasSetStream(compute_stream)");
    const ncclDataType_t nccl_type = NcclType<T>();

    struct PersistentEvents {
        bool initialized = false;
        cudaEvent_t panel_ready[2] = {};
        cudaEvent_t comm_done[2] = {};
        cudaEvent_t compute_done[2] = {};
    };
    static PersistentEvents events;
    if (!events.initialized) {
        for (int i = 0; i < 2; ++i) {
            AssertCuda(cudaEventCreateWithFlags(&events.panel_ready[i], cudaEventDisableTiming),
                       "cudaEventCreate panel_ready[i]");
            AssertCuda(cudaEventCreateWithFlags(&events.comm_done[i], cudaEventDisableTiming),
                       "cudaEventCreate comm_done[i]");
            AssertCuda(cudaEventCreateWithFlags(&events.compute_done[i], cudaEventDisableTiming),
                       "cudaEventCreate compute_done[i]");
        }
        events.initialized = true;
    }

    // Initialize reuse guards: both pack buffers start out free.
    AssertCuda(cudaEventRecord(events.comm_done[0], comm_stream), "cudaEventRecord comm_done[0]");
    AssertCuda(cudaEventRecord(events.comm_done[1], comm_stream), "cudaEventRecord comm_done[1]");
    AssertCuda(cudaEventRecord(events.compute_done[0], compute_stream),
               "cudaEventRecord compute_done[0]");
    AssertCuda(cudaEventRecord(events.compute_done[1], compute_stream),
               "cudaEventRecord compute_done[1]");

    auto prefetch_recv = [&](int inner, int buf) {
        if (panel_comm_mode != PanelCommMode::SendRecv) {
            return;
        }
        if (part.world_size <= 1) {
            return;
        }
        const int owner = OwnerOfPanel(inner, part);
        if (part.rank == owner) {
            return;
        }
        if (!RankHasColsAfter(part, part.rank, inner + kPanelWidth)) {
            return;
        }

        const int panel_rows = m - inner;
        T* d_pack_w = ws->d_pack_w[buf];
        T* d_pack_y = ws->d_pack_y[buf];
        AssertCuda(cudaStreamWaitEvent(comm_stream, events.compute_done[buf], 0),
                   "cudaStreamWaitEvent comm_stream <- compute_done[buf](prefetch)");
        AssertNccl(ncclGroupStart(), "ncclGroupStart panel W/Y(prefetch)");
        AssertNccl(ncclRecv(d_pack_w, static_cast<size_t>(panel_rows) * kPanelWidth, nccl_type,
                            owner, nccl_comm, comm_stream),
                   "ncclRecv panel W(prefetch)");
        AssertNccl(ncclRecv(d_pack_y, static_cast<size_t>(panel_rows) * kPanelWidth, nccl_type,
                            owner, nccl_comm, comm_stream),
                   "ncclRecv panel Y(prefetch)");
        AssertNccl(ncclGroupEnd(), "ncclGroupEnd panel W/Y(prefetch)");
        if (comm_profile) {
            comm_profile->bytes += 2ULL * static_cast<size_t>(panel_rows) * kPanelWidth * sizeof(T);
        }
        AssertCuda(cudaEventRecord(events.comm_done[buf], comm_stream),
                   "cudaEventRecord comm_done[buf](prefetch)");
    };

    auto count_active_receivers = [&](int inner, int owner) {
        int active_receivers = 0;
        for (int r = 0; r < part.world_size; ++r) {
            if (r == owner) {
                continue;
            }
            if (!RankHasColsAfter(part, r, inner + kPanelWidth)) {
                continue;
            }
            ++active_receivers;
        }
        return active_receivers;
    };

    auto prepare_owner_panel = [&](int block_begin_for_panel, int inner, int buf) {
        const int owner = OwnerOfPanel(inner, part);
        if (part.rank != owner) {
            return;
        }

        const int panel_rows = m - inner;
        T* d_pack_w = ws->d_pack_w[buf];
        T* d_pack_y = ws->d_pack_y[buf];

        // Ensure comm has finished consuming this pack buffer before overwriting it.
        AssertCuda(cudaStreamWaitEvent(compute_stream, events.comm_done[buf], 0),
                   "cudaStreamWaitEvent compute_stream <- comm_done[buf]");

        auto panel_prepare_range = distqr::nvtx::MakeScopedRangef(
            "panel_prepare r=%d b=%d i=%d o=%d", part.rank, block_begin_for_panel, inner, owner);

        const int local_panel_col = LocalColOffset(part, inner);
        if (local_panel_col < 0) {
            spdlog::error("Owner rank {} missing panel col {} in local layout.", owner, inner);
            std::exit(1);
        }

        T* panel_A = d_A_local + static_cast<size_t>(local_panel_col) * lda_local;
        T* panel_A_sub = panel_A + inner;

        tsqr<T>(cublas_handle, panel_rows, panel_A_sub, lda_local, ws->d_r_panel, kPanelWidth,
                ws->d_tsqr_work_panel, ws->tsqr_work_panel_elems, compute_stream);

        generate_wy<T>(panel_rows, kPanelWidth, panel_A_sub, lda_local, d_pack_y, panel_rows,
                       d_pack_w, panel_rows, compute_stream);

        const dim3 block_dim(16, 16);
        const dim3 grid_dim((kPanelWidth + block_dim.x - 1) / block_dim.x,
                            (kPanelWidth + block_dim.y - 1) / block_dim.y);
        write_upper_r_to_panel_kernel<<<grid_dim, block_dim, 0, compute_stream>>>(
            inner, ws->d_r_panel, kPanelWidth, panel_A, lda_local);
        AssertCuda(cudaGetLastError(), "write_upper_r_to_panel_kernel launch");

        AssertCuda(cudaEventRecord(events.panel_ready[buf], compute_stream),
                   "cudaEventRecord panel_ready[buf]");
        AssertCuda(cudaStreamWaitEvent(comm_stream, events.panel_ready[buf], 0),
                   "cudaStreamWaitEvent comm_stream <- panel_ready[buf]");
    };

    auto launch_panel_comm = [&](int block_begin_for_panel, int inner, int buf) {
        const int panel_rows = m - inner;
        const int owner = OwnerOfPanel(inner, part);
        const int active_receivers = count_active_receivers(inner, owner);
        T* d_pack_w = ws->d_pack_w[buf];
        T* d_pack_y = ws->d_pack_y[buf];

        if (part.world_size > 1) {
            if (panel_comm_mode == PanelCommMode::Broadcast) {
                {
                    auto panel_bcast_w_range = distqr::nvtx::MakeScopedRangef(
                        "panel_bcast_w r=%d b=%d i=%d o=%d", part.rank, block_begin_for_panel,
                        inner, owner);
                    AssertNccl(ncclBroadcast(d_pack_w, d_pack_w,
                                             static_cast<size_t>(panel_rows) * kPanelWidth,
                                             nccl_type, owner, nccl_comm, comm_stream),
                               "ncclBroadcast panel W");
                }
                {
                    auto panel_bcast_y_range = distqr::nvtx::MakeScopedRangef(
                        "panel_bcast_y r=%d b=%d i=%d o=%d", part.rank, block_begin_for_panel,
                        inner, owner);
                    AssertNccl(ncclBroadcast(d_pack_y, d_pack_y,
                                             static_cast<size_t>(panel_rows) * kPanelWidth,
                                             nccl_type, owner, nccl_comm, comm_stream),
                               "ncclBroadcast panel Y");
                }
                if (comm_profile && active_receivers > 0) {
                    comm_profile->bytes += 2ULL * static_cast<size_t>(panel_rows) * kPanelWidth *
                                           sizeof(T) * static_cast<size_t>(active_receivers);
                }
                AssertCuda(cudaEventRecord(events.comm_done[buf], comm_stream),
                           "cudaEventRecord comm_done[buf](bcast)");
            } else if (part.rank == owner) {
                const bool any_send = (active_receivers > 0);
                if (any_send) {
                    AssertNccl(ncclGroupStart(), "ncclGroupStart panel W/Y");
                }
                for (int r = 0; r < part.world_size; ++r) {
                    if (r == owner) {
                        continue;
                    }
                    if (!RankHasColsAfter(part, r, inner + kPanelWidth)) {
                        continue;
                    }
                    AssertNccl(ncclSend(d_pack_w, static_cast<size_t>(panel_rows) * kPanelWidth,
                                        nccl_type, r, nccl_comm, comm_stream),
                               "ncclSend panel W");
                    AssertNccl(ncclSend(d_pack_y, static_cast<size_t>(panel_rows) * kPanelWidth,
                                        nccl_type, r, nccl_comm, comm_stream),
                               "ncclSend panel Y");
                    if (comm_profile) {
                        comm_profile->bytes +=
                            2ULL * static_cast<size_t>(panel_rows) * kPanelWidth * sizeof(T);
                    }
                }
                if (any_send) {
                    AssertNccl(ncclGroupEnd(), "ncclGroupEnd panel W/Y");
                    AssertCuda(cudaEventRecord(events.comm_done[buf], comm_stream),
                               "cudaEventRecord comm_done[buf](send)");
                } else {
                    // No sends: buffer can be considered comm-free immediately.
                    AssertCuda(cudaEventRecord(events.comm_done[buf], comm_stream),
                               "cudaEventRecord comm_done[buf](nosend)");
                }
            }
        } else {
            // No NCCL comm: for non-owner there is nothing to receive; for owner the pack is
            // immediately usable and can be overwritten later due to in-stream ordering.
            AssertCuda(cudaEventRecord(events.comm_done[buf], comm_stream),
                       "cudaEventRecord comm_done[buf](np1)");
        }
    };

    int panel_seq = 0;
    bool pending_lookahead = false;
    int pending_block_begin = -1;
    int pending_inner = -1;
    int pending_buf = -1;
    for (int block_begin = 0; block_begin < n; block_begin += nb) {
        const int block_end = std::min(block_begin + nb, n);
        const int kb = block_end - block_begin;
        auto block_range =
            distqr::nvtx::MakeScopedRangef("qr_block r=%d b=%d:%d", part.rank, block_begin,
                                           block_end);

        const bool block_has_pending_first_panel =
            pending_lookahead && pending_block_begin == block_begin && pending_inner == block_begin;
        if (pending_lookahead && !block_has_pending_first_panel) {
            spdlog::error(
                "Invalid pending lookahead state (pending block_begin={} inner={}, current "
                "block_begin={}).",
                pending_block_begin, pending_inner, block_begin);
            std::exit(1);
        }

        // Receiver-side lookahead: start receiving the first panel of this block early.
        if (!block_has_pending_first_panel) {
            prefetch_recv(block_begin, panel_seq & 1);
        }

        for (int inner = block_begin; inner < block_end; inner += kPanelWidth) {
            const int buf = panel_seq & 1;
            const bool use_pending_panel =
                pending_lookahead && pending_block_begin == block_begin && pending_inner == inner;
            if (use_pending_panel && pending_buf != buf) {
                spdlog::error("Pending lookahead buffer mismatch (pending buf={}, current buf={}).",
                              pending_buf, buf);
                std::exit(1);
            }
            ++panel_seq;
            const int panel_rows = m - inner;
            const int owner = OwnerOfPanel(inner, part);
            const int block_col_off = inner - block_begin;
            const bool self_needs_panel = RankHasColsAfter(part, part.rank, inner + kPanelWidth);
            T* d_pack_w = ws->d_pack_w[buf];
            T* d_pack_y = ws->d_pack_y[buf];

            if (use_pending_panel) {
                pending_lookahead = false;
                pending_block_begin = -1;
                pending_inner = -1;
                pending_buf = -1;
            } else if (part.rank == owner) {
                prepare_owner_panel(block_begin, inner, buf);
            }

            if (!use_pending_panel) {
                launch_panel_comm(block_begin, inner, buf);
            }

            // Receiver-side lookahead: enqueue the next panel receive before we start the heavy
            // compute for this panel.
            const int next_inner = inner + kPanelWidth;
            if (next_inner < block_end) {
                prefetch_recv(next_inner, buf ^ 1);
            }

            // Apply this panel to local columns inside the current block using the original
            // panel WY (before block-WY update).
            if (part.rank == owner || self_needs_panel) {
                if (part.world_size > 1 && part.rank != owner) {
                    AssertCuda(cudaStreamWaitEvent(compute_stream, events.comm_done[buf], 0),
                               "cudaStreamWaitEvent compute_stream <- comm_done[buf](use)");
                }
                const int inblock_begin = inner + kPanelWidth;
                const int inblock_end = block_end;
                if (use_compact_local_gemm) {
                    const int local_begin = LocalColPrefix(part, inblock_begin);
                    const int cols_local = LocalColPrefix(part, inblock_end) - local_begin;
                    if (cols_local > 0) {
                        T* a_trail = d_A_local + static_cast<size_t>(local_begin) * lda_local;
                        auto panel_apply_range = distqr::nvtx::MakeScopedRangef(
                            "panel_apply r=%d i=%d seg=%d:%d", part.rank, inner, inblock_begin,
                            inblock_end);
                        panel_update_one_shot(cublas_handle, inner, panel_rows, cols_local,
                                              d_pack_w, d_pack_y, panel_rows, a_trail, lda_local,
                                              ws->d_tmp0, ws->tmp_elems);
                    }
                } else {
                    ForEachLocalSegment(
                        part, inblock_begin, inblock_end,
                        [&](int seg_begin, int seg_end, int local_begin) {
                            const int cols_local = seg_end - seg_begin;
                            T* a_trail = d_A_local + static_cast<size_t>(local_begin) * lda_local;
                            auto panel_apply_range = distqr::nvtx::MakeScopedRangef(
                                "panel_apply r=%d i=%d seg=%d:%d", part.rank, inner, seg_begin,
                                seg_end);
                            panel_update_one_shot(cublas_handle, inner, panel_rows, cols_local,
                                                  d_pack_w, d_pack_y, panel_rows, a_trail,
                                                  lda_local, ws->d_tmp0, ws->tmp_elems);
                        });
                }
            }

            if (part.rank == owner) {
                const int local_panel_col = LocalColOffset(part, inner);
                if (local_panel_col < 0) {
                    spdlog::error("Owner rank {} missing panel col {} in local layout.", owner,
                                  inner);
                    std::exit(1);
                }
                if (d_Y_local) {
                    AssertCuda(
                        cudaMemcpy2DAsync(
                            d_Y_local + static_cast<size_t>(local_panel_col) * lda_local + inner,
                            static_cast<size_t>(lda_local) * sizeof(T), d_pack_y,
                            static_cast<size_t>(panel_rows) * sizeof(T),
                            static_cast<size_t>(panel_rows) * sizeof(T), kPanelWidth,
                            cudaMemcpyDeviceToDevice, compute_stream),
                        "cudaMemcpy2DAsync pack_y -> d_Y_local");
                    const int zero_rows = inner - block_begin;
                    if (zero_rows > 0) {
                        T* z_y = d_Y_local + static_cast<size_t>(local_panel_col) * lda_local +
                                 block_begin;
                        AssertCuda(
                            cudaMemset2DAsync(z_y, static_cast<size_t>(lda_local) * sizeof(T), 0,
                                              static_cast<size_t>(zero_rows) * sizeof(T),
                                              kPanelWidth, compute_stream),
                            "cudaMemset2DAsync zero d_Y_local top");
                    }
                }
            }

            // Scatter packed W/Y into block buffers and zero top of the panel columns.
            if (part.rank == owner || self_needs_panel) {
                T* dst_w = ws->d_block_w + static_cast<size_t>(inner) +
                           static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
                T* dst_y = ws->d_block_y + static_cast<size_t>(inner) +
                           static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
                AssertCuda(cudaMemcpy2DAsync(dst_w, static_cast<size_t>(m) * sizeof(T), d_pack_w,
                                             static_cast<size_t>(panel_rows) * sizeof(T),
                                             static_cast<size_t>(panel_rows) * sizeof(T),
                                             kPanelWidth, cudaMemcpyDeviceToDevice, compute_stream),
                           "cudaMemcpy2DAsync pack_w -> block_w");
                AssertCuda(cudaMemcpy2DAsync(dst_y, static_cast<size_t>(m) * sizeof(T), d_pack_y,
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

                // Mark this pack buffer safe to overwrite (comm may still be using it, compute
                // only). Receivers will use this to avoid overwriting pack while compute reads it.
                AssertCuda(cudaEventRecord(events.compute_done[buf], compute_stream),
                           "cudaEventRecord compute_done[buf]");
            }

            // Now update W_i inside the block WY using the full block rows.
            if (part.rank == owner || self_needs_panel) {
                if (inner > block_begin) {
                    auto block_w_update_range = distqr::nvtx::MakeScopedRangef(
                        "block_w_update r=%d b=%d i=%d", part.rank, block_begin, inner);
                    const int k_prev = inner - block_begin;
                    const int block_rows = m - block_begin;
                    const T one = static_cast<T>(1);
                    const T zero = static_cast<T>(0);
                    const T minus_one = static_cast<T>(-1);

                    const T* y_prev_sub = ws->d_block_y + static_cast<size_t>(block_begin);
                    const T* w_prev_sub = ws->d_block_w + static_cast<size_t>(block_begin);
                    T* w_i_sub = ws->d_block_w + static_cast<size_t>(block_begin) +
                                 static_cast<size_t>(block_col_off) * static_cast<size_t>(m);

                    AssertCublas(
                        CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k_prev,
                                                  kPanelWidth, block_rows, &one, y_prev_sub, m,
                                                  w_i_sub, m, &zero, ws->d_tmp0, k_prev),
                        "tmp = Y_prev^T * W_i");
                    AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                           block_rows, kPanelWidth, k_prev,
                                                           &minus_one, w_prev_sub, m, ws->d_tmp0,
                                                           k_prev, &one, w_i_sub, m),
                                 "W_i -= W_prev * tmp");
                }
            }

            if (part.rank == owner && d_W_local) {
                const int local_panel_col = LocalColOffset(part, inner);
                if (local_panel_col < 0) {
                    spdlog::error("Owner rank {} missing panel col {} in local layout.", owner,
                                  inner);
                    std::exit(1);
                }
                const int block_rows = m - block_begin;
                T* src_w = ws->d_block_w + static_cast<size_t>(block_begin) +
                           static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
                AssertCuda(
                    cudaMemcpy2DAsync(
                        d_W_local + static_cast<size_t>(local_panel_col) * lda_local + block_begin,
                        static_cast<size_t>(lda_local) * sizeof(T), src_w,
                        static_cast<size_t>(m) * sizeof(T),
                        static_cast<size_t>(block_rows) * sizeof(T), kPanelWidth,
                        cudaMemcpyDeviceToDevice, compute_stream),
                    "cudaMemcpy2DAsync block_w -> d_W_local");
            }
        }

        const int next_block_begin = block_end;
        const int next_block_end = std::min(next_block_begin + nb, n);
        const int next_panel_end = std::min(next_block_begin + kPanelWidth, n);
        const int block_rows = m - block_begin;
        bool frontier_applied_here = false;
        if (next_block_begin < n) {
            const int next_buf = panel_seq & 1;
            const int next_owner = OwnerOfPanel(next_block_begin, part);
            if (panel_comm_mode == PanelCommMode::SendRecv) {
                prefetch_recv(next_block_begin, next_buf);
            }

            if (part.rank == next_owner) {
                const int local_panel_col = LocalColOffset(part, next_block_begin);
                if (local_panel_col < 0) {
                    spdlog::error("Next owner rank {} missing panel col {} in local layout.",
                                  next_owner, next_block_begin);
                    std::exit(1);
                }
                T* a_frontier = d_A_local + static_cast<size_t>(local_panel_col) * lda_local;
                const int frontier_cols = next_panel_end - next_block_begin;
                if (frontier_cols > 0) {
                    frontier_applied_here = true;
                    if (trail_one_shot) {
                        auto frontier_update_range = distqr::nvtx::MakeScopedRangef(
                            "frontier_update_1shot r=%d b=%d seg=%d:%d", part.rank, block_begin,
                            next_block_begin, next_panel_end);
                        block_update_one_shot(cublas_handle, block_begin, block_rows, kb,
                                              frontier_cols,
                                              ws->d_block_w + static_cast<size_t>(block_begin),
                                              ws->d_block_y + static_cast<size_t>(block_begin), m,
                                              a_frontier, lda_local, ws->d_tmp0, ws->tmp_elems);
                    } else {
                        auto frontier_update_range = distqr::nvtx::MakeScopedRangef(
                            "frontier_update_tiled r=%d b=%d seg=%d:%d", part.rank, block_begin,
                            next_block_begin, next_panel_end);
                        block_update_tile_pipeline(
                            cublas_handle, compute_stream, block_begin, block_rows, kb,
                            frontier_cols, tile_cols,
                            ws->d_block_w + static_cast<size_t>(block_begin),
                            ws->d_block_y + static_cast<size_t>(block_begin), m, a_frontier,
                            lda_local, ws->d_tmp0, ws->d_tmp1);
                    }
                }
                prepare_owner_panel(next_block_begin, next_block_begin, next_buf);
            }

            if (panel_comm_mode == PanelCommMode::Broadcast || part.rank == next_owner ||
                part.world_size <= 1) {
                launch_panel_comm(next_block_begin, next_block_begin, next_buf);
            }

            pending_lookahead = true;
            pending_block_begin = next_block_begin;
            pending_inner = next_block_begin;
            pending_buf = next_buf;
        }

        // Apply block WY to trailing columns after the block (full rows, like single-GPU).
        const int trail_begin =
            frontier_applied_here ? std::min(next_panel_end, next_block_end) : block_end;
        if (use_compact_local_gemm) {
            const int local_begin = LocalColPrefix(part, trail_begin);
            const int cols_local = LocalColPrefix(part, n) - local_begin;
            if (cols_local > 0) {
                T* a_trail = d_A_local + static_cast<size_t>(local_begin) * lda_local;
                if (trail_one_shot) {
                    auto trail_update_range = distqr::nvtx::MakeScopedRangef(
                        "trail_update_1shot r=%d b=%d seg=%d:%d", part.rank, block_begin,
                        trail_begin, n);
                    block_update_one_shot(cublas_handle, block_begin, block_rows, kb, cols_local,
                                          ws->d_block_w + static_cast<size_t>(block_begin),
                                          ws->d_block_y + static_cast<size_t>(block_begin), m,
                                          a_trail, lda_local, ws->d_tmp0, ws->tmp_elems);
                } else {
                    auto trail_update_range = distqr::nvtx::MakeScopedRangef(
                        "trail_update_tiled r=%d b=%d seg=%d:%d", part.rank, block_begin,
                        trail_begin, n);
                    block_update_tile_pipeline(cublas_handle, compute_stream, block_begin,
                                               block_rows, kb, cols_local, tile_cols,
                                               ws->d_block_w + static_cast<size_t>(block_begin),
                                               ws->d_block_y + static_cast<size_t>(block_begin), m,
                                               a_trail, lda_local, ws->d_tmp0, ws->d_tmp1);
                }
            }
        } else {
            ForEachLocalSegment(
                part, trail_begin, n, [&](int seg_begin, int seg_end, int local_begin) {
                    const int cols_local = seg_end - seg_begin;
                    T* a_trail = d_A_local + static_cast<size_t>(local_begin) * lda_local;
                    if (trail_one_shot) {
                        auto trail_update_range = distqr::nvtx::MakeScopedRangef(
                            "trail_update_1shot r=%d b=%d seg=%d:%d", part.rank, block_begin,
                            seg_begin, seg_end);
                        block_update_one_shot(cublas_handle, block_begin, block_rows, kb,
                                              cols_local,
                                              ws->d_block_w + static_cast<size_t>(block_begin),
                                              ws->d_block_y + static_cast<size_t>(block_begin), m,
                                              a_trail, lda_local, ws->d_tmp0, ws->tmp_elems);
                    } else {
                        auto trail_update_range = distqr::nvtx::MakeScopedRangef(
                            "trail_update_tiled r=%d b=%d seg=%d:%d", part.rank, block_begin,
                            seg_begin, seg_end);
                        block_update_tile_pipeline(cublas_handle, compute_stream, block_begin,
                                                   block_rows, kb, cols_local, tile_cols,
                                                   ws->d_block_w + static_cast<size_t>(block_begin),
                                                   ws->d_block_y + static_cast<size_t>(block_begin),
                                                   m, a_trail, lda_local, ws->d_tmp0, ws->d_tmp1);
                    }
                });
        }
    }
}

}  // namespace distributed_qr_col_blockcyclic
