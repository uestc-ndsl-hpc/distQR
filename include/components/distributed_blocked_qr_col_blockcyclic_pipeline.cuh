#pragma once
// Column-partitioned distributed blocked QR scaffold for a block-cyclic column layout.
//
// This file intentionally mirrors the structure of
// distributed_blocked_qr_col_blockcyclic.cuh, but only migrates the reusable
// distributed skeleton:
//   1. partition helpers,
//   2. owner-side panel factorization,
//   3. compact block-WY construction,
//   4. profiling / accounting hooks.
//
// The panel-by-panel propagation/update path from the existing implementation is
// intentionally not migrated. The long-term pipeline direction for this file is:
//   owner builds compact block-WY,
//   owner broadcasts block-WY k-tiles,
//   each rank performs a local row-block streamed trailing update.
//
// In other words, this file is a migration scaffold, not a finished algorithm.

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

namespace distributed_qr_col_blockcyclic_pipeline {

constexpr int kPanelWidth = 32;

template <typename T>
constexpr bool kSupportedQrType = std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
constexpr ncclDataType_t NcclType() {
    static_assert(kSupportedQrType<T>,
                  "distributed_qr_col_blockcyclic_pipeline only supports float and double.");
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
struct DistributedQrColBlockCyclicPipelineWorkspace {
    T* d_r_panel = nullptr;  // [kPanelWidth x kPanelWidth]
    T* d_tsqr_work_panel = nullptr;
    size_t tsqr_work_panel_elems = 0;

    // Owner-side staging for the current panel.
    // Unlike the existing implementation we do not keep a double-buffered
    // panel communication path here. The intended pipeline only communicates
    // block-level compact WY once panel processing for the block is complete.
    T* d_panel_w = nullptr;  // [panel_rows x kPanelWidth], worst-case panel_rows = m
    T* d_panel_y = nullptr;  // [panel_rows x kPanelWidth], worst-case panel_rows = m
    size_t panel_elems = 0;

    // Canonical block-WY storage for the owner.
    // Layout is compact: [block_rows x nb], leading dimension = block_rows.
    // This replaces the padded [m x nb] + compact copy pair from the existing
    // implementation.
    T* d_block_w = nullptr;
    T* d_block_y = nullptr;
    size_t block_elems = 0;

    // Row-block staging for the future streamed trailing update.
    // Each staging buffer is intended to hold one row-block slice of a single
    // k-tile: [row_block_rows x update_tile_cols].
    T* d_rowblock_w[2] = {nullptr, nullptr};
    T* d_rowblock_y[2] = {nullptr, nullptr};
    size_t rowblock_elems = 0;

    // Temporary workspaces for:
    //   tmp_tile = W_rb^T * A_rb_tile
    // and for the eventual apply pass.
    T* d_tmp0 = nullptr;
    T* d_tmp1 = nullptr;
    size_t tmp_elems = 0;
};

struct RowBlockPipelineConfig {
    // Width of each broadcast k-tile cut from the block-WY representation.
    // Must be a multiple of kPanelWidth once implemented.
    int update_tile_cols = 0;

    // Number of rows per streamed row-block slice of W/Y.
    // This determines staging size and GEMM granularity in the local pipeline.
    int row_block_rows = 0;

    // Number of trailing local columns processed per local GEMM tile.
    // This plays the same role as overlap_tile_cols / update_tile_cols in the
    // older implementation, but only for the local row-block pipeline.
    int trail_tile_cols = 0;
};

struct CommProfile {
    size_t bytes = 0;
};

enum class PhaseKind {
    PanelFactor = 0,
    WyBuild = 1,
    BlockWyMerge = 2,
    TailAccumulate = 3,
    TailApply = 4,
    Comm = 5,
};

struct PhaseInterval {
    PhaseKind kind = PhaseKind::PanelFactor;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

struct PhaseProfile {
    double panel_factor_ms = 0.0;
    double wy_build_ms = 0.0;
    double block_wy_merge_ms = 0.0;
    double tail_accumulate_ms = 0.0;
    double tail_apply_ms = 0.0;
    double comm_ms = 0.0;
    double tail_update_flops = 0.0;
    std::vector<PhaseInterval> intervals;
};

inline void ResetPhaseProfile(PhaseProfile* profile) {
    if (!profile) {
        return;
    }
    profile->panel_factor_ms = 0.0;
    profile->wy_build_ms = 0.0;
    profile->block_wy_merge_ms = 0.0;
    profile->tail_accumulate_ms = 0.0;
    profile->tail_apply_ms = 0.0;
    profile->comm_ms = 0.0;
    profile->tail_update_flops = 0.0;
    profile->intervals.clear();
}

inline size_t BeginPhaseInterval(PhaseProfile* profile, PhaseKind kind, cudaStream_t stream) {
    if (!profile) {
        return 0;
    }
    PhaseInterval interval{};
    interval.kind = kind;
    AssertCuda(cudaEventCreate(&interval.start), "cudaEventCreate phase_start");
    AssertCuda(cudaEventCreate(&interval.stop), "cudaEventCreate phase_stop");
    AssertCuda(cudaEventRecord(interval.start, stream), "cudaEventRecord phase_start");
    profile->intervals.push_back(interval);
    return profile->intervals.size() - 1;
}

inline void EndPhaseInterval(PhaseProfile* profile, size_t idx, cudaStream_t stream) {
    if (!profile) {
        return;
    }
    AssertCuda(cudaEventRecord(profile->intervals[idx].stop, stream), "cudaEventRecord phase_stop");
}

inline void FinalizePhaseProfile(PhaseProfile* profile) {
    if (!profile) {
        return;
    }
    for (auto& interval : profile->intervals) {
        float ms = 0.0f;
        AssertCuda(cudaEventElapsedTime(&ms, interval.start, interval.stop),
                   "cudaEventElapsedTime phase");
        switch (interval.kind) {
            case PhaseKind::PanelFactor:
                profile->panel_factor_ms += static_cast<double>(ms);
                break;
            case PhaseKind::WyBuild:
                profile->wy_build_ms += static_cast<double>(ms);
                break;
            case PhaseKind::BlockWyMerge:
                profile->block_wy_merge_ms += static_cast<double>(ms);
                break;
            case PhaseKind::TailAccumulate:
                profile->tail_accumulate_ms += static_cast<double>(ms);
                break;
            case PhaseKind::TailApply:
                profile->tail_apply_ms += static_cast<double>(ms);
                break;
            case PhaseKind::Comm:
                profile->comm_ms += static_cast<double>(ms);
                break;
        }
        AssertCuda(cudaEventDestroy(interval.start), "cudaEventDestroy phase_start");
        AssertCuda(cudaEventDestroy(interval.stop), "cudaEventDestroy phase_stop");
    }
    profile->intervals.clear();
}

template <typename T>
inline void ValidateWorkspace(const DistributedQrColBlockCyclicPipelineWorkspace<T>& ws,
                              int m,
                              int nb) {
    if (!ws.d_r_panel) {
        spdlog::error("Pipeline workspace d_r_panel is null.");
        std::exit(1);
    }

    const size_t tsqr_need = tsqr_work_elems<T>(m);
    if (tsqr_need > 0 && (!ws.d_tsqr_work_panel || ws.tsqr_work_panel_elems < tsqr_need)) {
        spdlog::error("Pipeline TSQR workspace too small (need {} elems, got {}).", tsqr_need,
                      ws.tsqr_work_panel_elems);
        std::exit(1);
    }

    const size_t panel_need = static_cast<size_t>(m) * static_cast<size_t>(kPanelWidth);
    if (!ws.d_panel_w || !ws.d_panel_y || ws.panel_elems < panel_need) {
        spdlog::error("Pipeline panel staging too small (need {} elems, got {}).", panel_need,
                      ws.panel_elems);
        std::exit(1);
    }

    const size_t block_need = static_cast<size_t>(m) * static_cast<size_t>(nb);
    if (!ws.d_block_w || !ws.d_block_y || ws.block_elems < block_need) {
        spdlog::error("Pipeline compact block-WY too small (need {} elems, got {}).", block_need,
                      ws.block_elems);
        std::exit(1);
    }

    if (!ws.d_tmp0 || !ws.d_tmp1) {
        spdlog::error("Pipeline tmp buffers are null.");
        std::exit(1);
    }
}

template <typename T>
inline void ScatterPanelToCompactBlockWy(int block_begin,
                                         int inner,
                                         int m,
                                         const T* d_panel_w,
                                         const T* d_panel_y,
                                         T* d_block_w,
                                         T* d_block_y,
                                         cudaStream_t stream) {
    const int block_rows = m - block_begin;
    const int row_offset = inner - block_begin;
    const int panel_rows = m - inner;
    const int block_col_off = inner - block_begin;

    T* dst_w = d_block_w + static_cast<size_t>(block_col_off) * static_cast<size_t>(block_rows);
    T* dst_y = d_block_y + static_cast<size_t>(block_col_off) * static_cast<size_t>(block_rows);

    if (row_offset > 0) {
        AssertCuda(cudaMemset2DAsync(dst_w, static_cast<size_t>(block_rows) * sizeof(T), 0,
                                     static_cast<size_t>(row_offset) * sizeof(T), kPanelWidth,
                                     stream),
                   "cudaMemset2DAsync zero compact block_w top");
        AssertCuda(cudaMemset2DAsync(dst_y, static_cast<size_t>(block_rows) * sizeof(T), 0,
                                     static_cast<size_t>(row_offset) * sizeof(T), kPanelWidth,
                                     stream),
                   "cudaMemset2DAsync zero compact block_y top");
    }

    AssertCuda(cudaMemcpy2DAsync(dst_w + row_offset, static_cast<size_t>(block_rows) * sizeof(T),
                                 d_panel_w, static_cast<size_t>(panel_rows) * sizeof(T),
                                 static_cast<size_t>(panel_rows) * sizeof(T), kPanelWidth,
                                 cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy2DAsync panel_w -> compact block_w");
    AssertCuda(cudaMemcpy2DAsync(dst_y + row_offset, static_cast<size_t>(block_rows) * sizeof(T),
                                 d_panel_y, static_cast<size_t>(panel_rows) * sizeof(T),
                                 static_cast<size_t>(panel_rows) * sizeof(T), kPanelWidth,
                                 cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy2DAsync panel_y -> compact block_y");
}

template <typename T>
inline void MergePanelIntoCompactBlockWy(cublasHandle_t cublas_handle,
                                         int block_begin,
                                         int inner,
                                         int m,
                                         T* d_block_w,
                                         const T* d_block_y,
                                         T* d_tmp,
                                         size_t tmp_elems) {
    if (inner <= block_begin) {
        return;
    }

    const int block_rows = m - block_begin;
    const int k_prev = inner - block_begin;
    const int block_col_off = inner - block_begin;
    const size_t need = static_cast<size_t>(k_prev) * static_cast<size_t>(kPanelWidth);
    if (tmp_elems < need) {
        spdlog::error("Pipeline compact block-WY merge tmp too small (need {} elems, got {}).",
                      need, tmp_elems);
        std::exit(1);
    }

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const T minus_one = static_cast<T>(-1);

    const T* y_prev = d_block_y;
    const T* w_prev = d_block_w;
    T* w_i = d_block_w + static_cast<size_t>(block_col_off) * static_cast<size_t>(block_rows);

    AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k_prev,
                                           kPanelWidth, block_rows, &one, y_prev, block_rows, w_i,
                                           block_rows, &zero, d_tmp, k_prev),
                 "compact block-WY tmp = Y_prev^T * W_i");
    AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, block_rows,
                                           kPanelWidth, k_prev, &minus_one, w_prev, block_rows,
                                           d_tmp, k_prev, &one, w_i, block_rows),
                 "compact block-WY W_i -= W_prev * tmp");
}

template <typename T>
inline void SaveOwnerPanelY(int block_begin,
                            int inner,
                            int panel_rows,
                            int local_panel_col,
                            T* d_Y_local,
                            int lda_local,
                            const T* d_panel_y,
                            cudaStream_t stream) {
    if (!d_Y_local) {
        return;
    }

    AssertCuda(cudaMemcpy2DAsync(d_Y_local + static_cast<size_t>(local_panel_col) * lda_local + inner,
                                 static_cast<size_t>(lda_local) * sizeof(T), d_panel_y,
                                 static_cast<size_t>(panel_rows) * sizeof(T),
                                 static_cast<size_t>(panel_rows) * sizeof(T), kPanelWidth,
                                 cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy2DAsync panel_y -> d_Y_local");

    const int zero_rows = inner - block_begin;
    if (zero_rows > 0) {
        T* z_y = d_Y_local + static_cast<size_t>(local_panel_col) * lda_local + block_begin;
        AssertCuda(cudaMemset2DAsync(z_y, static_cast<size_t>(lda_local) * sizeof(T), 0,
                                     static_cast<size_t>(zero_rows) * sizeof(T), kPanelWidth,
                                     stream),
                   "cudaMemset2DAsync zero d_Y_local top");
    }
}

template <typename T>
inline void SaveOwnerBlockW(int block_begin,
                            int inner,
                            int m,
                            int local_panel_col,
                            T* d_W_local,
                            int lda_local,
                            const T* d_block_w,
                            cudaStream_t stream) {
    if (!d_W_local) {
        return;
    }

    const int block_rows = m - block_begin;
    const int block_col_off = inner - block_begin;
    const T* src_w = d_block_w + static_cast<size_t>(block_col_off) * static_cast<size_t>(block_rows);
    AssertCuda(
        cudaMemcpy2DAsync(d_W_local + static_cast<size_t>(local_panel_col) * lda_local + block_begin,
                          static_cast<size_t>(lda_local) * sizeof(T), src_w,
                          static_cast<size_t>(block_rows) * sizeof(T),
                          static_cast<size_t>(block_rows) * sizeof(T), kPanelWidth,
                          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync compact block_w -> d_W_local");
}

template <typename T>
inline void StreamedTailUpdateScaffold(cublasHandle_t cublas_handle,
                                       ncclComm_t nccl_comm,
                                       const ColBlockCyclicPartition& part,
                                       int block_begin,
                                       int block_end,
                                       int m,
                                       int n,
                                       T* d_A_local,
                                       int lda_local,
                                       DistributedQrColBlockCyclicPipelineWorkspace<T>* ws,
                                       cudaStream_t compute_stream,
                                       cudaStream_t comm_stream,
                                       const RowBlockPipelineConfig& pipeline_cfg,
                                       CommProfile* comm_profile,
                                       PhaseProfile* phase_profile) {
    (void)cublas_handle;
    (void)nccl_comm;
    (void)part;
    (void)m;
    (void)n;
    (void)d_A_local;
    (void)lda_local;
    (void)ws;
    (void)compute_stream;
    (void)comm_stream;
    (void)pipeline_cfg;
    (void)comm_profile;
    (void)phase_profile;

    if (block_end >= n) {
        return;
    }

    // This is the key piece that still needs to be implemented for the new
    // pipeline design. The intended execution order is:
    //
    //   for each compact block-WY k-tile:
    //     1. owner broadcasts the k-tile,
    //     2. each rank streams the received tile as row-block slices,
    //     3. pass 1 accumulates tmp_tile = sum_rb(W_rb^T * A_rb_tile),
    //     4. pass 2 applies A_rb_tile -= Y_rb * tmp_tile.
    //
    // We fail explicitly here instead of silently reusing the old full-tile
    // update path, because the whole point of this file is to make the new
    // local row-block pipeline visible and hard to accidentally bypass.
    spdlog::error(
        "distributed_blocked_qr_col_blockcyclic_pipeline tail-update scaffold hit TODO at "
        "block [{}, {}). Implement row-block streamed trailing update before using this path.",
        block_begin, block_end);
    std::exit(1);
}

template <typename T>
void distributed_blocked_qr_factorize_col_blockcyclic_pipeline(
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
    DistributedQrColBlockCyclicPipelineWorkspace<T>* ws,
    cudaStream_t compute_stream,
    cudaStream_t comm_stream,
    const RowBlockPipelineConfig& pipeline_cfg = {},
    CommProfile* comm_profile = nullptr,
    PhaseProfile* phase_profile = nullptr) {
    static_assert(kSupportedQrType<T>,
                  "distributed_qr_col_blockcyclic_pipeline only supports float and double.");

    if (!ws) {
        spdlog::error("Pipeline factorization got null workspace.");
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

    ValidateWorkspace(*ws, m, nb);
    if (comm_profile) {
        comm_profile->bytes = 0;
    }
    ResetPhaseProfile(phase_profile);

    AssertCublas(cublasSetStream(cublas_handle, compute_stream), "cublasSetStream(compute_stream)");

    for (int block_begin = 0; block_begin < n; block_begin += nb) {
        const int block_end = std::min(block_begin + nb, n);
        const int kb = block_end - block_begin;
        const int block_rows = m - block_begin;
        const int block_owner = OwnerOfPanel(block_begin, part);

        // Compact block-WY is the canonical representation in this scaffold, so
        // the owner clears it once per outer block and then appends each panel
        // directly into the compact layout.
        if (part.rank == block_owner) {
            AssertCuda(cudaMemsetAsync(ws->d_block_w, 0,
                                       static_cast<size_t>(block_rows) * static_cast<size_t>(kb) *
                                           sizeof(T),
                                       compute_stream),
                       "cudaMemsetAsync compact block_w");
            AssertCuda(cudaMemsetAsync(ws->d_block_y, 0,
                                       static_cast<size_t>(block_rows) * static_cast<size_t>(kb) *
                                           sizeof(T),
                                       compute_stream),
                       "cudaMemsetAsync compact block_y");
        }

        for (int inner = block_begin; inner < block_end; inner += kPanelWidth) {
            const int owner = OwnerOfPanel(inner, part);
            if (owner != block_owner) {
                spdlog::error("Outer block [{}, {}) crossed owner boundary at panel {}.", block_begin,
                              block_end, inner);
                std::exit(1);
            }

            if (part.rank != owner) {
                continue;
            }

            const int local_panel_col = LocalColOffset(part, inner);
            if (local_panel_col < 0) {
                spdlog::error("Owner rank {} missing panel col {} in local layout.", owner, inner);
                std::exit(1);
            }

            // The owner still performs the same panel-local work as the existing
            // implementation:
            //   TSQR factorization,
            //   WY generation,
            //   write-back of explicit R to the panel in A.
            const int panel_rows = m - inner;
            T* panel_A = d_A_local + static_cast<size_t>(local_panel_col) * lda_local;
            T* panel_A_sub = panel_A + inner;

            const size_t panel_factor_idx =
                BeginPhaseInterval(phase_profile, PhaseKind::PanelFactor, compute_stream);
            tsqr<T>(cublas_handle, panel_rows, panel_A_sub, lda_local, ws->d_r_panel, kPanelWidth,
                    ws->d_tsqr_work_panel, ws->tsqr_work_panel_elems, compute_stream);
            EndPhaseInterval(phase_profile, panel_factor_idx, compute_stream);

            const size_t wy_build_idx =
                BeginPhaseInterval(phase_profile, PhaseKind::WyBuild, compute_stream);
            generate_wy<T>(panel_rows, kPanelWidth, panel_A_sub, lda_local, ws->d_panel_y,
                           panel_rows, ws->d_panel_w, panel_rows, compute_stream);
            EndPhaseInterval(phase_profile, wy_build_idx, compute_stream);

            const dim3 block_dim(16, 16);
            const dim3 grid_dim((kPanelWidth + block_dim.x - 1) / block_dim.x,
                                (kPanelWidth + block_dim.y - 1) / block_dim.y);
            write_upper_r_to_panel_kernel<<<grid_dim, block_dim, 0, compute_stream>>>(
                inner, ws->d_r_panel, kPanelWidth, panel_A, lda_local);
            AssertCuda(cudaGetLastError(), "write_upper_r_to_panel_kernel launch");

            SaveOwnerPanelY(block_begin, inner, panel_rows, local_panel_col, d_Y_local, lda_local,
                            ws->d_panel_y, compute_stream);

            // The new pipeline wants owner-side compact block-WY to be the only
            // block representation. We therefore append the panel directly into
            // the compact storage and merge W_i in-place there.
            ScatterPanelToCompactBlockWy(block_begin, inner, m, ws->d_panel_w, ws->d_panel_y,
                                         ws->d_block_w, ws->d_block_y, compute_stream);

            const size_t merge_idx =
                BeginPhaseInterval(phase_profile, PhaseKind::BlockWyMerge, compute_stream);
            MergePanelIntoCompactBlockWy(cublas_handle, block_begin, inner, m, ws->d_block_w,
                                         ws->d_block_y, ws->d_tmp0, ws->tmp_elems);
            EndPhaseInterval(phase_profile, merge_idx, compute_stream);

            SaveOwnerBlockW(block_begin, inner, m, local_panel_col, d_W_local, lda_local,
                            ws->d_block_w, compute_stream);

            // The piece intentionally omitted from this scaffold is the in-block
            // propagation that would make panel (inner + kPanelWidth) numerically
            // ready. Rather than silently computing the next panel on stale data,
            // we stop here as soon as the block needs a second panel.
            if (inner + kPanelWidth < block_end) {
                spdlog::error(
                    "distributed_blocked_qr_col_blockcyclic_pipeline is currently a scaffold. "
                    "In-block panel propagation/update is intentionally not migrated yet, so "
                    "outer blocks wider than one panel are not executable.");
                std::exit(1);
            }
        }

        // Once the owner has constructed compact block-WY, the new pipeline is
        // supposed to take over here. Tail update becomes the dominant stage and
        // is where the new row-block local streaming logic will live.
        StreamedTailUpdateScaffold(cublas_handle, nccl_comm, part, block_begin, block_end, m, n,
                                   d_A_local, lda_local, ws, compute_stream, comm_stream,
                                   pipeline_cfg, comm_profile, phase_profile);
    }
}

}  // namespace distributed_qr_col_blockcyclic_pipeline
