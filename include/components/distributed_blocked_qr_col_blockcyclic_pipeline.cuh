#pragma once
// Column-partitioned distributed blocked QR for a block-cyclic column layout.
//
// This variant keeps compact owner-side block-WY storage and performs the
// trailing update with a streamed row-block pipeline:
//   1. the owner factorizes panels and builds compact block-WY,
//   2. the owner repacks block-WY into row-block-major caches,
//   3. row blocks are broadcast one slab at a time and cached locally,
//   4. tail GEMMs consume the cached row blocks with overlap between row-block
//      communication and the accumulate pass.

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

    // Row-block-major caches used by the tail-update pipeline.
    // Each cache stores the full block-WY, but packed by row block:
    //   [row_block_0_rows x kb][row_block_1_rows x kb]...
    // Each row-block slab is itself column-major with lda=row_block_rows_i.
    T* d_block_w_rowmajor = nullptr;
    T* d_block_y_rowmajor = nullptr;
    size_t block_rowmajor_elems = 0;

    // Packed row-block communication cache.
    // Each row block is laid out as [W_slab][Y_slab] in a single contiguous
    // buffer so the row-block pipeline can use one broadcast per slab.
    T* d_rowblock_wy_packed = nullptr;
    size_t rowblock_wy_packed_elems = 0;

    // Temporary workspaces for:
    //   tmp_tile = W^T * A_tile
    // and an auxiliary scratch buffer for future local kernels.
    T* d_tmp0 = nullptr;
    T* d_tmp1 = nullptr;
    size_t tmp_elems = 0;
};

struct RowBlockPipelineConfig {
    enum class TailMode {
        Baseline = 0,
        Overlap = 1,
        Chained = 2,
    };

    // Reserved compatibility knob from the previous k-tile version.
    // The row-block communication path does not use it to partition traffic.
    int update_tile_cols = 0;

    // Number of rows per communicated/cached row block of W/Y.
    int row_block_rows = 0;

    // Number of trailing local columns processed per local GEMM tile.
    int trail_tile_cols = 0;

    TailMode tail_mode = TailMode::Baseline;
    bool skip_tail_update = false;
};

struct CommProfile {
    size_t bytes = 0;
};

enum class PhaseKind {
    PanelFactor = 0,
    WyBuild = 1,
    BlockWyMerge = 2,
    InnerBlockUpdate = 3,
    RowBlockPack = 4,
    RowBlockUnpack = 5,
    TailAccWait = 6,
    TailAccGemm = 7,
    TailApplyGemm = 8,
    Comm = 9,
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
    double inner_block_update_ms = 0.0;
    double rowblock_pack_ms = 0.0;
    double rowblock_unpack_ms = 0.0;
    double tail_acc_wait_ms = 0.0;
    double tail_acc_gemm_ms = 0.0;
    double tail_apply_gemm_ms = 0.0;
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
    profile->inner_block_update_ms = 0.0;
    profile->rowblock_pack_ms = 0.0;
    profile->rowblock_unpack_ms = 0.0;
    profile->tail_acc_wait_ms = 0.0;
    profile->tail_acc_gemm_ms = 0.0;
    profile->tail_apply_gemm_ms = 0.0;
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
            case PhaseKind::InnerBlockUpdate:
                profile->inner_block_update_ms += static_cast<double>(ms);
                break;
            case PhaseKind::RowBlockPack:
                profile->rowblock_pack_ms += static_cast<double>(ms);
                break;
            case PhaseKind::RowBlockUnpack:
                profile->rowblock_unpack_ms += static_cast<double>(ms);
                break;
            case PhaseKind::TailAccWait:
                profile->tail_acc_wait_ms += static_cast<double>(ms);
                break;
            case PhaseKind::TailAccGemm:
                profile->tail_acc_gemm_ms += static_cast<double>(ms);
                break;
            case PhaseKind::TailApplyGemm:
                profile->tail_apply_gemm_ms += static_cast<double>(ms);
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
    if (!ws.d_rowblock_wy_packed || ws.rowblock_wy_packed_elems < 2 * block_need) {
        spdlog::error(
            "Pipeline packed row-block communication cache too small (need {} elems, got {}).",
            2 * block_need, ws.rowblock_wy_packed_elems);
        std::exit(1);
    }

    if (!ws.d_tmp0 || !ws.d_tmp1) {
        spdlog::error("Pipeline tmp buffers are null.");
        std::exit(1);
    }
}

inline RowBlockPipelineConfig NormalizePipelineConfig(const RowBlockPipelineConfig& pipeline_cfg,
                                                      int nb) {
    RowBlockPipelineConfig cfg = pipeline_cfg;
    if (cfg.update_tile_cols <= 0) {
        cfg.update_tile_cols = nb;
    }
    if (cfg.row_block_rows <= 0) {
        cfg.row_block_rows = 1024;
    }
    if (cfg.trail_tile_cols <= 0) {
        cfg.trail_tile_cols = nb;
    }
    return cfg;
}

inline const char* TailModeString(RowBlockPipelineConfig::TailMode mode) {
    switch (mode) {
        case RowBlockPipelineConfig::TailMode::Baseline:
            return "baseline";
        case RowBlockPipelineConfig::TailMode::Overlap:
            return "overlap";
        case RowBlockPipelineConfig::TailMode::Chained:
            return "chained";
    }
    return "unknown";
}

template <typename T>
inline void ValidatePipelineConfig(const DistributedQrColBlockCyclicPipelineWorkspace<T>& ws,
                                   const RowBlockPipelineConfig& cfg,
                                   int nb) {
    if (cfg.update_tile_cols <= 0 || cfg.update_tile_cols % kPanelWidth != 0) {
        spdlog::error("Pipeline update_tile_cols must be positive and a multiple of {} (got {}).",
                      kPanelWidth, cfg.update_tile_cols);
        std::exit(1);
    }
    if (cfg.row_block_rows <= 0) {
        spdlog::error("Pipeline row_block_rows must be positive (got {}).", cfg.row_block_rows);
        std::exit(1);
    }
    if (cfg.trail_tile_cols <= 0) {
        spdlog::error("Pipeline trail_tile_cols must be positive (got {}).", cfg.trail_tile_cols);
        std::exit(1);
    }
    if (cfg.tail_mode != RowBlockPipelineConfig::TailMode::Baseline &&
        cfg.tail_mode != RowBlockPipelineConfig::TailMode::Overlap &&
        cfg.tail_mode != RowBlockPipelineConfig::TailMode::Chained) {
        spdlog::error("Pipeline tail_mode is invalid.");
        std::exit(1);
    }

    const size_t rowmajor_need = static_cast<size_t>(nb);
    if (!ws.d_block_w_rowmajor || !ws.d_block_y_rowmajor ||
        ws.block_rowmajor_elems < rowmajor_need) {
        spdlog::error(
            "Pipeline row-block-major caches are null or empty (need capacity for at least {} "
            "elems per cache, got {}).",
            rowmajor_need, ws.block_rowmajor_elems);
        std::exit(1);
    }
    if (!ws.d_rowblock_wy_packed || ws.rowblock_wy_packed_elems < 2 * rowmajor_need) {
        spdlog::error(
            "Pipeline packed row-block communication cache is null or empty (need capacity for "
            "at least {} elems, got {}).",
            2 * rowmajor_need, ws.rowblock_wy_packed_elems);
        std::exit(1);
    }

    const size_t tmp_need = static_cast<size_t>(nb) * static_cast<size_t>(cfg.trail_tile_cols);
    if (ws.tmp_elems < tmp_need) {
        spdlog::error("Pipeline tmp too small (need {} elems, got {}).", tmp_need, ws.tmp_elems);
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
        AssertCuda(
            cudaMemset2DAsync(dst_w, static_cast<size_t>(block_rows) * sizeof(T), 0,
                              static_cast<size_t>(row_offset) * sizeof(T), kPanelWidth, stream),
            "cudaMemset2DAsync zero compact block_w top");
        AssertCuda(
            cudaMemset2DAsync(dst_y, static_cast<size_t>(block_rows) * sizeof(T), 0,
                              static_cast<size_t>(row_offset) * sizeof(T), kPanelWidth, stream),
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

    AssertCuda(
        cudaMemcpy2DAsync(d_Y_local + static_cast<size_t>(local_panel_col) * lda_local + inner,
                          static_cast<size_t>(lda_local) * sizeof(T), d_panel_y,
                          static_cast<size_t>(panel_rows) * sizeof(T),
                          static_cast<size_t>(panel_rows) * sizeof(T), kPanelWidth,
                          cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync panel_y -> d_Y_local");

    const int zero_rows = inner - block_begin;
    if (zero_rows > 0) {
        T* z_y = d_Y_local + static_cast<size_t>(local_panel_col) * lda_local + block_begin;
        AssertCuda(
            cudaMemset2DAsync(z_y, static_cast<size_t>(lda_local) * sizeof(T), 0,
                              static_cast<size_t>(zero_rows) * sizeof(T), kPanelWidth, stream),
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
    const T* src_w =
        d_block_w + static_cast<size_t>(block_col_off) * static_cast<size_t>(block_rows);
    AssertCuda(cudaMemcpy2DAsync(
                   d_W_local + static_cast<size_t>(local_panel_col) * lda_local + block_begin,
                   static_cast<size_t>(lda_local) * sizeof(T), src_w,
                   static_cast<size_t>(block_rows) * sizeof(T),
                   static_cast<size_t>(block_rows) * sizeof(T), kPanelWidth,
                   cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy2DAsync compact block_w -> d_W_local");
}

template <typename T>
inline void PanelUpdateOneShot(cublasHandle_t cublas_handle,
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
        spdlog::error("Pipeline panel update work too small (need {} elems, got {}).", need,
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
                 "Pipeline in-block work = W^T * A");
    AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_rows,
                                           cols_local, kPanelWidth, &minus_one, Y, lda_wy, work,
                                           kPanelWidth, &one, a_sub, lda_local),
                 "Pipeline in-block A -= Y * work");
}

template <typename T>
inline void ApplyPanelToTrailingSegment(cublasHandle_t cublas_handle,
                                        int inner,
                                        int panel_rows,
                                        int cols_local,
                                        const T* d_panel_w,
                                        const T* d_panel_y,
                                        T* a_trail,
                                        int lda_local,
                                        T* d_tmp,
                                        size_t tmp_elems) {
    if (cols_local <= 0) {
        return;
    }
    const int max_chunk_cols =
        std::max(1, static_cast<int>(tmp_elems / static_cast<size_t>(kPanelWidth)));
    for (int col_off = 0; col_off < cols_local; col_off += max_chunk_cols) {
        const int chunk_cols = std::min(max_chunk_cols, cols_local - col_off);
        PanelUpdateOneShot(cublas_handle, inner, panel_rows, chunk_cols, d_panel_w, d_panel_y,
                           panel_rows, a_trail + static_cast<size_t>(col_off) * lda_local,
                           lda_local, d_tmp, tmp_elems);
    }
}

inline int RowBlockCount(int block_rows, int row_block_rows) {
    return (block_rows + row_block_rows - 1) / row_block_rows;
}

inline int RowBlockRowsAt(int block_rows, int row_block_rows, int rb_idx) {
    const int row_begin = rb_idx * row_block_rows;
    return std::min(row_block_rows, block_rows - row_begin);
}

inline size_t RowBlockOffsetElems(int row_block_rows, int kb, int rb_idx) {
    return static_cast<size_t>(rb_idx) * static_cast<size_t>(row_block_rows) *
           static_cast<size_t>(kb);
}

inline size_t RowBlockPackedOffsetElems(int row_block_rows, int kb, int rb_idx) {
    return 2 * RowBlockOffsetElems(row_block_rows, kb, rb_idx);
}

template <typename T>
inline void PackCompactRowBlockToRowMajorCache(int rb_idx,
                                               int block_rows,
                                               int kb,
                                               int row_block_rows,
                                               const T* d_block_w,
                                               const T* d_block_y,
                                               T* d_block_w_rowmajor,
                                               T* d_block_y_rowmajor,
                                               cudaStream_t stream) {
    const int row_begin = rb_idx * row_block_rows;
    const int row_count = RowBlockRowsAt(block_rows, row_block_rows, rb_idx);
    const size_t dst_off = RowBlockOffsetElems(row_block_rows, kb, rb_idx);
    T* dst_w = d_block_w_rowmajor + dst_off;
    T* dst_y = d_block_y_rowmajor + dst_off;

    AssertCuda(cudaMemcpy2DAsync(dst_w, static_cast<size_t>(row_count) * sizeof(T),
                                 d_block_w + row_begin, static_cast<size_t>(block_rows) * sizeof(T),
                                 static_cast<size_t>(row_count) * sizeof(T), kb,
                                 cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy2DAsync compact block_w -> rowmajor cache");
    AssertCuda(cudaMemcpy2DAsync(dst_y, static_cast<size_t>(row_count) * sizeof(T),
                                 d_block_y + row_begin, static_cast<size_t>(block_rows) * sizeof(T),
                                 static_cast<size_t>(row_count) * sizeof(T), kb,
                                 cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy2DAsync compact block_y -> rowmajor cache");
}

template <typename T>
inline void PackCompactWRowBlockToRowMajorCache(int rb_idx,
                                                int block_rows,
                                                int kb,
                                                int row_block_rows,
                                                const T* d_block_w,
                                                T* d_block_w_rowmajor,
                                                cudaStream_t stream) {
    const int row_begin = rb_idx * row_block_rows;
    const int row_count = RowBlockRowsAt(block_rows, row_block_rows, rb_idx);
    const size_t dst_off = RowBlockOffsetElems(row_block_rows, kb, rb_idx);
    T* dst_w = d_block_w_rowmajor + dst_off;

    AssertCuda(cudaMemcpy2DAsync(dst_w, static_cast<size_t>(row_count) * sizeof(T),
                                 d_block_w + row_begin, static_cast<size_t>(block_rows) * sizeof(T),
                                 static_cast<size_t>(row_count) * sizeof(T), kb,
                                 cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy2DAsync compact block_w -> rowmajor cache");
}

template <typename T>
inline void PackCompactRowBlockToPackedBroadcastCache(int rb_idx,
                                                      int block_rows,
                                                      int kb,
                                                      int row_block_rows,
                                                      const T* d_block_w,
                                                      const T* d_block_y,
                                                      T* d_rowblock_wy_packed,
                                                      cudaStream_t stream) {
    const int row_begin = rb_idx * row_block_rows;
    const int row_count = RowBlockRowsAt(block_rows, row_block_rows, rb_idx);
    const size_t elems = static_cast<size_t>(row_count) * static_cast<size_t>(kb);
    const size_t packed_off = RowBlockPackedOffsetElems(row_block_rows, kb, rb_idx);
    T* dst_w = d_rowblock_wy_packed + packed_off;
    T* dst_y = dst_w + elems;

    AssertCuda(cudaMemcpy2DAsync(dst_w, static_cast<size_t>(row_count) * sizeof(T),
                                 d_block_w + row_begin, static_cast<size_t>(block_rows) * sizeof(T),
                                 static_cast<size_t>(row_count) * sizeof(T), kb,
                                 cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy2DAsync compact block_w -> packed rowblock cache");
    AssertCuda(cudaMemcpy2DAsync(dst_y, static_cast<size_t>(row_count) * sizeof(T),
                                 d_block_y + row_begin, static_cast<size_t>(block_rows) * sizeof(T),
                                 static_cast<size_t>(row_count) * sizeof(T), kb,
                                 cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpy2DAsync compact block_y -> packed rowblock cache");
}

template <typename T>
inline void UnpackPackedRowBlockToRowMajorCache(int rb_idx,
                                                int block_rows,
                                                int kb,
                                                int row_block_rows,
                                                const T* d_rowblock_wy_packed,
                                                T* d_block_w_rowmajor,
                                                T* d_block_y_rowmajor,
                                                cudaStream_t stream) {
    const int row_count = RowBlockRowsAt(block_rows, row_block_rows, rb_idx);
    const size_t elems = static_cast<size_t>(row_count) * static_cast<size_t>(kb);
    const size_t packed_off = RowBlockPackedOffsetElems(row_block_rows, kb, rb_idx);
    const size_t rowmajor_off = RowBlockOffsetElems(row_block_rows, kb, rb_idx);
    const T* src_w = d_rowblock_wy_packed + packed_off;
    const T* src_y = src_w + elems;
    T* dst_w = d_block_w_rowmajor + rowmajor_off;
    T* dst_y = d_block_y_rowmajor + rowmajor_off;

    AssertCuda(cudaMemcpyAsync(dst_w, src_w, elems * sizeof(T), cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync packed rowblock W -> rowmajor cache");
    AssertCuda(cudaMemcpyAsync(dst_y, src_y, elems * sizeof(T), cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync packed rowblock Y -> rowmajor cache");
}

template <typename T>
struct CublasAxpyTraits;

template <>
struct CublasAxpyTraits<float> {
    static cublasStatus_t Axpy(cublasHandle_t handle,
                               int n,
                               const float* alpha,
                               const float* x,
                               int incx,
                               float* y,
                               int incy) {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    }
};

template <>
struct CublasAxpyTraits<double> {
    static cublasStatus_t Axpy(cublasHandle_t handle,
                               int n,
                               const double* alpha,
                               const double* x,
                               int incx,
                               double* y,
                               int incy) {
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }
};

template <typename T>
inline void TailAccumulateColTileFromRowBlockCache(
    cublasHandle_t tail_cublas_handle,
    const T* d_block_w_rowmajor,
    int row_offset,
    int block_rows,
    int kb,
    int row_block_rows,
    const T* a_tile,
    int lda_local,
    int cols_tile,
    T* d_tmp,
    size_t tmp_elems,
    cudaStream_t tail_update_stream,
    bool wait_for_each_rowblock,
    const std::vector<cudaEvent_t>& rowblock_comm_done,
    PhaseProfile* phase_profile) {
    if (cols_tile <= 0) {
        return;
    }

    const size_t need = static_cast<size_t>(kb) * static_cast<size_t>(cols_tile);
    if (tmp_elems < need) {
        spdlog::error("Pipeline tail tmp too small (need {} elems, got {}).", need, tmp_elems);
        std::exit(1);
    }

    const int row_block_count = RowBlockCount(block_rows, row_block_rows);
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    AssertCuda(cudaMemsetAsync(d_tmp, 0, need * sizeof(T), tail_update_stream),
               "cudaMemsetAsync tail accumulate tmp");
    for (int rb_idx = 0; rb_idx < row_block_count; ++rb_idx) {
        const int row_begin = rb_idx * row_block_rows;
        const int row_count = RowBlockRowsAt(block_rows, row_block_rows, rb_idx);
        const size_t off = RowBlockOffsetElems(row_block_rows, kb, rb_idx);
        if (wait_for_each_rowblock) {
            const size_t tail_wait_idx =
                BeginPhaseInterval(phase_profile, PhaseKind::TailAccWait, tail_update_stream);
            AssertCuda(cudaStreamWaitEvent(tail_update_stream, rowblock_comm_done[rb_idx], 0),
                       "cudaStreamWaitEvent tail_update_stream <- rowblock_comm_done");
            EndPhaseInterval(phase_profile, tail_wait_idx, tail_update_stream);
        }
        const T* w_rb = d_block_w_rowmajor + off;
        const T* a_rb = a_tile + row_offset + row_begin;
        const T* beta = (rb_idx == 0) ? &zero : &one;
        const size_t tail_acc_gemm_idx =
            BeginPhaseInterval(phase_profile, PhaseKind::TailAccGemm, tail_update_stream);
        AssertCublas(CublasGemmTraits<T>::Gemm(tail_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kb,
                                               cols_tile, row_count, &one, w_rb, row_count, a_rb,
                                               lda_local, beta, d_tmp, kb),
                     "Pipeline tail accumulate tmp += W_rb^T * A_rb");
        EndPhaseInterval(phase_profile, tail_acc_gemm_idx, tail_update_stream);
    }
}

template <typename T>
inline void TailAccumulateSingleRowBlockFromRowBlockCache(
    cublasHandle_t tail_cublas_handle,
    const T* d_block_w_rowmajor,
    int row_offset,
    int block_rows,
    int kb,
    int row_block_rows,
    int source_rb_idx,
    const T* a_tile,
    int lda_local,
    int cols_tile,
    T* d_tmp_partial,
    size_t tmp_elems,
    cudaStream_t tail_update_stream,
    const std::vector<cudaEvent_t>& rowblock_comm_done,
    PhaseProfile* phase_profile) {
    if (cols_tile <= 0) {
        return;
    }

    const size_t need = static_cast<size_t>(kb) * static_cast<size_t>(cols_tile);
    if (tmp_elems < need) {
        spdlog::error("Pipeline tail tmp too small (need {} elems, got {}).", need, tmp_elems);
        std::exit(1);
    }

    const int row_begin = source_rb_idx * row_block_rows;
    const int row_count = RowBlockRowsAt(block_rows, row_block_rows, source_rb_idx);
    const size_t off = RowBlockOffsetElems(row_block_rows, kb, source_rb_idx);
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);

    const size_t tail_wait_idx =
        BeginPhaseInterval(phase_profile, PhaseKind::TailAccWait, tail_update_stream);
    AssertCuda(cudaStreamWaitEvent(tail_update_stream, rowblock_comm_done[source_rb_idx], 0),
               "cudaStreamWaitEvent tail_update_stream <- rowblock_comm_done[source]");
    EndPhaseInterval(phase_profile, tail_wait_idx, tail_update_stream);

    const T* w_rb = d_block_w_rowmajor + off;
    const T* a_rb = a_tile + row_offset + row_begin;
    const size_t tail_acc_gemm_idx =
        BeginPhaseInterval(phase_profile, PhaseKind::TailAccGemm, tail_update_stream);
    AssertCublas(CublasGemmTraits<T>::Gemm(tail_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kb,
                                           cols_tile, row_count, &one, w_rb, row_count, a_rb,
                                           lda_local, &zero, d_tmp_partial, kb),
                 "Pipeline chained tail tmp_partial = W_rb^T * A_rb");
    EndPhaseInterval(phase_profile, tail_acc_gemm_idx, tail_update_stream);
}

template <typename T>
inline void TailApplySingleRowBlockFromRowBlockCache(cublasHandle_t tail_cublas_handle,
                                                     const T* d_block_y_rowmajor,
                                                     int row_offset,
                                                     int block_rows,
                                                     int kb,
                                                     int row_block_rows,
                                                     int target_rb_idx,
                                                     T* a_tile,
                                                     int lda_local,
                                                     int cols_tile,
                                                     const T* d_tmp,
                                                     cudaStream_t tail_update_stream,
                                                     PhaseProfile* phase_profile) {
    if (cols_tile <= 0) {
        return;
    }

    const int row_begin = target_rb_idx * row_block_rows;
    const int row_count = RowBlockRowsAt(block_rows, row_block_rows, target_rb_idx);
    const size_t off = RowBlockOffsetElems(row_block_rows, kb, target_rb_idx);
    const T one = static_cast<T>(1);
    const T minus_one = static_cast<T>(-1);
    const T* y_rb = d_block_y_rowmajor + off;
    T* a_rb = a_tile + row_offset + row_begin;

    const size_t tail_apply_gemm_idx =
        BeginPhaseInterval(phase_profile, PhaseKind::TailApplyGemm, tail_update_stream);
    AssertCublas(CublasGemmTraits<T>::Gemm(tail_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, row_count,
                                           cols_tile, kb, &minus_one, y_rb, row_count, d_tmp, kb,
                                           &one, a_rb, lda_local),
                 "Pipeline chained tail apply A_rb -= Y_rb * tmp");
    EndPhaseInterval(phase_profile, tail_apply_gemm_idx, tail_update_stream);
}

template <typename T>
inline void TailApplyColTileFromRowBlockCache(cublasHandle_t tail_cublas_handle,
                                              const T* d_block_y_rowmajor,
                                              int row_offset,
                                              int block_rows,
                                              int kb,
                                              int row_block_rows,
                                              T* a_tile,
                                              int lda_local,
                                              int cols_tile,
                                              const T* d_tmp,
                                              cudaStream_t tail_update_stream,
                                              PhaseProfile* phase_profile) {
    if (cols_tile <= 0) {
        return;
    }

    const int row_block_count = RowBlockCount(block_rows, row_block_rows);
    const T one = static_cast<T>(1);
    const T minus_one = static_cast<T>(-1);

    for (int rb_idx = 0; rb_idx < row_block_count; ++rb_idx) {
        const int row_begin = rb_idx * row_block_rows;
        const int row_count = RowBlockRowsAt(block_rows, row_block_rows, rb_idx);
        const size_t off = RowBlockOffsetElems(row_block_rows, kb, rb_idx);
        const T* y_rb = d_block_y_rowmajor + off;
        T* a_rb = a_tile + row_offset + row_begin;
        const size_t tail_apply_gemm_idx =
            BeginPhaseInterval(phase_profile, PhaseKind::TailApplyGemm, tail_update_stream);
        AssertCublas(CublasGemmTraits<T>::Gemm(tail_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               row_count, cols_tile, kb, &minus_one, y_rb,
                                               row_count, d_tmp, kb, &one, a_rb, lda_local),
                     "Pipeline tail apply A_rb -= Y_rb * tmp");
        EndPhaseInterval(phase_profile, tail_apply_gemm_idx, tail_update_stream);
    }
}

template <typename T>
inline void TailApplyCompactBlockToTile(cublasHandle_t tail_cublas_handle,
                                        const T* d_block_y,
                                        int row_offset,
                                        int block_rows,
                                        int kb,
                                        T* a_tile,
                                        int lda_local,
                                        int cols_tile,
                                        const T* d_tmp,
                                        cudaStream_t tail_update_stream,
                                        PhaseProfile* phase_profile) {
    if (cols_tile <= 0) {
        return;
    }

    const T one = static_cast<T>(1);
    const T minus_one = static_cast<T>(-1);
    T* a_sub = a_tile + row_offset;
    const size_t tail_apply_gemm_idx =
        BeginPhaseInterval(phase_profile, PhaseKind::TailApplyGemm, tail_update_stream);
    AssertCublas(CublasGemmTraits<T>::Gemm(tail_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           block_rows, cols_tile, kb, &minus_one, d_block_y,
                                           block_rows, d_tmp, kb, &one, a_sub, lda_local),
                 "Pipeline compact tail apply A -= Y * tmp");
    EndPhaseInterval(phase_profile, tail_apply_gemm_idx, tail_update_stream);
}

template <typename T>
inline void TailUpdateColTileFromRowBlockCache(cublasHandle_t tail_cublas_handle,
                                               const T* d_block_w_rowmajor,
                                               const T* d_block_y_rowmajor,
                                               int row_offset,
                                               int block_rows,
                                               int kb,
                                               int row_block_rows,
                                               T* a_tile,
                                               int lda_local,
                                               int cols_tile,
                                               T* d_tmp,
                                               size_t tmp_elems,
                                               cudaStream_t tail_update_stream,
                                               bool wait_for_each_rowblock,
                                               const std::vector<cudaEvent_t>& rowblock_comm_done,
                                               PhaseProfile* phase_profile) {
    TailAccumulateColTileFromRowBlockCache(
        tail_cublas_handle, d_block_w_rowmajor, row_offset, block_rows, kb, row_block_rows, a_tile,
        lda_local, cols_tile, d_tmp, tmp_elems, tail_update_stream, wait_for_each_rowblock,
        rowblock_comm_done, phase_profile);
    TailApplyColTileFromRowBlockCache(tail_cublas_handle, d_block_y_rowmajor, row_offset,
                                      block_rows, kb, row_block_rows, a_tile, lda_local, cols_tile,
                                      d_tmp, tail_update_stream, phase_profile);
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
    if (block_end >= n || pipeline_cfg.skip_tail_update) {
        return;
    }

    (void)cublas_handle;
    const int block_rows = m - block_begin;
    const int kb = block_end - block_begin;
    const int block_owner = OwnerOfPanel(block_begin, part);
    const int row_block_rows = std::max(1, pipeline_cfg.row_block_rows);
    const int row_block_count = RowBlockCount(block_rows, row_block_rows);
    const bool self_has_tail = RankHasColsAfter(part, part.rank, block_end);
    const bool overlap_tail = pipeline_cfg.tail_mode == RowBlockPipelineConfig::TailMode::Overlap;
    const bool chained_tail = pipeline_cfg.tail_mode == RowBlockPipelineConfig::TailMode::Chained;

    const size_t rowmajor_need = static_cast<size_t>(block_rows) * static_cast<size_t>(kb);
    if (ws->block_rowmajor_elems < rowmajor_need) {
        spdlog::error("Pipeline row-block-major cache too small (need {} elems, got {}).",
                      rowmajor_need, ws->block_rowmajor_elems);
        std::exit(1);
    }
    if (ws->rowblock_wy_packed_elems < 2 * rowmajor_need) {
        spdlog::error(
            "Pipeline packed row-block communication cache too small (need {} elems, got {}).",
            2 * rowmajor_need, ws->rowblock_wy_packed_elems);
        std::exit(1);
    }

    int block_receivers = 0;
    if (part.world_size > 1) {
        for (int r = 0; r < part.world_size; ++r) {
            if (r == block_owner) {
                continue;
            }
            if (!RankHasColsAfter(part, r, block_end)) {
                continue;
            }
            ++block_receivers;
        }
    }

    struct PersistentPipelineEvents {
        bool initialized = false;
        cudaStream_t rowblock_pack_stream = nullptr;
        cudaStream_t tail_acc_stream = nullptr;
        cudaStream_t tail_apply_stream = nullptr;
        cublasHandle_t tail_acc_cublas_handle = nullptr;
        cublasHandle_t tail_apply_cublas_handle = nullptr;
        cudaEvent_t block_wy_ready = nullptr;
        cudaEvent_t tail_update_done = nullptr;
        cudaEvent_t compact_y_comm_done = nullptr;
        cudaEvent_t tail_acc_done[2] = {nullptr, nullptr};
        cudaEvent_t tail_apply_done[2] = {nullptr, nullptr};
        std::vector<cudaEvent_t> rowblock_ready;
        std::vector<cudaEvent_t> rowblock_comm_done;
    };
    static PersistentPipelineEvents events;
    if (!events.initialized) {
        AssertCuda(cudaStreamCreateWithFlags(&events.rowblock_pack_stream, cudaStreamNonBlocking),
                   "cudaStreamCreate rowblock_pack_stream");
        AssertCuda(cudaStreamCreateWithFlags(&events.tail_acc_stream, cudaStreamNonBlocking),
                   "cudaStreamCreate tail_acc_stream");
        AssertCuda(cudaStreamCreateWithFlags(&events.tail_apply_stream, cudaStreamNonBlocking),
                   "cudaStreamCreate tail_apply_stream");
        AssertCublas(cublasCreate(&events.tail_acc_cublas_handle),
                     "cublasCreate tail_acc_cublas_handle");
        AssertCublas(cublasSetStream(events.tail_acc_cublas_handle, events.tail_acc_stream),
                     "cublasSetStream(tail_acc_stream)");
        AssertCublas(cublasCreate(&events.tail_apply_cublas_handle),
                     "cublasCreate tail_apply_cublas_handle");
        AssertCublas(cublasSetStream(events.tail_apply_cublas_handle, events.tail_apply_stream),
                     "cublasSetStream(tail_apply_stream)");
        AssertCuda(cudaEventCreateWithFlags(&events.block_wy_ready, cudaEventDisableTiming),
                   "cudaEventCreate block_wy_ready");
        AssertCuda(cudaEventCreateWithFlags(&events.tail_update_done, cudaEventDisableTiming),
                   "cudaEventCreate tail_update_done");
        AssertCuda(cudaEventCreateWithFlags(&events.compact_y_comm_done, cudaEventDisableTiming),
                   "cudaEventCreate compact_y_comm_done");
        for (int i = 0; i < 2; ++i) {
            AssertCuda(cudaEventCreateWithFlags(&events.tail_acc_done[i], cudaEventDisableTiming),
                       "cudaEventCreate tail_acc_done");
            AssertCuda(cudaEventCreateWithFlags(&events.tail_apply_done[i], cudaEventDisableTiming),
                       "cudaEventCreate tail_apply_done");
        }
        events.initialized = true;
    }
    while (static_cast<int>(events.rowblock_ready.size()) < row_block_count) {
        cudaEvent_t rowblock_ready = nullptr;
        cudaEvent_t rowblock_comm_done = nullptr;
        AssertCuda(cudaEventCreateWithFlags(&rowblock_ready, cudaEventDisableTiming),
                   "cudaEventCreate rowblock_ready");
        AssertCuda(cudaEventCreateWithFlags(&rowblock_comm_done, cudaEventDisableTiming),
                   "cudaEventCreate rowblock_comm_done");
        events.rowblock_ready.push_back(rowblock_ready);
        events.rowblock_comm_done.push_back(rowblock_comm_done);
    }
    AssertCuda(cudaEventRecord(events.tail_update_done, events.tail_apply_stream),
               "cudaEventRecord tail_update_done(reset)");
    AssertCuda(cudaEventRecord(events.compact_y_comm_done, comm_stream),
               "cudaEventRecord compact_y_comm_done(reset)");
    for (int i = 0; i < 2; ++i) {
        AssertCuda(cudaEventRecord(events.tail_apply_done[i], events.tail_apply_stream),
                   "cudaEventRecord tail_apply_done(reset)");
    }
    if (part.rank == block_owner) {
        AssertCuda(cudaEventRecord(events.block_wy_ready, compute_stream),
                   "cudaEventRecord block_wy_ready");
        AssertCuda(cudaStreamWaitEvent(events.rowblock_pack_stream, events.block_wy_ready, 0),
                   "cudaStreamWaitEvent rowblock_pack_stream <- block_wy_ready");
    }

    const ncclDataType_t nccl_type = NcclType<T>();
    if (overlap_tail) {
        for (int rb_idx = 0; rb_idx < row_block_count; ++rb_idx) {
            const int row_count = RowBlockRowsAt(block_rows, row_block_rows, rb_idx);
            const size_t off = RowBlockOffsetElems(row_block_rows, kb, rb_idx);
            const size_t elems = static_cast<size_t>(row_count) * static_cast<size_t>(kb);
            T* d_rowblock_w = ws->d_block_w_rowmajor + off;

            if (part.rank == block_owner) {
                const size_t pack_idx = BeginPhaseInterval(phase_profile, PhaseKind::RowBlockPack,
                                                           events.rowblock_pack_stream);
                PackCompactWRowBlockToRowMajorCache(rb_idx, block_rows, kb, row_block_rows,
                                                   ws->d_block_w, ws->d_block_w_rowmajor,
                                                   events.rowblock_pack_stream);
                EndPhaseInterval(phase_profile, pack_idx, events.rowblock_pack_stream);
                AssertCuda(
                    cudaEventRecord(events.rowblock_ready[rb_idx], events.rowblock_pack_stream),
                    "cudaEventRecord rowblock_ready");
                AssertCuda(cudaStreamWaitEvent(comm_stream, events.rowblock_ready[rb_idx], 0),
                           "cudaStreamWaitEvent comm_stream <- rowblock_ready");
            }

            if (block_receivers > 0) {
                const size_t comm_idx =
                    BeginPhaseInterval(phase_profile, PhaseKind::Comm, comm_stream);
                AssertNccl(ncclBroadcast(d_rowblock_w, d_rowblock_w, elems, nccl_type, block_owner,
                                         nccl_comm, comm_stream),
                           "ncclBroadcast overlap rowblock W");
                EndPhaseInterval(phase_profile, comm_idx, comm_stream);
                if (comm_profile) {
                    comm_profile->bytes += elems * sizeof(T) * static_cast<size_t>(block_receivers);
                }
            }
            AssertCuda(cudaEventRecord(events.rowblock_comm_done[rb_idx], comm_stream),
                       "cudaEventRecord rowblock_comm_done");
        }

        const size_t compact_y_elems =
            static_cast<size_t>(block_rows) * static_cast<size_t>(kb);
        if (block_receivers > 0) {
            const size_t comm_idx =
                BeginPhaseInterval(phase_profile, PhaseKind::Comm, comm_stream);
            AssertNccl(ncclBroadcast(ws->d_block_y, ws->d_block_y, compact_y_elems, nccl_type,
                                     block_owner, nccl_comm, comm_stream),
                       "ncclBroadcast overlap compact Y");
            EndPhaseInterval(phase_profile, comm_idx, comm_stream);
            if (comm_profile) {
                comm_profile->bytes +=
                    compact_y_elems * sizeof(T) * static_cast<size_t>(block_receivers);
            }
        }
        AssertCuda(cudaEventRecord(events.compact_y_comm_done, comm_stream),
                   "cudaEventRecord compact_y_comm_done");

        if (!self_has_tail) {
            AssertCuda(cudaStreamWaitEvent(compute_stream, events.compact_y_comm_done, 0),
                       "cudaStreamWaitEvent compute_stream <- compact_y_comm_done");
            return;
        }

        struct TailTileTask {
            T* a_tile = nullptr;
            int cols_tile = 0;
        };
        std::vector<TailTileTask> tail_tiles;
        int local_tail_cols = 0;
        ForEachLocalSegment(part, block_end, n, [&](int seg_begin, int seg_end, int local_begin) {
            const int cols_local = seg_end - seg_begin;
            local_tail_cols += cols_local;
            const int target_cols = std::min(pipeline_cfg.trail_tile_cols, cols_local);
            for (int col_off = 0; col_off < cols_local; col_off += target_cols) {
                const int cols_tile = std::min(target_cols, cols_local - col_off);
                tail_tiles.push_back(
                    {d_A_local + static_cast<size_t>(local_begin + col_off) * lda_local, cols_tile});
            }
        });

        for (const auto& tile : tail_tiles) {
            TailAccumulateColTileFromRowBlockCache(
                events.tail_acc_cublas_handle, ws->d_block_w_rowmajor, block_begin, block_rows, kb,
                row_block_rows, tile.a_tile, lda_local, tile.cols_tile, ws->d_tmp0, ws->tmp_elems,
                events.tail_acc_stream, true, events.rowblock_comm_done, phase_profile);

            const size_t tail_wait_idx =
                BeginPhaseInterval(phase_profile, PhaseKind::TailAccWait, events.tail_acc_stream);
            AssertCuda(cudaStreamWaitEvent(events.tail_acc_stream, events.compact_y_comm_done, 0),
                       "cudaStreamWaitEvent tail_acc_stream <- compact_y_comm_done");
            EndPhaseInterval(phase_profile, tail_wait_idx, events.tail_acc_stream);

            TailApplyCompactBlockToTile(events.tail_acc_cublas_handle, ws->d_block_y, block_begin,
                                        block_rows, kb, tile.a_tile, lda_local, tile.cols_tile,
                                        ws->d_tmp0, events.tail_acc_stream, phase_profile);
        }

        if (phase_profile && local_tail_cols > 0) {
            phase_profile->tail_update_flops += 4.0 * static_cast<double>(block_rows) *
                                                static_cast<double>(kb) *
                                                static_cast<double>(local_tail_cols);
        }

        AssertCuda(cudaEventRecord(events.tail_update_done, events.tail_acc_stream),
                   "cudaEventRecord tail_update_done");
        AssertCuda(cudaStreamWaitEvent(compute_stream, events.tail_update_done, 0),
                   "cudaStreamWaitEvent compute_stream <- tail_update_done");
        AssertCuda(cudaStreamWaitEvent(compute_stream, events.compact_y_comm_done, 0),
                   "cudaStreamWaitEvent compute_stream <- compact_y_comm_done");
        return;
    }

    for (int rb_idx = 0; rb_idx < row_block_count; ++rb_idx) {
        const int row_count = RowBlockRowsAt(block_rows, row_block_rows, rb_idx);
        const size_t off = RowBlockOffsetElems(row_block_rows, kb, rb_idx);
        const size_t packed_off = RowBlockPackedOffsetElems(row_block_rows, kb, rb_idx);
        const size_t elems = static_cast<size_t>(row_count) * static_cast<size_t>(kb);
        T* d_rowblock_wy_packed = ws->d_rowblock_wy_packed + packed_off;

        if (part.rank == block_owner) {
            const size_t pack_idx = BeginPhaseInterval(phase_profile, PhaseKind::RowBlockPack,
                                                       events.rowblock_pack_stream);
            PackCompactRowBlockToPackedBroadcastCache(
                rb_idx, block_rows, kb, row_block_rows, ws->d_block_w, ws->d_block_y,
                ws->d_rowblock_wy_packed, events.rowblock_pack_stream);
            EndPhaseInterval(phase_profile, pack_idx, events.rowblock_pack_stream);
            AssertCuda(cudaEventRecord(events.rowblock_ready[rb_idx], events.rowblock_pack_stream),
                       "cudaEventRecord rowblock_ready");
            AssertCuda(cudaStreamWaitEvent(comm_stream, events.rowblock_ready[rb_idx], 0),
                       "cudaStreamWaitEvent comm_stream <- rowblock_ready");
        }

        if (block_receivers > 0) {
            const size_t comm_idx = BeginPhaseInterval(phase_profile, PhaseKind::Comm, comm_stream);
            AssertNccl(ncclBroadcast(d_rowblock_wy_packed, d_rowblock_wy_packed, 2 * elems,
                                     nccl_type, block_owner, nccl_comm, comm_stream),
                       "ncclBroadcast pipeline packed rowblock WY");
            EndPhaseInterval(phase_profile, comm_idx, comm_stream);
            if (comm_profile) {
                comm_profile->bytes +=
                    2ULL * elems * sizeof(T) * static_cast<size_t>(block_receivers);
            }
        }
        if (part.rank == block_owner || block_receivers > 0) {
            const size_t unpack_idx =
                BeginPhaseInterval(phase_profile, PhaseKind::RowBlockUnpack, comm_stream);
            UnpackPackedRowBlockToRowMajorCache(rb_idx, block_rows, kb, row_block_rows,
                                                ws->d_rowblock_wy_packed, ws->d_block_w_rowmajor,
                                                ws->d_block_y_rowmajor, comm_stream);
            EndPhaseInterval(phase_profile, unpack_idx, comm_stream);
        }
        AssertCuda(cudaEventRecord(events.rowblock_comm_done[rb_idx], comm_stream),
                   "cudaEventRecord rowblock_comm_done");
    }

    if (!self_has_tail) {
        if (row_block_count > 0) {
            AssertCuda(cudaStreamWaitEvent(compute_stream,
                                           events.rowblock_comm_done[row_block_count - 1], 0),
                       "cudaStreamWaitEvent compute_stream <- rowblock_comm_done[last]");
        }
        return;
    }

    if (!overlap_tail && !chained_tail && row_block_count > 0) {
        const size_t tail_wait_idx =
            BeginPhaseInterval(phase_profile, PhaseKind::TailAccWait, events.tail_acc_stream);
        AssertCuda(cudaStreamWaitEvent(events.tail_acc_stream,
                                       events.rowblock_comm_done[row_block_count - 1], 0),
                   "cudaStreamWaitEvent tail_acc_stream <- rowblock_comm_done[last]");
        EndPhaseInterval(phase_profile, tail_wait_idx, events.tail_acc_stream);
    }

    struct TailTileTask {
        T* a_tile = nullptr;
        int cols_tile = 0;
    };
    std::vector<TailTileTask> tail_tiles;
    int local_tail_cols = 0;
    ForEachLocalSegment(part, block_end, n, [&](int seg_begin, int seg_end, int local_begin) {
        const int cols_local = seg_end - seg_begin;
        local_tail_cols += cols_local;
        const int target_cols = std::min(pipeline_cfg.trail_tile_cols, cols_local);
        for (int col_off = 0; col_off < cols_local; col_off += target_cols) {
            const int cols_tile = std::min(target_cols, cols_local - col_off);
            tail_tiles.push_back(
                {d_A_local + static_cast<size_t>(local_begin + col_off) * lda_local, cols_tile});
        }
    });

    if (chained_tail) {
        const T one = static_cast<T>(1);
        T* tmp_prefix = ws->d_tmp0;
        T* tmp_partial = ws->d_tmp1;
        for (const auto& tile : tail_tiles) {
            const size_t tile_need = static_cast<size_t>(kb) * static_cast<size_t>(tile.cols_tile);
            AssertCuda(
                cudaMemsetAsync(tmp_prefix, 0, tile_need * sizeof(T), events.tail_acc_stream),
                "cudaMemsetAsync chained tail tmp_prefix");
            for (int source_rb_idx = 0; source_rb_idx < row_block_count; ++source_rb_idx) {
                TailAccumulateSingleRowBlockFromRowBlockCache(
                    events.tail_acc_cublas_handle, ws->d_block_w_rowmajor, block_begin, block_rows,
                    kb, row_block_rows, source_rb_idx, tile.a_tile, lda_local, tile.cols_tile,
                    tmp_partial, ws->tmp_elems, events.tail_acc_stream, events.rowblock_comm_done,
                    phase_profile);
                AssertCublas(CublasAxpyTraits<T>::Axpy(events.tail_acc_cublas_handle,
                                                       static_cast<int>(tile_need), &one,
                                                       tmp_partial, 1, tmp_prefix, 1),
                             "Pipeline chained tmp_prefix += tmp_partial");
                for (int target_rb_idx = 0; target_rb_idx < source_rb_idx; ++target_rb_idx) {
                    TailApplySingleRowBlockFromRowBlockCache(
                        events.tail_acc_cublas_handle, ws->d_block_y_rowmajor, block_begin,
                        block_rows, kb, row_block_rows, target_rb_idx, tile.a_tile, lda_local,
                        tile.cols_tile, tmp_partial, events.tail_acc_stream, phase_profile);
                }
                TailApplySingleRowBlockFromRowBlockCache(
                    events.tail_acc_cublas_handle, ws->d_block_y_rowmajor, block_begin, block_rows,
                    kb, row_block_rows, source_rb_idx, tile.a_tile, lda_local, tile.cols_tile,
                    tmp_prefix, events.tail_acc_stream, phase_profile);
            }
        }
    } else if (overlap_tail) {
        T* tmp_buffers[2] = {ws->d_tmp0, ws->d_tmp1};
        for (size_t tile_idx = 0; tile_idx < tail_tiles.size(); ++tile_idx) {
            const int buf_idx = static_cast<int>(tile_idx % 2);
            const auto& tile = tail_tiles[tile_idx];
            TailAccumulateColTileFromRowBlockCache(
                events.tail_acc_cublas_handle, ws->d_block_w_rowmajor, block_begin, block_rows, kb,
                row_block_rows, tile.a_tile, lda_local, tile.cols_tile, tmp_buffers[buf_idx],
                ws->tmp_elems, events.tail_acc_stream, true, events.rowblock_comm_done,
                phase_profile);
            TailApplyColTileFromRowBlockCache(
                events.tail_acc_cublas_handle, ws->d_block_y_rowmajor, block_begin, block_rows, kb,
                row_block_rows, tile.a_tile, lda_local, tile.cols_tile, tmp_buffers[buf_idx],
                events.tail_acc_stream, phase_profile);
        }
    } else {
        for (const auto& tile : tail_tiles) {
            TailUpdateColTileFromRowBlockCache(
                events.tail_acc_cublas_handle, ws->d_block_w_rowmajor, ws->d_block_y_rowmajor,
                block_begin, block_rows, kb, row_block_rows, tile.a_tile, lda_local, tile.cols_tile,
                ws->d_tmp0, ws->tmp_elems, events.tail_acc_stream, false, events.rowblock_comm_done,
                phase_profile);
        }
    }

    if (phase_profile && local_tail_cols > 0) {
        if (chained_tail) {
            double apply_rows_weighted = 0.0;
            for (int rb_idx = 0; rb_idx < row_block_count; ++rb_idx) {
                apply_rows_weighted +=
                    static_cast<double>(RowBlockRowsAt(block_rows, row_block_rows, rb_idx)) *
                    static_cast<double>(row_block_count - rb_idx);
            }
            phase_profile->tail_update_flops +=
                2.0 * static_cast<double>(kb) * static_cast<double>(local_tail_cols) *
                (static_cast<double>(block_rows) + apply_rows_weighted);
        } else {
            phase_profile->tail_update_flops += 4.0 * static_cast<double>(block_rows) *
                                                static_cast<double>(kb) *
                                                static_cast<double>(local_tail_cols);
        }
    }

    AssertCuda(cudaEventRecord(events.tail_update_done, events.tail_acc_stream),
               "cudaEventRecord tail_update_done");
    AssertCuda(cudaStreamWaitEvent(compute_stream, events.tail_update_done, 0),
               "cudaStreamWaitEvent compute_stream <- tail_update_done");
    if (row_block_count > 0) {
        AssertCuda(
            cudaStreamWaitEvent(compute_stream, events.rowblock_comm_done[row_block_count - 1], 0),
            "cudaStreamWaitEvent compute_stream <- rowblock_comm_done[last]");
    }
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
    const RowBlockPipelineConfig cfg = NormalizePipelineConfig(pipeline_cfg, nb);
    ValidatePipelineConfig(*ws, cfg, nb);
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

        // Compact block-WY is the canonical block representation here, so the
        // owner clears it once per outer block and then appends each panel
        // directly into the compact layout.
        if (part.rank == block_owner) {
            AssertCuda(cudaMemsetAsync(
                           ws->d_block_w, 0,
                           static_cast<size_t>(block_rows) * static_cast<size_t>(kb) * sizeof(T),
                           compute_stream),
                       "cudaMemsetAsync compact block_w");
            AssertCuda(cudaMemsetAsync(
                           ws->d_block_y, 0,
                           static_cast<size_t>(block_rows) * static_cast<size_t>(kb) * sizeof(T),
                           compute_stream),
                       "cudaMemsetAsync compact block_y");
        }

        for (int inner = block_begin; inner < block_end; inner += kPanelWidth) {
            const int owner = OwnerOfPanel(inner, part);
            if (owner != block_owner) {
                spdlog::error("Outer block [{}, {}) crossed owner boundary at panel {}.",
                              block_begin, block_end, inner);
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

            ForEachLocalSegment(
                part, inner + kPanelWidth, block_end,
                [&](int seg_begin, int seg_end, int local_begin) {
                    const int cols_local = seg_end - seg_begin;
                    T* a_trail = d_A_local + static_cast<size_t>(local_begin) * lda_local;
                    const size_t inner_update_idx = BeginPhaseInterval(
                        phase_profile, PhaseKind::InnerBlockUpdate, compute_stream);
                    ApplyPanelToTrailingSegment(cublas_handle, inner, panel_rows, cols_local,
                                                ws->d_panel_w, ws->d_panel_y, a_trail, lda_local,
                                                ws->d_tmp0, ws->tmp_elems);
                    EndPhaseInterval(phase_profile, inner_update_idx, compute_stream);
                });
        }

        StreamedTailUpdateScaffold(cublas_handle, nccl_comm, part, block_begin, block_end, m, n,
                                   d_A_local, lda_local, ws, compute_stream, comm_stream, cfg,
                                   comm_profile, phase_profile);
    }
}

}  // namespace distributed_qr_col_blockcyclic_pipeline
