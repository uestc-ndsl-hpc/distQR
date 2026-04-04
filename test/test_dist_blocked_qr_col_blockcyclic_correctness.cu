#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "components/distributed_blocked_qr_col_blockcyclic.cuh"
#include "components/distributed_explicit_q_from_wy_col_blockcyclic.cuh"
#include "components/resourse_initial.cuh"

namespace {

MpiCudaEnv* g_env = nullptr;

void AssertCurand(curandStatus_t status, const char* context) {
    ASSERT_EQ(status, CURAND_STATUS_SUCCESS) << context;
}

template <typename T>
const char* DataTypeString() {
    if constexpr (std::is_same_v<T, float>) {
        return "float";
    }
    return "double";
}

template <typename T>
MPI_Datatype MpiType();

template <>
MPI_Datatype MpiType<float>() {
    return MPI_FLOAT;
}

template <>
MPI_Datatype MpiType<double>() {
    return MPI_DOUBLE;
}

template <typename T>
void FillDeviceRandom(T* device_data, size_t count, unsigned long long seed) {
    if (count == 0) {
        return;
    }

    curandGenerator_t gen = nullptr;
    AssertCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
    AssertCurand(curandSetPseudoRandomGeneratorSeed(gen, seed),
                 "curandSetPseudoRandomGeneratorSeed");

    if constexpr (std::is_same_v<T, float>) {
        AssertCurand(curandGenerateUniform(gen, device_data, count), "curandGenerateUniform");
    } else if constexpr (std::is_same_v<T, double>) {
        AssertCurand(curandGenerateUniformDouble(gen, device_data, count),
                     "curandGenerateUniformDouble");
    }

    AssertCurand(curandDestroyGenerator(gen), "curandDestroyGenerator");
}

template <typename T>
std::vector<T> CopyLocalMatrixToHost(const T* d_A, int lda, int rows, int cols) {
    std::vector<T> h_A(std::max(rows * cols, 1), static_cast<T>(0));
    if (rows == 0 || cols == 0) {
        return h_A;
    }

    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMemcpy2D(h_A.data(), static_cast<size_t>(rows) * sizeof(T), d_A,
                     static_cast<size_t>(lda) * sizeof(T), static_cast<size_t>(rows) * sizeof(T),
                     cols, cudaMemcpyDeviceToHost),
        "cudaMemcpy2D D2H local matrix");
    return h_A;
}

template <typename T>
struct PersistentWyAllocation {
    T* d_w = nullptr;
    T* d_y = nullptr;
    T* d_compact = nullptr;
    size_t compact_elems = 0;
    distributed_qr_col_blockcyclic::PersistentWyStorage<T> storage;
};

template <typename T>
PersistentWyAllocation<T> AllocatePersistentWyStorage(
    const distributed_qr_col_blockcyclic::ColBlockCyclicPartition& part,
    int m,
    int n,
    int nb,
    int lda_local,
    distributed_qr_col_blockcyclic::PersistentWyStorageMode mode) {
    PersistentWyAllocation<T> alloc{};
    if (mode == distributed_qr_col_blockcyclic::PersistentWyStorageMode::None) {
        alloc.storage = distributed_qr_col_blockcyclic::MakeNoPersistentWyStorage<T>();
        return alloc;
    }

    if (mode == distributed_qr_col_blockcyclic::PersistentWyStorageMode::Dense) {
        const size_t elems_alloc =
            static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(part.local_cols, 1));
        distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&alloc.d_w, elems_alloc * sizeof(T)),
                                                   "cudaMalloc persistent d_w");
        distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&alloc.d_y, elems_alloc * sizeof(T)),
                                                   "cudaMalloc persistent d_y");
        alloc.storage =
            distributed_qr_col_blockcyclic::MakeDensePersistentWyStorage(alloc.d_w, alloc.d_y,
                                                                         lda_local);
        return alloc;
    }

    alloc.compact_elems = distributed_qr_col_blockcyclic::CompactWyStorageRequiredElems(
        m, n, nb, part.block_starts, part.block_ends);
    if (alloc.compact_elems > 0) {
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMalloc(&alloc.d_compact, alloc.compact_elems * sizeof(T)),
            "cudaMalloc persistent compact WY");
    }
    alloc.storage = distributed_qr_col_blockcyclic::BuildCompactPersistentWyStorage(
        m, n, nb, part.block_starts, part.block_ends, alloc.d_compact, alloc.compact_elems);
    return alloc;
}

template <typename T>
void FreePersistentWyStorage(PersistentWyAllocation<T>* alloc) {
    ASSERT_NE(alloc, nullptr);
    if (alloc->d_w) {
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(alloc->d_w), "cudaFree persistent d_w");
    }
    if (alloc->d_y) {
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(alloc->d_y), "cudaFree persistent d_y");
    }
    if (alloc->d_compact) {
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(alloc->d_compact),
                                                   "cudaFree persistent compact WY");
    }
    *alloc = PersistentWyAllocation<T>{};
}

template <typename T>
void AllocateColBlockCyclicWorkspace(
    int m,
    int nb,
    int tile_cols,
    int panel_buffers,
    bool full_block_ping_pong,
    distributed_qr_col_blockcyclic::DistributedQrColBlockCyclicWorkspace<T>* ws) {
    ASSERT_NE(ws, nullptr);
    ws->pack_buffer_count = panel_buffers;
    ws->d_pack_w.assign(ws->pack_buffer_count, nullptr);
    ws->d_pack_y.assign(ws->pack_buffer_count, nullptr);
    ws->tsqr_work_panel_elems = std::max(tsqr_work_elems<T>(m), static_cast<size_t>(1));
    ws->pack_elems = static_cast<size_t>(m) *
                     static_cast<size_t>(distributed_qr_col_blockcyclic::kPanelWidth);
    ws->block_storage_elems = static_cast<size_t>(m) * static_cast<size_t>(nb);
    const size_t compact_need =
        static_cast<size_t>(nb + distributed_qr_col_blockcyclic::kPanelWidth) *
        static_cast<size_t>(nb);
    ws->block_compact_elems = full_block_ping_pong
                                  ? ws->block_storage_elems
                                  : std::max(compact_need, ws->block_storage_elems);
    ws->tmp_elems = static_cast<size_t>(nb) * static_cast<size_t>(tile_cols);

    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws->d_r_panel, static_cast<size_t>(distributed_qr_col_blockcyclic::kPanelWidth) *
                                      distributed_qr_col_blockcyclic::kPanelWidth * sizeof(T)),
        "cudaMalloc ws.d_r_panel");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws->d_tsqr_work_panel, ws->tsqr_work_panel_elems * sizeof(T)),
        "cudaMalloc ws.d_tsqr_work_panel");
    for (int i = 0; i < ws->pack_buffer_count; ++i) {
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMalloc(&ws->d_pack_w[i], ws->pack_elems * sizeof(T)), "cudaMalloc ws.d_pack_w[i]");
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMalloc(&ws->d_pack_y[i], ws->pack_elems * sizeof(T)), "cudaMalloc ws.d_pack_y[i]");
    }
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws->d_block_w, ws->block_storage_elems * sizeof(T)), "cudaMalloc ws.d_block_w");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws->d_block_y, ws->block_storage_elems * sizeof(T)), "cudaMalloc ws.d_block_y");
    if (full_block_ping_pong) {
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMalloc(&ws->d_block_w_alt, ws->block_storage_elems * sizeof(T)),
            "cudaMalloc ws.d_block_w_alt");
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMalloc(&ws->d_block_y_alt, ws->block_storage_elems * sizeof(T)),
            "cudaMalloc ws.d_block_y_alt");
    }
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws->d_block_w_compact, ws->block_compact_elems * sizeof(T)),
        "cudaMalloc ws.d_block_w_compact");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws->d_block_y_compact, ws->block_compact_elems * sizeof(T)),
        "cudaMalloc ws.d_block_y_compact");
    if (full_block_ping_pong) {
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMalloc(&ws->d_block_w_compact_alt, ws->block_compact_elems * sizeof(T)),
            "cudaMalloc ws.d_block_w_compact_alt");
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMalloc(&ws->d_block_y_compact_alt, ws->block_compact_elems * sizeof(T)),
            "cudaMalloc ws.d_block_y_compact_alt");
    }
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&ws->d_tmp0, ws->tmp_elems * sizeof(T)),
                                               "cudaMalloc ws.d_tmp0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&ws->d_tmp1, ws->tmp_elems * sizeof(T)),
                                               "cudaMalloc ws.d_tmp1");
}

template <typename T>
void FreeColBlockCyclicWorkspace(
    distributed_qr_col_blockcyclic::DistributedQrColBlockCyclicWorkspace<T>* ws) {
    ASSERT_NE(ws, nullptr);
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_r_panel), "cudaFree ws.d_r_panel");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_tsqr_work_panel),
                                               "cudaFree ws.d_tsqr_work_panel");
    for (int i = 0; i < ws->pack_buffer_count; ++i) {
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_pack_w[i]), "cudaFree ws.d_pack_w[i]");
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_pack_y[i]), "cudaFree ws.d_pack_y[i]");
    }
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_block_w), "cudaFree ws.d_block_w");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_block_y), "cudaFree ws.d_block_y");
    if (ws->d_block_w_alt) {
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_block_w_alt),
                                                   "cudaFree ws.d_block_w_alt");
    }
    if (ws->d_block_y_alt) {
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_block_y_alt),
                                                   "cudaFree ws.d_block_y_alt");
    }
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_block_w_compact),
                                               "cudaFree ws.d_block_w_compact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_block_y_compact),
                                               "cudaFree ws.d_block_y_compact");
    if (ws->d_block_w_compact_alt) {
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_block_w_compact_alt),
                                                   "cudaFree ws.d_block_w_compact_alt");
    }
    if (ws->d_block_y_compact_alt) {
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_block_y_compact_alt),
                                                   "cudaFree ws.d_block_y_compact_alt");
    }
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_tmp0), "cudaFree ws.d_tmp0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws->d_tmp1), "cudaFree ws.d_tmp1");
}

template <typename T>
std::vector<T> GatherColBlockCyclicMatrixToRoot(const std::vector<T>& h_local,
                                                int rows,
                                                int n_global,
                                                int block_cols,
                                                const MpiCudaEnv& env) {
    const auto part = distributed_qr_col_blockcyclic::MakeColBlockCyclicPartition(
        n_global, block_cols, env.size, env.rank);
    const int send_count = rows * part.local_cols;

    std::vector<int> recv_counts;
    std::vector<int> displs;
    int total_count = 0;
    if (env.rank == 0) {
        recv_counts.resize(env.size, 0);
        displs.resize(env.size, 0);
        for (int r = 0; r < env.size; ++r) {
            const auto part_r = distributed_qr_col_blockcyclic::MakeColBlockCyclicPartition(
                n_global, block_cols, env.size, r);
            recv_counts[r] = rows * part_r.local_cols;
            displs[r] = total_count;
            total_count += recv_counts[r];
        }
    }

    std::vector<T> gathered((env.rank == 0) ? std::max(total_count, 1) : 1, static_cast<T>(0));
    const int gather_status =
        MPI_Gatherv(h_local.data(), send_count, MpiType<T>(),
                    (env.rank == 0) ? gathered.data() : nullptr,
                    (env.rank == 0) ? recv_counts.data() : nullptr,
                    (env.rank == 0) ? displs.data() : nullptr, MpiType<T>(), 0,
                    MPI_COMM_WORLD);
    EXPECT_EQ(gather_status, MPI_SUCCESS);
    if (gather_status != MPI_SUCCESS) {
        return {};
    }

    if (env.rank != 0) {
        return {};
    }

    std::vector<T> h_global(static_cast<size_t>(rows) * static_cast<size_t>(n_global),
                            static_cast<T>(0));
    for (int r = 0; r < env.size; ++r) {
        const auto part_r = distributed_qr_col_blockcyclic::MakeColBlockCyclicPartition(
            n_global, block_cols, env.size, r);
        const T* src_rank = gathered.data() + displs[r];
        for (size_t b = 0; b < part_r.block_starts.size(); ++b) {
            const int global_begin = part_r.block_starts[b];
            const int cols = part_r.block_ends[b] - global_begin;
            const int local_begin = part_r.block_local_offsets[b];
            const T* src_block = src_rank + static_cast<size_t>(local_begin) * rows;
            T* dst_block = h_global.data() + static_cast<size_t>(global_begin) * rows;
            for (int col = 0; col < cols; ++col) {
                std::copy_n(src_block + static_cast<size_t>(col) * rows, rows,
                            dst_block + static_cast<size_t>(col) * rows);
            }
        }
    }
    return h_global;
}

template <typename T>
std::vector<double> ExtractUpperRToDouble(const std::vector<T>& h_Afact, int m, int n) {
    std::vector<double> h_R(static_cast<size_t>(std::max(n * n, 1)), 0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            h_R[static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(n)] =
                static_cast<double>(h_Afact[static_cast<size_t>(row) +
                                            static_cast<size_t>(col) * static_cast<size_t>(m)]);
        }
    }
    return h_R;
}

template <typename T>
double ExplicitQOrthogonalityError(const std::vector<T>& h_Q, int m, int n) {
    double diff_sq = 0.0;
    for (int col_j = 0; col_j < n; ++col_j) {
        for (int col_i = 0; col_i < n; ++col_i) {
            double dot = 0.0;
            for (int row = 0; row < m; ++row) {
                dot += static_cast<double>(
                           h_Q[static_cast<size_t>(row) +
                               static_cast<size_t>(col_i) * static_cast<size_t>(m)]) *
                       static_cast<double>(
                           h_Q[static_cast<size_t>(row) +
                               static_cast<size_t>(col_j) * static_cast<size_t>(m)]);
            }
            const double target = (col_i == col_j) ? 1.0 : 0.0;
            const double diff = dot - target;
            diff_sq += diff * diff;
        }
    }
    return std::sqrt(diff_sq) / std::max(std::sqrt(static_cast<double>(n)), 1.0e-12);
}

template <typename T>
double ExplicitQReconstructionError(const std::vector<T>& h_A0,
                                    const std::vector<T>& h_Q,
                                    const std::vector<double>& h_R,
                                    int m,
                                    int n) {
    double diff_sq = 0.0;
    double base_sq = 0.0;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double qr = 0.0;
            for (int k = 0; k <= col; ++k) {
                qr += static_cast<double>(
                          h_Q[static_cast<size_t>(row) +
                              static_cast<size_t>(k) * static_cast<size_t>(m)]) *
                      h_R[static_cast<size_t>(k) +
                          static_cast<size_t>(col) * static_cast<size_t>(n)];
            }
            const double a0 =
                static_cast<double>(h_A0[static_cast<size_t>(row) +
                                         static_cast<size_t>(col) * static_cast<size_t>(m)]);
            const double diff = a0 - qr;
            diff_sq += diff * diff;
            base_sq += a0 * a0;
        }
    }
    return std::sqrt(diff_sq) / std::max(std::sqrt(base_sq), 1.0e-12);
}

template <typename T>
void ApplyAllOuterPanelsQTToADense(
    T* d_A,
    int m,
    int n,
    int nb,
    int overlap_tile,
    cublasHandle_t cublas_handle,
    ncclComm_t nccl_comm,
    const distributed_qr_col_blockcyclic::ColBlockCyclicPartition& part,
    const T* d_W_local,
    const T* d_Y_local,
    distributed_qr_col_blockcyclic::DistributedQrColBlockCyclicWorkspace<T>* ws,
    cudaStream_t compute_stream) {
    const int lda_local = std::max(m, 1);
    const int tile_cols =
        (overlap_tile <= 0)
            ? nb
            : std::max(distributed_qr_col_blockcyclic::kPanelWidth, std::min(overlap_tile, nb));
    const ncclDataType_t nccl_type = distributed_qr_col_blockcyclic::NcclType<T>();

    distributed_qr_col_blockcyclic::AssertCublas(cublasSetStream(cublas_handle, compute_stream),
                                                 "cublasSetStream(compute_stream)");
    ASSERT_NE(ws, nullptr);
    ASSERT_NE(d_W_local, nullptr);
    ASSERT_NE(d_Y_local, nullptr);
    ASSERT_FALSE(ws->d_pack_w.empty());
    ASSERT_FALSE(ws->d_pack_y.empty());
    T* pack_w = ws->d_pack_w.front();
    T* pack_y = ws->d_pack_y.front();

    for (int block_begin = 0; block_begin < n; block_begin += nb) {
        const int block_end = std::min(block_begin + nb, n);
        const int kb = block_end - block_begin;
        const int rows = m - block_begin;

        for (int inner = block_begin; inner < block_end;
             inner += distributed_qr_col_blockcyclic::kPanelWidth) {
            const int owner = distributed_qr_col_blockcyclic::OwnerOfPanel(inner, part);
            const int local_panel_col = distributed_qr_col_blockcyclic::LocalColOffset(part, inner);
            const bool owner_has_panel = (owner == part.rank) && (local_panel_col >= 0);
            const int block_col_off = inner - block_begin;

            if (owner_has_panel) {
                distributed_qr_col_blockcyclic::AssertCuda(
                    cudaMemcpy2DAsync(
                        pack_w, static_cast<size_t>(rows) * sizeof(T),
                        d_W_local + static_cast<size_t>(local_panel_col) * lda_local + block_begin,
                        static_cast<size_t>(lda_local) * sizeof(T),
                        static_cast<size_t>(rows) * sizeof(T),
                        distributed_qr_col_blockcyclic::kPanelWidth, cudaMemcpyDeviceToDevice,
                        compute_stream),
                    "cudaMemcpy2DAsync local W -> pack");
                distributed_qr_col_blockcyclic::AssertCuda(
                    cudaMemcpy2DAsync(
                        pack_y, static_cast<size_t>(rows) * sizeof(T),
                        d_Y_local + static_cast<size_t>(local_panel_col) * lda_local + block_begin,
                        static_cast<size_t>(lda_local) * sizeof(T),
                        static_cast<size_t>(rows) * sizeof(T),
                        distributed_qr_col_blockcyclic::kPanelWidth, cudaMemcpyDeviceToDevice,
                        compute_stream),
                    "cudaMemcpy2DAsync local Y -> pack");
            }

            const size_t msg_elems =
                static_cast<size_t>(rows) *
                static_cast<size_t>(distributed_qr_col_blockcyclic::kPanelWidth);
            distributed_qr_col_blockcyclic::AssertNccl(ncclGroupStart(),
                                                       "ncclGroupStart panel W/Y(pack)");
            if (part.rank == owner) {
                for (int r = 0; r < part.world_size; ++r) {
                    if (r == owner) {
                        continue;
                    }
                    distributed_qr_col_blockcyclic::AssertNccl(
                        ncclSend(pack_w, msg_elems, nccl_type, r, nccl_comm, compute_stream),
                        "ncclSend panel W(pack)");
                    distributed_qr_col_blockcyclic::AssertNccl(
                        ncclSend(pack_y, msg_elems, nccl_type, r, nccl_comm, compute_stream),
                        "ncclSend panel Y(pack)");
                }
            } else {
                distributed_qr_col_blockcyclic::AssertNccl(
                    ncclRecv(pack_w, msg_elems, nccl_type, owner, nccl_comm, compute_stream),
                    "ncclRecv panel W(pack)");
                distributed_qr_col_blockcyclic::AssertNccl(
                    ncclRecv(pack_y, msg_elems, nccl_type, owner, nccl_comm, compute_stream),
                    "ncclRecv panel Y(pack)");
            }
            distributed_qr_col_blockcyclic::AssertNccl(ncclGroupEnd(),
                                                       "ncclGroupEnd panel W/Y(pack)");

            T* dst_w = ws->d_block_w + static_cast<size_t>(block_begin) +
                       static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
            T* dst_y = ws->d_block_y + static_cast<size_t>(block_begin) +
                       static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
            distributed_qr_col_blockcyclic::AssertCuda(
                cudaMemcpy2DAsync(dst_w, static_cast<size_t>(m) * sizeof(T), pack_w,
                                  static_cast<size_t>(rows) * sizeof(T),
                                  static_cast<size_t>(rows) * sizeof(T),
                                  distributed_qr_col_blockcyclic::kPanelWidth,
                                  cudaMemcpyDeviceToDevice, compute_stream),
                "cudaMemcpy2DAsync pack_w -> block_w");
            distributed_qr_col_blockcyclic::AssertCuda(
                cudaMemcpy2DAsync(dst_y, static_cast<size_t>(m) * sizeof(T), pack_y,
                                  static_cast<size_t>(rows) * sizeof(T),
                                  static_cast<size_t>(rows) * sizeof(T),
                                  distributed_qr_col_blockcyclic::kPanelWidth,
                                  cudaMemcpyDeviceToDevice, compute_stream),
                "cudaMemcpy2DAsync pack_y -> block_y");
        }

        distributed_qr_col_blockcyclic::ForEachLocalSegment(
            part, block_begin, n, [&](int seg_begin, int seg_end, int local_begin) {
                const int cols_local = seg_end - seg_begin;
                T* a_trail = d_A + static_cast<size_t>(local_begin) * lda_local;
                distributed_qr_col_blockcyclic::block_update_tile_pipeline<T>(
                    cublas_handle, compute_stream, block_begin, rows, kb, cols_local, tile_cols,
                    ws->d_block_w + static_cast<size_t>(block_begin),
                    ws->d_block_y + static_cast<size_t>(block_begin), m, a_trail, lda_local,
                    ws->d_tmp0, ws->d_tmp1);
            });
    }
}

template <typename T>
void MaterializePersistentWyToDense(
    const distributed_qr_col_blockcyclic::ColBlockCyclicPartition& part,
    int m,
    int n,
    int nb,
    int lda_local,
    const distributed_qr_col_blockcyclic::PersistentWyStorage<T>& persistent_wy,
    T* d_W_dense,
    T* d_Y_dense,
    cudaStream_t compute_stream) {
    ASSERT_NE(d_W_dense, nullptr);
    ASSERT_NE(d_Y_dense, nullptr);
    const size_t elems_alloc =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(part.local_cols, 1));
    distributed_qr_col_blockcyclic::AssertCuda(cudaMemsetAsync(d_W_dense, 0, elems_alloc * sizeof(T),
                                                               compute_stream),
                                               "cudaMemsetAsync materialize W");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMemsetAsync(d_Y_dense, 0, elems_alloc * sizeof(T),
                                                               compute_stream),
                                               "cudaMemsetAsync materialize Y");

    for (size_t i = 0; i < part.block_starts.size(); ++i) {
        for (int block_begin = part.block_starts[i]; block_begin < part.block_ends[i];
             block_begin += nb) {
            const int block_end = std::min(block_begin + nb, n);
            const int kb = block_end - block_begin;
            const int rows = m - block_begin;
            const int local_block_col = distributed_qr_col_blockcyclic::LocalColOffset(part, block_begin);
            ASSERT_GE(local_block_col, 0);
            distributed_qr_col_blockcyclic::LoadPersistentWyBlock(
                persistent_wy, block_begin, rows, kb, local_block_col,
                d_W_dense + static_cast<size_t>(local_block_col) * lda_local + block_begin,
                d_Y_dense + static_cast<size_t>(local_block_col) * lda_local + block_begin,
                lda_local, compute_stream);
        }
    }
}

struct ColBlockCyclicCorrectnessConfig {
    int m = 2048;
    int n = 512;
    int nb = 256;
    int block_cols = 256;
    int overlap_tile = 0;
    int panel_buffers = 2;
    distributed_qr_col_blockcyclic::PanelCommMode panel_comm_mode =
        distributed_qr_col_blockcyclic::PanelCommMode::SendRecv;
    distributed_qr_col_blockcyclic::BroadcastMode broadcast_mode =
        distributed_qr_col_blockcyclic::BroadcastMode::Panel;
    distributed_qr_col_blockcyclic::PersistentWyStorageMode wy_storage_mode =
        distributed_qr_col_blockcyclic::PersistentWyStorageMode::Dense;
    bool use_compact_local_gemm = true;
    const char* case_name = "default";
};

template <typename T>
void RunFactorizedAEqualsQtA0(const ColBlockCyclicCorrectnessConfig& cfg,
                              double rel_upper_tol,
                              double lower_ratio_tol) {
    ASSERT_NE(g_env, nullptr);
    const auto& env = *g_env;

    ASSERT_EQ(cfg.n % distributed_qr_col_blockcyclic::kPanelWidth, 0);
    ASSERT_EQ(cfg.nb % distributed_qr_col_blockcyclic::kPanelWidth, 0);
    ASSERT_GT(cfg.m, 0);
    ASSERT_GT(cfg.n, 0);
    ASSERT_GT(cfg.nb, 0);
    ASSERT_GT(cfg.block_cols, 0);
    ASSERT_GE(cfg.panel_buffers, 2);
    ASSERT_EQ(cfg.block_cols % cfg.nb, 0);
    ASSERT_EQ(cfg.block_cols % distributed_qr_col_blockcyclic::kPanelWidth, 0);
    ASSERT_NE(cfg.wy_storage_mode, distributed_qr_col_blockcyclic::PersistentWyStorageMode::None);

    const auto part = distributed_qr_col_blockcyclic::MakeColBlockCyclicPartition(
        cfg.n, cfg.block_cols, env.size, env.rank);
    const int lda_local = std::max(cfg.m, 1);
    const size_t elems_alloc =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(part.local_cols, 1));
    const size_t elems_used = static_cast<size_t>(cfg.m) * static_cast<size_t>(part.local_cols);

    T* d_A0 = nullptr;
    T* d_Afact = nullptr;
    T* d_Aqt = nullptr;
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_A0, elems_alloc * sizeof(T)),
                                               "cudaMalloc d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_Afact, elems_alloc * sizeof(T)),
                                               "cudaMalloc d_Afact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_Aqt, elems_alloc * sizeof(T)),
                                               "cudaMalloc d_Aqt");
    auto persistent_wy = AllocatePersistentWyStorage<T>(part, cfg.m, cfg.n, cfg.nb, lda_local,
                                                        cfg.wy_storage_mode);

    FillDeviceRandom(d_A0, elems_used, 20260220ULL + static_cast<unsigned long long>(env.rank));

    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMemcpy(d_Afact, d_A0, elems_alloc * sizeof(T), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Afact <- d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMemcpy(d_Aqt, d_A0, elems_alloc * sizeof(T), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Aqt <- d_A0");

    const int tile_cols =
        (cfg.overlap_tile <= 0)
            ? std::max(part.local_cols, 1)
            : std::max(distributed_qr_col_blockcyclic::kPanelWidth,
                       std::min(cfg.overlap_tile, cfg.nb));
    distributed_qr_col_blockcyclic::DistributedQrColBlockCyclicWorkspace<T> ws{};
    ws.pack_buffer_count = cfg.panel_buffers;
    const bool need_block_lookahead_buffers =
        cfg.panel_comm_mode == distributed_qr_col_blockcyclic::PanelCommMode::Broadcast &&
        distributed_qr_col_blockcyclic::IsBlockBroadcastMode(cfg.broadcast_mode);
    AllocateColBlockCyclicWorkspace<T>(cfg.m, cfg.nb, tile_cols, cfg.panel_buffers,
                                       need_block_lookahead_buffers, &ws);

    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking),
        "cudaStreamCreate compute_stream");
    int least_priority = 0;
    int greatest_priority = 0;
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
        "cudaDeviceGetStreamPriorityRange");
    (void)least_priority;
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaStreamCreateWithPriority(&comm_stream, cudaStreamNonBlocking, greatest_priority),
        "cudaStreamCreate comm_stream");

    cublasHandle_t cublas_handle = nullptr;
    distributed_qr_col_blockcyclic::AssertCublas(cublasCreate(&cublas_handle), "cublasCreate");
    distributed_qr_col_blockcyclic::AssertCublas(cublasSetStream(cublas_handle, compute_stream),
                                                 "cublasSetStream compute_stream");

    distributed_qr_col_blockcyclic::distributed_blocked_qr_factorize_col_blockcyclic<T>(
        cublas_handle, env.nccl_comm, part, cfg.m, cfg.n, cfg.nb, d_Afact, lda_local,
        persistent_wy.storage, &ws, compute_stream, comm_stream, cfg.overlap_tile, nullptr,
        cfg.panel_comm_mode, cfg.use_compact_local_gemm, cfg.broadcast_mode);

    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(compute_stream),
                                               "cudaStreamSynchronize factorize compute");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(comm_stream),
                                               "cudaStreamSynchronize factorize comm");

    T* d_W_apply = nullptr;
    T* d_Y_apply = nullptr;
    if (persistent_wy.storage.mode == distributed_qr_col_blockcyclic::PersistentWyStorageMode::Dense) {
        d_W_apply = persistent_wy.d_w;
        d_Y_apply = persistent_wy.d_y;
    } else {
        distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_W_apply, elems_alloc * sizeof(T)),
                                                   "cudaMalloc d_W_apply");
        distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_Y_apply, elems_alloc * sizeof(T)),
                                                   "cudaMalloc d_Y_apply");
        MaterializePersistentWyToDense(part, cfg.m, cfg.n, cfg.nb, lda_local, persistent_wy.storage,
                                       d_W_apply, d_Y_apply, compute_stream);
        distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(compute_stream),
                                                   "cudaStreamSynchronize materialize WY");
    }

    ApplyAllOuterPanelsQTToADense(d_Aqt, cfg.m, cfg.n, cfg.nb, cfg.overlap_tile, cublas_handle,
                                  env.nccl_comm, part, d_W_apply, d_Y_apply, &ws, compute_stream);

    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(compute_stream),
                                               "cudaStreamSynchronize applyQT compute");

    const auto h_Afact = CopyLocalMatrixToHost(d_Afact, lda_local, cfg.m, part.local_cols);
    const auto h_Aqt = CopyLocalMatrixToHost(d_Aqt, lda_local, cfg.m, part.local_cols);

    double local_upper_diff_sq = 0.0;
    double local_upper_base_sq = 0.0;
    double local_lower_sq = 0.0;

    for (int col = 0; col < part.local_cols; ++col) {
        int global_col = -1;
        int col_offset = col;
        for (size_t b = 0; b < part.block_starts.size(); ++b) {
            const int block_cols_local = part.block_ends[b] - part.block_starts[b];
            if (col_offset < block_cols_local) {
                global_col = part.block_starts[b] + col_offset;
                break;
            }
            col_offset -= block_cols_local;
        }
        if (global_col < 0) {
            continue;
        }

        for (int row = 0; row < cfg.m; ++row) {
            const size_t idx =
                static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(cfg.m);
            const double a_fact = static_cast<double>(h_Afact[idx]);
            const double a_qt = static_cast<double>(h_Aqt[idx]);
            if (row <= global_col) {
                const double diff = a_fact - a_qt;
                local_upper_diff_sq += diff * diff;
                local_upper_base_sq += a_fact * a_fact;
            } else {
                local_lower_sq += a_qt * a_qt;
            }
        }
    }

    double global_upper_diff_sq = 0.0;
    double global_upper_base_sq = 0.0;
    double global_lower_sq = 0.0;
    MPI_Allreduce(&local_upper_diff_sq, &global_upper_diff_sq, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_upper_base_sq, &global_upper_base_sq, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_lower_sq, &global_lower_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const double rel_upper =
        std::sqrt(global_upper_diff_sq) / std::max(std::sqrt(global_upper_base_sq), 1.0e-12);
    const double lower_ratio =
        std::sqrt(global_lower_sq) / std::max(std::sqrt(global_upper_base_sq), 1.0e-12);

    if (env.rank == 0) {
        spdlog::info(
            "Col blockcyclic correctness [{}] ({}): rel(upper(Afact), upper(Q^T*A0))={:.3e}, "
            "lower_ratio={:.3e}",
            cfg.case_name, DataTypeString<T>(), rel_upper, lower_ratio);
    }

    EXPECT_LT(rel_upper, rel_upper_tol);
    EXPECT_LT(lower_ratio, lower_ratio_tol);

    distributed_qr_col_blockcyclic::AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamDestroy(compute_stream),
                                               "cudaStreamDestroy compute");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamDestroy(comm_stream),
                                               "cudaStreamDestroy comm");

    FreeColBlockCyclicWorkspace(&ws);

    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_Afact), "cudaFree d_Afact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_Aqt), "cudaFree d_Aqt");
    if (persistent_wy.storage.mode != distributed_qr_col_blockcyclic::PersistentWyStorageMode::Dense) {
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_W_apply), "cudaFree d_W_apply");
        distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_Y_apply), "cudaFree d_Y_apply");
    }
    FreePersistentWyStorage(&persistent_wy);
}

template <typename T>
void RunExplicitQReconstructsA0(const ColBlockCyclicCorrectnessConfig& cfg,
                                double ortho_tol,
                                double recon_tol) {
    ASSERT_NE(g_env, nullptr);
    const auto& env = *g_env;

    ASSERT_EQ(cfg.n % distributed_qr_col_blockcyclic::kPanelWidth, 0);
    ASSERT_EQ(cfg.nb % distributed_qr_col_blockcyclic::kPanelWidth, 0);
    ASSERT_GT(cfg.m, 0);
    ASSERT_GT(cfg.n, 0);
    ASSERT_GT(cfg.nb, 0);
    ASSERT_GT(cfg.block_cols, 0);
    ASSERT_GE(cfg.panel_buffers, 2);
    ASSERT_EQ(cfg.block_cols % cfg.nb, 0);
    ASSERT_EQ(cfg.block_cols % distributed_qr_col_blockcyclic::kPanelWidth, 0);
    ASSERT_NE(cfg.wy_storage_mode, distributed_qr_col_blockcyclic::PersistentWyStorageMode::None);

    const auto part = distributed_qr_col_blockcyclic::MakeColBlockCyclicPartition(
        cfg.n, cfg.block_cols, env.size, env.rank);
    const int lda_local = std::max(cfg.m, 1);
    const size_t elems_alloc =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(part.local_cols, 1));
    const size_t elems_used = static_cast<size_t>(cfg.m) * static_cast<size_t>(part.local_cols);

    T* d_A0 = nullptr;
    T* d_Afact = nullptr;
    T* d_Q = nullptr;
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_A0, elems_alloc * sizeof(T)),
                                               "cudaMalloc d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_Afact, elems_alloc * sizeof(T)),
                                               "cudaMalloc d_Afact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_Q, elems_alloc * sizeof(T)),
                                               "cudaMalloc d_Q");
    auto persistent_wy = AllocatePersistentWyStorage<T>(part, cfg.m, cfg.n, cfg.nb, lda_local,
                                                        cfg.wy_storage_mode);

    FillDeviceRandom(d_A0, elems_used, 20260316ULL + static_cast<unsigned long long>(env.rank));

    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMemcpy(d_Afact, d_A0, elems_alloc * sizeof(T), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Afact <- d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMemset(d_Q, 0, elems_alloc * sizeof(T)),
                                               "cudaMemset d_Q");

    const int tile_cols =
        (cfg.overlap_tile <= 0)
            ? std::max(part.local_cols, 1)
            : std::max(distributed_qr_col_blockcyclic::kPanelWidth,
                       std::min(cfg.overlap_tile, cfg.nb));
    distributed_qr_col_blockcyclic::DistributedQrColBlockCyclicWorkspace<T> ws{};
    AllocateColBlockCyclicWorkspace<T>(cfg.m, cfg.nb, tile_cols, cfg.panel_buffers, true, &ws);

    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking),
        "cudaStreamCreate compute_stream");
    int least_priority = 0;
    int greatest_priority = 0;
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
        "cudaDeviceGetStreamPriorityRange");
    (void)least_priority;
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaStreamCreateWithPriority(&comm_stream, cudaStreamNonBlocking, greatest_priority),
        "cudaStreamCreate comm_stream");

    cublasHandle_t cublas_handle = nullptr;
    distributed_qr_col_blockcyclic::AssertCublas(cublasCreate(&cublas_handle), "cublasCreate");
    distributed_qr_col_blockcyclic::AssertCublas(cublasSetStream(cublas_handle, compute_stream),
                                                 "cublasSetStream compute_stream");

    distributed_qr_col_blockcyclic::distributed_blocked_qr_factorize_col_blockcyclic<T>(
        cublas_handle, env.nccl_comm, part, cfg.m, cfg.n, cfg.nb, d_Afact, lda_local,
        persistent_wy.storage, &ws, compute_stream, comm_stream, cfg.overlap_tile, nullptr,
        cfg.panel_comm_mode, cfg.use_compact_local_gemm, cfg.broadcast_mode);
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(compute_stream),
                                               "cudaStreamSynchronize factorize compute");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(comm_stream),
                                               "cudaStreamSynchronize factorize comm");

    distributed_qr_col_blockcyclic::generate_explicit_q_from_wy_col_blockcyclic<T>(
        cublas_handle, env.nccl_comm, part, cfg.m, cfg.n, cfg.nb, persistent_wy.storage, d_Q,
        lda_local, &ws, compute_stream, comm_stream, cfg.overlap_tile, cfg.panel_comm_mode);
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(compute_stream),
                                               "cudaStreamSynchronize explicitQ compute");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(comm_stream),
                                               "cudaStreamSynchronize explicitQ comm");

    const auto h_A0_local = CopyLocalMatrixToHost(d_A0, lda_local, cfg.m, part.local_cols);
    const auto h_Afact_local = CopyLocalMatrixToHost(d_Afact, lda_local, cfg.m, part.local_cols);
    const auto h_Q_local = CopyLocalMatrixToHost(d_Q, lda_local, cfg.m, part.local_cols);

    const auto h_A0 = GatherColBlockCyclicMatrixToRoot(h_A0_local, cfg.m, cfg.n, cfg.block_cols, env);
    const auto h_Afact =
        GatherColBlockCyclicMatrixToRoot(h_Afact_local, cfg.m, cfg.n, cfg.block_cols, env);
    const auto h_Q = GatherColBlockCyclicMatrixToRoot(h_Q_local, cfg.m, cfg.n, cfg.block_cols, env);

    double ortho_err = 0.0;
    double recon_err = 0.0;
    if (env.rank == 0) {
        const auto h_R = ExtractUpperRToDouble(h_Afact, cfg.m, cfg.n);
        ortho_err = ExplicitQOrthogonalityError(h_Q, cfg.m, cfg.n);
        recon_err = ExplicitQReconstructionError(h_A0, h_Q, h_R, cfg.m, cfg.n);
        spdlog::info("Col blockcyclic explicit Q [{}] ({}): orth_err={:.3e}, recon_err={:.3e}",
                     cfg.case_name, DataTypeString<T>(), ortho_err, recon_err);
    }
    ASSERT_EQ(MPI_Bcast(&ortho_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), MPI_SUCCESS);
    ASSERT_EQ(MPI_Bcast(&recon_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), MPI_SUCCESS);

    EXPECT_LT(ortho_err, ortho_tol);
    EXPECT_LT(recon_err, recon_tol);

    distributed_qr_col_blockcyclic::AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamDestroy(compute_stream),
                                               "cudaStreamDestroy compute");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamDestroy(comm_stream),
                                               "cudaStreamDestroy comm");
    FreeColBlockCyclicWorkspace(&ws);
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_Afact), "cudaFree d_Afact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_Q), "cudaFree d_Q");
    FreePersistentWyStorage(&persistent_wy);
}

TEST(DistBlockedQrColBlockCyclicCorrectnessTest, FactorizedAEqualsQtA0Float) {
    RunFactorizedAEqualsQtA0<float>({}, 8e-4, 8e-4);
}

TEST(DistBlockedQrColBlockCyclicCorrectnessTest, FactorizedAEqualsQtA0Double) {
    RunFactorizedAEqualsQtA0<double>({}, 1e-7, 1e-7);
}

TEST(DistBlockedQrColBlockCyclicCorrectnessTest, BroadcastBlockCols2NbTiledSegmentedFloat) {
    ColBlockCyclicCorrectnessConfig cfg;
    cfg.m = 8192;
    cfg.n = 8192;
    cfg.nb = 1024;
    cfg.block_cols = 2048;
    cfg.overlap_tile = 1024;
    cfg.panel_comm_mode = distributed_qr_col_blockcyclic::PanelCommMode::Broadcast;
    cfg.use_compact_local_gemm = false;
    cfg.case_name = "broadcast_block_cols_2048_tiled_segmented";
    RunFactorizedAEqualsQtA0<float>(cfg, 1.5e-3, 1.5e-3);
}

TEST(DistBlockedQrColBlockCyclicCorrectnessTest, BroadcastThreePanelBuffersFloat) {
    ColBlockCyclicCorrectnessConfig cfg;
    cfg.m = 4096;
    cfg.n = 4096;
    cfg.nb = 512;
    cfg.block_cols = 1024;
    cfg.overlap_tile = 512;
    cfg.panel_buffers = 3;
    cfg.panel_comm_mode = distributed_qr_col_blockcyclic::PanelCommMode::Broadcast;
    cfg.use_compact_local_gemm = false;
    cfg.case_name = "broadcast_three_panel_buffers";
    RunFactorizedAEqualsQtA0<float>(cfg, 1.5e-3, 1.5e-3);
}

TEST(DistBlockedQrColBlockCyclicCorrectnessTest,
     BlockBroadcastOnceBlockCols2NbTiledSegmentedFloat) {
    ColBlockCyclicCorrectnessConfig cfg;
    cfg.m = 8192;
    cfg.n = 8192;
    cfg.nb = 1024;
    cfg.block_cols = 2048;
    cfg.overlap_tile = 1024;
    cfg.panel_comm_mode = distributed_qr_col_blockcyclic::PanelCommMode::Broadcast;
    cfg.broadcast_mode = distributed_qr_col_blockcyclic::BroadcastMode::Block;
    cfg.wy_storage_mode = distributed_qr_col_blockcyclic::PersistentWyStorageMode::Compact;
    cfg.use_compact_local_gemm = false;
    cfg.case_name = "block_broadcast_once_block_cols_2048_tiled_segmented";
    RunFactorizedAEqualsQtA0<float>(cfg, 1.5e-3, 1.5e-3);
}

TEST(DistBlockedQrColBlockCyclicCorrectnessTest,
     BlockYTBroadcastOnceBlockCols2NbTiledSegmentedFloat) {
    ColBlockCyclicCorrectnessConfig cfg;
    cfg.m = 8192;
    cfg.n = 8192;
    cfg.nb = 1024;
    cfg.block_cols = 2048;
    cfg.overlap_tile = 1024;
    cfg.panel_comm_mode = distributed_qr_col_blockcyclic::PanelCommMode::Broadcast;
    cfg.broadcast_mode = distributed_qr_col_blockcyclic::BroadcastMode::BlockYT;
    cfg.wy_storage_mode = distributed_qr_col_blockcyclic::PersistentWyStorageMode::Compact;
    cfg.use_compact_local_gemm = false;
    cfg.case_name = "block_yt_broadcast_once_block_cols_2048_tiled_segmented";
    RunFactorizedAEqualsQtA0<float>(cfg, 1.5e-3, 1.5e-3);
}

TEST(DistBlockedQrColBlockCyclicCorrectnessTest, ExplicitQBroadcastBlockCols2NbFloat) {
    ColBlockCyclicCorrectnessConfig cfg;
    cfg.m = 1024;
    cfg.n = 512;
    cfg.nb = 128;
    cfg.block_cols = 256;
    cfg.overlap_tile = 128;
    cfg.panel_comm_mode = distributed_qr_col_blockcyclic::PanelCommMode::Broadcast;
    cfg.wy_storage_mode = distributed_qr_col_blockcyclic::PersistentWyStorageMode::Compact;
    cfg.use_compact_local_gemm = false;
    cfg.case_name = "explicit_q_broadcast_block_cols_256";
    RunExplicitQReconstructsA0<float>(cfg, 2.5e-3, 2.5e-3);
}

TEST(DistBlockedQrColBlockCyclicCorrectnessTest, ExplicitQBlockYTBroadcastBlockCols2NbFloat) {
    ColBlockCyclicCorrectnessConfig cfg;
    cfg.m = 1024;
    cfg.n = 512;
    cfg.nb = 128;
    cfg.block_cols = 256;
    cfg.overlap_tile = 128;
    cfg.panel_comm_mode = distributed_qr_col_blockcyclic::PanelCommMode::Broadcast;
    cfg.broadcast_mode = distributed_qr_col_blockcyclic::BroadcastMode::BlockYT;
    cfg.wy_storage_mode = distributed_qr_col_blockcyclic::PersistentWyStorageMode::Compact;
    cfg.use_compact_local_gemm = false;
    cfg.case_name = "explicit_q_block_yt_broadcast_block_cols_256";
    RunExplicitQReconstructsA0<float>(cfg, 2.5e-3, 2.5e-3);
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    spdlog::set_level(spdlog::level::info);

    MpiCudaEnv env = init_mpi_and_bind_gpu(&argc, &argv);
    init(&env);
    if (!init_nccl_comm(&env)) {
        finalize_mpi_if_needed(env);
        return 1;
    }

    g_env = &env;
    const int local_ret = RUN_ALL_TESTS();

    int local_fail = (local_ret == 0) ? 0 : 1;
    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    finalize_nccl_if_needed(&env);
    finalize_mpi_if_needed(env);
    return global_fail;
}
