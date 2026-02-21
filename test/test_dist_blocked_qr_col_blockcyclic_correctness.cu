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
#include "components/resourse_initial.cuh"

namespace {

MpiCudaEnv* g_env = nullptr;

void AssertCurand(curandStatus_t status, const char* context) {
    ASSERT_EQ(status, CURAND_STATUS_SUCCESS) << context;
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

std::vector<float> CopyLocalMatrixToHost(const float* d_A, int lda, int rows, int cols) {
    std::vector<float> h_A(std::max(rows * cols, 1), 0.0f);
    if (rows == 0 || cols == 0) {
        return h_A;
    }

    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMemcpy2D(h_A.data(), static_cast<size_t>(rows) * sizeof(float), d_A,
                     static_cast<size_t>(lda) * sizeof(float),
                     static_cast<size_t>(rows) * sizeof(float), cols, cudaMemcpyDeviceToHost),
        "cudaMemcpy2D D2H local matrix");
    return h_A;
}

void ApplyAllOuterPanelsQTToA(
    float* d_A,
    int m,
    int n,
    int nb,
    cublasHandle_t cublas_handle,
    ncclComm_t nccl_comm,
    const distributed_qr_col_blockcyclic::ColBlockCyclicPartition& part,
    const float* d_W_local,
    const float* d_Y_local,
    distributed_qr_col_blockcyclic::DistributedQrColBlockCyclicWorkspace<float>* ws,
    cudaStream_t compute_stream) {
    const int lda_local = std::max(m, 1);
    const int tile_cols = nb;
    const ncclDataType_t nccl_type = distributed_qr_col_blockcyclic::NcclType<float>();

    distributed_qr_col_blockcyclic::AssertCublas(cublasSetStream(cublas_handle, compute_stream),
                                                 "cublasSetStream(compute_stream)");
    ASSERT_NE(ws, nullptr);

    for (int block_begin = 0; block_begin < n; block_begin += nb) {
        const int block_end = std::min(block_begin + nb, n);
        const int kb = block_end - block_begin;
        const int rows = m - block_begin;

        for (int inner = block_begin; inner < block_end;
             inner += distributed_qr_col_blockcyclic::kPanelWidth) {
            const int owner = distributed_qr_col_blockcyclic::OwnerOfPanel(inner, part);
            const int local_panel_col =
                distributed_qr_col_blockcyclic::LocalColOffset(part, inner);
            const bool owner_has_panel = (owner == part.rank) && (local_panel_col >= 0);
            const int block_col_off = inner - block_begin;

            if (owner_has_panel) {
                distributed_qr_col_blockcyclic::AssertCuda(
                    cudaMemcpy2DAsync(
                        ws->d_pack_w, static_cast<size_t>(rows) * sizeof(float),
                        d_W_local + static_cast<size_t>(local_panel_col) * lda_local + block_begin,
                        static_cast<size_t>(lda_local) * sizeof(float),
                        static_cast<size_t>(rows) * sizeof(float),
                        distributed_qr_col_blockcyclic::kPanelWidth, cudaMemcpyDeviceToDevice,
                        compute_stream),
                    "cudaMemcpy2DAsync local W -> pack");
                distributed_qr_col_blockcyclic::AssertCuda(
                    cudaMemcpy2DAsync(
                        ws->d_pack_y, static_cast<size_t>(rows) * sizeof(float),
                        d_Y_local + static_cast<size_t>(local_panel_col) * lda_local + block_begin,
                        static_cast<size_t>(lda_local) * sizeof(float),
                        static_cast<size_t>(rows) * sizeof(float),
                        distributed_qr_col_blockcyclic::kPanelWidth, cudaMemcpyDeviceToDevice,
                        compute_stream),
                    "cudaMemcpy2DAsync local Y -> pack");
            }

            distributed_qr_col_blockcyclic::AssertNccl(
                ncclBroadcast(ws->d_pack_w, ws->d_pack_w,
                              static_cast<size_t>(rows) *
                                  distributed_qr_col_blockcyclic::kPanelWidth,
                              nccl_type, owner, nccl_comm, compute_stream),
                "ncclBroadcast panel W(pack)");
            distributed_qr_col_blockcyclic::AssertNccl(
                ncclBroadcast(ws->d_pack_y, ws->d_pack_y,
                              static_cast<size_t>(rows) *
                                  distributed_qr_col_blockcyclic::kPanelWidth,
                              nccl_type, owner, nccl_comm, compute_stream),
                "ncclBroadcast panel Y(pack)");

            float* dst_w = ws->d_block_w + static_cast<size_t>(block_begin) +
                           static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
            float* dst_y = ws->d_block_y + static_cast<size_t>(block_begin) +
                           static_cast<size_t>(block_col_off) * static_cast<size_t>(m);
            distributed_qr_col_blockcyclic::AssertCuda(
                cudaMemcpy2DAsync(dst_w, static_cast<size_t>(m) * sizeof(float), ws->d_pack_w,
                                  static_cast<size_t>(rows) * sizeof(float),
                                  static_cast<size_t>(rows) * sizeof(float),
                                  distributed_qr_col_blockcyclic::kPanelWidth,
                                  cudaMemcpyDeviceToDevice, compute_stream),
                "cudaMemcpy2DAsync pack_w -> block_w");
            distributed_qr_col_blockcyclic::AssertCuda(
                cudaMemcpy2DAsync(dst_y, static_cast<size_t>(m) * sizeof(float), ws->d_pack_y,
                                  static_cast<size_t>(rows) * sizeof(float),
                                  static_cast<size_t>(rows) * sizeof(float),
                                  distributed_qr_col_blockcyclic::kPanelWidth,
                                  cudaMemcpyDeviceToDevice, compute_stream),
                "cudaMemcpy2DAsync pack_y -> block_y");
        }

        distributed_qr_col_blockcyclic::ForEachLocalSegment(
            part, block_begin, n, [&](int seg_begin, int seg_end, int local_begin) {
                const int cols_local = seg_end - seg_begin;
                float* a_trail = d_A + static_cast<size_t>(local_begin) * lda_local;
                distributed_qr_col_blockcyclic::block_update_tile_pipeline<float>(
                    cublas_handle, compute_stream, block_begin, rows, kb, cols_local, tile_cols,
                    ws->d_block_w + static_cast<size_t>(block_begin),
                    ws->d_block_y + static_cast<size_t>(block_begin), m, a_trail, lda_local,
                    ws->d_tmp0, ws->d_tmp1);
            });
    }
}

TEST(DistBlockedQrColBlockCyclicCorrectnessTest, FactorizedAEqualsQtA0) {
    ASSERT_NE(g_env, nullptr);
    const auto& env = *g_env;

    constexpr int m = 2048;
    constexpr int n = 512;
    constexpr int nb = 256;
    constexpr int block_cols = nb;

    ASSERT_EQ(n % distributed_qr_col_blockcyclic::kPanelWidth, 0);
    ASSERT_EQ(nb % distributed_qr_col_blockcyclic::kPanelWidth, 0);

    const auto part = distributed_qr_col_blockcyclic::MakeColBlockCyclicPartition(
        n, block_cols, env.size, env.rank);
    const int lda_local = std::max(m, 1);
    const size_t elems_alloc =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(part.local_cols, 1));
    const size_t elems_used = static_cast<size_t>(m) * static_cast<size_t>(part.local_cols);

    float* d_A0 = nullptr;
    float* d_Afact = nullptr;
    float* d_Aqt = nullptr;
    float* d_W = nullptr;
    float* d_Y = nullptr;

    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_A0, elems_alloc * sizeof(float)),
                                               "cudaMalloc d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_Afact, elems_alloc * sizeof(float)),
                                               "cudaMalloc d_Afact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_Aqt, elems_alloc * sizeof(float)),
                                               "cudaMalloc d_Aqt");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_W, elems_alloc * sizeof(float)),
                                               "cudaMalloc d_W");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_Y, elems_alloc * sizeof(float)),
                                               "cudaMalloc d_Y");

    FillDeviceRandom(d_A0, elems_used, 20260220ULL + static_cast<unsigned long long>(env.rank));

    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMemcpy(d_Afact, d_A0, elems_alloc * sizeof(float), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Afact <- d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMemcpy(d_Aqt, d_A0, elems_alloc * sizeof(float), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Aqt <- d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMemset(d_W, 0, elems_alloc * sizeof(float)),
                                               "cudaMemset d_W");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMemset(d_Y, 0, elems_alloc * sizeof(float)),
                                               "cudaMemset d_Y");

    const int tile_cols = nb;
    distributed_qr_col_blockcyclic::DistributedQrColBlockCyclicWorkspace<float> ws{};
    ws.tsqr_work_panel_elems = std::max(tsqr_work_elems<float>(m), static_cast<size_t>(1));
    ws.pack_elems =
        static_cast<size_t>(m) * static_cast<size_t>(distributed_qr_col_blockcyclic::kPanelWidth);
    ws.block_storage_elems = static_cast<size_t>(m) * static_cast<size_t>(nb);
    ws.block_compact_elems =
        static_cast<size_t>(nb + distributed_qr_col_blockcyclic::kPanelWidth) *
        static_cast<size_t>(nb);
    ws.tmp_elems = static_cast<size_t>(nb) * static_cast<size_t>(tile_cols);

    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_r_panel, static_cast<size_t>(distributed_qr_col_blockcyclic::kPanelWidth) *
                                      distributed_qr_col_blockcyclic::kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_panel");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_tsqr_work_panel, ws.tsqr_work_panel_elems * sizeof(float)),
        "cudaMalloc ws.d_tsqr_work_panel");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_pack_w, ws.pack_elems * sizeof(float)),
        "cudaMalloc ws.d_pack_w");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_pack_y, ws.pack_elems * sizeof(float)),
        "cudaMalloc ws.d_pack_y");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_block_w, ws.block_storage_elems * sizeof(float)),
        "cudaMalloc ws.d_block_w");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_block_y, ws.block_storage_elems * sizeof(float)),
        "cudaMalloc ws.d_block_y");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_block_w_compact, ws.block_compact_elems * sizeof(float)),
        "cudaMalloc ws.d_block_w_compact");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_block_y_compact, ws.block_compact_elems * sizeof(float)),
        "cudaMalloc ws.d_block_y_compact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&ws.d_tmp0, ws.tmp_elems * sizeof(float)),
                                               "cudaMalloc ws.d_tmp0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&ws.d_tmp1, ws.tmp_elems * sizeof(float)),
                                               "cudaMalloc ws.d_tmp1");

    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking),
        "cudaStreamCreate compute_stream");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking),
        "cudaStreamCreate comm_stream");

    cublasHandle_t cublas_handle = nullptr;
    distributed_qr_col_blockcyclic::AssertCublas(cublasCreate(&cublas_handle), "cublasCreate");
    distributed_qr_col_blockcyclic::AssertCublas(
        cublasSetStream(cublas_handle, compute_stream), "cublasSetStream compute_stream");

    distributed_qr_col_blockcyclic::distributed_blocked_qr_factorize_col_blockcyclic<float>(
        cublas_handle, env.nccl_comm, part, m, n, nb, d_Afact, lda_local, d_W, d_Y, &ws,
        compute_stream, comm_stream);

    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(compute_stream),
                                               "cudaStreamSynchronize factorize compute");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(comm_stream),
                                               "cudaStreamSynchronize factorize comm");

    ApplyAllOuterPanelsQTToA(d_Aqt, m, n, nb, cublas_handle, env.nccl_comm, part, d_W, d_Y, &ws,
                             compute_stream);

    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(compute_stream),
                                               "cudaStreamSynchronize applyQT compute");

    const auto h_Afact = CopyLocalMatrixToHost(d_Afact, lda_local, m, part.local_cols);
    const auto h_Aqt = CopyLocalMatrixToHost(d_Aqt, lda_local, m, part.local_cols);

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

        for (int row = 0; row < m; ++row) {
            const size_t idx =
                static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(m);
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
            "Col blockcyclic correctness: rel(upper(Afact), upper(Q^T*A0))={:.3e}, "
            "lower_ratio={:.3e}",
            rel_upper, lower_ratio);
    }

    EXPECT_LT(rel_upper, 8e-4);
    EXPECT_LT(lower_ratio, 8e-4);

    distributed_qr_col_blockcyclic::AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamDestroy(compute_stream),
                                               "cudaStreamDestroy compute");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamDestroy(comm_stream),
                                               "cudaStreamDestroy comm");

    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_r_panel), "cudaFree ws.d_r_panel");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_tsqr_work_panel),
                                               "cudaFree ws.d_tsqr_work_panel");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_pack_w), "cudaFree ws.d_pack_w");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_pack_y), "cudaFree ws.d_pack_y");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_block_w), "cudaFree ws.d_block_w");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_block_y), "cudaFree ws.d_block_y");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_block_w_compact),
                                               "cudaFree ws.d_block_w_compact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_block_y_compact),
                                               "cudaFree ws.d_block_y_compact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_tmp0), "cudaFree ws.d_tmp0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_tmp1), "cudaFree ws.d_tmp1");

    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_Afact), "cudaFree d_Afact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_Aqt), "cudaFree d_Aqt");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_W), "cudaFree d_W");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_Y), "cudaFree d_Y");
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
