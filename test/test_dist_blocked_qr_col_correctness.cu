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

#include "components/distributed_blocked_qr_col.cuh"
#include "components/resourse_initial.cuh"
#include "utils/cublas_gemm_traits.cuh"

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

    distributed_qr_col::AssertCuda(
        cudaMemcpy2D(h_A.data(), static_cast<size_t>(rows) * sizeof(float), d_A,
                     static_cast<size_t>(lda) * sizeof(float),
                     static_cast<size_t>(rows) * sizeof(float), cols, cudaMemcpyDeviceToHost),
        "cudaMemcpy2D D2H local matrix");
    return h_A;
}

void ApplyAllOuterPanelsQTToA(float* d_A,
                              int m,
                              int n,
                              int nb,
                              cublasHandle_t cublas_handle,
                              ncclComm_t nccl_comm,
                              const distributed_qr_col::ColPartition& part,
                              const float* d_W_local,
                              const float* d_Y_local,
                              float* d_panel_w,
                              float* d_panel_y,
                              float* d_tmp,
                              cudaStream_t compute_stream) {
    const int lda_local = std::max(m, 1);
    const int tile_cols = nb;
    const int total_panels = n / distributed_qr_col::kPanelWidth;
    const ncclDataType_t nccl_type = distributed_qr_col::NcclType<float>();

    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;

    distributed_qr_col::AssertCublas(cublasSetStream(cublas_handle, compute_stream),
                                     "cublasSetStream(compute_stream)");

    for (int panel_col = 0; panel_col < n; panel_col += distributed_qr_col::kPanelWidth) {
        const int panel_idx = panel_col / distributed_qr_col::kPanelWidth;
        const int owner = distributed_qr_col::OwnerOfPanel(panel_idx, total_panels, part.world_size);
        const int local_panel_col = panel_col - part.col_start;
        const bool owner_has_panel =
            (owner == part.rank) && (local_panel_col >= 0) &&
            (local_panel_col + distributed_qr_col::kPanelWidth <= part.local_cols);
        const int panel_rows = m - panel_col;

        if (owner_has_panel) {
            distributed_qr_col::AssertCuda(
                cudaMemcpy2DAsync(d_panel_w + panel_col, static_cast<size_t>(m) * sizeof(float),
                                  d_W_local + static_cast<size_t>(local_panel_col) * lda_local +
                                      panel_col,
                                  static_cast<size_t>(lda_local) * sizeof(float),
                                  static_cast<size_t>(panel_rows) * sizeof(float),
                                  distributed_qr_col::kPanelWidth, cudaMemcpyDeviceToDevice,
                                  compute_stream),
                "cudaMemcpy2DAsync local panel W -> scratch");
            distributed_qr_col::AssertCuda(
                cudaMemcpy2DAsync(d_panel_y + panel_col, static_cast<size_t>(m) * sizeof(float),
                                  d_Y_local + static_cast<size_t>(local_panel_col) * lda_local +
                                      panel_col,
                                  static_cast<size_t>(lda_local) * sizeof(float),
                                  static_cast<size_t>(panel_rows) * sizeof(float),
                                  distributed_qr_col::kPanelWidth, cudaMemcpyDeviceToDevice,
                                  compute_stream),
                "cudaMemcpy2DAsync local panel Y -> scratch");
        }

        distributed_qr_col::AssertNccl(
            ncclBroadcast(d_panel_w + panel_col, d_panel_w + panel_col,
                          static_cast<size_t>(panel_rows) * distributed_qr_col::kPanelWidth,
                          nccl_type, owner, nccl_comm, compute_stream),
            "ncclBroadcast panel W");
        distributed_qr_col::AssertNccl(
            ncclBroadcast(d_panel_y + panel_col, d_panel_y + panel_col,
                          static_cast<size_t>(panel_rows) * distributed_qr_col::kPanelWidth,
                          nccl_type, owner, nccl_comm, compute_stream),
            "ncclBroadcast panel Y");

        const int local_begin_global = std::max(panel_col, part.col_start);
        if (local_begin_global >= part.col_end) {
            continue;
        }

        const int local_begin = local_begin_global - part.col_start;
        const int cols_local = part.col_end - local_begin_global;
        float* a_trail = d_A + static_cast<size_t>(local_begin) * lda_local;

        for (int col0 = 0; col0 < cols_local; col0 += tile_cols) {
            const int width = std::min(tile_cols, cols_local - col0);
            float* a_tile = a_trail + static_cast<size_t>(col0) * lda_local;
            float* a_tile_sub = a_tile + panel_col;

            distributed_qr_col::AssertCublas(
                CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                              distributed_qr_col::kPanelWidth, width, panel_rows,
                                              &one, d_panel_w + panel_col, m, a_tile_sub, lda_local,
                                              &zero, d_tmp,
                                              distributed_qr_col::kPanelWidth),
                "tmp = W^T * A_tile");

            distributed_qr_col::AssertCublas(
                CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_rows,
                                              width, distributed_qr_col::kPanelWidth, &minus_one,
                                              d_panel_y + panel_col, m, d_tmp,
                                              distributed_qr_col::kPanelWidth, &one, a_tile_sub,
                                              lda_local),
                "A_tile -= Y * tmp");
        }
    }
}

TEST(DistBlockedQrColCorrectnessTest, FactorizedAEqualsQtA0) {
    ASSERT_NE(g_env, nullptr);
    const auto& env = *g_env;

    constexpr int m = 2048;
    constexpr int n = 512;
    constexpr int nb = 256;

    ASSERT_EQ(n % distributed_qr_col::kPanelWidth, 0);
    ASSERT_EQ(nb % distributed_qr_col::kPanelWidth, 0);

    const auto part = distributed_qr_col::MakeColPartition(n, env.size, env.rank);
    const int lda_local = std::max(m, 1);
    const size_t elems_alloc =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(part.local_cols, 1));
    const size_t elems_used = static_cast<size_t>(m) * static_cast<size_t>(part.local_cols);

    float* d_A0 = nullptr;
    float* d_Afact = nullptr;
    float* d_Aqt = nullptr;
    float* d_W = nullptr;
    float* d_Y = nullptr;

    distributed_qr_col::AssertCuda(cudaMalloc(&d_A0, elems_alloc * sizeof(float)),
                                   "cudaMalloc d_A0");
    distributed_qr_col::AssertCuda(cudaMalloc(&d_Afact, elems_alloc * sizeof(float)),
                                   "cudaMalloc d_Afact");
    distributed_qr_col::AssertCuda(cudaMalloc(&d_Aqt, elems_alloc * sizeof(float)),
                                   "cudaMalloc d_Aqt");
    distributed_qr_col::AssertCuda(cudaMalloc(&d_W, elems_alloc * sizeof(float)), "cudaMalloc d_W");
    distributed_qr_col::AssertCuda(cudaMalloc(&d_Y, elems_alloc * sizeof(float)), "cudaMalloc d_Y");

    FillDeviceRandom(d_A0, elems_used, 20260220ULL + static_cast<unsigned long long>(env.rank));

    distributed_qr_col::AssertCuda(
        cudaMemcpy(d_Afact, d_A0, elems_alloc * sizeof(float), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Afact <- d_A0");
    distributed_qr_col::AssertCuda(
        cudaMemcpy(d_Aqt, d_A0, elems_alloc * sizeof(float), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Aqt <- d_A0");
    distributed_qr_col::AssertCuda(cudaMemset(d_W, 0, elems_alloc * sizeof(float)),
                                   "cudaMemset d_W");
    distributed_qr_col::AssertCuda(cudaMemset(d_Y, 0, elems_alloc * sizeof(float)),
                                   "cudaMemset d_Y");

    const int tile_cols = nb;
    distributed_qr_col::DistributedQrColWorkspace<float> ws{};
    ws.tsqr_work_panel_elems = std::max(tsqr_work_elems<float>(m), static_cast<size_t>(1));
    ws.tmp_elems = static_cast<size_t>(distributed_qr_col::kPanelWidth) * tile_cols;

    distributed_qr_col::AssertCuda(
        cudaMalloc(&ws.d_r_panel,
                   static_cast<size_t>(distributed_qr_col::kPanelWidth) *
                       distributed_qr_col::kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_panel");
    distributed_qr_col::AssertCuda(
        cudaMalloc(&ws.d_tsqr_work_panel, ws.tsqr_work_panel_elems * sizeof(float)),
        "cudaMalloc ws.d_tsqr_work_panel");
    distributed_qr_col::AssertCuda(
        cudaMalloc(&ws.d_panel_w,
                   static_cast<size_t>(m) * distributed_qr_col::kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_panel_w");
    distributed_qr_col::AssertCuda(
        cudaMalloc(&ws.d_panel_y,
                   static_cast<size_t>(m) * distributed_qr_col::kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_panel_y");
    distributed_qr_col::AssertCuda(cudaMalloc(&ws.d_tmp, ws.tmp_elems * sizeof(float)),
                                   "cudaMalloc ws.d_tmp");

    float* d_panel_w = nullptr;
    float* d_panel_y = nullptr;
    float* d_tmp = nullptr;
    distributed_qr_col::AssertCuda(
        cudaMalloc(&d_panel_w,
                   static_cast<size_t>(m) * distributed_qr_col::kPanelWidth * sizeof(float)),
        "cudaMalloc d_panel_w");
    distributed_qr_col::AssertCuda(
        cudaMalloc(&d_panel_y,
                   static_cast<size_t>(m) * distributed_qr_col::kPanelWidth * sizeof(float)),
        "cudaMalloc d_panel_y");
    distributed_qr_col::AssertCuda(
        cudaMalloc(&d_tmp, static_cast<size_t>(distributed_qr_col::kPanelWidth) * nb * sizeof(float)),
        "cudaMalloc d_tmp");

    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;
    distributed_qr_col::AssertCuda(
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking),
        "cudaStreamCreate compute_stream");
    distributed_qr_col::AssertCuda(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking),
                                   "cudaStreamCreate comm_stream");

    cublasHandle_t cublas_handle = nullptr;
    distributed_qr_col::AssertCublas(cublasCreate(&cublas_handle), "cublasCreate");
    distributed_qr_col::AssertCublas(cublasSetStream(cublas_handle, compute_stream),
                                     "cublasSetStream compute_stream");

    distributed_qr_col::distributed_blocked_qr_factorize_col<float>(
        cublas_handle, env.nccl_comm, part, m, n, nb, d_Afact, lda_local, d_W, d_Y, &ws,
        compute_stream, comm_stream);

    distributed_qr_col::AssertCuda(cudaStreamSynchronize(compute_stream),
                                   "cudaStreamSynchronize factorize compute");
    distributed_qr_col::AssertCuda(cudaStreamSynchronize(comm_stream),
                                   "cudaStreamSynchronize factorize comm");

    ApplyAllOuterPanelsQTToA(d_Aqt, m, n, nb, cublas_handle, env.nccl_comm, part, d_W, d_Y,
                             d_panel_w, d_panel_y, d_tmp, compute_stream);

    distributed_qr_col::AssertCuda(cudaStreamSynchronize(compute_stream),
                                   "cudaStreamSynchronize applyQT compute");

    const auto h_Afact = CopyLocalMatrixToHost(d_Afact, lda_local, m, part.local_cols);
    const auto h_Aqt = CopyLocalMatrixToHost(d_Aqt, lda_local, m, part.local_cols);

    double local_upper_diff_sq = 0.0;
    double local_upper_base_sq = 0.0;
    double local_lower_sq = 0.0;

    for (int col = 0; col < part.local_cols; ++col) {
        const int global_col = part.col_start + col;
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
        spdlog::info("Col correctness: rel(upper(Afact), upper(Q^T*A0))={:.3e}, lower_ratio={:.3e}",
                     rel_upper, lower_ratio);
    }

    EXPECT_LT(rel_upper, 8e-4);
    EXPECT_LT(lower_ratio, 8e-4);

    distributed_qr_col::AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    distributed_qr_col::AssertCuda(cudaStreamDestroy(compute_stream), "cudaStreamDestroy compute");
    distributed_qr_col::AssertCuda(cudaStreamDestroy(comm_stream), "cudaStreamDestroy comm");

    distributed_qr_col::AssertCuda(cudaFree(ws.d_r_panel), "cudaFree ws.d_r_panel");
    distributed_qr_col::AssertCuda(cudaFree(ws.d_tsqr_work_panel), "cudaFree ws.d_tsqr_work_panel");
    distributed_qr_col::AssertCuda(cudaFree(ws.d_panel_w), "cudaFree ws.d_panel_w");
    distributed_qr_col::AssertCuda(cudaFree(ws.d_panel_y), "cudaFree ws.d_panel_y");
    distributed_qr_col::AssertCuda(cudaFree(ws.d_tmp), "cudaFree ws.d_tmp");

    distributed_qr_col::AssertCuda(cudaFree(d_panel_w), "cudaFree d_panel_w");
    distributed_qr_col::AssertCuda(cudaFree(d_panel_y), "cudaFree d_panel_y");
    distributed_qr_col::AssertCuda(cudaFree(d_tmp), "cudaFree d_tmp");

    distributed_qr_col::AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    distributed_qr_col::AssertCuda(cudaFree(d_Afact), "cudaFree d_Afact");
    distributed_qr_col::AssertCuda(cudaFree(d_Aqt), "cudaFree d_Aqt");
    distributed_qr_col::AssertCuda(cudaFree(d_W), "cudaFree d_W");
    distributed_qr_col::AssertCuda(cudaFree(d_Y), "cudaFree d_Y");
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
