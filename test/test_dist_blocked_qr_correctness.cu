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

#include "components/distributed_blocked_qr.cuh"
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

    distributed_qr::AssertCuda(
        cudaMemcpy2D(h_A.data(), static_cast<size_t>(rows) * sizeof(float), d_A,
                     static_cast<size_t>(lda) * sizeof(float),
                     static_cast<size_t>(rows) * sizeof(float), cols, cudaMemcpyDeviceToHost),
        "cudaMemcpy2D D2H local matrix");
    return h_A;
}

void ApplyAllOuterBlocksQTToA(float* d_A,
                              int m,
                              int n,
                              int nb,
                              cublasHandle_t cublas_handle,
                              ncclComm_t nccl_comm,
                              const distributed_qr::RowPartition& part,
                              const float* d_W,
                              const float* d_Y,
                              float* d_work0,
                              float* d_work1,
                              cudaStream_t compute_stream,
                              cudaStream_t comm_stream) {
    const int lda_local = std::max(part.local_rows, 1);
    const int tile_cols = nb;

    for (int outer = 0; outer < n; outer += nb) {
        const int end = std::min(outer + nb, n);
        const int kb = end - outer;

        int local_row_off = 0;
        int m_sub_local = 0;
        distributed_qr::LocalSubRows(outer, m, part, &local_row_off, &m_sub_local);

        const float* w_big = d_W + static_cast<size_t>(local_row_off) +
                             static_cast<size_t>(outer) * static_cast<size_t>(lda_local);
        const float* y_big = d_Y + static_cast<size_t>(local_row_off) +
                             static_cast<size_t>(outer) * static_cast<size_t>(lda_local);
        float* a_sub = d_A + static_cast<size_t>(local_row_off) +
                       static_cast<size_t>(outer) * static_cast<size_t>(lda_local);

        const int n_sub = n - outer;
        distributed_qr::allreduce_panel_update_overlap(
            cublas_handle, nccl_comm, compute_stream, comm_stream, m_sub_local, kb, n_sub,
            tile_cols, w_big, lda_local, y_big, lda_local, a_sub, lda_local, d_work0, d_work1);
    }
}

TEST(DistBlockedQrCorrectnessTest, FactorizedAEqualsQtA0) {
    ASSERT_NE(g_env, nullptr);
    const auto& env = *g_env;

    constexpr int m = 2048;
    constexpr int n = 512;
    constexpr int nb = 256;

    ASSERT_EQ(n % distributed_qr::kPanelWidth, 0);
    ASSERT_EQ(nb % distributed_qr::kPanelWidth, 0);

    const auto part = distributed_qr::MakeRowPartition(m, env.size, env.rank);
    const int lda_local = std::max(part.local_rows, 1);
    const size_t elems_alloc = static_cast<size_t>(lda_local) * n;
    const size_t elems_used = static_cast<size_t>(part.local_rows) * n;

    float* d_A0 = nullptr;
    float* d_Afact = nullptr;
    float* d_Aqt = nullptr;
    float* d_W = nullptr;
    float* d_Y = nullptr;

    distributed_qr::AssertCuda(cudaMalloc(&d_A0, elems_alloc * sizeof(float)), "cudaMalloc d_A0");
    distributed_qr::AssertCuda(cudaMalloc(&d_Afact, elems_alloc * sizeof(float)),
                               "cudaMalloc d_Afact");
    distributed_qr::AssertCuda(cudaMalloc(&d_Aqt, elems_alloc * sizeof(float)), "cudaMalloc d_Aqt");
    distributed_qr::AssertCuda(cudaMalloc(&d_W, elems_alloc * sizeof(float)), "cudaMalloc d_W");
    distributed_qr::AssertCuda(cudaMalloc(&d_Y, elems_alloc * sizeof(float)), "cudaMalloc d_Y");

    FillDeviceRandom(d_A0, elems_used, 20260220ULL + static_cast<unsigned long long>(env.rank));

    distributed_qr::AssertCuda(
        cudaMemcpy(d_Afact, d_A0, elems_alloc * sizeof(float), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Afact <- d_A0");
    distributed_qr::AssertCuda(
        cudaMemcpy(d_Aqt, d_A0, elems_alloc * sizeof(float), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_Aqt <- d_A0");
    distributed_qr::AssertCuda(cudaMemset(d_W, 0, elems_alloc * sizeof(float)), "cudaMemset d_W");
    distributed_qr::AssertCuda(cudaMemset(d_Y, 0, elems_alloc * sizeof(float)), "cudaMemset d_Y");

    distributed_qr::DistributedQrWorkspace<float> ws{};
    const int stack_rows = env.size * distributed_qr::kPanelWidth;
    ws.tsqr_work_local_elems =
        std::max(tsqr_work_elems<float>(part.local_rows), static_cast<size_t>(1));
    ws.tsqr_work_stack_elems = std::max(tsqr_work_elems<float>(stack_rows), static_cast<size_t>(1));

    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_r_local, static_cast<size_t>(distributed_qr::kPanelWidth) *
                                      distributed_qr::kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_local");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_r_gathered, static_cast<size_t>(env.size) * distributed_qr::kPanelWidth *
                                         distributed_qr::kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_gathered");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_r_stack,
                   static_cast<size_t>(stack_rows) * distributed_qr::kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_stack");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_r_global, static_cast<size_t>(distributed_qr::kPanelWidth) *
                                       distributed_qr::kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_global");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_tsqr_work_local, ws.tsqr_work_local_elems * sizeof(float)),
        "cudaMalloc ws.d_tsqr_work_local");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_tsqr_work_stack, ws.tsqr_work_stack_elems * sizeof(float)),
        "cudaMalloc ws.d_tsqr_work_stack");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_panel_tmp, static_cast<size_t>(std::max(part.local_rows, 1)) *
                                        distributed_qr::kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_panel_tmp");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_work0, static_cast<size_t>(nb) * nb * sizeof(float)),
        "cudaMalloc ws.d_work0");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_work1, static_cast<size_t>(nb) * nb * sizeof(float)),
        "cudaMalloc ws.d_work1");

    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;
    distributed_qr::AssertCuda(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking),
                               "cudaStreamCreate compute_stream");
    distributed_qr::AssertCuda(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking),
                               "cudaStreamCreate comm_stream");

    cublasHandle_t cublas_handle = nullptr;
    distributed_qr::AssertCublas(cublasCreate(&cublas_handle), "cublasCreate");
    distributed_qr::AssertCublas(cublasSetStream(cublas_handle, compute_stream),
                                 "cublasSetStream compute_stream");

    distributed_qr::distributed_blocked_qr_factorize<float>(cublas_handle, env.nccl_comm, part, m,
                                                            n, nb, d_Afact, lda_local, d_W, d_Y,
                                                            &ws, compute_stream, comm_stream);

    distributed_qr::AssertCuda(cudaStreamSynchronize(compute_stream),
                               "cudaStreamSynchronize factorize compute");
    distributed_qr::AssertCuda(cudaStreamSynchronize(comm_stream),
                               "cudaStreamSynchronize factorize comm");

    ApplyAllOuterBlocksQTToA(d_Aqt, m, n, nb, cublas_handle, env.nccl_comm, part, d_W, d_Y,
                             ws.d_work0, ws.d_work1, compute_stream, comm_stream);

    distributed_qr::AssertCuda(cudaStreamSynchronize(compute_stream),
                               "cudaStreamSynchronize applyQT compute");
    distributed_qr::AssertCuda(cudaStreamSynchronize(comm_stream),
                               "cudaStreamSynchronize applyQT comm");

    const auto h_Afact = CopyLocalMatrixToHost(d_Afact, lda_local, part.local_rows, n);
    const auto h_Aqt = CopyLocalMatrixToHost(d_Aqt, lda_local, part.local_rows, n);

    double local_upper_diff_sq = 0.0;
    double local_upper_base_sq = 0.0;
    double local_lower_sq = 0.0;

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < part.local_rows; ++row) {
            const size_t idx = static_cast<size_t>(row) +
                               static_cast<size_t>(col) * static_cast<size_t>(part.local_rows);
            const double a_fact = static_cast<double>(h_Afact[idx]);
            const double a_qt = static_cast<double>(h_Aqt[idx]);
            const int global_row = part.row_start + row;
            if (global_row <= col) {
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
        spdlog::info("Correctness: rel(upper(Afact), upper(Q^T*A0))={:.3e}, lower_ratio={:.3e}",
                     rel_upper,
                     lower_ratio);
    }

    EXPECT_LT(rel_upper, 5e-4);
    EXPECT_LT(lower_ratio, 5e-4);

    distributed_qr::AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    distributed_qr::AssertCuda(cudaStreamDestroy(compute_stream), "cudaStreamDestroy compute");
    distributed_qr::AssertCuda(cudaStreamDestroy(comm_stream), "cudaStreamDestroy comm");

    distributed_qr::AssertCuda(cudaFree(ws.d_r_local), "cudaFree ws.d_r_local");
    distributed_qr::AssertCuda(cudaFree(ws.d_r_gathered), "cudaFree ws.d_r_gathered");
    distributed_qr::AssertCuda(cudaFree(ws.d_r_stack), "cudaFree ws.d_r_stack");
    distributed_qr::AssertCuda(cudaFree(ws.d_r_global), "cudaFree ws.d_r_global");
    distributed_qr::AssertCuda(cudaFree(ws.d_tsqr_work_local), "cudaFree ws.d_tsqr_work_local");
    distributed_qr::AssertCuda(cudaFree(ws.d_tsqr_work_stack), "cudaFree ws.d_tsqr_work_stack");
    distributed_qr::AssertCuda(cudaFree(ws.d_panel_tmp), "cudaFree ws.d_panel_tmp");
    distributed_qr::AssertCuda(cudaFree(ws.d_work0), "cudaFree ws.d_work0");
    distributed_qr::AssertCuda(cudaFree(ws.d_work1), "cudaFree ws.d_work1");

    distributed_qr::AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    distributed_qr::AssertCuda(cudaFree(d_Afact), "cudaFree d_Afact");
    distributed_qr::AssertCuda(cudaFree(d_Aqt), "cudaFree d_Aqt");
    distributed_qr::AssertCuda(cudaFree(d_W), "cudaFree d_W");
    distributed_qr::AssertCuda(cudaFree(d_Y), "cudaFree d_Y");
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
