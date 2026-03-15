#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverMp.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "components/resourse_initial.cuh"

namespace {

MpiCudaEnv* g_env = nullptr;

void AssertCuda(cudaError_t status, const char* context) {
    ASSERT_EQ(status, cudaSuccess) << context << ": " << cudaGetErrorString(status);
}

void AssertCusolver(cusolverStatus_t status, const char* context) {
    ASSERT_EQ(status, CUSOLVER_STATUS_SUCCESS) << context << ": " << static_cast<int>(status);
}

template <typename T>
cudaDataType CudaDataTypeValue() {
    if constexpr (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    }
    return CUDA_R_64F;
}

template <typename T>
const char* DataTypeString() {
    if constexpr (std::is_same_v<T, float>) {
        return "float";
    }
    return "double";
}

int ResolveBalancedGridRows(int world_size) {
    int rows = static_cast<int>(std::sqrt(static_cast<double>(world_size)));
    while (rows > 1 && world_size % rows != 0) {
        --rows;
    }
    return std::max(rows, 1);
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
void RunGeqrfOrmqrIdentityEndToEnd(int m, int n, int grid_block_size, double ortho_tol,
                                   double recon_tol) {
    ASSERT_NE(g_env, nullptr);
    const auto& env = *g_env;

    ASSERT_GT(m, 0);
    ASSERT_GT(n, 0);
    ASSERT_GT(grid_block_size, 0);
    ASSERT_GE(m, n);

    const int grid_rows = ResolveBalancedGridRows(env.size);
    const int grid_cols = env.size / grid_rows;

    cudaStream_t stream = nullptr;
    AssertCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    cusolverMpHandle_t mp_handle = nullptr;
    AssertCusolver(cusolverMpCreate(&mp_handle, env.device_id, stream), "cusolverMpCreate");

    cusolverMpGrid_t grid = nullptr;
    AssertCusolver(cusolverMpCreateDeviceGrid(mp_handle, &grid, env.nccl_comm, grid_rows, grid_cols,
                                              CUSOLVERMP_GRID_MAPPING_COL_MAJOR),
                   "cusolverMpCreateDeviceGrid");

    const int proc_row = env.rank % grid_rows;
    const int proc_col = env.rank / grid_rows;

    const int64_t local_rows_a = cusolverMpNUMROC(static_cast<int64_t>(m),
                                                  static_cast<int64_t>(grid_block_size), proc_row, 0,
                                                  grid_rows);
    const int64_t local_cols_a = cusolverMpNUMROC(static_cast<int64_t>(n),
                                                  static_cast<int64_t>(grid_block_size), proc_col, 0,
                                                  grid_cols);
    const int64_t lda_local_a = std::max<int64_t>(1, local_rows_a);
    const size_t local_elems_a =
        static_cast<size_t>(lda_local_a) * static_cast<size_t>(std::max<int64_t>(1, local_cols_a));
    const size_t local_bytes_a = local_elems_a * sizeof(T);

    const int64_t local_rows_q = cusolverMpNUMROC(static_cast<int64_t>(m),
                                                  static_cast<int64_t>(grid_block_size), proc_row, 0,
                                                  grid_rows);
    const int64_t local_cols_q = cusolverMpNUMROC(static_cast<int64_t>(n),
                                                  static_cast<int64_t>(grid_block_size), proc_col, 0,
                                                  grid_cols);
    const int64_t lda_local_q = std::max<int64_t>(1, local_rows_q);
    const size_t local_elems_q =
        static_cast<size_t>(lda_local_q) * static_cast<size_t>(std::max<int64_t>(1, local_cols_q));
    const size_t local_bytes_q = local_elems_q * sizeof(T);

    const int64_t local_tau = cusolverMpNUMROC(static_cast<int64_t>(n),
                                               static_cast<int64_t>(grid_block_size), proc_col, 0,
                                               grid_cols);
    const size_t tau_elems = static_cast<size_t>(std::max<int64_t>(1, local_tau));
    const size_t tau_bytes = tau_elems * sizeof(T);

    T* d_A0 = nullptr;
    T* d_Afact = nullptr;
    T* d_Q = nullptr;
    T* d_tau = nullptr;
    int* d_info = nullptr;
    void* d_work_geqrf = nullptr;
    void* d_work_ormqr = nullptr;

    AssertCuda(cudaMalloc(&d_A0, local_bytes_a), "cudaMalloc d_A0");
    AssertCuda(cudaMalloc(&d_Afact, local_bytes_a), "cudaMalloc d_Afact");
    AssertCuda(cudaMalloc(&d_Q, local_bytes_q), "cudaMalloc d_Q");
    AssertCuda(cudaMalloc(&d_tau, tau_bytes), "cudaMalloc d_tau");
    AssertCuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc d_info");

    cusolverMpMatrixDescriptor_t descA = nullptr;
    cusolverMpMatrixDescriptor_t descQ = nullptr;
    AssertCusolver(
        cusolverMpCreateMatrixDesc(&descA, grid, CudaDataTypeValue<T>(), static_cast<int64_t>(m),
                                   static_cast<int64_t>(n), static_cast<int64_t>(grid_block_size),
                                   static_cast<int64_t>(grid_block_size), 0, 0, lda_local_a),
        "cusolverMpCreateMatrixDesc descA");
    AssertCusolver(
        cusolverMpCreateMatrixDesc(&descQ, grid, CudaDataTypeValue<T>(), static_cast<int64_t>(m),
                                   static_cast<int64_t>(n), static_cast<int64_t>(grid_block_size),
                                   static_cast<int64_t>(grid_block_size), 0, 0, lda_local_q),
        "cusolverMpCreateMatrixDesc descQ");

    size_t geqrf_work_bytes_device = 0;
    size_t geqrf_work_bytes_host = 0;
    AssertCusolver(cusolverMpGeqrf_bufferSize(
                       mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_Afact, 1, 1,
                       descA, CudaDataTypeValue<T>(), &geqrf_work_bytes_device,
                       &geqrf_work_bytes_host),
                   "cusolverMpGeqrf_bufferSize");
    if (geqrf_work_bytes_device > 0) {
        AssertCuda(cudaMalloc(&d_work_geqrf, geqrf_work_bytes_device), "cudaMalloc d_work_geqrf");
    }
    std::vector<unsigned char> host_work_geqrf(std::max<size_t>(geqrf_work_bytes_host, 1), 0);

    size_t ormqr_work_bytes_device = 0;
    size_t ormqr_work_bytes_host = 0;
    AssertCusolver(cusolverMpOrmqr_bufferSize(
                       mp_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, static_cast<int64_t>(m),
                       static_cast<int64_t>(n), static_cast<int64_t>(n), d_Afact, 1, 1, descA,
                       d_tau, d_Q, 1, 1, descQ, CudaDataTypeValue<T>(), &ormqr_work_bytes_device,
                       &ormqr_work_bytes_host),
                   "cusolverMpOrmqr_bufferSize");
    if (ormqr_work_bytes_device > 0) {
        AssertCuda(cudaMalloc(&d_work_ormqr, ormqr_work_bytes_device), "cudaMalloc d_work_ormqr");
    }
    std::vector<unsigned char> host_work_ormqr(std::max<size_t>(ormqr_work_bytes_host, 1), 0);

    std::vector<T> h_A0;
    std::vector<T> h_Q0;
    if (env.rank == 0) {
        h_A0.resize(static_cast<size_t>(m) * static_cast<size_t>(n));
        h_Q0.resize(static_cast<size_t>(m) * static_cast<size_t>(n), static_cast<T>(0));

        std::mt19937 rng(20260316U);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < h_A0.size(); ++i) {
            h_A0[i] = static_cast<T>(dist(rng));
        }
        for (int col = 0; col < n; ++col) {
            h_Q0[static_cast<size_t>(col) + static_cast<size_t>(col) * static_cast<size_t>(m)] =
                static_cast<T>(1);
        }
    }

    AssertCusolver(cusolverMpMatrixScatterH2D(
                       mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_A0, 1, 1,
                       descA, 0, (env.rank == 0) ? h_A0.data() : nullptr, static_cast<int64_t>(m)),
                   "cusolverMpMatrixScatterH2D A0");
    AssertCusolver(cusolverMpMatrixScatterH2D(
                       mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_Q, 1, 1,
                       descQ, 0, (env.rank == 0) ? h_Q0.data() : nullptr, static_cast<int64_t>(m)),
                   "cusolverMpMatrixScatterH2D Q0");
    AssertCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize scatter");

    AssertCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info geqrf");
    AssertCuda(cudaMemcpyAsync(d_Afact, d_A0, local_bytes_a, cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync Afact <- A0");
    AssertCusolver(cusolverMpGeqrf(
                       mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_Afact, 1, 1,
                       descA, d_tau, CudaDataTypeValue<T>(), d_work_geqrf,
                       geqrf_work_bytes_device, host_work_geqrf.data(), geqrf_work_bytes_host,
                       d_info),
                   "cusolverMpGeqrf");
    AssertCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize geqrf");

    int info = 0;
    AssertCuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy d_info->info geqrf");
    int local_info_abs = std::abs(info);

    AssertCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info ormqr");
    AssertCusolver(cusolverMpOrmqr(
                       mp_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, static_cast<int64_t>(m),
                       static_cast<int64_t>(n), static_cast<int64_t>(n), d_Afact, 1, 1, descA,
                       d_tau, d_Q, 1, 1, descQ, CudaDataTypeValue<T>(), d_work_ormqr,
                       ormqr_work_bytes_device, host_work_ormqr.data(), ormqr_work_bytes_host,
                       d_info),
                   "cusolverMpOrmqr");
    AssertCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize ormqr");
    AssertCuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy d_info->info ormqr");
    local_info_abs = std::max(local_info_abs, std::abs(info));

    int global_info_abs = 0;
    MPI_Allreduce(&local_info_abs, &global_info_abs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    std::vector<T> h_Afact;
    std::vector<T> h_Q;
    if (env.rank == 0) {
        h_Afact.resize(static_cast<size_t>(m) * static_cast<size_t>(n));
        h_Q.resize(static_cast<size_t>(m) * static_cast<size_t>(n));
    }

    if (global_info_abs == 0) {
        AssertCusolver(cusolverMpMatrixGatherD2H(
                           mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_Afact, 1,
                           1, descA, 0, (env.rank == 0) ? h_Afact.data() : nullptr,
                           static_cast<int64_t>(m)),
                       "cusolverMpMatrixGatherD2H Afact");
        AssertCusolver(cusolverMpMatrixGatherD2H(
                           mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_Q, 1, 1,
                           descQ, 0, (env.rank == 0) ? h_Q.data() : nullptr,
                           static_cast<int64_t>(m)),
                       "cusolverMpMatrixGatherD2H Q");
        AssertCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize gather");
    }

    double ortho_err = std::numeric_limits<double>::infinity();
    double recon_err = std::numeric_limits<double>::infinity();
    if (env.rank == 0 && global_info_abs == 0) {
        const auto h_R = ExtractUpperRToDouble(h_Afact, m, n);
        ortho_err = ExplicitQOrthogonalityError(h_Q, m, n);
        recon_err = ExplicitQReconstructionError(h_A0, h_Q, h_R, m, n);
        spdlog::info("cuSOLVERMp GEQRF+ORMQR end2end ({}) m={} n={} nb={} grid={}x{}: orth_err="
                         "{:.3e}, recon_err={:.3e}",
                     DataTypeString<T>(), m, n, grid_block_size, grid_rows, grid_cols, ortho_err,
                     recon_err);
    }
    ASSERT_EQ(MPI_Bcast(&ortho_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), MPI_SUCCESS);
    ASSERT_EQ(MPI_Bcast(&recon_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), MPI_SUCCESS);

    EXPECT_EQ(global_info_abs, 0);
    EXPECT_LT(ortho_err, ortho_tol);
    EXPECT_LT(recon_err, recon_tol);

    AssertCusolver(cusolverMpDestroyMatrixDesc(descQ), "cusolverMpDestroyMatrixDesc descQ");
    AssertCusolver(cusolverMpDestroyMatrixDesc(descA), "cusolverMpDestroyMatrixDesc descA");
    AssertCusolver(cusolverMpDestroyGrid(grid), "cusolverMpDestroyGrid");
    AssertCusolver(cusolverMpDestroy(mp_handle), "cusolverMpDestroy");
    if (d_work_geqrf) {
        AssertCuda(cudaFree(d_work_geqrf), "cudaFree d_work_geqrf");
    }
    if (d_work_ormqr) {
        AssertCuda(cudaFree(d_work_ormqr), "cudaFree d_work_ormqr");
    }
    AssertCuda(cudaFree(d_info), "cudaFree d_info");
    AssertCuda(cudaFree(d_tau), "cudaFree d_tau");
    AssertCuda(cudaFree(d_Q), "cudaFree d_Q");
    AssertCuda(cudaFree(d_Afact), "cudaFree d_Afact");
    AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    AssertCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

TEST(CuSolverMpGeqrfOrmqrEndToEndTest, IdentityProducesThinQFloat) {
    RunGeqrfOrmqrIdentityEndToEnd<float>(1024, 512, 128, 5.0e-4, 5.0e-4);
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
