#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <mpi.h>
#include <nccl.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#include "components/distributed_blocked_qr.cuh"
#include "components/resourse_initial.cuh"

namespace {

using distributed_qr::kPanelWidth;

void AssertCurand(curandStatus_t status, const char* context) {
    if (status != CURAND_STATUS_SUCCESS) {
        spdlog::error("{}: curand error {}", context, static_cast<int>(status));
        std::exit(1);
    }
}

template <typename T>
void FillDeviceRandom(T* device_data, size_t count, unsigned long long seed) {
    if (count == 0) {
        return;
    }
    curandGenerator_t gen;
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
__global__ void count_non_finite_kernel(const T* data,
                                        size_t count,
                                        unsigned long long* out_count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const T v = data[idx];
    if (!isfinite(static_cast<double>(v))) {
        atomicAdd(out_count, 1ULL);
    }
}

struct Options {
    int m = 16384;
    int n = 1024;
    int nb = 1024;
    int warmup = 1;
    int iters = 3;
    int overlap_tile = 0;
};

Options ParseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            opts.m = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            opts.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nb") == 0 && i + 1 < argc) {
            opts.nb = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opts.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opts.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--overlap_tile") == 0 && i + 1 < argc) {
            opts.overlap_tile = std::atoi(argv[++i]);
        }
    }
    return opts;
}

}  // namespace

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::info);

    auto env = init_mpi_and_bind_gpu(&argc, &argv);
    init(&env);
    if (!init_nccl_comm(&env)) {
        finalize_mpi_if_needed(env);
        return 1;
    }

    const Options opts = ParseArgs(argc, argv);

    if (opts.n > opts.m || opts.n % kPanelWidth != 0 || opts.nb % kPanelWidth != 0 ||
        opts.nb > opts.n) {
        if (env.rank == 0) {
            spdlog::error(
                "Invalid args: require m>=n, n%{}==0, nb%{}==0, nb<=n (got m={} n={} "
                "nb={})",
                kPanelWidth, kPanelWidth, opts.m, opts.n, opts.nb);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }

    const auto part = distributed_qr::MakeRowPartition(opts.m, env.size, env.rank);
    const int lda_local = std::max(part.local_rows, 1);
    const size_t local_elems_alloc = static_cast<size_t>(lda_local) * static_cast<size_t>(opts.n);
    const size_t local_elems_used =
        static_cast<size_t>(part.local_rows) * static_cast<size_t>(opts.n);

    float* d_A0 = nullptr;
    float* d_A = nullptr;
    float* d_W = nullptr;
    float* d_Y = nullptr;

    distributed_qr::AssertCuda(cudaMalloc(&d_A0, local_elems_alloc * sizeof(float)),
                               "cudaMalloc d_A0");
    distributed_qr::AssertCuda(cudaMalloc(&d_A, local_elems_alloc * sizeof(float)),
                               "cudaMalloc d_A");
    distributed_qr::AssertCuda(cudaMalloc(&d_W, local_elems_alloc * sizeof(float)),
                               "cudaMalloc d_W");
    distributed_qr::AssertCuda(cudaMalloc(&d_Y, local_elems_alloc * sizeof(float)),
                               "cudaMalloc d_Y");

    FillDeviceRandom(d_A0, local_elems_used, 2026ULL + static_cast<unsigned long long>(env.rank));

    const int stack_rows = env.size * kPanelWidth;
    const size_t tsqr_work_local_elems =
        std::max(tsqr_work_elems<float>(part.local_rows), static_cast<size_t>(1));
    const size_t tsqr_work_stack_elems =
        std::max(tsqr_work_elems<float>(stack_rows), static_cast<size_t>(1));

    distributed_qr::DistributedQrWorkspace<float> ws{};
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_r_local, static_cast<size_t>(kPanelWidth) * kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_local");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_r_gathered,
                   static_cast<size_t>(env.size) * kPanelWidth * kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_gathered");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_r_stack, static_cast<size_t>(stack_rows) * kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_stack");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_r_global, static_cast<size_t>(kPanelWidth) * kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_global");

    ws.tsqr_work_local_elems = tsqr_work_local_elems;
    ws.tsqr_work_stack_elems = tsqr_work_stack_elems;
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_tsqr_work_local, ws.tsqr_work_local_elems * sizeof(float)),
        "cudaMalloc ws.d_tsqr_work_local");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_tsqr_work_stack, ws.tsqr_work_stack_elems * sizeof(float)),
        "cudaMalloc ws.d_tsqr_work_stack");

    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_panel_tmp,
                   static_cast<size_t>(std::max(part.local_rows, 1)) * kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_panel_tmp");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_work0, static_cast<size_t>(opts.nb) * opts.nb * sizeof(float)),
        "cudaMalloc ws.d_work0");
    distributed_qr::AssertCuda(
        cudaMalloc(&ws.d_work1, static_cast<size_t>(opts.nb) * opts.nb * sizeof(float)),
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
                                 "cublasSetStream(compute_stream)");

    for (int i = 0; i < opts.warmup; ++i) {
        distributed_qr::AssertCuda(cudaMemcpyAsync(d_A, d_A0, local_elems_alloc * sizeof(float),
                                                   cudaMemcpyDeviceToDevice, compute_stream),
                                   "cudaMemcpyAsync warmup A <- A0");
        distributed_qr::AssertCuda(
            cudaMemsetAsync(d_W, 0, local_elems_alloc * sizeof(float), compute_stream),
            "cudaMemsetAsync warmup W");
        distributed_qr::AssertCuda(
            cudaMemsetAsync(d_Y, 0, local_elems_alloc * sizeof(float), compute_stream),
            "cudaMemsetAsync warmup Y");

        distributed_qr::distributed_blocked_qr_factorize<float>(
            cublas_handle, env.nccl_comm, part, opts.m, opts.n, opts.nb, d_A, lda_local, d_W, d_Y,
            &ws, compute_stream, comm_stream, opts.overlap_tile);
        distributed_qr::AssertCuda(cudaStreamSynchronize(compute_stream),
                                   "cudaStreamSynchronize warmup compute");
        distributed_qr::AssertCuda(cudaStreamSynchronize(comm_stream),
                                   "cudaStreamSynchronize warmup comm");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int i = 0; i < opts.iters; ++i) {
        distributed_qr::AssertCuda(cudaMemcpyAsync(d_A, d_A0, local_elems_alloc * sizeof(float),
                                                   cudaMemcpyDeviceToDevice, compute_stream),
                                   "cudaMemcpyAsync timed A <- A0");
        distributed_qr::AssertCuda(
            cudaMemsetAsync(d_W, 0, local_elems_alloc * sizeof(float), compute_stream),
            "cudaMemsetAsync timed W");
        distributed_qr::AssertCuda(
            cudaMemsetAsync(d_Y, 0, local_elems_alloc * sizeof(float), compute_stream),
            "cudaMemsetAsync timed Y");

        distributed_qr::distributed_blocked_qr_factorize<float>(
            cublas_handle, env.nccl_comm, part, opts.m, opts.n, opts.nb, d_A, lda_local, d_W, d_Y,
            &ws, compute_stream, comm_stream, opts.overlap_tile);
    }

    distributed_qr::AssertCuda(cudaStreamSynchronize(compute_stream),
                               "cudaStreamSynchronize timed compute");
    distributed_qr::AssertCuda(cudaStreamSynchronize(comm_stream),
                               "cudaStreamSynchronize timed comm");

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double local_ms = (t1 - t0) * 1000.0 / static_cast<double>(opts.iters);
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    unsigned long long* d_bad = nullptr;
    distributed_qr::AssertCuda(cudaMalloc(&d_bad, sizeof(unsigned long long)), "cudaMalloc d_bad");
    distributed_qr::AssertCuda(cudaMemset(d_bad, 0, sizeof(unsigned long long)),
                               "cudaMemset d_bad");
    if (local_elems_used > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((local_elems_used + threads - 1) / threads);
        count_non_finite_kernel<<<blocks, threads>>>(d_A, local_elems_used, d_bad);
        distributed_qr::AssertCuda(cudaGetLastError(), "count_non_finite_kernel launch");
    }

    unsigned long long h_bad = 0;
    distributed_qr::AssertCuda(
        cudaMemcpy(&h_bad, d_bad, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
        "cudaMemcpy d_bad -> h_bad");
    unsigned long long total_bad = 0;
    MPI_Allreduce(&h_bad, &total_bad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (env.rank == 0) {
        const int effective_tile = (opts.overlap_tile <= 0) ? opts.nb : opts.overlap_tile;
        spdlog::info(
            "Distributed blocked QR (float): m={} n={} nb={} tile={} np={} avg {:.3f} ms", opts.m,
            opts.n, opts.nb, effective_tile, env.size, max_ms);
        if (total_bad > 0) {
            spdlog::error("Detected {} non-finite values after factorization.", total_bad);
        }
    }

    distributed_qr::AssertCuda(cudaFree(d_bad), "cudaFree d_bad");

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
    distributed_qr::AssertCuda(cudaFree(d_A), "cudaFree d_A");
    distributed_qr::AssertCuda(cudaFree(d_W), "cudaFree d_W");
    distributed_qr::AssertCuda(cudaFree(d_Y), "cudaFree d_Y");

    finalize_nccl_if_needed(&env);
    finalize_mpi_if_needed(env);

    return (total_bad == 0) ? 0 : 1;
}
