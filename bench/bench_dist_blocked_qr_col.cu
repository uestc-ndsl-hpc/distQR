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

#include "components/distributed_blocked_qr_col.cuh"
#include "components/resourse_initial.cuh"

namespace {

using distributed_qr_col::kPanelWidth;

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

    const auto part = distributed_qr_col::MakeColPartition(opts.n, env.size, env.rank);
    const int lda_local = std::max(opts.m, 1);
    const size_t local_elems_alloc =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(part.local_cols, 1));
    const size_t local_elems_used =
        static_cast<size_t>(opts.m) * static_cast<size_t>(part.local_cols);

    float* d_A0 = nullptr;
    float* d_A = nullptr;
    float* d_W = nullptr;
    float* d_Y = nullptr;

    distributed_qr_col::AssertCuda(cudaMalloc(&d_A0, local_elems_alloc * sizeof(float)),
                                   "cudaMalloc d_A0");
    distributed_qr_col::AssertCuda(cudaMalloc(&d_A, local_elems_alloc * sizeof(float)),
                                   "cudaMalloc d_A");
    distributed_qr_col::AssertCuda(cudaMalloc(&d_W, local_elems_alloc * sizeof(float)),
                                   "cudaMalloc d_W");
    distributed_qr_col::AssertCuda(cudaMalloc(&d_Y, local_elems_alloc * sizeof(float)),
                                   "cudaMalloc d_Y");

    FillDeviceRandom(d_A0, local_elems_used, 2026ULL + static_cast<unsigned long long>(env.rank));

    const int tile_target = (opts.overlap_tile <= 0) ? opts.nb : opts.overlap_tile;
    const int tile_cols = std::max(kPanelWidth, std::min(tile_target, opts.nb));
    distributed_qr_col::DistributedQrColWorkspace<float> ws{};
    ws.tsqr_work_panel_elems = std::max(tsqr_work_elems<float>(opts.m), static_cast<size_t>(1));
    ws.tmp_elems = static_cast<size_t>(kPanelWidth) * static_cast<size_t>(tile_cols);

    distributed_qr_col::AssertCuda(
        cudaMalloc(&ws.d_r_panel, static_cast<size_t>(kPanelWidth) * kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_r_panel");
    distributed_qr_col::AssertCuda(
        cudaMalloc(&ws.d_tsqr_work_panel, ws.tsqr_work_panel_elems * sizeof(float)),
        "cudaMalloc ws.d_tsqr_work_panel");
    distributed_qr_col::AssertCuda(
        cudaMalloc(&ws.d_panel_w, static_cast<size_t>(opts.m) * kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_panel_w");
    distributed_qr_col::AssertCuda(
        cudaMalloc(&ws.d_panel_y, static_cast<size_t>(opts.m) * kPanelWidth * sizeof(float)),
        "cudaMalloc ws.d_panel_y");
    distributed_qr_col::AssertCuda(cudaMalloc(&ws.d_tmp, ws.tmp_elems * sizeof(float)),
                                   "cudaMalloc ws.d_tmp");

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
                                     "cublasSetStream(compute_stream)");

    for (int i = 0; i < opts.warmup; ++i) {
        distributed_qr_col::AssertCuda(cudaMemcpyAsync(d_A, d_A0, local_elems_alloc * sizeof(float),
                                                       cudaMemcpyDeviceToDevice, compute_stream),
                                       "cudaMemcpyAsync warmup A <- A0");
        distributed_qr_col::AssertCuda(
            cudaMemsetAsync(d_W, 0, local_elems_alloc * sizeof(float), compute_stream),
            "cudaMemsetAsync warmup W");
        distributed_qr_col::AssertCuda(
            cudaMemsetAsync(d_Y, 0, local_elems_alloc * sizeof(float), compute_stream),
            "cudaMemsetAsync warmup Y");

        distributed_qr_col::distributed_blocked_qr_factorize_col<float>(
            cublas_handle, env.nccl_comm, part, opts.m, opts.n, opts.nb, d_A, lda_local, d_W, d_Y,
            &ws, compute_stream, comm_stream, opts.overlap_tile);
        distributed_qr_col::AssertCuda(cudaStreamSynchronize(compute_stream),
                                       "cudaStreamSynchronize warmup compute");
        distributed_qr_col::AssertCuda(cudaStreamSynchronize(comm_stream),
                                       "cudaStreamSynchronize warmup comm");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int i = 0; i < opts.iters; ++i) {
        distributed_qr_col::AssertCuda(cudaMemcpyAsync(d_A, d_A0, local_elems_alloc * sizeof(float),
                                                       cudaMemcpyDeviceToDevice, compute_stream),
                                       "cudaMemcpyAsync timed A <- A0");
        distributed_qr_col::AssertCuda(
            cudaMemsetAsync(d_W, 0, local_elems_alloc * sizeof(float), compute_stream),
            "cudaMemsetAsync timed W");
        distributed_qr_col::AssertCuda(
            cudaMemsetAsync(d_Y, 0, local_elems_alloc * sizeof(float), compute_stream),
            "cudaMemsetAsync timed Y");

        distributed_qr_col::distributed_blocked_qr_factorize_col<float>(
            cublas_handle, env.nccl_comm, part, opts.m, opts.n, opts.nb, d_A, lda_local, d_W, d_Y,
            &ws, compute_stream, comm_stream, opts.overlap_tile);
    }

    distributed_qr_col::AssertCuda(cudaStreamSynchronize(compute_stream),
                                   "cudaStreamSynchronize timed compute");
    distributed_qr_col::AssertCuda(cudaStreamSynchronize(comm_stream),
                                   "cudaStreamSynchronize timed comm");

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double local_ms = (t1 - t0) * 1000.0 / static_cast<double>(opts.iters);
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    unsigned long long* d_bad = nullptr;
    distributed_qr_col::AssertCuda(cudaMalloc(&d_bad, sizeof(unsigned long long)),
                                   "cudaMalloc d_bad");
    distributed_qr_col::AssertCuda(cudaMemset(d_bad, 0, sizeof(unsigned long long)),
                                   "cudaMemset d_bad");
    if (local_elems_used > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((local_elems_used + threads - 1) / threads);
        count_non_finite_kernel<<<blocks, threads>>>(d_A, local_elems_used, d_bad);
        distributed_qr_col::AssertCuda(cudaGetLastError(), "count_non_finite_kernel launch");
    }

    unsigned long long h_bad = 0;
    distributed_qr_col::AssertCuda(
        cudaMemcpy(&h_bad, d_bad, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
        "cudaMemcpy d_bad -> h_bad");
    unsigned long long total_bad = 0;
    MPI_Allreduce(&h_bad, &total_bad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (env.rank == 0) {
        const int effective_tile = (opts.overlap_tile <= 0) ? opts.nb : opts.overlap_tile;
        spdlog::info(
            "Distributed blocked QR [col-partition] (float): m={} n={} nb={} tile={} np={} avg "
            "{:.3f} ms",
            opts.m, opts.n, opts.nb, effective_tile, env.size, max_ms);
        if (total_bad > 0) {
            spdlog::error("Detected {} non-finite values after factorization.", total_bad);
        }
    }

    distributed_qr_col::AssertCuda(cudaFree(d_bad), "cudaFree d_bad");

    distributed_qr_col::AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    distributed_qr_col::AssertCuda(cudaStreamDestroy(compute_stream), "cudaStreamDestroy compute");
    distributed_qr_col::AssertCuda(cudaStreamDestroy(comm_stream), "cudaStreamDestroy comm");

    distributed_qr_col::AssertCuda(cudaFree(ws.d_r_panel), "cudaFree ws.d_r_panel");
    distributed_qr_col::AssertCuda(cudaFree(ws.d_tsqr_work_panel), "cudaFree ws.d_tsqr_work_panel");
    distributed_qr_col::AssertCuda(cudaFree(ws.d_panel_w), "cudaFree ws.d_panel_w");
    distributed_qr_col::AssertCuda(cudaFree(ws.d_panel_y), "cudaFree ws.d_panel_y");
    distributed_qr_col::AssertCuda(cudaFree(ws.d_tmp), "cudaFree ws.d_tmp");

    distributed_qr_col::AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    distributed_qr_col::AssertCuda(cudaFree(d_A), "cudaFree d_A");
    distributed_qr_col::AssertCuda(cudaFree(d_W), "cudaFree d_W");
    distributed_qr_col::AssertCuda(cudaFree(d_Y), "cudaFree d_Y");

    finalize_nccl_if_needed(&env);
    finalize_mpi_if_needed(env);

    return (total_bad == 0) ? 0 : 1;
}
