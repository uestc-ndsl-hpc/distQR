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
#include <vector>

#include "components/distributed_blocked_qr_col_blockcyclic.cuh"
#include "components/resourse_initial.cuh"
#include "utils/nvtx_range.cuh"

namespace {

using distributed_qr_col_blockcyclic::kPanelWidth;
using distributed_qr_col_blockcyclic::PanelCommMode;

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

template <typename T>
const char* DataTypeString() {
    if constexpr (std::is_same_v<T, float>) {
        return "float";
    }
    return "double";
}

struct Options {
    int m = 16384;
    int n = 1024;
    int nb = 1024;
    int warmup = 1;
    int iters = 3;
    int overlap_tile = 0;
    int block_cols = 0;
    bool print_per_rank = false;
    bool print_comm_bw = false;
    bool use_double = false;
    bool type_valid = true;
    std::string type_value = "float";
    PanelCommMode panel_comm_mode = PanelCommMode::SendRecv;
    bool panel_comm_valid = true;
    std::string panel_comm_value = "sendrecv";
    bool use_compact_local_gemm = false;
};

bool ParsePanelCommMode(const char* mode, PanelCommMode* out_mode) {
    if (std::strcmp(mode, "sendrecv") == 0 || std::strcmp(mode, "send_recv") == 0) {
        *out_mode = PanelCommMode::SendRecv;
        return true;
    }
    if (std::strcmp(mode, "broadcast") == 0) {
        *out_mode = PanelCommMode::Broadcast;
        return true;
    }
    return false;
}

bool ParseType(const char* type_str, bool* out_use_double) {
    if (std::strcmp(type_str, "float") == 0 || std::strcmp(type_str, "fp32") == 0) {
        *out_use_double = false;
        return true;
    }
    if (std::strcmp(type_str, "double") == 0 || std::strcmp(type_str, "fp64") == 0) {
        *out_use_double = true;
        return true;
    }
    return false;
}

const char* PanelCommModeToString(PanelCommMode mode) {
    if (mode == PanelCommMode::Broadcast) {
        return "broadcast";
    }
    return "sendrecv";
}

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
        } else if (std::strcmp(argv[i], "--block_cols") == 0 && i + 1 < argc) {
            opts.block_cols = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--print_per_rank") == 0) {
            opts.print_per_rank = true;
        } else if (std::strcmp(argv[i], "--print_comm_bw") == 0) {
            opts.print_comm_bw = true;
        } else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            opts.type_value = argv[++i];
            opts.type_valid = ParseType(opts.type_value.c_str(), &opts.use_double);
        } else if (std::strcmp(argv[i], "--panel-comm") == 0 && i + 1 < argc) {
            opts.panel_comm_value = argv[++i];
            opts.panel_comm_valid =
                ParsePanelCommMode(opts.panel_comm_value.c_str(), &opts.panel_comm_mode);
        } else if (std::strcmp(argv[i], "--compact-local-gemm") == 0) {
            opts.use_compact_local_gemm = true;
        } else if (std::strcmp(argv[i], "--segmented-local-gemm") == 0) {
            opts.use_compact_local_gemm = false;
        }
    }
    return opts;
}

template <typename T>
int RunBenchmarkTyped(const MpiCudaEnv& env, const Options& opts, int block_cols) {
    const auto part = distributed_qr_col_blockcyclic::MakeColBlockCyclicPartition(
        opts.n, block_cols, env.size, env.rank);
    const int lda_local = std::max(opts.m, 1);
    const size_t local_elems_alloc =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(part.local_cols, 1));
    const size_t local_elems_used =
        static_cast<size_t>(opts.m) * static_cast<size_t>(part.local_cols);

    T* d_A0 = nullptr;
    T* d_A = nullptr;
    T* d_W = nullptr;
    T* d_Y = nullptr;

    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_A0, local_elems_alloc * sizeof(T)),
                                               "cudaMalloc d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_A, local_elems_alloc * sizeof(T)),
                                               "cudaMalloc d_A");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_W, local_elems_alloc * sizeof(T)),
                                               "cudaMalloc d_W");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_Y, local_elems_alloc * sizeof(T)),
                                               "cudaMalloc d_Y");

    FillDeviceRandom(d_A0, local_elems_used, 2026ULL + static_cast<unsigned long long>(env.rank));

    const bool trail_one_shot = opts.overlap_tile <= 0;
    const int tile_cols = trail_one_shot
                              ? std::max(part.local_cols, 1)
                              : std::max(kPanelWidth, std::min(opts.overlap_tile, opts.nb));
    distributed_qr_col_blockcyclic::DistributedQrColBlockCyclicWorkspace<T> ws{};
    ws.tsqr_work_panel_elems = std::max(tsqr_work_elems<T>(opts.m), static_cast<size_t>(1));
    ws.pack_elems = static_cast<size_t>(opts.m) * static_cast<size_t>(kPanelWidth);
    ws.block_storage_elems = static_cast<size_t>(opts.m) * static_cast<size_t>(opts.nb);
    ws.block_compact_elems =
        static_cast<size_t>(opts.nb + kPanelWidth) * static_cast<size_t>(opts.nb);
    ws.tmp_elems = static_cast<size_t>(opts.nb) * static_cast<size_t>(tile_cols);

    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_r_panel, static_cast<size_t>(kPanelWidth) * kPanelWidth * sizeof(T)),
        "cudaMalloc ws.d_r_panel");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_tsqr_work_panel, ws.tsqr_work_panel_elems * sizeof(T)),
        "cudaMalloc ws.d_tsqr_work_panel");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_pack_w[0], ws.pack_elems * sizeof(T)), "cudaMalloc ws.d_pack_w[0]");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_pack_w[1], ws.pack_elems * sizeof(T)), "cudaMalloc ws.d_pack_w[1]");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_pack_y[0], ws.pack_elems * sizeof(T)), "cudaMalloc ws.d_pack_y[0]");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_pack_y[1], ws.pack_elems * sizeof(T)), "cudaMalloc ws.d_pack_y[1]");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_block_w, ws.block_storage_elems * sizeof(T)), "cudaMalloc ws.d_block_w");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_block_y, ws.block_storage_elems * sizeof(T)), "cudaMalloc ws.d_block_y");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_block_w_compact, ws.block_compact_elems * sizeof(T)),
        "cudaMalloc ws.d_block_w_compact");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMalloc(&ws.d_block_y_compact, ws.block_compact_elems * sizeof(T)),
        "cudaMalloc ws.d_block_y_compact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&ws.d_tmp0, ws.tmp_elems * sizeof(T)),
                                               "cudaMalloc ws.d_tmp0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&ws.d_tmp1, ws.tmp_elems * sizeof(T)),
                                               "cudaMalloc ws.d_tmp1");

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
                                                 "cublasSetStream(compute_stream)");

    for (int i = 0; i < opts.warmup; ++i) {
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMemcpyAsync(d_A, d_A0, local_elems_alloc * sizeof(T), cudaMemcpyDeviceToDevice,
                            compute_stream),
            "cudaMemcpyAsync warmup A <- A0");
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMemsetAsync(d_W, 0, local_elems_alloc * sizeof(T), compute_stream),
            "cudaMemsetAsync warmup W");
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMemsetAsync(d_Y, 0, local_elems_alloc * sizeof(T), compute_stream),
            "cudaMemsetAsync warmup Y");

        distributed_qr_col_blockcyclic::distributed_blocked_qr_factorize_col_blockcyclic<T>(
            cublas_handle, env.nccl_comm, part, opts.m, opts.n, opts.nb, d_A, lda_local, d_W, d_Y,
            &ws, compute_stream, comm_stream, opts.overlap_tile, nullptr, opts.panel_comm_mode,
            opts.use_compact_local_gemm);
        distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(compute_stream),
                                                   "cudaStreamSynchronize warmup compute");
        distributed_qr_col_blockcyclic::AssertCuda(cudaStreamSynchronize(comm_stream),
                                                   "cudaStreamSynchronize warmup comm");
    }

    cudaEvent_t timed_start = nullptr;
    cudaEvent_t timed_stop = nullptr;
    cudaEvent_t timed_comm_done = nullptr;
    distributed_qr_col_blockcyclic::AssertCuda(cudaEventCreate(&timed_start),
                                               "cudaEventCreate timed_start");
    distributed_qr_col_blockcyclic::AssertCuda(cudaEventCreate(&timed_stop),
                                               "cudaEventCreate timed_stop");
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaEventCreateWithFlags(&timed_comm_done, cudaEventDisableTiming),
        "cudaEventCreate timed_comm_done");
    std::vector<cudaEvent_t> comm_start_events;
    std::vector<cudaEvent_t> comm_end_events;
    std::vector<unsigned long long> comm_bytes_per_iter;
    if (opts.print_comm_bw) {
        comm_start_events.resize(opts.iters, nullptr);
        comm_end_events.resize(opts.iters, nullptr);
        comm_bytes_per_iter.resize(opts.iters, 0ULL);
        for (int i = 0; i < opts.iters; ++i) {
            distributed_qr_col_blockcyclic::AssertCuda(cudaEventCreate(&comm_start_events[i]),
                                                       "cudaEventCreate comm_start");
            distributed_qr_col_blockcyclic::AssertCuda(cudaEventCreate(&comm_end_events[i]),
                                                       "cudaEventCreate comm_end");
        }
    }

    float timed_total_ms = 0.0f;
    for (int i = 0; i < opts.iters; ++i) {
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMemcpyAsync(d_A, d_A0, local_elems_alloc * sizeof(T), cudaMemcpyDeviceToDevice,
                            compute_stream),
            "cudaMemcpyAsync timed A <- A0");
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMemsetAsync(d_W, 0, local_elems_alloc * sizeof(T), compute_stream),
            "cudaMemsetAsync timed W");
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaMemsetAsync(d_Y, 0, local_elems_alloc * sizeof(T), compute_stream),
            "cudaMemsetAsync timed Y");
        distributed_qr_col_blockcyclic::CommProfile comm_profile{};
        if (opts.print_comm_bw) {
            distributed_qr_col_blockcyclic::AssertCuda(
                cudaEventRecord(comm_start_events[i], comm_stream), "cudaEventRecord comm_start");
        }
        distributed_qr_col_blockcyclic::AssertCuda(cudaEventRecord(timed_start, compute_stream),
                                                   "cudaEventRecord timed_start");
        distributed_qr_col_blockcyclic::distributed_blocked_qr_factorize_col_blockcyclic<T>(
            cublas_handle, env.nccl_comm, part, opts.m, opts.n, opts.nb, d_A, lda_local, d_W, d_Y,
            &ws, compute_stream, comm_stream, opts.overlap_tile,
            opts.print_comm_bw ? &comm_profile : nullptr, opts.panel_comm_mode,
            opts.use_compact_local_gemm);
        if (opts.print_comm_bw) {
            distributed_qr_col_blockcyclic::AssertCuda(
                cudaEventRecord(comm_end_events[i], comm_stream), "cudaEventRecord comm_end");
            comm_bytes_per_iter[i] = static_cast<unsigned long long>(comm_profile.bytes);
        }
        distributed_qr_col_blockcyclic::AssertCuda(cudaEventRecord(timed_comm_done, comm_stream),
                                                   "cudaEventRecord timed_comm_done");
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaStreamWaitEvent(compute_stream, timed_comm_done, 0),
            "cudaStreamWaitEvent compute <- timed_comm_done");
        distributed_qr_col_blockcyclic::AssertCuda(cudaEventRecord(timed_stop, compute_stream),
                                                   "cudaEventRecord timed_stop");
        distributed_qr_col_blockcyclic::AssertCuda(cudaEventSynchronize(timed_stop),
                                                   "cudaEventSynchronize timed_stop");
        float iter_ms = 0.0f;
        distributed_qr_col_blockcyclic::AssertCuda(
            cudaEventElapsedTime(&iter_ms, timed_start, timed_stop), "cudaEventElapsedTime timed");
        timed_total_ms += iter_ms;
    }

    const double local_ms = static_cast<double>(timed_total_ms) / static_cast<double>(opts.iters);
    const double local_ms_no_barrier = local_ms;
    double local_comm_ms = 0.0;
    unsigned long long local_comm_bytes = 0ULL;
    if (opts.print_comm_bw) {
        for (int i = 0; i < opts.iters; ++i) {
            float ms = 0.0f;
            distributed_qr_col_blockcyclic::AssertCuda(
                cudaEventElapsedTime(&ms, comm_start_events[i], comm_end_events[i]),
                "cudaEventElapsedTime comm");
            local_comm_ms += static_cast<double>(ms);
            local_comm_bytes += comm_bytes_per_iter[i];
        }
        local_comm_ms /= static_cast<double>(opts.iters);
        local_comm_bytes /= static_cast<unsigned long long>(opts.iters);
    }

    distributed_qr_col_blockcyclic::AssertCuda(cudaEventDestroy(timed_start),
                                               "cudaEventDestroy timed_start");
    distributed_qr_col_blockcyclic::AssertCuda(cudaEventDestroy(timed_stop),
                                               "cudaEventDestroy timed_stop");
    distributed_qr_col_blockcyclic::AssertCuda(cudaEventDestroy(timed_comm_done),
                                               "cudaEventDestroy timed_comm_done");

    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> all_local_ms;
    std::vector<double> all_local_ms_no_barrier;
    std::vector<double> all_local_comm_ms;
    std::vector<unsigned long long> all_local_comm_bytes;
    if (opts.print_per_rank && env.rank == 0) {
        all_local_ms.resize(env.size, 0.0);
        all_local_ms_no_barrier.resize(env.size, 0.0);
    }
    if (opts.print_comm_bw && env.rank == 0) {
        all_local_comm_ms.resize(env.size, 0.0);
        all_local_comm_bytes.resize(env.size, 0ULL);
    }
    if (opts.print_per_rank) {
        MPI_Gather(&local_ms, 1, MPI_DOUBLE, (env.rank == 0) ? all_local_ms.data() : nullptr, 1,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_ms_no_barrier, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_local_ms_no_barrier.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
    }
    if (opts.print_comm_bw) {
        MPI_Gather(&local_comm_ms, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_local_comm_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_comm_bytes, 1, MPI_UNSIGNED_LONG_LONG,
                   (env.rank == 0) ? all_local_comm_bytes.data() : nullptr, 1,
                   MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    unsigned long long* d_bad = nullptr;
    distributed_qr_col_blockcyclic::AssertCuda(cudaMalloc(&d_bad, sizeof(unsigned long long)),
                                               "cudaMalloc d_bad");
    distributed_qr_col_blockcyclic::AssertCuda(cudaMemset(d_bad, 0, sizeof(unsigned long long)),
                                               "cudaMemset d_bad");
    if (local_elems_used > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((local_elems_used + threads - 1) / threads);
        count_non_finite_kernel<<<blocks, threads>>>(d_A, local_elems_used, d_bad);
        distributed_qr_col_blockcyclic::AssertCuda(cudaGetLastError(),
                                                   "count_non_finite_kernel launch");
    }

    unsigned long long h_bad = 0;
    distributed_qr_col_blockcyclic::AssertCuda(
        cudaMemcpy(&h_bad, d_bad, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
        "cudaMemcpy d_bad -> h_bad");
    unsigned long long total_bad = 0;
    MPI_Allreduce(&h_bad, &total_bad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (env.rank == 0) {
        spdlog::info(
            "Distributed blocked QR [col-blockcyclic] ({}): m={} n={} nb={} block_cols={} "
            "trail_update={} tile={} panel_comm={} local_update={} np={} avg {:.3f} ms",
            DataTypeString<T>(), opts.m, opts.n, opts.nb, block_cols,
            trail_one_shot ? "one-shot" : "tiled", trail_one_shot ? part.local_cols : tile_cols,
            PanelCommModeToString(opts.panel_comm_mode),
            opts.use_compact_local_gemm ? "compact" : "segmented", env.size, max_ms);
        if (opts.print_per_rank) {
            for (int r = 0; r < env.size; ++r) {
                spdlog::info("Per-rank time: rank {} -> {:.3f} ms (no-barrier {:.3f} ms)", r,
                             all_local_ms[r], all_local_ms_no_barrier[r]);
            }
        }
        if (opts.print_comm_bw) {
            for (int r = 0; r < env.size; ++r) {
                const double bytes_gb = static_cast<double>(all_local_comm_bytes[r]) / 1.0e9;
                const double bw_gbps = (all_local_comm_ms[r] > 0.0)
                                           ? (bytes_gb / (all_local_comm_ms[r] / 1000.0))
                                           : 0.0;
                spdlog::info("Per-rank comm(event): rank {} -> {:.3f} ms, {:.3f} MiB, {:.3f} GB/s",
                             r, all_local_comm_ms[r],
                             static_cast<double>(all_local_comm_bytes[r]) / (1024.0 * 1024.0),
                             bw_gbps);
            }
        }
        if (total_bad > 0) {
            spdlog::error("Detected {} non-finite values after factorization.", total_bad);
        }
    }

    if (opts.print_comm_bw) {
        for (int i = 0; i < opts.iters; ++i) {
            distributed_qr_col_blockcyclic::AssertCuda(cudaEventDestroy(comm_start_events[i]),
                                                       "cudaEventDestroy comm_start");
            distributed_qr_col_blockcyclic::AssertCuda(cudaEventDestroy(comm_end_events[i]),
                                                       "cudaEventDestroy comm_end");
        }
    }

    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_bad), "cudaFree d_bad");

    distributed_qr_col_blockcyclic::AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamDestroy(compute_stream),
                                               "cudaStreamDestroy compute");
    distributed_qr_col_blockcyclic::AssertCuda(cudaStreamDestroy(comm_stream),
                                               "cudaStreamDestroy comm");

    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_r_panel), "cudaFree ws.d_r_panel");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_tsqr_work_panel),
                                               "cudaFree ws.d_tsqr_work_panel");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_pack_w[0]), "cudaFree ws.d_pack_w[0]");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_pack_w[1]), "cudaFree ws.d_pack_w[1]");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_pack_y[0]), "cudaFree ws.d_pack_y[0]");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_pack_y[1]), "cudaFree ws.d_pack_y[1]");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_block_w), "cudaFree ws.d_block_w");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_block_y), "cudaFree ws.d_block_y");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_block_w_compact),
                                               "cudaFree ws.d_block_w_compact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_block_y_compact),
                                               "cudaFree ws.d_block_y_compact");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_tmp0), "cudaFree ws.d_tmp0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(ws.d_tmp1), "cudaFree ws.d_tmp1");

    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_A), "cudaFree d_A");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_W), "cudaFree d_W");
    distributed_qr_col_blockcyclic::AssertCuda(cudaFree(d_Y), "cudaFree d_Y");

    return (total_bad == 0) ? 0 : 1;
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
    const int block_cols = (opts.block_cols > 0) ? opts.block_cols : opts.nb;

    if (opts.n > opts.m || opts.n % kPanelWidth != 0 || opts.nb % kPanelWidth != 0 ||
        opts.nb > opts.n) {
        if (env.rank == 0) {
            spdlog::error(
                "Invalid args: require m>=n, n%{}==0, nb%{}==0, nb<=n (got m={} n={} nb={})",
                kPanelWidth, kPanelWidth, opts.m, opts.n, opts.nb);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }
    if (block_cols <= 0 || block_cols % opts.nb != 0 || block_cols % kPanelWidth != 0 ||
        block_cols > opts.n) {
        if (env.rank == 0) {
            spdlog::error(
                "Invalid args: require block_cols>0, block_cols%nb==0, block_cols%{}==0, "
                "block_cols<=n (got block_cols={} nb={} n={})",
                kPanelWidth, block_cols, opts.nb, opts.n);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }
    if (opts.warmup < 0 || opts.iters <= 0) {
        if (env.rank == 0) {
            spdlog::error(
                "Invalid args: require warmup >= 0 and iters > 0 (got warmup={} iters={})",
                opts.warmup, opts.iters);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }
    if (!opts.type_valid) {
        if (env.rank == 0) {
            spdlog::error("Invalid --type value '{}'. Supported values: float, double.",
                          opts.type_value);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }
    if (!opts.panel_comm_valid) {
        if (env.rank == 0) {
            spdlog::error("Invalid --panel-comm value '{}'. Supported values: sendrecv, broadcast.",
                          opts.panel_comm_value);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }
    if (env.rank == 0) {
        spdlog::info(
            "Custom NVTX backend: {} (compiled={}, runtime={}, set {}=0 to disable)",
            distqr::nvtx::kBackendName, distqr::nvtx::kCompiledIn ? "yes" : "no",
            distqr::nvtx::IsEnabled() ? "enabled" : "disabled",
            distqr::nvtx::kEnableEnvVarName);
    }

    const int local_ret = opts.use_double ? RunBenchmarkTyped<double>(env, opts, block_cols)
                                          : RunBenchmarkTyped<float>(env, opts, block_cols);

    finalize_nccl_if_needed(&env);
    finalize_mpi_if_needed(env);
    return local_ret;
}
