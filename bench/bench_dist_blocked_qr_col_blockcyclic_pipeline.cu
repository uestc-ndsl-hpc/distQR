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

#include "components/distributed_blocked_qr_col_blockcyclic_pipeline.cuh"
#include "components/resourse_initial.cuh"

namespace {

using distributed_qr_col_blockcyclic_pipeline::kPanelWidth;

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
    } else {
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

double QrFactorizationFlops(int m, int n) {
    const double m_d = static_cast<double>(m);
    const double n_d = static_cast<double>(n);
    return 2.0 * m_d * n_d * n_d - (2.0 / 3.0) * n_d * n_d * n_d;
}

double TflopsFromFlopsAndMs(double flops, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return flops / (ms * 1.0e9);
}

double GigabytesPerSecondFromBytesAndMs(double bytes, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return bytes / (ms * 1.0e6);
}

struct Options {
    int m = 16384;
    int n = 32;
    int nb = 32;
    int warmup = 1;
    int iters = 3;
    int block_cols = 0;
    int update_tile = 32;
    int row_block_rows = 1024;
    int trail_tile_cols = 256;
    bool print_per_rank = false;
    bool print_phase_timing = false;
    bool use_double = false;
    bool type_valid = true;
    std::string type_value = "float";
    distributed_qr_col_blockcyclic_pipeline::RowBlockPipelineConfig::TailMode row_block_mode =
        distributed_qr_col_blockcyclic_pipeline::RowBlockPipelineConfig::TailMode::Baseline;
    bool row_block_mode_valid = true;
    std::string row_block_mode_value = "baseline";
    bool skip_tail_update = false;
};

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

bool ParseRowBlockMode(
    const char* mode_str,
    distributed_qr_col_blockcyclic_pipeline::RowBlockPipelineConfig::TailMode* out_mode) {
    using TailMode = distributed_qr_col_blockcyclic_pipeline::RowBlockPipelineConfig::TailMode;
    if (std::strcmp(mode_str, "baseline") == 0) {
        *out_mode = TailMode::Baseline;
        return true;
    }
    if (std::strcmp(mode_str, "overlap") == 0) {
        *out_mode = TailMode::Overlap;
        return true;
    }
    return false;
}

Options ParseArgs(int argc, char** argv) {
    Options opts;
    auto parse_prefixed_int = [](const char* arg, const char* prefix, int* out_value) {
        const size_t prefix_len = std::strlen(prefix);
        if (std::strncmp(arg, prefix, prefix_len) != 0 || arg[prefix_len] != '=') {
            return false;
        }
        *out_value = std::atoi(arg + prefix_len + 1);
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            opts.m = std::atoi(argv[++i]);
        } else if (parse_prefixed_int(argv[i], "--m", &opts.m)) {
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            opts.n = std::atoi(argv[++i]);
        } else if (parse_prefixed_int(argv[i], "--n", &opts.n)) {
        } else if (std::strcmp(argv[i], "--nb") == 0 && i + 1 < argc) {
            opts.nb = std::atoi(argv[++i]);
        } else if (parse_prefixed_int(argv[i], "--nb", &opts.nb)) {
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opts.warmup = std::atoi(argv[++i]);
        } else if (parse_prefixed_int(argv[i], "--warmup", &opts.warmup)) {
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opts.iters = std::atoi(argv[++i]);
        } else if (parse_prefixed_int(argv[i], "--iters", &opts.iters)) {
        } else if (std::strcmp(argv[i], "--block_cols") == 0 && i + 1 < argc) {
            opts.block_cols = std::atoi(argv[++i]);
        } else if (parse_prefixed_int(argv[i], "--block_cols", &opts.block_cols)) {
        } else if (std::strcmp(argv[i], "--update_tile") == 0 && i + 1 < argc) {
            opts.update_tile = std::atoi(argv[++i]);
        } else if (parse_prefixed_int(argv[i], "--update_tile", &opts.update_tile)) {
        } else if (std::strcmp(argv[i], "--row_block_rows") == 0 && i + 1 < argc) {
            opts.row_block_rows = std::atoi(argv[++i]);
        } else if (parse_prefixed_int(argv[i], "--row_block_rows", &opts.row_block_rows)) {
        } else if (std::strcmp(argv[i], "--trail_tile_cols") == 0 && i + 1 < argc) {
            opts.trail_tile_cols = std::atoi(argv[++i]);
        } else if (parse_prefixed_int(argv[i], "--trail_tile_cols", &opts.trail_tile_cols)) {
        } else if (std::strcmp(argv[i], "--print_per_rank") == 0) {
            opts.print_per_rank = true;
        } else if (std::strcmp(argv[i], "--print_phase_timing") == 0) {
            opts.print_phase_timing = true;
        } else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            opts.type_value = argv[++i];
            opts.type_valid = ParseType(opts.type_value.c_str(), &opts.use_double);
        } else if (std::strncmp(argv[i], "--type=", 7) == 0) {
            opts.type_value = argv[i] + 7;
            opts.type_valid = ParseType(opts.type_value.c_str(), &opts.use_double);
        } else if (std::strcmp(argv[i], "--row_block_mode") == 0 && i + 1 < argc) {
            opts.row_block_mode_value = argv[++i];
            opts.row_block_mode_valid =
                ParseRowBlockMode(opts.row_block_mode_value.c_str(), &opts.row_block_mode);
        } else if (std::strncmp(argv[i], "--row_block_mode=", 17) == 0) {
            opts.row_block_mode_value = argv[i] + 17;
            opts.row_block_mode_valid =
                ParseRowBlockMode(opts.row_block_mode_value.c_str(), &opts.row_block_mode);
        } else if (std::strcmp(argv[i], "--rowblock_buffers") == 0 && i + 1 < argc) {
            ++i;
        } else if (std::strncmp(argv[i], "--rowblock_buffers=", 19) == 0) {
        } else if (std::strcmp(argv[i], "--skip_tail_update") == 0) {
            opts.skip_tail_update = true;
        }
    }
    return opts;
}

template <typename T>
int RunBenchmarkTyped(const MpiCudaEnv& env, const Options& opts, int block_cols) {
    const auto part = distributed_qr_col_blockcyclic_pipeline::MakeColBlockCyclicPartition(
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

    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&d_A0, local_elems_alloc * sizeof(T)), "cudaMalloc d_A0");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&d_A, local_elems_alloc * sizeof(T)), "cudaMalloc d_A");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&d_W, local_elems_alloc * sizeof(T)), "cudaMalloc d_W");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&d_Y, local_elems_alloc * sizeof(T)), "cudaMalloc d_Y");

    FillDeviceRandom(d_A0, local_elems_used, 2026ULL + static_cast<unsigned long long>(env.rank));

    distributed_qr_col_blockcyclic_pipeline::DistributedQrColBlockCyclicPipelineWorkspace<T> ws{};
    ws.tsqr_work_panel_elems = std::max(tsqr_work_elems<T>(opts.m), static_cast<size_t>(1));
    ws.panel_elems = static_cast<size_t>(opts.m) * static_cast<size_t>(kPanelWidth);
    ws.block_elems = static_cast<size_t>(opts.m) * static_cast<size_t>(opts.nb);
    ws.block_rowmajor_elems = ws.block_elems;
    ws.rowblock_wy_packed_elems = 2 * ws.block_elems;
    ws.tmp_elems = static_cast<size_t>(std::max(opts.nb, kPanelWidth)) *
                   static_cast<size_t>(std::max(opts.trail_tile_cols, 1));

    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_r_panel, static_cast<size_t>(kPanelWidth) * kPanelWidth * sizeof(T)),
        "cudaMalloc ws.d_r_panel");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_tsqr_work_panel, ws.tsqr_work_panel_elems * sizeof(T)),
        "cudaMalloc ws.d_tsqr_work_panel");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_panel_w, ws.panel_elems * sizeof(T)), "cudaMalloc ws.d_panel_w");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_panel_y, ws.panel_elems * sizeof(T)), "cudaMalloc ws.d_panel_y");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_block_w, ws.block_elems * sizeof(T)), "cudaMalloc ws.d_block_w");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_block_y, ws.block_elems * sizeof(T)), "cudaMalloc ws.d_block_y");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_block_w_rowmajor, ws.block_rowmajor_elems * sizeof(T)),
        "cudaMalloc ws.d_block_w_rowmajor");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_block_y_rowmajor, ws.block_rowmajor_elems * sizeof(T)),
        "cudaMalloc ws.d_block_y_rowmajor");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_rowblock_wy_packed, ws.rowblock_wy_packed_elems * sizeof(T)),
        "cudaMalloc ws.d_rowblock_wy_packed");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_tmp0, ws.tmp_elems * sizeof(T)), "cudaMalloc ws.d_tmp0");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&ws.d_tmp1, ws.tmp_elems * sizeof(T)), "cudaMalloc ws.d_tmp1");

    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking),
        "cudaStreamCreate compute_stream");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking),
        "cudaStreamCreate comm_stream");

    cublasHandle_t cublas_handle = nullptr;
    distributed_qr_col_blockcyclic_pipeline::AssertCublas(cublasCreate(&cublas_handle),
                                                          "cublasCreate");
    distributed_qr_col_blockcyclic_pipeline::AssertCublas(
        cublasSetStream(cublas_handle, compute_stream), "cublasSetStream(compute_stream)");

    distributed_qr_col_blockcyclic_pipeline::RowBlockPipelineConfig pipeline_cfg{};
    pipeline_cfg.update_tile_cols = opts.update_tile;
    pipeline_cfg.row_block_rows = opts.row_block_rows;
    pipeline_cfg.trail_tile_cols = opts.trail_tile_cols;
    pipeline_cfg.tail_mode = opts.row_block_mode;
    pipeline_cfg.skip_tail_update = opts.skip_tail_update;

    auto run_once = [&](distributed_qr_col_blockcyclic_pipeline::CommProfile* comm_profile,
                        distributed_qr_col_blockcyclic_pipeline::PhaseProfile* phase_profile) {
        distributed_qr_col_blockcyclic_pipeline::
            distributed_blocked_qr_factorize_col_blockcyclic_pipeline<T>(
                cublas_handle, env.nccl_comm, part, opts.m, opts.n, opts.nb, d_A, lda_local, d_W,
                d_Y, &ws, compute_stream, comm_stream, pipeline_cfg, comm_profile, phase_profile);
    };

    for (int i = 0; i < opts.warmup; ++i) {
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaMemcpyAsync(d_A, d_A0, local_elems_alloc * sizeof(T), cudaMemcpyDeviceToDevice,
                            compute_stream),
            "cudaMemcpyAsync warmup A <- A0");
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaMemsetAsync(d_W, 0, local_elems_alloc * sizeof(T), compute_stream),
            "cudaMemsetAsync warmup W");
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaMemsetAsync(d_Y, 0, local_elems_alloc * sizeof(T), compute_stream),
            "cudaMemsetAsync warmup Y");
        run_once(nullptr, nullptr);
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaStreamSynchronize(compute_stream),
                                                            "cudaStreamSynchronize warmup compute");
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaStreamSynchronize(comm_stream),
                                                            "cudaStreamSynchronize warmup comm");
    }

    cudaEvent_t timed_start = nullptr;
    cudaEvent_t timed_stop = nullptr;
    cudaEvent_t timed_comm_done = nullptr;
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaEventCreate(&timed_start),
                                                        "cudaEventCreate timed_start");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaEventCreate(&timed_stop),
                                                        "cudaEventCreate timed_stop");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaEventCreateWithFlags(&timed_comm_done, cudaEventDisableTiming),
        "cudaEventCreate timed_comm_done");

    float timed_total_ms = 0.0f;
    std::vector<double> phase_panel_ms_per_iter;
    std::vector<double> phase_wy_ms_per_iter;
    std::vector<double> phase_merge_ms_per_iter;
    std::vector<double> phase_inner_update_ms_per_iter;
    std::vector<double> phase_pack_ms_per_iter;
    std::vector<double> phase_unpack_ms_per_iter;
    std::vector<double> phase_comm_ms_per_iter;
    std::vector<double> phase_tail_wait_ms_per_iter;
    std::vector<double> phase_tail_acc_gemm_ms_per_iter;
    std::vector<double> phase_tail_apply_gemm_ms_per_iter;
    std::vector<double> phase_tail_flops_per_iter;
    std::vector<double> comm_bytes_per_iter;
    if (opts.print_phase_timing) {
        phase_panel_ms_per_iter.resize(opts.iters, 0.0);
        phase_wy_ms_per_iter.resize(opts.iters, 0.0);
        phase_merge_ms_per_iter.resize(opts.iters, 0.0);
        phase_inner_update_ms_per_iter.resize(opts.iters, 0.0);
        phase_pack_ms_per_iter.resize(opts.iters, 0.0);
        phase_unpack_ms_per_iter.resize(opts.iters, 0.0);
        phase_comm_ms_per_iter.resize(opts.iters, 0.0);
        phase_tail_wait_ms_per_iter.resize(opts.iters, 0.0);
        phase_tail_acc_gemm_ms_per_iter.resize(opts.iters, 0.0);
        phase_tail_apply_gemm_ms_per_iter.resize(opts.iters, 0.0);
        phase_tail_flops_per_iter.resize(opts.iters, 0.0);
        comm_bytes_per_iter.resize(opts.iters, 0.0);
    }

    for (int i = 0; i < opts.iters; ++i) {
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaMemcpyAsync(d_A, d_A0, local_elems_alloc * sizeof(T), cudaMemcpyDeviceToDevice,
                            compute_stream),
            "cudaMemcpyAsync timed A <- A0");
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaMemsetAsync(d_W, 0, local_elems_alloc * sizeof(T), compute_stream),
            "cudaMemsetAsync timed W");
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaMemsetAsync(d_Y, 0, local_elems_alloc * sizeof(T), compute_stream),
            "cudaMemsetAsync timed Y");

        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaEventRecord(timed_start, compute_stream), "cudaEventRecord timed_start");
        distributed_qr_col_blockcyclic_pipeline::PhaseProfile phase_profile{};
        distributed_qr_col_blockcyclic_pipeline::CommProfile comm_profile{};
        run_once(opts.print_phase_timing ? &comm_profile : nullptr,
                 opts.print_phase_timing ? &phase_profile : nullptr);
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaEventRecord(timed_comm_done, comm_stream), "cudaEventRecord timed_comm_done");
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaStreamWaitEvent(compute_stream, timed_comm_done, 0),
            "cudaStreamWaitEvent compute <- timed_comm_done");
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaEventRecord(timed_stop, compute_stream), "cudaEventRecord timed_stop");
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaEventSynchronize(timed_stop),
                                                            "cudaEventSynchronize timed_stop");

        if (opts.print_phase_timing) {
            distributed_qr_col_blockcyclic_pipeline::FinalizePhaseProfile(&phase_profile);
            phase_panel_ms_per_iter[i] = phase_profile.panel_factor_ms;
            phase_wy_ms_per_iter[i] = phase_profile.wy_build_ms;
            phase_merge_ms_per_iter[i] = phase_profile.block_wy_merge_ms;
            phase_inner_update_ms_per_iter[i] = phase_profile.inner_block_update_ms;
            phase_pack_ms_per_iter[i] = phase_profile.rowblock_pack_ms;
            phase_unpack_ms_per_iter[i] = phase_profile.rowblock_unpack_ms;
            phase_comm_ms_per_iter[i] = phase_profile.comm_ms;
            phase_tail_wait_ms_per_iter[i] = phase_profile.tail_acc_wait_ms;
            phase_tail_acc_gemm_ms_per_iter[i] = phase_profile.tail_acc_gemm_ms;
            phase_tail_apply_gemm_ms_per_iter[i] = phase_profile.tail_apply_gemm_ms;
            phase_tail_flops_per_iter[i] = phase_profile.tail_update_flops;
            comm_bytes_per_iter[i] = static_cast<double>(comm_profile.bytes);
        }

        float iter_ms = 0.0f;
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(
            cudaEventElapsedTime(&iter_ms, timed_start, timed_stop), "cudaEventElapsedTime timed");
        timed_total_ms += iter_ms;
    }

    const double local_ms = static_cast<double>(timed_total_ms) / static_cast<double>(opts.iters);
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> all_local_ms;
    std::vector<double> all_panel_ms;
    std::vector<double> all_wy_ms;
    std::vector<double> all_merge_ms;
    std::vector<double> all_inner_update_ms;
    std::vector<double> all_pack_ms;
    std::vector<double> all_unpack_ms;
    std::vector<double> all_comm_ms;
    std::vector<double> all_tail_wait_ms;
    std::vector<double> all_tail_acc_gemm_ms;
    std::vector<double> all_tail_apply_gemm_ms;
    std::vector<double> all_compute_ms;
    std::vector<double> all_local_copy_ms;
    std::vector<double> all_tail_gemm_tflops;
    std::vector<double> all_comm_bytes;
    std::vector<double> all_comm_gbps;
    double local_panel_ms = 0.0;
    double local_wy_ms = 0.0;
    double local_merge_ms = 0.0;
    double local_inner_update_ms = 0.0;
    double local_pack_ms = 0.0;
    double local_unpack_ms = 0.0;
    double local_comm_ms = 0.0;
    double local_tail_wait_ms = 0.0;
    double local_tail_acc_gemm_ms = 0.0;
    double local_tail_apply_gemm_ms = 0.0;
    double local_tail_flops = 0.0;
    double local_comm_bytes = 0.0;

    if (opts.print_phase_timing) {
        for (int i = 0; i < opts.iters; ++i) {
            local_panel_ms += phase_panel_ms_per_iter[i];
            local_wy_ms += phase_wy_ms_per_iter[i];
            local_merge_ms += phase_merge_ms_per_iter[i];
            local_inner_update_ms += phase_inner_update_ms_per_iter[i];
            local_pack_ms += phase_pack_ms_per_iter[i];
            local_unpack_ms += phase_unpack_ms_per_iter[i];
            local_comm_ms += phase_comm_ms_per_iter[i];
            local_tail_wait_ms += phase_tail_wait_ms_per_iter[i];
            local_tail_acc_gemm_ms += phase_tail_acc_gemm_ms_per_iter[i];
            local_tail_apply_gemm_ms += phase_tail_apply_gemm_ms_per_iter[i];
            local_tail_flops += phase_tail_flops_per_iter[i];
            local_comm_bytes += comm_bytes_per_iter[i];
        }
        local_panel_ms /= static_cast<double>(opts.iters);
        local_wy_ms /= static_cast<double>(opts.iters);
        local_merge_ms /= static_cast<double>(opts.iters);
        local_inner_update_ms /= static_cast<double>(opts.iters);
        local_pack_ms /= static_cast<double>(opts.iters);
        local_unpack_ms /= static_cast<double>(opts.iters);
        local_comm_ms /= static_cast<double>(opts.iters);
        local_tail_wait_ms /= static_cast<double>(opts.iters);
        local_tail_acc_gemm_ms /= static_cast<double>(opts.iters);
        local_tail_apply_gemm_ms /= static_cast<double>(opts.iters);
        local_tail_flops /= static_cast<double>(opts.iters);
        local_comm_bytes /= static_cast<double>(opts.iters);
    }

    const double local_compute_ms = local_panel_ms + local_wy_ms + local_merge_ms +
                                    local_inner_update_ms + local_tail_acc_gemm_ms +
                                    local_tail_apply_gemm_ms;
    const double local_local_copy_ms = local_pack_ms + local_unpack_ms;
    const double local_tail_gemm_tflops =
        TflopsFromFlopsAndMs(local_tail_flops, local_tail_acc_gemm_ms + local_tail_apply_gemm_ms);
    const double local_comm_gbps =
        GigabytesPerSecondFromBytesAndMs(local_comm_bytes, local_comm_ms);

    if (opts.print_per_rank && env.rank == 0) {
        all_local_ms.resize(env.size, 0.0);
    }
    if (opts.print_phase_timing && env.rank == 0) {
        all_panel_ms.resize(env.size, 0.0);
        all_wy_ms.resize(env.size, 0.0);
        all_merge_ms.resize(env.size, 0.0);
        all_inner_update_ms.resize(env.size, 0.0);
        all_pack_ms.resize(env.size, 0.0);
        all_unpack_ms.resize(env.size, 0.0);
        all_comm_ms.resize(env.size, 0.0);
        all_tail_wait_ms.resize(env.size, 0.0);
        all_tail_acc_gemm_ms.resize(env.size, 0.0);
        all_tail_apply_gemm_ms.resize(env.size, 0.0);
        all_compute_ms.resize(env.size, 0.0);
        all_local_copy_ms.resize(env.size, 0.0);
        all_tail_gemm_tflops.resize(env.size, 0.0);
        all_comm_bytes.resize(env.size, 0.0);
        all_comm_gbps.resize(env.size, 0.0);
    }
    if (opts.print_per_rank) {
        MPI_Gather(&local_ms, 1, MPI_DOUBLE, (env.rank == 0) ? all_local_ms.data() : nullptr, 1,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (opts.print_phase_timing) {
        MPI_Gather(&local_panel_ms, 1, MPI_DOUBLE, (env.rank == 0) ? all_panel_ms.data() : nullptr,
                   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_wy_ms, 1, MPI_DOUBLE, (env.rank == 0) ? all_wy_ms.data() : nullptr, 1,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_merge_ms, 1, MPI_DOUBLE, (env.rank == 0) ? all_merge_ms.data() : nullptr,
                   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_inner_update_ms, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_inner_update_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_pack_ms, 1, MPI_DOUBLE, (env.rank == 0) ? all_pack_ms.data() : nullptr, 1,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_unpack_ms, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_unpack_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_comm_ms, 1, MPI_DOUBLE, (env.rank == 0) ? all_comm_ms.data() : nullptr, 1,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_tail_wait_ms, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_tail_wait_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_tail_acc_gemm_ms, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_tail_acc_gemm_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_tail_apply_gemm_ms, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_tail_apply_gemm_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_compute_ms, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_compute_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_local_copy_ms, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_local_copy_ms.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_tail_gemm_tflops, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_tail_gemm_tflops.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_comm_bytes, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_comm_bytes.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        MPI_Gather(&local_comm_gbps, 1, MPI_DOUBLE,
                   (env.rank == 0) ? all_comm_gbps.data() : nullptr, 1, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
    }

    unsigned long long* d_bad = nullptr;
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMalloc(&d_bad, sizeof(unsigned long long)), "cudaMalloc d_bad");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMemset(d_bad, 0, sizeof(unsigned long long)), "cudaMemset d_bad");
    if (local_elems_used > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>((local_elems_used + threads - 1) / threads);
        count_non_finite_kernel<<<blocks, threads>>>(d_A, local_elems_used, d_bad);
        distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaGetLastError(),
                                                            "count_non_finite_kernel launch");
    }

    unsigned long long h_bad = 0;
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(
        cudaMemcpy(&h_bad, d_bad, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
        "cudaMemcpy d_bad -> h_bad");
    unsigned long long total_bad = 0;
    MPI_Allreduce(&h_bad, &total_bad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (env.rank == 0) {
        const double overall_tflops =
            TflopsFromFlopsAndMs(QrFactorizationFlops(opts.m, opts.n), max_ms);
        spdlog::info(
            "Distributed blocked QR [col-blockcyclic-pipeline] ({}): m={} n={} nb={} "
            "block_cols={} update_tile(compat)={} row_block_rows={} trail_tile_cols={} "
            "row_block_mode={} skip_tail_update={} np={} avg {:.3f} ms overall {:.3f} TFLOP/s",
            DataTypeString<T>(), opts.m, opts.n, opts.nb, block_cols, opts.update_tile,
            opts.row_block_rows, opts.trail_tile_cols,
            distributed_qr_col_blockcyclic_pipeline::TailModeString(opts.row_block_mode),
            opts.skip_tail_update, env.size, max_ms, overall_tflops);
        if (opts.print_per_rank) {
            for (int r = 0; r < env.size; ++r) {
                spdlog::info("Per-rank time: rank {} -> {:.3f} ms", r, all_local_ms[r]);
            }
        }
        if (opts.print_phase_timing) {
            for (int r = 0; r < env.size; ++r) {
                spdlog::info(
                    "Per-rank summary: rank {} -> compute {:.3f} ms, comm {:.3f} ms, "
                    "comm_bytes {:.3f} GB, comm_bw {:.3f} GB/s, local_copy {:.3f} ms, "
                    "tail_wait {:.3f} ms, tail_gemm {:.3f} TFLOP/s",
                    r, all_compute_ms[r], all_comm_ms[r], all_comm_bytes[r] / 1.0e9,
                    all_comm_gbps[r], all_local_copy_ms[r], all_tail_wait_ms[r],
                    all_tail_gemm_tflops[r]);
                spdlog::info(
                    "Per-rank phase: rank {} -> panel {:.3f} ms, WY {:.3f} ms, merge {:.3f} ms, "
                    "inner_update {:.3f} ms, pack {:.3f} ms, unpack {:.3f} ms, comm {:.3f} ms, "
                    "tail_wait {:.3f} ms, tail_acc_gemm {:.3f} ms, tail_apply_gemm {:.3f} ms",
                    r, all_panel_ms[r], all_wy_ms[r], all_merge_ms[r], all_inner_update_ms[r],
                    all_pack_ms[r], all_unpack_ms[r], all_comm_ms[r], all_tail_wait_ms[r],
                    all_tail_acc_gemm_ms[r], all_tail_apply_gemm_ms[r]);
            }
        }
        if (total_bad > 0) {
            spdlog::error("Detected {} non-finite values after factorization.", total_bad);
        }
    }

    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaEventDestroy(timed_start),
                                                        "cudaEventDestroy timed_start");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaEventDestroy(timed_stop),
                                                        "cudaEventDestroy timed_stop");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaEventDestroy(timed_comm_done),
                                                        "cudaEventDestroy timed_comm_done");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(d_bad), "cudaFree d_bad");

    distributed_qr_col_blockcyclic_pipeline::AssertCublas(cublasDestroy(cublas_handle),
                                                          "cublasDestroy");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaStreamDestroy(compute_stream),
                                                        "cudaStreamDestroy compute");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaStreamDestroy(comm_stream),
                                                        "cudaStreamDestroy comm");

    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_r_panel),
                                                        "cudaFree ws.d_r_panel");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_tsqr_work_panel),
                                                        "cudaFree ws.d_tsqr_work_panel");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_panel_w),
                                                        "cudaFree ws.d_panel_w");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_panel_y),
                                                        "cudaFree ws.d_panel_y");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_block_w),
                                                        "cudaFree ws.d_block_w");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_block_y),
                                                        "cudaFree ws.d_block_y");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_block_w_rowmajor),
                                                        "cudaFree ws.d_block_w_rowmajor");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_block_y_rowmajor),
                                                        "cudaFree ws.d_block_y_rowmajor");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_rowblock_wy_packed),
                                                        "cudaFree ws.d_rowblock_wy_packed");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_tmp0), "cudaFree ws.d_tmp0");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(ws.d_tmp1), "cudaFree ws.d_tmp1");

    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(d_A), "cudaFree d_A");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(d_W), "cudaFree d_W");
    distributed_qr_col_blockcyclic_pipeline::AssertCuda(cudaFree(d_Y), "cudaFree d_Y");

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
    if (!opts.row_block_mode_valid) {
        if (env.rank == 0) {
            spdlog::error(
                "Invalid --row_block_mode value '{}'. Supported values: baseline, overlap.",
                opts.row_block_mode_value);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }
    if (opts.update_tile <= 0 || opts.update_tile % kPanelWidth != 0) {
        if (env.rank == 0) {
            spdlog::error("Invalid --update_tile {}. Require positive multiple of {}.",
                          opts.update_tile, kPanelWidth);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }
    if (opts.row_block_rows <= 0 || opts.trail_tile_cols <= 0) {
        if (env.rank == 0) {
            spdlog::error(
                "Invalid args: require row_block_rows > 0 and trail_tile_cols > 0 (got {} and {}).",
                opts.row_block_rows, opts.trail_tile_cols);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }

    const int local_ret = opts.use_double ? RunBenchmarkTyped<double>(env, opts, block_cols)
                                          : RunBenchmarkTyped<float>(env, opts, block_cols);

    finalize_nccl_if_needed(&env);
    finalize_mpi_if_needed(env);
    return local_ret;
}
