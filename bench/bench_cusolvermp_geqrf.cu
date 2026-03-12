#include <cuda_runtime.h>
#include <cusolverMp.h>
#include <curand.h>
#include <mpi.h>
#include <nccl.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "components/resourse_initial.cuh"

namespace {

void AssertCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        spdlog::error("{}: {}", context, cudaGetErrorString(status));
        std::exit(1);
    }
}

void AssertCusolver(cusolverStatus_t status, const char* context) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        spdlog::error("{}: cusolver error {}", context, static_cast<int>(status));
        std::exit(1);
    }
}

void AssertCurand(curandStatus_t status, const char* context) {
    if (status != CURAND_STATUS_SUCCESS) {
        spdlog::error("{}: curand error {}", context, static_cast<int>(status));
        std::exit(1);
    }
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
    } else {
        AssertCurand(curandGenerateUniformDouble(gen, device_data, count),
                     "curandGenerateUniformDouble");
    }
    AssertCurand(curandDestroyGenerator(gen), "curandDestroyGenerator");
}

double QrFlops(int m, int n) {
    const double md = static_cast<double>(m);
    const double nd = static_cast<double>(n);
    return 2.0 * md * nd * nd - (2.0 / 3.0) * nd * nd * nd;
}

double FlopsToTflops(double flops, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return flops / (ms * 1.0e-3) / 1.0e12;
}

struct Options {
    int m = 16384;
    int n = 1024;
    int grid_block_size = 1024;
    int warmup = 1;
    int iters = 5;
    int grid_rows = 0;
    int grid_cols = 0;
    bool use_double = false;
    bool type_valid = true;
    std::string type_value = "float";
    std::vector<int> m_scan;
    std::vector<int> n_scan;
    std::vector<int> grid_block_scan;
    bool m_scan_valid = true;
    bool n_scan_valid = true;
    bool grid_block_scan_valid = true;
    std::string m_scan_value;
    std::string n_scan_value;
    std::string grid_block_scan_value;
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

std::vector<int> ParsePositiveIntList(const char* value, bool* valid) {
    std::vector<int> out;
    *valid = false;
    if (!value || value[0] == '\0') {
        return out;
    }

    const char* cursor = value;
    while (*cursor != '\0') {
        char* end_ptr = nullptr;
        const long parsed = std::strtol(cursor, &end_ptr, 10);
        if (end_ptr == cursor || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
            return out;
        }
        out.push_back(static_cast<int>(parsed));
        if (*end_ptr == '\0') {
            *valid = true;
            return out;
        }
        if (*end_ptr != ',') {
            return out;
        }
        cursor = end_ptr + 1;
        if (*cursor == '\0') {
            return out;
        }
    }

    *valid = true;
    return out;
}

Options ParseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            opts.m = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            opts.n = std::atoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "--grid-block-size") == 0 ||
                    std::strcmp(argv[i], "--grid_block_size") == 0 ||
                    std::strcmp(argv[i], "--nb") == 0) &&
                   i + 1 < argc) {
            opts.grid_block_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opts.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opts.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            opts.type_value = argv[++i];
            opts.type_valid = ParseType(opts.type_value.c_str(), &opts.use_double);
        } else if (std::strcmp(argv[i], "--grid-rows") == 0 && i + 1 < argc) {
            opts.grid_rows = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--grid-cols") == 0 && i + 1 < argc) {
            opts.grid_cols = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--m-scan") == 0 && i + 1 < argc) {
            opts.m_scan_value = argv[++i];
            opts.m_scan = ParsePositiveIntList(opts.m_scan_value.c_str(), &opts.m_scan_valid);
        } else if (std::strcmp(argv[i], "--n-scan") == 0 && i + 1 < argc) {
            opts.n_scan_value = argv[++i];
            opts.n_scan = ParsePositiveIntList(opts.n_scan_value.c_str(), &opts.n_scan_valid);
        } else if ((std::strcmp(argv[i], "--grid-block-scan") == 0 ||
                    std::strcmp(argv[i], "--grid_block_scan") == 0 ||
                    std::strcmp(argv[i], "--nb-scan") == 0) &&
                   i + 1 < argc) {
            opts.grid_block_scan_value = argv[++i];
            opts.grid_block_scan = ParsePositiveIntList(opts.grid_block_scan_value.c_str(),
                                                        &opts.grid_block_scan_valid);
        }
    }
    return opts;
}

bool ResolveGridDims(const Options& opts, int world_size, int* out_rows, int* out_cols) {
    int rows = opts.grid_rows;
    int cols = opts.grid_cols;
    if (rows <= 0 && cols <= 0) {
        rows = world_size;
        cols = 1;
    } else if (rows <= 0) {
        if (cols <= 0 || world_size % cols != 0) {
            return false;
        }
        rows = world_size / cols;
    } else if (cols <= 0) {
        if (world_size % rows != 0) {
            return false;
        }
        cols = world_size / rows;
    }

    if (rows <= 0 || cols <= 0 || rows * cols != world_size) {
        return false;
    }

    *out_rows = rows;
    *out_cols = cols;
    return true;
}

template <typename T>
int RunSingleCase(const MpiCudaEnv& env,
                  cusolverMpHandle_t mp_handle,
                  cusolverMpGrid_t grid,
                  const Options& opts,
                  int grid_rows,
                  int grid_cols,
                  int m,
                  int n,
                  int grid_block_size) {
    const int proc_row = env.rank % grid_rows;
    const int proc_col = env.rank / grid_rows;

    const int64_t local_rows = cusolverMpNUMROC(
        static_cast<int64_t>(m), static_cast<int64_t>(grid_block_size), proc_row, 0, grid_rows);
    const int64_t local_cols = cusolverMpNUMROC(
        static_cast<int64_t>(n), static_cast<int64_t>(grid_block_size), proc_col, 0, grid_cols);
    const int64_t lda_local = std::max<int64_t>(1, local_rows);
    const size_t local_elems =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max<int64_t>(1, local_cols));
    const size_t local_bytes = local_elems * sizeof(T);

    const int64_t local_tau = cusolverMpNUMROC(
        static_cast<int64_t>(n), static_cast<int64_t>(grid_block_size), proc_col, 0, grid_cols);
    const size_t tau_elems = static_cast<size_t>(std::max<int64_t>(1, local_tau));
    const size_t tau_bytes = tau_elems * sizeof(T);

    T* d_A0 = nullptr;
    T* d_A = nullptr;
    T* d_tau = nullptr;
    int* d_info = nullptr;
    void* d_work = nullptr;

    AssertCuda(cudaMalloc(&d_A0, local_bytes), "cudaMalloc d_A0");
    AssertCuda(cudaMalloc(&d_A, local_bytes), "cudaMalloc d_A");
    AssertCuda(cudaMalloc(&d_tau, tau_bytes), "cudaMalloc d_tau");
    AssertCuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc d_info");

    cusolverMpMatrixDescriptor_t descA = nullptr;
    AssertCusolver(
        cusolverMpCreateMatrixDesc(&descA, grid, CudaDataTypeValue<T>(), static_cast<int64_t>(m),
                                   static_cast<int64_t>(n), static_cast<int64_t>(grid_block_size),
                                   static_cast<int64_t>(grid_block_size), 0, 0, lda_local),
        "cusolverMpCreateMatrixDesc");

    size_t work_bytes_device = 0;
    size_t work_bytes_host = 0;
    AssertCusolver(cusolverMpGeqrf_bufferSize(
                       mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_A, 1, 1,
                       descA, CudaDataTypeValue<T>(), &work_bytes_device, &work_bytes_host),
                   "cusolverMpGeqrf_bufferSize");
    if (work_bytes_device > 0) {
        AssertCuda(cudaMalloc(&d_work, work_bytes_device), "cudaMalloc d_work");
    }
    const size_t host_alloc_bytes = std::max<size_t>(work_bytes_host, 1);
    std::vector<unsigned char> host_work(host_alloc_bytes, 0);
    void* h_work_ptr = host_work.data();

    cudaStream_t stream = nullptr;
    AssertCusolver(cusolverMpGetStream(mp_handle, &stream), "cusolverMpGetStream");

    FillDeviceRandom(
        d_A0, local_elems,
        20260305ULL + static_cast<unsigned long long>(env.rank) * 1315423911ULL);
    AssertCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize local random init");

    int info = 0;
    for (int i = 0; i < opts.warmup; ++i) {
        AssertCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info warmup");
        AssertCuda(cudaMemcpyAsync(d_A, d_A0, local_bytes, cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync warmup A <- A0");
        const cusolverStatus_t geqrf_st = cusolverMpGeqrf(
            mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_A, 1, 1, descA, d_tau,
            CudaDataTypeValue<T>(), d_work, work_bytes_device, h_work_ptr, work_bytes_host, d_info);
        if (geqrf_st != CUSOLVER_STATUS_SUCCESS) {
            int h_info = -777777;
            cudaError_t copy_st = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
            spdlog::error("cusolverMpGeqrf warmup failed: status={} info={} memcpy_status={}",
                          static_cast<int>(geqrf_st), h_info, static_cast<int>(copy_st));
            std::exit(1);
        }
        AssertCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");
        AssertCuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy d_info->info warmup");
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    AssertCuda(cudaEventCreate(&start), "cudaEventCreate start");
    AssertCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    MPI_Barrier(MPI_COMM_WORLD);

    float local_total_ms = 0.0f;
    int local_info_abs = std::abs(info);
    for (int i = 0; i < opts.iters; ++i) {
        AssertCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info timed");
        AssertCuda(cudaMemcpyAsync(d_A, d_A0, local_bytes, cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync timed A <- A0");
        AssertCuda(cudaEventRecord(start, stream), "cudaEventRecord start");
        const cusolverStatus_t geqrf_st = cusolverMpGeqrf(
            mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_A, 1, 1, descA, d_tau,
            CudaDataTypeValue<T>(), d_work, work_bytes_device, h_work_ptr, work_bytes_host, d_info);
        if (geqrf_st != CUSOLVER_STATUS_SUCCESS) {
            int h_info = -777777;
            cudaError_t copy_st = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
            spdlog::error("cusolverMpGeqrf timed failed: status={} info={} memcpy_status={}",
                          static_cast<int>(geqrf_st), h_info, static_cast<int>(copy_st));
            std::exit(1);
        }
        AssertCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
        AssertCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
        AssertCuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy d_info->info timed");
        float iter_ms = 0.0f;
        AssertCuda(cudaEventElapsedTime(&iter_ms, start, stop), "cudaEventElapsedTime");
        local_total_ms += iter_ms;
        local_info_abs = std::max(local_info_abs, std::abs(info));
    }

    const double local_ms = static_cast<double>(local_total_ms) / static_cast<double>(opts.iters);
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int global_info_abs = 0;
    MPI_Allreduce(&local_info_abs, &global_info_abs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (env.rank == 0) {
        const double tflops = FlopsToTflops(QrFlops(m, n), max_ms);
        spdlog::info(
            "cuSOLVERMp GEQRF [{}]: m={} n={} grid_block={} grid={}x{} avg {:.3f} ms ({:.3f} "
            "TFLOPS) info={}",
            DataTypeString<T>(), m, n, grid_block_size, grid_rows, grid_cols, max_ms, tflops,
            global_info_abs);
    }

    AssertCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    AssertCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    AssertCusolver(cusolverMpDestroyMatrixDesc(descA), "cusolverMpDestroyMatrixDesc");
    if (d_work) {
        AssertCuda(cudaFree(d_work), "cudaFree d_work");
    }
    AssertCuda(cudaFree(d_info), "cudaFree d_info");
    AssertCuda(cudaFree(d_tau), "cudaFree d_tau");
    AssertCuda(cudaFree(d_A), "cudaFree d_A");
    AssertCuda(cudaFree(d_A0), "cudaFree d_A0");

    return (global_info_abs == 0) ? 0 : 1;
}

template <typename T>
int RunBenchmarkTyped(const MpiCudaEnv& env, const Options& opts, int grid_rows, int grid_cols) {
    cudaStream_t stream = nullptr;
    AssertCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    cusolverMpHandle_t mp_handle = nullptr;
    AssertCusolver(cusolverMpCreate(&mp_handle, env.device_id, stream), "cusolverMpCreate");

    cusolverMpGrid_t grid = nullptr;
    AssertCusolver(cusolverMpCreateDeviceGrid(mp_handle, &grid, env.nccl_comm, grid_rows, grid_cols,
                                              CUSOLVERMP_GRID_MAPPING_COL_MAJOR),
                   "cusolverMpCreateDeviceGrid");

    const std::vector<int> m_values = opts.m_scan.empty() ? std::vector<int>{opts.m} : opts.m_scan;
    const std::vector<int> n_values = opts.n_scan.empty() ? std::vector<int>{opts.n} : opts.n_scan;
    const std::vector<int> grid_block_values = opts.grid_block_scan.empty()
                                                   ? std::vector<int>{opts.grid_block_size}
                                                   : opts.grid_block_scan;

    int executed_cases = 0;
    int global_ret = 0;
    for (int m : m_values) {
        for (int n : n_values) {
            for (int grid_block_size : grid_block_values) {
                if (m <= 0 || n <= 0 || grid_block_size <= 0 || m < n) {
                    if (env.rank == 0) {
                        spdlog::warn(
                            "Skip invalid case m={} n={} grid_block={} (require m>=n and >0)", m, n,
                            grid_block_size);
                    }
                    continue;
                }
                ++executed_cases;
                const int ret = RunSingleCase<T>(env, mp_handle, grid, opts, grid_rows, grid_cols,
                                                 m, n, grid_block_size);
                global_ret = std::max(global_ret, ret);
            }
        }
    }

    if (executed_cases == 0) {
        if (env.rank == 0) {
            spdlog::error("No valid benchmark cases to run.");
        }
        global_ret = 1;
    }

    AssertCusolver(cusolverMpDestroyGrid(grid), "cusolverMpDestroyGrid");
    AssertCusolver(cusolverMpDestroy(mp_handle), "cusolverMpDestroy");
    AssertCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return global_ret;
}

}  // namespace

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::info);

    const Options opts = ParseArgs(argc, argv);

    MPI_Init(&argc, &argv);
    MpiCudaEnv env{};
    env.mpi_initialized_here = true;
    MPI_Comm_rank(MPI_COMM_WORLD, &env.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &env.size);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &env.local_rank);
    MPI_Comm_size(local_comm, &env.local_size);
    MPI_Comm_free(&local_comm);

    AssertCuda(cudaGetDeviceCount(&env.device_count), "cudaGetDeviceCount");
    if (env.device_count <= 0) {
        if (env.rank == 0) {
            spdlog::error("No CUDA devices detected.");
        }
        MPI_Finalize();
        return 1;
    }

    env.device_id = env.local_rank;
    AssertCuda(cudaSetDevice(env.device_id), "cudaSetDevice");
    AssertCuda(cudaFree(nullptr), "cudaFree(nullptr)");

    ncclUniqueId id{};
    if (env.rank == 0) {
        ncclResult_t st = ncclGetUniqueId(&id);
        if (st != ncclSuccess) {
            spdlog::error("ncclGetUniqueId failed: {}", ncclGetErrorString(st));
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclResult_t init_res = ncclCommInitRank(&env.nccl_comm, env.size, id, env.rank);
    if (init_res != ncclSuccess) {
        if (env.rank == 0) {
            spdlog::error("ncclCommInitRank failed: {}", ncclGetErrorString(init_res));
        }
        MPI_Finalize();
        return 1;
    }
    env.nccl_initialized = true;

    if (opts.warmup < 0 || opts.iters <= 0) {
        if (env.rank == 0) {
            spdlog::error(
                "Invalid args: require warmup >= 0 and iters > 0 (got warmup={} iters={})",
                opts.warmup, opts.iters);
        }
        ncclCommDestroy(env.nccl_comm);
        MPI_Finalize();
        return 1;
    }
    if (!opts.type_valid) {
        if (env.rank == 0) {
            spdlog::error("Invalid --type value '{}'. Supported values: float, double.",
                          opts.type_value);
        }
        ncclCommDestroy(env.nccl_comm);
        MPI_Finalize();
        return 1;
    }
    if (!opts.m_scan_valid) {
        if (env.rank == 0) {
            spdlog::error("Invalid --m-scan value '{}'. Use comma-separated positive integers.",
                          opts.m_scan_value);
        }
        ncclCommDestroy(env.nccl_comm);
        MPI_Finalize();
        return 1;
    }
    if (!opts.n_scan_valid) {
        if (env.rank == 0) {
            spdlog::error("Invalid --n-scan value '{}'. Use comma-separated positive integers.",
                          opts.n_scan_value);
        }
        ncclCommDestroy(env.nccl_comm);
        MPI_Finalize();
        return 1;
    }
    if (!opts.grid_block_scan_valid) {
        if (env.rank == 0) {
            spdlog::error(
                "Invalid --grid-block-scan value '{}'. Use comma-separated positive integers.",
                opts.grid_block_scan_value);
        }
        ncclCommDestroy(env.nccl_comm);
        MPI_Finalize();
        return 1;
    }

    int grid_rows = 0;
    int grid_cols = 0;
    if (!ResolveGridDims(opts, env.size, &grid_rows, &grid_cols)) {
        if (env.rank == 0) {
            spdlog::error(
                "Invalid process grid: world_size={} grid_rows={} grid_cols={}. "
                "Require grid_rows*grid_cols == world_size.",
                env.size, opts.grid_rows, opts.grid_cols);
        }
        ncclCommDestroy(env.nccl_comm);
        MPI_Finalize();
        return 1;
    }

    if (env.rank == 0) {
        spdlog::info(
            "cuSOLVERMp bench: type={} warmup={} iters={} grid={}x{} m={} n={} grid_block={} "
            "m_scan='{}' n_scan='{}' grid_block_scan='{}'",
            opts.use_double ? "double" : "float", opts.warmup, opts.iters, grid_rows, grid_cols,
            opts.m, opts.n, opts.grid_block_size,
            opts.m_scan_value.empty() ? "-" : opts.m_scan_value,
            opts.n_scan_value.empty() ? "-" : opts.n_scan_value,
            opts.grid_block_scan_value.empty() ? "-" : opts.grid_block_scan_value);
    }

    const int local_ret = opts.use_double
                              ? RunBenchmarkTyped<double>(env, opts, grid_rows, grid_cols)
                              : RunBenchmarkTyped<float>(env, opts, grid_rows, grid_cols);

    ncclCommDestroy(env.nccl_comm);
    MPI_Finalize();
    return local_ret;
}
