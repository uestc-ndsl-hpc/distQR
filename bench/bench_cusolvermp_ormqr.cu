#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverMp.h>
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
void FillDeviceRandom(T* device_data,
                      size_t count,
                      unsigned long long seed,
                      cudaStream_t stream) {
    if (count == 0) {
        return;
    }

    curandGenerator_t gen = nullptr;
    AssertCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT),
                 "curandCreateGenerator");
    AssertCurand(curandSetPseudoRandomGeneratorSeed(gen, seed),
                 "curandSetPseudoRandomGeneratorSeed");
    AssertCurand(curandSetStream(gen, stream), "curandSetStream");
    if constexpr (std::is_same_v<T, float>) {
        AssertCurand(curandGenerateUniform(gen, device_data, count), "curandGenerateUniform");
    } else {
        AssertCurand(curandGenerateUniformDouble(gen, device_data, count),
                     "curandGenerateUniformDouble");
    }
    AssertCurand(curandDestroyGenerator(gen), "curandDestroyGenerator");
}

template <typename T>
__global__ void set_local_identity_blockcyclic_kernel(int m,
                                                      int n,
                                                      int block_size,
                                                      int proc_row,
                                                      int proc_col,
                                                      int grid_rows,
                                                      int grid_cols,
                                                      int local_rows,
                                                      int local_cols,
                                                      T* A_local,
                                                      int lda_local) {
    const int local_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_col >= local_cols) {
        return;
    }

    const int local_col_block = local_col / block_size;
    const int col_in_block = local_col % block_size;
    const int global_col_block = local_col_block * grid_cols + proc_col;
    const int global_col = global_col_block * block_size + col_in_block;
    if (global_col >= n || global_col >= m) {
        return;
    }

    const int owner_row = global_col_block % grid_rows;
    if (owner_row != proc_row) {
        return;
    }

    const int local_row_block = global_col_block / grid_rows;
    const int local_row = local_row_block * block_size + col_in_block;
    if (local_row >= local_rows) {
        return;
    }

    A_local[static_cast<size_t>(local_row) + static_cast<size_t>(local_col) * lda_local] =
        static_cast<T>(1);
}

template <typename T>
void SetLocalDistributedIdentity(int m,
                                 int n,
                                 int block_size,
                                 int proc_row,
                                 int proc_col,
                                 int grid_rows,
                                 int grid_cols,
                                 int local_rows,
                                 int local_cols,
                                 T* d_A_local,
                                 int lda_local,
                                 cudaStream_t stream) {
    const size_t elems =
        static_cast<size_t>(lda_local) * static_cast<size_t>(std::max(local_cols, 1));
    AssertCuda(cudaMemsetAsync(d_A_local, 0, elems * sizeof(T), stream),
               "cudaMemsetAsync distributed identity");
    if (local_cols <= 0 || local_rows <= 0) {
        return;
    }

    constexpr int kThreads = 256;
    const int blocks = (local_cols + kThreads - 1) / kThreads;
    set_local_identity_blockcyclic_kernel<T><<<blocks, kThreads, 0, stream>>>(
        m, n, block_size, proc_row, proc_col, grid_rows, grid_cols, local_rows, local_cols,
        d_A_local, lda_local);
    AssertCuda(cudaGetLastError(), "set_local_identity_blockcyclic_kernel launch");
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

double OrmqrLeftFlops(int m, int ncols, int k) {
    const double md = static_cast<double>(m);
    const double nd = static_cast<double>(ncols);
    const double kd = static_cast<double>(k);
    return 4.0 * nd * (kd * md - 0.5 * kd * (kd - 1.0));
}

double QrFlops(int m, int n) {
    const double md = static_cast<double>(m);
    const double nd = static_cast<double>(n);
    return 2.0 * md * nd * nd - (2.0 / 3.0) * nd * nd * nd;
}

double OrgqrFlops(int m, int n) {
    // Match bench_qr's normalization when ORMQR is used to form explicit Q
    // from an identity C with c_cols == n.
    return QrFlops(m, n);
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
    int c_cols = 0;
    int grid_block_size = 1024;
    int warmup = 1;
    int iters = 5;
    int grid_rows = 0;
    int grid_cols = 0;
    bool random_c = false;
    bool e2e = false;
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
        } else if ((std::strcmp(argv[i], "--c-cols") == 0 ||
                    std::strcmp(argv[i], "--c_cols") == 0 ||
                    std::strcmp(argv[i], "--rhs") == 0 ||
                    std::strcmp(argv[i], "--nrhs") == 0) &&
                   i + 1 < argc) {
            opts.c_cols = std::atoi(argv[++i]);
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
        } else if (std::strcmp(argv[i], "--random-c") == 0 ||
                   std::strcmp(argv[i], "--random_c") == 0) {
            opts.random_c = true;
        } else if (std::strcmp(argv[i], "--e2e") == 0) {
            opts.e2e = true;
        } else if (std::strcmp(argv[i], "--ormqr-only") == 0 ||
                   std::strcmp(argv[i], "--q-only") == 0) {
            opts.e2e = false;
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
                  int c_cols,
                  int grid_block_size) {
    const int proc_row = env.rank % grid_rows;
    const int proc_col = env.rank / grid_rows;

    const int64_t local_rows_a = cusolverMpNUMROC(
        static_cast<int64_t>(m), static_cast<int64_t>(grid_block_size), proc_row, 0, grid_rows);
    const int64_t local_cols_a = cusolverMpNUMROC(
        static_cast<int64_t>(n), static_cast<int64_t>(grid_block_size), proc_col, 0, grid_cols);
    const int64_t lda_local_a = std::max<int64_t>(1, local_rows_a);
    const size_t local_elems_a =
        static_cast<size_t>(lda_local_a) * static_cast<size_t>(std::max<int64_t>(1, local_cols_a));
    const size_t local_bytes_a = local_elems_a * sizeof(T);
    const size_t local_elems_a_used =
        static_cast<size_t>(std::max<int64_t>(local_rows_a, 0)) *
        static_cast<size_t>(std::max<int64_t>(local_cols_a, 0));

    const int64_t local_rows_c = cusolverMpNUMROC(
        static_cast<int64_t>(m), static_cast<int64_t>(grid_block_size), proc_row, 0, grid_rows);
    const int64_t local_cols_c = cusolverMpNUMROC(
        static_cast<int64_t>(c_cols), static_cast<int64_t>(grid_block_size), proc_col, 0,
        grid_cols);
    const int64_t lda_local_c = std::max<int64_t>(1, local_rows_c);
    const size_t local_elems_c =
        static_cast<size_t>(lda_local_c) * static_cast<size_t>(std::max<int64_t>(1, local_cols_c));
    const size_t local_bytes_c = local_elems_c * sizeof(T);
    const size_t local_elems_c_used =
        static_cast<size_t>(std::max<int64_t>(local_rows_c, 0)) *
        static_cast<size_t>(std::max<int64_t>(local_cols_c, 0));

    const int64_t local_tau = cusolverMpNUMROC(
        static_cast<int64_t>(n), static_cast<int64_t>(grid_block_size), proc_col, 0, grid_cols);
    const size_t tau_elems = static_cast<size_t>(std::max<int64_t>(1, local_tau));
    const size_t tau_bytes = tau_elems * sizeof(T);

    T* d_A0 = nullptr;
    T* d_Afact = nullptr;
    T* d_C0 = nullptr;
    T* d_C = nullptr;
    T* d_tau = nullptr;
    int* d_info = nullptr;
    void* d_work_geqrf = nullptr;
    void* d_work_ormqr = nullptr;

    AssertCuda(cudaMalloc(&d_A0, local_bytes_a), "cudaMalloc d_A0");
    AssertCuda(cudaMalloc(&d_Afact, local_bytes_a), "cudaMalloc d_Afact");
    AssertCuda(cudaMalloc(&d_C0, local_bytes_c), "cudaMalloc d_C0");
    AssertCuda(cudaMalloc(&d_C, local_bytes_c), "cudaMalloc d_C");
    AssertCuda(cudaMalloc(&d_tau, tau_bytes), "cudaMalloc d_tau");
    AssertCuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc d_info");

    cusolverMpMatrixDescriptor_t descA = nullptr;
    cusolverMpMatrixDescriptor_t descC = nullptr;
    AssertCusolver(
        cusolverMpCreateMatrixDesc(&descA, grid, CudaDataTypeValue<T>(), static_cast<int64_t>(m),
                                   static_cast<int64_t>(n), static_cast<int64_t>(grid_block_size),
                                   static_cast<int64_t>(grid_block_size), 0, 0, lda_local_a),
        "cusolverMpCreateMatrixDesc descA");
    AssertCusolver(
        cusolverMpCreateMatrixDesc(&descC, grid, CudaDataTypeValue<T>(), static_cast<int64_t>(m),
                                   static_cast<int64_t>(c_cols),
                                   static_cast<int64_t>(grid_block_size),
                                   static_cast<int64_t>(grid_block_size), 0, 0, lda_local_c),
        "cusolverMpCreateMatrixDesc descC");

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
                       static_cast<int64_t>(c_cols), static_cast<int64_t>(n), d_Afact, 1, 1,
                       descA, d_tau, d_C, 1, 1, descC, CudaDataTypeValue<T>(),
                       &ormqr_work_bytes_device, &ormqr_work_bytes_host),
                   "cusolverMpOrmqr_bufferSize");
    if (ormqr_work_bytes_device > 0) {
        AssertCuda(cudaMalloc(&d_work_ormqr, ormqr_work_bytes_device), "cudaMalloc d_work_ormqr");
    }
    std::vector<unsigned char> host_work_ormqr(std::max<size_t>(ormqr_work_bytes_host, 1), 0);

    cudaStream_t stream = nullptr;
    AssertCusolver(cusolverMpGetStream(mp_handle, &stream), "cusolverMpGetStream");

    FillDeviceRandom(d_A0, local_elems_a_used,
                     20260316ULL + static_cast<unsigned long long>(env.rank), stream);
    if (opts.random_c) {
        FillDeviceRandom(d_C0, local_elems_c_used,
                         20260316ULL + 1000003ULL + static_cast<unsigned long long>(env.rank),
                         stream);
    } else {
        SetLocalDistributedIdentity<T>(m, c_cols, grid_block_size, proc_row, proc_col, grid_rows,
                                       grid_cols, static_cast<int>(local_rows_c),
                                       static_cast<int>(local_cols_c), d_C0,
                                       static_cast<int>(lda_local_c), stream);
        AssertCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize identity init");
    }

    AssertCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info precompute");
    AssertCuda(cudaMemcpyAsync(d_Afact, d_A0, local_bytes_a, cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync precompute Afact <- A0");
    const cusolverStatus_t geqrf_st = cusolverMpGeqrf(
        mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_Afact, 1, 1, descA, d_tau,
        CudaDataTypeValue<T>(), d_work_geqrf, geqrf_work_bytes_device, host_work_geqrf.data(),
        geqrf_work_bytes_host, d_info);
    if (geqrf_st != CUSOLVER_STATUS_SUCCESS) {
        int h_info = -777777;
        cudaError_t copy_st = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        spdlog::error("cusolverMpGeqrf precompute failed: status={} info={} memcpy_status={}",
                      static_cast<int>(geqrf_st), h_info, static_cast<int>(copy_st));
        std::exit(1);
    }
    AssertCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize precompute");

    int info = 0;
    AssertCuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
               "cudaMemcpy d_info->info precompute");
    int local_info_abs = std::abs(info);

    for (int i = 0; i < opts.warmup; ++i) {
        AssertCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info warmup");
        if (opts.e2e) {
            AssertCuda(cudaMemcpyAsync(d_Afact, d_A0, local_bytes_a, cudaMemcpyDeviceToDevice,
                                       stream),
                       "cudaMemcpyAsync warmup Afact <- A0");
            const cusolverStatus_t geqrf_warmup_st = cusolverMpGeqrf(
                mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_Afact, 1, 1, descA,
                d_tau, CudaDataTypeValue<T>(), d_work_geqrf, geqrf_work_bytes_device,
                host_work_geqrf.data(), geqrf_work_bytes_host, d_info);
            if (geqrf_warmup_st != CUSOLVER_STATUS_SUCCESS) {
                int h_info = -777777;
                cudaError_t copy_st =
                    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
                spdlog::error(
                    "cusolverMpGeqrf warmup failed: status={} info={} memcpy_status={}",
                    static_cast<int>(geqrf_warmup_st), h_info, static_cast<int>(copy_st));
                std::exit(1);
            }
            AssertCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info warmup ormqr");
        }
        AssertCuda(cudaMemcpyAsync(d_C, d_C0, local_bytes_c, cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync warmup C <- C0");
        const cusolverStatus_t ormqr_st = cusolverMpOrmqr(
            mp_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, static_cast<int64_t>(m),
            static_cast<int64_t>(c_cols), static_cast<int64_t>(n), d_Afact, 1, 1, descA, d_tau,
            d_C, 1, 1, descC, CudaDataTypeValue<T>(), d_work_ormqr, ormqr_work_bytes_device,
            host_work_ormqr.data(), ormqr_work_bytes_host, d_info);
        if (ormqr_st != CUSOLVER_STATUS_SUCCESS) {
            int h_info = -777777;
            cudaError_t copy_st = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
            spdlog::error("cusolverMpOrmqr warmup failed: status={} info={} memcpy_status={}",
                          static_cast<int>(ormqr_st), h_info, static_cast<int>(copy_st));
            std::exit(1);
        }
        AssertCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");
        AssertCuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy d_info->info warmup");
        local_info_abs = std::max(local_info_abs, std::abs(info));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    AssertCuda(cudaEventCreate(&start), "cudaEventCreate start");
    AssertCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    MPI_Barrier(MPI_COMM_WORLD);

    float local_total_ms = 0.0f;
    for (int i = 0; i < opts.iters; ++i) {
        AssertCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info timed");
        if (opts.e2e) {
            AssertCuda(cudaMemcpyAsync(d_Afact, d_A0, local_bytes_a, cudaMemcpyDeviceToDevice,
                                       stream),
                       "cudaMemcpyAsync timed Afact <- A0");
        }
        AssertCuda(cudaMemcpyAsync(d_C, d_C0, local_bytes_c, cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync timed C <- C0");
        AssertCuda(cudaEventRecord(start, stream), "cudaEventRecord start");
        if (opts.e2e) {
            const cusolverStatus_t geqrf_timed_st = cusolverMpGeqrf(
                mp_handle, static_cast<int64_t>(m), static_cast<int64_t>(n), d_Afact, 1, 1, descA,
                d_tau, CudaDataTypeValue<T>(), d_work_geqrf, geqrf_work_bytes_device,
                host_work_geqrf.data(), geqrf_work_bytes_host, d_info);
            if (geqrf_timed_st != CUSOLVER_STATUS_SUCCESS) {
                int h_info = -777777;
                cudaError_t copy_st =
                    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
                spdlog::error("cusolverMpGeqrf timed failed: status={} info={} memcpy_status={}",
                              static_cast<int>(geqrf_timed_st), h_info,
                              static_cast<int>(copy_st));
                std::exit(1);
            }
            AssertCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info timed ormqr");
        }
        const cusolverStatus_t ormqr_st = cusolverMpOrmqr(
            mp_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, static_cast<int64_t>(m),
            static_cast<int64_t>(c_cols), static_cast<int64_t>(n), d_Afact, 1, 1, descA, d_tau,
            d_C, 1, 1, descC, CudaDataTypeValue<T>(), d_work_ormqr, ormqr_work_bytes_device,
            host_work_ormqr.data(), ormqr_work_bytes_host, d_info);
        if (ormqr_st != CUSOLVER_STATUS_SUCCESS) {
            int h_info = -777777;
            cudaError_t copy_st = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
            spdlog::error("cusolverMpOrmqr timed failed: status={} info={} memcpy_status={}",
                          static_cast<int>(ormqr_st), h_info, static_cast<int>(copy_st));
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
        const bool use_orgqr_norm = !opts.random_c && c_cols == n;
        const double qr_flops = QrFlops(m, n);
        const double apply_flops =
            use_orgqr_norm ? OrgqrFlops(m, n) : OrmqrLeftFlops(m, c_cols, n);
        const double flops = opts.e2e ? (qr_flops + apply_flops) : apply_flops;
        const double tflops = FlopsToTflops(flops, max_ms);
        spdlog::info(
            "cuSOLVERMp {} [{}]: m={} n={} c_cols={} grid_block={} grid={}x{} C={} flops_norm={} "
            "avg {:.3f} ms ({:.3f} TFLOPS) info={}",
            opts.e2e ? "GEQRF+ORMQR" : "ORMQR",
            DataTypeString<T>(), m, n, c_cols, grid_block_size, grid_rows, grid_cols,
            opts.random_c ? "random" : "identity",
            use_orgqr_norm ? "geqrf+orgqr" : "ormqr", max_ms, tflops, global_info_abs);
    }

    AssertCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    AssertCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    AssertCusolver(cusolverMpDestroyMatrixDesc(descC), "cusolverMpDestroyMatrixDesc descC");
    AssertCusolver(cusolverMpDestroyMatrixDesc(descA), "cusolverMpDestroyMatrixDesc descA");
    if (d_work_geqrf) {
        AssertCuda(cudaFree(d_work_geqrf), "cudaFree d_work_geqrf");
    }
    if (d_work_ormqr) {
        AssertCuda(cudaFree(d_work_ormqr), "cudaFree d_work_ormqr");
    }
    AssertCuda(cudaFree(d_info), "cudaFree d_info");
    AssertCuda(cudaFree(d_tau), "cudaFree d_tau");
    AssertCuda(cudaFree(d_C), "cudaFree d_C");
    AssertCuda(cudaFree(d_C0), "cudaFree d_C0");
    AssertCuda(cudaFree(d_Afact), "cudaFree d_Afact");
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
                const int c_cols = (opts.c_cols > 0) ? opts.c_cols : n;
                if (m <= 0 || n <= 0 || c_cols <= 0 || grid_block_size <= 0 || m < n) {
                    if (env.rank == 0) {
                        spdlog::warn(
                            "Skip invalid case m={} n={} c_cols={} grid_block={} "
                            "(require m>=n and >0)",
                            m, n, c_cols, grid_block_size);
                    }
                    continue;
                }
                ++executed_cases;
                const int ret = RunSingleCase<T>(env, mp_handle, grid, opts, grid_rows, grid_cols,
                                                 m, n, c_cols, grid_block_size);
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
    if (opts.c_cols < 0) {
        if (env.rank == 0) {
            spdlog::error("Invalid --c-cols value '{}'. Require c_cols >= 0.", opts.c_cols);
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
            "cuSOLVERMp {} bench: type={} warmup={} iters={} grid={}x{} m={} n={} c_cols={} "
            "grid_block={} C={} m_scan='{}' n_scan='{}' grid_block_scan='{}'",
            opts.e2e ? "GEQRF+ORMQR" : "ORMQR",
            opts.use_double ? "double" : "float", opts.warmup, opts.iters, grid_rows, grid_cols,
            opts.m, opts.n, (opts.c_cols > 0) ? opts.c_cols : opts.n, opts.grid_block_size,
            opts.random_c ? "random" : "identity",
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
