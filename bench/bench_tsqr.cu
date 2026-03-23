#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>

#include "components/panel_process.cuh"
#include "utils/cublas_gemm_traits.cuh"

namespace {

constexpr int kTsqrN = 32;
constexpr int kLegacyTsqrBlockSize = 256;
constexpr int kLegacyTsqrBlockDimX = 32;
constexpr int kLegacyTsqrBlockDimY = 32;
constexpr int kLegacyTsqrRowsPerThread = 8;

void AssertCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", context, cudaGetErrorString(status));
        std::exit(1);
    }
}

void AssertCublas(cublasStatus_t status, const char* context) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cublas error %d\n", context, static_cast<int>(status));
        std::exit(1);
    }
}

void AssertCusolver(cusolverStatus_t status, const char* context) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cusolver error %d\n", context, static_cast<int>(status));
        std::exit(1);
    }
}

void AssertCurand(curandStatus_t status, const char* context) {
    if (status != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: curand error %d\n", context, static_cast<int>(status));
        std::exit(1);
    }
}

template <typename T>
void FillDeviceRandom(T* device_data, size_t count, unsigned long long seed) {
    curandGenerator_t gen;
    AssertCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
    AssertCurand(curandSetPseudoRandomGeneratorSeed(gen, seed),
                 "curandSetPseudoRandomGeneratorSeed");
    if constexpr (std::is_same_v<T, float>) {
        AssertCurand(curandGenerateUniform(gen, device_data, count), "curandGenerateUniform");
    } else if constexpr (std::is_same_v<T, double>) {
        AssertCurand(curandGenerateUniformDouble(gen, device_data, count),
                     "curandGenerateUniformDouble");
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Only float/double supported.");
    }
    AssertCurand(curandDestroyGenerator(gen), "curandDestroyGenerator");
}

template <typename T>
void WarmupGemm(cublasHandle_t handle,
                int m,
                int n,
                int k,
                const T* A,
                int lda,
                const T* B,
                int ldb,
                T* C,
                int ldc,
                int iters) {
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    for (int i = 0; i < iters; ++i) {
        AssertCublas(CublasGemmTraits<T>::Gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A,
                                               lda, B, ldb, &zero, C, ldc),
                     "cublasGemm warmup");
    }
}

float TimeKernelMs(const std::function<void()>& setup, const std::function<void()>& fn, int iters) {
    cudaEvent_t start;
    cudaEvent_t stop;
    AssertCuda(cudaEventCreate(&start), "cudaEventCreate start");
    AssertCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        setup();
        AssertCuda(cudaEventRecord(start), "cudaEventRecord start");
        fn();
        AssertCuda(cudaEventRecord(stop), "cudaEventRecord stop");
        AssertCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
        float ms = 0.0f;
        AssertCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        total_ms += ms;
    }

    AssertCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    AssertCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return total_ms / static_cast<float>(iters);
}

double QrFlops(int m, int n) {
    const double md = static_cast<double>(m);
    const double nd = static_cast<double>(n);
    return 2.0 * md * nd * nd - (2.0 / 3.0) * nd * nd * nd;
}

double FlopsToTflops(double flops, float ms) {
    if (ms <= 0.0f) {
        return 0.0;
    }
    return flops / (static_cast<double>(ms) * 1e-3) / 1e12;
}

struct Options {
    int m = 32768;
    int iters = 20;
    int warmup = 5;
    bool use_double = false;
};

Options ParseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            opts.m = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opts.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opts.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            std::string type = argv[++i];
            opts.use_double = (type == "double" || type == "fp64");
        }
    }
    return opts;
}

constexpr int kOldestTsqrBlockSize = 256;
constexpr int kOldestTsqrBlockDimX = 32;
constexpr int kOldestTsqrBlockDimY = 32;
constexpr int kOldestTsqrRowsPerThread = 8;

// Frozen MGS-style baseline derived from the oldest LATER-era implementation.
__device__ __forceinline__ float OldestWarpAllReduceSum(float value) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, mask);
    }
    return value;
}

__global__ void oldest_tsqr_kernel(int m, int n, float* AA, int lda, float* RR, int ldr) {
    int mm = m - blockIdx.x * kOldestTsqrBlockSize;
    mm = (mm < kOldestTsqrBlockSize) ? mm : kOldestTsqrBlockSize;

    const int mnmin = (mm < n) ? mm : n;
    __shared__ float oldest_q[kOldestTsqrBlockSize];
    __shared__ float oldest_r[kTsqrN * kTsqrN];

    float ar[kOldestTsqrRowsPerThread] = {};

#pragma unroll 4
    for (int l = 0; l < kOldestTsqrRowsPerThread; ++l) {
        const int row_idx = threadIdx.x + l * kOldestTsqrBlockDimX;
        if (row_idx < mm && threadIdx.y < mnmin) {
            ar[l] = AA[blockIdx.x * kOldestTsqrBlockSize + row_idx + threadIdx.y * lda];
        }
    }

    __syncthreads();

    for (int k = 0; k < mnmin; ++k) {
        float nu = 0.0f;

        if (threadIdx.y == k) {
#pragma unroll 8
            for (int l = 0; l < kOldestTsqrRowsPerThread; ++l) {
                const int row_idx = threadIdx.x + l * kOldestTsqrBlockDimX;
                if (row_idx < mm) {
                    nu += ar[l] * ar[l];
                }
            }

            const float normx = sqrtf(OldestWarpAllReduceSum(nu));
            if (threadIdx.x == k) {
                oldest_r[k + k * kTsqrN] = normx;
            }
            const float scale = 1.0f / normx;

#pragma unroll 8
            for (int l = 0; l < kOldestTsqrRowsPerThread; ++l) {
                const int row_idx = threadIdx.x + l * kOldestTsqrBlockDimX;
                if (row_idx < mm) {
                    ar[l] *= scale;
                    oldest_q[row_idx] = ar[l];
                }
            }
        }

        __syncthreads();

        if (threadIdx.y > k) {
            float dot = 0.0f;

#pragma unroll 8
            for (int l = 0; l < kOldestTsqrRowsPerThread; ++l) {
                const int row_idx = threadIdx.x + l * kOldestTsqrBlockDimX;
                if (row_idx < mm) {
                    dot += oldest_q[row_idx] * ar[l];
                }
            }

            const float scale = OldestWarpAllReduceSum(dot);

#pragma unroll 8
            for (int l = 0; l < kOldestTsqrRowsPerThread; ++l) {
                const int row_idx = threadIdx.x + l * kOldestTsqrBlockDimX;
                if (row_idx < mm) {
                    ar[l] -= oldest_q[row_idx] * scale;
                }
            }

            if (threadIdx.x == k) {
                oldest_r[k + threadIdx.y * kTsqrN] = scale;
            }
        }

        __syncthreads();
    }

#pragma unroll 8
    for (int l = 0; l < kOldestTsqrRowsPerThread; ++l) {
        const int row_idx = threadIdx.x + l * kOldestTsqrBlockDimX;
        if (row_idx < mm && threadIdx.y < mnmin) {
            AA[blockIdx.x * kOldestTsqrBlockSize + row_idx + threadIdx.y * lda] = ar[l];
        }
    }

    if (threadIdx.x < mnmin && threadIdx.y < mnmin) {
        RR[blockIdx.x * kTsqrN + threadIdx.x + threadIdx.y * ldr] =
            (threadIdx.x <= threadIdx.y) ? oldest_r[threadIdx.x + threadIdx.y * kTsqrN] : 0.0f;
    }
}

bool OldestTsqrSupported(int m, int n) {
    return m >= n && n == kTsqrN;
}

size_t OldestTsqrWorkElems(int m, int n) {
    if (m <= kOldestTsqrBlockSize) {
        return 0;
    }

    size_t total = 0;
    int current = m;
    while (current > kOldestTsqrBlockSize) {
        const int block_num = (current + kOldestTsqrBlockSize - 1) / kOldestTsqrBlockSize;
        const int stack_rows = block_num * n;
        total += static_cast<size_t>(stack_rows) * n;
        current = stack_rows;
    }
    return total;
}

void OldestTsqr(cublasHandle_t cublas_handle,
                int m,
                int n,
                float* A,
                int lda,
                float* R,
                int ldr,
                float* work,
                size_t work_elems) {
    assert(OldestTsqrSupported(m, n));
    assert(m <= kOldestTsqrBlockSize || work != nullptr);

    const dim3 block_dim(kOldestTsqrBlockDimX, kOldestTsqrBlockDimY);
    if (m <= kOldestTsqrBlockSize) {
        oldest_tsqr_kernel<<<1, block_dim>>>(m, n, A, lda, R, ldr);
        return;
    }

    const int block_num = (m + kOldestTsqrBlockSize - 1) / kOldestTsqrBlockSize;
    const int stack_rows = block_num * n;
    const size_t stack_elems = static_cast<size_t>(stack_rows) * n;
    assert(stack_elems <= work_elems);
    const int ldwork = stack_rows;

    oldest_tsqr_kernel<<<block_num, block_dim>>>(m, n, A, lda, work, ldwork);
    OldestTsqr(cublas_handle, stack_rows, n, work, ldwork, R, ldr, work + stack_elems,
               work_elems - stack_elems);

    const float one = 1.0f;
    const float zero = 0.0f;
    const int full_blocks = m / kOldestTsqrBlockSize;
    if (full_blocks > 0) {
        AssertCublas(CublasGemmTraits<float>::GemmStridedBatched(
                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, kOldestTsqrBlockSize, n, n,
                         &one, A, lda, kOldestTsqrBlockSize, work, ldwork, n, &zero, A, lda,
                         kOldestTsqrBlockSize, full_blocks),
                     "oldest tsqr batched gemm");
    }

    const int remaining_rows = m % kOldestTsqrBlockSize;
    if (remaining_rows > 0) {
        AssertCublas(CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                   remaining_rows, n, n, &one,
                                                   A + (m - remaining_rows), lda,
                                                   work + full_blocks * n, ldwork, &zero,
                                                   A + (m - remaining_rows), lda),
                     "oldest tsqr tail gemm");
    }
}

template <typename T>
struct LegacySharedMemory;

template <>
struct LegacySharedMemory<float> {
    __device__ static float* GetPointer() {
        extern __shared__ float shared_mem_float[];
        return shared_mem_float;
    }
};

template <>
struct LegacySharedMemory<double> {
    __device__ static double* GetPointer() {
        extern __shared__ double shared_mem_double[];
        return shared_mem_double;
    }
};

template <typename T>
__device__ __forceinline__ T LegacyWarpAllReduceSum(T value) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, mask);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T LegacyAbs(T value) {
    return value >= static_cast<T>(0) ? value : -value;
}

template <typename T>
__device__ __forceinline__ T LegacySqrt(T value);

template <>
__device__ __forceinline__ float LegacySqrt<float>(float value) {
    return sqrtf(value);
}

template <>
__device__ __forceinline__ double LegacySqrt<double>(double value) {
    return sqrt(value);
}

template <typename T>
__device__ __forceinline__ T LegacyEpsilon();

template <>
__device__ __forceinline__ float LegacyEpsilon<float>() {
    return 1.0e-7f;
}

template <>
__device__ __forceinline__ double LegacyEpsilon<double>() {
    return 1.0e-12;
}

template <typename T>
__global__ void legacy_tsqr_kernel(int m, int n, T* A, int lda, T* R, int ldr) {
    T* shared_A = LegacySharedMemory<T>::GetPointer();
    constexpr int ldsa = kLegacyTsqrBlockSize;

    const int thread_idx_x = threadIdx.x;
    const int thread_idx_y = threadIdx.y;
    const int block_idx_x = blockIdx.x;
    const int block_size = min(kLegacyTsqrBlockSize, m - block_idx_x * kLegacyTsqrBlockSize);

    A += block_idx_x * kLegacyTsqrBlockSize;
    R += block_idx_x * n;

    const int num_data_col = (n + kLegacyTsqrBlockDimY - 1) / kLegacyTsqrBlockDimY;

    T acc[kLegacyTsqrRowsPerThread];
#pragma unroll
    for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
        const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
        if (row_idx < block_size) {
            for (int h = 0; h < num_data_col; ++h) {
                const int col_idx = thread_idx_y + h * kLegacyTsqrBlockDimY;
                if (col_idx < n) {
                    shared_A[row_idx + col_idx * ldsa] = A[row_idx + col_idx * lda];
                }
            }
        }
    }

    __syncthreads();

    T q[kLegacyTsqrRowsPerThread];
    for (int cols = 0; cols < n; ++cols) {
        T nu = static_cast<T>(0);
        if (thread_idx_y == cols % kLegacyTsqrBlockDimY) {
#pragma unroll
            for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
                acc[k] = static_cast<T>(0);
                const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
                if (row_idx >= cols && row_idx < block_size) {
                    q[k] = shared_A[row_idx + cols * ldsa];
                    acc[k] = q[k] * q[k];
                }
                nu += acc[k];
            }

            const T norm_x_square = LegacyWarpAllReduceSum(nu);
            const T norm_x = LegacySqrt(norm_x_square);
            const T epsilon = LegacyEpsilon<T>();

            if (norm_x > epsilon) {
                T scale = static_cast<T>(1) / norm_x;
#pragma unroll
                for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
                    const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
                    if (row_idx >= cols && row_idx < block_size) {
                        q[k] *= scale;
                    }
                }

                const int thread_idx = cols % kLegacyTsqrBlockDimX;
                const int thread_off = cols / kLegacyTsqrBlockDimX;
                T u1 = static_cast<T>(0);
                if (thread_idx_x == thread_idx) {
                    q[thread_off] += (q[thread_off] >= static_cast<T>(0)) ? static_cast<T>(1)
                                                                          : static_cast<T>(-1);
                    u1 = q[thread_off];
                    R[cols + cols * ldr] = (u1 >= static_cast<T>(0)) ? -norm_x : norm_x;
                }
                u1 = __shfl_sync(0xffffffff, u1, thread_idx);

                scale = static_cast<T>(1) / LegacySqrt(LegacyAbs(u1));
#pragma unroll
                for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
                    const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
                    if (row_idx >= cols && row_idx < block_size) {
                        shared_A[row_idx + cols * ldsa] = q[k] * scale;
                    }
                }
            } else {
                const int thread_idx = cols % kLegacyTsqrBlockDimX;
                if (thread_idx_x == thread_idx) {
                    R[cols + cols * ldr] = static_cast<T>(0);
                }
#pragma unroll
                for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
                    const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
                    if (row_idx >= cols && row_idx < block_size) {
                        shared_A[row_idx + cols * ldsa] = static_cast<T>(0);
                    }
                }
            }
        }

        __syncthreads();

        for (int h = 0; h < num_data_col; ++h) {
            const int op_cols = thread_idx_y + h * kLegacyTsqrBlockDimY;
            if (cols < op_cols && op_cols < n) {
                nu = static_cast<T>(0);
#pragma unroll
                for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
                    acc[k] = static_cast<T>(0);
                    const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
                    if (row_idx >= cols && row_idx < block_size) {
                        q[k] = shared_A[row_idx + cols * ldsa];
                        acc[k] = q[k] * shared_A[row_idx + op_cols * ldsa];
                    }
                    nu += acc[k];
                }
                const T utx = LegacyWarpAllReduceSum(nu);

#pragma unroll
                for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
                    const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
                    if (row_idx >= cols && row_idx < block_size) {
                        shared_A[row_idx + op_cols * ldsa] -= utx * q[k];
                    }
                }
            }
        }
    }

    __syncthreads();

    const int r_row_data_num = (n + kLegacyTsqrBlockDimX - 1) / kLegacyTsqrBlockDimX;
    for (int h = 0; h < num_data_col; ++h) {
        const int op_cols = thread_idx_y + h * kLegacyTsqrBlockDimY;
        if (op_cols >= n) {
            continue;
        }

#pragma unroll
        for (int k = 0; k < r_row_data_num; ++k) {
            const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
            if (row_idx < op_cols && row_idx < n) {
                R[row_idx + op_cols * ldr] = shared_A[row_idx + op_cols * ldsa];
                shared_A[row_idx + op_cols * ldsa] = static_cast<T>(0);
            }
            if (row_idx > op_cols && row_idx < n) {
                R[row_idx + op_cols * ldr] = static_cast<T>(0);
            }
        }
    }

    for (int h = 0; h < num_data_col; ++h) {
        const int op_cols = thread_idx_y + h * kLegacyTsqrBlockDimY;
        if (op_cols >= n) {
            continue;
        }

#pragma unroll
        for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
            const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
            q[k] = (row_idx == op_cols) ? static_cast<T>(1) : static_cast<T>(0);
        }
        __syncwarp();

        for (int cols = n - 1; cols >= 0; --cols) {
            if (op_cols >= cols) {
                T dot = static_cast<T>(0);
#pragma unroll
                for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
                    acc[k] = static_cast<T>(0);
                    const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
                    if (row_idx < block_size) {
                        acc[k] = shared_A[row_idx + cols * ldsa] * q[k];
                        dot += acc[k];
                    }
                }
                const T utq = LegacyWarpAllReduceSum(dot);

#pragma unroll
                for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
                    const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
                    if (row_idx < block_size) {
                        q[k] -= utq * shared_A[row_idx + cols * ldsa];
                    }
                }
                __syncwarp();
            }
        }

#pragma unroll
        for (int k = 0; k < kLegacyTsqrRowsPerThread; ++k) {
            const int row_idx = thread_idx_x + k * kLegacyTsqrBlockDimX;
            if (row_idx < block_size) {
                A[row_idx + op_cols * lda] = q[k];
            }
        }
    }
}

bool LegacyTsqrSupported(int m, int n) {
    return m >= n && n <= kTsqrN && m % n == 0 && ((m % kLegacyTsqrBlockSize) % n) == 0;
}

int LegacyTsqrLdwork(int m, int n) {
    if (m <= kLegacyTsqrBlockSize) {
        return 0;
    }
    return ((m + kLegacyTsqrBlockSize - 1) / kLegacyTsqrBlockSize) * n;
}

size_t LegacyTsqrWorkElems(int m, int n) {
    if (m <= kLegacyTsqrBlockSize) {
        return 0;
    }

    const int ldwork = LegacyTsqrLdwork(m, n);
    const size_t slab_elems = static_cast<size_t>(ldwork) * n;
    size_t total = 0;
    int current = m;
    while (current > kLegacyTsqrBlockSize) {
        total += slab_elems;
        current = ((current + kLegacyTsqrBlockSize - 1) / kLegacyTsqrBlockSize) * n;
    }
    return total;
}

template <typename T>
void LegacyTsqrFunc(cublasHandle_t cublas_handle,
                    int share_memory_size,
                    int m,
                    int n,
                    T* A,
                    int lda,
                    T* R,
                    int ldr,
                    T* work,
                    int ldwork) {
    const dim3 block_dim(kLegacyTsqrBlockDimX, kLegacyTsqrBlockDimY);

    if (m <= kLegacyTsqrBlockSize) {
        legacy_tsqr_kernel<T><<<1, block_dim, share_memory_size>>>(m, n, A, lda, R, ldr);
        return;
    }

    const int block_num = (m + kLegacyTsqrBlockSize - 1) / kLegacyTsqrBlockSize;
    legacy_tsqr_kernel<T><<<block_num, block_dim, share_memory_size>>>(m, n, A, lda, work, ldwork);

    LegacyTsqrFunc(cublas_handle, share_memory_size, block_num * n, n, work, ldwork, R, ldr,
                   work + static_cast<size_t>(ldwork) * n, ldwork);

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const int full_blocks = m / kLegacyTsqrBlockSize;
    if (full_blocks > 0) {
        AssertCublas(CublasGemmTraits<T>::GemmStridedBatched(
                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, kLegacyTsqrBlockSize, n, n, &one,
                         A, lda, kLegacyTsqrBlockSize, work, ldwork, n, &zero, A, lda,
                         kLegacyTsqrBlockSize, full_blocks),
                     "legacy tsqr batched gemm");
    }

    const int remaining_rows = m % kLegacyTsqrBlockSize;
    if (remaining_rows > 0) {
        AssertCublas(CublasGemmTraits<T>::Gemm(
                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, remaining_rows, n, n, &one,
                         A + (m - remaining_rows), lda, work + full_blocks * n, ldwork, &zero,
                         A + (m - remaining_rows), lda),
                     "legacy tsqr tail gemm");
    }
}

template <typename T>
void LegacyTsqr(cublasHandle_t cublas_handle,
                int m,
                int n,
                T* A,
                int lda,
                T* R,
                int ldr,
                T* work,
                size_t work_elems) {
    assert(LegacyTsqrSupported(m, n));
    assert(m <= kLegacyTsqrBlockSize || work != nullptr);

    const int ldwork = LegacyTsqrLdwork(m, n);
    const size_t required_work_elems = LegacyTsqrWorkElems(m, n);
    assert(required_work_elems <= work_elems);

    const int share_memory_size = kLegacyTsqrBlockSize * n * sizeof(T);
    AssertCuda(cudaFuncSetAttribute(legacy_tsqr_kernel<T>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    share_memory_size),
               "cudaFuncSetAttribute legacy_tsqr_kernel");

    LegacyTsqrFunc(cublas_handle, share_memory_size, m, n, A, lda, R, ldr, work, ldwork);
}

template <typename T>
struct CusolverGeqrfTraits;

template <>
struct CusolverGeqrfTraits<float> {
    static cusolverStatus_t BufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda,
                                       int* lwork) {
        return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork);
    }

    static cusolverStatus_t Geqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  float* A,
                                  int lda,
                                  float* tau,
                                  float* work,
                                  int lwork,
                                  int* info) {
        return cusolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    }
};

template <>
struct CusolverGeqrfTraits<double> {
    static cusolverStatus_t BufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda,
                                       int* lwork) {
        return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork);
    }

    static cusolverStatus_t Geqrf(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  double* A,
                                  int lda,
                                  double* tau,
                                  double* work,
                                  int lwork,
                                  int* info) {
        return cusolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    }
};

template <typename T>
void RunBench(const Options& opts, cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle) {
    const int m = opts.m;
    const int n = kTsqrN;
    const int lda = m;
    const int ldr = n;
    const double qr_flops = QrFlops(m, n);
    const size_t a_elems = static_cast<size_t>(lda) * n;
    const size_t a_bytes = a_elems * sizeof(T);
    const size_t r_bytes = static_cast<size_t>(ldr) * n * sizeof(T);
    const size_t tsqr_work_elems_count = tsqr_work_elems<T>(m);
    const bool legacy_supported = LegacyTsqrSupported(m, n);
    const size_t legacy_work_elems_count = legacy_supported ? LegacyTsqrWorkElems(m, n) : 0;
    const bool oldest_supported = std::is_same_v<T, float> && OldestTsqrSupported(m, n);
    const size_t oldest_work_elems_count = oldest_supported ? OldestTsqrWorkElems(m, n) : 0;

    T* d_A0 = nullptr;
    T* d_A_work = nullptr;
    T* d_R = nullptr;
    T* d_work_tsqr = nullptr;
    T* d_work_legacy = nullptr;
    float* d_work_oldest = nullptr;
    T* d_tau = nullptr;
    T* d_work_geqrf = nullptr;
    int* d_info = nullptr;

    AssertCuda(cudaMalloc(&d_A0, a_bytes), "cudaMalloc d_A0");
    AssertCuda(cudaMalloc(&d_A_work, a_bytes), "cudaMalloc d_A_work");
    AssertCuda(cudaMalloc(&d_R, r_bytes), "cudaMalloc d_R");
    AssertCuda(cudaMalloc(&d_tau, n * sizeof(T)), "cudaMalloc d_tau");
    AssertCuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc d_info");

    FillDeviceRandom(d_A0, a_elems, 1234ULL);

    if (tsqr_work_elems_count > 0) {
        AssertCuda(cudaMalloc(&d_work_tsqr, tsqr_work_elems_count * sizeof(T)),
                   "cudaMalloc d_work_tsqr");
    }
    if (legacy_work_elems_count > 0) {
        AssertCuda(cudaMalloc(&d_work_legacy, legacy_work_elems_count * sizeof(T)),
                   "cudaMalloc d_work_legacy");
    }
    if (oldest_work_elems_count > 0) {
        AssertCuda(cudaMalloc(&d_work_oldest, oldest_work_elems_count * sizeof(float)),
                   "cudaMalloc d_work_oldest");
    }

    int lwork = 0;
    AssertCusolver(CusolverGeqrfTraits<T>::BufferSize(cusolver_handle, m, n, d_A_work, lda, &lwork),
                   "cusolverDnXgeqrf_bufferSize");
    if (lwork > 0) {
        AssertCuda(cudaMalloc(&d_work_geqrf, lwork * sizeof(T)), "cudaMalloc d_work_geqrf");
    }

    T* d_B = nullptr;
    T* d_C = nullptr;
    AssertCuda(cudaMalloc(&d_B, static_cast<size_t>(n) * n * sizeof(T)), "cudaMalloc d_B");
    AssertCuda(cudaMalloc(&d_C, static_cast<size_t>(m) * n * sizeof(T)), "cudaMalloc d_C");
    FillDeviceRandom(d_B, static_cast<size_t>(n) * n, 5678ULL);
    WarmupGemm(cublas_handle, m, n, n, d_A0, lda, d_B, n, d_C, m, opts.warmup);
    AssertCuda(cudaFree(d_B), "cudaFree d_B");
    AssertCuda(cudaFree(d_C), "cudaFree d_C");
    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    const float current_tsqr_ms = TimeKernelMs(
        [&]() {
            AssertCuda(cudaMemcpy(d_A_work, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy D2D current tsqr");
        },
        [&]() {
            tsqr(cublas_handle, m, d_A_work, lda, d_R, ldr, d_work_tsqr, tsqr_work_elems_count,
                 nullptr);
            AssertCuda(cudaGetLastError(), "current tsqr launch");
        },
        opts.iters);

    float legacy_tsqr_ms = 0.0f;
    if (legacy_supported) {
        legacy_tsqr_ms = TimeKernelMs(
            [&]() {
                AssertCuda(cudaMemcpy(d_A_work, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpy D2D legacy tsqr");
            },
            [&]() {
                LegacyTsqr(cublas_handle, m, n, d_A_work, lda, d_R, ldr, d_work_legacy,
                           legacy_work_elems_count);
                AssertCuda(cudaGetLastError(), "legacy tsqr launch");
            },
            opts.iters);
    }

    float oldest_tsqr_ms = 0.0f;
    if constexpr (std::is_same_v<T, float>) {
        if (oldest_supported) {
            oldest_tsqr_ms = TimeKernelMs(
                [&]() {
                    AssertCuda(cudaMemcpy(d_A_work, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                               "cudaMemcpy D2D oldest tsqr");
                },
                [&]() {
                    OldestTsqr(cublas_handle, m, n, d_A_work, lda, d_R, ldr, d_work_oldest,
                               oldest_work_elems_count);
                    AssertCuda(cudaGetLastError(), "oldest tsqr launch");
                },
                opts.iters);
        }
    }

    const float geqrf_ms = TimeKernelMs(
        [&]() {
            AssertCuda(cudaMemcpy(d_A_work, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy D2D geqrf");
        },
        [&]() {
            AssertCusolver(CusolverGeqrfTraits<T>::Geqrf(cusolver_handle, m, n, d_A_work, lda,
                                                         d_tau, d_work_geqrf, lwork, d_info),
                           "cusolverDnXgeqrf");
        },
        opts.iters);

    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize bench");

    std::printf("Current TSQR avg: %.3f ms (%.3f TFLOPS)\n", current_tsqr_ms,
                FlopsToTflops(qr_flops, current_tsqr_ms));
    if (legacy_supported) {
        std::printf("Legacy  TSQR avg: %.3f ms (%.3f TFLOPS)\n", legacy_tsqr_ms,
                    FlopsToTflops(qr_flops, legacy_tsqr_ms));
        std::printf("Current/Legacy speedup: %.3fx\n", legacy_tsqr_ms / current_tsqr_ms);
    } else {
        std::printf("Legacy  TSQR avg: skipped (requires m %% n == 0 and tail block aligned to n)\n");
    }
    if (oldest_supported) {
        std::printf("Oldest  TSQR avg: %.3f ms (%.3f TFLOPS)\n", oldest_tsqr_ms,
                    FlopsToTflops(qr_flops, oldest_tsqr_ms));
        std::printf("Current/Oldest speedup: %.3fx\n", oldest_tsqr_ms / current_tsqr_ms);
    } else {
        std::printf("Oldest  TSQR avg: skipped (float-only frozen MGS baseline)\n");
    }
    std::printf("GEQRF avg:        %.3f ms (%.3f TFLOPS)\n", geqrf_ms, FlopsToTflops(qr_flops, geqrf_ms));

    AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    AssertCuda(cudaFree(d_A_work), "cudaFree d_A_work");
    AssertCuda(cudaFree(d_R), "cudaFree d_R");
    AssertCuda(cudaFree(d_work_tsqr), "cudaFree d_work_tsqr");
    AssertCuda(cudaFree(d_work_legacy), "cudaFree d_work_legacy");
    AssertCuda(cudaFree(d_work_oldest), "cudaFree d_work_oldest");
    AssertCuda(cudaFree(d_tau), "cudaFree d_tau");
    AssertCuda(cudaFree(d_work_geqrf), "cudaFree d_work_geqrf");
    AssertCuda(cudaFree(d_info), "cudaFree d_info");
}

}  // namespace

int main(int argc, char** argv) {
    const Options opts = ParseArgs(argc, argv);
    std::printf("TSQR bench: m=%d n=%d iters=%d warmup=%d type=%s\n", opts.m, kTsqrN, opts.iters,
                opts.warmup, opts.use_double ? "double" : "float");

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    AssertCublas(cublasCreate(&cublas_handle), "cublasCreate");
    AssertCusolver(cusolverDnCreate(&cusolver_handle), "cusolverDnCreate");

    if (opts.use_double) {
        RunBench<double>(opts, cublas_handle, cusolver_handle);
    } else {
        RunBench<float>(opts, cublas_handle, cusolver_handle);
    }

    AssertCusolver(cusolverDnDestroy(cusolver_handle), "cusolverDnDestroy");
    AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    return 0;
}
