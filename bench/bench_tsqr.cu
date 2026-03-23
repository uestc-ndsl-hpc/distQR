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
    int hybrid_switch_m = 4096;
    enum class DoubleVariant {
        kCurrent,
        kStatic192,
        kSplit256,
        kHybrid192,
        kBoth,
        kAll,
    };
    DoubleVariant double_variant = DoubleVariant::kCurrent;
};

const char* DoubleVariantName(Options::DoubleVariant variant) {
    switch (variant) {
        case Options::DoubleVariant::kCurrent:
            return "current";
        case Options::DoubleVariant::kStatic192:
            return "static192";
        case Options::DoubleVariant::kSplit256:
            return "split256";
        case Options::DoubleVariant::kHybrid192:
            return "hybrid192";
        case Options::DoubleVariant::kBoth:
            return "both";
        case Options::DoubleVariant::kAll:
            return "all";
    }
    return "unknown";
}

bool ShouldRunCurrentTsqr(const Options& opts) {
    return !opts.use_double || opts.double_variant == Options::DoubleVariant::kCurrent ||
           opts.double_variant == Options::DoubleVariant::kBoth ||
           opts.double_variant == Options::DoubleVariant::kAll;
}

bool ShouldRunStatic192Tsqr(const Options& opts) {
    return opts.use_double &&
           (opts.double_variant == Options::DoubleVariant::kStatic192 ||
            opts.double_variant == Options::DoubleVariant::kBoth ||
            opts.double_variant == Options::DoubleVariant::kAll);
}

bool ShouldRunSplit256Tsqr(const Options& opts) {
    return opts.use_double &&
           (opts.double_variant == Options::DoubleVariant::kSplit256 ||
            opts.double_variant == Options::DoubleVariant::kAll);
}

bool ShouldRunHybrid192Tsqr(const Options& opts) {
    return opts.use_double &&
           (opts.double_variant == Options::DoubleVariant::kHybrid192 ||
            opts.double_variant == Options::DoubleVariant::kAll);
}

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
        } else if (std::strcmp(argv[i], "--double-variant") == 0 && i + 1 < argc) {
            const std::string variant = argv[++i];
            if (variant == "current") {
                opts.double_variant = Options::DoubleVariant::kCurrent;
            } else if (variant == "static192") {
                opts.double_variant = Options::DoubleVariant::kStatic192;
            } else if (variant == "split256") {
                opts.double_variant = Options::DoubleVariant::kSplit256;
            } else if (variant == "hybrid192") {
                opts.double_variant = Options::DoubleVariant::kHybrid192;
            } else if (variant == "both") {
                opts.double_variant = Options::DoubleVariant::kBoth;
            } else if (variant == "all") {
                opts.double_variant = Options::DoubleVariant::kAll;
            }
        } else if (std::strcmp(argv[i], "--hybrid-switch-m") == 0 && i + 1 < argc) {
            opts.hybrid_switch_m = std::atoi(argv[++i]);
        }
    }
    return opts;
}

template <typename T>
__device__ __forceinline__ T PanelTsqrEpsilon();

template <typename T>
__device__ __forceinline__ T LegacyWarpAllReduceSum(T value);

template <typename T>
__device__ __forceinline__ T PanelTsqrSqrt(T value);

template <typename T>
__device__ __forceinline__ T PanelTsqrInvSqrtAbs(T value);

template <>
__device__ __forceinline__ float PanelTsqrEpsilon<float>() {
    return 1.0e-4f;
}

template <>
__device__ __forceinline__ double PanelTsqrEpsilon<double>() {
    return 1.0e-7;
}

template <>
__device__ __forceinline__ float PanelTsqrSqrt<float>(float value) {
    return __fsqrt_rn(value);
}

template <>
__device__ __forceinline__ double PanelTsqrSqrt<double>(double value) {
    return __dsqrt_rn(value);
}

template <>
__device__ __forceinline__ float PanelTsqrInvSqrtAbs<float>(float value) {
    return __frsqrt_rn(fabsf(value));
}

template <>
__device__ __forceinline__ double PanelTsqrInvSqrtAbs<double>(double value) {
    return __drcp_rn(__dsqrt_rn(fabs(value)));
}

template <typename T, int BlockSize>
__global__ void static_tsqr_n32_kernel(int m, T* A, int lda, T* R, int ldr) {
    constexpr int kN = kTsqrN;
    constexpr int kWarpSize = 32;
    constexpr int kRowsPerThread = (BlockSize + kWarpSize - 1) / kWarpSize;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int bx = blockIdx.x;

    A += bx * BlockSize;
    R += bx * kN;

    const int block_size = min(BlockSize, m - bx * BlockSize);

    __shared__ T shared_A[BlockSize * kN];
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        const int row_idx = lane_id + i * kWarpSize;
        if (row_idx < block_size) {
            shared_A[row_idx + warp_id * BlockSize] = A[row_idx + warp_id * lda];
        }
    }
    __syncthreads();

    T q[kRowsPerThread];
    for (int col = 0; col < kN; ++col) {
        T nu = static_cast<T>(0);
        if (warp_id == col) {
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx >= col && row_idx < block_size) {
                    q[i] = shared_A[row_idx + col * BlockSize];
                    nu += q[i] * q[i];
                }
            }
            const T norm_square = LegacyWarpAllReduceSum(nu);
            const T epsilon = PanelTsqrEpsilon<T>();
            if (norm_square > epsilon * epsilon) {
                T norm = PanelTsqrSqrt(norm_square);
                T scale = static_cast<T>(1) / norm;
#pragma unroll
                for (int i = 0; i < kRowsPerThread; ++i) {
                    const int row_idx = lane_id + i * kWarpSize;
                    if (row_idx >= col && row_idx < block_size) {
                        q[i] *= scale;
                    }
                }

                T u1 = static_cast<T>(0);
                const int thread_off = col / kWarpSize;
                if (lane_id == col % kWarpSize) {
                    q[thread_off] +=
                        (q[thread_off] >= static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(-1);
                    u1 = q[thread_off];
                    R[col + ldr * col] = (u1 >= static_cast<T>(0)) ? -norm : norm;
                }
                u1 = __shfl_sync(0xffffffff, u1, col % kWarpSize);
                scale = PanelTsqrInvSqrtAbs(u1);
#pragma unroll
                for (int i = 0; i < kRowsPerThread; ++i) {
                    const int row_idx = lane_id + i * kWarpSize;
                    if (row_idx >= col && row_idx < block_size) {
                        shared_A[row_idx + col * BlockSize] = q[i] * scale;
                    }
                }
            } else {
                if (lane_id == col % kWarpSize) {
                    R[col + ldr * col] = static_cast<T>(0);
                }
#pragma unroll
                for (int i = 0; i < kRowsPerThread; ++i) {
                    const int row_idx = lane_id + i * kWarpSize;
                    if (row_idx >= col && row_idx < block_size) {
                        shared_A[row_idx + col * BlockSize] = static_cast<T>(0);
                    }
                }
            }
        }
        __syncthreads();

        if (col < warp_id) {
            nu = static_cast<T>(0);
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx >= col && row_idx < block_size) {
                    q[i] = shared_A[row_idx + col * BlockSize];
                    nu += q[i] * shared_A[row_idx + warp_id * BlockSize];
                }
            }
            const T utx = LegacyWarpAllReduceSum(nu);
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx >= col && row_idx < block_size) {
                    shared_A[row_idx + warp_id * BlockSize] -= utx * q[i];
                }
            }
        }
    }

    __syncthreads();

    if (lane_id < warp_id) {
        R[lane_id + ldr * warp_id] = shared_A[lane_id + warp_id * BlockSize];
        shared_A[lane_id + warp_id * BlockSize] = static_cast<T>(0);
    }
    if (lane_id > warp_id) {
        R[lane_id + ldr * warp_id] = static_cast<T>(0);
    }

#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        const int row_idx = lane_id + i * kWarpSize;
        q[i] = (row_idx == warp_id) ? static_cast<T>(1) : static_cast<T>(0);
    }
    __syncwarp();
    for (int col = kN - 1; col >= 0; --col) {
        if (warp_id >= col) {
            T utq = static_cast<T>(0);
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx < block_size) {
                    utq += q[i] * shared_A[row_idx + col * BlockSize];
                }
            }
            utq = LegacyWarpAllReduceSum(utq);
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx < block_size) {
                    q[i] -= utq * shared_A[row_idx + col * BlockSize];
                }
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        const int row_idx = lane_id + i * kWarpSize;
        if (row_idx < block_size) {
            A[row_idx + warp_id * lda] = q[i];
        }
    }
}

template <int BlockSize, int StaticRows>
__device__ __forceinline__ double SplitSharedRead(const double* shared_head,
                                                  const double* shared_tail,
                                                  int row,
                                                  int col) {
    constexpr int kTailRows = BlockSize - StaticRows;
    static_assert(StaticRows > 0 && kTailRows > 0, "Split shared layout must have both regions.");
    if (row < StaticRows) {
        return shared_head[row + col * StaticRows];
    }
    return shared_tail[(row - StaticRows) + col * kTailRows];
}

template <int BlockSize, int StaticRows>
__device__ __forceinline__ void SplitSharedWrite(double* shared_head,
                                                 double* shared_tail,
                                                 int row,
                                                 int col,
                                                 double value) {
    constexpr int kTailRows = BlockSize - StaticRows;
    static_assert(StaticRows > 0 && kTailRows > 0, "Split shared layout must have both regions.");
    if (row < StaticRows) {
        shared_head[row + col * StaticRows] = value;
        return;
    }
    shared_tail[(row - StaticRows) + col * kTailRows] = value;
}

template <int BlockSize, int StaticRows>
__global__ void split_tsqr_n32_double_kernel(int m, double* A, int lda, double* R, int ldr) {
    constexpr int kN = kTsqrN;
    constexpr int kWarpSize = 32;
    constexpr int kRowsPerThread = (BlockSize + kWarpSize - 1) / kWarpSize;
    constexpr int kTailRows = BlockSize - StaticRows;
    constexpr double kEpsilon = 1.0e-7;
    static_assert(StaticRows > 0 && kTailRows > 0, "Split kernel needs static and dynamic rows.");

    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int bx = blockIdx.x;

    A += bx * BlockSize;
    R += bx * kN;

    const int block_size = min(BlockSize, m - bx * BlockSize);

    __shared__ double shared_head[StaticRows * kN];
    extern __shared__ double shared_tail[];

#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        const int row_idx = lane_id + i * kWarpSize;
        if (row_idx < block_size) {
            SplitSharedWrite<BlockSize, StaticRows>(shared_head, shared_tail, row_idx, warp_id,
                                                    A[row_idx + warp_id * lda]);
        }
    }
    __syncthreads();

    double q[kRowsPerThread];
    for (int col = 0; col < kN; ++col) {
        double nu = 0.0;
        if (warp_id == col) {
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx >= col && row_idx < block_size) {
                    q[i] = SplitSharedRead<BlockSize, StaticRows>(shared_head, shared_tail, row_idx,
                                                                  col);
                    nu += q[i] * q[i];
                }
            }
            const double norm_square = LegacyWarpAllReduceSum(nu);
            if (norm_square > kEpsilon * kEpsilon) {
                const double norm = __dsqrt_rn(norm_square);
                double scale = 1.0 / norm;
#pragma unroll
                for (int i = 0; i < kRowsPerThread; ++i) {
                    const int row_idx = lane_id + i * kWarpSize;
                    if (row_idx >= col && row_idx < block_size) {
                        q[i] *= scale;
                    }
                }
                double u1 = 0.0;
                const int thread_off = col / kWarpSize;
                if (lane_id == col % kWarpSize) {
                    q[thread_off] += (q[thread_off] >= 0.0) ? 1.0 : -1.0;
                    u1 = q[thread_off];
                    R[col + ldr * col] = (u1 >= 0.0) ? -norm : norm;
                }
                u1 = __shfl_sync(0xffffffff, u1, col % kWarpSize);
                scale = __drcp_rn(__dsqrt_rn(fabs(u1)));
#pragma unroll
                for (int i = 0; i < kRowsPerThread; ++i) {
                    const int row_idx = lane_id + i * kWarpSize;
                    if (row_idx >= col && row_idx < block_size) {
                        SplitSharedWrite<BlockSize, StaticRows>(shared_head, shared_tail, row_idx,
                                                                col, q[i] * scale);
                    }
                }
            } else {
                if (lane_id == col % kWarpSize) {
                    R[col + ldr * col] = 0.0;
                }
#pragma unroll
                for (int i = 0; i < kRowsPerThread; ++i) {
                    const int row_idx = lane_id + i * kWarpSize;
                    if (row_idx >= col && row_idx < block_size) {
                        SplitSharedWrite<BlockSize, StaticRows>(shared_head, shared_tail, row_idx,
                                                                col, 0.0);
                    }
                }
            }
        }
        __syncthreads();

        if (col < warp_id) {
            nu = 0.0;
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx >= col && row_idx < block_size) {
                    q[i] = SplitSharedRead<BlockSize, StaticRows>(shared_head, shared_tail, row_idx,
                                                                  col);
                    nu += q[i] * SplitSharedRead<BlockSize, StaticRows>(shared_head, shared_tail,
                                                                        row_idx, warp_id);
                }
            }
            const double utx = LegacyWarpAllReduceSum(nu);
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx >= col && row_idx < block_size) {
                    const double updated =
                        SplitSharedRead<BlockSize, StaticRows>(shared_head, shared_tail, row_idx,
                                                               warp_id) -
                        utx * q[i];
                    SplitSharedWrite<BlockSize, StaticRows>(shared_head, shared_tail, row_idx,
                                                            warp_id, updated);
                }
            }
        }
    }

    __syncthreads();

    if (lane_id < warp_id) {
        R[lane_id + ldr * warp_id] =
            SplitSharedRead<BlockSize, StaticRows>(shared_head, shared_tail, lane_id, warp_id);
        SplitSharedWrite<BlockSize, StaticRows>(shared_head, shared_tail, lane_id, warp_id, 0.0);
    }
    if (lane_id > warp_id) {
        R[lane_id + ldr * warp_id] = 0.0;
    }

#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        const int row_idx = lane_id + i * kWarpSize;
        q[i] = (row_idx == warp_id) ? 1.0 : 0.0;
    }
    __syncwarp();
    for (int col = kN - 1; col >= 0; --col) {
        if (warp_id >= col) {
            double utq = 0.0;
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx < block_size) {
                    utq += q[i] * SplitSharedRead<BlockSize, StaticRows>(shared_head, shared_tail,
                                                                         row_idx, col);
                }
            }
            utq = LegacyWarpAllReduceSum(utq);
#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const int row_idx = lane_id + i * kWarpSize;
                if (row_idx < block_size) {
                    q[i] -= utq * SplitSharedRead<BlockSize, StaticRows>(shared_head, shared_tail,
                                                                         row_idx, col);
                }
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        const int row_idx = lane_id + i * kWarpSize;
        if (row_idx < block_size) {
            A[row_idx + warp_id * lda] = q[i];
        }
    }
}

template <int BlockSize>
size_t StaticTsqrWorkElems(int m) {
    size_t total = 0;
    int current = m;
    while (current > BlockSize) {
        const int block_num = (current + BlockSize - 1) / BlockSize;
        const int stack_rows = block_num * kTsqrN;
        total += static_cast<size_t>(stack_rows) * kTsqrN;
        current = stack_rows;
    }
    return total;
}

template <typename T, int StaticBlockSize>
size_t HybridTsqrWorkElems(int m, int switch_m) {
    const int effective_switch_m = (switch_m < kTsqrN) ? kTsqrN : switch_m;
    if (m <= 0) {
        return 0;
    }
    if (m <= effective_switch_m) {
        return tsqr_work_elems<T>(m);
    }

    size_t total = 0;
    int current = m;
    while (current > effective_switch_m) {
        const int block_num = (current + StaticBlockSize - 1) / StaticBlockSize;
        const int stack_rows = block_num * kTsqrN;
        total += static_cast<size_t>(stack_rows) * kTsqrN;
        current = stack_rows;
    }
    total += tsqr_work_elems<T>(current);
    return total;
}

template <int BlockSize, int StaticRows>
void ConfigureSplitTsqrDoubleLaunch() {
    constexpr int kTailRows = BlockSize - StaticRows;
    constexpr int kTailSharedBytes = kTailRows * kTsqrN * static_cast<int>(sizeof(double));
    constexpr int kPreferSharedMemory = 100;
    AssertCuda(cudaFuncSetAttribute(split_tsqr_n32_double_kernel<BlockSize, StaticRows>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kTailSharedBytes),
               "cudaFuncSetAttribute split_tsqr_n32_double_kernel max dynamic shared");
    AssertCuda(cudaFuncSetAttribute(split_tsqr_n32_double_kernel<BlockSize, StaticRows>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout,
                                    kPreferSharedMemory),
               "cudaFuncSetAttribute split_tsqr_n32_double_kernel carveout");
}

template <typename T, int BlockSize>
void StaticTsqr(cublasHandle_t cublas_handle,
                int m,
                T* A,
                int lda,
                T* R,
                int ldr,
                T* work,
                size_t work_elems) {
    constexpr int kN = kTsqrN;
    const dim3 block(32, 32);

    if (m <= 0) {
        return;
    }

    if (m <= BlockSize) {
        static_tsqr_n32_kernel<T, BlockSize><<<1, block>>>(m, A, lda, R, ldr);
        return;
    }

    const int block_num = (m + BlockSize - 1) / BlockSize;
    const int stack_rows = block_num * kN;
    const size_t stack_elems = static_cast<size_t>(stack_rows) * kN;
    assert(stack_elems <= work_elems);
    const int ldwork = stack_rows;

    static_tsqr_n32_kernel<T, BlockSize><<<block_num, block>>>(m, A, lda, work, ldwork);

    StaticTsqr<T, BlockSize>(cublas_handle, stack_rows, work, ldwork, R, ldr, work + stack_elems,
                             work_elems - stack_elems);

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const int full_blocks = m / BlockSize;
    if (full_blocks > 0) {
        AssertCublas(CublasGemmTraits<T>::GemmStridedBatched(
                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, BlockSize, kN, kN, &one, A, lda,
                         BlockSize, work, ldwork, kN, &zero, A, lda, BlockSize, full_blocks),
                     "static tsqr batched gemm");
    }

    const int remaining_rows = m % BlockSize;
    if (remaining_rows > 0) {
        AssertCublas(CublasGemmTraits<T>::Gemm(
                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, remaining_rows, kN, kN, &one,
                         A + (m - remaining_rows), lda, work + full_blocks * kN, ldwork, &zero,
                         A + (m - remaining_rows), lda),
                     "static tsqr tail gemm");
    }
}

template <int BlockSize, int StaticRows>
void SplitTsqrDouble(cublasHandle_t cublas_handle,
                     int m,
                     double* A,
                     int lda,
                     double* R,
                     int ldr,
                     double* work,
                     size_t work_elems) {
    constexpr int kN = kTsqrN;
    constexpr int kTailRows = BlockSize - StaticRows;
    constexpr int kTailSharedBytes = kTailRows * kN * static_cast<int>(sizeof(double));
    const dim3 block(32, 32);

    if (m <= 0) {
        return;
    }

    ConfigureSplitTsqrDoubleLaunch<BlockSize, StaticRows>();

    if (m <= BlockSize) {
        split_tsqr_n32_double_kernel<BlockSize, StaticRows><<<1, block, kTailSharedBytes>>>(
            m, A, lda, R, ldr);
        return;
    }

    const int block_num = (m + BlockSize - 1) / BlockSize;
    const int stack_rows = block_num * kN;
    const size_t stack_elems = static_cast<size_t>(stack_rows) * kN;
    assert(stack_elems <= work_elems);
    const int ldwork = stack_rows;

    split_tsqr_n32_double_kernel<BlockSize, StaticRows><<<block_num, block, kTailSharedBytes>>>(
        m, A, lda, work, ldwork);

    SplitTsqrDouble<BlockSize, StaticRows>(cublas_handle, stack_rows, work, ldwork, R, ldr,
                                           work + stack_elems, work_elems - stack_elems);

    const double one = 1.0;
    const double zero = 0.0;
    const int full_blocks = m / BlockSize;
    if (full_blocks > 0) {
        AssertCublas(CublasGemmTraits<double>::GemmStridedBatched(
                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, BlockSize, kN, kN, &one, A, lda,
                         BlockSize, work, ldwork, kN, &zero, A, lda, BlockSize, full_blocks),
                     "split tsqr batched gemm");
    }

    const int remaining_rows = m % BlockSize;
    if (remaining_rows > 0) {
        AssertCublas(CublasGemmTraits<double>::Gemm(
                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, remaining_rows, kN, kN, &one,
                         A + (m - remaining_rows), lda, work + full_blocks * kN, ldwork, &zero,
                         A + (m - remaining_rows), lda),
                     "split tsqr tail gemm");
    }
}

template <typename T, int StaticBlockSize>
void HybridTsqr(cublasHandle_t cublas_handle,
                int switch_m,
                int m,
                T* A,
                int lda,
                T* R,
                int ldr,
                T* work,
                size_t work_elems) {
    const int effective_switch_m = (switch_m < kTsqrN) ? kTsqrN : switch_m;
    constexpr int kN = kTsqrN;
    const dim3 block(32, 32);

    if (m <= 0) {
        return;
    }
    if (m <= effective_switch_m) {
        tsqr(cublas_handle, m, A, lda, R, ldr, work, work_elems, nullptr);
        return;
    }
    if (m <= StaticBlockSize) {
        static_tsqr_n32_kernel<T, StaticBlockSize><<<1, block>>>(m, A, lda, R, ldr);
        return;
    }

    const int block_num = (m + StaticBlockSize - 1) / StaticBlockSize;
    const int stack_rows = block_num * kN;
    const size_t stack_elems = static_cast<size_t>(stack_rows) * kN;
    assert(stack_elems <= work_elems);
    const int ldwork = stack_rows;

    static_tsqr_n32_kernel<T, StaticBlockSize><<<block_num, block>>>(m, A, lda, work, ldwork);

    HybridTsqr<T, StaticBlockSize>(cublas_handle, effective_switch_m, stack_rows, work, ldwork, R,
                                   ldr, work + stack_elems, work_elems - stack_elems);

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const int full_blocks = m / StaticBlockSize;
    if (full_blocks > 0) {
        AssertCublas(CublasGemmTraits<T>::GemmStridedBatched(
                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, StaticBlockSize, kN, kN, &one,
                         A, lda, StaticBlockSize, work, ldwork, kN, &zero, A, lda, StaticBlockSize,
                         full_blocks),
                     "hybrid tsqr batched gemm");
    }

    const int remaining_rows = m % StaticBlockSize;
    if (remaining_rows > 0) {
        AssertCublas(CublasGemmTraits<T>::Gemm(
                         cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, remaining_rows, kN, kN, &one,
                         A + (m - remaining_rows), lda, work + full_blocks * kN, ldwork, &zero,
                         A + (m - remaining_rows), lda),
                     "hybrid tsqr tail gemm");
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
    return 1.0e-8f;
}

template <>
__device__ __forceinline__ double LegacyEpsilon<double>() {
    return 1.0e-14;
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
    const bool run_current_tsqr = ShouldRunCurrentTsqr(opts);
    const bool run_static192_tsqr = ShouldRunStatic192Tsqr(opts) && std::is_same_v<T, double>;
    const bool run_split256_tsqr = ShouldRunSplit256Tsqr(opts) && std::is_same_v<T, double>;
    const bool run_hybrid192_tsqr = ShouldRunHybrid192Tsqr(opts) && std::is_same_v<T, double>;
    const size_t static192_work_elems_count =
        run_static192_tsqr ? StaticTsqrWorkElems<192>(m) : static_cast<size_t>(0);
    const size_t split256_work_elems_count =
        run_split256_tsqr ? StaticTsqrWorkElems<256>(m) : static_cast<size_t>(0);
    const size_t hybrid192_work_elems_count =
        run_hybrid192_tsqr ? HybridTsqrWorkElems<T, 192>(m, opts.hybrid_switch_m)
                           : static_cast<size_t>(0);
    const bool legacy_supported = LegacyTsqrSupported(m, n);
    const size_t legacy_work_elems_count = legacy_supported ? LegacyTsqrWorkElems(m, n) : 0;

    T* d_A0 = nullptr;
    T* d_A_work = nullptr;
    T* d_R = nullptr;
    T* d_work_tsqr = nullptr;
    T* d_work_static192 = nullptr;
    T* d_work_split256 = nullptr;
    T* d_work_hybrid192 = nullptr;
    T* d_work_legacy = nullptr;
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
    if (static192_work_elems_count > 0) {
        AssertCuda(cudaMalloc(&d_work_static192, static192_work_elems_count * sizeof(T)),
                   "cudaMalloc d_work_static192");
    }
    if (split256_work_elems_count > 0) {
        AssertCuda(cudaMalloc(&d_work_split256, split256_work_elems_count * sizeof(T)),
                   "cudaMalloc d_work_split256");
    }
    if (hybrid192_work_elems_count > 0) {
        AssertCuda(cudaMalloc(&d_work_hybrid192, hybrid192_work_elems_count * sizeof(T)),
                   "cudaMalloc d_work_hybrid192");
    }
    if (legacy_work_elems_count > 0) {
        AssertCuda(cudaMalloc(&d_work_legacy, legacy_work_elems_count * sizeof(T)),
                   "cudaMalloc d_work_legacy");
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

    float current_tsqr_ms = 0.0f;
    if (run_current_tsqr) {
        current_tsqr_ms = TimeKernelMs(
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
    }

    float static192_tsqr_ms = 0.0f;
    if (run_static192_tsqr) {
        static192_tsqr_ms = TimeKernelMs(
            [&]() {
                AssertCuda(cudaMemcpy(d_A_work, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpy D2D static192 tsqr");
            },
            [&]() {
                StaticTsqr<T, 192>(cublas_handle, m, d_A_work, lda, d_R, ldr, d_work_static192,
                                   static192_work_elems_count);
                AssertCuda(cudaGetLastError(), "static192 tsqr launch");
            },
            opts.iters);
    }

    float split256_tsqr_ms = 0.0f;
    if constexpr (std::is_same_v<T, double>) {
        if (run_split256_tsqr) {
            split256_tsqr_ms = TimeKernelMs(
                [&]() {
                    AssertCuda(cudaMemcpy(d_A_work, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                               "cudaMemcpy D2D split256 tsqr");
                },
                [&]() {
                    SplitTsqrDouble<256, 192>(cublas_handle, m, d_A_work, lda, d_R, ldr,
                                              d_work_split256, split256_work_elems_count);
                    AssertCuda(cudaGetLastError(), "split256 tsqr launch");
                },
                opts.iters);
        }
    }

    float hybrid192_tsqr_ms = 0.0f;
    if (run_hybrid192_tsqr) {
        hybrid192_tsqr_ms = TimeKernelMs(
            [&]() {
                AssertCuda(cudaMemcpy(d_A_work, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpy D2D hybrid192 tsqr");
            },
            [&]() {
                HybridTsqr<T, 192>(cublas_handle, opts.hybrid_switch_m, m, d_A_work, lda, d_R, ldr,
                                   d_work_hybrid192, hybrid192_work_elems_count);
                AssertCuda(cudaGetLastError(), "hybrid192 tsqr launch");
            },
            opts.iters);
    }

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

    if (run_current_tsqr) {
        std::printf("Current TSQR avg: %.3f ms (%.3f TFLOPS)\n", current_tsqr_ms,
                    FlopsToTflops(qr_flops, current_tsqr_ms));
    } else {
        std::printf("Current TSQR avg: skipped by variant selection\n");
    }
    if (run_static192_tsqr) {
        std::printf("Static192 TSQR avg: %.3f ms (%.3f TFLOPS)\n", static192_tsqr_ms,
                    FlopsToTflops(qr_flops, static192_tsqr_ms));
    }
    if (run_split256_tsqr) {
        std::printf("Split256 TSQR avg: %.3f ms (%.3f TFLOPS)\n", split256_tsqr_ms,
                    FlopsToTflops(qr_flops, split256_tsqr_ms));
    }
    if (run_hybrid192_tsqr) {
        std::printf("Hybrid192 TSQR avg: %.3f ms (%.3f TFLOPS)\n", hybrid192_tsqr_ms,
                    FlopsToTflops(qr_flops, hybrid192_tsqr_ms));
    }
    if (legacy_supported) {
        std::printf("Legacy  TSQR avg: %.3f ms (%.3f TFLOPS)\n", legacy_tsqr_ms,
                    FlopsToTflops(qr_flops, legacy_tsqr_ms));
        if (run_current_tsqr) {
            std::printf("Current/Legacy speedup: %.3fx\n", legacy_tsqr_ms / current_tsqr_ms);
        }
        if (run_static192_tsqr) {
            std::printf("Static192/Legacy speedup: %.3fx\n", legacy_tsqr_ms / static192_tsqr_ms);
        }
        if (run_split256_tsqr) {
            std::printf("Split256/Legacy speedup: %.3fx\n", legacy_tsqr_ms / split256_tsqr_ms);
        }
        if (run_hybrid192_tsqr) {
            std::printf("Hybrid192/Legacy speedup: %.3fx\n", legacy_tsqr_ms / hybrid192_tsqr_ms);
        }
    } else {
        std::printf("Legacy  TSQR avg: skipped (requires m %% n == 0 and tail block aligned to n)\n");
    }
    if (run_current_tsqr && run_static192_tsqr) {
        std::printf("Current/Static192 speedup: %.3fx\n", static192_tsqr_ms / current_tsqr_ms);
    }
    if (run_current_tsqr && run_split256_tsqr) {
        std::printf("Current/Split256 speedup: %.3fx\n", split256_tsqr_ms / current_tsqr_ms);
    }
    if (run_current_tsqr && run_hybrid192_tsqr) {
        std::printf("Current/Hybrid192 speedup: %.3fx\n", hybrid192_tsqr_ms / current_tsqr_ms);
    }
    if (run_static192_tsqr && run_split256_tsqr) {
        std::printf("Static192/Split256 speedup: %.3fx\n", split256_tsqr_ms / static192_tsqr_ms);
    }
    if (run_static192_tsqr && run_hybrid192_tsqr) {
        std::printf("Static192/Hybrid192 speedup: %.3fx\n", hybrid192_tsqr_ms / static192_tsqr_ms);
    }
    if (run_split256_tsqr && run_hybrid192_tsqr) {
        std::printf("Split256/Hybrid192 speedup: %.3fx\n", hybrid192_tsqr_ms / split256_tsqr_ms);
    }
    std::printf("GEQRF avg:        %.3f ms (%.3f TFLOPS)\n", geqrf_ms, FlopsToTflops(qr_flops, geqrf_ms));

    AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    AssertCuda(cudaFree(d_A_work), "cudaFree d_A_work");
    AssertCuda(cudaFree(d_R), "cudaFree d_R");
    AssertCuda(cudaFree(d_work_tsqr), "cudaFree d_work_tsqr");
    AssertCuda(cudaFree(d_work_static192), "cudaFree d_work_static192");
    AssertCuda(cudaFree(d_work_split256), "cudaFree d_work_split256");
    AssertCuda(cudaFree(d_work_hybrid192), "cudaFree d_work_hybrid192");
    AssertCuda(cudaFree(d_work_legacy), "cudaFree d_work_legacy");
    AssertCuda(cudaFree(d_tau), "cudaFree d_tau");
    AssertCuda(cudaFree(d_work_geqrf), "cudaFree d_work_geqrf");
    AssertCuda(cudaFree(d_info), "cudaFree d_info");
}

}  // namespace

int main(int argc, char** argv) {
    const Options opts = ParseArgs(argc, argv);
    std::printf("TSQR bench: m=%d n=%d iters=%d warmup=%d type=%s", opts.m, kTsqrN, opts.iters,
                opts.warmup, opts.use_double ? "double" : "float");
    if (opts.use_double) {
        std::printf(" double_variant=%s hybrid_switch_m=%d", DoubleVariantName(opts.double_variant),
                    opts.hybrid_switch_m);
    }
    std::printf("\n");

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
