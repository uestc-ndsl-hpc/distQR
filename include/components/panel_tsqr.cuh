#include <cuda_runtime_api.h>
#include <cstddef>
#include <type_traits>

#include "utils/cublas_gemm_traits.cuh"

constexpr int double_block_size = 256;
constexpr int float_block_size = 256;
constexpr int kTsqrN32Cols = 32;
constexpr int kTsqrN32DoubleLegacySharedStride = double_block_size;
// Padding each double column by one element breaks cross-column bank aliasing.
constexpr int kTsqrN32DoublePaddedSharedStride = double_block_size + 1;
constexpr size_t kTsqrN32DoubleLegacySharedBytes =
    static_cast<size_t>(kTsqrN32DoubleLegacySharedStride) * static_cast<size_t>(kTsqrN32Cols) *
    sizeof(double);
constexpr size_t kTsqrN32DoublePaddedSharedBytes =
    static_cast<size_t>(kTsqrN32DoublePaddedSharedStride) * static_cast<size_t>(kTsqrN32Cols) *
    sizeof(double);

struct TsqrN32DoubleLaunchConfig {
    bool use_padded_stride = false;
    size_t shared_bytes = kTsqrN32DoubleLegacySharedBytes;
};

template <typename T>
static __inline__ __device__ T warp_all_reduce_sum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
constexpr int tsqr_block_size() {
    if constexpr (std::is_same_v<T, double>) {
        return double_block_size;
    } else {
        return float_block_size;
    }
}

template <typename T>
size_t tsqr_work_elems(int m) {
    constexpr int tsqr_n32_n = 32;
    const int block_size = tsqr_block_size<T>();
    size_t total = 0;
    int current = m;
    while (current > block_size) {
        const int block_num = (current + block_size - 1) / block_size;
        const int stack_rows = block_num * tsqr_n32_n;
        total += static_cast<size_t>(stack_rows) * tsqr_n32_n;
        current = stack_rows;
    }
    return total;
}

__global__ void tsqr_n32_float(int m, float* A, int lda, float* R, int ldr) {
    constexpr int tsqr_n32_block_size = float_block_size;
    constexpr int tsqr_n32_n = 32;
    constexpr int warp_size = 32;
    constexpr int tsqr_n32_data_num_per_thread =
        (tsqr_n32_block_size + warp_size - 1) / warp_size;
    constexpr float epsilon = 1e-4;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int bx = blockIdx.x;

    A += bx * tsqr_n32_block_size;
    R += bx * tsqr_n32_n;

    int block_size = min(tsqr_n32_block_size, m - bx * tsqr_n32_block_size);

    __shared__ float shared_A[tsqr_n32_block_size * tsqr_n32_n];
#pragma unroll
    for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
        auto row_idx = lane_id + i * warp_size;
        if (row_idx < block_size) {
            shared_A[row_idx + warp_id * tsqr_n32_block_size] = A[row_idx + warp_id * lda];
        }
    }
    __syncthreads();

    float q[tsqr_n32_data_num_per_thread];
    for (auto col = 0; col < tsqr_n32_n; ++col) {
        volatile float nu = 0.f;
        if (warp_id == col) {
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx >= col && row_idx < block_size) {
                    q[i] = shared_A[row_idx + col * tsqr_n32_block_size];
                    nu += q[i] * q[i];
                }
            }
            float norm_square = warp_all_reduce_sum(nu);
            if (norm_square > epsilon * epsilon) {
                float norm = __fsqrt_rn(norm_square);
                float scale = 1.f / norm;
#pragma unroll
                for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                    auto row_idx = lane_id + i * warp_size;
                    if (row_idx >= col && row_idx < block_size) {
                        q[i] *= scale;
                    }
                }
                float u1 = 0.f;
                auto thread_off = col / warp_size;
                if (lane_id == col % warp_size) {
                    q[thread_off] += (q[thread_off] >= 0) ? 1.f : -1.f;
                    u1 = q[thread_off];
                    R[col + ldr * col] = (u1 >= 0) ? -norm : norm;
                }
                u1 = __shfl_sync(0xffffffff, u1, col % warp_size);
                scale = __frsqrt_rn(fabs(u1));
#pragma unroll
                for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                    auto row_idx = lane_id + i * warp_size;
                    if (row_idx >= col && row_idx < block_size) {
                        shared_A[row_idx + col * tsqr_n32_block_size] = q[i] * scale;
                    }
                }
            } else {
                if (lane_id == col) {
                    R[col + ldr * col] = 0.f;
                }
#pragma unroll
                for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                    auto row_idx = lane_id + i * warp_size;
                    if (row_idx >= col && row_idx < block_size) {
                        shared_A[row_idx + col * tsqr_n32_block_size] = 0.f;
                    }
                }
            }
        }
        __syncthreads();

        if (col < warp_id) {
            volatile float nu = 0.f;
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx >= col && row_idx < block_size) {
                    q[i] = shared_A[row_idx + col * tsqr_n32_block_size];
                    nu += q[i] * shared_A[row_idx + warp_id * tsqr_n32_block_size];
                }
            }
            float utx = warp_all_reduce_sum(nu);
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx >= col && row_idx < block_size) {
                    shared_A[row_idx + warp_id * tsqr_n32_block_size] -= utx * q[i];
                }
            }
        }
    }

    __syncthreads();

    if (lane_id < warp_id) {
        R[lane_id + ldr * warp_id] = shared_A[lane_id + warp_id * tsqr_n32_block_size];
        shared_A[lane_id + warp_id * tsqr_n32_block_size] = 0.f;
    }
    if (lane_id > warp_id) {
        R[lane_id + ldr * warp_id] = 0.f;
    }

#pragma unroll
    for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
        auto row_idx = lane_id + i * warp_size;
        q[i] = (row_idx == warp_id) ? 1.f : 0.f;
    }
    __syncwarp();
    for (auto col = tsqr_n32_n - 1; col >= 0; --col) {
        if (warp_id >= col) {
            float nu = 0.f;
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx < block_size) {
                    nu += q[i] * shared_A[row_idx + col * tsqr_n32_block_size];
                }
            }
            float utq = warp_all_reduce_sum(nu);
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx < block_size) {
                    q[i] -= utq * shared_A[row_idx + col * tsqr_n32_block_size];
                }
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
        auto row_idx = lane_id + i * warp_size;
        if (row_idx < block_size) {
            A[row_idx + warp_id * lda] = q[i];
        }
    }
}

template <int SharedStride>
__device__ __forceinline__ double tsqr_n32_double_shared_read(const double* shared_A,
                                                              int row,
                                                              int col) {
    return shared_A[row + col * SharedStride];
}

template <int SharedStride>
__device__ __forceinline__ void tsqr_n32_double_shared_write(double* shared_A,
                                                             int row,
                                                             int col,
                                                             double value) {
    shared_A[row + col * SharedStride] = value;
}

template <int SharedStride>
__global__ void tsqr_n32_double(int m, double* A, int lda, double* R, int ldr) {
    constexpr int tsqr_n32_block_size = double_block_size;
    constexpr int tsqr_n32_n = 32;
    constexpr int warp_size = 32;
    constexpr int tsqr_n32_data_num_per_thread =
        (tsqr_n32_block_size + warp_size - 1) / warp_size;
    constexpr double epsilon = 1e-7;
    static_assert(SharedStride >= tsqr_n32_block_size,
                  "Double TSQR shared stride must cover the full block size.");
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int bx = blockIdx.x;

    A += bx * tsqr_n32_block_size;
    R += bx * tsqr_n32_n;

    int block_size = min(tsqr_n32_block_size, m - bx * tsqr_n32_block_size);

    extern __shared__ double shared_A[];
#pragma unroll
    for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
        auto row_idx = lane_id + i * warp_size;
        if (row_idx < block_size) {
            tsqr_n32_double_shared_write<SharedStride>(shared_A, row_idx, warp_id,
                                                       A[row_idx + warp_id * lda]);
        }
    }
    __syncthreads();

    double q[tsqr_n32_data_num_per_thread];
    for (auto col = 0; col < tsqr_n32_n; ++col) {
        double nu = 0.0;
        if (warp_id == col) {
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx >= col && row_idx < block_size) {
                    q[i] = tsqr_n32_double_shared_read<SharedStride>(shared_A, row_idx, col);
                    nu += q[i] * q[i];
                }
            }
            double norm_square = warp_all_reduce_sum(nu);
            if (norm_square > epsilon * epsilon) {
                double norm = __dsqrt_rn(norm_square);
                double scale = 1.0 / norm;
#pragma unroll
                for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                    auto row_idx = lane_id + i * warp_size;
                    if (row_idx >= col && row_idx < block_size) {
                        q[i] *= scale;
                    }
                }
                double u1 = 0.0;
                auto thread_off = col / warp_size;
                if (lane_id == col % warp_size) {
                    q[thread_off] += (q[thread_off] >= 0) ? 1.0 : -1.0;
                    u1 = q[thread_off];
                    R[col + ldr * col] = (u1 >= 0) ? -norm : norm;
                }
                u1 = __shfl_sync(0xffffffff, u1, col % warp_size);
                scale = rsqrt(fabs(u1));
#pragma unroll
                for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                    auto row_idx = lane_id + i * warp_size;
                    if (row_idx >= col && row_idx < block_size) {
                        tsqr_n32_double_shared_write<SharedStride>(shared_A, row_idx, col,
                                                                   q[i] * scale);
                    }
                }
            } else {
                if (lane_id == col) {
                    R[col + ldr * col] = 0.0;
                }
#pragma unroll
                for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                    auto row_idx = lane_id + i * warp_size;
                    if (row_idx >= col && row_idx < block_size) {
                        tsqr_n32_double_shared_write<SharedStride>(shared_A, row_idx, col, 0.0);
                    }
                }
            }
        }
        __syncthreads();

        if (col < warp_id) {
            double nu = 0.0;
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx >= col && row_idx < block_size) {
                    q[i] = tsqr_n32_double_shared_read<SharedStride>(shared_A, row_idx, col);
                    nu += q[i] *
                          tsqr_n32_double_shared_read<SharedStride>(shared_A, row_idx, warp_id);
                }
            }
            double utx = warp_all_reduce_sum(nu);
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx >= col && row_idx < block_size) {
                    const double updated =
                        tsqr_n32_double_shared_read<SharedStride>(shared_A, row_idx, warp_id) -
                        utx * q[i];
                    tsqr_n32_double_shared_write<SharedStride>(shared_A, row_idx, warp_id,
                                                               updated);
                }
            }
        }
    }

    __syncthreads();

    if (lane_id < warp_id) {
        R[lane_id + ldr * warp_id] =
            tsqr_n32_double_shared_read<SharedStride>(shared_A, lane_id, warp_id);
        tsqr_n32_double_shared_write<SharedStride>(shared_A, lane_id, warp_id, 0.0);
    }
    if (lane_id > warp_id) {
        R[lane_id + ldr * warp_id] = 0.0;
    }

#pragma unroll
    for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
        auto row_idx = lane_id + i * warp_size;
        q[i] = (row_idx == warp_id) ? 1.0 : 0.0;
    }
    __syncwarp();
    for (auto col = tsqr_n32_n - 1; col >= 0; --col) {
        if (warp_id >= col) {
            double nu = 0.0;
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx < block_size) {
                    nu += q[i] * tsqr_n32_double_shared_read<SharedStride>(shared_A, row_idx, col);
                }
            }
            double utq = warp_all_reduce_sum(nu);
#pragma unroll
            for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                auto row_idx = lane_id + i * warp_size;
                if (row_idx < block_size) {
                    q[i] -= utq * tsqr_n32_double_shared_read<SharedStride>(shared_A, row_idx, col);
                }
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
        auto row_idx = lane_id + i * warp_size;
        if (row_idx < block_size) {
            A[row_idx + warp_id * lda] = q[i];
        }
    }
}

template <int SharedStride>
static inline bool configure_tsqr_n32_double_launch(size_t shared_bytes) {
    constexpr int kPreferSharedMemory = 100;
    const int shared_bytes_int = static_cast<int>(shared_bytes);
    const auto max_dynamic_shared_status =
        cudaFuncSetAttribute(tsqr_n32_double<SharedStride>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes_int);
    if (max_dynamic_shared_status != cudaSuccess) {
        return false;
    }
    const auto carveout_status =
        cudaFuncSetAttribute(tsqr_n32_double<SharedStride>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             kPreferSharedMemory);
    return carveout_status == cudaSuccess;
}

static inline TsqrN32DoubleLaunchConfig select_tsqr_n32_double_launch_config() {
    TsqrN32DoubleLaunchConfig config;

    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
        int max_shared_bytes = 0;
        if (cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   device) == cudaSuccess &&
            max_shared_bytes >= static_cast<int>(kTsqrN32DoublePaddedSharedBytes) &&
            configure_tsqr_n32_double_launch<kTsqrN32DoublePaddedSharedStride>(
                kTsqrN32DoublePaddedSharedBytes)) {
            config.use_padded_stride = true;
            config.shared_bytes = kTsqrN32DoublePaddedSharedBytes;
            return config;
        }
    }

    configure_tsqr_n32_double_launch<kTsqrN32DoubleLegacySharedStride>(
        kTsqrN32DoubleLegacySharedBytes);
    return config;
}

static inline void launch_tsqr_n32_double(dim3 grid,
                                          dim3 block,
                                          cudaStream_t stream,
                                          int m,
                                          double* A,
                                          int lda,
                                          double* R,
                                          int ldr,
                                          const TsqrN32DoubleLaunchConfig& launch_config) {
    if (launch_config.use_padded_stride) {
        tsqr_n32_double<kTsqrN32DoublePaddedSharedStride>
            <<<grid, block, launch_config.shared_bytes, stream>>>(m, A, lda, R, ldr);
        return;
    }
    tsqr_n32_double<kTsqrN32DoubleLegacySharedStride>
        <<<grid, block, launch_config.shared_bytes, stream>>>(m, A, lda, R, ldr);
}

/**
Full TSQR for n == 32.
Note: Constructing the full Q and R requires recursive factorization of stacked
R and batched GEMM to apply the stacked Q to each block.
*/
template <typename T>
void tsqr_impl(cublasHandle_t cublas_handle,
               int m,
               T* A,
               int lda,
               T* R,
               int ldr,
               T* work,
               size_t work_elems,
               cudaStream_t stream,
               const TsqrN32DoubleLaunchConfig& double_launch_config) {
    const int tsqr_n32_block_size = tsqr_block_size<T>();
    constexpr int tsqr_n32_n = 32;
    constexpr int warp_size = 32;
    constexpr int warp_num = 32;

    if (m <= 0) {
        return;
    }

    const dim3 block(warp_size, warp_num);

    if (m <= tsqr_n32_block_size) {
        if constexpr (std::is_same_v<T, float>) {
            tsqr_n32_float<<<1, block, 0, stream>>>(m, A, lda, R, ldr);
        } else if constexpr (std::is_same_v<T, double>) {
            launch_tsqr_n32_double(dim3(1), block, stream, m, A, lda, R, ldr,
                                   double_launch_config);
        } else {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "tsqr_recursive only supports float and double.");
        }
        return;
    }

    const int block_num = (m + tsqr_n32_block_size - 1) / tsqr_n32_block_size;
    const int stack_rows = block_num * tsqr_n32_n;
    const size_t stack_elems = static_cast<size_t>(stack_rows) * tsqr_n32_n;
    if (stack_elems > work_elems) {
        return;
    }
    const int ldwork = stack_rows;

    if constexpr (std::is_same_v<T, float>) {
        tsqr_n32_float<<<block_num, block, 0, stream>>>(m, A, lda, work, ldwork);
    } else if constexpr (std::is_same_v<T, double>) {
        launch_tsqr_n32_double(dim3(block_num), block, stream, m, A, lda, work, ldwork,
                               double_launch_config);
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "tsqr_recursive only supports float and double.");
    }

    tsqr_impl(cublas_handle, stack_rows, work, ldwork, R, ldr, work + stack_elems,
              work_elems - stack_elems, stream, double_launch_config);

    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const int full_blocks = m / tsqr_n32_block_size;
    if (full_blocks > 0) {
        CublasGemmTraits<T>::GemmStridedBatched(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, tsqr_n32_block_size, tsqr_n32_n, tsqr_n32_n,
            &one, A, lda, tsqr_n32_block_size, work, ldwork, tsqr_n32_n, &zero, A, lda,
            tsqr_n32_block_size, full_blocks);
    }

    const int remaining_rows = m % tsqr_n32_block_size;
    if (remaining_rows > 0) {
        CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, remaining_rows,
                                  tsqr_n32_n, tsqr_n32_n, &one, A + (m - remaining_rows), lda,
                                  work + full_blocks * tsqr_n32_n, ldwork, &zero,
                                  A + (m - remaining_rows), lda);
    }
}

template <typename T>
void tsqr(cublasHandle_t cublas_handle,
          int m,
          T* A,
          int lda,
          T* R,
          int ldr,
          T* work,
          size_t work_elems,
          cudaStream_t stream) {
    if constexpr (std::is_same_v<T, double>) {
        const TsqrN32DoubleLaunchConfig double_launch_config =
            select_tsqr_n32_double_launch_config();
        tsqr_impl(cublas_handle, m, A, lda, R, ldr, work, work_elems, stream,
                  double_launch_config);
    } else {
        tsqr_impl(cublas_handle, m, A, lda, R, ldr, work, work_elems, stream,
                  TsqrN32DoubleLaunchConfig{});
    }
}
