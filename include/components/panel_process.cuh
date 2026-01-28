#include <cuda_runtime_api.h>
#include <cstddef>
#include <type_traits>

#include "utils/cublas_gemm_traits.cuh"

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
        return 192;
    } else {
        return 256;
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
    constexpr int tsqr_n32_block_size = 256;
    constexpr int tsqr_n32_n = 32;
    constexpr int tsqr_n32_data_num_per_thread = 8;
    constexpr int warp_size = 32;
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
        float nu = 0.f;
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
            float nu = 0.f;
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

__global__ void tsqr_n32_double(int m, double* A, int lda, double* R, int ldr) {
    constexpr int tsqr_n32_block_size = 192;
    constexpr int tsqr_n32_n = 32;
    constexpr int tsqr_n32_data_num_per_thread = 8;
    constexpr int warp_size = 32;
    constexpr double epsilon = 1e-7;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int bx = blockIdx.x;

    A += bx * tsqr_n32_block_size;
    R += bx * tsqr_n32_n;

    int block_size = min(tsqr_n32_block_size, m - bx * tsqr_n32_block_size);

    __shared__ double shared_A[tsqr_n32_block_size * tsqr_n32_n];
#pragma unroll
    for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
        auto row_idx = lane_id + i * warp_size;
        if (row_idx < block_size) {
            shared_A[row_idx + warp_id * tsqr_n32_block_size] = A[row_idx + warp_id * lda];
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
                    q[i] = shared_A[row_idx + col * tsqr_n32_block_size];
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
                scale = __drcp_rn(__dsqrt_rn(abs(u1)));
#pragma unroll
                for (auto i = 0; i < tsqr_n32_data_num_per_thread; ++i) {
                    auto row_idx = lane_id + i * warp_size;
                    if (row_idx >= col && row_idx < block_size) {
                        shared_A[row_idx + col * tsqr_n32_block_size] = q[i] * scale;
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
                        shared_A[row_idx + col * tsqr_n32_block_size] = 0.0;
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
                    q[i] = shared_A[row_idx + col * tsqr_n32_block_size];
                    nu += q[i] * shared_A[row_idx + warp_id * tsqr_n32_block_size];
                }
            }
            double utx = warp_all_reduce_sum(nu);
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
        shared_A[lane_id + warp_id * tsqr_n32_block_size] = 0.0;
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
                    nu += q[i] * shared_A[row_idx + col * tsqr_n32_block_size];
                }
            }
            double utq = warp_all_reduce_sum(nu);
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

/**
Full TSQR for n == 32.
Note: Constructing the full Q and R requires recursive factorization of stacked
R and batched GEMM to apply the stacked Q to each block.
*/
template <typename T>
void tsqr(
    cublasHandle_t cublas_handle, int m, T* A, int lda, T* R, int ldr, T* work, size_t work_elems) {
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
            tsqr_n32_float<<<1, block>>>(m, A, lda, R, ldr);
        } else if constexpr (std::is_same_v<T, double>) {
            tsqr_n32_double<<<1, block>>>(m, A, lda, R, ldr);
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
        tsqr_n32_float<<<block_num, block>>>(m, A, lda, work, ldwork);
    } else if constexpr (std::is_same_v<T, double>) {
        tsqr_n32_double<<<block_num, block>>>(m, A, lda, work, ldwork);
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "tsqr_recursive only supports float and double.");
    }

    tsqr(cublas_handle, stack_rows, work, ldwork, R, ldr, work + stack_elems,
         work_elems - stack_elems);

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
