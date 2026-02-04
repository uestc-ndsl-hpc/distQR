#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

#include <spdlog/spdlog.h>

#include "components/panel_process.cuh"

namespace {

constexpr int kPanelWidth = 32;

void AssertCuda(cudaError_t status, const char* context) {
    ASSERT_EQ(status, cudaSuccess) << context << ": " << cudaGetErrorString(status);
}

void AssertCublas(cublasStatus_t status, const char* context) {
    ASSERT_EQ(status, CUBLAS_STATUS_SUCCESS) << context;
}

void AssertCurand(curandStatus_t status, const char* context) {
    ASSERT_EQ(status, CURAND_STATUS_SUCCESS) << context;
}

struct DeviceBufferF {
    float* ptr = nullptr;
    size_t elems = 0;

    DeviceBufferF() = default;
    explicit DeviceBufferF(size_t n)
        : elems(n) {
        AssertCuda(cudaMalloc(&ptr, elems * sizeof(float)), "cudaMalloc DeviceBufferF");
    }
    DeviceBufferF(const DeviceBufferF&) = delete;
    DeviceBufferF& operator=(const DeviceBufferF&) = delete;
    DeviceBufferF(DeviceBufferF&& other) noexcept
        : ptr(other.ptr),
          elems(other.elems) {
        other.ptr = nullptr;
        other.elems = 0;
    }
    DeviceBufferF& operator=(DeviceBufferF&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            elems = other.elems;
            other.ptr = nullptr;
            other.elems = 0;
        }
        return *this;
    }
    ~DeviceBufferF() {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

__global__ void PackUpperTriangleKernel(int n, const float* A, int lda, float* out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row >= 0 && row <= col) {
        const int64_t idx = static_cast<int64_t>(col) * (col + 1) / 2 + row;
        out[idx] = A[row + col * lda];
    }
}

__global__ void PackUpperTriangleDiffKernel(
    int n, const float* A, int lda, const float* ref_upper, float* out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row >= 0 && row <= col) {
        const int64_t idx = static_cast<int64_t>(col) * (col + 1) / 2 + row;
        out[idx] = A[row + col * lda] - ref_upper[idx];
    }
}

__global__ void PackStrictLowerTriangleKernel(int n, const float* A, int lda, float* out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row > col && row < n) {
        const int64_t col_off =
            static_cast<int64_t>(col) * (2LL * n - col - 1) / 2;  // sum_{c<col} (n-c-1)
        const int64_t idx = col_off + (row - col - 1);
        out[idx] = A[row + col * lda];
    }
}

void FillNormal(float* d_A, size_t elems, unsigned long long seed) {
    curandGenerator_t gen = nullptr;
    AssertCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
    AssertCurand(curandSetPseudoRandomGeneratorSeed(gen, seed),
                 "curandSetPseudoRandomGeneratorSeed");
    AssertCurand(curandGenerateNormal(gen, d_A, elems, 0.0f, 1.0f), "curandGenerateNormal");
    AssertCurand(curandDestroyGenerator(gen), "curandDestroyGenerator");
}

void SingleCardBlockedQrFactorize(float* d_A,
                                  int m,
                                  int n,
                                  int nb,
                                  cublasHandle_t cublas_handle,
                                  float* d_W,
                                  float* d_Y,
                                  float* d_rtmp,
                                  float* d_tsqr_work) {
    const int lda = m;
    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;

    for (int outer_index = 0; outer_index < n; outer_index += nb) {
        const int end = std::min(outer_index + nb, n);
        const int kb = end - outer_index;
        const int m_sub = m - outer_index;
        auto w_big = d_W + outer_index * lda + outer_index;
        auto y_big = d_Y + outer_index * lda + outer_index;

        for (int inner_index = outer_index; inner_index < end; inner_index += kPanelWidth) {
            const int panel_height = m - inner_index;
            auto panel_A = d_A + inner_index * lda + inner_index;
            auto panel_W = d_W + inner_index * lda + inner_index;
            auto panel_Y = d_Y + inner_index * lda + inner_index;

            tsqr<float>(cublas_handle, panel_height, panel_A, lda, d_rtmp, kPanelWidth, d_tsqr_work,
                        tsqr_work_elems<float>(panel_height), nullptr);
            generate_wy(panel_height, kPanelWidth, panel_A, lda, panel_Y, lda, panel_W, lda,
                        nullptr);
            write_back_R2A(kPanelWidth, kPanelWidth, d_rtmp, kPanelWidth, panel_A, lda, nullptr);

            // Update remaining columns inside this outer block:
            // A(inner:m, inner+b:end) <- Q_i^T * A(inner:m, inner+b:end)
            // Q_i = I - W_i * Y_i^T  =>  Q_i^T = I - Y_i * W_i^T
            // A -= Y_i * (W_i^T * A)
            const int n_remain_in_block = end - (inner_index + kPanelWidth);
            if (n_remain_in_block > 0) {
                auto a_remain = panel_A + kPanelWidth * lda;
                auto work = d_rtmp;  // (b x n_remain) fits in (nb x nb)
                CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kPanelWidth,
                                              n_remain_in_block, panel_height, &one, panel_W, lda,
                                              a_remain, lda, &zero, work, kPanelWidth);
                CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_height,
                                              n_remain_in_block, kPanelWidth, &minus_one, panel_Y,
                                              lda, work, kPanelWidth, &one, a_remain, lda);
            }

            // Accumulate (W,Y) for the whole outer block (compact WY):
            // Q_prev = I - W_prev * Y_prev^T
            // Q_new  = Q_prev * Q_i = I - [W_prev, Q_prev*W_i] * [Y_prev, Y_i]^T
            // Update W_i <- Q_prev * W_i = W_i - W_prev * (Y_prev^T * W_i)
            if (inner_index > outer_index) {
                const int k_prev = inner_index - outer_index;
                auto w_prev = w_big;                                   // (m_sub x k_prev)
                auto y_prev = y_big;                                   // (m_sub x k_prev)
                auto w_i_sub = d_W + inner_index * lda + outer_index;  // (m_sub x b)

                // tmp (k_prev x b) = Y_prev^T * W_i, stored in d_rtmp with leading dimension nb.
                CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, k_prev,
                                              kPanelWidth, m_sub, &one, y_prev, lda, w_i_sub, lda,
                                              &zero, d_rtmp, nb);
                // W_i -= W_prev * tmp  (alpha=-1, beta=1)
                CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_sub,
                                              kPanelWidth, k_prev, &minus_one, w_prev, lda, d_rtmp,
                                              nb, &one, w_i_sub, lda);
            }
        }

        // Outer-block update on trailing matrix:
        // A(outer:m, end:n) <- Q_block^T * A = A - Y_big * (W_big^T * A)
        const int n_trail = n - end;
        if (n_trail > 0) {
            auto a_trail = d_A + end * lda + outer_index;  // (m_sub x n_trail)
            for (int col0 = 0; col0 < n_trail; col0 += nb) {
                const int tile = std::min(nb, n_trail - col0);
                auto a_tile = a_trail + static_cast<size_t>(col0) * lda;  // (m_sub x tile)

                // work (kb x tile) = W_big^T * A_tile, stored in d_rtmp
                CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kb, tile,
                                              m_sub, &one, w_big, lda, a_tile, lda, &zero, d_rtmp,
                                              kb);
                // A_tile -= Y_big * work
                CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_sub, tile,
                                              kb, &minus_one, y_big, lda, d_rtmp, kb, &one, a_tile,
                                              lda);
            }
        }
    }
}

void ApplyAllOuterBlocksQT(float* d_A,
                           int m,
                           int n,
                           int nb,
                           cublasHandle_t cublas_handle,
                           const float* d_W,
                           const float* d_Y,
                           float* d_rtmp) {
    const int lda = m;
    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;

    // R = Q^T * A0 = (Q_k^T ... Q_0^T) * A0, so apply blocks in forward order.
    for (int outer_index = 0; outer_index < n; outer_index += nb) {
        const int end = std::min(outer_index + nb, n);
        const int kb = end - outer_index;
        const int m_sub = m - outer_index;
        const int n_sub = n - outer_index;
        auto w_big = d_W + outer_index * lda + outer_index;
        auto y_big = d_Y + outer_index * lda + outer_index;
        auto a_sub = d_A + outer_index * lda + outer_index;  // (m_sub x n_sub)

        for (int col0 = 0; col0 < n_sub; col0 += nb) {
            const int tile = std::min(nb, n_sub - col0);
            auto a_tile = a_sub + static_cast<size_t>(col0) * lda;  // (m_sub x tile)
            // work (kb x tile) = W_big^T * A_tile
            CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kb, tile, m_sub,
                                          &one, w_big, lda, a_tile, lda, &zero, d_rtmp, kb);
            // A_tile -= Y_big * work
            CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_sub, tile, kb,
                                          &minus_one, y_big, lda, d_rtmp, kb, &one, a_tile, lda);
        }
    }
}

}  // namespace

TEST(SingleCardQrTest, Norms16384UsingCublasNrm2) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "No CUDA device available.";
    }

    const int m = 16384;
    const int n = 16384;
    const int nb = 1024;

    ASSERT_EQ(n % kPanelWidth, 0);
    ASSERT_EQ(nb % kPanelWidth, 0);
    ASSERT_GE(m, n);

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    AssertCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    const size_t elems_A = static_cast<size_t>(m) * n;
    const size_t elems_WY = elems_A;
    const size_t elems_rtmp = static_cast<size_t>(nb) * nb;
    const size_t elems_tsqr = tsqr_work_elems<float>(m);
    const size_t elems_upper = static_cast<size_t>(n) * (n + 1) / 2;
    const size_t elems_lower = static_cast<size_t>(n) * (n - 1) / 2;
    const size_t bytes_needed = (elems_A + elems_WY + elems_WY + elems_rtmp + elems_tsqr +
                                 elems_upper + std::max(elems_upper, elems_lower)) *
                                sizeof(float);

    if (bytes_needed > static_cast<size_t>(static_cast<double>(free_bytes) * 0.85)) {
        GTEST_SKIP() << "Insufficient free GPU memory for 16384 test. Need ~"
                     << (bytes_needed / (1024.0 * 1024.0 * 1024.0)) << " GiB, have ~"
                     << (free_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB free.";
    }

    spdlog::info("SingleCardQrTest 16384: allocating ~{:.2f} GiB (free {:.2f} / total {:.2f} GiB)",
                 bytes_needed / (1024.0 * 1024.0 * 1024.0), free_bytes / (1024.0 * 1024.0 * 1024.0),
                 total_bytes / (1024.0 * 1024.0 * 1024.0));

    DeviceBufferF d_A(elems_A);
    DeviceBufferF d_W(elems_WY);
    DeviceBufferF d_Y(elems_WY);
    DeviceBufferF d_rtmp(elems_rtmp);
    DeviceBufferF d_tsqr(elems_tsqr > 0 ? elems_tsqr : 1);
    DeviceBufferF d_Rupper(elems_upper);
    DeviceBufferF d_pack(std::max(elems_upper, elems_lower));

    AssertCuda(cudaMemset(d_W.ptr, 0, elems_WY * sizeof(float)), "cudaMemset d_W");
    AssertCuda(cudaMemset(d_Y.ptr, 0, elems_WY * sizeof(float)), "cudaMemset d_Y");

    FillNormal(d_A.ptr, elems_A, 1234ULL);

    cublasHandle_t handle = nullptr;
    AssertCublas(cublasCreate(&handle), "cublasCreate");

    // Factorize once to get (W,Y) and capture R's upper triangle.
    SingleCardBlockedQrFactorize(d_A.ptr, m, n, nb, handle, d_W.ptr, d_Y.ptr, d_rtmp.ptr,
                                 d_tsqr.ptr);

    {
        dim3 block_dim(16, 16);
        dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);
        PackUpperTriangleKernel<<<grid_dim, block_dim>>>(n, d_A.ptr, m, d_Rupper.ptr);
        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize PackUpperTriangleKernel");
        AssertCuda(cudaGetLastError(), "cudaGetLastError PackUpperTriangleKernel");
    }

    // Regenerate A0 in-place (same seed), compute ||A0||_2, then apply Q^T from (W,Y).
    FillNormal(d_A.ptr, elems_A, 1234ULL);

    float norm_A0 = 0.0f;
    AssertCublas(cublasSnrm2(handle, static_cast<int>(elems_A), d_A.ptr, 1, &norm_A0),
                 "cublasSnrm2 A0");
    ASSERT_GT(norm_A0, 0.0f);

    ApplyAllOuterBlocksQT(d_A.ptr, m, n, nb, handle, d_W.ptr, d_Y.ptr, d_rtmp.ptr);

    // ||strictly_lower(Q^T*A0)||_2 should be small (ideally ~0 for square QR).
    float norm_lower = 0.0f;
    {
        dim3 block_dim(16, 16);
        dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);
        PackStrictLowerTriangleKernel<<<grid_dim, block_dim>>>(n, d_A.ptr, m, d_pack.ptr);
        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize PackStrictLowerTriangleKernel");
        AssertCuda(cudaGetLastError(), "cudaGetLastError PackStrictLowerTriangleKernel");
        AssertCublas(cublasSnrm2(handle, static_cast<int>(elems_lower), d_pack.ptr, 1, &norm_lower),
                     "cublasSnrm2 lower");
    }

    // ||upper(Q^T*A0) - R_upper||_2 should be small.
    float norm_upper_diff = 0.0f;
    {
        dim3 block_dim(16, 16);
        dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (n + block_dim.y - 1) / block_dim.y);
        PackUpperTriangleDiffKernel<<<grid_dim, block_dim>>>(n, d_A.ptr, m, d_Rupper.ptr,
                                                             d_pack.ptr);
        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize PackUpperTriangleDiffKernel");
        AssertCuda(cudaGetLastError(), "cudaGetLastError PackUpperTriangleDiffKernel");
        AssertCublas(
            cublasSnrm2(handle, static_cast<int>(elems_upper), d_pack.ptr, 1, &norm_upper_diff),
            "cublasSnrm2 upper diff");
    }

    spdlog::info("A0 nrm2={:.6e}, lower nrm2={:.6e}, upper-diff nrm2={:.6e}", norm_A0, norm_lower,
                 norm_upper_diff);

    const double rel_lower = norm_lower / static_cast<double>(norm_A0);
    const double rel_upper_diff = norm_upper_diff / static_cast<double>(norm_A0);
    spdlog::info("rel_lower={:.6e}, rel_upper_diff={:.6e}", rel_lower, rel_upper_diff);

    // Tolerances are intentionally loose for a large, multi-stage TSQR/WY pipeline.
    EXPECT_LT(rel_lower, 1e-6);
    EXPECT_LT(rel_upper_diff, 1e-6);

    AssertCublas(cublasDestroy(handle), "cublasDestroy");
}
