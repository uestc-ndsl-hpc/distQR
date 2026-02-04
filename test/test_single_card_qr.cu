#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

#include "components/panel_process.cuh"

namespace {

constexpr int kPanelWidth = 32;

static inline size_t Idx2D(int row, int col, int ld) {
    return static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(ld);
}

void AssertCuda(cudaError_t status, const char* context) {
    ASSERT_EQ(status, cudaSuccess) << context << ": " << cudaGetErrorString(status);
}

void AssertCublas(cublasStatus_t status, const char* context) {
    ASSERT_EQ(status, CUBLAS_STATUS_SUCCESS) << context;
}

void AssertCurand(curandStatus_t status, const char* context) {
    ASSERT_EQ(status, CURAND_STATUS_SUCCESS) << context;
}

struct ReusableDeviceBufferF {
    float* ptr = nullptr;
    size_t capacity_elems = 0;

    ~ReusableDeviceBufferF() { Release(); }
    ReusableDeviceBufferF() = default;
    ReusableDeviceBufferF(const ReusableDeviceBufferF&) = delete;
    ReusableDeviceBufferF& operator=(const ReusableDeviceBufferF&) = delete;

    void Release() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            capacity_elems = 0;
        }
    }

    bool Ensure(size_t elems) {
        if (elems <= capacity_elems) {
            return true;
        }
        Release();
        cudaError_t st = cudaMalloc(&ptr, elems * sizeof(float));
        if (st != cudaSuccess) {
            ptr = nullptr;
            capacity_elems = 0;
            return false;
        }
        capacity_elems = elems;
        return true;
    }
};

struct ReusableNormalRng {
    curandGenerator_t gen = nullptr;

    ~ReusableNormalRng() {
        if (gen) {
            curandDestroyGenerator(gen);
        }
    }

    void EnsureCreated() {
        if (!gen) {
            AssertCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT),
                         "curandCreateGenerator");
        }
    }

    void Fill(float* d_A, size_t elems, unsigned long long seed) {
        EnsureCreated();
        AssertCurand(curandSetPseudoRandomGeneratorSeed(gen, seed),
                     "curandSetPseudoRandomGeneratorSeed");
        AssertCurand(curandGenerateNormal(gen, d_A, elems, 0.0f, 1.0f), "curandGenerateNormal");
    }
};

double Nrm2Large(cublasHandle_t handle, const float* d_x, size_t n) {
    // Even though the API takes `int n`, very large `n` near INT_MAX can fail on some systems
    // (observed as CUBLAS_STATUS_EXECUTION_FAILED). Use a conservative chunk size.
    constexpr size_t kMaxChunk = static_cast<size_t>(1) << 26;  // 67,108,864 elements
    double sumsq = 0.0;
    size_t offset = 0;
    while (offset < n) {
        const int chunk = static_cast<int>(std::min(kMaxChunk, n - offset));
        float chunk_norm = 0.0f;
        AssertCublas(cublasSnrm2(handle, chunk, d_x + offset, 1, &chunk_norm), "cublasSnrm2 chunk");
        const double cn = static_cast<double>(chunk_norm);
        sumsq += cn * cn;
        offset += static_cast<size_t>(chunk);
    }
    return std::sqrt(sumsq);
}

__global__ void PackFullBlockKernel(int b, const float* A, int lda, float* out, int ldout) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < b && col < b) {
        out[row + col * ldout] = A[row + col * lda];
    }
}

__global__ void PackFullBlockDiffKernel(
    int b, const float* A, int lda, const float* B, int ldb, float* out, int ldout) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < b && col < b) {
        out[row + col * ldout] = A[row + col * lda] - B[row + col * ldb];
    }
}

__global__ void PackStrictLowerBlockKernel(int b, const float* A, int lda, float* out, int ldout) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < b && col < b) {
        out[row + col * ldout] = (row > col) ? A[row + col * lda] : 0.0f;
    }
}

__global__ void PackUpperBlockDiffKernel(
    int b, const float* A, int lda, const float* B, int ldb, float* out, int ldout) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < b && col < b) {
        out[row + col * ldout] = (row <= col) ? (A[row + col * lda] - B[row + col * ldb]) : 0.0f;
    }
}

void SingleCardBlockedQrFactorize(float* d_Afact,
                                  int m,
                                  int n,
                                  int nb,
                                  cublasHandle_t cublas_handle,
                                  float* d_W,
                                  float* d_Y,
                                  float* d_rtmp,
                                  float* d_tsqr_work,
                                  size_t tsqr_work_elems_m) {
    const int lda = m;
    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;

    for (int outer_index = 0; outer_index < n; outer_index += nb) {
        const int end = std::min(outer_index + nb, n);
        const int kb = end - outer_index;
        const int m_sub = m - outer_index;
        auto w_big = d_W + Idx2D(outer_index, outer_index, lda);
        auto y_big = d_Y + Idx2D(outer_index, outer_index, lda);

        for (int inner_index = outer_index; inner_index < end; inner_index += kPanelWidth) {
            const int panel_height = m - inner_index;
            auto panel_A = d_Afact + Idx2D(inner_index, inner_index, lda);
            auto panel_W = d_W + Idx2D(inner_index, inner_index, lda);
            auto panel_Y = d_Y + Idx2D(inner_index, inner_index, lda);

            tsqr<float>(cublas_handle, panel_height, panel_A, lda, d_rtmp, kPanelWidth, d_tsqr_work,
                        tsqr_work_elems_m, nullptr);
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
                auto w_prev = w_big;                                        // (m_sub x k_prev)
                auto y_prev = y_big;                                        // (m_sub x k_prev)
                auto w_i_sub = d_W + Idx2D(outer_index, inner_index, lda);  // (m_sub x b)

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
            auto a_trail = d_Afact + Idx2D(outer_index, end, lda);  // (m_sub x n_trail)
            for (int col0 = 0; col0 < n_trail; col0 += nb) {
                const int tile = std::min(nb, n_trail - col0);
                auto a_tile = a_trail + static_cast<size_t>(col0) *
                                            static_cast<size_t>(lda);  // (m_sub x tile)

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

void ApplyAllOuterBlocksQT(float* d_A0,
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

    for (int outer_index = 0; outer_index < n; outer_index += nb) {
        const int end = std::min(outer_index + nb, n);
        const int kb = end - outer_index;
        const int m_sub = m - outer_index;
        const int n_sub = n - outer_index;
        auto w_big = d_W + Idx2D(outer_index, outer_index, lda);
        auto y_big = d_Y + Idx2D(outer_index, outer_index, lda);
        auto a_sub = d_A0 + Idx2D(outer_index, outer_index, lda);  // (m_sub x n_sub)

        for (int col0 = 0; col0 < n_sub; col0 += nb) {
            const int tile = std::min(nb, n_sub - col0);
            auto a_tile =
                a_sub + static_cast<size_t>(col0) * static_cast<size_t>(lda);  // (m_sub x tile)
            CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kb, tile, m_sub,
                                          &one, w_big, lda, a_tile, lda, &zero, d_rtmp, kb);
            CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_sub, tile, kb,
                                          &minus_one, y_big, lda, d_rtmp, kb, &one, a_tile, lda);
        }
    }
}

struct QrCase {
    int n = 0;
    int nb = 0;
};

void ApplyAllOuterBlocksQTToMatrix(float* d_X,
                                   int rows,
                                   int cols,
                                   int nb,
                                   cublasHandle_t cublas_handle,
                                   const float* d_W,
                                   const float* d_Y,
                                   float* d_rtmp) {
    const int lda = rows;
    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;

    for (int outer_index = 0; outer_index < rows; outer_index += nb) {
        const int end = std::min(outer_index + nb, rows);
        const int kb = end - outer_index;
        const int m_sub = rows - outer_index;
        auto w_big = d_W + Idx2D(outer_index, outer_index, lda);
        auto y_big = d_Y + Idx2D(outer_index, outer_index, lda);
        auto x_sub = d_X + Idx2D(outer_index, 0, lda);  // (m_sub x cols)

        for (int col0 = 0; col0 < cols; col0 += nb) {
            const int tile = std::min(nb, cols - col0);
            auto x_tile = x_sub + static_cast<size_t>(col0) * static_cast<size_t>(lda);
            CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kb, tile, m_sub,
                                          &one, w_big, lda, x_tile, lda, &zero, d_rtmp, kb);
            CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_sub, tile, kb,
                                          &minus_one, y_big, lda, d_rtmp, kb, &one, x_tile, lda);
        }
    }
}

void ApplyAllOuterBlocksQToMatrix(float* d_X,
                                  int rows,
                                  int cols,
                                  int nb,
                                  cublasHandle_t cublas_handle,
                                  const float* d_W,
                                  const float* d_Y,
                                  float* d_rtmp) {
    const int lda = rows;
    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;

    // If the overall Q^T is formed by applying block Q_i^T in increasing `outer_index`
    // during factorization, then Q itself is the reverse product. Apply blocks in
    // reverse order so that X <- Q * X matches (Q^T)^T.
    const int last_block = ((rows + nb - 1) / nb - 1) * nb;
    for (int outer_index = last_block; outer_index >= 0; outer_index -= nb) {
        const int end = std::min(outer_index + nb, rows);
        const int kb = end - outer_index;
        const int m_sub = rows - outer_index;
        auto w_big = d_W + Idx2D(outer_index, outer_index, lda);
        auto y_big = d_Y + Idx2D(outer_index, outer_index, lda);
        auto x_sub = d_X + Idx2D(outer_index, 0, lda);  // (m_sub x cols)

        for (int col0 = 0; col0 < cols; col0 += nb) {
            const int tile = std::min(nb, cols - col0);
            auto x_tile = x_sub + static_cast<size_t>(col0) * static_cast<size_t>(lda);
            // work = Y^T * X_tile
            CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kb, tile, m_sub,
                                          &one, y_big, lda, x_tile, lda, &zero, d_rtmp, kb);
            // X_tile -= W * work
            CublasGemmTraits<float>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_sub, tile, kb,
                                          &minus_one, w_big, lda, d_rtmp, kb, &one, x_tile, lda);
        }
    }
}

__global__ void SubVectorKernel(size_t n, float* y, const float* x) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] -= x[idx];
    }
}

static void RunQrCase(const QrCase& test_case,
                      ReusableDeviceBufferF& d_A0,
                      ReusableDeviceBufferF& d_Afact,
                      ReusableDeviceBufferF& d_W,
                      ReusableDeviceBufferF& d_Y,
                      ReusableDeviceBufferF& d_rtmp,
                      ReusableDeviceBufferF& d_tsqr,
                      ReusableDeviceBufferF& d_pack,
                      ReusableDeviceBufferF& d_X,
                      ReusableDeviceBufferF& d_Xorig,
                      ReusableNormalRng& rng) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "No CUDA device available.";
    }

    const int n = test_case.n;
    const int m = n;
    const int nb = test_case.nb;

    ASSERT_GT(m, 0);
    ASSERT_GT(n, 0);
    ASSERT_GT(nb, 0);
    ASSERT_EQ(n % kPanelWidth, 0);
    ASSERT_EQ(nb % kPanelWidth, 0);
    ASSERT_EQ(n % nb, 0) << "This test assumes n is divisible by nb.";

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    AssertCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    const size_t elems_A = static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t elems_rtmp = static_cast<size_t>(nb) * static_cast<size_t>(nb);
    const size_t elems_tsqr = std::max(tsqr_work_elems<float>(m), static_cast<size_t>(1));
    constexpr int kOrthoCols = 8;
    const size_t elems_X = static_cast<size_t>(m) * static_cast<size_t>(kOrthoCols);

    // Two A's (A0 + Afact) + W + Y + rtmp + tsqr + pack(nb*nb).
    const size_t bytes_needed =
        (elems_A * 4 + elems_rtmp * 2 + elems_tsqr + elems_X * 2) * sizeof(float);

    auto release_all = [&]() {
        d_A0.Release();
        d_Afact.Release();
        d_W.Release();
        d_Y.Release();
        d_rtmp.Release();
        d_tsqr.Release();
        d_pack.Release();
    };

    auto have_enough_mem = [&](size_t free_b) {
        return bytes_needed <= static_cast<size_t>(static_cast<double>(free_b) * 0.85);
    };

    if (!have_enough_mem(free_bytes)) {
        // If previous tests left buffers allocated, free them and re-check.
        release_all();
        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after Release");
        AssertCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo after Release");
    }

    if (!have_enough_mem(free_bytes)) {
        GTEST_SKIP() << "Insufficient free GPU memory. Need ~"
                     << (bytes_needed / (1024.0 * 1024.0 * 1024.0)) << " GiB, have ~"
                     << (free_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB free.";
    }

    spdlog::info("SingleCardQrTest N={} NB={}: need ~{:.2f} GiB (free {:.2f} / total {:.2f} GiB)",
                 n, nb, bytes_needed / (1024.0 * 1024.0 * 1024.0),
                 free_bytes / (1024.0 * 1024.0 * 1024.0), total_bytes / (1024.0 * 1024.0 * 1024.0));

    ASSERT_TRUE(d_A0.Ensure(elems_A));
    ASSERT_TRUE(d_Afact.Ensure(elems_A));
    ASSERT_TRUE(d_W.Ensure(elems_A));
    ASSERT_TRUE(d_Y.Ensure(elems_A));
    ASSERT_TRUE(d_rtmp.Ensure(elems_rtmp));
    ASSERT_TRUE(d_tsqr.Ensure(elems_tsqr));
    ASSERT_TRUE(d_pack.Ensure(elems_rtmp));
    ASSERT_TRUE(d_X.Ensure(elems_X));
    ASSERT_TRUE(d_Xorig.Ensure(elems_X));

    // A0 + Afact share the same initial random matrix.
    rng.Fill(d_A0.ptr, elems_A, 1234ULL);
    AssertCuda(cudaMemcpy(d_Afact.ptr, d_A0.ptr, elems_A * sizeof(float), cudaMemcpyDeviceToDevice),
               "cudaMemcpy D2D A0->Afact");

    // W/Y need to be clean because we index into the (m x n) buffers at (outer, outer).
    AssertCuda(cudaMemset(d_W.ptr, 0, elems_A * sizeof(float)), "cudaMemset d_W");
    AssertCuda(cudaMemset(d_Y.ptr, 0, elems_A * sizeof(float)), "cudaMemset d_Y");

    cublasHandle_t handle = nullptr;
    AssertCublas(cublasCreate(&handle), "cublasCreate");

    // Factorize Afact to get WY (and R in Afact's upper triangle).
    SingleCardBlockedQrFactorize(d_Afact.ptr, m, n, nb, handle, d_W.ptr, d_Y.ptr, d_rtmp.ptr,
                                 d_tsqr.ptr, elems_tsqr);
    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after factorize");
    AssertCuda(cudaGetLastError(), "cudaGetLastError after factorize");

    const double norm_A0 = Nrm2Large(handle, d_A0.ptr, elems_A);
    ASSERT_GT(norm_A0, 0.0);

    // Apply Q^T (from WY) to A0: A0 <- Q^T * A0.
    ApplyAllOuterBlocksQT(d_A0.ptr, m, n, nb, handle, d_W.ptr, d_Y.ptr, d_rtmp.ptr);
    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after apply Q^T");
    AssertCuda(cudaGetLastError(), "cudaGetLastError after apply Q^T");

    // Orthogonality check (Q^T Q ≈ I):
    // Use a small random matrix X (m x k) and verify Q^T(QX) ≈ X and ||QX||≈||X||.
    rng.Fill(d_Xorig.ptr, elems_X, 5678ULL);
    AssertCuda(cudaMemcpy(d_X.ptr, d_Xorig.ptr, elems_X * sizeof(float), cudaMemcpyDeviceToDevice),
               "cudaMemcpy D2D Xorig->X");
    const double norm_X = Nrm2Large(handle, d_X.ptr, elems_X);
    ASSERT_GT(norm_X, 0.0);

    ApplyAllOuterBlocksQToMatrix(d_X.ptr, m, kOrthoCols, nb, handle, d_W.ptr, d_Y.ptr, d_rtmp.ptr);
    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after apply Q");
    AssertCuda(cudaGetLastError(), "cudaGetLastError after apply Q");
    const double norm_QX = Nrm2Large(handle, d_X.ptr, elems_X);

    ApplyAllOuterBlocksQTToMatrix(d_X.ptr, m, kOrthoCols, nb, handle, d_W.ptr, d_Y.ptr, d_rtmp.ptr);
    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after apply Q^T to X");
    AssertCuda(cudaGetLastError(), "cudaGetLastError after apply Q^T to X");

    {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned int>((elems_X + block.x - 1) / block.x));
        SubVectorKernel<<<grid, block>>>(elems_X, d_X.ptr, d_Xorig.ptr);
        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize SubVectorKernel");
        AssertCuda(cudaGetLastError(), "cudaGetLastError SubVectorKernel");
    }
    const double norm_QtQX_minus_X = Nrm2Large(handle, d_X.ptr, elems_X);
    const double rel_norm_preserve = std::abs(norm_QX - norm_X) / norm_X;
    const double rel_ortho = norm_QtQX_minus_X / norm_X;
    spdlog::info(
        "N={} NB={} Q-orth: ||X||2={:.6e}, ||QX||2={:.6e} (rel {:.3e}), ||Q^TQX-X||2={:.6e} (rel "
        "{:.3e})",
        n, nb, norm_X, norm_QX, rel_norm_preserve, norm_QtQX_minus_X, rel_ortho);
    EXPECT_LT(rel_norm_preserve, 1e-4);
    EXPECT_LT(rel_ortho, 1e-4);

    // Norm checks via cublasNrm2 on packed nb*nb blocks.
    const dim3 block_dim(16, 16);
    const dim3 grid_dim((nb + block_dim.x - 1) / block_dim.x, (nb + block_dim.y - 1) / block_dim.y);

    double sumsq_lower = 0.0;
    double sumsq_upper_diff = 0.0;
    for (int j0 = 0; j0 < n; j0 += nb) {
        for (int i0 = 0; i0 < n; i0 += nb) {
            const float* a_block = d_A0.ptr + Idx2D(i0, j0, m);
            const float* r_block = d_Afact.ptr + Idx2D(i0, j0, m);

            if (i0 > j0) {
                PackFullBlockKernel<<<grid_dim, block_dim>>>(nb, a_block, m, d_pack.ptr, nb);
                float bnrm = 0.0f;
                AssertCublas(cublasSnrm2(handle, nb * nb, d_pack.ptr, 1, &bnrm),
                             "cublasSnrm2 lower block");
                sumsq_lower += static_cast<double>(bnrm) * static_cast<double>(bnrm);
            } else if (i0 == j0) {
                PackStrictLowerBlockKernel<<<grid_dim, block_dim>>>(nb, a_block, m, d_pack.ptr, nb);
                float bnrm = 0.0f;
                AssertCublas(cublasSnrm2(handle, nb * nb, d_pack.ptr, 1, &bnrm),
                             "cublasSnrm2 lower diag");
                sumsq_lower += static_cast<double>(bnrm) * static_cast<double>(bnrm);

                PackUpperBlockDiffKernel<<<grid_dim, block_dim>>>(nb, a_block, m, r_block, m,
                                                                  d_pack.ptr, nb);
                float unrm = 0.0f;
                AssertCublas(cublasSnrm2(handle, nb * nb, d_pack.ptr, 1, &unrm),
                             "cublasSnrm2 upper diff diag");
                sumsq_upper_diff += static_cast<double>(unrm) * static_cast<double>(unrm);
            } else {  // i0 < j0
                PackFullBlockDiffKernel<<<grid_dim, block_dim>>>(nb, a_block, m, r_block, m,
                                                                 d_pack.ptr, nb);
                float unrm = 0.0f;
                AssertCublas(cublasSnrm2(handle, nb * nb, d_pack.ptr, 1, &unrm),
                             "cublasSnrm2 upper diff block");
                sumsq_upper_diff += static_cast<double>(unrm) * static_cast<double>(unrm);
            }
        }
    }
    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize pack/nrm2");
    AssertCuda(cudaGetLastError(), "cudaGetLastError pack/nrm2");

    const double norm_lower = std::sqrt(sumsq_lower);
    const double norm_upper_diff = std::sqrt(sumsq_upper_diff);

    const double rel_lower = norm_lower / norm_A0;
    const double rel_upper_diff = norm_upper_diff / norm_A0;

    spdlog::info(
        "N={} NB={} ||A0||2={:.6e}, ||lower||2={:.6e} (rel {:.3e}), ||upper-diff||2={:.6e} (rel "
        "{:.3e})",
        n, nb, norm_A0, norm_lower, rel_lower, norm_upper_diff, rel_upper_diff);

    EXPECT_LT(rel_lower, 2e-2);
    EXPECT_LT(rel_upper_diff, 2e-2);

    AssertCublas(cublasDestroy(handle), "cublasDestroy");
}

static ReusableDeviceBufferF g_A0;
static ReusableDeviceBufferF g_Afact;
static ReusableDeviceBufferF g_W;
static ReusableDeviceBufferF g_Y;
static ReusableDeviceBufferF g_rtmp;
static ReusableDeviceBufferF g_tsqr;
static ReusableDeviceBufferF g_pack;
static ReusableDeviceBufferF g_X;
static ReusableDeviceBufferF g_Xorig;
static ReusableNormalRng g_rng;

TEST(SingleCardQrTest, Norms16384_NB1024) {
    RunQrCase({16384, 1024}, g_A0, g_Afact, g_W, g_Y, g_rtmp, g_tsqr, g_pack, g_X, g_Xorig, g_rng);
}

TEST(SingleCardQrTest, Norms49152_NB1024) {
    RunQrCase({49152, 1024}, g_A0, g_Afact, g_W, g_Y, g_rtmp, g_tsqr, g_pack, g_X, g_Xorig, g_rng);
}

TEST(SingleCardQrTest, Norms65536_NB1024) {
    RunQrCase({65536, 1024}, g_A0, g_Afact, g_W, g_Y, g_rtmp, g_tsqr, g_pack, g_X, g_Xorig, g_rng);
}

}  // namespace
