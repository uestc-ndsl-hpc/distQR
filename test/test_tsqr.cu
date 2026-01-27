#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <vector>

#include "components/panel_process.cuh"

namespace {

void AssertCuda(cudaError_t status, const char* context) {
    ASSERT_EQ(status, cudaSuccess) << context << ": " << cudaGetErrorString(status);
}

void AssertCublas(cublasStatus_t status, const char* context) {
    ASSERT_EQ(status, CUBLAS_STATUS_SUCCESS) << context;
}

void AssertCurand(curandStatus_t status, const char* context) {
    ASSERT_EQ(status, CURAND_STATUS_SUCCESS) << context;
}

template <typename T>
void FillDeviceRandom(T* device_data, size_t count, unsigned long long seed) {
    curandGenerator_t gen;
    AssertCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT),
                 "curandCreateGenerator");
    AssertCurand(curandSetPseudoRandomGeneratorSeed(gen, seed),
                 "curandSetPseudoRandomGeneratorSeed");
    if constexpr (std::is_same_v<T, float>) {
        AssertCurand(curandGenerateUniform(gen, device_data,
                                           static_cast<int>(count)),
                     "curandGenerateUniform");
    } else if constexpr (std::is_same_v<T, double>) {
        AssertCurand(curandGenerateUniformDouble(gen, device_data,
                                                 static_cast<int>(count)),
                     "curandGenerateUniformDouble");
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Only float/double supported.");
    }
    AssertCurand(curandDestroyGenerator(gen), "curandDestroyGenerator");
}

template <typename T>
std::vector<T> MultiplyQR(int m, int n, int lda, const std::vector<T>& Q,
                          int ldr, const std::vector<T>& R) {
    std::vector<T> result(static_cast<size_t>(m) * n);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k) {
                acc += static_cast<double>(Q[row + k * lda]) *
                       static_cast<double>(R[k + col * ldr]);
            }
            result[row + col * m] = static_cast<T>(acc);
        }
    }
    return result;
}

template <typename T>
double RelativeFrobeniusError(const std::vector<T>& A,
                              const std::vector<T>& B) {
    double diff = 0.0;
    double base = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        double a = static_cast<double>(A[i]);
        double b = static_cast<double>(B[i]);
        double d = a - b;
        diff += d * d;
        base += a * a;
    }
    const double denom = (base > 0.0) ? std::sqrt(base) : 1.0;
    return std::sqrt(diff) / denom;
}

template <typename T>
double MaxOrthoError(int m, int n, int lda, const std::vector<T>& Q) {
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int k = 0; k < m; ++k) {
                acc += static_cast<double>(Q[k + i * lda]) *
                       static_cast<double>(Q[k + j * lda]);
            }
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = std::abs(acc - expected);
            if (diff > max_abs) {
                max_abs = diff;
            }
        }
    }
    return max_abs;
}

template <typename T>
double MaxLowerTriError(int n, int ldr, const std::vector<T>& R) {
    double max_abs = 0.0;
    for (int col = 0; col < n; ++col) {
        for (int row = col + 1; row < n; ++row) {
            double v = std::abs(static_cast<double>(R[row + col * ldr]));
            if (v > max_abs) {
                max_abs = v;
            }
        }
    }
    return max_abs;
}

template <typename T>
void RunTsqrTest(int m, double tol_recon, double tol_ortho,
                 double tol_tri) {
    constexpr int n = 32;
    const int lda = m;
    const int ldr = n;

    std::vector<T> h_A0(static_cast<size_t>(lda) * n);

    T* d_A = nullptr;
    T* d_R = nullptr;
    T* d_work = nullptr;

    const size_t a_bytes = static_cast<size_t>(lda) * n * sizeof(T);
    const size_t r_bytes = static_cast<size_t>(ldr) * n * sizeof(T);

    AssertCuda(cudaMalloc(&d_A, a_bytes), "cudaMalloc d_A");
    AssertCuda(cudaMalloc(&d_R, r_bytes), "cudaMalloc d_R");

    FillDeviceRandom(d_A, static_cast<size_t>(lda) * n, 123ULL);
    AssertCuda(cudaMemcpy(h_A0.data(), d_A, a_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H d_A (baseline)");

    size_t work_elems = tsqr_work_elems<T>(m);
    if (work_elems > 0) {
        AssertCuda(cudaMalloc(&d_work, work_elems * sizeof(T)),
                   "cudaMalloc d_work");
    }

    cublasHandle_t handle;
    AssertCublas(cublasCreate(&handle), "cublasCreate");

    tsqr(handle, m, d_A, lda, d_R, ldr, d_work, work_elems);

    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    AssertCuda(cudaGetLastError(), "cudaGetLastError");

    std::vector<T> h_Q(static_cast<size_t>(m) * n);
    std::vector<T> h_R(static_cast<size_t>(n) * n);

    AssertCuda(cudaMemcpy(h_Q.data(), d_A, a_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H d_A");
    AssertCuda(cudaMemcpy(h_R.data(), d_R, r_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H d_R");

    std::vector<T> h_recon = MultiplyQR(m, n, lda, h_Q, ldr, h_R);

    const double recon_err = RelativeFrobeniusError(h_A0, h_recon);
    const double ortho_err = MaxOrthoError(m, n, lda, h_Q);
    const double tri_err = MaxLowerTriError(n, ldr, h_R);

    EXPECT_LT(recon_err, tol_recon);
    EXPECT_LT(ortho_err, tol_ortho);
    EXPECT_LT(tri_err, tol_tri);

    AssertCublas(cublasDestroy(handle), "cublasDestroy");
    AssertCuda(cudaFree(d_A), "cudaFree d_A");
    AssertCuda(cudaFree(d_R), "cudaFree d_R");
    if (d_work) {
        AssertCuda(cudaFree(d_work), "cudaFree d_work");
    }
}

}  // namespace

TEST(TsqrTest, FloatSmallMatrix) {
    RunTsqrTest<float>(128, 5e-3, 5e-3, 5e-3);
}

TEST(TsqrTest, FloatLargeMatrix) {
    RunTsqrTest<float>(32768, 8e-7, 6e-7, 6e-3);
}

TEST(TsqrTest, DoubleSmallMatrix) {
    RunTsqrTest<double>(128, 1e-14, 1e-14, 1e-14);
}

TEST(TsqrTest, DoubleLargeMatrix) {
    RunTsqrTest<double>(32768, 5e-14, 5e-14, 5e-14);
}
