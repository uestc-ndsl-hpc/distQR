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
std::vector<T> ReconstructQFromWY(int m,
                                  int n,
                                  int ldw,
                                  const std::vector<T>& W,
                                  int ldy,
                                  const std::vector<T>& Y) {
    std::vector<T> result(static_cast<size_t>(m) * n);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k) {
                acc += static_cast<double>(W[row + k * ldw]) *
                       static_cast<double>(Y[col + k * ldy]);
            }
            double val = (row == col) ? 1.0 : 0.0;
            result[row + col * m] = static_cast<T>(val - acc);
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

void RunTsqrWyTest(int m, double tol) {
    constexpr int n = 32;
    const int lda = m;
    const int ldr = n;
    const int ldy = m;
    const int ldw = m;

    float* d_A = nullptr;
    float* d_R = nullptr;
    float* d_Y = nullptr;
    float* d_W = nullptr;
    float* d_work = nullptr;

    const size_t a_bytes = static_cast<size_t>(lda) * n * sizeof(float);
    const size_t r_bytes = static_cast<size_t>(ldr) * n * sizeof(float);
    const size_t wy_bytes = static_cast<size_t>(m) * n * sizeof(float);

    AssertCuda(cudaMalloc(&d_A, a_bytes), "cudaMalloc d_A");
    AssertCuda(cudaMalloc(&d_R, r_bytes), "cudaMalloc d_R");
    AssertCuda(cudaMalloc(&d_Y, wy_bytes), "cudaMalloc d_Y");
    AssertCuda(cudaMalloc(&d_W, wy_bytes), "cudaMalloc d_W");

    FillDeviceRandom(d_A, static_cast<size_t>(lda) * n, 123ULL);

    size_t work_elems = tsqr_work_elems<float>(m);
    if (work_elems > 0) {
        AssertCuda(cudaMalloc(&d_work, work_elems * sizeof(float)),
                   "cudaMalloc d_work");
    }

    cublasHandle_t handle;
    AssertCublas(cublasCreate(&handle), "cublasCreate");

    tsqr(handle, m, d_A, lda, d_R, ldr, d_work, work_elems);
    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize tsqr");
    AssertCuda(cudaGetLastError(), "cudaGetLastError tsqr");

    generate_wy(m, n, d_A, lda, d_Y, ldy, d_W, ldw);
    AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize generate_wy");
    AssertCuda(cudaGetLastError(), "cudaGetLastError generate_wy");

    std::vector<float> h_Q(static_cast<size_t>(m) * n);
    std::vector<float> h_Y(static_cast<size_t>(m) * n);
    std::vector<float> h_W(static_cast<size_t>(m) * n);

    AssertCuda(cudaMemcpy(h_Q.data(), d_A, a_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H d_A");
    AssertCuda(cudaMemcpy(h_Y.data(), d_Y, wy_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H d_Y");
    AssertCuda(cudaMemcpy(h_W.data(), d_W, wy_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H d_W");

    std::vector<float> h_Q_wy = ReconstructQFromWY(m, n, ldw, h_W, ldy, h_Y);
    const double wy_err = RelativeFrobeniusError(h_Q, h_Q_wy);
    EXPECT_LT(wy_err, tol);

    AssertCublas(cublasDestroy(handle), "cublasDestroy");
    AssertCuda(cudaFree(d_A), "cudaFree d_A");
    AssertCuda(cudaFree(d_R), "cudaFree d_R");
    AssertCuda(cudaFree(d_Y), "cudaFree d_Y");
    AssertCuda(cudaFree(d_W), "cudaFree d_W");
    if (d_work) {
        AssertCuda(cudaFree(d_work), "cudaFree d_work");
    }
}

}  // namespace

TEST(TsqrWyTest, FloatSmallMatrix) {
    RunTsqrWyTest(128, 1e-5);
}

TEST(TsqrWyTest, FloatLargeMatrix) {
    RunTsqrWyTest(32768, 1e-5);
}
