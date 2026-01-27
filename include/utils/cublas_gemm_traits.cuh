#pragma once

#include <cublas_v2.h>

template <typename T>
struct CublasGemmTraits;

template <>
struct CublasGemmTraits<float> {
    static cublasStatus_t GemmStridedBatched(cublasHandle_t handle,
                                             cublasOperation_t transa,
                                             cublasOperation_t transb, int m,
                                             int n, int k, const float* alpha,
                                             const float* A, int lda,
                                             long long int strideA,
                                             const float* B, int ldb,
                                             long long int strideB,
                                             const float* beta, float* C,
                                             int ldc,
                                             long long int strideC,
                                             int batchCount) {
        return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha,
                                         A, lda, strideA, B, ldb, strideB, beta,
                                         C, ldc, strideC, batchCount);
    }

    static cublasStatus_t Gemm(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int m, int n, int k,
                               const float* alpha, const float* A, int lda,
                               const float* B, int ldb, const float* beta,
                               float* C, int ldc) {
        return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B,
                           ldb, beta, C, ldc);
    }
};

template <>
struct CublasGemmTraits<double> {
    static cublasStatus_t GemmStridedBatched(cublasHandle_t handle,
                                             cublasOperation_t transa,
                                             cublasOperation_t transb, int m,
                                             int n, int k, const double* alpha,
                                             const double* A, int lda,
                                             long long int strideA,
                                             const double* B, int ldb,
                                             long long int strideB,
                                             const double* beta, double* C,
                                             int ldc,
                                             long long int strideC,
                                             int batchCount) {
        return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha,
                                         A, lda, strideA, B, ldb, strideB, beta,
                                         C, ldc, strideC, batchCount);
    }

    static cublasStatus_t Gemm(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int m, int n, int k,
                               const double* alpha, const double* A, int lda,
                               const double* B, int ldb, const double* beta,
                               double* C, int ldc) {
        return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B,
                           ldb, beta, C, ldc);
    }
};
