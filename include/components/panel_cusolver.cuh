#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>

template <typename T>
struct CusolverPanelQrTraits;

template <>
struct CusolverPanelQrTraits<float> {
    static cusolverStatus_t GeqrfBufferSize(cusolverDnHandle_t handle,
                                            int m,
                                            int n,
                                            float* A,
                                            int lda,
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

    static cusolverStatus_t OrgqrBufferSize(cusolverDnHandle_t handle,
                                            int m,
                                            int n,
                                            int k,
                                            float* A,
                                            int lda,
                                            float* tau,
                                            int* lwork) {
        return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    }

    static cusolverStatus_t Orgqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  float* A,
                                  int lda,
                                  float* tau,
                                  float* work,
                                  int lwork,
                                  int* info) {
        return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    }
};

template <>
struct CusolverPanelQrTraits<double> {
    static cusolverStatus_t GeqrfBufferSize(cusolverDnHandle_t handle,
                                            int m,
                                            int n,
                                            double* A,
                                            int lda,
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

    static cusolverStatus_t OrgqrBufferSize(cusolverDnHandle_t handle,
                                            int m,
                                            int n,
                                            int k,
                                            double* A,
                                            int lda,
                                            double* tau,
                                            int* lwork) {
        return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    }

    static cusolverStatus_t Orgqr(cusolverDnHandle_t handle,
                                  int m,
                                  int n,
                                  int k,
                                  double* A,
                                  int lda,
                                  double* tau,
                                  double* work,
                                  int lwork,
                                  int* info) {
        return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    }
};

template <typename T>
__global__ void pack_upper_triangle_kernel(int m, int n, const T* A, int lda, T* R, int ldr) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m && col < n) {
        R[row + col * ldr] = (row <= col) ? A[row + col * lda] : static_cast<T>(0);
    }
}

template <typename T>
void pack_upper_triangle(int m, int n, const T* A, int lda, T* R, int ldr, cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    pack_upper_triangle_kernel<T><<<grid, block, 0, stream>>>(m, n, A, lda, R, ldr);
}
