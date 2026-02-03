#include "panel_tsqr.cuh"
#include "panel_wy_generate.cuh"

template <typename T>
__global__ void write_back_R2A_kernel(int m, int n, T* R, int ldr, T* A, int lda) {
    const auto tx = threadIdx.x;
    const auto ty = threadIdx.y;
    const auto bx = blockIdx.x;
    const auto by = blockIdx.y;
    const auto dim = 32;

    auto row_idx = tx + bx * dim;
    auto col_idx = ty + by * dim;
    if (row_idx < m && col_idx < n) {
        if (row_idx <= col_idx)
            A[row_idx + col_idx * lda] = R[row_idx + col_idx * ldr];
        else
            A[row_idx + col_idx * lda] = 0;
    }
}

/**
write back R to A to save memory
*/
template <typename T>
void write_back_R2A(int m, int n, T* R, int ldr, T* A, int lda, cudaStream_t stream) {
    auto grid_dim = dim3((m + 31) / 32, (n + 31) / 32);
    auto block_dim = dim3(32, 32);
    write_back_R2A_kernel<T><<<grid_dim, block_dim, 0, stream>>>(m, n, R, ldr, A, lda);
    cudaStreamSynchronize(stream);
}