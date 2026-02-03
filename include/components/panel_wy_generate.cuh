#include <cuda_runtime.h>
#include <stdexcept>

constexpr int tsqr_n32_n = 32;
constexpr int warp_size = 32;

__global__ void generate_wy_float_kernel(
    int m, int n, float* A, int lda, float* Y, int ldy, float* W, int ldw) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row_block = blockIdx.x;
    constexpr int columns_per_warp = 4;
    constexpr int row_tile = tsqr_n32_n * 4;
    const int block_col_base = 0;
    const int warp_col_base = warp_id * columns_per_warp;
    const int row_tile_base = row_block * row_tile;

    // process I - Q && LU
    __shared__ float lu[tsqr_n32_n * tsqr_n32_n];

#pragma unroll
    for (int col_offset = 0; col_offset < columns_per_warp; ++col_offset) {
        int col = warp_col_base + col_offset;
        int global_col = block_col_base + col;
        if (col < tsqr_n32_n && global_col < n) {
            float a_val = A[lane_id + global_col * lda];
            lu[lane_id + col * tsqr_n32_n] = (lane_id == col) ? 1.f - a_val : -a_val;
        }
    }

    for (auto i = 0; i < tsqr_n32_n - 1; i++) {
        const int global_i = block_col_base + i;
        const bool pivot_valid = global_i < n;
        __syncthreads();
        if (lane_id > i && pivot_valid) {
            int local_offset = i - warp_col_base;
            if (local_offset >= 0 && local_offset < columns_per_warp) {
                lu[lane_id + i * tsqr_n32_n] /= lu[i + i * tsqr_n32_n];
            }
        }
        __syncthreads();
        if (lane_id > i && pivot_valid) {
#pragma unroll
            for (int col_offset = 0; col_offset < columns_per_warp; ++col_offset) {
                int col = warp_col_base + col_offset;
                int global_col = block_col_base + col;
                if (col > i && col < tsqr_n32_n && global_col < n) {
                    lu[lane_id + col * tsqr_n32_n] -=
                        lu[lane_id + i * tsqr_n32_n] * lu[i + col * tsqr_n32_n];
                }
            }
        }
    }

    __syncthreads();

    const int rows_below = m - n;
    if (row_block == 0 && lane_id < n) {
#pragma unroll
        for (int col_offset = 0; col_offset < columns_per_warp; ++col_offset) {
            int col = warp_col_base + col_offset;
            int global_col = block_col_base + col;
            if (col < tsqr_n32_n && global_col < n) {
                float val = lu[lane_id + col * tsqr_n32_n];
                if (lane_id > col) {
                    Y[lane_id + global_col * ldy] = val;
                } else if (lane_id == col) {
                    Y[lane_id + global_col * ldy] = 1.f;
                } else {
                    Y[lane_id + global_col * ldy] = 0.f;
                }
            }
        }
    }

    const int row_in_block = warp_id * warp_size + lane_id;
    const bool row_active = row_in_block < row_tile;
    const int global_row = row_tile_base + row_in_block;
    if (row_active && global_row < rows_below) {
        const int actual_row = global_row + n;
        float local[tsqr_n32_n];
#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                local[c] = -A[actual_row + c * lda];
            } else {
                local[c] = 0.f;
            }
        }

        for (int c = 0; c < n; ++c) {
            float diag = lu[c * tsqr_n32_n + c];
            local[c] /= diag;
#pragma unroll
            for (int k = c + 1; k < tsqr_n32_n; ++k) {
                if (k < n) {
                    local[k] -= local[c] * lu[k * tsqr_n32_n + c];
                }
            }
        }

#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                Y[actual_row + c * ldy] = local[c];
            }
        }
    }

    if (row_block == 0 && warp_id == 0 && lane_id < n) {
        const int row = lane_id;
        float local[tsqr_n32_n];
#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                float a_val = A[row + c * lda];
                float val = -a_val;
                if (c == row) {
                    val += 1.f;
                }
                local[c] = val;
            } else {
                local[c] = 0.f;
            }
        }

        for (int c = 0; c < n; ++c) {
#pragma unroll
            for (int k = c + 1; k < tsqr_n32_n; ++k) {
                if (k < n) {
                    local[k] -= local[c] * lu[k + c * tsqr_n32_n];
                }
            }
        }

#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                W[row + c * ldw] = local[c];
            }
        }
    }

    if (row_active && global_row < rows_below) {
        const int row = global_row + n;
        float local[tsqr_n32_n];
#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                float a_val = A[row + c * lda];
                local[c] = -a_val;
            } else {
                local[c] = 0.f;
            }
        }

        for (int c = 0; c < n; ++c) {
#pragma unroll
            for (int k = c + 1; k < tsqr_n32_n; ++k) {
                if (k < n) {
                    local[k] -= local[c] * lu[k + c * tsqr_n32_n];
                }
            }
        }

#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                W[row + c * ldw] = local[c];
            }
        }
    }
}

__global__ void generate_wy_double_kernel(
    int m, int n, double* A, int lda, double* Y, int ldy, double* W, int ldw) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row_block = blockIdx.x;
    constexpr int columns_per_warp = 4;
    constexpr int row_tile = tsqr_n32_n * 4;
    const int block_col_base = 0;
    const int warp_col_base = warp_id * columns_per_warp;
    const int row_tile_base = row_block * row_tile;

    // process I - Q && LU
    __shared__ double lu[tsqr_n32_n * tsqr_n32_n];

#pragma unroll
    for (int col_offset = 0; col_offset < columns_per_warp; ++col_offset) {
        int col = warp_col_base + col_offset;
        int global_col = block_col_base + col;
        if (col < tsqr_n32_n && global_col < n) {
            double a_val = A[lane_id + global_col * lda];
            lu[lane_id + col * tsqr_n32_n] = (lane_id == col) ? 1.0 - a_val : -a_val;
        }
    }

    for (auto i = 0; i < tsqr_n32_n - 1; i++) {
        const int global_i = block_col_base + i;
        const bool pivot_valid = global_i < n;
        __syncthreads();
        if (lane_id > i && pivot_valid) {
            int local_offset = i - warp_col_base;
            if (local_offset >= 0 && local_offset < columns_per_warp) {
                lu[lane_id + i * tsqr_n32_n] /= lu[i + i * tsqr_n32_n];
            }
        }
        __syncthreads();
        if (lane_id > i && pivot_valid) {
#pragma unroll
            for (int col_offset = 0; col_offset < columns_per_warp; ++col_offset) {
                int col = warp_col_base + col_offset;
                int global_col = block_col_base + col;
                if (col > i && col < tsqr_n32_n && global_col < n) {
                    lu[lane_id + col * tsqr_n32_n] -=
                        lu[lane_id + i * tsqr_n32_n] * lu[i + col * tsqr_n32_n];
                }
            }
        }
    }

    __syncthreads();

    const int rows_below = m - n;
    if (row_block == 0 && lane_id < n) {
#pragma unroll
        for (int col_offset = 0; col_offset < columns_per_warp; ++col_offset) {
            int col = warp_col_base + col_offset;
            int global_col = block_col_base + col;
            if (col < tsqr_n32_n && global_col < n) {
                double val = lu[lane_id + col * tsqr_n32_n];
                if (lane_id > col) {
                    Y[lane_id + global_col * ldy] = val;
                } else if (lane_id == col) {
                    Y[lane_id + global_col * ldy] = 1.0;
                } else {
                    Y[lane_id + global_col * ldy] = 0.0;
                }
            }
        }
    }

    const int row_in_block = warp_id * warp_size + lane_id;
    const bool row_active = row_in_block < row_tile;
    const int global_row = row_tile_base + row_in_block;
    if (row_active && global_row < rows_below) {
        const int actual_row = global_row + n;
        double local[tsqr_n32_n];
#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                local[c] = -A[actual_row + c * lda];
            } else {
                local[c] = 0.0;
            }
        }

        for (int c = 0; c < n; ++c) {
            double diag = lu[c * tsqr_n32_n + c];
            local[c] /= diag;
#pragma unroll
            for (int k = c + 1; k < tsqr_n32_n; ++k) {
                if (k < n) {
                    local[k] -= local[c] * lu[k * tsqr_n32_n + c];
                }
            }
        }

#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                Y[actual_row + c * ldy] = local[c];
            }
        }
    }

    if (row_block == 0 && warp_id == 0 && lane_id < n) {
        const int row = lane_id;
        double local[tsqr_n32_n];
#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                double a_val = A[row + c * lda];
                double val = -a_val;
                if (c == row) {
                    val += 1.0;
                }
                local[c] = val;
            } else {
                local[c] = 0.0;
            }
        }

        for (int c = 0; c < n; ++c) {
#pragma unroll
            for (int k = c + 1; k < tsqr_n32_n; ++k) {
                if (k < n) {
                    local[k] -= local[c] * lu[k + c * tsqr_n32_n];
                }
            }
        }

#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                W[row + c * ldw] = local[c];
            }
        }
    }

    if (row_active && global_row < rows_below) {
        const int row = global_row + n;
        double local[tsqr_n32_n];
#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                double a_val = A[row + c * lda];
                local[c] = -a_val;
            } else {
                local[c] = 0.0;
            }
        }

        for (int c = 0; c < n; ++c) {
#pragma unroll
            for (int k = c + 1; k < tsqr_n32_n; ++k) {
                if (k < n) {
                    local[k] -= local[c] * lu[k + c * tsqr_n32_n];
                }
            }
        }

#pragma unroll
        for (int c = 0; c < tsqr_n32_n; ++c) {
            if (c < n) {
                W[row + c * ldw] = local[c];
            }
        }
    }
}

template <typename T>
void generate_wy(int m, int n, T* A, int lda, T* Y, int ldy, T* W, int ldw, cudaStream_t stream) {
    constexpr int columns_per_warp = 4;
    constexpr int warps_per_block = tsqr_n32_n / columns_per_warp;
    constexpr int row_tile = tsqr_n32_n * 4;
    if (m <= 0 || n <= 0 || m < n || n > tsqr_n32_n) {
        return;
    }
    dim3 block_dim(warp_size, warps_per_block);
    int rows_below = m - n;
    int row_blocks = (rows_below + row_tile - 1) / row_tile;
    if (row_blocks < 1) {
        row_blocks = 1;
    }
    dim3 grid_dim(row_blocks);
    if constexpr (std::is_same_v<T, float>) {
        generate_wy_float_kernel<<<grid_dim, block_dim, 0, stream>>>(m, n, A, lda, Y, ldy, W, ldw);
    } else if constexpr (std::is_same_v<T, double>) {
        generate_wy_double_kernel<<<grid_dim, block_dim, 0, stream>>>(m, n, A, lda, Y, ldy, W, ldw);
    } else {
        throw std::runtime_error("Unsupported type: unknown");
    }
    cudaStreamSynchronize(stream);
}
