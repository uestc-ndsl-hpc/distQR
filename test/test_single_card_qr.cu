#include <thrust/device_vector.h>
#include <spdlog/spdlog.h>
#include <curand.h>

#include "../include/components/panel_process.cuh"

int main() {
    const size_t m = 16384;
    const size_t n = 16384;
    const auto lda = m;
    const size_t count = m * n;

    thrust::device_vector<float> device_matrix(count);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    auto device_matrix_raw_pointer = thrust::raw_pointer_cast(device_matrix.data());
    curandGenerateNormal(gen, device_matrix_raw_pointer, count, 0.0f, 1.0f);

    size_t nb = 1024;
    size_t b = 32;

    thrust::device_vector<float> device_work(lda * n * 2 + b * b + tsqr_work_elems<float>(m));

    auto w_raw_pointer = thrust::raw_pointer_cast(device_work.data());
    auto y_raw_pointer = w_raw_pointer + lda * n;
    auto r_raw_pointer = y_raw_pointer + lda * n;
    auto tsqr_work_space_raw_pointer = r_raw_pointer + b * b;

    spdlog::info("Perform QR factorization m:{}, n:{}, nb:{}, b:{}", m, n, nb, b);

    cublasHandle_t cublas_handle = nullptr;
    cublasCreate(&cublas_handle);

    // outer loop for large update
    for (auto outer_index = 0; outer_index < n; outer_index += nb) {
        // inner loop for inner update
        for (auto inner_index = outer_index; inner_index < nb; inner_index += b) {
            // tsqr
            auto panel_height = m - inner_index;
            auto panel_raw_pointer = device_matrix_raw_pointer + inner_index * lda + inner_index;
            auto panel_w_raw_pointer = w_raw_pointer + inner_index * lda + inner_index;
            auto panel_y_raw_pointer = y_raw_pointer + inner_index * lda + inner_index;
            tsqr<float>(cublas_handle, panel_height, panel_raw_pointer, lda, r_raw_pointer, b,
                        tsqr_work_space_raw_pointer, tsqr_work_elems<float>(panel_height), nullptr);
            // generate WY for next panel
            generate_wy(panel_height, b, panel_raw_pointer, lda, panel_y_raw_pointer, lda,
                        panel_w_raw_pointer, lda, nullptr);
            // write back R to A
            write_back_R2A(b, b, r_raw_pointer, b, panel_raw_pointer, lda, nullptr);

            // update next panel
            if (inner_index + b >= nb) break;
            auto next_panel_raw_pointer = panel_raw_pointer + b * lda;
            auto work_t_raw_pointer = r_raw_pointer;
            float one = 1.0f;
            float zero = 0.0f;
            float minus_one = -1.0f;
            CublasGemmTraits<float>::Gemm(
                cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, b, b, panel_height, &one,
                panel_y_raw_pointer, lda, next_panel_raw_pointer, lda, &zero, work_t_raw_pointer,
                b);
            CublasGemmTraits<float>::Gemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, panel_height, b, b, &minus_one,
                panel_w_raw_pointer, lda, work_t_raw_pointer, b, &one, next_panel_raw_pointer,
                lda);
        }
    }
}
