#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <thrust/device_ptr.h>
#include <vector>

class MatrixHelper {
public:
    template <typename T>
    static void printMatrixToFile(T* A,
                                  size_t lda,
                                  size_t m,
                                  size_t n,
                                  int precision = 4,
                                  const std::string& filename = "log/matrix_output.txt") {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            spdlog::error("Failed to open file for writing: {}", filename);
            return;
        }

        ofs << std::fixed << std::setprecision(precision);
        ofs << "Matrix dimensions: " << m << "x" << n << std::endl;

        for (auto i = 0; i < m; i++) {
            for (auto j = 0; j < n; j++) {
                ofs << A[i * lda + j] << " ";
            }
            ofs << std::endl;
        }

        ofs.close();
        spdlog::info("Matrix has been written to file: {}", filename);
    }

    template <typename T>
    static void printMatrixToFile(thrust::device_ptr<T> A,
                                  size_t lda,
                                  size_t m,
                                  size_t n,
                                  int precision = 4,
                                  const std::string& filename = "log/matrix_output.txt") {
        std::vector<T> host(m * n);
        const T* device_ptr = thrust::raw_pointer_cast(A);
        cudaError_t err = cudaMemcpy2D(host.data(), n * sizeof(T), device_ptr, lda * sizeof(T),
                                       n * sizeof(T), m, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            spdlog::error("cudaMemcpy2D failed: {}", cudaGetErrorString(err));
            return;
        }

        MatrixHelper::printMatrixToFile(host.data(), n, m, n, precision, filename);
    }
};
