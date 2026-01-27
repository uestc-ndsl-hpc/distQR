#include <cuda_runtime.h>
#include <curand.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "utils/matrix_helper.cuh"

namespace {

void AssertCuda(cudaError_t status, const char* context) {
    ASSERT_EQ(status, cudaSuccess) << context << ": " << cudaGetErrorString(status);
}

void AssertCurand(curandStatus_t status, const char* context) {
    ASSERT_EQ(status, CURAND_STATUS_SUCCESS) << context;
}

}  // namespace

TEST(MatrixHelperTest, PrintMatrixToFileFromDevice) {
    const size_t m = 3;
    const size_t n = 4;
    const size_t count = m * n;

    float* device_matrix = nullptr;
    AssertCuda(cudaMalloc(&device_matrix, count * sizeof(float)), "cudaMalloc");

    curandGenerator_t gen;
    AssertCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
    AssertCurand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL),
                 "curandSetPseudoRandomGeneratorSeed");
    AssertCurand(curandGenerateUniform(gen, device_matrix, count), "curandGenerateUniform");

    std::vector<float> host_matrix(count);
    AssertCuda(cudaMemcpy(host_matrix.data(), device_matrix, count * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy");

    const std::string filename = "log/test_matrix_output.txt";
    MatrixHelper::printMatrixToFile(thrust::device_pointer_cast(device_matrix), n, m, n, 4,
                                    filename);

    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.is_open());

    std::string line;
    ASSERT_TRUE(std::getline(ifs, line));
    EXPECT_EQ(line, "Matrix dimensions: 3x4");

    for (size_t i = 0; i < m; ++i) {
        ASSERT_TRUE(std::getline(ifs, line));
        std::ostringstream expected;
        expected << std::fixed << std::setprecision(4);
        for (size_t j = 0; j < n; ++j) {
            expected << host_matrix[i * n + j] << " ";
        }
        EXPECT_EQ(line, expected.str());
    }

    std::string extra;
    EXPECT_FALSE(std::getline(ifs, extra));

    ifs.close();
    // std::remove(filename.c_str());

    AssertCurand(curandDestroyGenerator(gen), "curandDestroyGenerator");
    AssertCuda(cudaFree(device_matrix), "cudaFree");
}
