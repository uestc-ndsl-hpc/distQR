#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>

#include "components/panel_process.cuh"
#include "utils/cublas_gemm_traits.cuh"

namespace {

void AssertCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", context, cudaGetErrorString(status));
        std::exit(1);
    }
}

void AssertCublas(cublasStatus_t status, const char* context) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cublas error %d\n", context, static_cast<int>(status));
        std::exit(1);
    }
}

void AssertCusolver(cusolverStatus_t status, const char* context) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cusolver error %d\n", context, static_cast<int>(status));
        std::exit(1);
    }
}

void AssertCurand(curandStatus_t status, const char* context) {
    if (status != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: curand error %d\n", context, static_cast<int>(status));
        std::exit(1);
    }
}

template <typename T>
void FillDeviceRandom(T* device_data, size_t count, unsigned long long seed) {
    curandGenerator_t gen;
    AssertCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
    AssertCurand(curandSetPseudoRandomGeneratorSeed(gen, seed),
                 "curandSetPseudoRandomGeneratorSeed");
    if constexpr (std::is_same_v<T, float>) {
        AssertCurand(curandGenerateUniform(gen, device_data, static_cast<int>(count)),
                     "curandGenerateUniform");
    } else if constexpr (std::is_same_v<T, double>) {
        AssertCurand(curandGenerateUniformDouble(gen, device_data, static_cast<int>(count)),
                     "curandGenerateUniformDouble");
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Only float/double supported.");
    }
    AssertCurand(curandDestroyGenerator(gen), "curandDestroyGenerator");
}

template <typename T>
void WarmupGemm(cublasHandle_t handle,
                int m,
                int n,
                int k,
                const T* A,
                int lda,
                const T* B,
                int ldb,
                T* C,
                int ldc,
                int iters) {
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    for (int i = 0; i < iters; ++i) {
        AssertCublas(CublasGemmTraits<T>::Gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A,
                                               lda, B, ldb, &zero, C, ldc),
                     "cublasGemm warmup");
    }
}

float TimeKernelMs(const std::function<void()>& setup, const std::function<void()>& fn, int iters) {
    cudaEvent_t start;
    cudaEvent_t stop;
    AssertCuda(cudaEventCreate(&start), "cudaEventCreate start");
    AssertCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        setup();
        AssertCuda(cudaEventRecord(start), "cudaEventRecord start");
        fn();
        AssertCuda(cudaEventRecord(stop), "cudaEventRecord stop");
        AssertCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
        float ms = 0.0f;
        AssertCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        total_ms += ms;
    }

    AssertCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    AssertCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return total_ms / static_cast<float>(iters);
}

double QrFlops(int m, int n) {
    const double md = static_cast<double>(m);
    const double nd = static_cast<double>(n);
    return 2.0 * md * nd * nd - (2.0 / 3.0) * nd * nd * nd;
}

double FlopsToTflops(double flops, float ms) {
    if (ms <= 0.0f) {
        return 0.0;
    }
    return flops / (static_cast<double>(ms) * 1e-3) / 1e12;
}

struct Options {
    int m = 32768;
    int iters = 20;
    int warmup = 5;
    bool use_double = false;
};

Options ParseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            opts.m = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opts.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opts.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            std::string type = argv[++i];
            opts.use_double = (type == "double" || type == "fp64");
        }
    }
    return opts;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opts = ParseArgs(argc, argv);
    const int m = opts.m;
    const int n = 32;
    const double qr_flops = QrFlops(m, n);

    std::printf("TSQR bench: m=%d n=%d iters=%d warmup=%d type=%s\n", m, n, opts.iters, opts.warmup,
                opts.use_double ? "double" : "float");

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    AssertCublas(cublasCreate(&cublas_handle), "cublasCreate");
    AssertCusolver(cusolverDnCreate(&cusolver_handle), "cusolverDnCreate");

    if (opts.use_double) {
        using T = double;
        const int lda = m;
        const int ldr = n;
        const size_t a_elems = static_cast<size_t>(lda) * n;
        const size_t a_bytes = a_elems * sizeof(T);
        const size_t r_bytes = static_cast<size_t>(ldr) * n * sizeof(T);

        T* d_A0 = nullptr;
        T* d_A_tsqr = nullptr;
        T* d_A_geqrf = nullptr;
        T* d_R = nullptr;
        T* d_work_tsqr = nullptr;
        T* d_tau = nullptr;
        T* d_work_geqrf = nullptr;
        int* d_info = nullptr;

        AssertCuda(cudaMalloc(&d_A0, a_bytes), "cudaMalloc d_A0");
        AssertCuda(cudaMalloc(&d_A_tsqr, a_bytes), "cudaMalloc d_A_tsqr");
        AssertCuda(cudaMalloc(&d_A_geqrf, a_bytes), "cudaMalloc d_A_geqrf");
        AssertCuda(cudaMalloc(&d_R, r_bytes), "cudaMalloc d_R");
        AssertCuda(cudaMalloc(&d_tau, n * sizeof(T)), "cudaMalloc d_tau");
        AssertCuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc d_info");

        FillDeviceRandom(d_A0, a_elems, 1234ULL);

        const size_t work_elems = tsqr_work_elems<T>(m);
        if (work_elems > 0) {
            AssertCuda(cudaMalloc(&d_work_tsqr, work_elems * sizeof(T)), "cudaMalloc d_work_tsqr");
        }

        int lwork = 0;
        AssertCusolver(cusolverDnDgeqrf_bufferSize(cusolver_handle, m, n, d_A_geqrf, lda, &lwork),
                       "cusolverDnDgeqrf_bufferSize");
        if (lwork > 0) {
            AssertCuda(cudaMalloc(&d_work_geqrf, lwork * sizeof(T)), "cudaMalloc d_work_geqrf");
        }

        T* d_B = nullptr;
        T* d_C = nullptr;
        AssertCuda(cudaMalloc(&d_B, static_cast<size_t>(n) * n * sizeof(T)), "cudaMalloc d_B");
        AssertCuda(cudaMalloc(&d_C, static_cast<size_t>(m) * n * sizeof(T)), "cudaMalloc d_C");
        FillDeviceRandom(d_B, static_cast<size_t>(n) * n, 5678ULL);
        WarmupGemm(cublas_handle, m, n, n, d_A0, lda, d_B, n, d_C, m, opts.warmup);
        AssertCuda(cudaFree(d_B), "cudaFree d_B");
        AssertCuda(cudaFree(d_C), "cudaFree d_C");
        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

        const float tsqr_ms = TimeKernelMs(
            [&]() {
                AssertCuda(cudaMemcpy(d_A_tsqr, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpy D2D tsqr");
            },
            [&]() {
                tsqr(cublas_handle, m, d_A_tsqr, lda, d_R, ldr, d_work_tsqr, work_elems);
                AssertCuda(cudaGetLastError(), "tsqr launch");
            },
            opts.iters);

        const float geqrf_ms = TimeKernelMs(
            [&]() {
                AssertCuda(cudaMemcpy(d_A_geqrf, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpy D2D geqrf");
            },
            [&]() {
                AssertCusolver(cusolverDnDgeqrf(cusolver_handle, m, n, d_A_geqrf, lda, d_tau,
                                                d_work_geqrf, lwork, d_info),
                               "cusolverDnDgeqrf");
            },
            opts.iters);

        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize bench");

        const double tsqr_tflops = FlopsToTflops(qr_flops, tsqr_ms);
        const double geqrf_tflops = FlopsToTflops(qr_flops, geqrf_ms);
        std::printf("TSQR avg:   %.3f ms (%.3f TFLOPS)\n", tsqr_ms, tsqr_tflops);
        std::printf("GEQRF avg:  %.3f ms (%.3f TFLOPS)\n", geqrf_ms, geqrf_tflops);

        AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
        AssertCuda(cudaFree(d_A_tsqr), "cudaFree d_A_tsqr");
        AssertCuda(cudaFree(d_A_geqrf), "cudaFree d_A_geqrf");
        AssertCuda(cudaFree(d_R), "cudaFree d_R");
        AssertCuda(cudaFree(d_tau), "cudaFree d_tau");
        AssertCuda(cudaFree(d_work_tsqr), "cudaFree d_work_tsqr");
        if (d_work_geqrf) {
            AssertCuda(cudaFree(d_work_geqrf), "cudaFree d_work_geqrf");
        }
        AssertCuda(cudaFree(d_info), "cudaFree d_info");
    } else {
        using T = float;
        const int lda = m;
        const int ldr = n;
        const size_t a_elems = static_cast<size_t>(lda) * n;
        const size_t a_bytes = a_elems * sizeof(T);
        const size_t r_bytes = static_cast<size_t>(ldr) * n * sizeof(T);

        T* d_A0 = nullptr;
        T* d_A_tsqr = nullptr;
        T* d_A_geqrf = nullptr;
        T* d_R = nullptr;
        T* d_work_tsqr = nullptr;
        T* d_tau = nullptr;
        T* d_work_geqrf = nullptr;
        int* d_info = nullptr;

        AssertCuda(cudaMalloc(&d_A0, a_bytes), "cudaMalloc d_A0");
        AssertCuda(cudaMalloc(&d_A_tsqr, a_bytes), "cudaMalloc d_A_tsqr");
        AssertCuda(cudaMalloc(&d_A_geqrf, a_bytes), "cudaMalloc d_A_geqrf");
        AssertCuda(cudaMalloc(&d_R, r_bytes), "cudaMalloc d_R");
        AssertCuda(cudaMalloc(&d_tau, n * sizeof(T)), "cudaMalloc d_tau");
        AssertCuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc d_info");

        FillDeviceRandom(d_A0, a_elems, 1234ULL);

        const size_t work_elems = tsqr_work_elems<T>(m);
        if (work_elems > 0) {
            AssertCuda(cudaMalloc(&d_work_tsqr, work_elems * sizeof(T)), "cudaMalloc d_work_tsqr");
        }

        int lwork = 0;
        AssertCusolver(cusolverDnSgeqrf_bufferSize(cusolver_handle, m, n, d_A_geqrf, lda, &lwork),
                       "cusolverDnSgeqrf_bufferSize");
        if (lwork > 0) {
            AssertCuda(cudaMalloc(&d_work_geqrf, lwork * sizeof(T)), "cudaMalloc d_work_geqrf");
        }

        T* d_B = nullptr;
        T* d_C = nullptr;
        AssertCuda(cudaMalloc(&d_B, static_cast<size_t>(n) * n * sizeof(T)), "cudaMalloc d_B");
        AssertCuda(cudaMalloc(&d_C, static_cast<size_t>(m) * n * sizeof(T)), "cudaMalloc d_C");
        FillDeviceRandom(d_B, static_cast<size_t>(n) * n, 5678ULL);
        WarmupGemm(cublas_handle, m, n, n, d_A0, lda, d_B, n, d_C, m, opts.warmup);
        AssertCuda(cudaFree(d_B), "cudaFree d_B");
        AssertCuda(cudaFree(d_C), "cudaFree d_C");
        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

        const float tsqr_ms = TimeKernelMs(
            [&]() {
                AssertCuda(cudaMemcpy(d_A_tsqr, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpy D2D tsqr");
            },
            [&]() {
                tsqr(cublas_handle, m, d_A_tsqr, lda, d_R, ldr, d_work_tsqr, work_elems);
                AssertCuda(cudaGetLastError(), "tsqr launch");
            },
            opts.iters);

        const float geqrf_ms = TimeKernelMs(
            [&]() {
                AssertCuda(cudaMemcpy(d_A_geqrf, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpy D2D geqrf");
            },
            [&]() {
                AssertCusolver(cusolverDnSgeqrf(cusolver_handle, m, n, d_A_geqrf, lda, d_tau,
                                                d_work_geqrf, lwork, d_info),
                               "cusolverDnSgeqrf");
            },
            opts.iters);

        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize bench");

        const double tsqr_tflops = FlopsToTflops(qr_flops, tsqr_ms);
        const double geqrf_tflops = FlopsToTflops(qr_flops, geqrf_ms);
        std::printf("TSQR avg:   %.3f ms (%.3f TFLOPS)\n", tsqr_ms, tsqr_tflops);
        std::printf("GEQRF avg:  %.3f ms (%.3f TFLOPS)\n", geqrf_ms, geqrf_tflops);

        AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
        AssertCuda(cudaFree(d_A_tsqr), "cudaFree d_A_tsqr");
        AssertCuda(cudaFree(d_A_geqrf), "cudaFree d_A_geqrf");
        AssertCuda(cudaFree(d_R), "cudaFree d_R");
        AssertCuda(cudaFree(d_tau), "cudaFree d_tau");
        AssertCuda(cudaFree(d_work_tsqr), "cudaFree d_work_tsqr");
        if (d_work_geqrf) {
            AssertCuda(cudaFree(d_work_geqrf), "cudaFree d_work_geqrf");
        }
        AssertCuda(cudaFree(d_info), "cudaFree d_info");
    }

    AssertCusolver(cusolverDnDestroy(cusolver_handle), "cusolverDnDestroy");
    AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    return 0;
}
