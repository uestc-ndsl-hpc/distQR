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

constexpr int kPanelWidth = 32;

static inline size_t Idx2D(int row, int col, int ld) {
    return static_cast<size_t>(row) + static_cast<size_t>(col) * static_cast<size_t>(ld);
}

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

template <typename T>
void BlockedQrFactorize(cublasHandle_t cublas_handle,
                        int m,
                        int n,
                        int nb,
                        T* d_A,
                        int lda,
                        T* d_W,
                        T* d_Y,
                        T* d_rtmp,
                        T* d_tsqr_work,
                        size_t tsqr_work_elems_m,
                        cudaStream_t stream) {
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);
    const T minus_one = static_cast<T>(-1);

    for (int outer_index = 0; outer_index < n; outer_index += nb) {
        const int end = std::min(outer_index + nb, n);
        const int kb = end - outer_index;
        const int m_sub = m - outer_index;
        T* w_big = d_W + Idx2D(outer_index, outer_index, lda);
        T* y_big = d_Y + Idx2D(outer_index, outer_index, lda);

        for (int inner_index = outer_index; inner_index < end; inner_index += kPanelWidth) {
            const int panel_height = m - inner_index;
            T* panel_A = d_A + Idx2D(inner_index, inner_index, lda);
            T* panel_W = d_W + Idx2D(inner_index, inner_index, lda);
            T* panel_Y = d_Y + Idx2D(inner_index, inner_index, lda);

            tsqr<T>(cublas_handle, panel_height, panel_A, lda, d_rtmp, kPanelWidth, d_tsqr_work,
                    tsqr_work_elems_m, stream);
            generate_wy(panel_height, kPanelWidth, panel_A, lda, panel_Y, lda, panel_W, lda, stream);
            write_back_R2A(kPanelWidth, kPanelWidth, d_rtmp, kPanelWidth, panel_A, lda, stream);

            const int n_remain_in_block = end - (inner_index + kPanelWidth);
            if (n_remain_in_block > 0) {
                T* a_remain = panel_A + kPanelWidth * lda;
                T* work = d_rtmp;  // (b x n_remain)
                AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                      kPanelWidth, n_remain_in_block, panel_height,
                                                      &one, panel_W, lda, a_remain, lda, &zero,
                                                      work, kPanelWidth),
                             "in-block W^T*A");
                AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                      panel_height, n_remain_in_block, kPanelWidth,
                                                      &minus_one, panel_Y, lda, work, kPanelWidth,
                                                      &one, a_remain, lda),
                             "in-block A -= Y*work");
            }

            if (inner_index > outer_index) {
                const int k_prev = inner_index - outer_index;
                T* w_prev = w_big;  // (m_sub x k_prev)
                T* y_prev = y_big;  // (m_sub x k_prev)
                T* w_i_sub = d_W + Idx2D(outer_index, inner_index, lda);  // (m_sub x b)

                AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                      k_prev, kPanelWidth, m_sub, &one, y_prev,
                                                      lda, w_i_sub, lda, &zero, d_rtmp, nb),
                             "WY tmp = Y_prev^T * W_i");
                AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_sub,
                                                      kPanelWidth, k_prev, &minus_one, w_prev, lda,
                                                      d_rtmp, nb, &one, w_i_sub, lda),
                             "WY W_i -= W_prev*tmp");
            }
        }

        const int n_trail = n - end;
        if (n_trail > 0) {
            T* a_trail = d_A + Idx2D(outer_index, end, lda);  // (m_sub x n_trail)
            for (int col0 = 0; col0 < n_trail; col0 += nb) {
                const int tile = std::min(nb, n_trail - col0);
                T* a_tile = a_trail + static_cast<size_t>(col0) * static_cast<size_t>(lda);
                AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kb,
                                                      tile, m_sub, &one, w_big, lda, a_tile, lda,
                                                      &zero, d_rtmp, kb),
                             "trail work = W^T*A");
                AssertCublas(CublasGemmTraits<T>::Gemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m_sub,
                                                      tile, kb, &minus_one, y_big, lda, d_rtmp, kb,
                                                      &one, a_tile, lda),
                             "trail A -= Y*work");
            }
        }
    }
}

struct Options {
    int m = 16384;
    int n = 16384;
    int nb = 1024;
    int iters = 10;
    int warmup = 2;
    bool use_double = false;
    bool run_geqrf = true;
};

Options ParseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            opts.m = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            opts.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nb") == 0 && i + 1 < argc) {
            opts.nb = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opts.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opts.warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            std::string type = argv[++i];
            opts.use_double = (type == "double" || type == "fp64");
        } else if (std::strcmp(argv[i], "--no-geqrf") == 0) {
            opts.run_geqrf = false;
        }
    }
    return opts;
}

template <typename T>
void RunBench(const Options& opts, cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle) {
    const int m = opts.m;
    const int n = opts.n;
    const int nb = opts.nb;
    const int lda = m;
    const double qr_flops = QrFlops(m, n);

    const size_t a_elems = static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t a_bytes = a_elems * sizeof(T);
    const size_t wy_bytes = a_bytes;
    const size_t rtmp_bytes = static_cast<size_t>(nb) * static_cast<size_t>(nb) * sizeof(T);
    const size_t tsqr_work_elems_m = tsqr_work_elems<T>(m);
    const size_t tsqr_work_bytes = tsqr_work_elems_m * sizeof(T);

    T* d_A0 = nullptr;
    T* d_A = nullptr;
    T* d_W = nullptr;
    T* d_Y = nullptr;
    T* d_rtmp = nullptr;
    T* d_work_tsqr = nullptr;

    AssertCuda(cudaMalloc(&d_A0, a_bytes), "cudaMalloc d_A0");
    AssertCuda(cudaMalloc(&d_A, a_bytes), "cudaMalloc d_A");
    AssertCuda(cudaMalloc(&d_W, wy_bytes), "cudaMalloc d_W");
    AssertCuda(cudaMalloc(&d_Y, wy_bytes), "cudaMalloc d_Y");
    AssertCuda(cudaMalloc(&d_rtmp, rtmp_bytes), "cudaMalloc d_rtmp");
    if (tsqr_work_elems_m > 0) {
        AssertCuda(cudaMalloc(&d_work_tsqr, tsqr_work_bytes), "cudaMalloc d_work_tsqr");
    }

    FillDeviceRandom(d_A0, a_elems, 1234ULL);

    // warmup with one iteration (not timed)
    for (int i = 0; i < opts.warmup; ++i) {
        AssertCuda(cudaMemcpy(d_A, d_A0, a_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D warmup");
        AssertCuda(cudaMemset(d_W, 0, wy_bytes), "cudaMemset d_W warmup");
        AssertCuda(cudaMemset(d_Y, 0, wy_bytes), "cudaMemset d_Y warmup");
        BlockedQrFactorize<T>(cublas_handle, m, n, nb, d_A, lda, d_W, d_Y, d_rtmp, d_work_tsqr,
                              tsqr_work_elems_m, nullptr);
        AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");
    }

    const float blocked_ms = TimeKernelMs(
        [&]() {
            AssertCuda(cudaMemcpy(d_A, d_A0, a_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D A");
            AssertCuda(cudaMemset(d_W, 0, wy_bytes), "cudaMemset d_W");
            AssertCuda(cudaMemset(d_Y, 0, wy_bytes), "cudaMemset d_Y");
        },
        [&]() {
            BlockedQrFactorize<T>(cublas_handle, m, n, nb, d_A, lda, d_W, d_Y, d_rtmp, d_work_tsqr,
                                  tsqr_work_elems_m, nullptr);
            AssertCuda(cudaGetLastError(), "blocked qr launch");
        },
        opts.iters);

    const double blocked_tflops = FlopsToTflops(qr_flops, blocked_ms);
    std::printf("BlockedQR avg: %.3f ms (%.3f TFLOPS)\n", blocked_ms, blocked_tflops);

    if (opts.run_geqrf) {
        T* d_A_geqrf = nullptr;
        T* d_tau = nullptr;
        T* d_work_geqrf = nullptr;
        int* d_info = nullptr;
        AssertCuda(cudaMalloc(&d_A_geqrf, a_bytes), "cudaMalloc d_A_geqrf");
        AssertCuda(cudaMalloc(&d_tau, static_cast<size_t>(n) * sizeof(T)), "cudaMalloc d_tau");
        AssertCuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc d_info");

        int lwork = 0;
        if constexpr (std::is_same_v<T, float>) {
            AssertCusolver(cusolverDnSgeqrf_bufferSize(cusolver_handle, m, n, d_A_geqrf, lda, &lwork),
                           "cusolverDnSgeqrf_bufferSize");
        } else {
            AssertCusolver(cusolverDnDgeqrf_bufferSize(cusolver_handle, m, n, d_A_geqrf, lda, &lwork),
                           "cusolverDnDgeqrf_bufferSize");
        }
        if (lwork > 0) {
            AssertCuda(cudaMalloc(&d_work_geqrf, static_cast<size_t>(lwork) * sizeof(T)),
                       "cudaMalloc d_work_geqrf");
        }

        // warmup
        for (int i = 0; i < opts.warmup; ++i) {
            AssertCuda(cudaMemcpy(d_A_geqrf, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy D2D geqrf warmup");
            if constexpr (std::is_same_v<T, float>) {
                AssertCusolver(cusolverDnSgeqrf(cusolver_handle, m, n, d_A_geqrf, lda, d_tau,
                                                d_work_geqrf, lwork, d_info),
                               "cusolverDnSgeqrf");
            } else {
                AssertCusolver(cusolverDnDgeqrf(cusolver_handle, m, n, d_A_geqrf, lda, d_tau,
                                                d_work_geqrf, lwork, d_info),
                               "cusolverDnDgeqrf");
            }
            AssertCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize geqrf warmup");
        }

        const float geqrf_ms = TimeKernelMs(
            [&]() {
                AssertCuda(cudaMemcpy(d_A_geqrf, d_A0, a_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpy D2D geqrf");
            },
            [&]() {
                if constexpr (std::is_same_v<T, float>) {
                    AssertCusolver(cusolverDnSgeqrf(cusolver_handle, m, n, d_A_geqrf, lda, d_tau,
                                                    d_work_geqrf, lwork, d_info),
                                   "cusolverDnSgeqrf");
                } else {
                    AssertCusolver(cusolverDnDgeqrf(cusolver_handle, m, n, d_A_geqrf, lda, d_tau,
                                                    d_work_geqrf, lwork, d_info),
                                   "cusolverDnDgeqrf");
                }
            },
            opts.iters);

        const double geqrf_tflops = FlopsToTflops(qr_flops, geqrf_ms);
        std::printf("GEQRF avg:    %.3f ms (%.3f TFLOPS)\n", geqrf_ms, geqrf_tflops);

        if (d_work_geqrf) {
            AssertCuda(cudaFree(d_work_geqrf), "cudaFree d_work_geqrf");
        }
        AssertCuda(cudaFree(d_A_geqrf), "cudaFree d_A_geqrf");
        AssertCuda(cudaFree(d_tau), "cudaFree d_tau");
        AssertCuda(cudaFree(d_info), "cudaFree d_info");
    }

    AssertCuda(cudaFree(d_A0), "cudaFree d_A0");
    AssertCuda(cudaFree(d_A), "cudaFree d_A");
    AssertCuda(cudaFree(d_W), "cudaFree d_W");
    AssertCuda(cudaFree(d_Y), "cudaFree d_Y");
    AssertCuda(cudaFree(d_rtmp), "cudaFree d_rtmp");
    if (d_work_tsqr) {
        AssertCuda(cudaFree(d_work_tsqr), "cudaFree d_work_tsqr");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options opts = ParseArgs(argc, argv);

    if (opts.m <= 0 || opts.n <= 0 || opts.nb <= 0) {
        std::fprintf(stderr, "Invalid args: require m,n,nb > 0\n");
        return 1;
    }
    if (opts.m < opts.n) {
        std::fprintf(stderr, "Invalid args: require m >= n\n");
        return 1;
    }
    if (opts.n % kPanelWidth != 0 || opts.nb % kPanelWidth != 0) {
        std::fprintf(stderr, "Invalid args: require n and nb to be multiples of %d\n", kPanelWidth);
        return 1;
    }

    std::printf("QR bench: m=%d n=%d nb=%d b=%d iters=%d warmup=%d type=%s %s\n",
                opts.m, opts.n, opts.nb, kPanelWidth, opts.iters, opts.warmup,
                opts.use_double ? "double" : "float",
                opts.run_geqrf ? "" : "(no geqrf)");

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    AssertCublas(cublasCreate(&cublas_handle), "cublasCreate");
    AssertCusolver(cusolverDnCreate(&cusolver_handle), "cusolverDnCreate");

    if (opts.use_double) {
        RunBench<double>(opts, cublas_handle, cusolver_handle);
    } else {
        RunBench<float>(opts, cublas_handle, cusolver_handle);
    }

    AssertCusolver(cusolverDnDestroy(cusolver_handle), "cusolverDnDestroy");
    AssertCublas(cublasDestroy(cublas_handle), "cublasDestroy");
    return 0;
}

