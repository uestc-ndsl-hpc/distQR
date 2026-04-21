#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "components/panel_cusolver.cuh"
#include "components/panel_tsqr.cuh"

namespace {

constexpr int kPanelWidth = 32;

void AssertCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", context, cudaGetErrorString(status));
        std::exit(1);
    }
}

void AssertCusolver(cusolverStatus_t status, const char* context) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cusolver error %d\n", context, static_cast<int>(status));
        std::exit(1);
    }
}

double BytesToGiB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

enum class PanelBackend {
    Tsqr,
    Cusolver,
};

const char* PanelBackendToString(PanelBackend backend) {
    switch (backend) {
        case PanelBackend::Tsqr:
            return "tsqr";
        case PanelBackend::Cusolver:
            return "cusolver";
    }
    return "unknown";
}

bool ParsePanelBackend(const char* value, PanelBackend* backend) {
    if (!value || !backend) {
        return false;
    }
    if (std::strcmp(value, "tsqr") == 0) {
        *backend = PanelBackend::Tsqr;
        return true;
    }
    if (std::strcmp(value, "cusolver") == 0) {
        *backend = PanelBackend::Cusolver;
        return true;
    }
    return false;
}

bool ParseInt(const char* value, int* out) {
    if (!value || !out) {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(parsed);
    return true;
}

bool ParseNonNegativeInt(const char* value, int* out) {
    if (!value || !out) {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(parsed);
    return true;
}

bool ParseDouble(const char* value, double* out) {
    if (!value || !out) {
        return false;
    }
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (!end || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0) {
        return false;
    }
    *out = parsed;
    return true;
}

struct Options {
    int device = 0;
    int nb = 1024;
    int step = 1024;
    int trail_tile_cols = 1024;
    bool trail_one_shot = false;
    bool budget_explicit = false;
    size_t budget_bytes = 0;
    PanelBackend panel_backend = PanelBackend::Tsqr;
};

Options ParseArgs(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!ParseNonNegativeInt(argv[++i], &opts.device)) {
                std::fprintf(stderr, "Invalid --device value: %s\n", argv[i]);
                std::exit(1);
            }
        } else if (std::strcmp(argv[i], "--nb") == 0 && i + 1 < argc) {
            if (!ParseInt(argv[++i], &opts.nb)) {
                std::fprintf(stderr, "Invalid --nb value: %s\n", argv[i]);
                std::exit(1);
            }
        } else if (std::strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            if (!ParseInt(argv[++i], &opts.step)) {
                std::fprintf(stderr, "Invalid --step value: %s\n", argv[i]);
                std::exit(1);
            }
        } else if (std::strcmp(argv[i], "--trail-tile-cols") == 0 && i + 1 < argc) {
            if (!ParseInt(argv[++i], &opts.trail_tile_cols)) {
                std::fprintf(stderr, "Invalid --trail-tile-cols value: %s\n", argv[i]);
                std::exit(1);
            }
        } else if (std::strcmp(argv[i], "--trail-one-shot") == 0) {
            opts.trail_one_shot = true;
        } else if (std::strcmp(argv[i], "--trail-tiled") == 0) {
            opts.trail_one_shot = false;
        } else if (std::strcmp(argv[i], "--budget-mib") == 0 && i + 1 < argc) {
            double mib = 0.0;
            if (!ParseDouble(argv[++i], &mib)) {
                std::fprintf(stderr, "Invalid --budget-mib value: %s\n", argv[i]);
                std::exit(1);
            }
            opts.budget_bytes = static_cast<size_t>(mib * 1024.0 * 1024.0);
            opts.budget_explicit = true;
        } else if (std::strcmp(argv[i], "--budget-gib") == 0 && i + 1 < argc) {
            double gib = 0.0;
            if (!ParseDouble(argv[++i], &gib)) {
                std::fprintf(stderr, "Invalid --budget-gib value: %s\n", argv[i]);
                std::exit(1);
            }
            opts.budget_bytes = static_cast<size_t>(gib * 1024.0 * 1024.0 * 1024.0);
            opts.budget_explicit = true;
        } else if (std::strcmp(argv[i], "--panel-backend") == 0 && i + 1 < argc) {
            if (!ParsePanelBackend(argv[++i], &opts.panel_backend)) {
                std::fprintf(stderr,
                             "Invalid --panel-backend value: %s (supported: tsqr, cusolver)\n",
                             argv[i]);
                std::exit(1);
            }
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::printf(
                "Usage: bench_qr_max_fp32_size [options]\n"
                "  --device <int>             CUDA device index (default: 0)\n"
                "  --nb <int>                 blocked-QR outer width (default: 1024)\n"
                "  --step <int>               scan step for square N (default: 1024)\n"
                "  --trail-one-shot           use one-shot trailing update sizing\n"
                "  --trail-tiled              use tiled trailing update sizing (default)\n"
                "  --trail-tile-cols <int>    tiled update width (default: 1024)\n"
                "  --panel-backend <tsqr|cusolver>\n"
                "  --budget-mib <float>       explicit memory budget in MiB\n"
                "  --budget-gib <float>       explicit memory budget in GiB\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            std::exit(1);
        }
    }
    return opts;
}

size_t MatrixBytes(int m, int n) {
    return static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
}

size_t BaseBytes(int m, int n) {
    // bench_qr keeps both d_A0 and d_A.
    return 2 * MatrixBytes(m, n);
}

size_t PanelBackendExtraBytes(PanelBackend backend,
                              int m,
                              float* d_dummy_a,
                              float* d_dummy_tau,
                              cusolverDnHandle_t cusolver_handle) {
    if (backend == PanelBackend::Tsqr) {
        return tsqr_work_elems<float>(m) * sizeof(float);
    }

    int panel_lwork_geqrf = 0;
    int panel_lwork_orgqr = 0;
    AssertCusolver(CusolverPanelQrTraits<float>::GeqrfBufferSize(cusolver_handle, m, kPanelWidth,
                                                                 d_dummy_a, m,
                                                                 &panel_lwork_geqrf),
                   "panel cusolver geqrf_bufferSize");
    AssertCusolver(CusolverPanelQrTraits<float>::OrgqrBufferSize(cusolver_handle, m, kPanelWidth,
                                                                 kPanelWidth, d_dummy_a, m,
                                                                 d_dummy_tau,
                                                                 &panel_lwork_orgqr),
                   "panel cusolver orgqr_bufferSize");

    return static_cast<size_t>(std::max(panel_lwork_geqrf, panel_lwork_orgqr)) * sizeof(float) +
           static_cast<size_t>(kPanelWidth) * sizeof(float) + sizeof(int);
}

size_t BenchQrNoWyExtraBytes(const Options& opts,
                             int m,
                             int n,
                             float* d_dummy_a,
                             float* d_dummy_tau,
                             cusolverDnHandle_t cusolver_handle) {
    const size_t wy_bytes_each =
        static_cast<size_t>(m) * static_cast<size_t>(opts.nb) * sizeof(float);
    const int trail_tile_cols = opts.trail_tile_cols > 0 ? opts.trail_tile_cols : opts.nb;
    const int max_trail_cols =
        opts.trail_one_shot ? std::max(opts.nb, n - opts.nb) : std::max(opts.nb, trail_tile_cols);
    const size_t rtmp_bytes =
        static_cast<size_t>(opts.nb) * static_cast<size_t>(max_trail_cols) * sizeof(float);

    return 2 * wy_bytes_each +
           PanelBackendExtraBytes(opts.panel_backend, m, d_dummy_a, d_dummy_tau, cusolver_handle) +
           rtmp_bytes;
}

size_t CusolverGeqrfExtraBytes(int m, int n, float* d_dummy_a, cusolverDnHandle_t cusolver_handle) {
    int lwork_geqrf = 0;
    AssertCusolver(cusolverDnSgeqrf_bufferSize(cusolver_handle, m, n, d_dummy_a, m,
                                               &lwork_geqrf),
                   "cusolverDnSgeqrf_bufferSize");
    return static_cast<size_t>(lwork_geqrf) * sizeof(float) +
           static_cast<size_t>(n) * sizeof(float) + sizeof(int);
}

size_t BenchQrNoWyTotalBytes(const Options& opts,
                             int m,
                             int n,
                             float* d_dummy_a,
                             float* d_dummy_tau,
                             cusolverDnHandle_t cusolver_handle) {
    return BaseBytes(m, n) +
           BenchQrNoWyExtraBytes(opts, m, n, d_dummy_a, d_dummy_tau, cusolver_handle);
}

size_t CusolverGeqrfTotalBytes(int m,
                               int n,
                               float* d_dummy_a,
                               cusolverDnHandle_t cusolver_handle) {
    return BaseBytes(m, n) + CusolverGeqrfExtraBytes(m, n, d_dummy_a, cusolver_handle);
}

size_t BenchQrSizeTotalBytes(const Options& opts,
                             int m,
                             int n,
                             float* d_dummy_a,
                             float* d_dummy_tau,
                             cusolverDnHandle_t cusolver_handle) {
    const size_t ours_extra =
        BenchQrNoWyExtraBytes(opts, m, n, d_dummy_a, d_dummy_tau, cusolver_handle);
    const size_t cusolver_extra = CusolverGeqrfExtraBytes(m, n, d_dummy_a, cusolver_handle);
    return BaseBytes(m, n) + std::max(ours_extra, cusolver_extra);
}

struct ScanResult {
    int best_n = 0;
    size_t best_total_bytes = 0;
    size_t best_extra_bytes = 0;
};

void PrintResult(const char* label, const ScanResult& result) {
    if (result.best_n <= 0) {
        std::printf("%-20s : no fit\n", label);
        return;
    }
    std::printf("%-20s : N=%7d  total=%8.2f GiB  extra=%8.2f GiB\n", label, result.best_n,
                BytesToGiB(result.best_total_bytes), BytesToGiB(result.best_extra_bytes));
}

}  // namespace

int main(int argc, char** argv) {
    const Options opts = ParseArgs(argc, argv);

    if (opts.nb <= 0 || opts.step <= 0 || opts.trail_tile_cols <= 0) {
        std::fprintf(stderr, "Require nb, step, trail_tile_cols > 0.\n");
        return 1;
    }
    if (opts.nb % kPanelWidth != 0 || opts.step % kPanelWidth != 0) {
        std::fprintf(stderr, "Require nb and step to be multiples of %d.\n", kPanelWidth);
        return 1;
    }

    int device_count = 0;
    AssertCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        std::fprintf(stderr, "No CUDA device available.\n");
        return 1;
    }
    if (opts.device < 0 || opts.device >= device_count) {
        std::fprintf(stderr, "Invalid device index %d (device_count=%d).\n", opts.device,
                     device_count);
        return 1;
    }

    AssertCuda(cudaSetDevice(opts.device), "cudaSetDevice");
    AssertCuda(cudaFree(nullptr), "cudaFree(nullptr)");

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    AssertCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    const size_t budget_bytes = opts.budget_explicit ? opts.budget_bytes : free_bytes;

    cusolverDnHandle_t cusolver_handle;
    AssertCusolver(cusolverDnCreate(&cusolver_handle), "cusolverDnCreate");

    float* d_dummy_a = nullptr;
    float* d_dummy_tau = nullptr;
    AssertCuda(cudaMalloc(&d_dummy_a, sizeof(float)), "cudaMalloc d_dummy_a");
    AssertCuda(cudaMalloc(&d_dummy_tau, static_cast<size_t>(kPanelWidth) * sizeof(float)),
               "cudaMalloc d_dummy_tau");

    ScanResult ours;
    ScanResult cusolver;
    ScanResult combined;

    for (int n = opts.step;;) {
        const int m = n;
        const size_t ours_total =
            BenchQrNoWyTotalBytes(opts, m, n, d_dummy_a, d_dummy_tau, cusolver_handle);
        const size_t ours_extra =
            ours_total > BaseBytes(m, n) ? ours_total - BaseBytes(m, n) : 0;

        const size_t cusolver_total = CusolverGeqrfTotalBytes(m, n, d_dummy_a, cusolver_handle);
        const size_t cusolver_extra =
            cusolver_total > BaseBytes(m, n) ? cusolver_total - BaseBytes(m, n) : 0;

        const size_t combined_total =
            BenchQrSizeTotalBytes(opts, m, n, d_dummy_a, d_dummy_tau, cusolver_handle);
        const size_t combined_extra =
            combined_total > BaseBytes(m, n) ? combined_total - BaseBytes(m, n) : 0;

        if (ours_total <= budget_bytes) {
            ours.best_n = n;
            ours.best_total_bytes = ours_total;
            ours.best_extra_bytes = ours_extra;
        }
        if (cusolver_total <= budget_bytes) {
            cusolver.best_n = n;
            cusolver.best_total_bytes = cusolver_total;
            cusolver.best_extra_bytes = cusolver_extra;
        }
        if (combined_total <= budget_bytes) {
            combined.best_n = n;
            combined.best_total_bytes = combined_total;
            combined.best_extra_bytes = combined_extra;
        }

        if (ours_total > budget_bytes && cusolver_total > budget_bytes) {
            break;
        }
        if (n > std::numeric_limits<int>::max() - opts.step) {
            break;
        }
        n += opts.step;
    }

    std::printf("GPU %d free=%zu bytes (%.2f GiB), total=%zu bytes (%.2f GiB)\n", opts.device,
                free_bytes, BytesToGiB(free_bytes), total_bytes, BytesToGiB(total_bytes));
    std::printf("Budget=%zu bytes (%.2f GiB)\n", budget_bytes, BytesToGiB(budget_bytes));
    std::printf("Scan: FP32 square matrices, step=%d, nb=%d, panel_backend=%s, trail=%s, "
                "trail_tile_cols=%d\n",
                opts.step, opts.nb, PanelBackendToString(opts.panel_backend),
                opts.trail_one_shot ? "one-shot" : "tiled", opts.trail_tile_cols);
    std::printf("Note: bench_qr_size = 2*A + max(extra_bench_qr_no_wy, extra_cusolver_geqrf)\n");

    PrintResult("bench_qr(no-WY)", ours);
    PrintResult("cusolver_geqrf", cusolver);
    PrintResult("bench_qr_size", combined);

    AssertCuda(cudaFree(d_dummy_tau), "cudaFree d_dummy_tau");
    AssertCuda(cudaFree(d_dummy_a), "cudaFree d_dummy_a");
    AssertCusolver(cusolverDnDestroy(cusolver_handle), "cusolverDnDestroy");
    return 0;
}
