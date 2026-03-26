#include "components/resourse_initial.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Options {
    size_t bytes = size_t{1} << 30;
    int warmup = 2;
    int iters = 5;
    bool check = false;
    std::string mode = "all";
};

struct CommContext {
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    ncclComm_t nccl_comm = nullptr;
    int rank = -1;
    int size = 0;
};

struct BenchResult {
    std::string mode;
    double avg_ms = 0.0;
    double logical_gbps = 0.0;
    double inter_node_gbps = 0.0;
};

constexpr int kRootRank = 0;
constexpr unsigned char kPatternMul = 131;
constexpr unsigned char kPatternAdd = 17;

__global__ void FillPatternKernel(unsigned char* data, size_t bytes) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= bytes) {
        return;
    }
    data[idx] = static_cast<unsigned char>((idx * kPatternMul + kPatternAdd) & 0xff);
}

unsigned char ExpectedByte(size_t idx) {
    return static_cast<unsigned char>((idx * kPatternMul + kPatternAdd) & 0xff);
}

bool CheckCuda(cudaError_t err, const char* expr) {
    if (err == cudaSuccess) {
        return true;
    }
    spdlog::error("CUDA error at {}: {}", expr, cudaGetErrorString(err));
    return false;
}

bool CheckNccl(ncclResult_t err, const char* expr) {
    if (err == ncclSuccess) {
        return true;
    }
    spdlog::error("NCCL error at {}: {}", expr, ncclGetErrorString(err));
    return false;
}

bool InitSubComm(const MpiCudaEnv& env, MPI_Comm mpi_comm, CommContext* ctx) {
    if (!ctx || mpi_comm == MPI_COMM_NULL) {
        return false;
    }

    ctx->mpi_comm = mpi_comm;
    MPI_Comm_rank(mpi_comm, &ctx->rank);
    MPI_Comm_size(mpi_comm, &ctx->size);

    ncclUniqueId id{};
    if (ctx->rank == 0) {
        if (!CheckNccl(ncclGetUniqueId(&id), "ncclGetUniqueId")) {
            return false;
        }
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);

    spdlog::info("sub-comm init start: world rank {} node {} subrank {} size {}", env.rank,
                 env.node_name, ctx->rank, ctx->size);
    if (!CheckNccl(ncclCommInitRank(&ctx->nccl_comm, ctx->size, id, ctx->rank),
                   "ncclCommInitRank(subcomm)")) {
        return false;
    }
    spdlog::info("sub-comm init done: world rank {} node {} subrank {} size {}", env.rank,
                 env.node_name, ctx->rank, ctx->size);
    return true;
}

void DestroySubComm(CommContext* ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->nccl_comm != nullptr) {
        ncclCommDestroy(ctx->nccl_comm);
        ctx->nccl_comm = nullptr;
    }
    if (ctx->mpi_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&ctx->mpi_comm);
        ctx->mpi_comm = MPI_COMM_NULL;
    }
}

bool ParsePositiveInt(const char* arg, int* value) {
    if (!arg || !value) {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(arg, &end, 10);
    if (end == arg || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool ParsePositiveSize(const char* arg, size_t* value) {
    if (!arg || !value) {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(arg, &end, 10);
    if (end == arg || *end != '\0' || parsed == 0) {
        return false;
    }
    *value = static_cast<size_t>(parsed);
    return true;
}

void PrintUsage(const char* prog) {
    spdlog::info("Usage: {} [--bytes N | --mb N] [--warmup N] [--iters N] "
                 "[--mode broadcast|hierarchical|all] [--check]",
                 prog);
}

bool ParseOptions(int argc, char** argv, Options* opts) {
    if (!opts) {
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bytes" && i + 1 < argc) {
            if (!ParsePositiveSize(argv[++i], &opts->bytes)) {
                spdlog::error("Invalid value for --bytes: {}", argv[i]);
                return false;
            }
        } else if (arg == "--mb" && i + 1 < argc) {
            size_t mb = 0;
            if (!ParsePositiveSize(argv[++i], &mb)) {
                spdlog::error("Invalid value for --mb: {}", argv[i]);
                return false;
            }
            opts->bytes = mb * size_t{1024} * size_t{1024};
        } else if (arg == "--warmup" && i + 1 < argc) {
            if (!ParsePositiveInt(argv[++i], &opts->warmup)) {
                spdlog::error("Invalid value for --warmup: {}", argv[i]);
                return false;
            }
        } else if (arg == "--iters" && i + 1 < argc) {
            if (!ParsePositiveInt(argv[++i], &opts->iters)) {
                spdlog::error("Invalid value for --iters: {}", argv[i]);
                return false;
            }
        } else if (arg == "--mode" && i + 1 < argc) {
            opts->mode = argv[++i];
            if (opts->mode != "broadcast" && opts->mode != "hierarchical" &&
                opts->mode != "all") {
                spdlog::error("Invalid value for --mode: {}", opts->mode);
                return false;
            }
        } else if (arg == "--check") {
            opts->check = true;
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            spdlog::error("Unknown argument: {}", arg);
            return false;
        }
    }
    return true;
}

bool FillRootBuffer(const MpiCudaEnv& env, unsigned char* device_buffer, size_t bytes,
                    cudaStream_t stream) {
    if (env.rank != kRootRank) {
        return CheckCuda(cudaMemsetAsync(device_buffer, 0, bytes, stream),
                         "cudaMemsetAsync(non-root)");
    }

    constexpr int threads = 256;
    const size_t blocks = (bytes + threads - 1) / threads;
    FillPatternKernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(device_buffer,
                                                                                 bytes);
    if (!CheckCuda(cudaGetLastError(), "FillPatternKernel launch")) {
        return false;
    }
    return true;
}

bool RunBroadcastOnce(ncclComm_t global_comm, unsigned char* device_buffer, size_t bytes,
                      cudaStream_t stream) {
    return CheckNccl(
        ncclBroadcast(device_buffer, device_buffer, bytes, ncclUint8, kRootRank, global_comm, stream),
        "ncclBroadcast(global)");
}

bool RunHierarchicalOnce(const MpiCudaEnv& env, const CommContext& local_ctx,
                         const CommContext* leader_ctx, int node_count, unsigned char* device_buffer,
                         size_t bytes, cudaStream_t stream) {
    if (env.local_rank == 0 && node_count > 1 && leader_ctx != nullptr) {
        ncclGroupStart();
        if (env.rank == kRootRank) {
            for (int peer = 1; peer < leader_ctx->size; ++peer) {
                if (!CheckNccl(
                        ncclSend(device_buffer, bytes, ncclUint8, peer, leader_ctx->nccl_comm, stream),
                        "ncclSend(leader)")) {
                    ncclGroupEnd();
                    return false;
                }
            }
        } else {
            if (!CheckNccl(
                    ncclRecv(device_buffer, bytes, ncclUint8, 0, leader_ctx->nccl_comm, stream),
                    "ncclRecv(leader)")) {
                ncclGroupEnd();
                return false;
            }
        }
        if (!CheckNccl(ncclGroupEnd(), "ncclGroupEnd(leader send/recv)")) {
            return false;
        }
    }

    return CheckNccl(ncclBroadcast(device_buffer, device_buffer, bytes, ncclUint8, 0,
                                   local_ctx.nccl_comm, stream),
                     "ncclBroadcast(local)");
}

bool VerifyBuffer(const MpiCudaEnv& env, unsigned char* device_buffer, size_t bytes) {
    const std::vector<size_t> sample_offsets = {
        0,
        bytes / 7,
        bytes / 5,
        bytes / 3,
        bytes / 2,
        (bytes * 3) / 4,
        bytes > 0 ? bytes - 1 : 0,
    };

    std::vector<unsigned char> host_values(sample_offsets.size(), 0);
    for (size_t i = 0; i < sample_offsets.size(); ++i) {
        const size_t offset = std::min(sample_offsets[i], bytes - 1);
        if (!CheckCuda(cudaMemcpy(&host_values[i], device_buffer + offset, sizeof(unsigned char),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(sample)")) {
            return false;
        }
        if (host_values[i] != ExpectedByte(offset)) {
            spdlog::error("Rank {} verification failed at offset {}: got {}, expected {}", env.rank,
                          offset, static_cast<int>(host_values[i]),
                          static_cast<int>(ExpectedByte(offset)));
            return false;
        }
    }
    return true;
}

template <typename Runner>
bool TimeMode(const MpiCudaEnv& env, const Options& opts, const std::string& mode_name,
              int node_count, Runner&& runner, BenchResult* result) {
    if (!result) {
        return false;
    }

    unsigned char* device_buffer = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    auto cleanup = [&]() {
        if (start != nullptr) {
            cudaEventDestroy(start);
            start = nullptr;
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
            stop = nullptr;
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        if (device_buffer != nullptr) {
            cudaFree(device_buffer);
            device_buffer = nullptr;
        }
    };

    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_buffer), opts.bytes), "cudaMalloc")) {
        return false;
    }
    if (!CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags")) {
        cleanup();
        return false;
    }
    if (!CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)") ||
        !CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
        cleanup();
        return false;
    }

    for (int i = 0; i < opts.warmup; ++i) {
        if (!FillRootBuffer(env, device_buffer, opts.bytes, stream)) {
            cleanup();
            return false;
        }
        if (!runner(device_buffer, stream)) {
            cleanup();
            return false;
        }
        if (!CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)")) {
            cleanup();
            return false;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double total_ms = 0.0;
    for (int iter = 0; iter < opts.iters; ++iter) {
        if (!FillRootBuffer(env, device_buffer, opts.bytes, stream)) {
            cleanup();
            return false;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (!CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord(start)")) {
            cleanup();
            return false;
        }
        if (!runner(device_buffer, stream)) {
            cleanup();
            return false;
        }
        if (!CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)")) {
            cleanup();
            return false;
        }
        if (!CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)")) {
            cleanup();
            return false;
        }

        float local_ms = 0.0f;
        if (!CheckCuda(cudaEventElapsedTime(&local_ms, start, stop), "cudaEventElapsedTime")) {
            cleanup();
            return false;
        }

        double global_ms = static_cast<double>(local_ms);
        MPI_Allreduce(MPI_IN_PLACE, &global_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        total_ms += global_ms;
    }

    if (opts.check) {
        if (!FillRootBuffer(env, device_buffer, opts.bytes, stream)) {
            cleanup();
            return false;
        }
        if (!runner(device_buffer, stream)) {
            cleanup();
            return false;
        }
        if (!CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(check)")) {
            cleanup();
            return false;
        }

        int local_ok = VerifyBuffer(env, device_buffer, opts.bytes) ? 1 : 0;
        int global_ok = 0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (env.rank == 0) {
            spdlog::info("Correctness [{}]: {}", mode_name, global_ok ? "PASS" : "FAIL");
        }
        if (!global_ok) {
            cleanup();
            return false;
        }
    }

    cleanup();

    result->mode = mode_name;
    result->avg_ms = total_ms / static_cast<double>(opts.iters);

    const double logical_bytes = static_cast<double>(opts.bytes) * static_cast<double>(env.size - 1);
    result->logical_gbps = logical_bytes / (result->avg_ms * 1.0e6);

    const double inter_node_bytes =
        mode_name == "hierarchical"
            ? static_cast<double>(opts.bytes) * static_cast<double>(std::max(0, node_count - 1))
            : logical_bytes;
    result->inter_node_gbps = inter_node_bytes / (result->avg_ms * 1.0e6);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    auto env = init_mpi_and_bind_gpu(&argc, &argv);
    init(&env);

    if (!init_nccl_comm(&env)) {
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }

    Options opts;
    if (!ParseOptions(argc, argv, &opts)) {
        if (env.rank == 0) {
            PrintUsage(argv[0]);
        }
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }

    MPI_Comm local_mpi_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, env.rank, MPI_INFO_NULL, &local_mpi_comm);

    CommContext local_ctx;
    if (!InitSubComm(env, local_mpi_comm, &local_ctx)) {
        DestroySubComm(&local_ctx);
        finalize_nccl_if_needed(&env);
        finalize_mpi_if_needed(env);
        return 1;
    }

    MPI_Comm leader_mpi_comm = MPI_COMM_NULL;
    CommContext leader_ctx;
    CommContext* leader_ctx_ptr = nullptr;
    if (env.local_rank == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, 0, env.rank, &leader_mpi_comm);
        if (!InitSubComm(env, leader_mpi_comm, &leader_ctx)) {
            DestroySubComm(&local_ctx);
            DestroySubComm(&leader_ctx);
            finalize_nccl_if_needed(&env);
            finalize_mpi_if_needed(env);
            return 1;
        }
        leader_ctx_ptr = &leader_ctx;
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, env.rank, &leader_mpi_comm);
    }

    int node_count = 0;
    int is_leader = env.local_rank == 0 ? 1 : 0;
    MPI_Allreduce(&is_leader, &node_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (env.rank == 0) {
        spdlog::info(
            "NCCL transfer bench: bytes={} MiB={} ranks={} nodes={} mode={} warmup={} iters={} check={}",
            opts.bytes, static_cast<double>(opts.bytes) / (1024.0 * 1024.0), env.size, node_count,
            opts.mode, opts.warmup, opts.iters, opts.check ? "on" : "off");
    }

    std::vector<BenchResult> results;
    results.reserve(2);

    if (opts.mode == "broadcast" || opts.mode == "all") {
        BenchResult result;
        const bool ok = TimeMode(
            env, opts, "broadcast", node_count,
            [&](unsigned char* device_buffer, cudaStream_t stream) {
                return RunBroadcastOnce(env.nccl_comm, device_buffer, opts.bytes, stream);
            },
            &result);
        if (!ok) {
            DestroySubComm(&local_ctx);
            DestroySubComm(&leader_ctx);
            finalize_nccl_if_needed(&env);
            finalize_mpi_if_needed(env);
            return 1;
        }
        results.push_back(result);
    }

    if (opts.mode == "hierarchical" || opts.mode == "all") {
        BenchResult result;
        const bool ok = TimeMode(
            env, opts, "hierarchical", node_count,
            [&](unsigned char* device_buffer, cudaStream_t stream) {
                return RunHierarchicalOnce(env, local_ctx, leader_ctx_ptr, node_count, device_buffer,
                                           opts.bytes, stream);
            },
            &result);
        if (!ok) {
            DestroySubComm(&local_ctx);
            DestroySubComm(&leader_ctx);
            finalize_nccl_if_needed(&env);
            finalize_mpi_if_needed(env);
            return 1;
        }
        results.push_back(result);
    }

    if (env.rank == 0) {
        for (const auto& result : results) {
            spdlog::info("[{}] avg {:.3f} ms | logical {:.3f} GB/s | inter-node {:.3f} GB/s",
                         result.mode, result.avg_ms, result.logical_gbps, result.inter_node_gbps);
        }
    }

    DestroySubComm(&local_ctx);
    DestroySubComm(&leader_ctx);
    finalize_nccl_if_needed(&env);
    finalize_mpi_if_needed(env);
    return 0;
}
