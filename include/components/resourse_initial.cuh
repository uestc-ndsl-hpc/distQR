#pragma once

#include <cuda_runtime.h>
#include <mpi.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <string>
#include <unistd.h>

struct MpiCudaEnv {
    int rank = 0;
    int size = 1;
    int local_rank = 0;
    int local_size = 1;
    int device_id = 0;
    int device_count = 0;
    std::string node_name;
    bool mpi_initialized_here = false;
};

inline bool should_print_on_this_rank(const MpiCudaEnv* env) {
    if (env) {
        return env->local_rank == 0;
    }

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        return true;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    return local_rank == 0;
}

inline void init(const MpiCudaEnv* env = nullptr) {
    if (!should_print_on_this_rank(env)) {
        return;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        spdlog::error("cudaGetDeviceCount failed: {}", cudaGetErrorString(err));
        return;
    }

    if (device_count == 0) {
        spdlog::warn("No CUDA devices detected.");
        return;
    }

    spdlog::info("Detected {} CUDA device(s).", device_count);
    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) {
            spdlog::error("cudaGetDeviceProperties failed for device {}: {}", device,
                          cudaGetErrorString(err));
            continue;
        }

        const size_t mem_mb = static_cast<size_t>(prop.totalGlobalMem / (1024 * 1024));
        const int clock_mhz = prop.clockRate / 1000;
        spdlog::info("GPU {}: {} (SM {}.{}, {} MB, {} MHz, {} SMs)", device, prop.name, prop.major,
                     prop.minor, mem_mb, clock_mhz, prop.multiProcessorCount);
    }
}

inline bool has_mpi_launcher_env() {
    const char* env_vars[] = {
        "PMI_RANK",
        "PMI_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "MV2_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_PROCID",
        "SLURM_NTASKS",
    };
    for (const char* var : env_vars) {
        if (std::getenv(var) != nullptr) {
            return true;
        }
    }
    return false;
}

inline std::string get_hostname_string() {
    char hostname[256] = {};
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
    return std::string("unknown");
}

inline MpiCudaEnv init_mpi_and_bind_gpu(int* argc, char*** argv) {
    MpiCudaEnv env{};

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    const bool launched_with_mpi = has_mpi_launcher_env();
    if (!mpi_initialized && launched_with_mpi) {
        MPI_Init(argc, argv);
        env.mpi_initialized_here = true;
    }

    if (mpi_initialized || launched_with_mpi) {
        MPI_Comm_rank(MPI_COMM_WORLD, &env.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &env.size);

        int name_len = 0;
        char node_name[MPI_MAX_PROCESSOR_NAME] = {};
        MPI_Get_processor_name(node_name, &name_len);
        env.node_name.assign(node_name, name_len);

        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, env.rank, MPI_INFO_NULL,
                            &local_comm);
        MPI_Comm_rank(local_comm, &env.local_rank);
        MPI_Comm_size(local_comm, &env.local_size);
        MPI_Comm_free(&local_comm);
    } else {
        env.node_name = get_hostname_string();
        spdlog::warn("MPI launcher not detected; running in single-process mode.");
    }

    cudaError_t err = cudaGetDeviceCount(&env.device_count);
    if (err != cudaSuccess) {
        spdlog::error("cudaGetDeviceCount failed: {}", cudaGetErrorString(err));
        return env;
    }
    if (env.device_count == 0) {
        spdlog::error("No CUDA devices detected on node {}.", env.node_name);
        return env;
    }

    env.device_id = env.local_rank % env.device_count;
    err = cudaSetDevice(env.device_id);
    if (err != cudaSuccess) {
        spdlog::error("cudaSetDevice({}) failed on node {}: {}", env.device_id, env.node_name,
                      cudaGetErrorString(err));
        return env;
    }
    err = cudaFree(0);
    if (err != cudaSuccess) {
        spdlog::error("cudaFree(0) failed on node {}: {}", env.node_name,
                      cudaGetErrorString(err));
        return env;
    }

    spdlog::info("node {} rank {}/{} local {}/{} -> device {}/{}",
                 env.node_name,
                 env.rank,
                 env.size,
                 env.local_rank,
                 env.local_size,
                 env.device_id,
                 env.device_count);
    return env;
}

inline void finalize_mpi_if_needed(const MpiCudaEnv& env) {
    if (!env.mpi_initialized_here) {
        return;
    }
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        MPI_Finalize();
    }
}
