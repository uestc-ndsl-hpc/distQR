#include <spdlog/spdlog.h>

#include "components/resourse_initial.cuh"

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::info);
    auto env = init_mpi_and_bind_gpu(&argc, &argv);
    init(&env);
    init_nccl_comm(&env);
    finalize_nccl_if_needed(&env);
    finalize_mpi_if_needed(env);
    return 0;
}
