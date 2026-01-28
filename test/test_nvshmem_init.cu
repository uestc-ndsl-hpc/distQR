#include <spdlog/spdlog.h>

#include "components/resourse_initial.cuh"

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::info);
    auto env = init_mpi_and_bind_gpu(&argc, &argv);
    init(&env);
    finalize_mpi_if_needed(env);
    return 0;
}
