#include <spdlog/spdlog.h>

#include "components/resourse_initial.cuh"

int main() {
    spdlog::set_level(spdlog::level::info);
    init();
    return 0;
}
