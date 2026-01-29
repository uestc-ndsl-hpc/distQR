cmake_minimum_required(VERSION 3.20)

project(distQR LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(third_party/spdlog)
add_subdirectory(third_party/googletest)

# CUDA
find_package(CUDAToolkit REQUIRED)
# MPI
find_package(MPI REQUIRED)
# NCCL
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(NCCL REQUIRED)

# Test
add_subdirectory(test)

# Bench
add_subdirectory(bench)
