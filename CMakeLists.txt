cmake_minimum_required(VERSION 3.20)

project(distQR LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(third_party/spdlog)
add_subdirectory(third_party/googletest)

# CUDA
find_package(CUDAToolkit REQUIRED)

add_library(distqr_nvtx_headers INTERFACE)
if(CUDAToolkit_FOUND)
  get_filename_component(_DISTQR_CUDA_ROOT_FROM_BIN "${CUDAToolkit_BIN_DIR}" DIRECTORY)
endif()

find_path(DISTQR_NVTX_INCLUDE_DIR
  NAMES nvtx3/nvToolsExt.h nvToolsExt.h
  HINTS
    ${CUDAToolkit_INCLUDE_DIRS}
    ${_DISTQR_CUDA_ROOT_FROM_BIN}
    $ENV{CONDA_PREFIX}
    /usr/local/cuda
    /usr/local/cuda-12.8
  PATH_SUFFIXES include targets/x86_64-linux/include)

if(DISTQR_NVTX_INCLUDE_DIR)
  target_include_directories(distqr_nvtx_headers INTERFACE ${DISTQR_NVTX_INCLUDE_DIR})
  message(STATUS "Found NVTX headers: ${DISTQR_NVTX_INCLUDE_DIR}")
else()
  message(STATUS "NVTX headers not found; custom NVTX ranges will be disabled.")
endif()

# MPI
find_package(MPI REQUIRED)
# NCCL
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(NCCL REQUIRED)

# Test
add_subdirectory(test)

# Bench
add_subdirectory(bench)
