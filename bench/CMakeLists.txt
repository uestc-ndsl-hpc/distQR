add_executable(bench_tsqr bench_tsqr.cu)
target_include_directories(bench_tsqr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_tsqr PRIVATE CUDA::cublas CUDA::cusolver CUDA::curand CUDA::cudart)

add_executable(bench_qr bench_qr.cu)
target_include_directories(bench_qr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_qr PRIVATE CUDA::cublas CUDA::cusolver CUDA::curand CUDA::cudart spdlog::spdlog)

add_executable(bench_dist_blocked_qr bench_dist_blocked_qr.cu)
target_include_directories(bench_dist_blocked_qr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_dist_blocked_qr PRIVATE CUDA::cublas CUDA::curand CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL)

add_executable(bench_dist_blocked_qr_col bench_dist_blocked_qr_col.cu)
target_include_directories(bench_dist_blocked_qr_col PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_dist_blocked_qr_col PRIVATE CUDA::cublas CUDA::curand CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL)

add_executable(bench_dist_blocked_qr_col_blockcyclic bench_dist_blocked_qr_col_blockcyclic.cu)
target_include_directories(bench_dist_blocked_qr_col_blockcyclic PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_dist_blocked_qr_col_blockcyclic PRIVATE CUDA::cublas CUDA::curand CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL distqr_nvtx_headers)

add_executable(bench_dist_orgqr_col_blockcyclic bench_dist_orgqr_col_blockcyclic.cu)
target_include_directories(bench_dist_orgqr_col_blockcyclic PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_dist_orgqr_col_blockcyclic PRIVATE CUDA::cublas CUDA::curand CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL distqr_nvtx_headers)

if(CUDAToolkit_FOUND)
  get_filename_component(_CUDA_ROOT_FROM_BIN "${CUDAToolkit_BIN_DIR}" DIRECTORY)
endif()

set(_CUSOLVERMP_ROOT_HINTS $ENV{CUSOLVERMP_ROOT} $ENV{CUSOLVERMP_HOME} $ENV{CUSOLVERMP_DIR})

find_path(CUSOLVERMP_INCLUDE_DIR
  NAMES cusolverMp.h
  HINTS
    ${CUDAToolkit_INCLUDE_DIRS}
    ${_CUDA_ROOT_FROM_BIN}
    ${_CUSOLVERMP_ROOT_HINTS}
  PATH_SUFFIXES include targets/x86_64-linux/include)

find_library(CUSOLVERMP_LIBRARY
  NAMES cusolverMp
  HINTS
    ${CUDAToolkit_LIBRARY_DIR}
    ${CUDAToolkit_LIBRARY_ROOT}
    ${_CUDA_ROOT_FROM_BIN}
    ${_CUSOLVERMP_ROOT_HINTS}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)

if(CUSOLVERMP_INCLUDE_DIR AND CUSOLVERMP_LIBRARY)
  add_executable(bench_cusolvermp_geqrf bench_cusolvermp_geqrf.cu)
  target_include_directories(bench_cusolvermp_geqrf PRIVATE ${CMAKE_SOURCE_DIR}/include ${CUSOLVERMP_INCLUDE_DIR})
  target_link_libraries(bench_cusolvermp_geqrf PRIVATE
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
    CUDA::cudart
    MPI::MPI_CXX
    spdlog::spdlog
    NCCL::NCCL
    ${CUSOLVERMP_LIBRARY})
else()
  message(STATUS "Skip bench_cusolvermp_geqrf: cuSOLVERMp not found (need cusolverMp.h and libcusolverMp).")
endif()
