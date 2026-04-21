add_executable(test_spdlog test_spdlog.cu)
target_link_libraries(test_spdlog PRIVATE spdlog::spdlog)

add_executable(test_matrix_helper test_matrix_helper.cu)
target_include_directories(test_matrix_helper PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_matrix_helper PRIVATE GTest::gtest_main CUDA::curand CUDA::cudart spdlog::spdlog)

add_executable(test_tsqr test_tsqr.cu)
target_include_directories(test_tsqr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_tsqr PRIVATE GTest::gtest_main CUDA::cublas CUDA::curand CUDA::cudart)

add_executable(test_tsqr_wy test_tsqr_wy.cu)
target_include_directories(test_tsqr_wy PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_tsqr_wy PRIVATE GTest::gtest_main CUDA::cublas CUDA::curand CUDA::cudart)

add_executable(test_nccl_init test_nccl_init.cu)
target_include_directories(test_nccl_init PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_nccl_init PRIVATE CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL)

add_executable(test_single_card_qr test_single_card_qr.cu)
target_link_libraries(test_single_card_qr PRIVATE GTest::gtest_main CUDA::cublas CUDA::cusolver CUDA::curand CUDA::cudart spdlog::spdlog)
target_include_directories(test_single_card_qr PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_dist_blocked_qr_correctness test_dist_blocked_qr_correctness.cu)
target_include_directories(test_dist_blocked_qr_correctness PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_dist_blocked_qr_correctness PRIVATE GTest::gtest CUDA::cublas CUDA::curand CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL)

add_executable(test_dist_blocked_qr_col_correctness test_dist_blocked_qr_col_correctness.cu)
target_include_directories(test_dist_blocked_qr_col_correctness PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_dist_blocked_qr_col_correctness PRIVATE GTest::gtest CUDA::cublas CUDA::curand CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL)

add_executable(test_dist_blocked_qr_col_blockcyclic_correctness test_dist_blocked_qr_col_blockcyclic_correctness.cu)
target_include_directories(test_dist_blocked_qr_col_blockcyclic_correctness PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_dist_blocked_qr_col_blockcyclic_correctness PRIVATE GTest::gtest CUDA::cublas CUDA::curand CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL distqr_nvtx_headers)

set(_CUSOLVERMP_ROOT_HINTS $ENV{CUSOLVERMP_ROOT} $ENV{CUSOLVERMP_HOME} $ENV{CUSOLVERMP_DIR})

find_path(CUSOLVERMP_INCLUDE_DIR
  NAMES cusolverMp.h
  HINTS
    ${CUDAToolkit_INCLUDE_DIRS}
    ${_DISTQR_CUDA_ROOT_FROM_BIN}
    ${_CUSOLVERMP_ROOT_HINTS}
  PATH_SUFFIXES include targets/x86_64-linux/include)

find_library(CUSOLVERMP_LIBRARY
  NAMES cusolverMp
  HINTS
    ${CUDAToolkit_LIBRARY_DIR}
    ${CUDAToolkit_LIBRARY_ROOT}
    ${_DISTQR_CUDA_ROOT_FROM_BIN}
    ${_CUSOLVERMP_ROOT_HINTS}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)

if(CUSOLVERMP_INCLUDE_DIR AND CUSOLVERMP_LIBRARY)
  add_executable(test_cusolvermp_geqrf_ormqr test_cusolvermp_geqrf_ormqr.cu)
  target_include_directories(test_cusolvermp_geqrf_ormqr PRIVATE ${CMAKE_SOURCE_DIR}/include ${CUSOLVERMP_INCLUDE_DIR})
  target_link_libraries(test_cusolvermp_geqrf_ormqr PRIVATE
    GTest::gtest
    CUDA::cudart
    MPI::MPI_CXX
    spdlog::spdlog
    NCCL::NCCL
    ${CUSOLVERMP_LIBRARY})
else()
  message(STATUS "Skip test_cusolvermp_geqrf_ormqr: cuSOLVERMp not found (need cusolverMp.h and libcusolverMp).")
endif()
