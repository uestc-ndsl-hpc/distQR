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
target_link_libraries(test_dist_blocked_qr_col_blockcyclic_correctness PRIVATE GTest::gtest CUDA::cublas CUDA::curand CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL)
