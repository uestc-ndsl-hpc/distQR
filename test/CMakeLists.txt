add_executable(test_spdlog test_spdlog.cu)
target_link_libraries(test_spdlog PRIVATE spdlog::spdlog)

add_executable(test_matrix_helper test_matrix_helper.cu)
target_include_directories(test_matrix_helper PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_matrix_helper PRIVATE GTest::gtest_main CUDA::curand CUDA::cudart spdlog::spdlog)

add_executable(test_tsqr test_tsqr.cu)
target_include_directories(test_tsqr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_tsqr PRIVATE GTest::gtest_main CUDA::cublas CUDA::curand CUDA::cudart)
