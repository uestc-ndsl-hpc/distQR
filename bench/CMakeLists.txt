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
target_link_libraries(bench_dist_blocked_qr_col_blockcyclic PRIVATE CUDA::cublas CUDA::curand CUDA::cudart MPI::MPI_CXX spdlog::spdlog NCCL::NCCL)
