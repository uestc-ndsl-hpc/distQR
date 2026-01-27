add_executable(bench_tsqr bench_tsqr.cu)
target_include_directories(bench_tsqr PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_tsqr PRIVATE CUDA::cublas CUDA::cusolver CUDA::curand CUDA::cudart)
