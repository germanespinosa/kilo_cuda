### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 
    )

# Specify include directories
include_directories(src)
	
# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    kilo_cuda
    src/main.cpp
    src/robot.cpp
    src/util.cpp
	src/kilo_kernels.cu
    )

