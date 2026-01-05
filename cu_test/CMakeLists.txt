cmake_minimum_required(VERSION 3.10)
project(infer_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

find_package(CUDA REQUIRED)
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)

include(${PROJECT_SOURCE_DIR}/cmake/modules/libnvinferConfig.cmake)
message("Using Cuda Version: ${CUDA_VERSION} and TensorRT Version: ${nvinfer_VERSION}")

find_package(OpenCV REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${nvinfer_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/infer/include
)

add_subdirectory(infer)

add_executable(main_infer main.cpp)
target_link_libraries(main_infer
    infer
    ${nvinfer_LIBRARY}
    cudart
    nvonnxparser
    ${OpenCV_LIBS}
)
