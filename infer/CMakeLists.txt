# infer 静态库（TensorRT + CUDA）
#
# 相对旧版的主要差异（为把 .cu 真正编进库，否则往往只有 BaseInfer.cpp）：
#   - CMake 3.18+、project(... LANGUAGES CXX CUDA)
#   - find_package(CUDAToolkit) + target_link_libraries(... CUDA::cudart)
#   - CUDA_SEPARABLE_COMPILATION 与 infer 内多文件 CUDA 一致
#
# 临时视频测试：默认关闭，需要时 -DINFER_BUILD_VIDEO_TEST=ON；后续会删并接入 auto_aim。

cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0079 NEW)

project(infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

find_package(CUDAToolkit REQUIRED)
include(../cmake/modules/libnvinferConfig.cmake)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE INFER_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_library(infer ${INFER_SRCS})
target_include_directories(infer PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${nvinfer_INCLUDE_DIRS}
)
target_link_libraries(infer PUBLIC
    ${nvinfer_LIBRARY}
    CUDA::cudart
    nvonnxparser
    ${OpenCV_LIBS}
)
set_property(TARGET infer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

option(INFER_BUILD_VIDEO_TEST "临时：编译 test/yolo_video_test（将迁入 auto_aim 后可删）" OFF)
if(INFER_BUILD_VIDEO_TEST)
    add_executable(yolo_video_test ${PROJECT_SOURCE_DIR}/test/yolo_video_test.cpp)
    target_link_libraries(yolo_video_test PRIVATE infer)
    if(UNIX AND NOT APPLE)
        target_link_libraries(yolo_video_test PRIVATE stdc++fs)
    endif()
endif()
