# infer：TensorRT + CUDA 共享库（libinfer.so，见下方 SHARED 说明）
#
# 仅当工程根目录 CMake 开启 BOF_WITH_INFER 并 add_subdirectory(infer) 时本文件才会执行；
# 单独在 infer/ 目录下 cmake -B build 会因无顶层 CMakeLists.txt 而失败。
#
# 相对旧版：CMake 3.18+、CUDA_SEPARABLE_COMPILATION、链 CUDA::cudart 等。
# 临时视频测试：-DINFER_BUILD_VIDEO_TEST=ON

cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0079 NEW)

# 在 project() 之前必须找到 nvcc。若 CMakeCache 里 CMAKE_CUDA_COMPILER 指向已删除的路径，
# 仍会进入下方逻辑并 FORCE 覆盖。
if(NOT CMAKE_CUDA_COMPILER OR NOT EXISTS "${CMAKE_CUDA_COMPILER}")
  foreach(
    _nvcc
    "$ENV{CUDACXX}"
    "$ENV{CUDA_HOME}/bin/nvcc"
    "$ENV{CUDA_PATH}/bin/nvcc"
    /usr/local/cuda/bin/nvcc
    /usr/lib/cuda/bin/nvcc)
    if(_nvcc AND EXISTS "${_nvcc}")
      set(CMAKE_CUDA_COMPILER "${_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
      break()
    endif()
  endforeach()
endif()
# PATH 里的 nvcc（上面列表未覆盖时）
if(NOT CMAKE_CUDA_COMPILER OR NOT EXISTS "${CMAKE_CUDA_COMPILER}")
  find_program(_infer_nvcc nvcc)
  if(_infer_nvcc)
    set(CMAKE_CUDA_COMPILER "${_infer_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
endif()
if(NOT CMAKE_CUDA_COMPILER OR NOT EXISTS "${CMAKE_CUDA_COMPILER}")
  message(
    FATAL_ERROR
      "infer: 未找到 CUDA 编译器 nvcc。\n"
      "  - 确认已安装 CUDA Toolkit；或 export CUDACXX=/path/to/nvcc\n"
      "  - 或 cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ...\n"
      "  - 若曾配置失败，删掉 build 目录后重新 cmake（避免 CACHE 里残留错误路径）\n"
      "  - 根目录需 -DBOF_WITH_INFER=ON，否则不会进入 infer 子目录")
endif()
message(STATUS "infer: CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")

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
# 使用 SHARED：CUDA_SEPARABLE_COMPILATION 下静态库 .a 不会在归档时做 device link，
# 纯 C++ 可执行文件链接时出现 __cudaRegisterLinkedBinary_* 未定义；共享库在链 libinfer.so 时会完成 device link。
add_library(infer SHARED ${INFER_SRCS})
set_target_properties(infer PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
