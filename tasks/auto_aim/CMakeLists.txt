cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_subdirectory(planner/tinympc)

set(AUTO_AIM_SRCS
    armor.cpp
    classifier.cpp
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

if(BOF_WITH_INFER)
  list(APPEND AUTO_AIM_SRCS yolos/yolov5_trt.cpp)
endif()

add_library(auto_aim OBJECT ${AUTO_AIM_SRCS})

if(BOF_WITH_INFER)
  target_compile_definitions(auto_aim PUBLIC BOF_WITH_INFER=1)
  target_link_libraries(auto_aim PUBLIC infer)
endif()

target_link_libraries(auto_aim PUBLIC io openvino::runtime tinympcstatic)
