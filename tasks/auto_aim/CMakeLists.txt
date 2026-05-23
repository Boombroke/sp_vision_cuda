cmake_minimum_required(VERSION 3.16)

# 后端来源说明：
#   - 传统识别 detector + classifier、OpenVINO 系 yolov5/v8/11、多线程 mt_detector
#     都依赖 OpenVINO，统一 gate 在 BOF_WITH_OPENVINO 下。
#   - yolov5_trt 是 TRT 路径，gate 在 BOF_WITH_INFER 下。
#   - tracker/target/solver/armor/aimer/hero_aimer/shooter/planner 与后端无关，全平台必带。

add_subdirectory(planner/tinympc)

# 全平台共用源
set(AUTO_AIM_SRCS
    armor.cpp
    solver.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    aimer.cpp
    hero_aimer.cpp
    shooter.cpp
    yolo.cpp
    multithread/commandgener.cpp
    planner/planner.cpp
)

# OpenVINO 后端：ONNX 分类器 + OpenVINO yolo + 多线程 OpenVINO 推理 + 传统识别 fallback
if(BOF_WITH_OPENVINO)
  list(APPEND AUTO_AIM_SRCS
      classifier.cpp
      detector.cpp
      yolos/yolov5.cpp
      yolos/yolov8.cpp
      yolos/yolo11.cpp
      multithread/mt_detector.cpp
  )
endif()

# TensorRT 后端
if(BOF_WITH_INFER)
  list(APPEND AUTO_AIM_SRCS yolos/yolov5_trt.cpp)
endif()

add_library(auto_aim OBJECT ${AUTO_AIM_SRCS})
target_link_libraries(auto_aim PUBLIC io tinympcstatic)

if(BOF_WITH_OPENVINO)
  target_compile_definitions(auto_aim PUBLIC BOF_WITH_OPENVINO=1)
  target_link_libraries(auto_aim PUBLIC openvino::runtime)
endif()

if(BOF_WITH_INFER)
  target_compile_definitions(auto_aim PUBLIC BOF_WITH_INFER=1)
  target_link_libraries(auto_aim PUBLIC infer)
endif()
