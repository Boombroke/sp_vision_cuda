#ifndef __YOLO_HPP__
#define __YOLO_HPP__

#include <NvInfer.h>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "BaseInfer.hpp"
#include "InferTool.hpp"
#include "logging.h"

using namespace nvinfer1;

static tdt_yolov5_predictor::Logger gLogger;

namespace yolo {

enum class Type : int {
    V5 = 0,
    X = 1,
    V3 = 2,
    V7 = 3,
    V8 = 5,
    V8Seg = 6,
    V5Face = 7,
    /** RP 装甲 YOLOv5 头（与 tasks/auto_aim OpenVINO 版同一解码语义；原 0708） */
    RP = 8
};

struct InstanceSegmentMap {
    int            width = 0, height = 0;
    unsigned char* data = nullptr;

    InstanceSegmentMap(int width, int height);
    virtual ~InstanceSegmentMap();
};

struct Box {
    float left, top, right, bottom, confidence;
    int   class_label;
    /** Type::RP 时为 0–3（与 OpenVINO parse 颜色 argmax 一致）；否则为 -1 */
    int   color_id = -1;
    std::shared_ptr<InstanceSegmentMap> seg;
    std::vector<float>                  points;
    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence,
        int class_label)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          confidence(confidence),
          class_label(class_label)
    {
    }
    Box(float left, float top, float right, float bottom, float confidence,
        int class_label, std::vector<float> points, int color_id = -1)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          confidence(confidence),
          class_label(class_label),
          color_id(color_id),
          points(std::move(points))
    {
    }
};

typedef std::vector<Box> BoxArray;

/**
 * 加载 YOLO TensorRT 引擎。
 * @param type 模型类型；Type::RP 为左上角 letterbox+黑边、22 维输出（8 关键点+logit+4 色+9 类），需匹配导出的 engine；结果中 `Box.color_id` 为 0–3，不做颜色过滤。
 */
std::shared_ptr<bof_vision::Infer<BoxArray>>
load(const std::string& engine_file, Type type,
     float confidence_threshold = 0.45f, float nms_threshold = 0.6f);

const char*                           type_name(Type type);
std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
};  // namespace yolo

#endif  // __YOLO_HPP__