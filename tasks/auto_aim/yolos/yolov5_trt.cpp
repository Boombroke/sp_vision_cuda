#include "yolov5_trt.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <stdexcept>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
YOLOV5_TRT::YOLOV5_TRT(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  const std::string engine_path = yaml["yolov5_trt_model_path"].as<std::string>();
  if (yaml["yolov5_trt_confidence"])
    trt_confidence_ = yaml["yolov5_trt_confidence"].as<float>();
  if (yaml["yolov5_trt_nms"])
    trt_nms_ = yaml["yolov5_trt_nms"].as<float>();

  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f((float)x, (float)y);

  infer_ = yolo::load(engine_path, yolo::Type::RP, trt_confidence_, trt_nms_);
  if (!infer_) {
    throw std::runtime_error("YOLOV5_TRT: load TensorRT engine failed: " + engine_path);
  }

  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    throw std::runtime_error("YOLOV5_TRT: cudaStreamCreate failed");
  }
}

YOLOV5_TRT::~YOLOV5_TRT()
{
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

std::list<Armor> YOLOV5_TRT::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    cv::Rect roi = roi_;
    if (roi.width == -1)
      roi.width = raw_img.cols;
    if (roi.height == -1)
      roi.height = raw_img.rows;
    bgr_img = raw_img(roi);
  } else {
    bgr_img = raw_img;
  }

  cv::Mat infer_in = bgr_img;
  if (!infer_in.isContinuous())
    infer_in = infer_in.clone();

  bof_vision::Image img(infer_in.data, infer_in.cols, infer_in.rows);
  yolo::BoxArray     boxes = infer_->forward(img, stream_);

  std::list<Armor> armors;
  for (const auto & b : boxes) {
    if (b.points.size() < 8)
      continue;

    std::vector<cv::Point2f> kpts;
    kpts.reserve(4);
    for (size_t k = 0; k < 8; k += 2)
      kpts.emplace_back(b.points[k], b.points[k + 1]);

    const int bl = (int)b.left;
    const int bt = (int)b.top;
    const int bw = std::max(1, (int)(b.right - b.left));
    const int bh = std::max(1, (int)(b.bottom - b.top));
    cv::Rect  rect(bl, bt, bw, bh);

    if (use_roi_) {
      armors.emplace_back(
        b.color_id, b.class_label, b.confidence, rect, kpts, offset_);
    } else {
      armors.emplace_back(b.color_id, b.class_label, b.confidence, rect, kpts);
    }
  }

  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }
    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    if (use_traditional_)
      detector_.detect(*it, raw_img);

    it->center_norm = get_center_norm(raw_img, it->center);
    ++it;
  }

  if (debug_)
    draw_detections(raw_img, armors, frame_count);

  return armors;
}

std::list<Armor> YOLOV5_TRT::postprocess(
  double /*scale*/, cv::Mat & /*output*/, const cv::Mat & /*bgr_img*/, int /*frame_count*/)
{
  tools::logger()->warn(
    "YOLOV5_TRT::postprocess is not supported; use detect() or OpenVINO yolo for async pipeline.");
  return {};
}

bool YOLOV5_TRT::check_name(const Armor & armor) const
{
  const bool name_ok = armor.name != ArmorName::not_armor;
  const bool confidence_ok = armor.confidence > min_confidence_;
  return name_ok && confidence_ok;
}

bool YOLOV5_TRT::check_type(const Armor & armor) const
{
  const bool name_ok = (armor.type == ArmorType::small)
                         ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                         : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                            armor.name != ArmorName::outpost);
  return name_ok;
}

cv::Point2f YOLOV5_TRT::get_center_norm(const cv::Mat & img, const cv::Point2f & center) const
{
  const float h = (float)img.rows;
  const float w = (float)img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5_TRT::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  cv::Mat detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection_trt", detection);
}

}  // namespace auto_aim
