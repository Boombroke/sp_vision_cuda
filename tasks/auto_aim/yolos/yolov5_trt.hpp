#ifndef AUTO_AIM__YOLOV5_TRT_HPP
#define AUTO_AIM__YOLOV5_TRT_HPP

#include <cuda_runtime.h>
#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

#include "BaseInfer.hpp"
#include "yolos.hpp"

namespace auto_aim
{
/**
 * TensorRT 装甲检测（infer::yolo::Type::RP），与 tasks/auto_aim/yolos/yolov5.cpp 流程对齐：
 * ROI、Armor 构造、check_name/type、传统 refine、debug 绘制。
 * 配置：yolo_name: yolov5_trt，yolov5_trt_model_path 指向 .engine（RP 头，与 0708 等一致）。
 */
class YOLOV5_TRT : public YOLOBase
{
public:
  YOLOV5_TRT(const std::string & config_path, bool debug);
  ~YOLOV5_TRT();

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  bool debug_, use_roi_, use_traditional_;
  double min_confidence_;

  float trt_confidence_{0.45f};
  float trt_nms_{0.6f};

  std::shared_ptr<bof_vision::Infer<yolo::BoxArray>> infer_;
  cudaStream_t                                      stream_{};

  cv::Rect    roi_;
  cv::Point2f offset_;
  Detector    detector_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & img, const cv::Point2f & center) const;

  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV5_TRT_HPP
