#ifndef AUTO_AIM__ARMOR_HPP
#define AUTO_AIM__ARMOR_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_aim
{
// 颜色 枚举
// 红 蓝 熄灭 紫色（基地专用）
enum Color
{
  red,
  blue,
  extinguish,
  purple
};
const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};

// 装甲板类型 枚举
// 大 小
enum ArmorType
{
  big,
  small,
  building
};
const std::vector<std::string> ARMOR_TYPES = {"big", "small", "building"};


// 装甲板 名字 枚举
// 一 二 三 四 五 哨兵 前哨站 基地 非装甲
enum ArmorName
{
  one,
  two,
  three,
  four,
  five,
  sentry,
  outpost,
  base,
  not_armor
};

const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                              "sentry", "outpost", "base",  "not_armor"};

// 装甲板 优先级 枚举                                              
enum ArmorPriority
{
  first = 1,
  second,
  third,
  forth,
  fifth
};

// 装甲板属性 列表
// clang-format off
const std::vector<std::tuple<Color, ArmorName, ArmorType>> armor_properties = {
  {blue, sentry, small},     {red, sentry, small},     {extinguish, sentry, small},
  {blue, one, small},        {red, one, small},        {extinguish, one, small},
  {blue, two, small},        {red, two, small},        {extinguish, two, small},
  {blue, three, small},      {red, three, small},      {extinguish, three, small},
  {blue, four, small},       {red, four, small},       {extinguish, four, small},
  {blue, five, small},       {red, five, small},       {extinguish, five, small},
  {blue, outpost, building}, {red, outpost, building}, {extinguish, outpost, building},
  {blue, base, building},    {red, base, building},    {extinguish, base, building},  {purple, base, building},
  {blue, base, building},    {red, base, building},    {extinguish, base, building},  {purple, base, building},
  {blue, three, big},        {red, three, big},        {extinguish, three, big}, 
  {blue, four, big},         {red, four, big},         {extinguish, four, big},  
  {blue, five, big},         {red, five, big},         {extinguish, five, big}};
// clang-format on

// 灯条 结构体
// id 颜色 中心 顶点 底点 顶到底向量 四个角点 角度 角度误差 长度 宽度 比值 旋转矩形
struct Lightbar
{
  std::size_t id;
  Color color;
  cv::Point2f center, top, bottom, top2bottom;
  std::vector<cv::Point2f> points;
  double angle, angle_error, length, width, ratio;
  cv::RotatedRect rotated_rect;

  Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id);
  Lightbar() {};
};


// 装甲板 结构体
// 颜色 左灯条 右灯条 中心 归一化坐标 点 集合
// 两灯条中点连线与长灯条长度之比 长灯条与短灯条长度之比
// 灯条和中点连线所成夹角与π/2的差
// 类型 名字 优先级 类别id 包围盒 图案 置信度 是否重复
// 相机坐标系xyz 世界坐标系xyz 云台坐标系ypr 世界坐标系ypr 世界坐标系ypd
// 原始偏航角
struct Armor
{
  Color color;
  Lightbar left, right;     //used to be const
  cv::Point2f center;       // 不是对角线交点，不能作为实际中心！
  cv::Point2f center_norm;  // 归一化坐标
  std::vector<cv::Point2f> points;

  double ratio;              // 两灯条的中点连线与长灯条的长度之比
  double side_ratio;         // 长灯条与短灯条的长度之比
  double rectangular_error;  // 灯条和中点连线所成夹角与π/2的差值

  ArmorType type;
  ArmorName name;
  ArmorPriority priority;
  int class_id;
  cv::Rect box;
  cv::Mat pattern;
  double confidence;
  bool duplicated;

  Eigen::Vector3d xyz_in_gimbal;  // 单位：m
  Eigen::Vector3d xyz_in_world;   // 单位：m
  Eigen::Vector3d ypr_in_gimbal;  // 单位：rad
  Eigen::Vector3d ypr_in_world;   // 单位：rad
  Eigen::Vector3d ypd_in_world;   // 球坐标系

  double yaw_raw;  // rad

  Armor(const Lightbar & left, const Lightbar & right);
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints);
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints,
    cv::Point2f offset);
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints);
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_HPP