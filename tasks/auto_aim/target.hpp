#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

// 瞄准的目标
class Target
{
public:
  ArmorName name; // 装甲板名字
  ArmorType armor_type;// 装甲板类型
  ArmorPriority priority;// 装甲板优先级
  bool jumped;// 是否跳变
  int last_id;  // debug only

  Target() = default;

  // 构造函数
  // 识别到的装甲板
  // 时间点
  // 半径
  // 装甲板数量
  // 预测量协方差
  Target(
    const Armor & armor, 
    std::chrono::steady_clock::time_point t, 
    double radius, 
    int armor_num,
    Eigen::VectorXd P0_dig);


  // 这个好像没有用到
  Target(double x, double vyaw, double radius, double h);

  //预测 此刻
  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  
  // 更新 
  void update(const Armor & armor);

  // 外部调用的接口
  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;// 装甲板数量
  int switch_count_;// 切换次数
  int update_count_;// 更新次数

  bool is_switch_, is_converged_;// 是否切换, 是否收敛

  tools::ExtendedKalmanFilter ekf_;// 扩展卡尔曼滤波器
  std::chrono::steady_clock::time_point t_;// 时间点

  //更新装甲板状态
  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  // 推断出 此外的装甲板中心的状态 x y 
  // x是车子中心的状态 ekf_.x
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  // 
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP