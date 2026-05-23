#include "shooter.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Shooter::Shooter(const std::string & config_path) : last_command_{false, false, 0, 0}
{
  auto yaml = YAML::LoadFile(config_path);
  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;    // degree to rad
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;  // degree to rad
  judge_distance_ = yaml["judge_distance"].as<double>();
  auto_fire_ = yaml["auto_fire"].as<bool>();
  cmd_stable_window_ = yaml["cmd_stable_window"].as<int>(5);
  cmd_stable_thresh_ = yaml["cmd_stable_thresh"].as<double>(4.0) / 57.3;  // degree to rad
}

bool Shooter::shoot(
  const io::Command & command, const auto_aim::Aimer & aimer,
  const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos)
{
  if (!command.control || targets.empty() || !auto_fire_) {
    cmd_yaw_history_.clear();  // 失控时清空历史，恢复后从头攒
    return false;
  }

  auto target_x = targets.front().ekf_x()[0];
  auto target_y = targets.front().ekf_x()[2];
  auto tolerance = std::sqrt(tools::square(target_x) + tools::square(target_y)) > judge_distance_
                     ? second_tolerance_
                     : first_tolerance_;

  // 维护 cmd_yaw 历史，看连续 N 帧 cmd 最大跳变是否都小于阈值
  cmd_yaw_history_.push_back(command.yaw);
  while ((int)cmd_yaw_history_.size() > cmd_stable_window_) cmd_yaw_history_.pop_front();
  bool cmd_stable = (int)cmd_yaw_history_.size() >= cmd_stable_window_;
  if (cmd_stable) {
    double max_jump = 0;
    for (size_t i = 1; i < cmd_yaw_history_.size(); ++i) {
      double d = std::abs(cmd_yaw_history_[i] - cmd_yaw_history_[i - 1]);
      if (d > max_jump) max_jump = d;
    }
    if (max_jump > cmd_stable_thresh_) cmd_stable = false;
  }

  if (
    std::abs(last_command_.yaw - command.yaw) < tolerance * 2 &&
    std::abs(gimbal_pos[0] - last_command_.yaw) < tolerance &&
    aimer.debug_aim_point.valid && cmd_stable) {
    last_command_ = command;
    return true;
  }

  last_command_ = command;
  return false;
}

}  // namespace auto_aim
