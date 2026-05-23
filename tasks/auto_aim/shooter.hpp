#ifndef AUTO_AIM__SHOOTER_HPP
#define AUTO_AIM__SHOOTER_HPP

#include <deque>
#include <string>

#include "io/command.hpp"
#include "tasks/auto_aim/aimer.hpp"

namespace auto_aim
{
class Shooter
{
public:
  Shooter(const std::string & config_path);

  bool shoot(
    const io::Command & command, const auto_aim::Aimer & aimer,
    const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos);

private:
  io::Command last_command_;
  double judge_distance_;
  double first_tolerance_;
  double second_tolerance_;
  bool auto_fire_;

  // cmd 稳态闸门：连续 N 帧 cmd 最大变化都小于阈值才允许开火
  // 用来抑制高速小陀螺时 cmd 9° 假跳变带来的喷水
  int cmd_stable_window_;
  double cmd_stable_thresh_;  // rad
  std::deque<double> cmd_yaw_history_;
};
}  // namespace auto_aim

#endif  // AUTO_AIM__SHOOTER_HPP