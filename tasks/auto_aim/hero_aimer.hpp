#ifndef AUTO_AIM__HERO_AIMER_HPP
#define AUTO_AIM__HERO_AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <deque>
#include <list>
#include <string>

#include "io/command.hpp"
#include "tasks/auto_aim/target.hpp"

namespace auto_aim
{

struct HeroAimPoint
{
  bool valid;
  Eigen::Vector4d xyza;
};

class HeroAimer
{
public:
  explicit HeroAimer(const std::string & config_path);
  ~HeroAimer() = default;

  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true, Eigen::Vector3d ypr = Eigen::Vector3d::Zero());

  Eigen::Vector3d get_outpost_hit_xyz(const Target & target) const;
  Eigen::Vector3d get_top_front(const Target & target_origin) const;

  HeroAimPoint debug_aim_point_;

  /** 与 aim() 中「转速过大瞄中心」分支一致，单位 rad/s */
  double center_mode_w_threshold() const { return center_mode_w_threshold_; }

private:
  double center_mode_w_threshold_;
  double shoot_delay_;
  double yaw_offset_;
  double pitch_offset_;
  double max_fire_error_yaw_;
  double max_fire_error_pitch_;
  double max_shoot_angle_;

  // 慢云台专用：响应时间常数 + cmd 低通 + 开火窗口 + 稳态判定
  double gimbal_yaw_tau_;
  double gimbal_pitch_tau_;
  double cmd_smooth_tau_;
  double fire_pos_window_;       // rad, 板位置进瞄准方向的角度窗
  double fire_facing_window_;    // rad, 板法线偏离视线的最大角（防侧倾）
  int fire_steady_frames_;

  // 内部状态
  bool cmd_lp_init_ = false;
  double yaw_cmd_lp_ = 0;
  double pitch_cmd_lp_ = 0;
  std::chrono::steady_clock::time_point last_cmd_time_;
  std::deque<double> yaw_err_hist_;
  std::deque<double> pitch_err_hist_;

  HeroAimPoint choose_aim_point(const Target & target);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__HERO_AIMER_HPP
