#include "hero_aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
HeroAimer::HeroAimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  shoot_delay_ = yaml["shoot_delay"].as<double>();
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
  max_fire_error_yaw_ = yaml["max_fire_error_yaw"].as<double>();
  max_fire_error_pitch_ = yaml["max_fire_error_pitch"].as<double>();
  max_shoot_angle_ = yaml["max_shoot_angle"].as<double>();
  center_mode_w_threshold_ = yaml["center_mode_w_threshold"].as<double>(8.0);
  gimbal_yaw_tau_ = yaml["gimbal_yaw_tau"].as<double>(0.15);
  gimbal_pitch_tau_ = yaml["gimbal_pitch_tau"].as<double>(0.20);
  cmd_smooth_tau_ = yaml["cmd_smooth_tau"].as<double>(0.30);
  fire_pos_window_ = yaml["fire_pos_window"].as<double>(0.05);
  fire_facing_window_ = yaml["fire_facing_window"].as<double>(0.40);
  fire_steady_frames_ = yaml["fire_steady_frames"].as<int>(5);
}

// 瞄准前哨站
// hit_point_xy 是在中心连线上的点 距离中心点的距离为ekf_x[8]
Eigen::Vector3d HeroAimer::get_outpost_hit_xyz(const Target & target) const
{
  auto ekf_x = target.ekf_x();
  Eigen::Vector3d center = {ekf_x[0], ekf_x[2], ekf_x[4]};
  const auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  Eigen::Vector2d center_xy = center.head(2);
  const double horizonal_distance = center_xy.norm() - ekf_x[8];
  Eigen::Vector2d hit_point_xy = (horizonal_distance / center_xy.norm()) * center_xy;

  double z = ekf_x[4];
  const auto armors = target.armor_xyza_list();
  if (!armors.empty()) {
    std::size_t best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < armors.size(); ++i) {
      const double d = std::abs(tools::limit_rad(armors[i][3] - center_yaw));
      if (d < best_d) {
        best_d = d;
        best = i;
      }
    }
    z = armors[best][2];
  }
  return {hit_point_xy[0], hit_point_xy[1], z};
}

// 瞄准车子的装甲板
Eigen::Vector3d HeroAimer::get_top_front(const Target & target_origin) const
{
  auto ekf_x = target_origin.ekf_x();

  Eigen::Vector3d center = {ekf_x[0], ekf_x[2], ekf_x[4]};
  const auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  Eigen::Vector2d center_xy = center.head(2);
  const double horizonal_distance = center_xy.norm() - ekf_x[8];
  Eigen::Vector2d hit_point_xy = (horizonal_distance / center_xy.norm()) * center_xy;

  const auto armors = target_origin.armor_xyza_list();
  if (armors.empty()) {
    return {hit_point_xy[0], hit_point_xy[1], ekf_x[4]};
  }

  const double w = ekf_x[7];
  if (std::fabs(w) < 1e-6) {
    return {hit_point_xy[0], hit_point_xy[1], ekf_x[4]};
  }

  // 找未来最先转到正前方（板法线 == 中心 LOS）那块板，pitch 跟它的 z 走
  // 这样云台 yaw 锁在车中心方向不动，只 pitch 缓慢调整高度
  const double T = 2 * M_PI / std::fabs(w);
  const int sig_w = w > 0 ? 1 : -1;
  std::size_t armor_id = 0;
  double dt_min = INFINITY;
  for (std::size_t k = 0; k < armors.size(); ++k) {
    const double ay = armors[k][3];
    double t;
    if (sig_w == 1) {
      // CCW: 板 yaw 在递增, 找正向到达 center_yaw 的最短时间
      const double diff = tools::limit_rad(center_yaw - ay);
      t = (diff >= 0) ? diff / w : (diff + 2 * M_PI) / w;
    } else {
      // CW: 板 yaw 在递减
      const double diff = tools::limit_rad(ay - center_yaw);
      t = (diff >= 0) ? diff / std::fabs(w) : (diff + 2 * M_PI) / std::fabs(w);
    }
    if (t < dt_min) {
      armor_id = k;
      dt_min = t;
    }
  }
  return {hit_point_xy[0], hit_point_xy[1], armors[armor_id][2]};
}

io::Command HeroAimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now, Eigen::Vector3d ypr)
{
  if (targets.empty()) {
    return {false, false, 0, 0};
  }

  auto target = targets.front();
  auto target_predicted = target;
  auto t_fire = timestamp;
  // 设置弹速
  if (bullet_speed != 0.0) {
    bullet_speed = 11.8;
  }
  auto ekf_x = target_predicted.ekf_x();

  // 高速分支：云台 yaw 锁在"车中心连线 - r"方向（不追板），pitch 跟将正对那块板的高度
  // cmd 加低通防抖，开火条件改为"板进瞄准方向窗口 + 板法线朝你 + 云台稳态到位"
  if (std::abs(ekf_x[7]) > center_mode_w_threshold_) {
    const auto t_decide = std::chrono::steady_clock::now();

    // 命中时刻 = 现在 + 发弹延时 + 飞行时间 + 云台响应时间
    // 飞行时间用一个粗略初值（按瞄准点到当前的距离估），随后用真解迭代一次
    auto target_rotate = target;
    target_rotate.predict(t_decide);

    const double t_gimbal = std::max(gimbal_yaw_tau_, gimbal_pitch_tau_);

    // 第一遍：拿当前位置算粗弹道，得 fly_time
    Eigen::Vector3d xyz_now;
    if (target_rotate.name == ArmorName::outpost) {
      xyz_now = get_outpost_hit_xyz(target_rotate);
    } else {
      xyz_now = get_top_front(target_rotate);
    }
    const double d_now = std::hypot(xyz_now[0], xyz_now[1]);
    tools::Trajectory traj_init(bullet_speed, d_now, xyz_now[2]);
    if (traj_init.unsolvable) {
      tools::logger()->debug(
        "[HeroAimer] Unsolvable trajectory_init: {:.2f} {:.2f} {:.2f}", bullet_speed, d_now,
        xyz_now[2]);
      return {false, false, 0, 0};
    }

    // 命中时刻预测
    const double dt_total = shoot_delay_ + traj_init.fly_time + t_gimbal;
    const auto t_hit =
      t_decide + std::chrono::microseconds(static_cast<int>(dt_total * 1e6));
    target_rotate.predict(t_hit);
    ekf_x = target_rotate.ekf_x();

    // 命中时刻的瞄准点（xy 在车中心连线上，z 跟将正对那块板）
    Eigen::Vector3d xyz0;
    if (target_rotate.name == ArmorName::outpost) {
      xyz0 = get_outpost_hit_xyz(target_rotate);
    } else {
      xyz0 = get_top_front(target_rotate);
    }

    const double d0 = std::hypot(xyz0[0], xyz0[1]);
    tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);
    if (trajectory0.unsolvable) {
      tools::logger()->debug(
        "[HeroAimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
      return {false, false, 0, 0};
    }

    const double yaw_raw = std::atan2(xyz0[1], xyz0[0]) + yaw_offset_;
    const double pitch_raw = -(trajectory0.pitch + pitch_offset_);

    // cmd 一阶低通：alpha = dt / (cmd_smooth_tau + dt)
    // tau 越大 cmd 越慢，云台越跟得动；副作用：cmd 自身也滞后
    double yaw_cmd = yaw_raw;
    double pitch_cmd = pitch_raw;
    if (!cmd_lp_init_) {
      yaw_cmd_lp_ = yaw_raw;
      pitch_cmd_lp_ = pitch_raw;
      cmd_lp_init_ = true;
      last_cmd_time_ = t_decide;
    } else {
      const double dt_cmd = std::chrono::duration<double>(t_decide - last_cmd_time_).count();
      const double dt_eff = std::max(dt_cmd, 1e-3);
      const double alpha = dt_eff / (cmd_smooth_tau_ + dt_eff);
      // yaw 跨 ±π 边界处理：把残差 limit_rad 后再加回去
      const double yaw_residual = tools::limit_rad(yaw_raw - yaw_cmd_lp_);
      yaw_cmd_lp_ = tools::limit_rad(yaw_cmd_lp_ + alpha * yaw_residual);
      pitch_cmd_lp_ = pitch_cmd_lp_ + alpha * (pitch_raw - pitch_cmd_lp_);
      last_cmd_time_ = t_decide;
      yaw_cmd = yaw_cmd_lp_;
      pitch_cmd = pitch_cmd_lp_;
    }

    debug_aim_point_ = {true, {xyz0[0], xyz0[1], xyz0[2], 0}};

    // 开火判定：
    //   闸门 1：板"位置"接近瞄准方向（板会撞上来）
    //   闸门 2：板"法线"朝你（命中时不是侧倾跳弹）
    //   闸门 3：云台连续 N 帧到位且稳定（不是瞬时穿过）
    const auto armors_hit = target_rotate.armor_xyza_list();
    const double aim_yaw_world = yaw_cmd - yaw_offset_;

    bool any_armor_in_window = false;
    for (const Eigen::Vector4d & ap : armors_hit) {
      const double armor_los_yaw = std::atan2(ap[1], ap[0]);
      const double pos_diff = std::abs(tools::limit_rad(armor_los_yaw - aim_yaw_world));
      if (pos_diff > fire_pos_window_) continue;
      // 板法线相对该板视线的偏差，越小越正对
      const double facing_diff = std::abs(tools::limit_rad(ap[3] - armor_los_yaw));
      if (facing_diff > fire_facing_window_) continue;
      any_armor_in_window = true;
      break;
    }

    // 云台稳态历史：用 limit_rad 处理 yaw 跨边界
    const double yaw_err_now = std::abs(tools::limit_rad(ypr[0] - yaw_cmd));
    const double pitch_err_now = std::abs(ypr[1] - pitch_cmd);
    yaw_err_hist_.push_back(yaw_err_now);
    pitch_err_hist_.push_back(pitch_err_now);
    while ((int)yaw_err_hist_.size() > fire_steady_frames_) yaw_err_hist_.pop_front();
    while ((int)pitch_err_hist_.size() > fire_steady_frames_) pitch_err_hist_.pop_front();

    bool gimbal_steady = (int)yaw_err_hist_.size() >= fire_steady_frames_;
    if (gimbal_steady) {
      for (double e : yaw_err_hist_) {
        if (e > max_fire_error_yaw_) { gimbal_steady = false; break; }
      }
    }
    if (gimbal_steady) {
      for (double e : pitch_err_hist_) {
        if (e > max_fire_error_pitch_) { gimbal_steady = false; break; }
      }
    }

    const bool fire = any_armor_in_window && gimbal_steady;
    if (fire) tools::logger()->info("########## fire ##########");
    return {true, fire, yaw_cmd, pitch_cmd};
  }

  // 离开高速分支：清空稳态历史和低通状态，避免下次进入时用到陈旧数据
  cmd_lp_init_ = false;
  yaw_err_hist_.clear();
  pitch_err_hist_.clear();

  if (to_now) {
    const double dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + shoot_delay_;
    t_fire = timestamp + std::chrono::microseconds(static_cast<int>(dt * 1e6));
    target_predicted.predict(t_fire);
  }

  const auto aim_point0 = choose_aim_point(target_predicted);
  debug_aim_point_ = aim_point0;
  if (!aim_point0.valid) {
    return {false, false, 0, 0};
  }

  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  double d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);
  if (trajectory0.unsolvable) {
    tools::logger()->debug("[HeroAimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point_.valid = false;
    return {false, false, 0, 0};
  }

  const auto t_hit = t_fire + std::chrono::microseconds(static_cast<int>(trajectory0.fly_time * 1e6));
  target_predicted.predict(t_hit);

  const auto aim_point1 = choose_aim_point(target_predicted);
  debug_aim_point_ = aim_point1;
  if (!aim_point1.valid) {
    return {false, false, 0, 0};
  }

  const Eigen::Vector3d xyz1 = aim_point1.xyza.head(3);
  const double d1 = std::sqrt(xyz1[0] * xyz1[0] + xyz1[1] * xyz1[1]);
  tools::Trajectory trajectory1(bullet_speed, d1, xyz1[2]);
  if (trajectory1.unsolvable) {
    tools::logger()->debug("[HeroAimer] Unsolvable trajectory1: {:.2f} {:.2f} {:.2f}", bullet_speed, d1, xyz1[2]);
    debug_aim_point_.valid = false;
    return {false, false, 0, 0};
  }

  const double time_error = trajectory1.fly_time - trajectory0.fly_time;
  if (std::abs(time_error) > 0.1) {
    tools::logger()->debug("[HeroAimer] Large time error: {:.3f}", time_error);
    debug_aim_point_.valid = false;
    return {false, false, 0, 0};
  }

  const double yaw = std::atan2(xyz1[1], xyz1[0]) + yaw_offset_;
  const double pitch = -(trajectory1.pitch + pitch_offset_);

  bool is_fire = false;
  if ((std::abs(ypr[0] - yaw) < max_fire_error_yaw_) && (std::abs(ypr[1] - pitch) < max_fire_error_pitch_)) {
    is_fire = true;
  }

  return {true, is_fire, yaw, pitch};
}

// HeroAimPoint HeroAimer::choose_aim_point(const Target & target)
// {
//   Eigen::VectorXd ekf_x = target.ekf_x();
//   const std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
//   const std::size_t n = armor_xyza_list.size();

//   if (!target.jumped) return {true, armor_xyza_list[0]};

//   const auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

//   int first_id = -1, second_id = -1;
//   float min1 = std::numeric_limits<float>::max();
//   float min2 = std::numeric_limits<float>::max();

//   // 选择两个距离中心最近的装甲板 yaw差值最小的两个装甲板
//   for (std::size_t i = 0; i < n; ++i) {
//     const Eigen::Vector4d & pl = armor_xyza_list[i];
//     const float delta = static_cast<float>(std::abs(pl[3] - center_yaw));

//     if (delta < min1) {
//       min2 = min1;
//       second_id = first_id;

//       min1 = delta;
//       first_id = static_cast<int>(i);
//     } else if (delta < min2) {
//       min2 = delta;
//       second_id = static_cast<int>(i);
//     }
//   }

//   int dmin_first_id = -1, dmin_second_id = -1;
//   float dmin1 = std::numeric_limits<float>::max();
//   float dmin2 = std::numeric_limits<float>::max();

//   // 选择两个距离中心最近的装甲板 xy差值最小的两个装甲板
//   for (std::size_t i = 0; i < n; ++i) {
//     const Eigen::Vector4d & pl = armor_xyza_list[i];
//     const float delta = pl[0] * pl[0] + pl[1] * pl[1];

//     if (delta < dmin1) {
//       dmin2 = dmin1;
//       dmin_second_id = dmin_first_id;

//       dmin1 = delta;
//       dmin_first_id = static_cast<int>(i);
//     } else if (delta < dmin2) {
//       dmin2 = delta;
//       dmin_second_id = static_cast<int>(i);
//     }
//   }

//   // 
//   const auto plate = [&armor_xyza_list](int id) -> const Eigen::Vector4d & {
//     return armor_xyza_list[static_cast<std::size_t>(id)];
//   };
//   const Eigen::Vector4d & p1 = plate(first_id);
//   const Eigen::Vector4d & p2 = plate(second_id);
//   const Eigen::Vector3d xyz1 = p1.head(3);
//   const Eigen::Vector3d xyz2 = p2.head(3);

//   const double yaw1 = std::atan2(xyz1[1], xyz1[0]);
//   const double yaw2 = std::atan2(xyz2[1], xyz2[0]);

//   // 如果第一个装甲板不是xy差值最小的两个装甲板中的一个，则返回第二个装甲板
//   if (first_id != dmin_first_id && first_id != dmin_second_id) return {true, plate(second_id)};

//   // 如果第二个装甲板不是xy差值最小的两个装甲板中的一个，则返回第一个装甲板
//   if (second_id != dmin_first_id && second_id != dmin_second_id) return {true, plate(first_id)};

//   // 如果第一个装甲板的yaw差值大于最大射击角度，则返回第二个装甲板
//   if (std::abs(p1[3] - yaw1) > max_shoot_angle_ / 57.3) return {true, p2};

//   // 如果第二个装甲板的yaw差值大于最大射击角度，则返回第一个装甲板
//   if (std::abs(p2[3] - yaw2) > max_shoot_angle_ / 57.3) return {true, p1};

//   // 如果第一个装甲板和第二个装甲板的yaw差值都小于最大射击角度，则返回yaw差值较小的装甲板
//   if (lock_id_ != first_id && lock_id_ != second_id)
//     lock_id_ = (std::abs(p1[3] - yaw1) < std::abs(p2[3] - yaw2)) ? first_id : second_id;

//   return {true, plate(lock_id_)};
// }

HeroAimPoint HeroAimer::choose_aim_point(const Target & target)
{
  const std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  const auto armor_num = armor_xyza_list.size();
  if (!target.jumped) return {true, armor_xyza_list[0]};

  const double shoot_angle = max_shoot_angle_ / 57.3;

  // 板面 yaw 相对视线方位角的偏差，|delta| 越小越正对射手
  std::vector<double> delta_angle_list;
  delta_angle_list.reserve(armor_num);
  for (std::size_t i = 0; i < armor_num; ++i) {
    const Eigen::Vector4d & pl = armor_xyza_list[i];
    const double los_yaw = std::atan2(pl[1], pl[0]);
    delta_angle_list.push_back(tools::limit_rad(pl[3] - los_yaw));
  }

  // xy 平面距离最近的两块装甲板
  int nearest_id = 0;
  int second_nearest_id = -1;
  float min_dist1 = std::numeric_limits<float>::max();
  float min_dist2 = std::numeric_limits<float>::max();
  for (std::size_t i = 0; i < armor_num; ++i) {
    const Eigen::Vector4d & pl = armor_xyza_list[i];
    const float dist_sq = static_cast<float>(pl[0] * pl[0] + pl[1] * pl[1]);

    if (dist_sq < min_dist1) {
      min_dist2 = min_dist1;
      second_nearest_id = nearest_id;

      min_dist1 = dist_sq;
      nearest_id = static_cast<int>(i);
    } else if (dist_sq < min_dist2) {
      min_dist2 = dist_sq;
      second_nearest_id = static_cast<int>(i);
    }
  }

  const int id0 = nearest_id;
  const int id1 = (second_nearest_id < 0) ? nearest_id : second_nearest_id;
  const int best_id =
    (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

  if (std::abs(delta_angle_list[best_id]) > shoot_angle) {
    tools::logger()->warn("[HeroAimer] Empty id list!");
    return {false, armor_xyza_list[0]};
  }
  return {true, armor_xyza_list[best_id]};
}

}  // namespace auto_aim
