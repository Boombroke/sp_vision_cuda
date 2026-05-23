#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

// 前哨站三块装甲板的高度阶梯（顺时针 1→2→3 高度递增 10.2cm）
// EKF 中 id 按 yaw 递增（CCW），对应 CCW 序列 L→H→M
// DZ[phase][id] = 在相位 phase 下，EKF id 号板相对 L 板的 z 偏移
constexpr double OUTPOST_STEP = 0.102;
constexpr double OUTPOST_DZ[3][3] = {
  {0.0,             2 * OUTPOST_STEP, OUTPOST_STEP    },  // phase=0: plate-0 = L
  {2 * OUTPOST_STEP, OUTPOST_STEP,    0.0             },  // phase=1: plate-0 = H
  {OUTPOST_STEP,    0.0,              2 * OUTPOST_STEP},  // phase=2: plate-0 = M
};

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

  // 前哨站相位（plate-0 当前对应 L/H/M 中哪一阶）和旋转方向锁定状态
  bool outpost_phase_locked_ = false;
  bool outpost_w_sign_locked_ = false;
  int outpost_phase_ = 0;

  // 前哨站每块板观测 z 的 EMA（phase 锁定后启用），armor_xyza_list 用其替代模型 z
  // 解决"高板打低/低板打高"：DZ 表压缩了真实高度差，直接用观测更准
  // -1000.0 表示尚未有观测（哨兵值）
  double outpost_z_ema_[3] = {-1000.0, -1000.0, -1000.0};

  // 重捕时由 Tracker 调用：把 EKF 的 w 项直接写到指定值
  // （仅给 outpost 重捕沿用旧 sign(w) 用，其他场景不应该用）
  void set_w(double w);

  // 前哨站转速钳位幅值（rad/s），由 Tracker 从 yaml 加载后写入
  static double outpost_w_clamp;

private:
  int armor_num_;// 装甲板数量
  int switch_count_;// 切换次数
  int update_count_;// 更新次数

  bool is_switch_, is_converged_;// 是否切换, 是否收敛

  // 前哨站 phase 识别用：每块板的 z 观测累积（最后 N 次）
  std::vector<std::vector<double>> outpost_z_obs_{3};

  tools::ExtendedKalmanFilter ekf_;// 扩展卡尔曼滤波器
  std::chrono::steady_clock::time_point t_;// 时间点

  //更新装甲板状态
  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  // 推断出 此外的装甲板中心的状态 x y 
  // x是车子中心的状态 ekf_.x
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  // 
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;

  // 新模型下 z_c (x[4]) 表示 L 板高度，三块板 z = z_c + DZ[phase][id]
  double getoutpost_armor_z(int id, const Eigen::VectorXd x) const {
    if (id < 0 || id > 2) return x[4];
    return x[4] + OUTPOST_DZ[outpost_phase_][id];
  }
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP