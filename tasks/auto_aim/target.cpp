#include "tasks/auto_aim/target.hpp"

#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

double Target::outpost_w_clamp = 2.51;

// 构造函数
Target::Target(
  const Armor & armor, 
  std::chrono::steady_clock::time_point t, 
  double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  // 这步再看看
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}


Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}



// 预测dt段的状态
void Target::predict(double dt)
{
  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on


  
  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1, v2, q_l, q_h;
  if (name == ArmorName::outpost) {
    v1 = 10;     // 前哨站加速度方差
    // 关键：phase/w 锁定前 v2 必须大，否则 EKF 跟不上 2.5rad/s 的转速
    // 锁定后 v2 调小，让 w 稳定在 ±2.51 附近
    v2 = outpost_w_sign_locked_ ? 0.05 : 5.0;
    q_l = 0;     // 高度阶梯由 DZ 表给出，l/h 不再更新
    q_h = 0;
  } else {
    v1 = 100;  // 加速度方差
    v2 = 400;  // 角加速度方差
    q_l = 0;   // 4装甲板的l参数过程噪声（根据实际情况设置）
    q_h = 0;   // 4装甲板的h参数过程噪声（根据实际情况设置）


  }
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;



  // 预测过程噪声偏差的方差
  // clang-format off
  Eigen::MatrixXd Q{

    // x       vx         y     vy        z     vz       a     w     r   l h
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, q_l, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, q_h}
  };
  // clang-format on

  
  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  
  // 前哨站转速钳位：sign(w) 一旦锁定就钉到 ±outpost_w_clamp
  if (this->name == ArmorName::outpost && this->outpost_w_sign_locked_) {
    this->ekf_.x[7] = (this->ekf_.x[7] > 0) ? outpost_w_clamp : -outpost_w_clamp;
  }

  
    ekf_.predict(F, Q, f);
}


// 更新装甲板
void Target::update(const Armor & armor)
{
  int id;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  // 得到 pair vector: (xyza, id)
  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  // 按照距离排序
  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });



  // 取前3个distance最小的装甲板
  for (int i = 0; i < 3; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    // 前哨站 phase 锁定后，加入 z 残差作为额外鉴别（10cm 高度差区分度极高）
    if (name == ArmorName::outpost && outpost_phase_locked_) {
      double z_pred = ekf_.x[4] + OUTPOST_DZ[outpost_phase_][xyza_i_list[i].second];
      angle_error += 2.0 * std::abs(armor.xyz_in_world[2] - z_pred);  // 权重 2/m
    }

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  // 前哨站：识别旋转方向和相位
  if (name == ArmorName::outpost && update_count_ >= 1) {
    // 锁定 sign(w)：要求 EKF 已经足够稳定（update_count > 20 + |w_ekf| > 0.5）
    // 否则第一次"假切板"会把方向锁错
    if (
      !outpost_w_sign_locked_ && is_switch_ && update_count_ >= 20 &&
      std::abs(ekf_.x[7]) > 0.5) {
      // 优先信任 EKF 自己估出来的 w 方向，而不是 id 跳变方向
      // （id 关联可能在切板瞬间不准）
      double w_ekf = ekf_.x[7];
      int delta = ((id - last_id) % 3 + 3) % 3;
      ekf_.x[7] = (w_ekf > 0) ? +outpost_w_clamp : -outpost_w_clamp;
      outpost_w_sign_locked_ = true;
      tools::logger()->info(
        "[Target] outpost w sign locked: w = {:.2f} "
        "(ekf_w={:.2f}, last_id={}, id={}, delta={})",
        ekf_.x[7], w_ekf, last_id, id, delta);
    }

    // 每帧都累积 z 观测（不限制必须 switch），但限制每个 id 的样本数
    if (outpost_z_obs_[id].size() < 32) {
      outpost_z_obs_[id].push_back(armor.xyz_in_world[2]);
    }

    // phase 锁定后维护每块板 z 的 EMA，给 armor_xyza_list 用
    if (outpost_phase_locked_) {
      const double z_obs = armor.xyz_in_world[2];
      if (outpost_z_ema_[id] < -100.0) {
        outpost_z_ema_[id] = z_obs;  // 第一次直接初始化
      } else {
        // 异常值保护：超过 30cm 跳变多半是 id 关联错，丢弃这次观测
        if (std::abs(z_obs - outpost_z_ema_[id]) < 0.30) {
          outpost_z_ema_[id] = 0.85 * outpost_z_ema_[id] + 0.15 * z_obs;
        }
      }
    }

    // 锁定 phase：要求 3 块不同 id 都有观测（防几何欠定）+ EKF 已经稳定
    // 注：rms[0]==rms[2] 的对称性意味着只看到 2 块板时 phase LS 二选一不可靠
    if (!outpost_phase_locked_) {
      int seen = 0;
      int total = 0;
      int min_per_id = 999;
      for (auto & v : outpost_z_obs_) {
        if (!v.empty()) seen++;
        total += v.size();
        min_per_id = std::min<int>(min_per_id, v.size());
      }
      // 三块板都见过 + 每块至少 3 个样本 + EKF 至少更新 30 次
      if (seen >= 3 && min_per_id >= 3 && update_count_ >= 30) {
        // 对每个候选 phase 求 z_c 的最小二乘解，再计算残差
        int best_phi = 0;
        double best_err = 1e18;
        double best_zc = ekf_.x[4];
        for (int p = 0; p < 3; p++) {
          // z_obs - DZ[p][k] 在所有观测上的均值即 z_c 的 LS 解
          double sum = 0;
          int n = 0;
          for (int k = 0; k < 3; k++) {
            for (double z : outpost_z_obs_[k]) {
              sum += (z - OUTPOST_DZ[p][k]);
              n++;
            }
          }
          if (n == 0) continue;
          double zc = sum / n;
          double err = 0;
          for (int k = 0; k < 3; k++) {
            for (double z : outpost_z_obs_[k]) {
              double r = z - (zc + OUTPOST_DZ[p][k]);
              err += r * r;
            }
          }
          if (err < best_err) { best_err = err; best_phi = p; best_zc = zc; }
        }
        // 诊断：把所有候选都打出来便于分析
        double err0 = 0, err1 = 0, err2 = 0;
        int n_total = 0;
        for (int p = 0; p < 3; p++) {
          double sum = 0; int n = 0;
          for (int k = 0; k < 3; k++)
            for (double z : outpost_z_obs_[k]) { sum += (z - OUTPOST_DZ[p][k]); n++; }
          if (n == 0) continue;
          double zc = sum / n;
          double err = 0;
          for (int k = 0; k < 3; k++)
            for (double z : outpost_z_obs_[k]) {
              double r = z - (zc + OUTPOST_DZ[p][k]);
              err += r * r;
            }
          (p == 0 ? err0 : (p == 1 ? err1 : err2)) = err;
          n_total = n;
        }
        outpost_phase_ = best_phi;
        outpost_phase_locked_ = true;
        ekf_.x[4] = best_zc;
        tools::logger()->info(
          "[Target] phase locked: phase={}, z_c={:.3f}, n={}, "
          "rms[0]={:.3f}, rms[1]={:.3f}, rms[2]={:.3f}",
          best_phi, best_zc, n_total,
          std::sqrt(err0 / std::max(n_total, 1)),
          std::sqrt(err1 / std::max(n_total, 1)),
          std::sqrt(err2 / std::max(n_total, 1)));
      }
    }
  }

  last_id = id;
  update_count_++;

  // 更新 yaw pitch distance angle
  update_ypda(armor, id);
}



// 观测 和 预测 更新
void Target::update_ypda(const Armor & armor, int id)
{
  // 观测噪声
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  // R[1] 是 pitch 通道（垂直方向），影响 z 估计
  // 前哨站 phase 锁定前，pitch 残差可能高达 10cm（DZ 错配），完全忽略 z 观测；
  // 锁定后 z 残差仍有 5-10cm 量级（PnP 暗光噪声），R 给宽一些防止周期性 NIS 失败
  double R_pitch = 4e-3;
  if (name == ArmorName::outpost) {
    R_pitch = outpost_phase_locked_ ? 5e-2 : 1.0;
  }
  // 最后一项 R[3] 是 armor_yaw（板法线 yaw）的观测噪声
  // 原 9e-2 比球坐标 yaw（4e-3）大 25 倍，EKF 不信观测靠 predict 外推
  // 实测 EKF a 比 PnP 观测系统性超前 ~12° → 子弹打到目标已转过的位置
  // 降到 2e-2 让 EKF 更紧跟观测
  Eigen::VectorXd R_dig{
    {4e-3, R_pitch, log(std::abs(delta_angle) + 1) + 1, log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 2e-2}};
  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();// 转换为对角矩阵

  // R = [
  //   [4e-3,    0,    0,    0    ]
  //   [0,    4e-3,    0,    0    ]
  //   [0,       0,  var2,   0    ]  // var2 = log(|delta_angle|+1) + 1
  //   [0,       0,    0,  var3   ]  // var3 = log(|distance|+1)/200 + 9e-2
  // ]



  // h(x) 表示从状态 x（EKF 预测的状态）计算出预测的观测值

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  // z反映的才是真正的识别出的观测数值
  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  ekf_.update(z, H, R, h, z_subtract);
}



Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

void Target::set_w(double w) { ekf_.x[7] = w; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }



// 返回 ekf_x 的装甲板序列
// 前哨站：z 一律用模型 z (ekf_x[4] + DZ[phase][id])，避免 EMA 把 PnP 在板侧倾时
// 的 z 误差（高/低 ~20cm）平均后系统性偏估，导致 cmd_pitch 异常使子弹打高
std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;
  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}




// 判断滤波器是否发散
bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  // auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  // if (r_ok && l_ok) return false;

  if (name == ArmorName::outpost) {
    // 前哨站：高度阶梯由 DZ 表给出，不再依赖 l/h；只看 r 是否合理
    if (r_ok) return false;
} else {
    // 4装甲板：检查半径差
    auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;
    if (r_ok && l_ok) return false;
}


  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

// 判断滤波器是否收敛
bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站：phase 锁定且 update_count_ > 4 即可视为收敛
  if (
    this->name == ArmorName::outpost && this->outpost_phase_locked_ &&
    update_count_ > 4 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 通过ekf_x（车体中心）计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  auto outpost = armor_num_ == 3;


  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  // auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];
  auto armor_z = (outpost) ? getoutpost_armor_z(id, x) : (use_l_h) ? x[4] + x[10] : x[4];


  
  return {armor_x, armor_y, armor_z};
}


// 没看懂VvV
// 观测雅可比矩阵
Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);// 角度
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);// 维护长短轴半径
  auto outpost = (armor_num_ == 3); 

  
  auto r = (use_l_h) ? x[8] + x[9] : x[8];// 半径
  auto dx_da = r * std::sin(angle);// 角度对x的偏导
  auto dy_da = -r * std::cos(angle);// 角度对y的偏导

  auto dx_dr = -std::cos(angle);// 半径对x的偏导
  auto dy_dr = -std::sin(angle);// 半径对y的偏导
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;// 半径差对x的偏导
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;// 半径差对y的偏导

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;// 高度对z的偏导
  // 前哨站新模型：z 偏移 DZ[phase][id] 是常量，对 x[9]/x[10] 偏导为 0
  (void)outpost;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0,     dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  // 得到装甲板观测的雅可比矩阵
  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}


bool Target::checkinit() { return isinit; }


}  // namespace auto_aim
