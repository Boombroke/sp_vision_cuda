#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "tasks/auto_aim/armor.hpp"
#include "solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  // 自瞄跟踪
  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t, int enemy_color = -1);
    
  // 全向感知跟踪
  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, int enemy_color = -1);

private:
  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  double nis_thresh_default_;
  double nis_thresh_outpost_;
  double lost_timeout_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;

  // 前哨站 phase/sign(w) 跨 set_target 缓存：跟踪同一台前哨站时保留物理几何
  bool outpost_cache_valid_ = false;
  bool outpost_cache_phase_locked_ = false;
  bool outpost_cache_w_sign_locked_ = false;
  int outpost_cache_phase_ = 0;
  double outpost_cache_w_ = 0.0;  // 锁定的 w 值（±2.51）

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP