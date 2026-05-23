#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/hero_aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/signal_recorder.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/hero.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  // 录制器：按 yaml 开关启用，缺省/未配置时默认开（兼容旧行为）
  std::unique_ptr<tools::Recorder> recorder;
  std::unique_ptr<tools::SignalRecorder> signal_recorder;
  {
    auto yaml = YAML::LoadFile(config_path);
    bool record_video = !yaml["record_video"] || yaml["record_video"].as<bool>();
    if (record_video) {
      recorder = std::make_unique<tools::Recorder>();
      tools::logger()->info("[Main] Recorder (video+quat) enabled.");
    }
    if (yaml["record_signals"] && yaml["record_signals"].as<bool>()) {
      signal_recorder = std::make_unique<tools::SignalRecorder>();
      tools::logger()->info("[Main] SignalRecorder enabled.");
    }
  }

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO detector(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::HeroAimer hero_aimer(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  auto mode = io::GimbalMode::IDLE;
  auto last_mode = io::GimbalMode::IDLE;

  while (!exiter.exit()) {
    auto t0 = std::chrono::steady_clock::now();

    camera.read(img, t);

    q = gimbal.q(t - 2ms);

    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode;
    }

    if (recorder) recorder->record(img, q, t);

    solver.set_R_gimbal2world(q);

    auto gimbal_state = gimbal.state();

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    auto armors = detector.detect(img);

    auto targets = tracker.track(armors, t, gimbal_state.enemy_color);

    auto command =
      hero_aimer.aim(targets, t, gimbal_state.bullet_speed, true, ypr);

    gimbal.send(
      command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);

    tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

    nlohmann::json data;
    data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
    data["tracker_state"] = tracker.state();

    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      auto min_x = 1e10;
      auto & armor = armors.front();
      for (auto & a : armors) {
        if (a.center.x < min_x) {
          min_x = a.center.x;
          armor = a;
        }
      }
      solver.solve(armor);
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
    }

    if (!targets.empty()) {
      auto target = targets.front();

      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); ++i) {
        const auto & xyza = armor_xyza_list[i];
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
        if (image_points.empty()) continue;
        float rx = image_points[0].x, ty = image_points[0].y;
        for (const auto & p : image_points) {
          if (p.x > rx) rx = p.x;
          if (p.y < ty) ty = p.y;
        }
        tools::draw_text(
          img, fmt::format("{}", i), {static_cast<int>(rx) - 8, static_cast<int>(ty) + 18},
          {0, 255, 255}, 1.2, 2);
      }

      // EKF 旋转中心 (x,y,z) → 像素；|w| 超过 hero 瞄中心阈值时红圈，否则绿圈
      {
        Eigen::VectorXd xc = target.ekf_x();
        const bool aim_center_mode = std::abs(xc[7]) > hero_aimer.center_mode_w_threshold();
        const std::vector<cv::Point3f> center_world{cv::Point3f(
          static_cast<float>(xc[0]), static_cast<float>(xc[2]), static_cast<float>(xc[4]))};
        auto center_px = solver.world2pixel(center_world);
        if (!center_px.empty()) {
          const auto & p = center_px[0];
          const cv::Scalar color = aim_center_mode ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
          cv::circle(
            img, cv::Point(static_cast<int>(p.x + 0.5f), static_cast<int>(p.y + 0.5f)), 14, color,
            2);
        }
      }

      auto aim_point = hero_aimer.debug_aim_point_;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid)
        tools::draw_points(img, image_points, {0, 0, 255});
      else
        tools::draw_points(img, image_points, {255, 0, 0});

      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

      // EKF 协方差对角线：看 EKF 是不是"信不信自己"，是 yaw 抖动诊断关键
      const auto & P = target.ekf().P;
      data["P_x"] = P(0, 0);
      data["P_y"] = P(2, 2);
      data["P_z"] = P(4, 4);
      data["P_a"] = P(6, 6);
      data["P_w"] = P(7, 7);
      data["P_r"] = P(8, 8);

      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    }

    data["gimbal_yaw"] = gimbal_state.yaw * 57.3;
    data["gimbal_pitch"] = gimbal_state.pitch * 57.3;
    data["gimbal_yaw_vel"] = gimbal_state.yaw_vel;
    data["gimbal_pitch_vel"] = gimbal_state.pitch_vel;
    data["bullet_speed"] = gimbal_state.bullet_speed;
    data["bullet_count"] = gimbal_state.bullet_count;

    data["cmd_yaw"] = command.yaw * 57.3;
    data["cmd_pitch"] = command.pitch * 57.3;
    data["cmd_shoot"] = command.shoot ? 1 : 0;     // PlotJuggler 看是否在开火（0/1 方波）
    data["cmd_control"] = command.control ? 1 : 0; // 是否在跟踪控制

    plotter.plot(data);
    if (signal_recorder) signal_recorder->record(t, data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}
