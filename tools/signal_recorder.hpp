#ifndef TOOLS__SIGNAL_RECORDER_HPP
#define TOOLS__SIGNAL_RECORDER_HPP

#include <atomic>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "tools/thread_safe_queue.hpp"

namespace tools
{
// JSONL 格式的"时间序列中间量"记录器：每帧一行 JSON，写入
// records/<timestamp>_signals.jsonl，便于离线用 pandas.read_json(lines=True)
// 复盘 EKF 状态、cmd、Tracker 状态等信号。
//
// 使用模式与 tools::Recorder 一致：构造时只生成路径并建好 records/ 目录，
// 文件在第一次 record() 调用时才打开，并启动后台保存线程。JSON 序列化在保
// 存线程里做，避免主循环卡顿。
class SignalRecorder
{
public:
  SignalRecorder();
  ~SignalRecorder();

  // 把一帧的多个信号合并成一行 JSON 写入磁盘。
  // - t       帧时间戳（steady_clock）；写入文件时换算成相对构造时刻的秒数
  // - signals 该帧的所有信号，例如 {"ekf_x": 1.2, "cmd_yaw": 0.3, ...}
  void record(std::chrono::steady_clock::time_point t, const nlohmann::json & signals);

private:
  struct FrameData
  {
    double since_begin;     // 相对 start_time_ 的秒数
    nlohmann::json signals;  // 该帧的信号字典
  };

  bool init_;
  std::atomic<bool> stop_thread_;
  std::string text_path_;
  std::ofstream text_writer_;
  std::chrono::steady_clock::time_point start_time_;
  tools::ThreadSafeQueue<FrameData> queue_;
  std::thread saving_thread_;

  void init();
  void save_to_file();
};

}  // namespace tools

#endif  // TOOLS__SIGNAL_RECORDER_HPP
