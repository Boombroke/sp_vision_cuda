#include "signal_recorder.hpp"

#include <fmt/chrono.h>

#include <filesystem>

#include "math_tools.hpp"

namespace tools
{
SignalRecorder::SignalRecorder() : init_(false), stop_thread_(false), queue_(1024)
{
  start_time_ = std::chrono::steady_clock::now();

  auto folder_path = "records";
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  text_path_ = fmt::format("{}/{}_signals.jsonl", folder_path, file_name);

  std::filesystem::create_directory(folder_path);
}

SignalRecorder::~SignalRecorder()
{
  stop_thread_ = true;
  // 退出时给队列额外推入一个空帧，避免 pop 一直等待
  queue_.push({0.0, nlohmann::json::object()});
  if (saving_thread_.joinable()) saving_thread_.join();

  if (!init_) return;
  text_writer_.close();
}

void SignalRecorder::record(
  std::chrono::steady_clock::time_point t, const nlohmann::json & signals)
{
  if (!init_) init();
  auto since_begin = tools::delta_time(t, start_time_);
  // 不在主线程做 dump，把 json 对象推进队列让保存线程序列化
  queue_.push({since_begin, signals});
}

void SignalRecorder::init()
{
  text_writer_.open(text_path_);
  saving_thread_ = std::thread(&SignalRecorder::save_to_file, this);
  init_ = true;
}

void SignalRecorder::save_to_file()
{
  while (!stop_thread_) {
    FrameData frame;
    queue_.pop(frame);

    // 跳过析构时塞进来的"哨兵"空 json
    if (frame.signals.empty()) continue;

    // 合并 t 和 signals 写一行 JSON。signals 已经是 object 时直接把 t 塞进去，
    // 否则用 {"t": ..., "value": signals} 兜底（虽然按约定调用方都传 object）。
    nlohmann::json line;
    if (frame.signals.is_object()) {
      line = frame.signals;
      line["t"] = frame.since_begin;
    } else {
      line["t"] = frame.since_begin;
      line["value"] = frame.signals;
    }

    // 每行写完立刻 flush，断电不丢数据
    text_writer_ << line.dump() << std::endl;
  }
}

}  // namespace tools
