/** 临时：engine + 视频，forward 后画框与关键点，打印推理耗时。
 *  可视化风格参考 tasks/auto_aim + tools/img_tools（putText + 更粗的框与标签底）。 */

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "BaseInfer.hpp"
#include "yolos.hpp"

namespace {

/** 与 tasks/auto_aim/armor.hpp ARMOR_NAMES 顺序一致（RP 9 类） */
constexpr const char* kArmorNames[] = {"one",    "two",     "three", "four",     "five",
                                       "sentry", "outpost", "base",  "not_armor"};
constexpr int         kNumArmorNames = int(sizeof kArmorNames / sizeof kArmorNames[0]);

/** 与 tasks/auto_aim/armor.hpp COLORS 顺序一致 */
constexpr const char* kColorNames[] = {"red", "blue", "extinguish", "purple"};
constexpr int         kNumColorNames = int(sizeof kColorNames / sizeof kColorNames[0]);

/** BGR：亮绿装甲框 */
const cv::Scalar kArmorBgr(0, 255, 64);
const cv::Scalar kWhite(255, 255, 255);

/** 仿 tools::draw_text，略放大字号与线宽便于观看 */
void draw_text_vis(cv::Mat& img, const std::string& text, cv::Point org, const cv::Scalar& color,
                   double font_scale = 1.15, int thickness = 2)
{
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0),
                thickness + 2, cv::LINE_AA);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
                cv::LINE_AA);
}

/** 类型、置信度等：白字黑描边，贴在框上沿外或下沿外 */
void draw_detection_label(cv::Mat& img, const yolo::Box& b, int box_top, int box_bottom)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << b.confidence;
    if (b.class_label >= 0 && b.class_label < kNumArmorNames)
        oss << ' ' << kArmorNames[b.class_label];
    else
        oss << " cls" << b.class_label;
    if (b.color_id >= 0 && b.color_id < kNumColorNames)
        oss << ' ' << kColorNames[b.color_id];
    const std::string text = oss.str();

    const double font_scale = 1.15;
    const int    thick = 2;
    int          baseline = 0;
    cv::Size     sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thick, &baseline);
    int          x = (int)b.left;
    int          y = box_top - 4;
    if (y < sz.height + 6)
        y = box_bottom + sz.height + 8;
    x = std::max(0, std::min(x, img.cols - sz.width - 2));
    y = std::max(sz.height + 2, std::min(y, img.rows - 2));

    cv::rectangle(img, cv::Rect(x - 2, y - sz.height - 4, sz.width + 6, sz.height + baseline + 8),
                  cv::Scalar(0, 0, 0), -1);
    draw_text_vis(img, text, cv::Point(x, y), kWhite, font_scale, thick);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <engine.trt> <video_path> [output_dir|-]\n";
        return -1;
    }

    const std::string engine = argv[1];
    const std::string video = argv[2];
    const std::string out =
        (argc >= 4 && std::string(argv[3]) != "-") ? argv[3] : "";

    if (!out.empty())
        std::filesystem::create_directories(out);

    cv::VideoCapture cap(video);
    if (!cap.isOpened()) {
        std::cerr << "open video failed: " << video << std::endl;
        return -1;
    }

    auto infer = yolo::load(engine, yolo::Type::RP, 0.45f, 0.6f);
    if (!infer) {
        std::cerr << "load engine failed: " << engine << std::endl;
        return -1;
    }

    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    cv::Mat frame;
    for (int fi = 0;; ++fi) {
        cap >> frame;
        if (frame.empty())
            break;
        if (!frame.isContinuous())
            frame = frame.clone();

        tdt_radar::Image img(frame.data, frame.cols, frame.rows);

        auto t0 = std::chrono::high_resolution_clock::now();
        // Infer<BoxArray>::forward 返回一整帧的 yolo::BoxArray（vector<yolo::Box>）
        yolo::BoxArray boxes = infer->forward(img, stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;

        for (size_t i = 0; i < boxes.size(); ++i) {
            const yolo::Box& b = boxes[i];
            const int        bt = (int)b.top;
            const int        bb = (int)b.bottom;
            const int        bl = (int)b.left;
            const int        bw = std::max(1, (int)(b.right - b.left));
            const int        bh = std::max(1, (int)(b.bottom - b.top));
            cv::rectangle(frame, cv::Rect(bl, bt, bw, bh), kArmorBgr, 3);
            draw_detection_label(frame, b, bt, bb);
            if (b.points.size() >= 8) {
                std::vector<cv::Point2f> kv;
                for (size_t k = 0; k < 8; k += 2)
                    kv.emplace_back(b.points[k], b.points[k + 1]);
                for (const auto& p : kv)
                    cv::circle(frame, p, 5, kWhite, -1);
                for (size_t k = 0; k < kv.size(); ++k)
                    cv::line(frame, kv[k], kv[(k + 1) % kv.size()], kWhite, 2);
            }
        }

        {
            std::ostringstream hud;
            hud << '[' << fi << "] " << boxes.size() << " det  " << std::fixed
                << std::setprecision(2) << ms << " ms";
            draw_text_vis(frame, hud.str(), cv::Point(12, 36), kWhite, 1.0, 2);
        }

        std::cout << "frame " << fi << " infer " << std::fixed << std::setprecision(2) << ms
                  << " ms  det " << boxes.size() << std::endl;

        if (!out.empty()) {
            std::ostringstream fn;
            fn << out << "/f_" << std::setfill('0') << std::setw(6) << fi << ".jpg";
            cv::imwrite(fn.str(), frame);
        }
        cv::imshow("rp", frame);
        if (cv::waitKey(1) == 27)
            break;
    }

    cudaStreamDestroy(stream);
    return 0;
}
