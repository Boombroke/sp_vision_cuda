#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "yolos.hpp"

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "Usage: " << argv[0]
                  << " <yolov8_engine.trt> <video_path>" << std::endl;
        return -1;
    }

    std::string engine_file = argv[1];
    std::string video_path = argv[2];

    // 打开视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return -1;
    }

    // 加载 YOLOv8 推理引擎
    auto infer = yolo::load(engine_file, yolo::Type::V0708, 0.65f, 0.40f);
    if (!infer) {
        std::cerr << "Failed to load engine: " << engine_file << std::endl;
        return -1;
    }

    // 创建 CUDA stream（整段视频循环复用）
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream." << std::endl;
        return -1;
    }

    cv::Mat frame;
    int     frame_idx = 0;
    
    // FPS计算相关变量
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_fps_time = start_time;
    int fps_counter = 0;
    double current_fps = 0.0;
    double avg_fps = 0.0;
    
    while (true) {
        // 记录帧开始时间
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        cap >> frame;
        if (frame.empty())
            break;

        tdt_radar::Image input(frame.data, frame.cols, frame.rows);

        // 前向推理，得到检测框
        yolo::BoxArray boxes = infer->forward(input, stream);
        
        // 记录推理结束时间
        auto inference_end = std::chrono::high_resolution_clock::now();

        // 在图像上绘制检测结果（参考OpenVINO版本）
        for (size_t i = 0; i < boxes.size(); i++) {
            const auto& box = boxes[i];
            
            // 根据类别选择绘制颜色（暂时用class_label，后续可以添加color字段）
            // 红色(0)或蓝色(1)，这里暂时用随机颜色
            cv::Scalar box_color = (box.class_label % 2 == 0) ? 
                                   cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0); // 红色或蓝色
            
            // 绘制边界框
            cv::Rect rect((int)box.left, (int)box.top,
                         (int)(box.right - box.left), 
                         (int)(box.bottom - box.top));
            cv::rectangle(frame, rect, box_color, 2);
            
            // 绘制关键点（如果有）
            if (!box.points.empty() && box.points.size() >= 8) {
                std::vector<cv::Point2f> keypoints;
                // 关键点顺序：左上顺时针 [0,1] [2,3] [4,5] [6,7]
                for (size_t j = 0; j < 8; j += 2) {
                    keypoints.push_back(cv::Point2f(box.points[j], box.points[j + 1]));
                }
                
                // 绘制关键点圆圈（绿色）
                for (const auto& pt : keypoints) {
                    cv::circle(frame, pt, 3, cv::Scalar(0, 255, 0), -1); // 绿色实心圆
                }
                
                // 连接关键点（按顺序形成闭合图形）
                for (size_t j = 0; j < keypoints.size(); j++) {
                    cv::line(frame, keypoints[j], 
                            keypoints[(j + 1) % keypoints.size()], 
                            cv::Scalar(0, 255, 0), 1); // 绿色线条
                }
            }
            
            // 准备标签文本
            // 类别名称（根据OpenVINO版本）
            std::string class_names[] = {"G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"};
            std::string class_name = (box.class_label >= 0 && box.class_label < 9) ? 
                                    class_names[box.class_label] : "Unknown";
            std::string label_text = class_name + " " + 
                                    std::to_string(box.confidence).substr(0, 4);
            
            // 绘制标签背景
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point text_org(rect.x, rect.y - 5);
            if (text_org.y < text_size.height) {
                text_org.y = rect.y + text_size.height + 5;
            }
            cv::rectangle(frame, 
                         cv::Point(text_org.x, text_org.y - text_size.height - 2),
                         cv::Point(text_org.x + text_size.width, text_org.y + baseline),
                         cv::Scalar(0, 0, 0), -1); // 黑色背景
            
            // 绘制标签文本
            cv::putText(frame, label_text, text_org, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
        
        // 打印检测结果
        std::cout << "Frame " << frame_idx++ 
                  << " 检测到 " << boxes.size() << " 个目标:" << std::endl;
        for (size_t i = 0; i < boxes.size(); i++) {
            const auto& box = boxes[i];
            std::cout << "  目标 " << i + 1 << ": "
                     << "类别=" << box.class_label << ", "
                     << "置信度=" << box.confidence << ", "
                     << "边界框=(" << box.left << "," << box.top << "," 
                     << (box.right - box.left) << "," << (box.bottom - box.top) << ")";
            if (!box.points.empty() && box.points.size() >= 8) {
                std::cout << ", 关键点=[";
                for (size_t j = 0; j < 8; j += 2) {
                    std::cout << "(" << box.points[j] << "," << box.points[j + 1] << ")";
                    if (j < 6) std::cout << ",";
                }
                std::cout << "]";
            }
            std::cout << std::endl;
        }
        
        // 计算FPS
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame_end - frame_start).count();
        current_fps = 1000.0 / frame_duration;  // 当前帧FPS
        
        // 计算平均FPS（每30帧更新一次）
        fps_counter++;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame_end - last_fps_time).count();
        if (elapsed >= 1000) {  // 每秒更新一次平均FPS
            avg_fps = fps_counter * 1000.0 / elapsed;
            fps_counter = 0;
            last_fps_time = frame_end;
        }
        
        // 在图像上显示FPS
        std::string fps_text = "FPS: " + std::to_string(current_fps).substr(0, 5);
        std::string avg_fps_text = "Avg FPS: " + std::to_string(avg_fps).substr(0, 5);
        cv::putText(frame, fps_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, avg_fps_text, cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // 显示推理时间（可选）
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            inference_end - frame_start).count();
        std::string inference_text = "Inference: " + std::to_string(inference_time) + "ms";
        cv::putText(frame, inference_text, cv::Point(10, 90),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        
        cv::imshow("YOLOv8 video", frame);
        if (cv::waitKey(1) == 27 || cv::waitKey(1) == 'q')  // ESC 或 'q' 退出
            break;
    }
    
    // 计算总平均FPS
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    if (total_time > 0 && frame_idx > 0) {
        double total_avg_fps = frame_idx * 1000.0 / total_time;
        std::cout << "\n总平均FPS: " << total_avg_fps << std::endl;
    }

    cudaStreamDestroy(stream);
    return 0;
}


