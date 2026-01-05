//
// Created by mozihe on 24-11-21.
//
#include "BaseInfer.hpp"
#include <cstring>

using namespace tdt_radar;

/**
 * @brief 创建均值-标准差归一化配置
 * 
 * 使用 Z-score 归一化方法：(pixel / alpha - mean) / std
 * 常用于 ImageNet 预训练模型，如 YOLO、ResNet 等
 * 
 * @param mean[3] RGB 三个通道的均值，例如 [0.485, 0.456, 0.406]
 * @param std[3] RGB 三个通道的标准差，例如 [0.229, 0.224, 0.225]
 * @param alpha 像素值缩放因子，默认 1/255.0，将 [0,255] 缩放到 [0,1]
 * @param channel_type 通道处理类型，是否交换 R/B 通道（BGR ↔ RGB）
 * @return Norm 归一化配置对象
 */
Norm Norm::mean_std(const float mean[3], const float std[3], float alpha,
                    ChannelType channel_type)
{
    Norm out;
    out.type = NormType::MeanStd;  // 设置归一化类型为均值-标准差
    out.alpha = alpha;             // 设置缩放因子
    out.channel_type = channel_type;  // 设置通道处理类型
    memcpy(out.mean, mean, sizeof(out.mean));  // 复制均值数组
    memcpy(out.std, std, sizeof(out.std));     // 复制标准差数组
    return out;
}

/**
 * @brief 创建线性归一化配置
 * 
 * 使用线性变换：pixel * alpha + beta
 * 常用于简单的像素值归一化，如将 [0,255] 缩放到 [0,1]
 * 
 * @param alpha 缩放系数，例如 1/255.0 将像素值从 [0,255] 缩放到 [0,1]
 * @param beta 偏移量，默认 0
 * @param channel_type 通道处理类型，是否交换 R/B 通道（BGR ↔ RGB）
 * @return Norm 归一化配置对象
 */
Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type)
{
    Norm out;
    out.type = NormType::AlphaBeta;  // 设置归一化类型为线性变换
    out.alpha = alpha;                // 设置缩放系数
    out.beta = beta;                  // 设置偏移量
    out.channel_type = channel_type;  // 设置通道处理类型
    return out;
}

/**
 * @brief 创建无归一化配置
 * 
 * 不进行任何归一化处理，直接使用原始像素值
 * 
 * @return Norm 归一化配置对象（type = NormType::None）
 */
Norm Norm::None()
{
    return Norm();  // 返回默认构造的 Norm 对象，type 默认为 None
}

/**
 * @brief 计算仿射变换矩阵
 * 
 * 计算从原始图像尺寸到网络输入尺寸的仿射变换矩阵
 * 使用等比例缩放并居中，保持图像宽高比不变
 * 
 * @param from 原始图像尺寸 (width, height)
 * @param to 目标尺寸，即网络输入尺寸 (width, height)
 * 
 * 变换矩阵格式（2x3 矩阵，按行优先存储为 6 个元素）：
 * [a b tx]  其中 a,b 是缩放和旋转参数，tx,ty 是平移参数
 * [c d ty]
 * 
 * i2d: Image to Device，从原始图像坐标到网络输入坐标的变换
 * d2i: Device to Image，从网络输入坐标到原始图像坐标的变换（i2d 的逆矩阵）
 */
void AffineMatrix::compute(const std::tuple<int, int>& from,
                           const std::tuple<int, int>& to)
{
    // 计算 x 和 y 方向的缩放比例
    float scale_x = std::get<0>(to) / (float)std::get<0>(from);
    float scale_y = std::get<1>(to) / (float)std::get<1>(from);
    // 使用较小的缩放比例，保持宽高比不变（等比例缩放）
    float scale = std::min(scale_x, scale_y);
    
    // 计算 i2d 矩阵（Image to Device）
    // 矩阵形式：[scale  0    tx]
    //          [0      scale ty]
    i2d[0] = scale;  // x 方向缩放
    i2d[1] = 0;       // 无旋转
    // x 方向平移：将图像中心对齐到目标中心
    i2d[2] = -scale * std::get<0>(from) * 0.5 + std::get<0>(to) * 0.5 +
             scale * 0.5 - 0.5;
    i2d[3] = 0;       // 无旋转
    i2d[4] = scale;   // y 方向缩放
    // y 方向平移：将图像中心对齐到目标中心
    i2d[5] = -scale * std::get<1>(from) * 0.5 + std::get<1>(to) * 0.5 +
             scale * 0.5 - 0.5;

    // 计算 i2d 矩阵的逆矩阵，得到 d2i（Device to Image）
    // 对于 2x2 矩阵 [a b; c d]，逆矩阵为 (1/D) * [d -b; -c a]
    // 其中 D = ad - bc 是行列式
    double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];  // 计算行列式
    D = D != 0. ? double(1.) / D : double(0.);      // 计算行列式的倒数
    // 计算逆矩阵的元素
    double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D,
           A21 = -i2d[3] * D;
    // 计算逆变换的平移参数
    double b1 = -A11 * i2d[2] - A12 * i2d[5];
    double b2 = -A21 * i2d[2] - A22 * i2d[5];

    // 存储 d2i 矩阵（Device to Image，i2d 的逆矩阵）
    d2i[0] = A11;
    d2i[1] = A12;
    d2i[2] = b1;
    d2i[3] = A21;
    d2i[4] = A22;
    d2i[5] = b2;
}
