#ifndef __BASEINFER_HPP__
#define __BASEINFER_HPP__

#include <tuple>
#include <vector>

namespace tdt_radar {

#define GPU_BLOCK_THREADS 512


// 图像信息类，用于存储图像信息
// 包含图像指针、宽度、高度
// 两个构造
struct Image {
    const void* bgrptr = nullptr;
    int         width = 0, height = 0;

    Image() = default;
    Image(const void* bgrptr, int width, int height)
        : bgrptr(bgrptr), width(width), height(height)
    {
    }
};
// 归一化类型
enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };
// 通道类型
enum class ChannelType : int { None = 0, SwapRB = 1 };


// 基类Infer，用于实现推理
template <typename T>
class Infer {
public:
    virtual T forward(const Image& image, void* stream = nullptr) = 0;
    virtual std::vector<T> forwards(const std::vector<Image>& images,
                                    void* stream = nullptr) = 0;
};

// 归一化类，用于存储归一化信息
// 包含均值、方差、alpha、beta、类型、通道类型
// 三个静态方法，用于创建归一化对象
struct Norm {
    float       mean[3];
    float       std[3];
    float       alpha, beta;
    NormType    type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    static Norm mean_std(const float mean[3], const float std[3],
                         float       alpha = 1 / 255.0f,
                         ChannelType channel_type = ChannelType::None);

    static Norm alpha_beta(float alpha, float beta = 0,
                           ChannelType channel_type = ChannelType::None);

    static Norm None();
};

// 仿射矩阵类，用于存储仿射矩阵信息
// 包含i2d、d2i
// 一个方法，用于计算仿射矩阵
// 网络到图像的仿射矩阵，i2d，图像到网络的仿射矩阵，d2i
struct AffineMatrix {
    float i2d[6]; // 输入到输出
    float d2i[6]; // 输出到输入

    void compute(const std::tuple<int, int>& from,
                 const std::tuple<int, int>& to);
};

}  // namespace tdt_radar

#endif  //__BASEINFER_HPP__
