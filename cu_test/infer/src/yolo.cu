#include "BaseInfer.hpp"
#include "NvidiaInterface.hpp"
#include "yolos.hpp"

namespace yolo {

using namespace std;
using namespace tdt_radar;
static const int point_num = 4;//关键点数量

const int NUM_BOX_ELEMENT = 15;//   
const int MAX_IMAGE_BOXES = 1024;//最大检测框数量

inline int upbound(int n, int align = 32)//向上取整
{
    return (n + align - 1) / align * align;
}//向上取整

//仿射变换
static __host__ __device__ void
affine_project(float* matrix, float x, float y, float* ox, float* oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

// Sigmoid函数（用于0708模型）
static __device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// 解码核心
static __global__ void
decode_kernel_common(float* predict, int num_bboxes, int num_classes,
                     int output_cdim, float confidence_threshold,
                     float* invert_affine_matrix, float* parray,
                     int MAX_IMAGE_BOXES)
{
    // ========== CUDA 并行处理机制 ==========
    // 关键：每个GPU线程处理一个预测框，不需要循环！
    // 
    // 例如：如果有 25200 个框，GPU 会启动 25200 个线程
    // - 线程0 处理框0 (position=0)
    // - 线程1 处理框1 (position=1)
    // - 线程2 处理框2 (position=2)
    // - ...
    // - 线程25199 处理框25199 (position=25199)
    // 
    // 所有线程同时并行执行，不需要 for 循环！
    // ======================================
    
    // 计算当前线程的唯一ID（position）
    // blockDim.x: 每个block的线程数（例如512）
    // blockIdx.x: 当前block的索引
    // threadIdx.x: 当前线程在block内的索引
    // 公式：position = blockIdx.x * blockDim.x + threadIdx.x
    // 例如：block 0 的线程 0: position = 0*512+0 = 0
    //       block 0 的线程 1: position = 0*512+1 = 1
    //       block 1 的线程 0: position = 1*512+0 = 512
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 边界检查：如果线程ID超出框的数量，直接返回
    if (position >= num_bboxes)
        return;
    
    // 根据 position 计算当前线程要处理的框在 predict 数组中的起始位置
    // predict: TensorRT输出的原始数据 [num_bboxes, output_cdim]
    // output_cdim: 每个框的维度（例如：4坐标+1objectness+num_classes = 22）
    // position: 当前线程处理的框索引（0, 1, 2, ..., num_bboxes-1）
    // 
    // 例如：output_cdim=22, position=5
    //      pitem 指向 predict[5*22] = predict[110]，即第6个框的数据
    float* pitem = predict + output_cdim * position;

    // ========== YOLO 输出格式 ==========
    // pitem[0-3]: 边界框坐标 (cx, cy, width, height)
    // pitem[4]:   objectness score（物体存在性分数：这个框里是否有物体）
    // pitem[5:]:  各个类别的置信度分数（这个框是哪个类别的概率）
    // ===================================
    
    // 第一步：检查物体存在性分数（objectness）
    // objectness 表示这个预测框包含物体的概率（不关心是什么类别）
    // 如果 objectness 太低，说明这个框很可能没有物体，直接过滤掉（早期过滤，提高效率）
    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;  // 提前返回，不处理这个框

    // 第二步：找到置信度最高的类别
    // 从 pitem[5] 开始是各个类别的置信度分数
    float* class_confidence = pitem + 5;
    float  confidence = *class_confidence++;  // 初始化为第0类的置信度
    int    label = 0;  // 初始化为第0类
    
    // 遍历所有类别，找到置信度最高的类别
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;  // 更新最大置信度
            label = i;  // 更新类别标签
        }
    }

    // 第三步：计算最终置信度
    // 最终置信度 = 物体存在性分数 × 类别置信度
    // 例如：objectness=0.9, 类别置信度=0.8 → 最终置信度=0.72
    confidence *= objectness;
    
    // 再次检查最终置信度，如果太低就过滤掉
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;

    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
}

static __global__ void
decode_kernel_v5_face(float* predict, int num_bboxes, int num_classes,
                      int output_cdim, float confidence_threshold,
                      float* invert_affine_matrix, float* parray,
                      int MAX_IMAGE_BOXES)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float* pitem = predict + output_cdim * position;

    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;

    float* class_confidence = pitem + 5 + 2 * point_num;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    objectness = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    float landmark_array[point_num * 2];

    for (int i = 0; i < point_num; i++) {
        landmark_array[2 * i] = *pitem++;
        landmark_array[2 * i + 1] = *pitem++;
    }

    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
    for (int i = 0; i < point_num; i++) {
        affine_project(invert_affine_matrix, landmark_array[2 * i],
                       landmark_array[2 * i + 1], &landmark_array[2 * i],
                       &landmark_array[2 * i + 1]);
    }
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    for (int i = 0; i < point_num; i++) {
        *pout_item++ = landmark_array[2 * i];
        *pout_item++ = landmark_array[2 * i + 1];
    }
}

static __global__ void decode_kernel_v8(float* predict, int num_bboxes,
                                        int num_classes, int output_cdim,
                                        float  confidence_threshold,
                                        float* invert_affine_matrix,
                                        float* parray, int MAX_IMAGE_BOXES)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float* pitem = predict + output_cdim * position;
    float* class_confidence = pitem + 4;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = position;
}





// 0708解码核心
// 输出格式：[0-7:关键点, 8:confidence, 9-12:颜色, 13-21:类别]
static __global__ void decode_kernel_V0708(float* predict, int num_bboxes, int num_classes,
                     int output_cdim, float confidence_threshold,
                     float scale, float offset_x, float offset_y, int detect_color, float* parray,
                     int MAX_IMAGE_BOXES)
{

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 边界检查：如果线程ID超出框的数量，直接返回
    if (position >= num_bboxes)
        return;

    float* pitem = predict + output_cdim * position;


    float confidence = sigmoid(pitem[8]);
    if (confidence < confidence_threshold)
        return;  // 提前返回，不处理这个框


    float* color_scores = pitem + 9;
    float max_color_score = *color_scores;
    int color_id = 0;
    for (int i = 1; i < 4; ++i) {
        if (color_scores[i] > max_color_score) {
            max_color_score = color_scores[i];
            color_id = i;
        }
    }



    // 颜色过滤：丢弃灰色(2)和紫色(3)
    if (color_id == 2 || color_id == 3)
        return;

    // detect_color过滤：
    // detect_color == 0: 检测蓝色，跳过红色
    // detect_color == 1: 检测红色，跳过蓝色
    if (detect_color == 0 && color_id == 0)  // 检测蓝色，跳过红色
        return;
    if (detect_color == 1 && color_id == 1)   // 检测红色，跳过蓝色
        return;


    // 第三步：读取类别分数，找到最大值
    float* class_scores = pitem + 13;
    float max_class_score = *class_scores;
    int class_id = 0;
    for (int i = 1; i < num_classes; ++i) {
        if (class_scores[i] > max_class_score) {
            max_class_score = class_scores[i];
            class_id = i;
            }
        }


        // 第四步：读取关键点坐标并还原到原图尺寸
        // OpenVINO的resize+padding将图像放在左上角，CUDA的仿射变换是居中放置
        // 需要先减去居中偏移，再除以scale还原
        float landmarks[8];
        for (int i = 0; i < 8; i += 2) {
            // x坐标：减去x偏移，除以scale
            landmarks[i] = (pitem[i] - offset_x) / scale;
            // y坐标：减去y偏移，除以scale
            landmarks[i + 1] = (pitem[i + 1] - offset_y) / scale;
        }
        
        // 第五步：从关键点计算bounding box（min/max x/y）
        // landmarks顺序：左上逆时针 [0,1] [2,3] [4,5] [6,7]
        float min_x = landmarks[0];
        float max_x = landmarks[0];
        float min_y = landmarks[1];
        float max_y = landmarks[1];
        
        for (int i = 2; i < 8; i += 2) {
            if (landmarks[i] < min_x) min_x = landmarks[i];
            if (landmarks[i] > max_x) max_x = landmarks[i];
            if (landmarks[i + 1] < min_y) min_y = landmarks[i + 1];
            if (landmarks[i + 1] > max_y) max_y = landmarks[i + 1];
        }
        
        int index = atomicAdd(parray, 1);
        if (index >= MAX_IMAGE_BOXES)
            return;
        

        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = min_x;   // left
        *pout_item++ = min_y;   // top
        *pout_item++ = max_x;   // right
        *pout_item++ = max_y;   // bottom
        *pout_item++ = max_class_score;  // confidence（使用类别分数，不是confidence）
        *pout_item++ = class_id;  // label
        *pout_item++ = 1;  // 1 = keep, 0 = ignore
        
        // 存储关键点（左上顺时针顺序）
        *pout_item++ = landmarks[0];  // 点0: [0,1]
        *pout_item++ = landmarks[1];
        *pout_item++ = landmarks[6];  // 点1: [6,7]
        *pout_item++ = landmarks[7];
        *pout_item++ = landmarks[4];  // 点2: [4,5]
        *pout_item++ = landmarks[5];
        *pout_item++ = landmarks[2];  // 点3: [2,3]
        *pout_item++ = landmarks[3];




}










static __device__ float box_iou(float aleft, float atop, float aright,
                                float abottom, float bleft, float btop,
                                float bright, float bbottom)
{
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float* bboxes, int MAX_IMAGE_BOXES,
                                       float threshold)
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, MAX_IMAGE_BOXES);
    if (position >= count)
        return;

    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5])
            continue;

        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou =
                box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                        pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold) {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

static dim3 grid_dims(int numJobs)
{
    int numBlockThreads =
        numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs)
{
    return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static void decode_kernel_invoker(float* predict, int num_bboxes,
                                  int num_classes, int output_cdim,
                                  float  confidence_threshold,
                                  float  nms_threshold,
                                  float* invert_affine_matrix,
                                  float* parray, int MAX_IMAGE_BOXES,
                                  Type type, cudaStream_t stream,
                                  float scale = 1.0f, float offset_x = 0.0f, float offset_y = 0.0f, int detect_color = 0)
{
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    if (type == Type::V8 || type == Type::V8Seg) {
        checkKernel(decode_kernel_v8<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray,
            MAX_IMAGE_BOXES));
    } else if (type == Type::V5Face) {
        checkKernel(decode_kernel_v5_face<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray,
            MAX_IMAGE_BOXES));
    } else if (type == Type::V0708) {
        checkKernel(decode_kernel_V0708<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, scale, offset_x, offset_y, detect_color, parray,
            MAX_IMAGE_BOXES));
    } 
        else {
        checkKernel(decode_kernel_common<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray,
            MAX_IMAGE_BOXES));
    }

    grid = grid_dims(MAX_IMAGE_BOXES);
    block = block_dims(MAX_IMAGE_BOXES);
    // nms
    checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(
        parray, MAX_IMAGE_BOXES, nms_threshold));
}

static __global__ void
decode_single_mask_kernel(int left, int top, float* mask_weights,
                          float* mask_predict, int mask_width,
                          int mask_height, unsigned char* mask_out,
                          int mask_dim, int out_width, int out_height)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= out_width || dy >= out_height)
        return;

    int sx = left + dx;
    int sy = top + dy;
    if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
        mask_out[dy * out_width + dx] = 0;
        return;
    }

    float cumprod = 0;
    for (int ic = 0; ic < mask_dim; ++ic) {
        float cval =
            mask_predict[(ic * mask_height + sy) * mask_width + sx];
        float wval = mask_weights[ic];
        cumprod += cval * wval;
    }

    float alpha = 1.0f / (1.0f + exp(-cumprod));
    mask_out[dy * out_width + dx] = alpha * 255;
}

static void decode_single_mask(float left, float top, float* mask_weights,
                               float* mask_predict, int mask_width,
                               int mask_height, unsigned char* mask_out,
                               int mask_dim, int out_width, int out_height,
                               cudaStream_t stream)
{
    // mask_weights is mask_dim(32 element) gpu pointer
    dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
        left, top, mask_weights, mask_predict, mask_width, mask_height,
        mask_out, mask_dim, out_width, out_height));
}

const char* type_name(Type type)
{
    switch (type) {
    case Type::V5:
        return "YoloV5";
    case Type::V3:
        return "YoloV3";
    case Type::V7:
        return "YoloV7";
    case Type::X:
        return "YoloX";
    case Type::V8:
        return "YoloV8";
    case Type::V5Face:
        return "YoloV5Face";
    case Type::V0708:
        return "Yolo0708";
    default:
        return "Unknow";
    }
}

struct AffineMatrix {
    float i2d[6];
    float d2i[6];

    void compute(const std::tuple<int, int>& from,
                 const std::tuple<int, int>& to)
    {
        float scale_x = get<0>(to) / (float)get<0>(from);
        float scale_y = get<1>(to) / (float)get<1>(from);
        float scale = std::min(scale_x, scale_y);
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 +
                 scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 +
                 scale * 0.5 - 0.5;

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D,
               A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};

InstanceSegmentMap::InstanceSegmentMap(int width, int height)
{
    this->width = width;
    this->height = height;
    checkRuntime(cudaMallocHost(&this->data, width * height));
}

InstanceSegmentMap::~InstanceSegmentMap()
{
    if (this->data) {
        checkRuntime(cudaFreeHost(this->data));
        this->data = nullptr;
    }
    this->width = 0;
    this->height = 0;
}

/**
 * @brief YOLO 推理器的具体实现类
 * 
 * 继承自 Infer<BoxArray>，实现 YOLO 目标检测模型的推理功能
 * 包括模型加载、图像预处理、TensorRT 推理、后处理等完整流程
 */
class InferImpl : public Infer<BoxArray> {
public:
    // ========== TensorRT 相关 ==========
    shared_ptr<trt::Infer> trt_;  // TensorRT 推理引擎，用于执行模型推理
    
    // ========== 模型配置 ==========
    string engine_file_;           // TensorRT 引擎文件路径（.engine 文件）
    Type   type_;                  // YOLO 模型类型（V5/V8/V8Seg/V5Face 等）
    
    // ========== 检测阈值 ==========
    float confidence_threshold_;   // 置信度阈值，低于此值的检测框将被过滤
    float nms_threshold_;          // 非极大值抑制（NMS）阈值，用于去除重叠的检测框
    
    // ========== 内存管理 ==========
    vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;  // 预处理缓冲区，用于图像预处理时的临时内存
    trt::Memory<float> input_buffer_;      // 输入缓冲区，存储预处理后的图像数据（GPU）
    trt::Memory<float> bbox_predict_;      // 边界框预测输出缓冲区，存储模型原始输出（GPU）
    trt::Memory<float> output_boxarray_;  // 最终检测结果缓冲区，存储解码后的检测框（GPU/CPU）
    trt::Memory<float> segment_predict_;   // 分割预测输出缓冲区，用于实例分割模型（GPU）
    vector<shared_ptr<trt::Memory<unsigned char>>> box_segment_cache_;  // 分割掩码缓存，用于存储每个检测框的分割结果
    
    // ========== 网络配置 ==========
    int network_input_width_;   // 网络输入图像宽度（模型要求的输入尺寸）
    int network_input_height_;  // 网络输入图像高度（模型要求的输入尺寸）
    Norm normalize_;            // 归一化配置，包含均值、标准差等归一化参数
    
    // ========== 输出维度 ==========
    vector<int> bbox_head_dims_;    // 边界框输出头的维度 [batch, num_bboxes, num_classes+5]
    vector<int> segment_head_dims_;  // 分割输出头的维度 [batch, num_prototypes, mask_h, mask_w]
    
    // ========== 模型特性 ==========
    int  num_classes_ = 0;        // 检测类别数量
    bool has_segment_ = false;    // 是否支持实例分割（如 YOLO V8Seg）
    bool has_keyPoint = false;    // 是否支持关键点检测（如 YOLO V5Face）
    bool isdynamic_model_ = false; // 是否为动态输入尺寸模型
    
    // ========== 0708模型专用 ==========
    vector<float> scales_;        // 存储每个batch的scale值（用于0708模型后处理坐标还原）

    virtual ~InferImpl() = default;

    // 调整内存，根据批量大小调整内存大小
    void adjust_memory(int batch_size)
    {
        size_t input_numel =
            network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] *
                          bbox_head_dims_[2]);
        output_boxarray_.gpu(batch_size *
                             (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
        output_boxarray_.cpu(batch_size *
                             (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

        if (has_segment_)
            segment_predict_.gpu(batch_size * segment_head_dims_[1] *
                                 segment_head_dims_[2] *
                                 segment_head_dims_[3]);

        if ((int)preprocess_buffers_.size() < batch_size) {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                preprocess_buffers_.push_back(
                    make_shared<trt::Memory<unsigned char>>());
        }
        
        // ========== 0708模型：预分配scales_空间 ==========
        if (type_ == Type::V0708) {
            scales_.resize(batch_size);
        }
        // ===============================================
    }

    // 预处理 
    // 输入批次索引、图像、预处理缓冲区、仿射矩阵、流
    void
    preprocess(int ibatch, const Image& image,
               shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
               AffineMatrix& affine, void* stream = nullptr)
    {
        affine.compute(
            make_tuple(image.width, image.height),
            make_tuple(network_input_width_, network_input_height_));

        // ========== 0708模型：保存scale值用于后处理坐标还原 ==========
        if (type_ == Type::V0708) {
            float scale = affine.i2d[0];  // scale值存储在i2d[0]
            if (scales_.size() <= ibatch) {
                scales_.resize(ibatch + 1);
            }
            scales_[ibatch] = scale;
        }
        // ============================================================

        size_t input_numel =
            network_input_width_ * network_input_height_ * 3;
        float*   input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t   size_image = image.width * image.height * 3;
        size_t   size_matrix = upbound(sizeof(affine.d2i), 32);
        uint8_t* gpu_workspace =
            preprocess_buffer->gpu(size_matrix + size_image);
        float*   affine_matrix_device = (float*)gpu_workspace;
        uint8_t* image_device = gpu_workspace + size_matrix;

        uint8_t* cpu_workspace =
            preprocess_buffer->cpu(size_matrix + size_image);
        float*   affine_matrix_host = (float*)cpu_workspace;
        uint8_t* image_host = cpu_workspace + size_matrix;

        cudaStream_t stream_ = (cudaStream_t)stream;
        std::chrono ::high_resolution_clock::time_point a1 =
            std::chrono::high_resolution_clock::now();
        memcpy(image_host, image.bgrptr, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        std::chrono ::high_resolution_clock::time_point a2 =
            std::chrono::high_resolution_clock::now();
        auto time_used2 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a2 -
                                                                      a1);
        checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image,
                                     cudaMemcpyHostToDevice, stream_));
        checkRuntime(cudaMemcpyAsync(affine_matrix_device,
                                     affine_matrix_host, sizeof(affine.d2i),
                                     cudaMemcpyHostToDevice, stream_));
        
        // ========== 根据模型类型选择填充值 ==========
        // 0708模型使用黑色填充(0)，其他模型使用114
        uint8_t pad_value = (type_ == Type::V0708) ? 0 : 114;
        warp_affine_bilinear_and_normalize_plane(
            image_device, image.width * 3, image.width, image.height,
            input_device, network_input_width_, network_input_height_,
            affine_matrix_device, pad_value, normalize_, stream_);
        // ==========================================
    }

    
    // impl 加载参数
    bool load(const string& engine_file, Type type,
              float confidence_threshold, float nms_threshold)
    {
        trt_ = trt::load(engine_file);
        if (trt_ == nullptr)
            return false;

        trt_->print();

        this->type_ = type;
        this->confidence_threshold_ = confidence_threshold;
        this->nms_threshold_ = nms_threshold;

        auto input_dim = trt_->static_dims(0);//输入维度
        bbox_head_dims_ = trt_->static_dims(1);//边界框输出头的维度

        has_segment_ = type == Type::V8Seg;//是否支持实例分割
        has_keyPoint = (type == Type::V5Face) || (type == Type::V0708);//是否支持关键点检测
        if (has_segment_) {
            bbox_head_dims_ = trt_->static_dims(2);
            segment_head_dims_ = trt_->static_dims(1);
        }
        network_input_width_ = input_dim[3];//网络输入宽度
        network_input_height_ = input_dim[2];//网络输入高度
        isdynamic_model_ = trt_->has_dynamic_dim();//是否支持动态输入尺寸

        // 归一化配置 类别数量
        if (type == Type::V5 || type == Type::V3 || type == Type::V7) {
            normalize_ =
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 5;
        } else if (type == Type::V8) {
            normalize_ =
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4;
        } else if (type == Type::V8Seg) {
            normalize_ =
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
        } else if (type == Type::X) {
            normalize_ = Norm::None();
            num_classes_ = bbox_head_dims_[2] - 5;
        } else if (type == Type::V5Face) {
            normalize_ =
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 5 - 2 * point_num;
        } else if (type == Type::V0708) {
            normalize_ =
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            // 0708模型输出格式：[0-7:关键点, 8:confidence, 9-12:颜色, 13-21:类别]
            // 所以 num_classes = output_cdim - 13 = 22 - 13 = 9
            num_classes_ = 9;  // 0708模型固定9个类别
        } else {
            INFO("Unsupport type %d", type);
        }
        return true;


    }

    // 前向推理
    // 输入图像、流
    // 返回框数组
    // {image}
    virtual BoxArray forward(const Image& image,
                             void*        stream = nullptr) override
    {
        auto output = forwards({image}, stream);
        if (output.empty())
            return {};
        return output[0];
    }

    virtual vector<BoxArray> forwards(const vector<Image>& images,
                                      void* stream = nullptr) override
    {
        int num_image = images.size();
        if (num_image == 0)
            return {};
        // 处理图片动态数量与模型 batch size 的匹配。
        //处理实际输入图片数量与模型 batch size 的匹配。
        auto input_dims = trt_->static_dims(0);//输入维度
        int  infer_batch_size = input_dims[0];//推理批量大小
        if (infer_batch_size != num_image) {
            if (isdynamic_model_) {
                infer_batch_size = num_image;
                input_dims[0] = num_image;
                if (!trt_->set_run_dims(0, input_dims))
                    return {};
            } else {
                if (infer_batch_size < num_image) {
                    INFO("When using static shape model, number of "
                         "images[%d] must be "
                         "less than or equal to the maximum batch[%d].",
                         num_image, infer_batch_size);
                    return {};
                }
            }
        }

        std::chrono ::high_resolution_clock::time_point a1 =
            std::chrono::high_resolution_clock::now();
        adjust_memory(infer_batch_size);//调整内存 根据批量大小，为输入、输出、结果和预处理分配 GPU/CPU 内存。
        std::chrono ::high_resolution_clock::time_point a2 =
            std::chrono::high_resolution_clock::now();
        auto time_used2 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a2 -
                                                                      a1);
        // 将图像转换为网络
        vector<AffineMatrix> affine_matrixs(num_image);//仿射矩阵
        std::chrono ::high_resolution_clock::time_point a3 =
            std::chrono::high_resolution_clock::now();

        // 预处理 对每个图片进行仿射变换和归一化处理
        // cuda 流           
        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i)
            preprocess(i, images[i], preprocess_buffers_[i],
                       affine_matrixs[i], stream);

        float*        bbox_output_device = bbox_predict_.gpu();
        vector<void*> bindings{input_buffer_.gpu(), bbox_output_device};

        if (has_segment_) {
            bindings = {input_buffer_.gpu(), segment_predict_.gpu(),
                        bbox_output_device};
        }
        std::chrono ::high_resolution_clock::time_point a3d1 =
            std::chrono::high_resolution_clock::now();
        auto time_used3 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a3d1 -
                                                                      a3);
        
        // 执行模型推理 
        if (!trt_->forward(bindings, stream)) {
            INFO("Failed to tensorRT forward.");
            return {};
        }
        std::chrono ::high_resolution_clock::time_point a3d2 =
            std::chrono::high_resolution_clock::now();
        auto time_used3d2 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a3d2 -
                                                                      a3d1);
        // 理解为后处理把
        // 解码 对每个图片进行解码
        // 输入边界框输出、边界框头维度、类别数量、置信度阈值、非极大值抑制阈值、仿射矩阵、边界框数组、最大检测框数量、模型类型、流
        // 输出边界框数组
        for (int ib = 0; ib < num_image; ++ib) {
            // 创建输出缓存区域
            // 输出缓存区域 包含边界框数组、仿射矩阵、图像基于边界框输出
            float* boxarray_device =
                output_boxarray_.gpu() +
                ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            // 网络到图像的仿射矩阵
                float* affine_matrix_device =
                (float*)preprocess_buffers_[ib]->gpu();
            // 图像基于边界框输出
                float* image_based_bbox_output =
                bbox_output_device +
                ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
            // 清空输出缓存区域
            checkRuntime(
                cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
            
            // 解码核心
            // 对于0708模型，需要传递scale、偏移量和detect_color参数
            if (type_ == Type::V0708) {
                float scale = (ib < scales_.size()) ? scales_[ib] : 1.0f;
                // 获取仿射变换的偏移量（i2d[2]和i2d[5]）
                float offset_x = affine_matrixs[ib].i2d[2];
                float offset_y = affine_matrixs[ib].i2d[5];
                int detect_color = 0;  // 默认检测所有颜色，可以根据需要修改
                decode_kernel_invoker(
                    image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                    bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                    affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES,
                    type_, stream_, scale, offset_x, offset_y, detect_color);
            } else {
                decode_kernel_invoker(
                    image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                    bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                    affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES,
                    type_, stream_);
            }
        }
        std::chrono ::high_resolution_clock::time_point a3d3 =
            std::chrono::high_resolution_clock::now();
        auto time_used3d3 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a3d3 -
                                                                      a3d2);
        std::chrono ::high_resolution_clock::time_point a3d4 =
            std::chrono::high_resolution_clock::now();
        // 将输出缓存区域从 GPU 复制到 CPU
        checkRuntime(cudaMemcpyAsync(
            output_boxarray_.cpu(), output_boxarray_.gpu(),
            output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        // 等待流完成
        checkRuntime(cudaStreamSynchronize(stream_));
        std::chrono ::high_resolution_clock::time_point a4 =
            std::chrono::high_resolution_clock::now();
        auto time_used3d4 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a4 -
                                                                      a3d4);
        auto time_used4 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a4 -
                                                                      a3);
        INFO("forward and decode_kernel_invoker time: %f",
             time_used4.count() * 1000);
        std::chrono ::high_resolution_clock::time_point a5 =
            std::chrono::high_resolution_clock::now();


        // 转换 为 BoxArray
        vector<BoxArray> arrout(num_image);
        int              imemory = 0;
        for (int ib = 0; ib < num_image; ++ib) {
            float* parray = output_boxarray_.cpu() +
                            ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            int       count = min(MAX_IMAGE_BOXES, (int)*parray);
            BoxArray& output = arrout[ib];
            output.reserve(count);
            for (int i = 0; i < count; ++i) {
                float*             pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                int                a1 = pbox[0];
                int                a2 = pbox[1];
                int                a3 = pbox[2];
                int                a4 = pbox[3];
                int                a5 = pbox[4];
                int                label = pbox[5];
                int                keepflag = pbox[6];
                std::vector<float> points;
                for (int i = 1; i < 9; i++) {
                    points.push_back(pbox[6 + i]);
                }
                if (keepflag == 1) {
                    Box result_object_box(pbox[0], pbox[1], pbox[2],
                                          pbox[3], pbox[4], label, points);

                    if (has_segment_) {
                        int    row_index = pbox[7];
                        int    mask_dim = segment_head_dims_[1];
                        float* mask_weights =
                            bbox_output_device +
                            (ib * bbox_head_dims_[1] + row_index) *
                                bbox_head_dims_[2] +
                            num_classes_ + 4;

                        float* mask_head_predict = segment_predict_.gpu();
                        float  left, top, right, bottom;
                        float* i2d = affine_matrixs[ib].i2d;
                        affine_project(i2d, pbox[0], pbox[1], &left, &top);
                        affine_project(i2d, pbox[2], pbox[3], &right,
                                       &bottom);

                        float box_width = right - left;
                        float box_height = bottom - top;

                        float scale_to_predict_x =
                            segment_head_dims_[3] /
                            (float)network_input_width_;
                        float scale_to_predict_y =
                            segment_head_dims_[2] /
                            (float)network_input_height_;
                        int mask_out_width =
                            box_width * scale_to_predict_x + 0.5f;
                        int mask_out_height =
                            box_height * scale_to_predict_y + 0.5f;

                        if (mask_out_width > 0 && mask_out_height > 0) {
                            if (imemory >= (int)box_segment_cache_.size()) {
                                box_segment_cache_.push_back(
                                    std::make_shared<
                                        trt::Memory<unsigned char>>());
                            }

                            int bytes_of_mask_out =
                                mask_out_width * mask_out_height;
                            auto box_segment_output_memory =
                                box_segment_cache_[imemory];
                            result_object_box.seg =
                                make_shared<InstanceSegmentMap>(
                                    mask_out_width, mask_out_height);

                            unsigned char* mask_out_device =
                                box_segment_output_memory->gpu(
                                    bytes_of_mask_out);
                            unsigned char* mask_out_host =
                                result_object_box.seg->data;
                            decode_single_mask(
                                left * scale_to_predict_x,
                                top * scale_to_predict_y, mask_weights,
                                mask_head_predict +
                                    ib * segment_head_dims_[1] *
                                        segment_head_dims_[2] *
                                        segment_head_dims_[3],
                                segment_head_dims_[3],
                                segment_head_dims_[2], mask_out_device,
                                mask_dim, mask_out_width, mask_out_height,
                                stream_);
                            checkRuntime(cudaMemcpyAsync(
                                mask_out_host, mask_out_device,
                                box_segment_output_memory->gpu_bytes(),
                                cudaMemcpyDeviceToHost, stream_));
                        }
                    }
                    output.emplace_back(result_object_box);
                }
            }
        }
        std::chrono ::high_resolution_clock::time_point a6 =
            std::chrono::high_resolution_clock::now();
        auto time_used6 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a6 -
                                                                      a5);
        if (has_segment_)
            checkRuntime(cudaStreamSynchronize(stream_));

        return arrout;
    }
};

Infer<BoxArray>* loadraw(const std::string& engine_file, Type type,
                         float confidence_threshold, float nms_threshold)
{
    InferImpl* impl = new InferImpl();
    if (!impl->load(engine_file, type, confidence_threshold,
                    nms_threshold)) {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

shared_ptr<Infer<BoxArray>> load(const string& engine_file, Type type,
                                 float confidence_threshold,
                                 float nms_threshold)
{
    return std::shared_ptr<InferImpl>((InferImpl*)loadraw(
        engine_file, type, confidence_threshold, nms_threshold));
}

// 不重要的东西
// 为每个id生成一个随机颜色
// HSV转BGR
std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int   h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float       r, g, b;
    switch (h_i) {
    case 0:
        r = v, g = t, b = p;
        break;
    case 1:
        r = q, g = v, b = p;
        break;
    case 2:
        r = p, g = v, b = t;
        break;
    case 3:
        r = p, g = q, b = v;
        break;
    case 4:
        r = t, g = p, b = v;
        break;
    case 5:
        r = v, g = p, b = q;
        break;
    default:
        r = 1, g = 1, b = 1;
        break;
    }
    return make_tuple(static_cast<uint8_t>(b * 255),
                      static_cast<uint8_t>(g * 255),
                      static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

};  // namespace yolo