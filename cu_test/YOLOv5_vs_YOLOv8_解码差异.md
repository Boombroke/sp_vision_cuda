# YOLOv5 vs YOLOv8 解码差异详解

## 核心差异总结

| 特性 | YOLOv5 (V3/V5/V7) | YOLOv8 |
|------|-------------------|--------|
| **输出格式** | `[cx, cy, w, h, obj, cls0, cls1, ..., clsN-1]` | `[cx, cy, w, h, cls0, cls1, ..., clsN-1]` |
| **Objectness** | ✅ 有独立的 objectness 分支 | ❌ 没有 objectness，直接使用类别分数 |
| **置信度计算** | `confidence = obj × max(cls)` | `confidence = max(cls)` |
| **类别数计算** | `num_classes = output_dim - 5` | `num_classes = output_dim - 4` |
| **解码 Kernel** | `decode_kernel_common` | `decode_kernel_v8` |

---

## 详细代码对比

### 1. 输出维度差异

#### YOLOv5 输出格式
```
每个 anchor 的输出维度 = 5 + num_classes
例如 COCO (80类): 5 + 80 = 85 维

布局: [cx, cy, w, h, obj, cls0, cls1, ..., cls79]
索引:   0   1   2  3   4    5    6   ...   84
```

#### YOLOv8 输出格式
```
每个 anchor 的输出维度 = 4 + num_classes
例如 COCO (80类): 4 + 80 = 84 维

布局: [cx, cy, w, h, cls0, cls1, ..., cls79]
索引:   0   1   2  3    4    5   ...   83
```

**关键差异**: YOLOv8 **少了 objectness 这一维**！

---

### 2. 解码流程对比

#### YOLOv5 解码流程 (`decode_kernel_common`)

```cpp
// 步骤 1: 读取预测值
float* pitem = predict + output_cdim * position;
float cx = pitem[0];        // 中心点 x
float cy = pitem[1];        // 中心点 y
float width = pitem[2];     // 宽度
float height = pitem[3];    // 高度
float objectness = pitem[4]; // ⭐ objectness 分数

// 步骤 2: 先用 objectness 过滤
if (objectness < confidence_threshold)
    return;  // 如果 objectness 太低，直接丢弃

// 步骤 3: 找最大类别分数
float* class_confidence = pitem + 5;  // 跳过前 5 个元素
float confidence = *class_confidence++;
int label = 0;
for (int i = 1; i < num_classes; ++i) {
    if (class_confidence[i] > confidence) {
        confidence = class_confidence[i];
        label = i;
    }
}

// 步骤 4: ⭐ 最终置信度 = objectness × 类别置信度
confidence *= objectness;
if (confidence < confidence_threshold)
    return;

// 步骤 5: 坐标转换和写入
// ... (后续相同)
```

#### YOLOv8 解码流程 (`decode_kernel_v8`)

```cpp
// 步骤 1: 读取预测值
float* pitem = predict + output_cdim * position;
// ⚠️ 注意：没有 objectness，直接从第 4 个元素开始是类别分数

// 步骤 2: ⭐ 直接找最大类别分数（没有 objectness 过滤）
float* class_confidence = pitem + 4;  // 跳过前 4 个元素 (cx, cy, w, h)
float confidence = *class_confidence++;
int label = 0;
for (int i = 1; i < num_classes; ++i) {
    if (class_confidence[i] > confidence) {
        confidence = class_confidence[i];
        label = i;
    }
}

// 步骤 3: ⭐ 直接使用类别分数作为置信度（不乘以 objectness）
if (confidence < confidence_threshold)
    return;

// 步骤 4: 读取 bbox 坐标
float cx = *pitem++;
float cy = *pitem++;
float width = *pitem++;
float height = *pitem++;

// 步骤 5: 坐标转换和写入
// ... (后续相同)
```

---

## 关键差异点详解

### 差异 1: Objectness 分支

#### YOLOv5 的设计思路
```
网络输出两个分支：
1. Objectness 分支: 预测"这里是否有目标"（二分类）
2. Classification 分支: 预测"目标是什么类别"（多分类）

最终置信度 = P(有目标) × P(是某类别)
         = objectness × class_confidence
```

**优点**:
- 可以提前过滤掉"没有目标"的区域，减少计算
- Objectness 和分类解耦，训练更稳定

**缺点**:
- 需要额外的 objectness 分支，增加模型复杂度
- 两个分支可能不一致（objectness 高但分类低，或反之）

#### YOLOv8 的设计思路
```
网络只输出一个分支：
- Classification 分支: 直接预测类别概率（包含"背景"类别）

最终置信度 = max(class_confidence)
```

**优点**:
- 模型更简洁，少一个输出维度
- 类别分数直接反映置信度，更直观
- 训练时不需要平衡 objectness 和分类损失

**缺点**:
- 无法提前用 objectness 过滤，所有 anchor 都要计算类别分数

---

### 差异 2: 置信度阈值的使用

#### YOLOv5: 两次过滤
```cpp
// 第一次过滤：objectness 阈值
if (objectness < confidence_threshold)  // 例如 0.25
    return;

// 第二次过滤：最终置信度阈值
confidence = objectness * max_class;
if (confidence < confidence_threshold)  // 例如 0.45
    return;
```

**实际效果**:
- 如果 `confidence_threshold = 0.45`，objectness 至少要 `0.45` 才能通过
- 即使类别分数很高（如 0.9），如果 objectness 低（如 0.4），也会被过滤

#### YOLOv8: 一次过滤
```cpp
// 只有一次过滤：类别分数阈值
confidence = max_class;
if (confidence < confidence_threshold)  // 例如 0.45
    return;
```

**实际效果**:
- 直接使用类别分数，更简单直接
- 如果某个 anchor 的类别分数 > 0.45，就保留

---

### 差异 3: 类别数计算

在 `load` 函数中：

#### YOLOv5
```cpp
if (type == Type::V5 || type == Type::V3 || type == Type::V7) {
    num_classes_ = bbox_head_dims_[2] - 5;  // 85 - 5 = 80
}
```

#### YOLOv8
```cpp
else if (type == Type::V8) {
    num_classes_ = bbox_head_dims_[2] - 4;  // 84 - 4 = 80
}
```

---

## 实际影响

### 1. 性能影响

**YOLOv5**:
- 可以先过滤 objectness < threshold 的 anchor，减少后续计算
- 但需要额外的 objectness 分支计算

**YOLOv8**:
- 所有 anchor 都要计算类别分数，无法提前过滤
- 但模型更简洁，推理可能更快

### 2. 检测质量影响

**YOLOv5**:
- Objectness 和分类解耦，可能更稳定
- 但两个分支不一致时可能漏检

**YOLOv8**:
- 类别分数直接反映置信度，更直观
- 但可能对低置信度目标更敏感

### 3. 代码复杂度

**YOLOv5** (`decode_kernel_common`):
```cpp
// 需要处理 objectness
float objectness = pitem[4];
if (objectness < threshold) return;
// ...
confidence *= objectness;
```

**YOLOv8** (`decode_kernel_v8`):
```cpp
// 直接使用类别分数，更简洁
float* class_confidence = pitem + 4;
// ...
// 不需要乘以 objectness
```

---

## 代码位置

- **YOLOv5 解码**: `yolo.cu:34-97` (`decode_kernel_common`)
- **YOLOv8 解码**: `yolo.cu:182-218` (`decode_kernel_v8`)
- **类型选择**: `yolo.cu:298-315` (`decode_kernel_invoker`)

```cpp
if (type == Type::V8 || type == Type::V8Seg) {
    decode_kernel_v8(...);  // 使用 V8 解码
} else {
    decode_kernel_common(...);  // 使用 V5/V3/V7 解码
}
```

---

## 总结

| 方面 | YOLOv5 | YOLOv8 |
|------|--------|--------|
| **输出维度** | 5 + num_classes | 4 + num_classes |
| **Objectness** | ✅ 有 | ❌ 无 |
| **置信度** | obj × cls | cls |
| **过滤策略** | 两次过滤 | 一次过滤 |
| **模型复杂度** | 更高 | 更低 |
| **代码复杂度** | 稍高 | 稍低 |

**核心思想差异**:
- **YOLOv5**: "先判断有没有目标，再判断是什么目标"（两阶段思想）
- **YOLOv8**: "直接判断是什么目标"（单阶段思想，更简洁）







