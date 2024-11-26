# 损失函数代码库汇总
## 损失函数

### 复合损失函数
### (instance imbalance aware loss functions for semantic segmentation)
## 函数说明
### 辅助函数
- **`vprint(*args, verbose=True)`**  
  打印调试信息的辅助函数，用于根据 `verbose` 参数控制调试输出。

### 损失计算函数
1. **`compute_compound_loss`**  
   - **功能**: 计算复合损失函数，支持多种损失函数的加权组合。
   - **参数**:
     - `criterion_dict`: 损失函数配置字典。
     - `raw_network_outputs`: 网络的原始输出。
     - `label`: 真实标签。
     - `blob_loss_mode`: 是否计算 blob（区域）损失。
     - `masked`: 是否在 blob 损失中应用掩码。
     - `verbose`: 是否启用调试打印。
   - **返回值**: 计算的总损失值。

2. **`compute_blob_loss_multi`**  
   - **功能**: 计算每个 batch 元素的 blob（区域）损失。
   - **参数**:
     - `criterion`: 损失函数。
     - `network_outputs`: 网络输出张量。
     - `multi_label`: 每个像素的多标签。
     - `verbose`: 是否启用调试打印。
   - **返回值**: 平均 blob 损失值。

3. **`compute_no_masking_multi`**  
   - **功能**: 计算不使用掩码的 blob 损失。
   - **参数**:
     - `criterion`: 损失函数。
     - `network_outputs`: 网络输出张量。
     - `multi_label`: 每个像素的多标签。
     - `verbose`: 是否启用调试打印。
   - **返回值**: 平均 blob 损失值。

4. **`compute_loss`**  
   - **功能**: 计算总损失，包括全局主损失和局部 blob 损失。
   - **参数**:
     - `blob_loss_dict`: 包含主损失和 blob 损失的权重配置。
     - `criterion_dict`: 用于主损失的损失函数配置字典。
     - `blob_criterion_dict`: 用于 blob 损失的损失函数配置字典。
     - `raw_network_outputs`: 网络的原始输出。
     - `binary_label`: 全局二值标签。
     - `multi_label`: 多标签。
     - `verbose`: 是否启用调试打印。
   - **返回值**: 一个元组 `(总损失, 主损失, blob 损失)`。


### 出处
[索引](https://github.com/neuronflow/blob_loss/)

(https://arxiv.org/abs/2205.08209)

---
### cross_entropy_loss
## 函数名称

1. **`__init__(self, weight=None, ignore_index=255)`**
   - 初始化类的参数：
     - `weight`: 可选，类别权重，用于类别不平衡时的损失调整。
     - `ignore_index`: 忽略类别的索引，默认为 255。

2. **`forward(self, inputs, targets)`**
   - 计算损失值：
     - `inputs`: 网络的输出张量，形状为 `(batch_size, num_classes, H, W)`。
     - `targets`: 目标标签张量，形状为 `(batch_size, H, W)`。
     - 返回值为计算的损失张量。

### 出处
[索引](https://github.com/yutao1008/margin_calibration)

---
###  dice_loss
## 函数名称

### 方法
1. **`__init__(self, smooth=1e-5)`**
   - 初始化类的参数：
     - `smooth`: 平滑因子，默认值为 1e-5，用于避免分母为零的问题。

2. **`forward(self, net_output, gt)`**
   - 计算广义Dice损失：
     - `net_output`: 网络输出张量，形状为 `(batch_size, num_classes, H, W)`。
     - `gt`: 目标标签张量，形状为 `(batch_size, H, W)`。
     - 返回值为一个标量损失值，表示网络输出与目标之间的差距。

### 出处
[索引](https://github.com/yutao1008/margin_calibration)

---
###  focal_loss
## 函数名称

1. **`__init__(self, gamma=2, alpha=None)`**
   - 初始化类的参数：
     - `gamma`: 调节因子，默认值为 2，控制对易分类样本的损失权重。
     - `alpha`: 类别权重，默认为 `None`，可以设置为浮点数或类别权重列表。

2. **`forward(self, inputs, targets)`**
   - 计算 Focal Loss：
     - `inputs`: 网络输出张量，形状为 `(batch_size, num_classes, H, W)`。
     - `targets`: 目标标签张量，形状为 `(batch_size, H, W)`。
     - 返回值为 Focal Loss 的标量值。

### 使用场景
- Focal Loss 适用于类别分布不平衡的任务（例如目标检测或语义分割），通过调节 `gamma` 和 `alpha` 来优化模型性能。
- `gamma` 控制易分类样本的损失权重，`alpha` 可以设置类别权重。

### 出处
[索引](https://github.com/yutao1008/margin_calibration)

---
###  lovasz_loss
## 简介
本库实现了 Lovasz 损失函数，特别适用于优化交并比（IoU）指标的语义分割任务。支持二值分割和多类分割。
### 函数
### 二值损失函数
- **`lovasz_hinge`**: 二值分割的 Lovasz Hinge 损失。
- **`binary_xloss`**: 二值交叉熵损失。

### 多类损失函数
- **`lovasz_softmax`**: 多类分割的 Lovasz Softmax 损失。
- **`xloss`**: 多类交叉熵损失。

### 辅助函数
- **`lovasz_grad`**: 计算 Lovasz 损失的梯度。
- **`iou_binary`**: 计算二值 IoU。
- **`iou`**: 计算多类 IoU。
- **`flatten_binary_scores`**: 将二值预测和标签扁平化。
- **`flatten_probas`**: 将多类概率和标签扁平化。

### 出处
[索引](https://github.com/yutao1008/margin_calibration)

---
###  margin_loss
### 函数名称
- **`MarginLogLoss`**:     带边界约束的损失函数

### 出处
[索引](https://github.com/yutao1008/margin_calibration)

---
###  nrdice_loss
### 函数名称
- **`NR_DiceLoss`**:        噪声鲁棒的Dice损失

### 出处
[索引](https://github.com/yutao1008/margin_calibration)

---
###  tversky_loss
### 函数名称
- **`TverskyLoss`**:       Tversky损失
    
### 出处
[索引](https://github.com/yutao1008/margin_calibration)

---
###  soft_dice_loss
### 函数名称
- **`SoftDiceLoss`**:      Soft Dice损失

### 出处
[索引](https://github.com/by-liu/SegLossBias)

---
### custom_loss
### 函数名称
- **`CrossEntropyWithL1`**:       交叉熵损失与L1正则化
- **`CrossEntropyWithKL`**:       交叉熵损失与KL散度

### 出处
[索引](https://arxiv.org/abs/2007.10033)

---
### clDice_Loss
### 函数名称
- **`clDiceLoss`**:       clDice损失

### 出处
[索引](https://github.com/jocpae/clDice)
https://arxiv.org/abs/2003.07311

---
###  topoloss
### 函数名称
- **`TopoLoss`**:       拓扑损失
内容：本库实现了拓扑损失函数，用于保持图像分割结果的拓扑结构。支持 PyTorch 框架。

### 出处
[索引](https://github.com/HuXiaoling/TopoLoss)

---

### Pyramid_Loss
### 函数名称
- **`PyramidLoss`**:       金字塔损失
- 内容：本库实现了金字塔损失函数，用于语义分割任务。支持 PyTorch 框架。
- 输入：预测结果，真实标签
- 输出：损失值

### 出处
[索引](https://github.com/ZJULearning/RMI)https://arxiv.org/abs/1910.12037

---
###  LA_Hausdorff_Loss
### 函数名称
- **`hd_loss `**:       局部-全局哈尔斯多夫损失
- 内容：本库实现了局部-全局哈尔斯多夫损失函数，用于语义分割任务。支持 PyTorch 框架。
- 输入：预测结果，真实标签
- 输出：损失值

### 出处
[索引](https://github.com/JunMa11/SegWithDistMap/blob/5a67153bc730eb82de396ef63f57594f558e23cd/code/train_LA_HD.py#L106)

---
###  LovaszSoftmax_Loss
### 函数名称
- **`LovaszHingeLoss`**:       Lovasz-Softmax损失
- 内容：本库实现了 Lovasz-Softmax 损失函数，用于优化交并比（IoU）指标的语义分割任务。支持 PyTorch 和 TensorFlow 框架。

### 出处
[索引](https://github.com/bermanmaxim/LovaszSoftmax)