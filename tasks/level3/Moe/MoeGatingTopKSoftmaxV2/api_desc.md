# MoeGatingTopKSoftmaxV2

## 功能描述

### 算子功能
MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取topK操作；如果renorm=1，先对x的输出做topK操作，再进行Softmax操作。其中yOut为softmax的topK结果；expertIdxOut为topK的值的索引结果，即对应的专家序号；softmaxResultOutOptional为计算过程中Softmax的完整结果（当renorm=0时有效）。如果expertIdxOut中对应的行的finished为True，则直接填num_expert值（即x的最后一个轴大小）。

### 计算公式

#### 当 renorm = 0 时：

**Step 1: Softmax计算**

对输入 x 做Softmax归一化：

$$
softmaxResultOutOptional = Softmax(x, axis=-1)
$$

**Step 2: Top-K 专家选择**

对归一化结果取Top-K得到专家索引和对应值：

$$
yOut, expertIdxOut = TopK(softmaxResultOutOptional, k=k)
$$

#### 当 renorm = 1 时：

**Step 1: Top-K 专家选择**

先对输入 x 取Top-K得到专家索引和对应值：

$$
topkOut, expertIdxOut = TopK(x, k=k)
$$

**Step 2: Softmax计算**

对Top-K结果做Softmax归一化：

$$
yOut = Softmax(topkOut, axis=-1)
$$

**Step 3: 处理finished标志**

如果对应行的finished为True，将专家索引设置为num_expert：

$$
expertIdxOut = where(finished, num_expert, expertIdxOut)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_gating_top_k_softmax_v2()` 函数形式提供：

```python
def moe_gating_top_k_softmax_v2(x: Tensor, finished_optional: Tensor, k: int, renorm: int):
    """
    实现 moe_gating_top_k_softmax_v2 操作。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        finished_optional (Tensor): 可选输入张量，表示该行是否参与运算，Device侧的张量，数据格式支持ND。
        k (int): TopK的k值，专家数。
        renorm (int): Softmax和TopK的顺序，0表示先计算Softmax再计算TopK，1表示顺序相反。

    返回:
        tuple: 包含三个张量的元组，分别是：
            - 输出张量yOut，数据类型与输入x一致
            - 专家索引张量expertIdxOut，数据类型为int32
            - Softmax结果张量softmaxResultOutOptional，数据类型为float32（仅当renorm=0时有效）
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn([5, 3], dtype=torch.float32)
finished_optional = torch.randint(0, 2, (5,), dtype=torch.bool)
k = 3
renorm = 0  # 先计算Softmax，再计算TopK

# 调用算子
out, indices, softmax_result = kernel_gen_ops.moe_gating_top_k_softmax_v2(x, finished_optional, k, renorm)

print("Output shape:", out.shape)
print("Indices shape:", indices.shape)
print("Softmax result shape:", softmax_result.shape)

# 使用renorm=1的情况
renorm = 1  # 先计算TopK，再计算Softmax
out, indices, softmax_result = kernel_gen_ops.moe_gating_top_k_softmax_v2(x, finished_optional, k, renorm)

print("\nWith renorm=1:")
print("Output shape:", out.shape)
print("Indices shape:", indices.shape)
print("Softmax result shape:", softmax_result.shape)
```

## 约束与限制

- 输入 x 的数据类型支持 FLOAT16、BFLOAT16、FLOAT32。
- 输入 x 的维度支持 2D-3D，每一维大小应不大于int32的最大值2147483647。
- 输入 finished_optional 的shape要求为 x_shape[:-1]，数据类型为BOOL。
- 要求 0 ≤ k ≤ x的最后一维大小，且k不大于1024。
- renorm参数只支持取值0和1：
  - 0：表示先计算Softmax，再计算TopK
  - 1：表示先计算TopK，再计算Softmax
- 输出张量yOut的非最后一维与x的对应轴大小一致，最后一维大小为k。
- 输出张量expertIdxOut的shape与yOut一致，数据类型为INT32。
- 输出张量softmaxResultOutOptional的shape与x一致，数据类型为FLOAT32。
- 当renorm=0时，softmaxResultOutOptional包含完整的Softmax结果；当renorm=1时，该参数不生效。
- 如果expertIdxOut中对应的行的finished为True，则直接填num_expert值（即x的最后一个轴大小）。