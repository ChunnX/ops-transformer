# MoeGatingTopKSoftmax

## 功能描述

### 算子功能
MoE计算中，对输入x做Softmax计算，然后对计算结果取Top-K操作。其中yOut为Softmax的Top-K结果；expertIdxOut为Top-K值的索引结果，即对应的专家序号；sourceRowOut为与expertIdxOut相同shape的张量，指示每个位置对应的原始行位置。如果expertIdxOut中对应的行的finished为True，则直接填num_expert值（即x的最后一个轴大小）。

### 计算公式

**Step 1: Softmax计算**

对输入 x 做Softmax归一化：

$$
softmaxOut = Softmax(x, axis=-1)
$$

**Step 2: Top-K 专家选择**

对归一化结果取Top-K得到专家索引和对应值：

$$
yOut, expertIdxOut = TopK(softmaxOut, k=k)
$$

**Step 3: 生成源行索引**

生成与专家索引形状相同的源行索引张量：

$$
rowIdxRange = arange(expertIdxOut.shape[0] * expertIdxOut.shape[1])
$$

$$
sourceRowOut = rowIdxRange.reshape([expertIdxOut.shape[1], expertIdxOut.shape[0]]).transpose(1, 0)
$$

**Step 4: 处理finished标志**

如果对应行的finished为True，将专家索引设置为num_expert：

$$
expertIdxOut = where(finished, num_expert, expertIdxOut)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_gating_top_k_softmax()` 函数形式提供：

```python
def moe_gating_top_k_softmax(x: Tensor, finished_optional: Tensor, k: int):
    """
    实现 moe_gating_top_k_softmax 操作。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        finished_optional (Tensor): 可选输入张量，表示该行是否参与运算，Device侧的张量，数据格式支持ND。
        k (int): TopK的k值。

    返回:
        tuple: 包含三个张量的元组，分别是：
            - 输出张量yOut，数据类型与输入x一致
            - 专家索引张量expertIdxOut，数据类型为int32
            - 源行索引张量sourceRowOut，数据类型为int32
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

# 调用算子
out, indices, source_row_out = kernel_gen_ops.moe_gating_top_k_softmax(x, finished_optional, k)

print("Output shape:", out.shape)
print("Indices shape:", indices.shape)
print("Source row out shape:", source_row_out.shape)
```

## 约束与限制

- 输入 x 的数据类型支持 FLOAT16、BFLOAT16、FLOAT32。
- 输入 x 的维度支持 2D-3D，每一维大小应不大于int32的最大值2147483647。
- 输入 x 最后一维的大小（即专家数）要求不大于2048。
- 输入 finished_optional 的shape要求为 x_shape[:-1]，数据类型为BOOL。
- 要求 0 ≤ k ≤ x的最后一维大小，且k不大于1024。
- 输出张量yOut的非最后一维与x的对应轴大小一致，最后一维大小为k。
- 输出张量expertIdxOut和sourceRowOut的shape与yOut一致。
- 如果expertIdxOut中对应的行的finished为True，则直接填num_expert值（即x的最后一个轴大小）。
