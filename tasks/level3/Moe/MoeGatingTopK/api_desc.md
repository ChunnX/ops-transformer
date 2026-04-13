# MoeGatingTopK

## 功能描述

### 算子功能
MoE计算中，对输入x做Sigmoid或者SoftMax计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。

### 计算公式

**Step 1: 归一化**

根据 normType 对输入 x 做归一化：

$$
normOut = \begin{cases}
    SoftMax(x), & normType = 0 \\
    Sigmoid(x), & normType = 1
\end{cases}
$$

**Step 2: 加偏置**

若 bias 不为空，加偏置得到用于选择的值：

$$
normValue = normOut + bias
$$

否则 $normValue = normOut$。

**Step 3: 分组筛选**（仅 groupCount > 1 时执行）

将 normValue 按 groupCount 分组，根据 groupSelectMode 计算每组得分：

$$
groupedValue = Reshape(normValue,\ [batch,\ groupCount,\ -1])
$$

$$
groupScore = \begin{cases}
    ReduceMax(groupedValue,\ dim=-1), & groupSelectMode = 0 \\
    ReduceSum(TopK(groupedValue,\ k=2,\ dim=-1),\ dim=-1), & groupSelectMode = 1
\end{cases}
$$

选取得分最高的 kGroup 个组，将未选中组的对应位置置为 $-\infty$：

$$
groupIdx = TopK(groupScore,\ k=kGroup).indices
$$

$$
normValue = Mask(groupedValue,\ groupIdx,\ fillValue=-\infty)
$$

**Step 4: Top-K 专家选择**

对 normValue 取 Top-K 得到专家索引：

$$
y, expertIdxOut = TopK(normValue[groupIdx, :],\ k=k)
$$

**Step 5: Renorm 与缩放**

normType=1 时做归一化；normType=0 时，renorm 参数生效，renorm=1 时做renorm：

$$
if\ (normType = 1)\ or\ (normType = 0\ and\ renorm = 1):
$$

$$
\quad yOut = \frac{normOut}{ReduceSum(normOut,\ dim=-1) + eps}
$$

最终输出：

$$
yOut = yOut \times routedScalingFactor
$$

**Step 6: 可选输出**

若 outFlag 为 True，第三个输出为 normOut；否则为空。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_gating_top_k()` 函数形式提供：

```python
def moe_gating_top_k(x: Tensor, finished_optional: Tensor, k: int):
    """
    实现 moe_gating_top_k 操作。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        finished_optional (Tensor): 可选输入张量，Device侧的张量，数据格式支持ND。
        k (int): topk的k值。

    返回:
        tuple: 包含三个张量的元组，分别是：
            - 输出张量，数据类型与输入 x 一致
            - 专家索引张量，数据类型为int32
            - 源行索引张量，数据类型为int32
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
out, indices, source_row_out = kernel_gen_ops.moe_gating_top_k(x, finished_optional, k)

print("Output shape:", out.shape)
print("Indices shape:", indices.shape)
print("Source row out shape:", source_row_out.shape)
```

## 约束与限制

- 输入 x 的数据类型支持 FLOAT16、BFLOAT16、FLOAT32。
- 输入 x 的最后一维（即专家数）要求不大于2048。
- 要求 1 <= k <= x_shape[-1] / groupCount * kGroup。
- 要求 1 <= kGroup <= groupCount，并且 kGroup * x_shape[-1] / groupCount 的值要大于等于 k。
- 要求 groupCount > 0，x_shape[-1] 能够被 groupCount 整除且整除后的结果大于 groupSelectMode，并且整除的结果按照32个数对齐后乘 groupCount 的结果不大于2048。
- renorm 仅支持0，表示先进行norm操作，再计算topk。
- groupSelectMode 取值0和1，0表示使用最大值对group进行排序, 1表示使用topk2的sum值对group进行排序。
- normType 取值0和1，0表示使用Softmax函数，1表示使用Sigmoid函数。
- outFlag 取值true和false，true表示输出，false表示不输出。