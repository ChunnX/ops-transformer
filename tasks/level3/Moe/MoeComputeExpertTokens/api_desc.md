# MoeComputeExpertTokens

## 功能描述

### 算子功能
MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。

### 计算公式

$$
out_{i}=BinarySearch(sortedExperts,numExperts)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_compute_expert_tokens()` 函数形式提供：


```python
def moe_compute_expert_tokens(sorted_experts: Tensor, num_experts: int):
    """
    实现 moe_compute_expert_tokens 操作。

    参数:
        sorted_experts (Tensor): 排序后的专家索引，Device侧的张量，数据类型为INT32，数据格式支持ND。
        num_experts (int): 总专家数，取值范围大于0且不超过2048。

    返回:
        Tensor: 输出张量，数据类型为INT32，数据格式保持不变，表示每个专家处理的最后一行的位置。
    """

```

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

# 创建输入张量并移到 NPU
sorted_experts = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32).to('npu')
num_experts = 3

# 调用算子（输入张量需要在 NPU 上）
result = kernel_gen_ops.moe_compute_expert_tokens(sorted_experts, num_experts)

# 将结果移回 CPU 以便打印
result_cpu = result.to('cpu')
print(result_cpu)
# 输出: tensor([2, 5, 6], dtype=torch.int32)

# 注：输出表示每个专家处理的最后一行的位置（从0开始计数）
# 例如：专家0的最后一行位置是1（索引从0开始），所以输出2
# 专家1的最后一行位置是4，所以输出5
# 专家2的最后一行位置是5，所以输出6
```
## 约束与限制

- sortedExperts的shape大小需要小于2^24。
- numExperts的输入常值需要大于0，但不能超过2048。
- 输入shape大小不要超过device可分配的内存上限，否则会导致异常终止。
- 输入sorted_experts的数据类型必须为INT32。
- 输出out的数据类型为INT32，与输入sorted_experts一致。