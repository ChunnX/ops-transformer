# aclnnMoeTokenUnpermuteWithEp

## 功能描述

### 算子功能
根据 sortedIndices 中存储的下标位置，获取 permutedTokens 中的输入数据与 probs 相乘，并进行合并累加，支持 EP（Expert Parallelism）分区范围的指定。

### 计算公式

$$sortedIndices = sortedIndices[rangeOptional[0] \leq i < rangeOptional[1]]$$

（1）probs 非 None 时，其中 $i \in \{0, 1, ..., num\_tokens - 1\}$，$j \in \{0, 1, ..., topK\_num - 1\}$：

$$permutedTokens_{k} = permutedTokens_{sortedIndices[k] - rangeStart} \times probs_{i,j}$$

$$out_{i} = \sum_{k=i \times topK\_num}^{(i+1) \times topK\_num - 1} permutedTokens_{k}$$

（2）probs 为 None 时：

$$out_{i} = \sum_{k=i \times topK\_num}^{(i+1) \times topK\_num - 1} permutedTokens_{sortedIndices[k] - rangeStart}$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.custom_pybind_api()` 函数形式提供：

```python
def custom_pybind_api(
    permuted_tokens: torch.Tensor,     # [num_local_tokens, hidden_size]
    sorted_indices: torch.Tensor,      # [num_tokens * top_k]
    probs: torch.Tensor,               # [num_tokens, top_k] 或空张量
    num_topk: int,                     # top-k 专家数
    range: List[int],                  # [range_start, range_end]
    padded_mode: bool,                 # 是否启用 padded 模式（仅支持 False）
    restore_shape: List[int],          # 恢复形状（padded_mode=True 时生效，需为空）
    out: torch.Tensor                  # [num_tokens, hidden_size]
) -> torch.Tensor
```

## 使用案例
```python
import torch
import kernel_gen_ops

# 示例：3个token，top_k=2，hidden_size=2，range=[1,5]
permuted_tokens = torch.tensor([[2., 2.], [1., 1.], [3., 3.], [2., 2.]], dtype=torch.float32).npu()
sorted_indices = torch.tensor([2, 0, 4, 1, 5, 3], dtype=torch.int32).npu()
probs = torch.tensor([[1., 1.], [1., 1.], [1., 1.]], dtype=torch.float32).npu()

num_tokens = 3
top_k = 2
out = torch.zeros(num_tokens, 2, dtype=torch.float32).npu()

result = kernel_gen_ops.custom_pybind_api(
    permuted_tokens, sorted_indices, probs,
    top_k, [1, 5], False, [], out
)
```

## 约束与限制
- 支持的数据类型：permuted_tokens 和 probs 支持 BFLOAT16、FLOAT16、FLOAT32；sorted_indices 仅支持 INT32
- 支持的数据格式：ND
- topK_num <= 512
- 不支持 padded_mode 为 True
- 当 range 为空时，忽略 numTopk，执行逻辑回退到 MoeTokenUnpermute
- sorted_indices 中的元素值需 >= 0 且 < 2134372523
