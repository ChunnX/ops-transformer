# aclnnMoeTokenUnpermuteWithRoutingMap

## 功能描述

### 算子功能
根据 sortedIndices 中存储的下标位置，获取 permutedTokens 中的输入数据，结合 routingMap 和 probs 进行加权合并累加，实现 token 的反向排列（unpermute）。支持 drop-and-pad 模式。

### 计算公式

**非 padded 模式（drop_and_pad=false）：**

（1）probs 非 None 且 routingMap 非 None 时：

$$permuteProbs = MaskedSelect(probs, routingMap)$$

$$out_{i} = \sum_{k=i \times topK}^{(i+1) \times topK - 1} permutedTokens_{sortedIndices[k]} \times permuteProbs_{k}$$

（2）probs 为 None 时：

$$out_{i} = \sum_{k=i \times topK}^{(i+1) \times topK - 1} permutedTokens_{sortedIndices[k]}$$

**padded 模式（drop_and_pad=true）：**

$$out_{posIdx} = probs_{sortedIndices[posIdx] \times numExperts + posIdx / capacity}$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.moe_token_unpermute_with_routing_map()` 函数形式提供：

```python
def moe_token_unpermute_with_routing_map(
    permuted_tokens: torch.Tensor,     # [num_out_tokens, hidden_size]
    sorted_indices: torch.Tensor,      # [total_length]
    routing_map: Optional[torch.Tensor],  # [num_tokens, num_experts] bool/int8
    probs: Optional[torch.Tensor],     # [num_tokens, num_experts]
    drop_and_pad: bool,                # 是否启用 padded 模式
    restore_shape: List[int],          # [num_tokens]
    unpermuted_tokens: torch.Tensor,   # [num_tokens, hidden_size] (输出)
    out_index: torch.Tensor,           # [total_length] (输出)
    permute_token_id: torch.Tensor,    # [total_length] (输出)
    permute_probs: torch.Tensor        # [max_size] (输出)
) -> torch.Tensor
```

## 使用案例
```python
import torch
import kernel_gen_ops

num_tokens = 4
num_experts = 2
topK = 2
hidden_size = 64

permuted_tokens = torch.randn(num_tokens * topK, hidden_size, dtype=torch.float32).npu()
sorted_indices = torch.randint(0, num_tokens * topK, (num_tokens * topK,), dtype=torch.int32).npu()
routing_map = torch.ones(num_tokens, num_experts, dtype=torch.bool).npu()
probs = torch.rand(num_tokens, num_experts, dtype=torch.float32).npu()

unpermuted_tokens = torch.zeros(num_tokens, hidden_size, dtype=torch.float32).npu()
out_index = torch.zeros(num_tokens * topK, dtype=torch.int32).npu()
permute_token_id = torch.zeros(num_tokens * topK, dtype=torch.int32).npu()
permute_probs = torch.zeros(num_tokens * num_experts, dtype=torch.float32).npu()

result = kernel_gen_ops.moe_token_unpermute_with_routing_map(
    permuted_tokens, sorted_indices, routing_map, probs,
    False, [num_tokens],
    unpermuted_tokens, out_index, permute_token_id, permute_probs
)
```

## 约束与限制
- 支持的数据类型：permuted_tokens 和 probs 支持 BFLOAT16、FLOAT16、FLOAT32；sorted_indices 仅支持 INT32；routing_map 支持 BOOL 和 INT8
- 支持的数据格式：ND
- topK <= 512
- 当 probs 非 None 时，routingMap 也应提供
- sorted_indices 中的元素值需 >= 0 且 < num_out_tokens
