# aclnnMoeTokenUnpermuteWithEpGrad

## 功能描述

### 算子功能
MoeTokenUnpermuteWithEp 的反向传播算子，用于计算混合专家（Mixture of Experts）模型中 token 重排操作的反向梯度。

### 计算公式

**probs 非 None 时：**

1. 当 `range[0] <= sortedIndices[i] < range[1]` 时：
$$permutedTokensGrad[sortedIndices[i]-range[0]] = unpermutedTokensGrad[i // topK] \times probs[i // topK, i \% topK]$$

2. probs 梯度：
$$probsGrad[tokenIdx, kIdx] = \sum_{h}(permutedTokens[sortedIndices[i]-range[0], h] \times unpermutedTokensGrad[tokenIdx, h])$$

**probs 为 None 时：**

当 `range[0] <= sortedIndices[i] < range[1]` 时：
$$permutedTokensGrad[sortedIndices[i]-range[0]] = unpermutedTokensGrad[i // topK]$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.moe_token_unpermute_with_ep_grad()` 函数形式提供：

```python
def moe_token_unpermute_with_ep_grad(
    unpermuted_tokens_grad: Tensor,   # (tokens_num, hidden_size)
    sorted_indices: Tensor,            # (tokens_num * topk_num,)
    permuted_tokens: Tensor,           # (tokens_num * topk_num, hidden_size)
    probs: Tensor,                     # (tokens_num, topk_num) 或 None
    padded_mode: bool,                 # 目前仅支持 False
    restore_shape: List[int],          # 目前仅支持 [1, 1]
    range: List[int],                  # EP 切分范围 [start, end]
    topk_num: int,                     # 每个token被选中的专家个数
) -> Tuple[Tensor, Tensor]:            # (permuted_tokens_grad, probs_grad)
```

## 使用案例
```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops

tokens_num = 4
hidden_size = 32
topk_num = 2
dtype = torch.float32
device = "npu:0"

unpermuted_tokens_grad = torch.randn(tokens_num, hidden_size, dtype=dtype, device=device)
sorted_indices = torch.randperm(tokens_num * topk_num).int().to(device)
permuted_tokens = torch.randn(tokens_num * topk_num, hidden_size, dtype=dtype, device=device)
probs = torch.randn(tokens_num, topk_num, dtype=dtype, device=device)

permuted_tokens_grad, probs_grad = kernel_gen_ops.moe_token_unpermute_with_ep_grad(
    unpermuted_tokens_grad, sorted_indices, permuted_tokens, probs,
    False, [1, 1], [-1, -1], topk_num
)
```

## 约束与限制
- 支持的数据类型：BFLOAT16、FLOAT16、FLOAT32
- sorted_indices 仅支持 INT32
- 支持的数据格式：ND
- topk_num <= 512
- padded_mode 目前仅支持 False
- range[0] < range[1]，为 [-1, -1] 时不生效
