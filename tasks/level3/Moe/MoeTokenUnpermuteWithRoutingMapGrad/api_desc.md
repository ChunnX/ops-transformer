# aclnnMoeTokenUnpermuteWithRoutingMapGrad

## 功能描述

### 算子功能
aclnnMoeTokenUnpermuteWithRoutingMap的反向传播算子，根据正向传播的输出梯度计算输入的梯度。

### 计算公式

**probs非None时：**

$$permutedTokensGrad[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]$$

$$permutedProbsGrad = permutedTokensGrad \times permutedTokensOptional$$

$$probsGradExpertOrder = \sum_{j=0}^{hidden\_size}(permutedProbsGrad_{i,j})$$

- **paddedMode为false时：**

$$probsGradOut = masked\_scatter(routingMapOptional.T, probsGradExpertOrder)$$

$$permutedProbs = probsOptional.T.masked\_select(routingMapOptional.T)$$

$$permutedTokensGradOut = permutedProbs.unsqueeze(-1) \times permutedTokensGrad$$

- **paddedMode为true时：**

$$probsGradOut[permuteTokenId[i], outIndex[i]/capacity] = probsGradExpertOrder[outIndex[i]]$$

$$permutedProbs[outIndex[i]] = probsOptional.view(1)[i]$$

$$permutedTokensGradOut = permutedProbs \times permutedTokensGrad$$

**probs为None时：**

$$permutedTokensGradOut[outIndex[i]] = unpermutedTokensGrad[permuteTokenId[i]]$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，以 `kernel_gen_ops.moe_token_unpermute_with_routing_map_grad()` 函数形式提供：

```python
def moe_token_unpermute_with_routing_map_grad(
    unpermuted_tokens_grad: Tensor,    # [tokens_num, hidden_size]
    out_index: Tensor,                  # [total_length] int32
    permute_token_id: Tensor,           # [total_length] int32
    routing_map: Optional[Tensor],      # [tokens_num, experts_num] bool/int8
    permuted_tokens: Optional[Tensor],  # [total_length, hidden_size]
    probs: Optional[Tensor],            # [tokens_num, experts_num]
    drop_and_pad: bool,
    restore_shape: List[int],
    permuted_tokens_grad: Tensor,       # [out_length, hidden_size] (output)
    probs_grad: Optional[Tensor]        # [tokens_num, experts_num] (output)
) -> Tuple[Tensor, Tensor]:
    """计算 MoeTokenUnpermuteWithRoutingMap 的反向传播"""
```

## 使用案例
```python
import torch
import kernel_gen_ops

# 基本参数
num_tokens, hidden_size, num_experts, top_k = 4, 64, 2, 2
total_length = num_tokens * top_k

# 构造输入
unpermuted_tokens_grad = torch.randn(num_tokens, hidden_size, dtype=torch.float32)
out_index = torch.arange(total_length, dtype=torch.int32)
permute_token_id = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
routing_map = torch.ones(num_tokens, num_experts, dtype=torch.bool)
permuted_tokens = torch.randn(total_length, hidden_size, dtype=torch.float32)
probs = torch.rand(num_tokens, num_experts, dtype=torch.float32)

# 预分配输出
permuted_tokens_grad = torch.zeros(total_length, hidden_size, dtype=torch.float32)
probs_grad = torch.zeros(num_tokens, num_experts, dtype=torch.float32)

# 调用算子
result = kernel_gen_ops.moe_token_unpermute_with_routing_map_grad(
    unpermuted_tokens_grad, out_index, permute_token_id,
    routing_map, permuted_tokens, probs,
    False, [num_tokens],
    permuted_tokens_grad, probs_grad
)
```

## 约束与限制
- **支持的数据类型**：unpermuted_tokens_grad 支持 BFLOAT16、FLOAT16、FLOAT32；out_index/permute_token_id 支持 INT32；routing_map 支持 BOOL、INT8；probs 支持 BFLOAT16、FLOAT16、FLOAT32
- **支持的数据格式**：ND
- **约束**：topK <= 512；topK <= experts_num；capacity <= tokens_num
- **平台支持**：Atlas A2 训练系列、Atlas A3 训练/推理系列
