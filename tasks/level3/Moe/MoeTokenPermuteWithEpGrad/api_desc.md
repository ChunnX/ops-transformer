# MoeTokenPermuteWithEpGrad算子接口文档

## 功能描述

MoeTokenPermuteWithEpGrad是MoeTokenPermuteWithEp的反向传播算子，用于计算permutedTokensOutputGrad和permutedProbsOutputGrad对输入token和probs的梯度。

## 计算公式

1. **计算tokenGradOut**：
   - 当rangeOptional[0] <= sortedIndices[i] < rangeOptional[1]时：
     $$
tokenGradOut[i] = permutedTokensOutputGrad[sortedIndices[i]-rangeOptional[0]]
$$
   - 否则：
     $$
tokenGradOut[i] = 0
$$

2. **reshape并沿topK维度求和**：
   $$
tokenGradOut = tokenGradOut.reshape(-1, numTopk, hiddenSize)
$$
   $$
tokenGradOut = tokenGradOut.sum(dim = 1)
$$

## Python接口定义

```python
def moe_token_permute_with_ep_grad(
    unpermuted_tokens_grad: torch.Tensor,
    sorted_indices: torch.Tensor,
    permuted_probs_output_grad: Optional[torch.Tensor],
    topk_num: int,
    range_vals: List[int],
    padded_mode: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    MoeTokenPermuteWithEp的反向传播计算
    
    参数：
    - unpermuted_tokens_grad: 表示正向输出permutedTokens的梯度，形状为[num_tokens, hidden_size]
    - sorted_indices: 表示正向输出的sortedIndices，形状为[num_tokens]
    - permuted_probs_output_grad: 表示正向输出permutedProbs的梯度（可选），形状为[num_tokens]
    - topk_num: 表示topK值
    - range_vals: 表示范围参数，形状为[2]
    - padded_mode: 表示是否使用padding模式（当前版本仅支持False）
    
    返回值：
    - token_grad_out: 表示输入token的梯度，形状为[num_tokens/topk_num, hidden_size]
    - probs_grad_out: 表示输入probs的梯度（可选），形状为[num_tokens/topk_num, topk_num]
    """
```

## 支持的数据类型

| 输入数据类型 | 支持情况 |
| :----------- | :------- |
| BFLOAT16     | √        |
| FLOAT16      | √        |
| FLOAT32      | √        |

## 约束条件

| 约束类型           | 约束描述                             |
| :----------------- | :----------------------------------- |
| numTopk范围        | numTopk > 0 且 numTopk <= 512        |
| padded_mode支持    | 当前版本仅支持False                  |
| sorted_indices类型 | 必须为int32类型                     |
| range_vals长度     | 必须为2，表示[start, end)范围        |
| 形状匹配           | unpermuted_tokens_grad.shape[0] == sorted_indices.shape[0] |
|                    | 如果提供permuted_probs_output_grad，则permuted_probs_output_grad.shape[0] == sorted_indices.shape[0] |

## 使用案例

```python
import torch
from kernel_gen_ops import moe_token_permute_with_ep_grad

# 准备输入数据
unpermuted_tokens_grad = torch.randn(9638, 5376, dtype=torch.bfloat16, device='npu')
sorted_indices = torch.randint(0, 9638, (9638,), dtype=torch.int32, device='npu')
permuted_probs_output_grad = torch.randn(9638, dtype=torch.bfloat16, device='npu')
topk_num = 1
range_vals = [0, 9638]
padded_mode = False

# 调用算子
token_grad_out, probs_grad_out = moe_token_permute_with_ep_grad(
    unpermuted_tokens_grad, 
    sorted_indices, 
    permuted_probs_output_grad, 
    topk_num, 
    range_vals, 
    padded_mode
)

print(f"token_grad_out shape: {token_grad_out.shape}")
print(f"probs_grad_out shape: {probs_grad_out.shape if probs_grad_out is not None else None}")
```

## 错误处理

| 错误类型           | 错误描述                             |
| :----------------- | :----------------------------------- |
| 输入形状不匹配     | unpermuted_tokens_grad或permuted_probs_output_grad的形状与sorted_indices不匹配 |
| numTopk超出范围    | numTopk <= 0 或 numTopk > 512        |
| padded_mode不支持  | padded_mode为True                     |
| sorted_indices类型错误 | sorted_indices不是int32类型         |
| range_vals长度错误 | range_vals的长度不等于2              |
