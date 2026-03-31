# aclnnMoeTokenUnpermuteGrad

## 功能描述

### 算子功能
该算子实现aclnnMoeTokenUnpermute的反向传播，用于MoE（Mixture of Experts）模型中token重排操作的梯度计算。

### 计算公式

- probs非None：

  $$
  unpermutedTokens[i] = permutedTokens[sortedIndices[i]]
  $$

  $$
  unpermutedTokens = unpermutedTokens.reshape(-1, topK\_num, hiddenSize)
  $$

  $$
  unpermutedTokens = unpermutedTokensGrad.unsqueeze(1) * unpermutedTokens
  $$

  $$
  probsGrad = \sum_{k=0}^{K}(unpermutedTokens_{i,j,k})
  $$

  $$
  permutedTokensGrad[sortedIndices[i]] = ((unpermutedTokensGrad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, hiddenSize))[i]
  $$

- probs为None：

  $$
  permutedTokensGrad[sortedIndices[i]] = unpermutedTokensGrad[i]
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_token_unpermute_grad()` 函数形式提供：

```python
def moe_token_unpermute_grad(permuted_tokens: Tensor, 
                           unpermuted_tokens_grad: Tensor, 
                           sorted_indices: Tensor, 
                           probs_optional: Optional[Tensor] = None,
                           padded_mode: bool = False):
    """
    实现 MoE Token Unpermute 的反向传播操作。

    参数:
        permuted_tokens (Tensor): 输入张量，表示重排后的token，形状为(tokens_num * topK_num, hidden_size)
        unpermuted_tokens_grad (Tensor): 输入张量，表示未重排token的梯度，形状为(tokens_num, hidden_size)
        sorted_indices (Tensor): 输入张量，表示输入输出梯度的映射关系，形状为(tokens_num * topK_num)
        probs_optional (Optional[Tensor]): 可选输入张量，表示token选择指定专家的权重，形状为(tokens_num, topK_num)
        padded_mode (bool): 是否开启padded模式，目前仅支持False

    返回:
        Tuple[Tensor, Optional[Tensor]]: 包含两个张量的元组
            - permuted_tokens_grad: 重排token的梯度，形状与permuted_tokens相同
            - probs_grad: probs的梯度，当probs_optional不为None时返回，否则为None
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 场景1：不带probs的情况
# 创建输入张量
tokens_num = 32
hidden_size = 512
topK_num = 1

permuted_tokens = torch.randn(tokens_num * topK_num, hidden_size, dtype=torch.float32)
unpermuted_tokens_grad = torch.randn(tokens_num, hidden_size, dtype=torch.float32)
sorted_indices = torch.randint(0, tokens_num * topK_num, (tokens_num * topK_num,), dtype=torch.int32)

# 执行反向传播计算
permuted_tokens_grad, probs_grad = kernel_gen_ops.moe_token_unpermute_grad(
    permuted_tokens, unpermuted_tokens_grad, sorted_indices
)

print(f"permuted_tokens_grad shape: {permuted_tokens_grad.shape}")
print(f"probs_grad: {probs_grad}")  # 应为None

# 场景2：带probs的情况
topK_num = 4
permuted_tokens = torch.randn(tokens_num * topK_num, hidden_size, dtype=torch.float32)
unpermuted_tokens_grad = torch.randn(tokens_num, hidden_size, dtype=torch.float32)
sorted_indices = torch.randint(0, tokens_num * topK_num, (tokens_num * topK_num,), dtype=torch.int32)
probs = torch.randn(tokens_num, topK_num, dtype=torch.float32)

# 执行反向传播计算
permuted_tokens_grad, probs_grad = kernel_gen_ops.moe_token_unpermute_grad(
    permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs
)

print(f"permuted_tokens_grad shape: {permuted_tokens_grad.shape}")
print(f"probs_grad shape: {probs_grad.shape}")
```

## 约束与限制

- 输入张量数据类型支持：BFLOAT16、FLOAT16、FLOAT32
- sorted_indices的数据类型必须为INT32
- sorted_indices的取值范围是[0, tokens_num * topK_num - 1]，且不能有重复索引
- 当probs_optional不为空时，topK_num等于probs_optional的第2维；当probs_optional为空时，topK_num=1
- padded_mode目前仅支持False
- restore_shape_optional当前仅支持None
- 输出张量均为连续张量
- Atlas A2/A3系列产品：topK_num <= 512
- 确定性计算：aclnnMoeTokenUnpermuteGrad默认确定性实现
