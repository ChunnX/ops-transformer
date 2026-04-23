# aclnnMoeTokenPermute

## 功能描述

### 算子功能
该算子执行 MoE（混合专家模型）的 permute 计算，根据索引 indices 将 tokens 广播并排序。

### 计算公式

- paddedMode 为 `false` 时，公式如下，其中 topK 指一个 token 选择的专家个数，Indices 维度为 2 时 topK 等于 Indices 最后一维大小，Indices 维度为 1 时 topK 等于 1：

  $$
sortedIndicesFirst=argSort(\text{flatten}(Indices))
  $$

  $$
sortedIndicesOut=argSort(sortedIndicesFirst)
  $$

  $$
permuteTokensOut[sortedIndicesOut[i]]=tokens[i//topK]
  $$

- paddedMode 为 `true` 时（暂不支持）：

  $$
permuteTokensOut[i]=tokens[indices[i]]
  $$

  $$
sortedIndicesOut=indices
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_token_permute()` 函数形式提供：

```python
def moe_token_permute(tokens: Tensor, indices: Tensor, num_out_tokens: int, padded_mode: bool = False):
    """
    实现 MoeTokenPermute 操作。

    参数:
        tokens (Tensor): 输入 token 特征，shape 为 (num_tokens, ...)，维度大于等于 2。
        indices (Tensor): 输入 indices 索引，shape 为 (num_tokens, topK) 或 (num_tokens)。
        num_out_tokens (int): 有效输出 token 数，0 表示不删除任何 token。
        padded_mode (bool): 是否为填充模式，目前仅支持 False。

    返回:
        Tuple[Tensor, Tensor]: 
            - permute_tokens: 根据 indices 进行扩展并排序过的 tokens，shape 为 (min(num_tokens * topK, num_out_tokens), ...)。
            - sorted_indices: permuteTokensOut 和 tokens 的映射关系，shape 为 (num_tokens * topK)。
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
num_tokens = 3
topK = 2
hidden_size = 4

# 生成 tokens
tokens = torch.tensor([[0.1, 0.1, 0.1, 0.1],
                       [0.2, 0.2, 0.2, 0.2],
                       [0.3, 0.3, 0.3, 0.3]], dtype=torch.float32)

# 生成 indices
indices = torch.tensor([[1, 2], [0, 1], [0, 2]], dtype=torch.int32)

# 使用 moe_token_permute 执行操作
permute_tokens, sorted_indices = kernel_gen_ops.moe_token_permute(tokens, indices, 0, False)
print(f"Input tokens shape: {tokens.shape}")
print(f"Output permute_tokens shape: {permute_tokens.shape}")
print(f"Output sorted_indices shape: {sorted_indices.shape}")
```

## 约束与限制

- 输入 tokens 要求为一个维度大于等于 2 的 Tensor，第一维的大小为 num_tokens。
- 输入 indices 要求 shape 为 2D 或 1D：
  - paddedMode 为 false 时表示每一个输入 token 对应的 topK 个处理专家索引，shape 为 (num_tokens, topK) 或 (num_tokens)。
  - paddedMode 为 true 时表示每个专家选中的 token 索引（暂不支持）。
- indices 元素个数小于 16777215，值大于等于 0 且小于 16777215。
- numOutTokens 值范围为任意整数；0 表示不会删除任何 token，大于 0 时会按照 numOutTokens 对按照专家排序好的 token 进行切片，保留前 numOutTokens 个 token，小于 0 时按负的切片索引进行处理。
- paddedMode 目前仅支持 false。
- tokens 支持的数据类型为 FLOAT16、BFLOAT16、FLOAT32。
- indices 支持的数据类型为 INT32、INT64。
- permuteTokensOut 的数据类型同 tokens。
- sortedIndicesOut 的数据类型为 INT32。