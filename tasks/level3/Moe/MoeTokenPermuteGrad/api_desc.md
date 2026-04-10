# aclnnMoeTokenPermuteGrad

## 功能描述

### 算子功能
该算子是 aclnnMoeTokenPermute 的反向传播计算，根据 permutedOutputGrad 和 sortedIndices 计算输入 token 的梯度。

### 计算公式

$$
inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)
$$

$$
inputGrad = inputGrad.reshape(-1, numTopk, hiddenSize)
$$

$$
inputGrad = inputGrad.sum(dim = 1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_token_permute_grad()` 函数形式提供：

```python
def moe_token_permute_grad(permuted_output_grad: Tensor, sorted_indices: Tensor, num_topk: int, padded_mode: bool = False):
    """
    实现 MoeTokenPermuteGrad 操作。

    参数:
        permuted_output_grad (Tensor): 正向输出 permutedTokens 的梯度，shape为（tokens_num * topK_num，hidden_size）。
        sorted_indices (Tensor): 排序的索引值，shape为（tokens_num * topK_num）。
        num_topk (int): 被选中的专家个数。
        padded_mode (bool): 是否开启 paddedMode 模式，目前仅支持 false。

    返回:
        Tensor: 输入 token 的梯度，shape为（tokens_num，hidden_size），数据类型与输入一致。
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
tokens_num = 8
topK_num = 2
hidden_size = 64

# 生成 permutedOutputGrad
permuted_output_grad = torch.randn(tokens_num * topK_num, hidden_size, dtype=torch.float32)

# 生成 sortedIndices
sorted_indices = torch.randint(0, tokens_num * topK_num, (tokens_num * topK_num,), dtype=torch.int32)

# 使用 moe_token_permute_grad 执行操作
result = kernel_gen_ops.moe_token_permute_grad(permuted_output_grad, sorted_indices, topK_num)
print(f"Input shape: {permuted_output_grad.shape}")
print(f"Output shape: {result.shape}")
```

## 约束与限制

- 输入 sorted_indices 的取值范围是 (0, tokens_num * topK_num - 1)，且没有重复索引。
- paddedMode 目前仅支持 false。
- permutedOutputGrad 支持的数据类型为 BFLOAT16、FLOAT16、FLOAT32。
- sortedIndices 支持的数据类型为 INT32。
- 输出 out 的数据类型与 permutedOutputGrad 一致。
- numTopk <= 512。