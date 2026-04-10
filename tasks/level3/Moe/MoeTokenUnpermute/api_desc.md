# aclnnMoeTokenUnpermute

## 功能描述

### 算子功能
该算子根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘；最后进行累加求和，并输出计算结果。

### 计算公式

- probs非None计算公式如下：
  
  $$
  T[k] = T[S[k]]
  $$
  
  $$
  T[k] = T[k] * P[i][j]
  $$

  $$
  O[i] = \sum_{k=i*topK}^{(i+1)*topK - 1 } T[k]
  $$
  
  其中$i \in {0,1,...,tokens-1}$；$j \in {0,1,...,topK-1}$；$k \in {0,1,...,tokens*topK-1}$；T表示permutedTokens；S表示sortedIndices；P表示probs；O表示out；topK表示topK\_num；tokens表示tokens_num。

- probs为None时，此时topK\_num=1，计算公式如下：

  $$
  T[i] = T[S[i]]
  $$

  $$
  O[i] = T[i]
  $$

  其中 $i \in {0,1,...,tokens-1}$；T表示permutedTokens；S表示sortedIndices；O表示out；tokens表示tokens_num。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_token_unpermute()` 函数形式提供：

```python
def moe_token_unpermute(permuted_tokens: Tensor, sorted_indices: Tensor, probs: Optional[Tensor] = None, padded_mode: bool = False):
    """
    实现 MoeTokenUnpermute 操作。

    参数:
        permuted_tokens (Tensor): 输入Tokens，shape为（tokens_num * topK_num，hidden_size）。
        sorted_indices (Tensor): 表示需要计算的数据在permutedTokens中的位置，shape为（tokens_num * topK_num）。
        probs (Optional[Tensor]): 可选的概率值，shape为（tokens_num，topK_num）。
        padded_mode (bool): 是否开启paddedMode，目前仅支持false。

    返回:
        Tensor: 输出张量，shape为（tokens_num，hidden_size），数据类型与输入一致。
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
padded_mode = False

# 生成 permutedTokens
permuted_tokens = torch.randn(tokens_num * topK_num, hidden_size, dtype=torch.float32)

# 生成 sortedIndices
sorted_indices = torch.randint(0, tokens_num * topK_num, (tokens_num * topK_num,), dtype=torch.int32)

# 生成 probs
probs = torch.randn(tokens_num, topK_num, dtype=torch.float32)

# 使用 moe_token_unpermute 执行操作
result = kernel_gen_ops.moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode)
print(f"Input shape: {permuted_tokens.shape}")
print(f"Output shape: {result.shape}")
```

## 约束与限制

- 输入 sorted_indices 的取值范围是 [0, tokens_num * topK_num - 1]，且没有重复索引。
- 当 probs 传入时，topK_num 等于 probs 的第二维；当 probs 不传时，topK_num=1。
- paddedMode 目前仅支持 false。
- permutedTokens 和 probsOptional 支持的数据类型为 BFLOAT16、FLOAT16、FLOAT32。
- sortedIndices 支持的数据类型为 INT32。
- 输出 out 的数据类型与 permutedTokens 一致。
- topK_num <= 512。
- hidden_size是128的倍数且小于10240