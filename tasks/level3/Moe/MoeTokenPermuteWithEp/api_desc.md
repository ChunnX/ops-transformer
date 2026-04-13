# MoeTokenPermuteWithEp

## 功能描述

### 算子功能
MoE计算中，根据专家索引indices将tokens和可选probs广播后排序，并按照range_vals中的范围进行切片。

### 计算公式

#### paddedMode为false时（当前仅支持此模式）

topK表示每个token选择的专家数量。如果indices为2维，则topK等于indices最后一维的大小。如果indices为1维，则topK为1。

$$
sortedIndicesFirst=argSort(\text{flatten}(indices))
$$

$$
sortedIndicesOut=argSort(sortedIndicesFirst)
$$

当range_vals[0] <= sortedIndicesOut[i] < range_vals[1]时:

$$
permuteTokensOut[sortedIndicesOut[i] - range_vals[0]]=tokens[i//topK]
$$

$$
permuteProbsOut[sortedIndicesOut[i] - range_vals[0]]=probs[i]
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_token_permute_with_ep()` 函数形式提供：


```python
def moe_token_permute_with_ep(x: Tensor, indices: Tensor, probs: Tensor, range_vals: List[int], num_token_out: int, pad_mode: bool):
    """
    实现 moe_token_permute_with_ep 操作。

    参数:
        x (Tensor): 输入tokens张量，Device侧的张量，数据类型支持BFLOAT16、FLOAT16、FLOAT32，shape为(num_tokens, hidden_size)。
        indices (Tensor): 输入tokens对应的专家索引，Device侧的张量，数据类型支持INT32、INT64，shape为(num_tokens, topK_num)或(num_tokens)。
        probs (Tensor): 输入tokens对应的专家概率，Device侧的张量，数据类型同x，shape与indices的shape相同。
        range_vals (List[int]): ep切分的有效范围，长度为2的列表。
        num_token_out (int): 有效输出token数。
        pad_mode (bool): 是否为填充模式，当前仅支持False。

    返回:
        Tuple[Tensor, Tensor, Tensor]: 输出元组，包含：
            - permuteTokensOut: 根据indices进行扩展并排序过的tokens，shape为(range_vals[1] - range_vals[0], hidden_size)。
            - sortedIndicesOut: permuteTokensOut和tokens的映射关系，shape为(num_tokens * topK_num)。
            - permuteProbsOut: 根据indices进行扩展并排序过的probs，shape为(range_vals[1] - range_vals[0])。
    """

```

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

# 创建输入张量并移到 NPU
x = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]], dtype=torch.float32).to('npu')
indices = torch.tensor([[1, 2], [3, 1], [2, 3]], dtype=torch.int32).to('npu')
probs = torch.tensor([[0.5, 0.3], [0.4, 0.2], [0.5, 0.4]], dtype=torch.float32).to('npu')
range_vals = [1, 5]
num_token_out = 4
pad_mode = False

# 调用算子（输入张量需要在 NPU 上）
expanded_x, sorted_indices, expanded_probs = kernel_gen_ops.moe_token_permute_with_ep(x, indices, probs, range_vals, num_token_out, pad_mode)

# 将结果移回 CPU 以便打印
expanded_x_cpu = expanded_x.to('cpu')
sorted_indices_cpu = sorted_indices.to('cpu')
expanded_probs_cpu = expanded_probs.to('cpu')

print("expanded_x:", expanded_x_cpu)
print("sorted_indices:", sorted_indices_cpu)
print("expanded_probs:", expanded_probs_cpu)
```

## 约束与限制

- indices的元素个数必须小于16777215，且元素值必须大于等于0且小于16777215。
- indices的shape必须为1D或2D。当pad_mode为false时，indices的shape第一维必须与x的第一维相等。
- 当pad_mode为true时暂不支持。
- x的数据类型必须为BFLOAT16、FLOAT16或FLOAT32。
- indices的数据类型必须为INT32或INT64。
- probs的数据类型必须与x相同。
- range_vals的长度必须为2。
- 输入shape大小不要超过device可分配的内存上限，否则会导致异常终止。
- sortedIndicesOut的数据类型为INT32，与输入indices的数据类型无关。
- 输出permuteTokensOut的数据类型与输入x相同。
- 输出permuteProbsOut的数据类型与输入probs相同。