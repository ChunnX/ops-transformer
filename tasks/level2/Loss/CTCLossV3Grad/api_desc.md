# CTCLossV3Grad

## 功能描述

### 算子功能
该算子计算连续（未分段）时间序列与目标序列之间的损失的反向。

### 计算公式

$$
O^{ML}(S,N_w) = - \sum_{x,z\in S} {\ln{p(z|x)}}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.ctc_loss_v3_grad()` 函数形式提供：


```python
def ctc_loss_v3_grad(grad_out: Tensor, log_probs: Tensor, targets: Tensor, input_lengths: List[int], target_lengths: List[int], neg_log_likelihood: Tensor, log_alpha: Tensor, blank: int, zero_infinity: bool):
    """
    实现 ctc_loss 操作。

    参数:
        grad_out (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        log_probs (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        targets (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        input_lengths (List[int]): 输入序列的长度。
        target_lengths (List[int]): 目标序列的长度。
        blank（Int）：空白标识，数值必须小于C且大于等于0。
        zero_infinity（bool）：表示是否将无限损耗和相关梯度归零。

    返回:
        neg_log_likelihood (Tensor): 输出张量，数据类型与输入log_probs一致，数据格式保持不变。
        log_alpha (Tensor): 输出张量，数据类型与输入log_probs一致，数据格式保持不变。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
grad = torch.randn([1], dtype=torch.float32)
log_probs = torch.nn.functional.log_softmax(torch.randn([9, 1, 15], dtype=torch.float32), -1)
targets = torch.randint(0, 10, [1, 1],dtype=torch.int32)
input_lengths = [1]
target_lengths = [1]
neg_log_likelihood = torch.randn([1], dtype=torch.float32)
log_aplha = torch.randn([1, 9, 8], dtype=torch.float32)

result = kernel_gen_ops.ctc_loss_v3_grad(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_aplha, 6, True)
```
## 约束与限制

- grad_out的元素数量必须为N。
- log_probs的shape为(T, N, C)，其中T>=max(input_lengths)。
- targets的shape为(N, S)或者(sum(target_lengths),)，其中S>=max(target_lengths)，值为[0, C)。
- input_lengths的元素数量必须为N，值为[0, T]。
- target_lengths的元素数量必须为N，值为[1, S]。
- neg_log_likelihood的shape必须为1维(N)。
- log_alpha的shape必须为3维(N, T, X)，其中X=((2 * S + 1) + 7) / 8 * 8。
- blank的值域为[0, C - 1]。