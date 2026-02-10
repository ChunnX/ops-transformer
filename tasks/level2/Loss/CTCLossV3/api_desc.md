# CtcLossV3

## 功能描述

### 算子功能
该算子计算连续（未分段）时间序列与目标序列之间的损失。

### 计算公式

定义$y_{k}^{t}$表示在时刻$t$时真实字符为$k$的概率。（一般地，$y_{k}^{t}$是经过softmax之后的输出矩阵中的一个元素）。将字符集$L^{'}$可以构成的所有序列的集合称为$L^{'T}$，将$L^{'T}$中的任意一个序列称为路径，并标记为$π$。$π$的分布为公式(1)：

$$
p(π|x)=\prod_{t=1}^{T}y^{t}_{π_{t}} , \forall π \in L'^{T}. \tag{1}
$$

定义多对一(many to one)映射B: $L^{'T} \to L^{\leq T}$，通过映射B计算得到$l \in L^{\leq T}$的条件概率，等于对应于$l$的所有可能路径的概率之和，公式(2):

$$
p(l|x)=\sum_{π \in B^{-1}(l)}p(π|x).\tag{2}
$$

将找到使$p(l|x)$值最大的$l$的路径的任务称为解码，公式(3)：

$$
h(x)=^{arg \  max}_{l \in L^{ \leq T}} \ p(l|x).\tag{3}
$$

当zeroInfinity为True时

$$
h(x)=\begin{cases}0,&h(x) == Inf \text{ or } h(x) == -Inf \\h(x),&\text { else }\end{cases}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.ctc_loss_v3()` 函数形式提供：


```python
def ctc_loss_v3(log_probs: Tensor, targets: Tensor, input_lengths: List[int], target_lengths: List[int], blank: int, zero_infinity: bool):
    """
    实现 ctc_loss 操作。

    参数:
        log_probs (Tensor): 输入张量，Device侧的张量，公式中的y，数据格式支持ND。
        targets (Tensor): 输入张量，Device侧的张量，公式中的π，数据格式支持ND。
        input_lengths (List[int]): 公式中的T为input_lengths中的元素。
        target_lengths (List[int]): 公式中的l的长度为target_lengths中的元素。
        blank（Int）：空白标识，数值必须小于C且大于等于0。
        zero_infinity（bool）：表示是否将无限损耗和相关梯度归零，公式中的zeroInfinity。

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
log_probs = torch.nn.functional.log_softmax(torch.randn([1, 1, 9], dtype=torch.float32), -1)
targets = torch.randint(0, 5, [1],dtype=torch.int32)
input_lengths = [1]
target_lengths = [1]

result = kernel_gen_ops.ctc_loss_v3(log_probs, targets, input_lengths, target_lengths 6, True)
```
## 约束与限制

- log_probs的shape为(T, N, C)，其中T>=max(input_lengths)
- targets的shape为(N, S)或者(sum(target_lengths),)，其中S>=max(target_lengths)。
- blank表示空白标识，数值必须小于C且大于等于0。
- input_lengths和target_lengths的长度必须为N。
- neg_log_likelihood的shape为(N)。
- log_alpha的shape为(N, T, $(2*max(targetLengths)+8)/8*8$)。
- $log\_alpha.shape[2] - 1 / 2 <= 1000$
- $log\_probs.shape[2] <= 200000$
- $log\_probs.shape[1] <= 8180$