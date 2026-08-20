# MoE Gating TopK 算子说明与对比

本文说明以下三个算子的作用、计算公式、输入输出及主要区别：

- `moe_gating_top_k`
- `moe_gating_top_k_softmax`
- `moe_gating_top_k_softmax_v2`

三个算子都用于 MoE（Mixture of Experts）的门控路由。对每个 token，算子根据专家 logits 计算路由权重，并选择 `k` 个专家。

以下以二维输入为例：

```
x.shape = [B, E]
```

其中 `B` 是 token 数，`E` 是专家数。

## 1. `moe_gating_top_k`

这是功能最完整的门控 TopK 算子，支持：

- Softmax、Sigmoid、SqrtSoftplus 归一化
- bias
- 专家分组选择
- TopK 专家选择
- selected weight renorm
- routed scaling
- Hash 模式预定义专家索引

### 计算公式

首先根据 `normType` 计算归一化结果：

$$
n_{b,e}=
\begin{cases}
\operatorname{Softmax}(x_{b,:})_e,& normType=0\\
\operatorname{Sigmoid}(x_{b,e}),& normType=1\\
\sqrt{\operatorname{Softplus}(x_{b,e})},& normType=2
\end{cases}
$$

其中 `normType=2` 主要由 Ascend 950 路径支持。

如果存在 bias：

$$
s_{b,e}=n_{b,e}+bias_e
$$

将专家划分为 `groupCount` 组，每组有 `E / groupCount` 个专家。

- `groupSelectMode=0`：

$$
q_{b,g}=\max_{e\in g}s_{b,e}
$$

- `groupSelectMode=1`：

$$
q_{b,g}=\text{组内最大两个值之和}
$$

选择得分最高的 `kGroup` 个组，然后在这些组内选择 TopK 专家：

$$
I_b=\operatorname{TopK}(s_{b,:}, k)
$$

专家选择依据是 `n + bias`，但最终输出权重使用不带 bias 的 `n`：

$$
a_{b,j}=n_{b,I_{b,j}}
$$

当需要 renorm 时，实际实现对选中的 K 个权重求和：

$$
y_{b,j}=\frac{a_{b,j}}
{\sum_{t=1}^{k}a_{b,t}+\epsilon}
\times routedScalingFactor
$$

否则：

$$
y_{b,j}=a_{b,j}\times routedScalingFactor
$$

### 输入输出

对于 `x.shape = [B, E]`：

- `x`：`[B, E]`，专家 logits
- `biasOptional`：`[E]`，可选偏置
- `yOut`：`[B, k]`，选中专家的权重，dtype 与 `x` 相同
- `expertIdxOut`：`[B, k]`，专家编号，`int32`
- `outOut`：`[B, E]`，当 `outFlag=true` 时输出完整的归一化结果，通常为 `float32`

如果同时提供 `inputIds` 和 `tid2eid`，可以启用 Hash 模式：

$$
expertIdxOut=tid2eid[inputIds,:]
$$

此时跳过专家排序，直接使用预先计算的专家索引。

### 示例

```
x    = [[ 2, 1, 0, -1 ]]
bias = [ 0, 0, 0.5, 0.5 ]

k = 2
kGroup = 1
groupCount = 2
groupSelectMode = 0
normType = 1          # Sigmoid
renorm = 1
```

Sigmoid 结果：

```
n = [0.880797, 0.731059, 0.500000, 0.268941]
```

加 bias 后：

```
s = [0.880797, 0.731059, 1.000000, 0.768941]
```

两组得分：

```
group 0: max(0.880797, 0.731059) = 0.880797
group 1: max(1.000000, 0.768941) = 1.000000
```

选择第 1 组，对应专家 `2、3`：

```
expertIdxOut = [[2, 3]]
```

选中权重为：

```
a = [0.500000, 0.268941]
```

renorm 后：

```
yOut = [[0.650245, 0.349755]]
```

如果 `outFlag=true`：

```
outOut = [[0.880797, 0.731059, 0.500000, 0.268941]]
```

## 2. `moe_gating_top_k_softmax`

这是较简单的固定版本，只执行：

1. 对 logits 做完整 Softmax
2. 选择 Softmax 概率最大的 TopK 专家

### 计算公式

$$
p_{b,e}=\operatorname{Softmax}(x_{b,:})_e
$$

$$
I_b=\operatorname{TopK}(p_{b,:},k)
$$

$$
y_{b,j}=p_{b,I_{b,j}}
$$

这里不会对选中的 K 个概率再次归一化，因此通常有：

$$
\sum_{j=1}^{k}y_{b,j}<1
$$

### 输入输出

- `x`：2D 或 3D logits Tensor
- `finishedOptional`：可选布尔 Tensor，shape 为 `x.shape[:-1]`
- `yOut`：TopK Softmax 概率
- `expertIdxOut`：TopK 专家索引
- `rowIdxOut`：与 TopK 结果对应的原始 token 行索引

对于 `x.shape = [B, E]`，输出 shape 均为 `[B, k]`。

如果某行 `finished=True`，该行的 `expertIdxOut` 会被设置为 `E`，即专家数，作为无效专家的哨兵值。

### 示例

```
x = [
  [2, 1, 0],
  [0, 1, 2]
]
k = 2
```

完整 Softmax：

```
p = [
  [0.665241, 0.244728, 0.090031],
  [0.090031, 0.244728, 0.665241]
]
```

输出：

```
yOut      = [[0.665241, 0.244728],
             [0.665241, 0.244728]]

expertIdx = [[0, 1],
             [2, 1]]

rowIdxOut = [[0, 2],
             [1, 3]]
```

如果：

```
finished = [False, True]
```

则第二行的专家索引为：

```
expertIdx = [[0, 1],
             [3, 3]]
```

这里 `3` 等于专家数 `E=3`，表示无效专家。

## 3. `moe_gating_top_k_softmax_v2`

这是 `moe_gating_top_k_softmax` 的扩展版本，增加了 `renorm`，可以控制 Softmax 和 TopK 的执行顺序。

### `renorm=0`

先对全部专家做 Softmax，再取 TopK：

$$
p=\operatorname{Softmax}(x)
$$

$$
I=\operatorname{TopK}(x,k)
$$

$$
y=p[I]
$$

此时输出的 K 个权重是完整专家 Softmax 概率中的一部分，通常不加和为 1。

### `renorm=1`

先取 logits 的 TopK，再只对这 K 个值做 Softmax：

$$
I=\operatorname{TopK}(x,k)
$$

$$
z=x[I]
$$

$$
y=\operatorname{Softmax}(z)
$$

因此：

$$
\sum_{j=1}^{k}y_j=1
$$

### 输入输出

- `x`：2D 或 3D logits Tensor
- `finishedOptional`：可选布尔 Tensor
- `yOut`：`[... , k]`
- `expertIdxOut`：与 `yOut` 同 shape，`int32`
- `softmaxResultOutOptional`：与 `x` 同 shape，`float32`

当：

```
renorm = 0
outputSoftmaxResultFlag = true
```

才输出完整 Softmax 结果。`renorm=1` 时该标记不生效，不输出完整 Softmax。

### 示例

仍使用：

```
x = [
  [2, 1, 0],
  [0, 1, 2]
]
k = 2
```

`renorm=0`：

```
yOut      = [[0.665241, 0.244728],
             [0.665241, 0.244728]]

expertIdx = [[0, 1],
             [2, 1]]

softmaxResult = [
  [0.665241, 0.244728, 0.090031],
  [0.090031, 0.244728, 0.665241]
]
```

`renorm=1`：

```
expertIdx = [[0, 1],
             [2, 1]]

yOut = [[0.731059, 0.268941],
        [0.731059, 0.268941]]
```

因为此时 Softmax 只在选中的 `[2, 1]` 两个 logits 上进行。

## 4. 主要区别

| 算子 | 归一化与选择顺序 | 额外能力 | 主要辅助输出 |
|---|---|---|---|
| `moe_gating_top_k` | 可选归一化，再 bias、分组、TopK | bias、分组、Sigmoid、Hash、renorm、缩放 | `outOut`：完整归一化结果 |
| `moe_gating_top_k_softmax` | 完整 Softmax 后 TopK | `finished` | `rowIdxOut` |
| `moe_gating_top_k_softmax_v2` | `renorm=0`：Softmax 后 TopK；`renorm=1`：TopK 后 Softmax | `finished`、可选输出完整 Softmax | `softmaxResultOutOptional` |

简单选择建议：

- 只需要标准 Softmax TopK 和行索引：使用 `moe_gating_top_k_softmax`
- 需要控制 TopK 后是否重新归一化：使用 `moe_gating_top_k_softmax_v2`
- 需要 bias、专家分组、Sigmoid、Hash 或路由缩放：使用 `moe_gating_top_k`

## 5. 参考源码文档

- [aclnnMoeGatingTopK.md](moe_gating_top_k/docs/aclnnMoeGatingTopK.md)
- [aclnnMoeGatingTopKSoftmax.md](moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md)
- [aclnnMoeGatingTopKSoftmaxV2.md](moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)
