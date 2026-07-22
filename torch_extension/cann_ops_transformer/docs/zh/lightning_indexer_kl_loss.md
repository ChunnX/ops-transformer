# lightning\_indexer\_kl\_loss

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- API 功能：

  计算 teacher 侧原始未归一化主注意力分数（`target_score`）与 student 侧 indexer softmax 后概率分布（`index_probs`）之间的 KL 散度。teacher 侧用 `clamp_min` 防止 y=0 处 log(0) 导致 `0·(−inf)=NaN`，student 侧用 `+eps` 保住 Y→0 处的梯度。

- 计算公式：

  $$
  P = \frac{y}{\sum(y) + \varepsilon}
  $$

  $$
  \text{log\_P} = \log(\text{clamp\_min}(P, \varepsilon))
  $$

  $$
  \text{log\_Y} = \log(Y + \varepsilon)
  $$

  $$
  \text{loss} = \sum_{i} (\text{log\_P}_i - \text{log\_Y}_i) \cdot y_i
  $$

  其中 $y$ 为 `target_score`，$Y$ 为 `index_probs`，$\varepsilon$ 为 `eps`。

## 函数原型

```python
from cann_ops_transformer import lightning_indexer_kl_loss

lightning_indexer_kl_loss(
    target_score,
    index_probs,
    eps=1e-9
) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- | :--- |
| target_score | 必选输入 | teacher 侧压缩段原始未归一化主注意力分数，sum != 1。 | `float16`、`bfloat16`、`float32` | `(..., D)` |
| index_probs | 必选输入 | student 侧 indexer softmax 后的概率分布。 | 与 `target_score` 一致 | `(..., D)`，与 `target_score` 一致 |
| eps | 可选属性 | 数值稳定常数。 | `float` | - |

## 返回值说明

| 输出 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- |
| loss | KL 散度标量损失。 | 与输入一致 | `()`，标量 |

## 约束说明

- `target_score` 与 `index_probs` 的数据类型必须一致，且最后一维 `D` 相等。
- `eps` 必须大于 0。
- `target_score` 与 `index_probs` 需在同一 NPU 设备上。

## 调用示例

### 单算子调用

```python
import torch
import torch_npu
from cann_ops_transformer import lightning_indexer_kl_loss

torch_npu.npu.set_device(0)
device = torch.device("npu:0")

B, H, D = 4, 10, 128

target_score = torch.rand(B, H, D, dtype=torch.float32, device=device)
index_probs = torch.softmax(torch.randn(B, H, D, dtype=torch.float32, device=device), dim=-1)

loss = lightning_indexer_kl_loss(target_score, index_probs)
torch.npu.synchronize()
print(f"KL loss: {loss.item()}")
```

### 混合精度调用

```python
import torch
import torch_npu
from cann_ops_transformer import lightning_indexer_kl_loss

torch_npu.npu.set_device(0)
device = torch.device("npu:0")

B, H, D = 4, 10, 128

target_score = torch.rand(B, H, D, dtype=torch.float16, device=device)
index_probs = torch.softmax(torch.randn(B, H, D, dtype=torch.float16, device=device), dim=-1)

loss = lightning_indexer_kl_loss(target_score, index_probs, eps=1e-9)
torch.npu.synchronize()
print(f"KL loss (fp16): {loss.item()}")
```
