# <OpName> 算子

## 功能描述

<!-- TODO: 用 1-3 句话描述算子的数学功能，说明输入输出之间的关系 -->

该算子对输入张量执行逐元素 <操作描述>，输出与输入形状相同的张量。

## 接口定义

```python
kernel_gen_ops.<kernel_name>(x: torch.Tensor, ...) -> torch.Tensor
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `torch.Tensor` | 输入张量，形状为 `(*)` |
| ... | ... | ... |
| **返回值** | `torch.Tensor` | 输出张量，形状与输入相同 |

## 使用示例

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops

# 准备输入
x = torch.randn(4, 8, dtype=torch.float32, device="npu")

# 调用自定义算子
y = kernel_gen_ops.<kernel_name>(x)

print(y.shape)  # torch.Size([4, 8])
print(y.device)  # npu:0
```

## 约束与限制

### 支持的数据类型
- float32
- float16
<!-- TODO: 根据实际算子列出，如 bfloat16, int32 -->

### 张量格式
- 输入/输出必须为 ND 格式（NCHW 或任意维度）

### 其他限制
<!-- TODO: 列出 shape 约束、取值范围等 -->
- 输入张量的形状必须 > 0
