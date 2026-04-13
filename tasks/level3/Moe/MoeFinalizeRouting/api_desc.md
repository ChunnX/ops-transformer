# MoeFinalizeRouting

## 功能描述

### 算子功能
MoE计算中，最后处理合并MoE FFN的输出结果。

### 计算公式

$$
expertid=expertIdx[i,k]
$$

$$
out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_finalize_routing()` 函数形式提供：


```python
def moe_finalize_routing(expandedX: Tensor, expandedRowIdx: Tensor, x1Optional: Tensor, x2Optional: Tensor, biasOptional: Tensor, scalesOptional: Tensor, expertIdxOptional: Tensor, dropPadMode: int):
    """
    实现 moe_finalize_routing 操作。

    参数:
        expandedX (Tensor): MoE的FFN输出，Device侧的张量，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
        expandedRowIdx (Tensor): 扩展行索引，Device侧的张量，数据类型为INT32，数据格式支持ND。
        x1Optional (Tensor): 可选输入x1，Device侧的张量，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
        x2Optional (Tensor): 可选输入x2，Device侧的张量，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
        biasOptional (Tensor): 可选偏置，Device侧的张量，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
        scalesOptional (Tensor): 可选缩放因子，Device侧的张量，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
        expertIdxOptional (Tensor): 可选专家索引，Device侧的张量，数据类型为INT32，数据格式支持ND。
        dropPadMode (int): 表示是否支持丢弃模式，expandedRowIdx的排列方式。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变，表示合并后的结果。
    """

```

## 使用案例

```
import torch
import torch_npu
import kernel_gen_ops

# 创建输入张量并移到 NPU
batch_size = 16
num_experts = 8
hidden_size = 256
k = 2

expandedX = torch.randn(batch_size * k, hidden_size, dtype=torch.float16).to('npu')
expandedRowIdx = torch.randint(0, batch_size, (batch_size * k,), dtype=torch.int32).to('npu')
x1Optional = torch.randn(batch_size, hidden_size, dtype=torch.float16).to('npu')
x2Optional = torch.randn(batch_size, hidden_size, dtype=torch.float16).to('npu')
biasOptional = torch.randn(num_experts, hidden_size, dtype=torch.float16).to('npu')
scalesOptional = torch.randn(batch_size, k, dtype=torch.float16).to('npu')
expertIdxOptional = torch.randint(0, num_experts, (batch_size, k), dtype=torch.int32).to('npu')
dropPadMode = 0

# 调用算子（输入张量需要在 NPU 上）
result = kernel_gen_ops.moe_finalize_routing(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional, expertIdxOptional, dropPadMode)

# 将结果移回 CPU 以便打印
output_cpu = result.to('cpu')
print(output_cpu.shape)
# 输出: torch.Size([16, 256])
```
## 约束与限制

- 输入张量的shape大小不要超过device可分配的内存上限，否则会导致异常终止。
- 输入expandedRowIdx和expertIdxOptional的数据类型必须为INT32。
- 其他输入张量的数据类型必须为FLOAT16、BFLOAT16或FLOAT32，且保持一致。
- 输出张量的数据类型与输入一致。