# aclnnDequantBias（DequantBias）

## 功能描述

对 INT32 张量 `x` 做反量化：按 `weight_scale`、可选的 `activate_scale` 与可选的 `bias` 组合计算，输出 **FLOAT16** 或 **BFLOAT16** 的二维张量 `y`。语义与公式分支见 CANN 文档；CPU golden 与 **`ops-nn-dev/quant/dequant_bias/tests/st/aclnnDequantBias/executor_aclnnDequantBias.py`** 一致。

## 接口定义

### Python 接口

该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 **`kernel_gen_ops.dequant_bias`** 形式提供（`kernel_name` 为 **`dequant_bias`**）：

```python
def dequant_bias(
    x: Tensor,
    weight_scale: Tensor,
    activate_scale: Optional[Tensor],
    bias: Optional[Tensor],
    output_dtype: int,
):
    """
    参数:
        x: [M, N]，INT32，ND。
        weight_scale: [N]，BF16/FLOAT 等与算子定义一致。
        activate_scale: 可选，[M]，FLOAT；无则传 None。
        bias: 可选，[N]；可为 INT32（与 x 相加再乘 scale）或 FP16/BF16/FP32（乘加语义见 ST executor）。
        output_dtype: 输出类型枚举，**1** 表示 FP16，**27** 表示 BF16（与 ge 定义一致）。

    返回:
        长度为 1 的列表，元素为 [M, N] 的 FP16 或 BF16 张量。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

x = torch.randint(-10, 10, (4, 16), device="npu", dtype=torch.int32)
w = torch.rand(16, device="npu", dtype=torch.bfloat16) * 0.1
act = torch.ones(4, device="npu", dtype=torch.float32)
y_list = kernel_gen_ops.dequant_bias(x, w, act, None, 1)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含 conda、CANN、shell、`run_multi_test` 或运行环境说明。
- `x`、`y` 为 **2 维**；`weight_scale` 为 **1 维** 且长度等于 `N`；若提供 `activate_scale`，长度为 **M**；若提供 `bias`，长度为 **N**。
- `output_dtype` 仅支持 **1（FP16）** 与 **27（BF16）**。
- **dtype 白名单（EZ1001）**：输出为 **FP16（1）** 时，`weight_scale` 宜为 **FLOAT32**；输出为 **BF16（27）** 时，`weight_scale` 宜为 **BFLOAT16**（与 CANN 登记的 SupportInfo 一致）。`activate_scale` 为 **FLOAT32**。
