# aclnnRotaryPositionEmbedding

## 功能描述

### 算子功能

对输入 `x` 与 `cos`、`sin` 做旋转位置编码，支持多种 **mode**（half / interleave / quarter / interleave-half），输出 `y` 与 `x` 同形状。

### 计算公式

见产品文档 `posembedding/rotary_position_embedding/README.md` 中四种模式的定义；实现上在 float32 上累加再 cast 回输入 dtype 可与设备结果对齐。

## 接口定义

### Python 接口

通过 PyBind11 封装，Python 侧为 `kernel_gen_ops.rotary_position_embedding(x, cos, sin, mode, out)`（由编译流程将 `custom_pybind_api` 绑定为该名）：

```python
def rotary_position_embedding(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    mode: int,
    out: Tensor,
) -> Tensor:
    """
    参数:
        x: 输入，FLOAT16/BFLOAT16/FLOAT32，ND。
        cos/sin: 与 x dtype 一致，可与 x broadcast。
        mode: 0～3，分别对应 half、interleave、quarter、interleave-half。
        out: 预分配输出，与 x 同 shape、同 dtype。

    返回:
        与 out 为同一 Tensor。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

B, N, S, D = 1, 2, 2, 64
x = torch.randn(B, N, S, D, dtype=torch.float16, device="npu")
cos = torch.randn(1, 1, S, D, dtype=torch.float16, device="npu")
sin = torch.randn(1, 1, S, D, dtype=torch.float16, device="npu")
out = torch.empty_like(x)
kernel_gen_ops.rotary_position_embedding(x, cos, sin, 0, out)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含 conda、CANN、shell、`run_multi_test`、设备或排障说明。
- **mode=2（quarter）**、**mode=3（interleave-half / DeepSeek）** 在部分芯片上由 **apt** 核实现；当前 Bench 默认目标（如 Atlas A2 910B）上 **kernel 主文件仅包含 mode 0/1**，validation 用例仅覆盖 **mode 0 与 1**。mode 2/3 的约束与公式仍以产品 README 为准。
- **mode=2** 要求 **D 能被 4 整除**；mode 0/1/3 常用要求 **D 为偶数**（以产品约束为准）。
- 数据格式为 **ND**；dtype 组合需满足算子定义。
