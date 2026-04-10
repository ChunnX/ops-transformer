# aclnnRotaryPositionEmbeddingGrad

## 功能描述

### 算子功能

对正向算子 `aclnnRotaryPositionEmbedding` 做反向，根据 **mode**（half / interleave 等）计算 **dx、dcos、dsin**。其中 **dcos、dsin** 需对正向中 **broadcast** 的轴求和，形状分别与 **cos、sin** 输入一致。

### 计算公式

见产品文档 `posembedding/rotary_position_embedding_grad/README.md` 与 `docs/aclnnRotaryPositionEmbeddingGrad.md`（half / interleave 下对 `dy、cos、sin、x` 的分块与 `dx、dcos、dsin` 定义）。

## 接口定义

### Python 接口

通过 PyBind11 封装，Python 侧为 `kernel_gen_ops.rotary_position_embedding_grad(...)`（由编译流程将 `custom_pybind_api` 绑定为该名）：

```python
def rotary_position_embedding_grad(
    dy: Tensor,
    cos: Tensor,
    sin: Tensor,
    x: Tensor,
    mode: int,
    dx: Tensor,
    dcos: Tensor,
    dsin: Tensor,
) -> List[Tensor]:
    """
    参数:
        dy: 上游梯度，与正向输出 y 同形，FLOAT16/BFLOAT16/FLOAT32，ND。
        cos/sin: 与正向一致，可与 dy broadcast；dtype 与 dy 一致。
        x: 正向输入 x（可选语义在 aclnn 侧为指针；Bench 中传入与 dy 同形张量），dtype 一致。
        mode: 0～3；当前 validation 在默认芯片上仅覆盖 **0（half）** 与 **1（interleave）**（与正向主核能力一致）。
        dx/dcos/dsin: 预分配输出，形状分别为 dy.shape、cos.shape、sin.shape，dtype 各自对齐输入。

    返回:
        [dx, dcos, dsin]。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

B, N, S, D = 1, 2, 2, 64
dy = torch.randn(B, N, S, D, dtype=torch.float16, device="npu")
x = torch.randn(B, N, S, D, dtype=torch.float16, device="npu")
cos = torch.randn(1, 1, S, D, dtype=torch.float16, device="npu")
sin = torch.randn(1, 1, S, D, dtype=torch.float16, device="npu")
dx = torch.empty_like(dy)
dcos = torch.empty_like(cos)
dsin = torch.empty_like(sin)
kernel_gen_ops.rotary_position_embedding_grad(dy, cos, sin, x, 0, dx, dcos, dsin)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含 conda、CANN、shell、`run_multi_test`、设备或排障说明。
- **mode 2（quarter）**、**mode 3（interleave-half）** 在部分芯片上由 apt 核实现；默认 Bench 目标与正向一致时，validation 用例仅含 **mode 0、1**。
- 数据格式为 **ND**；dtype 组合需满足算子定义。
