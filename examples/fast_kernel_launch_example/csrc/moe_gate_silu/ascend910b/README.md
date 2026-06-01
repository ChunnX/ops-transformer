# MoeGateSilu

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas A2 训练系列产品 | 是 |

## 功能说明

MoeGateSilu 用于 MoE (Mixture of Experts) 架构的 Expert MLP 阶段，融合实现 SwiGLU (Gated SiLU) 激活计算，并支持通过 `total_rows` 在动态 Shape 场景下读取真实有效行数。

公开的 PyTorch 调用接口：

```python
torch.ops.ascend_ops.moe_gate_silu(x, total_rows)
```

CPU 参考公式：

```python
gate, up = x[:, :hidden], x[:, hidden:]
z = torch.nn.functional.silu(gate.float()) * up.float()
z = z.to(x.dtype)
```

其中 `hidden = x.shape[1] // 2`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | Expert MLP 的 Gate 和 Up 投影融合输入，shape 为 `(rows, 2 * hidden)` | float16, bfloat16 | ND |
| total_rows | 输入 | 动态 Shape 场景下的真实有效行数，kernel 读取最后一个元素作为 active rows | int64 | ND |
| z | 输出 | SwiGLU 激活后的输出，shape 为 `(rows, hidden)` | 与 `x` 相同 | ND |

内部 kernel launch 参数由 PyTorch wrapper 根据输入推导：

| 参数名 | 描述 |
| :--- | :--- |
| rows | `x.size(0)` |
| per_count | `x.size(1)` |
| per_total_count | `total_rows.numel()` |
| block_cnt | 当前芯片 AIV core 数 |
| stream | 当前 NPU stream |

## 约束说明

- `x` 必须是 2-D contiguous NPU Tensor。
- `x.shape[1]` 必须是偶数，表示合并后的 Gate 和 Up 投影。
- `hidden = x.shape[1] // 2` 必须满足当前 dtype 的 32B 对齐要求。
- `hidden <= 8192`。
- `total_rows` 必须是 contiguous NPU int64 Tensor，且 `total_rows[-1] <= x.size(0)`。
- 当前 PyTorch wrapper 支持 `float16` 和 `bfloat16`。

## 价值/作用

在 LLaMA、Qwen、DeepSeek 等主流大语言模型中，Expert MLP 普遍采用 SwiGLU 作为 FFN 激活函数。模型实现通常将 Gate 投影和 Up 投影合并为一次矩阵乘法，输出维度翻倍。

本算子将张量拆分、SiLU 激活、逐元素乘法三个操作融合在一个 AI Core kernel 内部执行，并支持动态有效行数读取。该实现减少中间结果在 UB 和 GM 之间的读写，降低访存带宽压力，适配推理引擎中因动态 batch、paged attention 或 padding 造成的真实 token 数变化。

## 设计方案

### Tiling 策略

- 按 `rows` 维度分核。
- 每个 core 处理若干行数据。
- 使用 tail 处理机制：无法整除的余数行分配给前若干个 core。
- 每行输入拆为左半 `gate` 和右半 `up`，输出写回对应的半维度位置。

### Kernel 侧设计

Kernel 包含 `InitTotal`、`ProcessTotal`、`Init`、`Process` 四个主要阶段。

- `InitTotal` / `ProcessTotal`
  - 从 `total_rows` 读取最后一个元素，作为真实有效行数 `real_rows`。
  - 建立 `total_rows` 的 GM Tensor 映射和输入队列。
- `Init`
  - 计算 `per_dim = per_count / 2`。
  - 按 32B 对齐计算 `align_count`。
  - 根据 UB 容量估算 `rows_per_core_per_loop`。
  - 初始化 `x`、`z` 的 GM Tensor 映射，以及输入、输出和 FP32 工作区。
- `Process`
  - 循环执行 `CopyIn -> Compute -> CopyOut`。
- `Compute`
  - 将 `gate` cast 到 FP32。
  - 计算 `silu(gate)`。
  - 将 `up` cast 到 FP32 并与 `silu(gate)` 相乘。
  - 将结果 cast 回输出 dtype。

## 调用说明

```python
import torch
import torch_npu
import ascend_ops

rows, hidden = 128, 1024
x = torch.randn(rows, hidden * 2, dtype=torch.bfloat16).npu()
total_rows = torch.tensor([rows], dtype=torch.int64).npu()

out = torch.ops.ascend_ops.moe_gate_silu(x, total_rows)
```

测试与精度对比见 `tests/moe_gate_silu`。
