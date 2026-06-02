# MoeQuant

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Atlas A2 训练系列产品 | 是 |

## 功能说明

MoeQuant 用于 MoE (Mixture of Experts) 架构的 W8A8 低比特推理场景，对路由分发后进入 Expert MLP 的激活值执行动态 Per-Channel 量化。

公开的 PyTorch 调用接口：

```python
torch.ops.npu_ops_transformer_ext.moe_quant(x, local_rows, quant_scale)
```

CPU 参考公式：

```python
out = torch.empty_like(x, dtype=torch.int8)
start = 0
for expert_idx, end in enumerate(local_rows.tolist()):
    scaled = x[start:end].float() * quant_scale[expert_idx].float()
    scaled = scaled.clamp(-128.0, 127.0).to(torch.float16)
    out[start:end] = torch.round(scaled).to(torch.int8)
    start = end
```

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | Expert MLP 的输入激活值，shape 为 `(rows, hidden)` | bfloat16 | ND |
| local_rows | 输入 | 每个 Expert 处理 token 数的前缀和，shape 为 `(expert_num,)` | int64 | ND |
| quant_scale | 输入 | Per-Channel 缩放因子，shape 为 `(expert_num, hidden)` | float32 | ND |
| outputX | 输出 | 量化后的激活值，shape 为 `(rows, hidden)` | int8 | ND |

内部 kernel launch 参数由 PyTorch wrapper 根据输入推导：

| 参数名 | 描述 |
| :--- | :--- |
| token_rows | `x.size(0)` |
| expert_num | `local_rows.numel()` |
| hidden | `x.size(1)` |
| block_dim | 当前实现使用 40 个 AIV core |
| stream | 当前 NPU stream |

## 约束说明

- `x` 必须是 2-D contiguous NPU Tensor，dtype 为 `bfloat16`。
- `local_rows` 必须是 1-D contiguous NPU Tensor，dtype 为 `int64`。
- `quant_scale` 必须是 2-D contiguous NPU Tensor，dtype 为 `float32`。
- `quant_scale.shape == (local_rows.numel(), x.shape[1])`。
- `local_rows` 表示前缀和，测试用例按 `local_rows[-1] == x.shape[0]` 构造。
- `hidden` 需要满足 bfloat16 的 32B 对齐要求，即 `hidden` 是 16 的整数倍。

## 价值/作用

在 LLaMA、Qwen 等模型的 W8A8 量化推理部署中，激活值量化是 INT8 Matmul 的前置步骤。本算子支持按 Expert 对激活值做 Per-Channel 量化，减少 Python 或框架侧逐 Expert 切分、缩放、clamp、round 的开销。

## 设计方案

### Tiling 策略

- 按 Expert 维度分核。
- 每个 core 处理一个或多个 Expert。
- 通过 `local_rows` 的前缀和计算当前 Expert 对应的 token 范围。
- 对每个 token 的 `hidden` 维执行 per-channel 缩放和 int8 量化。

### Kernel 侧设计

Kernel 包含 `Init` 和 `Process` 两个阶段。

- `Init`
  - 计算 `hidden` 和 `local_rows` 的 32B 对齐长度。
  - 建立 `x`、`local_rows`、`quant_scale`、`outputX` 的 GM Tensor 映射。
  - 初始化输入、输出队列和 FP32/FP16 工作区。
- `Process`
  - 搬入 `local_rows`。
  - 遍历当前 core 分配到的 Expert。
  - 读取 Expert 的 token 起止范围和对应 `quant_scale`。
  - 对每个 token 执行 `x * quant_scale -> clamp[-128, 127] -> fp16 -> round int8`。
  - 将结果写回 `outputX`。

## 调用说明

```python
import torch
import torch_npu
import npu_ops_transformer_ext

counts = torch.tensor([2, 0, 3], dtype=torch.int64)
local_rows = torch.cumsum(counts, dim=0).npu()
x = torch.randn(int(local_rows[-1]), 64, dtype=torch.bfloat16).npu()
quant_scale = torch.rand(3, 64, dtype=torch.float32).npu()

out = torch.ops.npu_ops_transformer_ext.moe_quant(x, local_rows, quant_scale)
```

测试与精度对比见 `test/test_moe_quant.py`。
