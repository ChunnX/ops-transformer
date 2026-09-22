# engram_gate_dsl

`engram_gate` 是 Engram（DSV4.x 系列模型）残差门算子的 CANNBotDSL 实现：融合
双路 RMS 归一化、加权点积、signed-sqrt sigmoid 门控与残差更新，为单融合 kernel
（纯单机向量计算，无通信）。本算子与 mc2 目录下的
[engram_fetch](../../mc2/engram_fetch)/[engram_fetch_grad](../../mc2/engram_fetch_grad)
等通信算子同属 Engram 机制：fetch 从分布式 Engram 存储取回 key/value，本算子
在其后完成门控残差计算。

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练/推理系列产品 | × |
| Atlas A2 训练/推理系列产品 | × |

## 接口

```python
from engram_gate import engram_gate

y = engram_gate(
    x,
    key,
    value,
    weight,
    image_mask=None,
    eps=1e-6,
    clamp_value=1e-6,
)
```

| 参数 | dtype | shape |
| :--- | :--- | :--- |
| `x`, `key` | BF16 | `[..., hc_mult, dim]`，ndim ≥ 2；前导维折叠为 token，2-D 表示 `hc_mult=1` |
| `value` | BF16 | `[..., dim]`（与 `x` 前导维一致，跨 hc 共享） |
| `weight` | FP32 | `[hc_mult, dim]` |
| `image_mask` | BOOL，可选 | `[...]`（与 `x` 前导维一致），`True` 的 token 原样返回 `x` |
| `y` | BF16 | 与 `x` 同 shape |

`dim` 支持任意正整数：不足 64 的尾部按掩码处理；超过单行驻留 UB 预算
（`ceil(dim/64)*64 > 8448`）时自动切换为列分块流式路径，无 dim 上限。模型默认
配置为 `dim=5120, hc_mult=4`。

计算公式：

```text
rstd = rsqrt(mean(x²) + eps) * rsqrt(mean(key²) + eps)
dot  = sum(x * weight * key) * rstd / sqrt(dim)
gate = sigmoid(copysign(sqrt(max(abs(dot), clamp_value)), dot))
y    = bf16(x + gate * value)      # image_mask=True 的 token 直接返回 x
```

## 依赖

- Python ≥ 3.10，`torch`、`torch_npu`
- `cannbotdsl`（CANNBotDSL，本算子为 Python DSL 实现，无需 CMake 编译；首次调用
  按形状 JIT/AOT 编译并缓存）

## 使用示例

```python
import torch
import torch_npu
from engram_gate import engram_gate

x = torch.randn(72, 4, 5120, dtype=torch.bfloat16, device="npu")
key = torch.randn(72, 4, 5120, dtype=torch.bfloat16, device="npu")
value = torch.randn(72, 5120, dtype=torch.bfloat16, device="npu")
weight = torch.randn(4, 5120, dtype=torch.float32, device="npu")
y = engram_gate(x, key, value, weight)          # [72, 4, 5120] BF16
```

预编译（可选，消除首次调用编译开销）：

```python
import cannbotdsl
from engram_gate import EngramGate

EngramGate(eps=1e-6, clamp_value=1e-6).run.compile(
    cannbotdsl.TensorSpec((72, 4, 5120), cannbotdsl.dtypes.bfloat16),
    cannbotdsl.TensorSpec((72, 4, 5120), cannbotdsl.dtypes.bfloat16),
    cannbotdsl.TensorSpec((72, 5120), cannbotdsl.dtypes.bfloat16),
    cannbotdsl.TensorSpec((4, 5120), cannbotdsl.dtypes.float32),
    cannbotdsl.TensorSpec((72, 4, 5120), cannbotdsl.dtypes.bfloat16),
)
```

## 实现说明与性能

经 Ascend950DT_9574（CANN 9.2.0-beta.1）性能调优交付，机制：image_mask select
融入单融合 kernel、fp32 向量累加 + 权重 UB 驻留 + 56 核行级映射、跨行流水装载
提前、out ping-pong + 输入三缓冲旋转。公开 API 与数值契约（BF16 输入、FP32 累
加、golden 阈值 atol=rtol=2e-2）在调优前后保持不变。

- 10 case 冻结全集（T=1…8192，msprof 10 组 warmup→target 中位）相对基线实现：
  10/10 有效提升、0 有效回退，几何加速比 **3.42×**
- 参考形状 T=72, hc_mult=4, dim=5120：kernel 耗时 14.33 → **6.60 us（2.17×）**

## torch_extension（图模式接入）

dynamo/torch.compile 无法直接穿透 CANNBotDSL 的 jit 派发；[torch_extension](torch_extension)
以 `torch.library.custom_op` 注册算子边界（含 FakeTensor 实现），供图模式使用：

```python
from torch_extension import engram_gate_torch

out = engram_gate_torch(x, key, value, weight)          # 与 engram_gate 语义一致
```

## 运行测试

```bash
bash experimental/mc2/engram_gate_dsl/tests/test.sh
# 或
pytest -q experimental/mc2/engram_gate_dsl/tests/
```

覆盖：默认形状编译、模型形状精度（含 image_mask 位级一致）、任意 dim（尾部掩码）、
任意 ndim（前导维折叠）、超大 dim（列分块路径，最大 100K）、ACL Graph 图捕获与
重放——含 `torch.npu.NPUGraph` 原生捕获（重放对新输入数据/新 mask 值结果正确）
与 torchair `npugraph_ex`（经 torch_extension 边界 dynamo 全图捕获重放，依赖
torchair，缺失时自动跳过）。
