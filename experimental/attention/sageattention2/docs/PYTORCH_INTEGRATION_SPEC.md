# SageAttention2 PyTorch 接入规范与端到端集成指南

本文档介绍如何在 PyTorch 大模型训练与推理管线中接入和使用华为昇腾 Atlas 950PR 的 `SageAttention2` 算子。

## 一、接口入口概述

`sageattention2` 提供两套 Python 调用接口：

1. **统一入口 `sageattn`**：
   - 高级通用入口，自适应路由至 NPU 优化的 INT8 QK^T + FP8 PV 高性能计算流。
2. **底层直调入口 `sageattn_qk_int8_pv_fp8_asc`**：
   - 显式量化配置与全量控制参数入口，具备精细调节能力。
3. **无侵入替换器 `replace_sdpa`**：
   - 基于 Python 上下文管理器的无侵入式替换，将 PyTorch 标准 `F.scaled_dot_product_attention` 自动重定向至 `sageattn`。

## 二、快速上手代码示例

### 1. 基础调用示例（HND 布局与默认参数）

```python
import torch
from sageattention2 import sageattn

# 构造输入张量 [Batch, Head, Seq, Dim]
batch, n_heads, seq_len, head_dim = 1, 32, 4096, 128
q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

# 执行 SageAttention2 高性能前向计算
out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
print("Output shape:", out.shape)  # [1, 32, 4096, 128]
```

### 2. GQA 分组查询注意力与非对称序列

```python
# Query 具有 32 头，Key/Value 具有 8 头 (4:1 GQA)
q = torch.randn(1, 32, 4096, 128, dtype=torch.bfloat16)
k = torch.randn(1, 8, 8192, 128, dtype=torch.bfloat16)
v = torch.randn(1, 8, 8192, 128, dtype=torch.bfloat16)

out = sageattn(q, k, v, is_causal=False)
assert out.shape == (1, 32, 4096, 128)
```

### 3. 使用 `replace_sdpa` 实现模型零修改加速

```python
import torch.nn.functional as F
from sageattention2 import replace_sdpa

# 在需要加速的 Transformer 层作用域中使用上下文管理器
with replace_sdpa():
    # 原生调用将自动重定向至 sageattn
    out = F.scaled_dot_product_attention(q, k, v)
```

## 三、动态 Shape 与非法边界防御

为确保生产环境运行的健壮性，算子内置了完备的防御性边界拦截机制：

1. **输入合法性前置拦截**：
   - 输入张量必须同设备且同数据类型（FP16 或 BF16）；
   - 最后一维必须在内存中连续分布（`stride(-1) == 1`），杜绝因内存重整引入隐藏错误；
   - GQA 必须满足整除关系（`Hq % Hk == 0`）；
   - 因果注意力模式（`is_causal=True`）要求输入序列为方阵（`Sq == Sk`）。
2. **特征维度超界拦截**：
   - 特征维度 $D$ 必须严格满足 $0 < D \le 128$，对于 $D \le 0$ 或 $D > 128$ 严格抛出 `ValueError`。
3. **未知传参安全检查**：
   - 底层入口对未声明的可变关键字参数（`**kwargs`）实施严格类型拦截，禁止非法参数被静默吞掉。

## 四、相关文件索引

- 接口实现：[python/sageattention2/core.py](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/python/sageattention2/core.py)
- SDPA 替换器：[python/sageattention2/sdpa_replace.py](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/python/sageattention2/sdpa_replace.py)
- 测试验证矩阵：[tests/test_functional_matrix.py](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/test_functional_matrix.py)
