# aclnnRopeQuantKvcache

## 功能描述

### 算子功能

将 `qkv` 在最后一维按 `size_splits` 切分为 **q、k、v**，对 **q、k** 做旋转位置编码（RoPE），再对 **k、v** 按量化参数写入 **k_cache、v_cache**（由 `indice` 指定写入位置）。

### 数据流（概念）

切分 → RoPE(q)、RoPE(k) → 量化(k)、量化(v) → 更新 cache。

## 接口定义

### Python 接口

通过 PyBind11 封装，Python 侧函数名为 `kernel_gen_ops.rope_quant_kvcache`（由编译流程将 `custom_pybind_api` 替换而来）：

```python
def rope_quant_kvcache(
    qkv: Tensor,
    cos: Tensor,
    sin: Tensor,
    quant_scale: Tensor,
    quant_offset: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    indices: Tensor,
    size_splits: List[int],
    layout: str,
    kv_output: bool,
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> List[Tensor]:
    """
    参数:
        qkv: [B,S,H]，FLOAT16，H = size_splits 三者之和。
        cos/sin: 与 layout 一致，可与 k 广播。
        quant_scale: FLOAT32，长度与 head 维 D 一致，按最后一维广播。
        quant_offset: INT32，与 quant_scale 同形状语义。
        k_cache / v_cache: INT8，与输出 cache 原地更新。
        indices: INT32，[B]，写入 cache 的序列起点下标。
        size_splits: [sq, sk, sv] 为最后一维三段长度（元素个数）。
        layout: 如 \"BSND\"。
        kv_output: 是否输出 k/v 浮点（具体行为以算子实现为准）。
        q/k/v: 预分配的 FLOAT16 输出缓冲区。

    返回:
        [q, k, v, k_cache, v_cache]（与实现一致的张量列表）。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

B, S, n_q, n_kv, D = 1, 1, 2, 1, 32
sq, sk, sv = n_q * D, n_kv * D, n_kv * D
qkv = torch.randn(B, S, sq + sk + sv, dtype=torch.float16, device="npu")
cos = torch.randn(B, S, 1, D, dtype=torch.float16, device="npu")
sin = torch.randn(B, S, 1, D, dtype=torch.float16, device="npu")
quant_scale = torch.rand(D, dtype=torch.float32, device="npu") * 0.05 + 0.01
quant_offset = torch.zeros(D, dtype=torch.int32, device="npu")
k_cache = torch.zeros(B, 16, n_kv, D, dtype=torch.int8, device="npu")
v_cache = torch.zeros_like(k_cache)
indices = torch.zeros(B, dtype=torch.int32, device="npu")
q = torch.empty(B, S, n_q, D, dtype=torch.float16, device="npu")
k = torch.empty(B, S, n_kv, D, dtype=torch.float16, device="npu")
v = torch.empty_like(k)
kernel_gen_ops.rope_quant_kvcache(
    qkv, cos, sin, quant_scale, quant_offset, k_cache, v_cache, indices,
    [sq, sk, sv], "BSND", True, q, k, v,
)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含 conda、CANN、shell、`run_multi_test`、设备或排障说明。
- 当前实现路径下，部分形状与 **S=1** 的融合内核假设一致；使用前请对照产品约束与图编译结果。
- 输入输出 dtype、维数需满足算子定义（见 `rope_quant_kvcache_def.cpp`）。
