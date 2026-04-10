# aclnnDequantRopeQuantKvcache

## 功能描述

对输入 `x` 可选 dequant 后，按 `size_splits` 在最后一维切分为 q、k、v；对 q、k 做 RoPE；对 k、v 做量化并按 `indices` 写入 k/v cache（支持 contiguous / page 等模式，由属性控制）。

## 接口定义

### Python

通过 PyBind 封装为 `kernel_gen_ops.dequant_rope_quant_kvcache`，参数顺序与 `aclnnDequantRopeQuantKvcacheGetWorkspaceSize` 一致：输入张量与可选张量、`size_splits`（长度为 3 的列表）、`quant_mode`、`layout`、`kv_output`、`cache_mode`，以及预分配的输出缓冲区 `q`、`k`、`v`；算子就地更新 `k_cache`、`v_cache` 并写出 `q`、`k`、`v`。

```python
def dequant_rope_quant_kvcache(
    x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v,
    offset_k, offset_v, weight_scale, activation_scale, bias,
    size_splits, quant_mode, layout, kv_output, cache_mode,
    q, k, v,
):
    ...
```

## 约束与限制

- **本文档仅描述算子功能与接口**；不包含 conda、CANN、shell、`run_multi_test`、设备错误或任何运行环境 / 排障说明。
- `x` 为 3 维 `[B, S, H]` 或 2 维 `[B, H]`（`H = size_splits` 之和）；`cos`/`sin` 与文档中 BSND 布局及维数要求一致。
- `k_cache`/`v_cache` 为 4 维 INT8；`scale_k`/`scale_v` 为 FLOAT，长度与 `n_kv * D` 一致。
- INT8 输入 `x` 的 dequant 路径需提供 `weight_scale`（及可选 `bias`/`activation_scale`），与 CANN 定义一致。
