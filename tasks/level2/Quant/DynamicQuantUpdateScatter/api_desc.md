# DynamicQuantUpdateScatter

## 功能简述

对 `updates`（fp16/bf16）做与 DynamicQuant 一致的 per-切片动态量化，并按 `indices` 将结果 scatter 写回 `var`（int8）与 `var_scale`（float）。`reduce` 为必填字符串（tiling 校验允许 `update` / `none` / 空串）；`axis` 可选，验证侧使用 **-2** 与 `op_host` 中 `GetTilingNeg2` 路径一致。

## 输入输出

| 名称 | 类型 | 说明 |
|------|------|------|
| var | int8 | 被就地更新的量化张量 |
| var_scale | float32 | 与 `var` **同秩**，最后一维为 **1**（如 `var` `[D0,D1,D2,D3]` → `var_scale` `[D0,D1,D2,1]`），就地更新 |
| indices | int32/int64 | 1D 或 2D（最后一维 2） |
| updates | float16/bf16 | 与 `var` 同形状 |
| smooth_scales | 可选 fp16/bf16 | 与 `updates` 最后一维长度一致 |

输出与输入 `var`、`var_scale` 同形（就地语义）。

## Validation

- **golden**：`validation/module.py` 按 `dynamic_quant_update_scatter_comm.h::ComputeQuant` 与 `GetDetOffsetNeg2` 实现 CPU 参考。
- **精度**：`prepare_inputs.custom_check_precision` 对 int8 允许逐元素 |Δ|≤1，对 `var_scale` 使用框架 `check_precision`。
