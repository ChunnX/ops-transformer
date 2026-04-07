# DynamicQuantV2

## 功能简述

对输入 `x`（fp16/bf16）做动态量化，输出 `y`（默认 int8，由 `dst_type` 指定）、`scale`、`offset`。可选 `smooth_scales`、`group_index`。默认 infershape 为 **pertoken** 的 scale/offset 形状（与 `x` 去掉最后一维一致）。

## Validation

- **golden**：`validation/module.py` 按非对称 int8 路径（`dynamic_quant_base.h::GetScaleAndOffset` + `ComputAsymmetric`）实现；与 NPU 上 offset 缓冲区非空时行为一致。
- **pybind**：`EXEC_NPU_CMD` 参数顺序与 `build_out/autogen/aclnn_dynamic_quant_v2.h` 中 `GetWorkspaceSize` 一致。
- **精度**：`custom_check_precision` 对 `y` 允许逐元素 |Δ|≤1，对 `scale`、`offset` 使用框架 `check_precision`。
