# 9343.diff 修改总结

## 概述

为 `mega_moe` 算子新增 **SiTU 激活函数**支持（此前仅支持 SwiGLU），并引入 `activation_alpha`、`activation_beta` 两个标量参数。改动贯穿 API、Host Tiling、Kernel、Python 扩展四层。

## 核心改动

### 1. 新增激活模式

| 模式 | actMode | 公式 |
|------|---------|------|
| SwiGLU | 0 | `swish(x1) * x2`，不支持 alpha/beta（必须为 1.0） |
| SiTU | 1 | `situ_a = beta * tanh(gate/beta) * sigmoid(gate)` |

SiTU 子模式（`actSubMode`）：
- DEFAULT(0)：`out = situ_a * up`
- LINEAR(1)：`out = situ_a * (alpha * tanh(up/alpha))`（仅当 alpha != 1.0 时启用）

### 2. API / 属性层

- `aclnn_mega_moe.cpp`：`aclnnMegaMoeGetWorkspaceSize` 新增 `float activationAlpha, float activationBeta` 入参
- `mega_moe_def.cpp`：新增可选属性 `activation_alpha`(默认1.0)、`activation_beta`(默认1.0)
- `mega_moe.h`(host)：`MegaMoeConfig` 属性索引调整，alpha=11、beta=12，后续顺延

### 3. Tiling 层 (arch35)

- 新增常量 `ACT_MODE_SWIGLU/SITU`、`ACT_SUB_MODE_DEFAULT/LINEAR`
- **属性校验**：
  - activation 必须为 `"swiglu"` 或 `"situ"`
  - alpha/beta 不能为 0、不能为 NAN
  - swiglu 不支持 alpha/beta != 1.0
  - URMA topology 不支持 situ
- `SetAttrParams`：根据 activation 值和 alpha 值决定 actMode/actSubMode，写入 tilingData
- `mega_moe_tiling.h`：TilingData 新增 `actMode`/`actSubMode`(uint8)、`activationAlpha`/`activationBeta`(float)

### 4. Kernel 层 (block_epilogue_swiglu_mx_quant.h)

**重构：将原内联 swiglu 计算拆分为独立激活函数 + 分派器**

- 新增 `ActMode`/`ActSubMode` 枚举
- 新增 `ActivationContext<DataTypeInT>` 结构体：封装源数据地址、对齐参数、分块循环控制、尾循环 mask/地址、权重预取地址、激活参数（beta/invBeta/alpha/invAlpha），可跨函数传递
- 新增 `COMPUTE_TANH_TWOPATH` 宏：两段式 tanh（|x|<0.6 走多项式 Horner，|x|>=0.6 走 sigmoid 分解），保号无相消
- 新增 `ActivationFlowSwiGLU()`：SwiGLU 完整计算流（主循环+尾循环+pad）
- 新增 `ActivationFlowSiTU<ActSub>()`：SiTU 完整计算流，按子模式分支
- `VFDoSwigluAndQuantForMX` 重构为分派器：填充 ActivationContext → 按模板参数 `Activation` 分派 → 量化部分原样保留
- 调用点（`VFDoSwigluForMX`）改为运行时按 `actMode_`/`actSubMode_` 分派 4 个模板实例
- `mega_moe.h`/`mega_moe_layered.h`：Init 调用新增激活参数（layered 硬编码 SWIGLU/DEFAULT）

### 5. Python 扩展层

- `csrc/mega_moe.cpp`：`NpuMegaMoe` 新增 `activation_alpha`/`activation_beta` 可选参数
- `mega_moe.py`：OpBuilder schema、`_npu_mega_moe`、`mega_moe()` 均新增两参数
- `graph_convert_mega_moe.py`：图转换 ATTR 定义新增两属性

### 6. 单元测试

- `test_mega_moe_tiling.cpp`、`test_mega_moe_infershape.cpp`：所有测试用例补齐 `activation_alpha=1.0f`、`activation_beta=1.0f` 属性

## 设计要点

1. **寄存器隔离**：RegTensor 不能跨函数传递，故每个 ActivationFlow* 函数自含全部寄存器声明与 `__VEC_SCOPE__`
2. **数值稳定**：tanh 采用两段式避免大输入时的 catastrophic cancellation
3. **向后兼容**：alpha/beta 默认 1.0，SwiGLU 路径行为不变
4. **编译期分派**：通过模板参数 `ActMode`/`ActSubMode` 生成 4 个特化版本，运行时按 tiling 字段选择

## 涉及文件

| 层 | 文件 |
|----|------|
| API | `op_api/aclnn_mega_moe.cpp` |
| Host Def | `op_host/mega_moe_def.cpp` |
| Host Tiling | `op_host/op_tiling/arch35/mega_moe_tiling.cpp`、`op_host/op_tiling/mega_moe.h` |
| Kernel | `op_kernel/arch35/block_epilogue_swiglu_mx_quant.h`、`op_kernel/arch35/mega_moe.h`、`op_kernel/arch35/mega_moe_layered.h`、`op_kernel/arch35/mega_moe_tiling.h` |
| UT | `tests/ut/op_host/arch35/test_mega_moe_tiling.cpp`、`tests/ut/op_host/test_mega_moe_infershape.cpp` |
| Python | `torch_extension/.../csrc/mega_moe.cpp`、`ops/mega_moe.py`、`ops/graph_convert/graph_convert_mega_moe.py` |
