# 9375.diff 修改总结

## 背景

SiTU 激活接口仍在评审中，本 PR 先行合入 **Kernel 代码框架**，并修复之前 PR 遗留的问题。Host Tiling 侧仅搭好 TilingData 字段并写入默认值（`actMode=SWIGLU`，功能关闭），暂不读取/校验新属性。

## 核心改动

### 1. Kernel 激活框架重构

**文件**：`op_kernel/arch35/block_epilogue_swiglu_mx_quant.h`

将原 `VFDoSwigluAndQuantForMX` 中内联的 SwiGLU 计算拆分为独立的激活函数 + 分派器架构，为后续 SiTU 激活预留扩展点。

**新增类型定义**：
- `ActMode` 枚举：`SWIGLU=0`、`SITU=1`
- `ActSubMode` 枚举：`DEFAULT=0`、`LINEAR=1`
- `ActivationContext<DataTypeInT>` 结构体：封装源数据地址、对齐参数、分块循环控制、尾循环 mask/地址、权重预取地址、激活参数（beta/invBeta/alpha/invAlpha），仅含标量与 UB 指针，可安全跨函数传递（RegTensor 不能跨函数，故不含寄存器）

**新增 `COMPUTE_TANH_TWOPATH` 宏**：两段式 tanh 实现
- 多项式路（|x|<0.6）：`x*(1 + c1*x^2 + c2*x^4 + c3*x^6 + c4*x^8)`，Horner + FusedMulDstAdd
- sigmoid 分解路（|x|>=0.6）：`2/(1+e^{-2x}) - 1`，天然保号无相消
- 系数：c1=-0.333327681, c2=0.133152977, c3=-0.0523039624, c4=0.0157396831

**新增激活函数**：
- `ActivationFlowSwiGLU()`：`swish(x1) * x2`，含主循环 + 尾循环 + pad 填零
- `ActivationFlowSiTU<ActSub>()`：
  - `situ_a = beta * tanh(gate/beta) * sigmoid(gate)`
  - LINEAR：`out = situ_a * (alpha * tanh(up/alpha))`
  - DEFAULT：`out = situ_a * up`
  - tanh 采用两段式，每个函数自含全部寄存器声明与 `__VEC_SCOPE__`

**重构 `VFDoSwigluAndQuantForMX`**：
- 新增模板参数 `ActMode Activation`、`ActSubMode ActSub`
- 标量分块计算原样保留
- 填充 `ActivationContext` 后按 `Activation` 模板参数分派到对应激活函数
- 量化部分原样保留

**重构调用点**（`VFDoSwigluForMX`）：
- 运行时按 `actMode_`/`actSubMode_` 分派 4 个模板实例（SITU+LINEAR / SITU+DEFAULT / SWIGLU+DEFAULT）

**Init 新增成员**：
- `Arguments` 结构体新增 `actMode`/`actSubMode`/`activationAlpha`/`activationBeta`
- 类成员新增对应私有字段，Init 时从 params 拷贝

### 2. Kernel Init 传参

| 文件 | 改动 |
|------|------|
| `mega_moe.h` | `epilogueOp_.Init` / `sharedEpilogueOp_.Init` 新增 actMode/actSubMode/alpha/beta 参数，从 tilingData 读取 |
| `mega_moe_layered.h` | 同上，但硬编码 `ActMode::SWIGLU` / `ActSubMode::DEFAULT` / `1.0f` |
| `mega_moe_tiling.h` | `MegaMoeTilingData` 新增 4 个字段：`actMode`(uint8)、`actSubMode`(uint8)、`activationAlpha`(float)、`activationBeta`(float) |

### 3. Host Tiling 字段搭建（功能关闭）

**文件**：`op_host/op_tiling/arch35/mega_moe_tiling.cpp`

`SetAttrParams` 中硬编码默认值，不读取激活属性：
```cpp
tilingData->actMode = 0U;            // ACT_MODE_SWIGLU, 功能关闭
tilingData->actSubMode = 0U;         // ACT_SUB_MODE_DEFAULT
tilingData->activationAlpha = 1.0f;
tilingData->activationBeta = 1.0f;
```
同时新增 `maxTilesPerExpert = 0` 初始化（原代码仅在 prefetch 路径赋值，非 prefetch 路径未初始化）。

### 4. 修复之前 PR 遗留问题

#### 4a. SymmBuffer 缺少 topk_weights_type 传递

**文件**：`torch_extension/cann_ops_transformer/ops/mega_moe.py`

`SymmBuffer.__init__` 中某调用补齐遗漏的 `topk_weights_type` 参数。

#### 4b. 参数对齐格式修复

**文件**：`op_kernel/arch35/mega_moe.h`

`GroupMatmulWithCombine` 函数签名中 `vecSetSyncCom`/`gmTileSequence` 缩进对齐修正。

## 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `op_host/op_tiling/arch35/mega_moe_tiling.cpp` | TilingData 新增字段默认值 + maxTilesPerExpert 初始化 |
| `op_kernel/arch35/block_epilogue_swiglu_mx_quant.h` | 激活框架重构 |
| `op_kernel/arch35/mega_moe.h` | Init 传参 + 参数对齐修复 |
| `op_kernel/arch35/mega_moe_layered.h` | Init 传参（硬编码 SWIGLU） |
| `op_kernel/arch35/mega_moe_tiling.h` | TilingData 新增字段 |
| `torch_extension/cann_ops_transformer/ops/mega_moe.py` | 修复 SymmBuffer 遗漏 topk_weights_type |

## 兼容性

`actMode=SWIGLU` + `alpha/beta=1.0`，运行时只走 SwiGLU 路径，行为与重构前完全一致。SiTU 路径代码已就绪但被 Host 关闭，待接口评审通过后启用。
