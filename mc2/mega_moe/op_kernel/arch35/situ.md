# SiTU 激活函数实现方案

## 1. 背景与目标

### 1.1 当前状态

`block_epilogue_swiglu_mx_quant.h` 中的 epilogue 块在 GMM1 输出后执行 **SwiGLU** 激活（即 SILU 门控）：

```
SwiGLU(gate, up) = silu(gate) * up = gate / (1 + e^(-gate)) * up
```

GMM1 输出按 hidden 维拆成前后两半：
- `firstSrc`（前半）= gate 路输入（当前代码注释为 `swishInput:x`）
- `secondSrc`（后半）= up 路输入（当前代码注释为 `gateInput:y`）

两者均为 BF16，经 Cast 升 FP32 计算后降回 BF16，再走 MX 量化输出 FP8/FP4。

### 1.2 目标

在不破坏现有 SILU 路径的前提下，新增 **SiTU** 激活可选。SiTU 来自 `tanhcal.py` 的 `SituAndMul`：

```python
# tanhcal.py
gate = x[..., :d]   # 前半
up   = x[..., d:]   # 后半
situ_a = beta * tanh(gate / beta) * sigmoid(gate)
if linear_beta is not None:
    up = linear_beta * tanh(up / linear_beta)
out = (situ_a * up).to(x.dtype)
```

### 1.3 约束

| 约束 | 说明 |
|------|------|
| `__VEC_SCOPE__` 内必须使用 `if constexpr` | 运行时 `if` 破坏 RegBase 向量流水，编译期消除死分支才能保证性能 |
| `__VEC_SCOPE__` 内不能出现非 `if constexpr` 形式的分支 | "up 路是否做 tanh 变换"是结构性差异，必须编码进编译期常量，不能用运行时 `if (alpha != 1.0)` 判断 |
| 外层传参尽量简单 | 不改 apt.cpp / MegaMoe 类模板宏 / impl 函数模板链；激活模式通过 tilingData 运行时字段传入 epilogue 成员变量，在 epilogue 内部用函数模板参数转化为编译期常量 |
| 不使用 TilingKey | 避免二进制翻倍 |
| 不依赖 clamp 保证数值稳定性 | clamp 默认值为 `FLT_MAX`，等于无 clamp，单分支 tanh 必然溢出 |

### 1.4 总体方案

**核心思路**：`ActMode` 不作为 epilogue 类的模板参数（否则 `MegaMoe` 类成员 `epilogueOp_` 的类型依赖 `ActMode`，必须从 apt.cpp 一路透传模板参数）。改为：

1. epilogue 类新增运行时成员 `actMode_`（`uint8_t`）、`activationAlpha_`（`float`）、`activationBeta_`（`float`），从 tilingData 读入
2. epilogue 类内部的 `VFDoSwigluAndQuantForMX` 新增函数模板参数 `ActMode M`（带默认值 `SILU`），在 `__VEC_SCOPE__` 内用 `if constexpr (M == ...)` 分支
3. epilogue 类内部的 `VFDoSwigluForMX`（调用 `VFDoSwigluAndQuantForMX` 的上层）用运行时 `if/else if/else` 按 `actMode_` 分发到 `<SILU>` / `<SITU>` / `<SITU_LINEAR>` 实例

```
tilingData.actMode / activationAlpha / activationBeta (运行时)
       │
       │ (Init 时存入成员)
       ▼
epilogue.actMode_ / activationAlpha_ / activationBeta_ (运行时成员)
       │
       │ (VFDoSwigluForMX 内运行时 if/else if/else 分发, 不在 __VEC_SCOPE__ 内)
       ├── if (SITU_LINEAR) → VFDoSwigluAndQuantForMX<..., ActMode::SITU_LINEAR>(...)
       ├── else if (SITU)   → VFDoSwigluAndQuantForMX<..., ActMode::SITU>(...)
       └── else             → VFDoSwigluAndQuantForMX<..., ActMode::SILU>(...)
              │
              │ (M 是函数模板参数, 编译期常量)
              ▼
       __VEC_SCOPE__ 内: if constexpr (M == SILU) {...}
                         else { /* gate 路 */ if constexpr (M == SITU_LINEAR) { /* up 路 */ } }
```

- **1 份二进制**：不使用 TilingKey，apt.cpp / MegaMoe 类 / impl 全部不改
- **`if constexpr` 在 `__VEC_SCOPE__` 内**：`M` 是函数模板参数，满足编译期常量约束
- **运行时 `if/else if/else` 在 `VFDoSwigluForMX`**：不在 `__VEC_SCOPE__` 内，每个 tile 执行一次，`actMode_` 恒定，分支预测 ~100% 命中
- **向后兼容**：不传 `activation` 属性 → 默认 `"swiglu"` → `actMode=0` → SILU 路径

---

## 2. 数学原理

### 2.1 当前 SILU-GLU 实现

```
SwiGLU(gate, up) = silu(gate) * up = gate / (1 + e^(-gate)) * up
```

SILU 对大负 gate 天然安全：`gate → -∞` 时 `e^(-gate) → +∞`，`gate / e^(-gate) → 0`，不产生 NaN。

### 2.2 目标 SiTU 公式

来自 `tanhcal.py`：

```
gate = firstSrc   (前半)
up   = secondSrc  (后半)

situ_a = beta * tanh(gate / beta) * sigmoid(gate)

当 linear_beta 有效时（SITU_LINEAR 模式）:
    up_transformed = alpha * tanh(up / alpha)
    out = situ_a * up_transformed

当 linear_beta 无效时（SITU 模式, beta=1.0 默认退化为 tanh(gate)*sigmoid(gate)*up）:
    out = situ_a * up
```

其中：
- `sigmoid(gate) = gate / (exp(-gate) + 1)`
- `tanh(z)` 用双分支数值稳定实现

### 2.3 tanh 双分支数值稳定方案

tanh 有两个等价指数形式：

| 形式 | 公式 | 指数项 | 溢出方向 |
|------|------|--------|---------|
| **A** | `(e^(2z) - 1) / (e^(2z) + 1)` | `e^(2z)` | z > 0 时溢出 |
| **B** | `(1 - e^(-2z)) / (1 + e^(-2z))` | `e^(-2z)` | z < 0 时溢出 |

按 z 符号选安全形式：

| z 范围 | 选用形式 | 指数 | e^(指数) 范围 | 安全性 |
|--------|---------|------|-------------|--------|
| z < 0 | A: `(e^(2z)-1)/(e^(2z)+1)` | 2z < 0 | (0, 1) | ✓ 永不溢出 |
| z ≥ 0 | B: `(1-e^(-2z))/(1+e^(-2z))` | -2z ≤ 0 | (0, 1] | ✓ 永不溢出 |

SiTU 中有两处 tanh：
1. `tanh(gate/beta)` — gate 路
2. `tanh(up/alpha)` — up 路（仅 SITU_LINEAR）

两处均用双分支，先对输入做标量缩放（`gate/beta`、`up/alpha`），再对缩放后的值做双分支 tanh。由于缩放后的值仍可正可负，双分支依然必要。

### 2.4 数值验证

以 `tanh(z)` 为例（z = gate/beta 或 up/alpha）：

| z | 选中分支 | 指数值 | 计算结果 | tanh 真值 | 正确 |
|---|--------|--------|---------|----------|------|
| -100 | A (e^(2z)≈0) | ≈0 | (0-1)/(0+1) = -1 | -1 | ✓ |
| -1 | A (e^(-2)≈0.135) | 0.135 | (0.135-1)/(0.135+1) = -0.762 | -0.762 | ✓ |
| 0 | B (e^(0)=1) | 1 | (1-1)/(1+1) = 0 | 0 | ✓ |
| 1 | B (e^(-2)≈0.135) | 0.135 | (1-0.135)/(1+0.135) = 0.762 | 0.762 | ✓ |
| 100 | B (e^(-200)≈0) | ≈0 | (1-0)/(1+0) = 1 | 1 | ✓ |

参考实现：`mish_dag.h:81-100` 使用相同的双分支 + `CompareScalar<LT>` + `Select` 模式。

### 2.5 beta=1.0 退化行为

当 `beta=1.0`（默认）时：
```
situ_a = 1.0 * tanh(gate/1.0) * sigmoid(gate) = tanh(gate) * sigmoid(gate)
out = situ_a * up
```
即 `tanh(gate) * sigmoid(gate) * up`，与之前 TANH 方案的 `tanh(gate) * up` 相比多了 `sigmoid(gate)` 因子。这是预期行为。

---

## 3. ActMode 编码设计

### 3.1 核心问题：up 路是否变换

Python 侧 `linear_beta` 可为 `None`（跳过 up 变换）或具体值（做 `alpha*tanh(up/alpha)`）。`__VEC_SCOPE__` 内不能用运行时 `if (alpha != 1.0)` 判断——必须用 `if constexpr`。

**解决方案**：将"up 路是否做 tanh 变换"这个**结构性差异**编码进 `ActMode` 枚举，使其成为编译期常量。`alpha` 和 `beta` 的**数值本身**是运行时标量，用在 `Muls`/`Div` 等向量原语中，不产生分支。

### 3.2 枚举定义

```cpp
enum class ActMode : uint8_t {
    SILU = 0,        // swiglu: silu(gate) * up
    SITU = 1,        // situ:  beta*tanh(gate/beta)*sigmoid(gate) * up
    SITU_LINEAR = 2, // situ + linear: gate 路同上, up 路: alpha*tanh(up/alpha)
};
```

### 3.3 alpha 默认值约定

| 属性 | 默认值 | 含义 |
|------|--------|------|
| `activation_alpha` | `1.0f` | 1.0 = 不对 up 做变换（等价 linear_beta=None）→ host tiling 设 `SITU` |
| `activation_beta` | `1.0f` | beta 默认 1.0 |

host tiling 判断逻辑：

```cpp
if (strcmp(activationPtr, "situ") == 0) {
    if (*activationAlphaPtr != 1.0f) {
        tilingData->actMode = ACT_MODE_SITU_LINEAR;  // up 路做变换
    } else {
        tilingData->actMode = ACT_MODE_SITU;          // up 路不变换
    }
}
```

`alpha == 1.0` 表示不启用 up 路变换（等价 Python 侧 `linear_beta=None`）。alpha 作分母（`up/alpha`），不能为 0，host tiling 校验排除 0 和 NaN。

### 3.4 `__VEC_SCOPE__` 内分支结构

```cpp
if constexpr (M == ActMode::SILU) {
    // silu(gate) * up
} else {
    // === gate 路（SITU 和 SITU_LINEAR 共用）===
    // sigmoid(gate) = gate / (exp(-gate)+1)
    // tanh(gate/beta) 双分支
    // gate_result = beta * tanh(gate/beta) * sigmoid(gate)

    if constexpr (M == ActMode::SITU_LINEAR) {
        // === up 路: alpha * tanh(up/alpha) 双分支 ===
    }
    // out = gate_result * up_result
    // (SITU: up_result = 原始 up; SITU_LINEAR: up_result = alpha*tanh(up/alpha))
}
```

三种模式的编译结果：

| M | 编译的代码 |
|---|----------|
| SILU | silu 分支 |
| SITU | gate 路 + `Mul(out, gate_result, up)` |
| SITU_LINEAR | gate 路 + up 双分支 tanh + `Mul(out, gate_result, up_transformed)` |

---

## 4. 寄存器设计

### 4.1 现有寄存器清单

`__VEC_SCOPE__` 内的 FP32 scratch 寄存器（SILU 路径使用）：

| 寄存器 | 类型 | SILU 用途 |
|--------|------|----------|
| `vregX1F` | `RegTensor<float>` | gate（前半），升 FP32 后的值 |
| `vregX2F` | `RegTensor<float>` | up（后半），升 FP32 后的值 |
| `negReg` | `RegTensor<float>` | scratch: -x |
| `expReg` | `RegTensor<float>` | scratch: e^(-x) |
| `addsReg` | `RegTensor<float>` | scratch: e^(-x)+1 |
| `sigmoidReg` | `RegTensor<float>` | scratch: silu(x) |
| `outFReg` | `RegTensor<float>` | 输出: silu(x)*y |
| `cmpMaskReg` | `MaskReg` | tanh 双分支比较掩码 |

### 4.2 SITU 寄存器复用策略

gate 路和 up 路串行计算，scratch 寄存器可跨路复用。

#### gate 路（SITU 和 SITU_LINEAR 共用）

```
// sigmoid(gate) = gate / (exp(-gate)+1) → 存入 sigmoidReg（需存活到最终乘法）
Muls(negReg, vregX1F, -1)          // -gate
Exp(expReg, negReg)                 // exp(-gate)
Adds(addsReg, expReg, 1)           // exp(-gate)+1
Div(sigmoidReg, vregX1F, addsReg)  // sigmoid(gate) → sigmoidReg 存活

// tanh(gate/beta): 先原地缩放 vregX1F
Muls(vregX1F, vregX1F, invBeta)    // gate ← gate/beta（原 gate 不再需要，sigmoid 已存好）

// 双分支 tanh(gate/beta):
// 分支 A: resA → expReg
Muls(negReg, vregX1F, 2)           // 2*(gate/beta)
Exp(expReg, negReg)                 // e^(2*gate/beta)
Adds(addsReg, expReg, 1)           // denA
Adds(sigmoidReg_tmp, expReg, -1)   // numA  ← 但 sigmoidReg 正存活！
```

**问题**：`sigmoidReg` 存着 `sigmoid(gate)` 需要存活到最终乘法，但分支 A 的 `numA` 也需要 scratch 寄存器。

**解决**：gate 路计算中 `sigmoid(gate)` 必须存活到 `beta * tanh(gate/beta) * sigmoid(gate)` 这一步。调整计算顺序——**先算 tanh，再算 sigmoid**，因为 sigmoid 的结果可以在 tanh 完成后立即用：

```
// 步骤 1: tanh(gate/beta) → 存入 outFReg（outFReg 此时空闲）
Muls(vregX1F, vregX1F, invBeta)    // gate ← gate/beta

// 双分支 tanh:
Muls(negReg, vregX1F, 2)           // 2*(gate/beta)
Exp(expReg, negReg)                 // e^(2*gate/beta)
Adds(addsReg, expReg, 1)           // denA
Adds(sigmoidReg, expReg, -1)       // numA（sigmoidReg 此时空闲）
Div(expReg, sigmoidReg, addsReg)   // resA → expReg

Muls(negReg, vregX1F, -2)          // -2*(gate/beta)
Exp(negReg, negReg)                 // e^(-2*gate/beta)
Adds(addsReg, negReg, 1)           // denB
Adds(sigmoidReg, negReg, -1)       // e^(-2*gate/beta)-1
Muls(sigmoidReg, sigmoidReg, -1)   // numB = 1-e^(-2*gate/beta)
Div(sigmoidReg, sigmoidReg, addsReg) // resB → sigmoidReg

CompareScalar<LT>(cmpMaskReg, vregX1F, 0)
Select(outFReg, expReg, sigmoidReg, cmpMaskReg)  // tanh(gate/beta) → outFReg

// 步骤 2: beta * tanh(gate/beta)
Muls(outFReg, outFReg, beta)       // outFReg = beta*tanh(gate/beta)

// 步骤 3: sigmoid(gate)  ← 但 vregX1F 已被覆写为 gate/beta！
```

**新问题**：`vregX1F` 被缩放为 `gate/beta` 后，原始 gate 值丢失，无法再算 `sigmoid(gate)`。

**最终方案**：保留原始 gate，用**额外寄存器** `outFReg` 存 gate/beta 的双分支中间结果，原始 gate 留在 `vregX1F` 不动：

```
// 用 outFReg 临时存 gate/beta
Muls(outFReg, vregX1F, invBeta)    // outFReg = gate/beta（vregX1F 保留原始 gate）

// 双分支 tanh(outFReg): resA → expReg, resB → sigmoidReg
// ...（negReg/expReg/addsReg/sigmoidReg 作为 scratch，此时全空闲）
// Select(sigmoidReg, expReg, sigmoidReg, cmpMask)  → sigmoidReg = tanh(gate/beta)

// beta * tanh(gate/beta)
Muls(sigmoidReg, sigmoidReg, beta)  // sigmoidReg = beta*tanh(gate/beta)

// sigmoid(gate) = gate / (exp(-gate)+1)  ← vregX1F 仍有原始 gate
Muls(negReg, vregX1F, -1)
Exp(expReg, negReg)
Adds(addsReg, expReg, 1)
Div(expReg, vregX1F, addsReg)       // expReg = sigmoid(gate)（复用 expReg 存结果）

// gate_result = beta*tanh(gate/beta) * sigmoid(gate)
Mul(outFReg, sigmoidReg, expReg)    // outFReg = gate_result
// 此时 sigmoidReg、expReg、negReg、addsReg 全部释放
```

#### up 路（仅 SITU_LINEAR）

```
// tanh(up/alpha): 用 outFReg 临时存 up/alpha（vregX2F 保留原始 up）
Muls(outFReg, vregX2F, invAlpha)    // outFReg = up/alpha（vregX2F 保留原始 up）

// 双分支 tanh(outFReg): resA → expReg, resB → sigmoidReg（此时全空闲）
// Select(sigmoidReg, expReg, sigmoidReg, cmpMask) → sigmoidReg = tanh(up/alpha)

Muls(sigmoidReg, sigmoidReg, alpha)  // sigmoidReg = alpha*tanh(up/alpha)
```

#### 最终乘法

```
// SITU:      out = gate_result * up         → Mul(outFReg, outFReg, vregX2F)
// SITU_LINEAR: out = gate_result * up_transformed → Mul(outFReg, outFReg, sigmoidReg)
```

### 4.3 寄存器增量总结

| 类型 | SILU | SITU | SITU_LINEAR |
|------|------|------|-------------|
| FP32 scratch (negReg/expReg/addsReg/sigmoidReg) | 4 | 4 (同) | 4 (同) |
| FP32 (vregX1F/vregX2F/outFReg) | 3 | 3 (同) | 3 (同) |
| MaskReg | 4 (mask/mask1/mask2/mask3) | 5 (+cmpMaskReg) | 5 (+cmpMaskReg) |
| **增量** | 基准 | +1 MaskReg | +1 MaskReg |

**零数据寄存器增量**。gate 路和 up 路串行计算，scratch 寄存器跨路复用。关键：用 `outFReg` 临时存缩放后的输入（gate/beta 或 up/alpha），原始输入保留在 `vregX1F`/`vregX2F` 不动，避免数据丢失。

### 4.4 声明隔离

与现有 `weightReg` 一致：`cmpMaskReg` 在 `__VEC_SCOPE__` 顶部无条件声明，靠 `if constexpr` 隔离指令使用。声明无法隔离的原因见 C++ 作用域规则（`if constexpr` 块内声明的变量在块外不可见，而使用点在 for 循环体内）。MicroAPI 后端对未使用的 `MaskReg` 声明会优化掉。

---

## 5. 属性与接口透传

### 5.1 新增属性

在 `mega_moe_def.cpp` 中，`activation_clamp`（index 10）和 `activation_out_dtype`（原 index 11）之间插入两个新属性：

```cpp
this->Attr("activation").AttrType(OPTIONAL).String("swiglu");           // index 9, 现有
this->Attr("activation_clamp").AttrType(OPTIONAL).Float(std::numeric_limits<float>::max());  // index 10, 现有
this->Attr("activation_alpha").AttrType(OPTIONAL).Float(1.0f);          // index 11, 新增
this->Attr("activation_beta").AttrType(OPTIONAL).Float(1.0f);           // index 12, 新增
this->Attr("activation_out_dtype").AttrType(OPTIONAL).Int(...);         // index 13, 原 11 后移
// 后续属性索引全部 +2
```

### 5.2 属性索引更新（`op_tiling/mega_moe.h`）

```cpp
uint32_t attrActivationIndex = 9U;         // 不变
uint32_t attrActivationClampIndex = 10U;   // 不变
uint32_t attrActivationAlphaIndex = 11U;   // 新增
uint32_t attrActivationBetaIndex = 12U;    // 新增
uint32_t attrActivationOutDtypeIndex = 13U;// 原 11U → 13U
uint32_t attrTransposeWeight1Index = 14U;  // 原 12U → 14U
uint32_t attrTransposeWeight2Index = 15U;  // 原 13U → 15U
uint32_t attrWeight1InterleaveIndex = 16U; // 原 14U → 16U
uint32_t attrTopoTypeIndex = 17U;          // 原 15U → 17U
uint32_t attrRankNumPerServerIndex = 18U;  // 原 16U → 18U
uint32_t attrTopkWeightsTypeIndex = 19U;   // 原 17U → 19U
```

### 5.3 tilingData 字段（`mega_moe_tiling.h`）

```cpp
struct MegaMoeTilingData {
    // ... 现有字段 ...
    uint8_t actMode;            // 已有：0=swiglu(silu), 1=situ, 2=situ_linear
    float activationAlpha;      // 新增：linear_beta, 默认 1.0
    float activationBeta;       // 新增：beta, 默认 1.0
};
```

### 5.4 host tiling（`mega_moe_tiling.cpp`）

#### 校验（`CheckAttrPtrNullptr`）

```cpp
auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
auto activationAlphaPtr = attrs->GetAttrPointer<float>(config.attrActivationAlphaIndex);
auto activationBetaPtr = attrs->GetAttrPointer<float>(config.attrActivationBetaIndex);

OP_TILING_CHECK(activationPtr == nullptr, ...);
OP_TILING_CHECK(strcmp(activationPtr, "swiglu") != 0 && strcmp(activationPtr, "situ") != 0,
                OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr,
                                          "should be 'swiglu' or 'situ'"),
                return ge::GRAPH_FAILED);
OP_TILING_CHECK(activationAlphaPtr == nullptr, ...);
OP_TILING_CHECK(activationBetaPtr == nullptr, ...);
OP_TILING_CHECK(*activationBetaPtr <= 0 || std::isnan(*activationBetaPtr),
                OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation_beta", ..., "should be > 0 and not NAN"),
                return ge::GRAPH_FAILED);
OP_TILING_CHECK(*activationAlphaPtr < 0 || std::isnan(*activationAlphaPtr),
                OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation_alpha", ..., "should not be 0 and not NAN"),
                return ge::GRAPH_FAILED);
```

#### 赋值（`SetAttrParams`）

```cpp
constexpr uint8_t ACT_MODE_SWIGLU = 0U;
constexpr uint8_t ACT_MODE_SITU = 1U;
constexpr uint8_t ACT_MODE_SITU_LINEAR = 2U;

tilingData->activationAlpha = *activationAlphaPtr;
tilingData->activationBeta = *activationBetaPtr;

if (std::strcmp(activationPtr, "situ") == 0) {
    if (*activationAlphaPtr != 1.0f) {
        tilingData->actMode = ACT_MODE_SITU_LINEAR;
    } else {
        tilingData->actMode = ACT_MODE_SITU;
    }
} else {
    tilingData->actMode = ACT_MODE_SWIGLU;
}
```

#### 打印（`PrintMegaMoeTilingData`）

```cpp
OP_LOGD(nodeName, "actMode is %u", tilingData->actMode);
OP_LOGD(nodeName, "activationAlpha is %f", tilingData->activationAlpha);
OP_LOGD(nodeName, "activationBeta is %f", tilingData->activationBeta);
```

### 5.5 epilogue 类新增成员

#### Arguments 结构体

```cpp
struct Arguments {
    // ... 现有字段 ...
    float clampLimit{0.0f};
    uint8_t actMode{0};
    float activationAlpha{1.0f};   // 新增
    float activationBeta{1.0f};    // 新增
    Arguments() = default;
};
```

#### 类成员变量

```cpp
float clampLimit_{0.0f};
uint8_t actMode_{0};
float activationAlpha_{1.0f};   // 新增
float activationBeta_{1.0f};    // 新增
```

#### Init 读取

```cpp
clampLimit_ = params.clampLimit;
actMode_ = params.actMode;
activationAlpha_ = params.activationAlpha;   // 新增
activationBeta_ = params.activationBeta;     // 新增
```

### 5.6 epilogue Init 调用点（`mega_moe.h` + `mega_moe_layered.h`）

共 4 处 Init 调用需追加 `activationAlpha` / `activationBeta`：

```cpp
// 改造前（当前已有 actMode）
epilogueOp_.Init({..., tilingData->clampLimit, tilingData->actMode});

// 改造后
epilogueOp_.Init({..., tilingData->clampLimit, tilingData->actMode,
                  tilingData->activationAlpha, tilingData->activationBeta});
```

4 处调用点：

| 文件 | 调用对象 |
|------|---------|
| `mega_moe.h` | `epilogueOp_.Init(...)` |
| `mega_moe.h` | `sharedEpilogueOp_.Init(...)` |
| `mega_moe_layered.h` | `epilogueOp_.Init(...)` |
| `mega_moe_layered.h` | `sharedEpilogueOp_.Init(...)` |

### 5.7 不改动的文件

| 文件 | 不改原因 |
|------|---------|
| `mega_moe_apt.cpp` | epilogue 类类型不依赖 `ActMode`，`MegaMoe` 模板参数不变 |
| `mega_moe.h`（模板宏 + 类型别名） | `BlockEpilogueSwigluMxQuant<...>` 参数列表不变 |
| `mega_moe_impl.h` | `GroupMatmulSwigluQuant` / `GroupMatmulSwigluQuantA8W4` 模板参数不变 |
| `mega_moe_tiling_key.h` | 不使用 TilingKey |

---

## 6. 代码实现

### 6.1 枚举定义

在 `block_epilogue_swiglu_mx_quant.h` 文件头部（`SwigluQuantMsg` namespace 之后）：

```cpp
enum class ActMode : uint8_t {
    SILU = 0,
    SITU = 1,
    SITU_LINEAR = 2,
};
```

### 6.2 新增常量

在 `VFDoSwigluAndQuantForMX` 函数体内新增：

```cpp
const float scalarOne = 1.0f;
const float negScalarOne = -1.0f;
const float scalarTwo = 2.0f;
const float negScalarTwo = -2.0f;
const float scalarZero = 0.0f;
```

### 6.3 预计算标量

在 `VFDoSwigluForMX`（`__VEC_SCOPE__` 之外）预计算倒数，避免在向量循环内做 `Div`：

```cpp
float invBeta = (activationBeta_ != 0.0f) ? (1.0f / activationBeta_) : 1.0f;
float invAlpha = (activationAlpha_ != 0.0f) ? (1.0f / activationAlpha_) : 1.0f;
```

这两个值传入 `VFDoSwigluAndQuantForMX`，在 `__VEC_SCOPE__` 内用于 `Muls` 标量乘法——不产生分支。

### 6.4 函数模板参数

`VFDoSwigluAndQuantForMX` 类内声明加默认值，类外定义不带默认值：

```cpp
// 类内声明（带默认值）
template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc = false, ActMode M = ActMode::SILU>
__aicore__ inline void VFDoSwigluAndQuantForMX(...);

// 类外定义（不带默认值）
template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc, ActMode M>
__aicore__ inline void BlockEpilogueSwigluMxQuant<...>::VFDoSwigluAndQuantForMX(...)
```

### 6.5 `VFDoSwigluForMX` 运行时分发

从 2 分支扩展为 3 分支（每个 `IsInterleaved_` 分支内 3 路）：

```cpp
if constexpr (IsInterleaved_) {
    __ubuf__ DataTypeIn *l0cOutUbSecondAddr = l0cOutUbBase + singleN_;
    if (actMode_ == static_cast<uint8_t>(ActMode::SITU_LINEAR)) {
        VFDoSwigluAndQuantForMX<..., true, ActMode::SITU_LINEAR>(...);
    } else if (actMode_ == static_cast<uint8_t>(ActMode::SITU)) {
        VFDoSwigluAndQuantForMX<..., true, ActMode::SITU>(...);
    } else {
        VFDoSwigluAndQuantForMX<..., true, ActMode::SILU>(...);
    }
} else {
    __ubuf__ DataTypeIn *l0cOutUbSecondAddr = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    if (actMode_ == static_cast<uint8_t>(ActMode::SITU_LINEAR)) {
        VFDoSwigluAndQuantForMX<..., false, ActMode::SITU_LINEAR>(...);
    } else if (actMode_ == static_cast<uint8_t>(ActMode::SITU)) {
        VFDoSwigluAndQuantForMX<..., false, ActMode::SITU>(...);
    } else {
        VFDoSwigluAndQuantForMX<..., false, ActMode::SILU>(...);
    }
}
```

### 6.6 主循环激活段

```cpp
if constexpr (M == ActMode::SILU) {
    // SILU: silu(gate) * up = gate / (exp(-gate)+1) * up
    Muls(negReg, vregX1F, negScalarOne, mask);   // -gate
    Exp(expReg, negReg, mask);                    // exp(-gate)
    Adds(addsReg, expReg, scalarOne, mask);       // exp(-gate)+1
    Div(sigmoidReg, vregX1F, addsReg, mask);      // silu(gate)
    Mul(outFReg, sigmoidReg, vregX2F, mask);      // silu(gate)*up
} else {
    // === gate 路（SITU 和 SITU_LINEAR 共用）===
    // tanh(gate/beta) 双分支, 用 outFReg 临时存 gate/beta
    Muls(outFReg, vregX1F, invBeta, mask);         // outFReg = gate/beta

    // 分支 A: (e^(2z)-1)/(e^(2z)+1), z=gate/beta
    Muls(negReg, outFReg, scalarTwo, mask);
    Exp(expReg, negReg, mask);
    Adds(addsReg, expReg, scalarOne, mask);        // denA
    Adds(sigmoidReg, expReg, negScalarOne, mask);  // numA
    Div(expReg, sigmoidReg, addsReg, mask);         // resA → expReg

    // 分支 B: (1-e^(-2z))/(1+e^(-2z))
    Muls(negReg, outFReg, negScalarTwo, mask);
    Exp(negReg, negReg, mask);
    Adds(addsReg, negReg, scalarOne, mask);         // denB
    Adds(sigmoidReg, negReg, negScalarOne, mask);   // e^(-2z)-1
    Muls(sigmoidReg, sigmoidReg, negScalarOne, mask); // numB = 1-e^(-2z)
    Div(sigmoidReg, sigmoidReg, addsReg, mask);      // resB → sigmoidReg

    // 选择: z<0 → resA(expReg), z>=0 → resB(sigmoidReg)
    CompareScalar<float, CMPMODE::LT>(cmpMaskReg, outFReg, scalarZero, mask);
    Select(sigmoidReg, expReg, sigmoidReg, cmpMaskReg);  // sigmoidReg = tanh(gate/beta)

    // beta * tanh(gate/beta)
    Muls(sigmoidReg, sigmoidReg, activationBeta_, mask);  // sigmoidReg = beta*tanh(gate/beta)

    // sigmoid(gate) = gate / (exp(-gate)+1), vregX1F 仍有原始 gate
    Muls(negReg, vregX1F, negScalarOne, mask);
    Exp(expReg, negReg, mask);
    Adds(addsReg, expReg, scalarOne, mask);
    Div(expReg, vregX1F, addsReg, mask);               // expReg = sigmoid(gate)

    // gate_result = beta*tanh(gate/beta) * sigmoid(gate)
    Mul(outFReg, sigmoidReg, expReg, mask);             // outFReg = gate_result
    // sigmoidReg, expReg, negReg, addsReg 全部释放

    if constexpr (M == ActMode::SITU_LINEAR) {
        // === up 路: alpha * tanh(up/alpha) 双分支 ===
        Muls(outFReg_tmp, vregX2F, invAlpha, mask);    // 需要保存 gate_result
        // ... 双分支 tanh(up/alpha) ...
        Muls(sigmoidReg, sigmoidReg, activationAlpha_, mask);  // alpha*tanh(up/alpha)
        Mul(outFReg, gateResult, sigmoidReg, mask);     // 最终结果
    } else {
        // SITU: out = gate_result * up
        Mul(outFReg, outFReg, vregX2F, mask);
    }
}
```

**注意**：SITU_LINEAR 路径中 `gate_result` 存在 `outFReg`，但 up 路的双分支 tanh 也需要 `outFReg` 临时存 `up/alpha`。需要用另一个寄存器存 `gate_result`，或调整计算顺序。具体实现时需解决这个寄存器冲突——可用 `vregX1F`（gate 已消费完）暂存 `gate_result`。

### 6.7 尾部循环激活段

与主循环结构完全相同，仅 mask 换成 `mask1`。

### 6.8 指令序列完整对照

| 模式 | 指令数 | 说明 |
|------|--------|------|
| SILU | 5 | silu(gate)*up |
| SITU | 24 | gate 路 sigmoid(4) + gate 路双分支 tanh(11) + beta 乘(1) + sigmoid 乘(1) + gate_result(1) + 最终 up 乘(1) + 缩放(1) + 选择(1) + 其他(3) |
| SITU_LINEAR | 38 | SITU 的 24 条 + up 路双分支 tanh(11) + alpha 乘(1) + 最终乘(1) + 额外寄存器周转(1) |

### 6.9 clamp 处理

现有 clamp 代码对三个分支都保留：

```cpp
Mins(vregX1F, vregX1F, clampLimit_, mask);
Mins(vregX2F, vregX2F, clampLimit_, mask);
Maxs(vregX2F, vregX2F, -clampLimit_, mask);
```

- **SILU 路径**：行为不变
- **SITU/SITU_LINEAR 路径**：双分支 tanh 自身数值稳定，clamp 多余但无害。保留可避免分支。

---

## 7. 改动文件清单

| # | 文件 | 改动内容 | 改动量 |
|---|------|---------|--------|
| 1 | `mega_moe_def.cpp` | `activation_clamp` 和 `activation_out_dtype` 之间新增 `activation_alpha`（Float, 默认 1.0）、`activation_beta`（Float, 默认 1.0）两个属性 | 小 |
| 2 | `op_tiling/mega_moe.h` | 新增 `attrActivationAlphaIndex=11U`、`attrActivationBetaIndex=12U`；后续索引 +2 | 小 |
| 3 | `mega_moe_tiling.h` | `MegaMoeTilingData` 新增 `float activationAlpha`、`float activationBeta` 字段（`actMode` 已有） | 小 |
| 4 | `mega_moe_tiling.cpp` | 匿名 namespace 加 `ACT_MODE_SITU`/`ACT_MODE_SITU_LINEAR` 常量；校验 alpha/beta；`SetAttrParams` 读取 alpha/beta + 按 alpha 判断 SITU/SITU_LINEAR；打印 | 小 |
| 5 | `block_epilogue_swiglu_mx_quant.h` | `ActMode` 枚举改 SILU/SITU/SITU_LINEAR（替换现有 TANH）；`Arguments` 加 `activationAlpha`/`activationBeta`；成员变量 + Init 读取；`VFDoSwigluAndQuantForMX` 加模板参数 `M`（已有）；常量 `scalarTwo/negScalarTwo/scalarZero`（已有）；`cmpMaskReg` 声明（已有）；主循环 + 尾部循环 `if constexpr` 三分支（替换现有双分支）；`VFDoSwigluForMX` 运行时 3 路分发（替换现有 2 路）；预计算 `invBeta`/`invAlpha` | 中（核心） |
| 6 | `mega_moe.h` | 2 处 Init 调用追加 `tilingData->activationAlpha`/`tilingData->activationBeta` | 小 |
| 7 | `mega_moe_layered.h` | 2 处 Init 调用追加（同上） | 小 |
| 8 | `op_api/aclnn_mega_moe.cpp` | `aclnnMegaMoeGetWorkspaceSize` 签名 + 内部调用透传（见 8.2 节） | 小 |
| 9 | `torch_extension/.../csrc/mega_moe.cpp` | `NpuMegaMoe` 签名 + `ACLNN_CMD` 透传（见 8.3 节） | 小 |
| 10 | `torch_extension/.../ops/mega_moe.py` | schema + register_meta + `_npu_mega_moe` + `mega_moe()` 签名 + 透传（见 8.4-8.5 节） | 小 |
| 11 | `torch_extension/.../ops/graph_convert/graph_convert_mega_moe.py` | `MegaMoe()` + `convert_npu_mega_moe` 签名 + attrs + IrDef（见 8.6 节） | 小 |

**不改动的文件**：`mega_moe_apt.cpp`、`mega_moe_impl.h`、`mega_moe_tiling_key.h`。

---

## 8. aclnn 前置接口透传

新增的 `activation_alpha` / `activation_beta` 参数需要从 Python 用户接口一路透传到 host tiling。完整调用链：

```
Python: mega_moe() → _npu_mega_moe() → npu_mega_moe (pybind)
C++ binding: NpuMegaMoe() → ACLNN_CMD(aclnnMegaMoe)
aclnn API: aclnnMegaMegaMoeGetWorkspaceSize() → aclnnInnerMegaMoeGetWorkspaceSize()
op def: mega_moe_def.cpp (REG_OP .ATTR)
graph_convert: graph_convert_mega_moe.py (GE op attrs)
host tiling: mega_moe_tiling.cpp (读取 attr)
```

### 8.1 op 定义（`mega_moe_def.cpp`）

见 5.1 节。在 `activation_clamp` 和 `activation_out_dtype` 之间新增两个 ATTR。

### 8.2 aclnn API（`op_api/aclnn_mega_moe.cpp`）

`aclnnMegaMoeGetWorkspaceSize` 函数签名新增两个参数，透传给 `aclnnInnerMegaMoeGetWorkspaceSize`。

**函数签名**（`:104`，在 `activationClamp` 后、`topoType` 前插入）：

```cpp
aclnnStatus aclnnMegaMoeGetWorkspaceSize(
    ...,
    const char *activation, float activationClamp,
    float activationAlpha, float activationBeta,  // 新增
    int64_t topoType, ...)
```

**内部调用**（`:185-192`，`aclnnInnerMegaMoeGetWorkspaceSize` 调用处插入）：

```cpp
aclnnStatus getWorkspaceSizesRes = aclnnInnerMegaMoeGetWorkspaceSize(
    ...,
    const_cast<char *>(activation), activationClamp,
    activationAlpha, activationBeta,              // 新增
    ge::DT_UNDEFINED, false, false, 0,
    topoType, ...);
```

### 8.3 C++ binding（`torch_extension/cann_ops_transformer/ops/csrc/mega_moe.cpp`）

`NpuMegaMoe` 函数签名新增两个参数，透传给 `ACLNN_CMD`。

**函数签名**（`:32`，在 `activationClamp` 后、`dispatchQuantOutDtype` 前插入）：

```cpp
std::tuple<at::Tensor, at::Tensor> NpuMegaMoe(
    ...,
    std::string activation, c10::optional<float> activationClamp,
    c10::optional<float> activationAlpha, c10::optional<float> activationBeta,  // 新增
    c10::optional<int64_t> dispatchQuantOutDtype, ...)
```

**参数处理**（`:96` 附近，与 `activationClampValue` 同款）：

```cpp
float activationAlphaValue = activationAlpha.value_or(1.0f);
float activationBetaValue = activationBeta.value_or(1.0f);
```

**ACLNN_CMD 调用**（`:127-132`，在 `activationClampValue` 后插入）：

```cpp
ACLNN_CMD(aclnnMegaMoe, context, x, topkIds, topkWeights, ...,
          moeExpertNum, epWorldSize, cclBufferSize, maxRecvTokenNum, dispatchQuantMode, dispatchQuantResultType,
          combineQuantMode, commAlgPtr, numMaxTokensPerRank, activationPtr, activationClampValue,
          activationAlphaValue, activationBetaValue,  // 新增
          topoTypeValue, rankNumPerServerValue, topkWeightsType, y, expertTokenNums);
```

### 8.4 Python OpBuilder schema（`torch_extension/cann_ops_transformer/ops/mega_moe.py`）

三处需同步新增参数：

**a) `schema()` 字符串**（`:39-43`，在 `activation_clamp` 后插入）：

```python
'float? activation_clamp=None, '
'float? activation_alpha=None, '   # 新增
'float? activation_beta=None, '    # 新增
'int? dispatch_quant_out_dtype=None, '
```

**b) `register_meta()` / `_npu_mega_moe()` 函数签名**（`:75` 和 `:126`，在 `activation_clamp` 后插入）：

```python
activation_clamp=None,
activation_alpha=None,    # 新增
activation_beta=None,     # 新增
dispatch_quant_out_dtype=None,
```

**c) `_npu_mega_moe()` 内部调用**（`:162`，在 `activation_clamp` 后插入）：

```python
activation,
activation_clamp,
activation_alpha,    # 新增
activation_beta,     # 新增
dispatch_quant_out_dtype,
```

### 8.5 Python 用户接口（`mega_moe()` 函数）

**函数签名**（`:333-334`，在 `activation_clamp` 后插入）：

```python
activation: str = "swiglu",
activation_clamp: Optional[float] = None,
activation_alpha: Optional[float] = None,    # 新增
activation_beta: Optional[float] = None,     # 新增
weight1_type: Optional[int] = None,
```

**内部调用**（`:370-371`，在 `activation_clamp` 后插入）：

```python
activation=activation,
activation_clamp=activation_clamp,
activation_alpha=activation_alpha,    # 新增
activation_beta=activation_beta,      # 新增
dispatch_quant_out_dtype=sym_buffer.dispatch_quant_out_dtype,
```

### 8.6 graph_convert（`graph_convert_mega_moe.py`）

三处需同步：

**a) `MegaMoe()` 函数签名**（`:86-88`，在 `activation_clamp` 后插入）：

```python
activation: str = "swiglu",
activation_clamp: float = 3.4028234663852886e38,
activation_alpha: float = 1.0,    # 新增
activation_beta: float = 1.0,     # 新增
topk_weights_type: int = 0,
```

**b) `attrs` 字典**（`:148`，在 `activation_clamp` 后插入）：

```python
"activation_clamp": attr.Float(activation_clamp),
"activation_alpha": attr.Float(activation_alpha),    # 新增
"activation_beta": attr.Float(activation_beta),      # 新增
"topk_weights_type": attr.Int(topk_weights_type),
```

**c) `IrDef` 属性声明**（`:197`，在 `activation_clamp` 后插入）：

```python
.attr("activation_clamp", attr.Float(3.4028234663852886e38))
.attr("activation_alpha", attr.Float(1.0))    # 新增
.attr("activation_beta", attr.Float(1.0))     # 新增
.attr("topk_weights_type", attr.Int(0))
```

**d) `convert_npu_mega_moe()` 函数签名 + 透传**（`:225-226` 和 `:255-256`，同款新增）。

### 8.7 前置接口改动文件清单

| # | 文件 | 改动内容 | 改动量 |
|---|------|---------|--------|
| 1 | `mc2/mega_moe/op_api/aclnn_mega_moe.cpp` | `aclnnMegaMoeGetWorkspaceSize` 签名加 2 参数；`aclnnInnerMegaMoeGetWorkspaceSize` 调用透传 | 小 |
| 2 | `torch_extension/.../ops/csrc/mega_moe.cpp` | `NpuMegaMoe` 签名加 2 参数；`value_or` 处理；`ACLNN_CMD` 透传 | 小 |
| 3 | `torch_extension/.../ops/mega_moe.py` | `schema()` 字符串加 2 参数；`register_meta` / `_npu_mega_moe` 签名 + 透传；`mega_moe()` 签名 + 透传 | 小 |
| 4 | `torch_extension/.../ops/graph_convert/graph_convert_mega_moe.py` | `MegaMoe()` 签名 + attrs 字典 + IrDef 声明 + `convert_npu_mega_moe` 签名 + 透传 | 小 |

### 8.8 参数默认值一致性

各层默认值需保持一致：

| 层 | activation_alpha 默认 | activation_beta 默认 |
|----|---------------------|---------------------|
| op 定义 `mega_moe_def.cpp` | `1.0f` | `1.0f` |
| graph_convert `IrDef` | `1.0` | `1.0` |
| C++ binding `value_or` | `1.0f` | `1.0f` |
| Python `schema()` | `None`（→ C++ optional） | `None`（→ C++ optional） |
| Python `mega_moe()` | `None` | `None` |

Python 侧用 `None`（透传到 C++ 为 `c10::optional`），C++ 侧 `value_or` 转为默认值。graph_convert 的 `MegaMoe()` 因直接构建 GE 算子，用具体默认值 `1.0` / `1.0`。

---

## 9. 验证要点

### 9.1 精度验证

| 测试场景 | 输入 | 期望输出 | 验证点 |
|---------|------|---------|--------|
| SILU 回归 | gate/up 任意 | silu(gate)*up | 默认模式不回归 |
| SITU beta=1.0 | gate=0, up=1 | tanh(0)*sigmoid(0)*1 = 0 | 零点 |
| SITU beta=1.0 | gate=1, up=1 | tanh(1)*sigmoid(1) ≈ 0.4621*0.7311 ≈ 0.3378 | 常规值 |
| SITU beta=1.0 | gate=100, up=1 | tanh(100)*sigmoid(100) ≈ 1*1 = 1 | 正向饱和，无 NaN |
| SITU beta=1.0 | gate=-100, up=1 | tanh(-100)*sigmoid(-100) ≈ -1*0 = 0 | 负向饱和 |
| SITU beta=2.0 | gate=2, up=1 | 2*tanh(1)*sigmoid(2) ≈ 2*0.7616*0.8808 ≈ 1.3426 | beta 缩放 |
| SITU_LINEAR alpha=1.0 | gate=1, up=2 | (tanh(1)*sigmoid(1)) * (1*tanh(2)) ≈ 0.3378*0.9640 ≈ 0.3257 | up 路变换 |
| SITU_LINEAR alpha=2.0 | gate=1, up=2 | (tanh(1)*sigmoid(1)) * (2*tanh(1)) ≈ 0.3378*1.5233 ≈ 0.5146 | alpha 缩放 |
| 量化输出 | 任意 | MX FP8/FP4 + scale | 量化流水不受激活切换影响 |

### 9.2 编译验证

- SILU 路径（`actMode=0`）：`if constexpr` 编译 SILU 分支
- SITU 路径（`actMode=1`）：`if constexpr` 编译 gate 路 + `Mul(out, gate_result, up)`
- SITU_LINEAR 路径（`actMode=2`）：`if constexpr` 编译 gate 路 + up 双分支 + `Mul(out, gate_result, up_transformed)`
- `__VEC_SCOPE__` 内无运行时 `if` 分支
- 3 份 `VFDoSwigluAndQuantForMX` 模板实例在同一编译单元内，1 份二进制

### 9.3 接口验证

- 不传 `activation` 属性 → 默认 `"swiglu"` → `actMode=0` → SILU 路径
- 传 `activation="situ"` + `activation_alpha=1.0`（默认）→ `actMode=1` → SITU 路径
- 传 `activation="situ"` + `activation_alpha=2.0` → `actMode=2` → SITU_LINEAR 路径
- 传 `activation="situ"` + `activation_beta=0` → tiling 报错（beta 不能为 0）
- 传 `activation="invalid"` → tiling 报错
- 传 `activation="swiglu"` + `activation_alpha=2.0` → tiling 报错（swiglu 不支持 alpha/beta 非 1.0）

---

## 10. 注意事项

### 10.1 alpha 默认值

`activation_alpha = 1.0f` 表示不对 up 做变换（等价 Python 侧 `linear_beta=None`），host tiling 据此设 `actMode=SITU`。alpha 作分母（`up/alpha`），不能为 0，host tiling 校验排除 0 和 NaN。

### 10.2 beta 校验

`beta` 作分母（`gate/beta`），不能为 0。host tiling 校验 `activationBeta == 0 || isnan` 时报错。允许负值（数学上合法，仅缩放方向翻转）。alpha 同理。

### 10.3 SITU_LINEAR 寄存器冲突

SITU_LINEAR 路径中，gate_result 存在 `outFReg`，但 up 路双分支 tanh 需要临时寄存器存 `up/alpha`。解决方案：gate 路完成后 `vregX1F`（原始 gate）已消费完，可用 `vregX1F` 暂存 `gate_result`，释放 `outFReg` 供 up 路使用。

### 10.4 预计算倒数

`invBeta = 1.0f / beta` 和 `invAlpha = 1.0f / alpha` 在 `VFDoSwigluForMX`（`__VEC_SCOPE__` 之外）预计算，避免在向量循环内做标量 `Div`。这两个值作为局部变量传入 `VFDoSwigluAndQuantForMX`。

### 10.5 函数模板参数 vs 类模板参数

`ActMode` 是 `VFDoSwigluAndQuantForMX` 的**函数模板参数**，而非 `BlockEpilogueSwigluMxQuant` 的**类模板参数**。这样 epilogue 类型不依赖 `ActMode`，`MegaMoe` 类成员 `epilogueOp_` 类型不变，避免从 apt.cpp 到 impl 的全链路模板参数透传。

### 10.6 命名说明

`ActMode::TANH`（当前代码中的旧命名）需替换为 `ActMode::SITU` / `ActMode::SITU_LINEAR`。SiTU 中包含 tanh 函数，但 tanh 不是激活类型本身。类名/函数名中的 `Swiglu` 暂不更名（约 90+ 处引用，风险高，建议独立 PR 处理）。

### 10.7 参考实现

| 参考文件 | 参考内容 |
|---------|---------|
| `tanhcal.py` | SituAndMul 完整公式 |
| `ops-nn/activation/mish/op_kernel/arch35/mish_dag.h:81-100` | 双分支 + `CompareScalar<LT>` + `Select` 模式 |
| `block_epilogue_swiglu_mx_quant.h`（现有 `TopkWeightsPrefetch`） | `__VEC_SCOPE__` 内 `if constexpr` 先例；`weightReg` 无条件声明 + `if constexpr` 隔离使用的现有模式 |
