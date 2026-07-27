# Tanh 激活函数实现方案

## 1. 背景与目标

### 1.1 当前状态

`block_epilogue_swiglu_mx_quant.h` 中的 epilogue 块在 GMM1 输出后执行 **SwiGLU** 激活（即 SILU 门控）：

```
SwiGLU(x, y) = silu(x) * y = x / (1 + e^(-x)) * y
```

其中 `x` 是 swish 输入（GMM1 前半输出），`y` 是 gate 输入（GMM1 后半输出），两者均为 BF16，经 Cast 升 FP32 计算后降回 BF16，再走 MX 量化输出 FP8/FP4。

### 1.2 目标

在不破坏现有 SILU 路径的前提下，新增 **TANH 门控**可选：

```
TanhGLU(x, y) = tanh(x) * y
```

### 1.3 约束

| 约束 | 说明 |
|------|------|
| `__VEC_SCOPE__` 内必须使用 `if constexpr` | 运行时 `if` 破坏 RegBase 向量流水，编译期消除死分支才能保证性能 |
| 激活模式在 `__VEC_SCOPE__` 内必须是编译期常量 | `if constexpr` 的条件必须是模板参数或 `constexpr` 值 |
| 外层传参尽量简单 | 不改 apt.cpp / MegaMoe 类模板宏 / impl 函数模板链；激活模式通过 tilingData 运行时字段传入 epilogue 成员变量，在 epilogue 内部用函数模板参数转化为编译期常量 |
| 不使用 TilingKey | 避免二进制翻倍（`TopkWeightsPrefetch` 已翻倍，再加 `ActMode` 会 ×4） |
| 不依赖 clamp 保证数值稳定性 | clamp 默认值为 `FLT_MAX`（`mega_moe_def.cpp:171`），等于无 clamp，单分支 tanh 必然溢出 |

### 1.4 总体方案

**核心思路**：`ActMode` 不作为 epilogue 类的模板参数（否则 `MegaMoe` 类成员 `epilogueOp_` 的类型依赖 `ActMode`，必须从 apt.cpp 一路透传模板参数）。改为：

1. epilogue 类新增运行时成员 `actMode_`（`uint8_t`），从 tilingData 读入
2. epilogue 类内部的 `VFDoSwigluAndQuantForMX` 新增函数模板参数 `ActMode M`（带默认值 `SILU`），在 `__VEC_SCOPE__` 内用 `if constexpr (M == ...)` 分支
3. epilogue 类内部的 `VFDoSwigluForMX`（调用 `VFDoSwigluAndQuantForMX` 的上层）用运行时 `if/else` 按 `actMode_` 分发到 `<SILU>` 或 `<TANH>` 实例

```
tilingData.actMode (uint8_t, 运行时)
       │
       │ (Init 时存入成员)
       ▼
epilogue.actMode_ (uint8_t, 运行时成员)
       │
       │ (VFDoSwigluForMX 内运行时 if/else 分发, 不在 __VEC_SCOPE__ 内)
       ├── if (actMode_ == TANH) → VFDoSwigluAndQuantForMX<..., ActMode::TANH>(...)
       └── else                  → VFDoSwigluAndQuantForMX<..., ActMode::SILU>(...)
              │
              │ (M 是函数模板参数, 编译期常量)
              ▼
       __VEC_SCOPE__ 内: if constexpr (M == SILU) {...} else {...}
```

- **1 份二进制**：不使用 TilingKey，apt.cpp / MegaMoe 类 / impl 全部不改
- **`if constexpr` 在 `__VEC_SCOPE__` 内**：`M` 是函数模板参数，满足编译期常量约束
- **运行时 `if/else` 在 `VFDoSwigluForMX`**：不在 `__VEC_SCOPE__` 内，每个 tile 执行一次，`actMode_` 恒定，分支预测 ~100% 命中
- **向后兼容**：不传 `activation` 属性 → 默认 `"swiglu"` → `actMode=0` → SILU 路径

### 1.5 方案对比

| 方案 | 二进制份数 | 改动文件数 | apt.cpp | MegaMoe 类 | impl | `__VEC_SCOPE__` 机制 |
|------|----------|----------|---------|-----------|------|---------------------|
| TilingKey | 2 | 7 | 不改 | 改 | 改 | if constexpr |
| 类模板参数 + apt.cpp if/else | 1 | 6 | 改 | 改 | 改 | if constexpr |
| **本方案：运行时成员 + 内部函数模板** | **1** | **5** | **不改** | **仅 Init 调用** | **不改** | **if constexpr** |

本方案优势：改动收敛在 epilogue 文件 + host tiling + 2 个 Init 调用点，不牵动模板参数链。

---

## 2. 数学原理

### 2.1 当前 SILU-GLU 实现

当前代码（`block_epilogue_swiglu_mx_quant.h`，SILU 分支）：

```cpp
Muls(negReg, x, -1.0f);           // -x
Exp(expReg, negReg);              // e^(-x)
Adds(addsReg, expReg, 1.0f);     // e^(-x) + 1
Div(sigmoidReg, x, addsReg);     // silu(x) = x / (e^(-x) + 1)
Mul(outFReg, sigmoidReg, y);     // silu(x) * y
```

SILU 对大负 x 天然安全：`x → -∞` 时 `e^(-x) → +∞`，`x / e^(-x) → 0`（IEEE 规定有限/inf = 0），不产生 NaN。因此 SILU 不依赖 clamp。

### 2.2 目标 TANH-GLU 公式

```
tanh(x) * y
```

tanh 有两个等价指数形式：

| 形式 | 公式 | 指数项 | 溢出方向 |
|------|------|--------|---------|
| **A** | `(e^(2x) - 1) / (e^(2x) + 1)` | `e^(2x)` | x > 0 时 2x > 0，e^(2x) 可能溢出 |
| **B** | `(1 - e^(-2x)) / (1 + e^(-2x))` | `e^(-2x)` | x < 0 时 -2x > 0，e^(-2x) 可能溢出 |

### 2.3 单分支的数值溢出问题

以形式 B（`(1 - e^(-2x)) / (1 + e^(-2x))`）为例：

| x | e^(-2x) | 分子 1-e^(-2x) | 分母 1+e^(-2x) | tanh 结果 |
|---|---------|---------------|---------------|----------|
| 0 | 1 | 0 | 2 | 0 ✓ |
| 5 | e^(-10) ≈ 0 | ≈1 | ≈1 | ≈1 ✓ |
| -20 | e^(40) ≈ 2.4e17 | -2.4e17 | 2.4e17 | -1 ✓ |
| **-45** | **e^(90) ≈ 1.2e39 > FLT_MAX** | **-inf** | **+inf** | **NaN ✗** |

当 `x < -44.4` 时 `e^(-2x) > FLT_MAX(3.4e38)`，`inf / inf = NaN`。

形式 A 同理在 `x > 44` 时溢出。**单分支无论选哪个形式，都有一侧必然溢出**。

当前算子的 `clampLimit` 默认值为 `std::numeric_limits<float>::max()`（`mega_moe_def.cpp:171`），`Mins(x, FLT_MAX)` 形同虚设，无法防溢出。

### 2.4 双分支数值稳定方案

核心思想：**按 x 符号选安全的形式，使两个分支的指数恒 ≤ 0**。

| x 范围 | 选用形式 | 指数 | e^(指数) 范围 | 安全性 |
|--------|---------|------|-------------|--------|
| x < 0 | A: `(e^(2x)-1)/(e^(2x)+1)` | 2x < 0 | (0, 1) | ✓ 永不溢出 |
| x ≥ 0 | B: `(1-e^(-2x))/(1+e^(-2x))` | -2x ≤ 0 | (0, 1] | ✓ 永不溢出 |

两个分支的指数恒为负或零，`e^(非正数)` 永远 ∈ (0, 1]，**完全不依赖 clamp**。

### 2.5 数值验证

| x | 选中分支 | 指数值 | 计算结果 | tanh 真值 | 正确 |
|---|--------|--------|---------|----------|------|
| -100 | A (e^(2x)=e^(-200)≈0) | ≈0 | (0-1)/(0+1) = -1 | -1 | ✓ |
| -1 | A (e^(-2)≈0.135) | 0.135 | (0.135-1)/(0.135+1) = -0.762 | -0.762 | ✓ |
| 0 | B (e^(0)=1) | 1 | (1-1)/(1+1) = 0 | 0 | ✓ |
| 1 | B (e^(-2)≈0.135) | 0.135 | (1-0.135)/(1+0.135) = 0.762 | 0.762 | ✓ |
| 100 | B (e^(-200)≈0) | ≈0 | (1-0)/(1+0) = 1 | 1 | ✓ |

参考实现：`mish_dag.h:81-100` 使用相同的双分支 + `CompareScalar<LT>` + `Select` 模式实现 `tanh(softplus(x))`，已验证可行。

---

## 3. 寄存器设计

### 3.1 现有 SILU 寄存器清单

`__VEC_SCOPE__` 内当前声明的寄存器：

| 寄存器 | 类型 | SILU 用途 | 备注 |
|--------|------|----------|------|
| `vregX1` | `RegTensor<DataTypeIn>` | 搬入 x（BF16） | 两个分支共用 |
| `vregX2` | `RegTensor<DataTypeIn>` | 搬入 y（BF16） | 两个分支共用 |
| `vregX1F` | `RegTensor<float>` | x 升 FP32 | 两个分支共用 |
| `vregX2F` | `RegTensor<float>` | y 升 FP32 | 两个分支共用 |
| `negReg` | `RegTensor<float>` | -x | scratch |
| `expReg` | `RegTensor<float>` | e^(-x) | scratch |
| `addsReg` | `RegTensor<float>` | e^(-x)+1 | scratch |
| `sigmoidReg` | `RegTensor<float>` | silu(x) | scratch |
| `outFReg` | `RegTensor<float>` | silu(x)*y | 输出 |
| `weightReg` | `RegTensor<float>` | topk 权重 | **无条件声明，仅 `TopkWeightsPrefetch` 使用** |
| `outTReg` | `RegTensor<bfloat16_t>` | FP32→BF16 | 两个分支共用 |
| `zeroReg` | `RegTensor<bfloat16_t>` | pad 填零 | 两个分支共用 |
| `mask` | `MaskReg` | 全 1 掩码 | 主循环 |
| `mask1` | `MaskReg` | 尾部掩码 | tail 循环 |
| `mask2` | `MaskReg` | 尾部存储掩码 | tail 循环 |
| `mask3` | `MaskReg` | pad 存储掩码 | tail 循环 |

### 3.2 TANH 双分支寄存器复用映射

TANH 双分支的关键：**两个分支串行计算，resA 和 resB 分别落入不同寄存器，供 `Select` 同时读取**。通过精心规划生命周期，复用 SILU 的同一组 scratch 寄存器：

#### 分支 A：`resA = (e^(2x)-1)/(e^(2x)+1)`，结果落入 `expReg`

```
Muls(negReg, x, 2.0f)           // negReg = 2x
Exp(expReg, negReg)             // expReg = e^(2x)
Adds(addsReg, expReg, 1.0f)    // addsReg = e^(2x)+1  (denA)
Adds(sigmoidReg, expReg, -1.0f) // sigmoidReg = e^(2x)-1 (numA)
Div(expReg, sigmoidReg, addsReg) // expReg = resA  ← 覆写 e^(2x)，此时 e^(2x) 已无用
```

此时寄存器状态：
- `expReg` = resA **(存活，分支 B 不可覆写)**
- `negReg` / `addsReg` / `sigmoidReg` = 空闲

#### 分支 B：`resB = (1-e^(-2x))/(1+e^(-2x))`，结果落入 `sigmoidReg`

```
Muls(negReg, x, -2.0f)          // negReg = -2x
Exp(negReg, negReg)             // negReg = e^(-2x)  ← 覆写 -2x
Adds(addsReg, negReg, 1.0f)    // addsReg = 1+e^(-2x) (denB)  ← 覆写 denA
Adds(sigmoidReg, negReg, -1.0f) // sigmoidReg = e^(-2x)-1  ← 覆写 numA
Muls(sigmoidReg, sigmoidReg, -1.0f) // sigmoidReg = 1-e^(-2x) (numB)
Div(sigmoidReg, sigmoidReg, addsReg) // sigmoidReg = resB  ← 覆写，得 tanh(B)
```

此时寄存器状态：
- `expReg` = resA **(仍存活)**
- `sigmoidReg` = resB **(存活)**
- `negReg` / `addsReg` = 空闲

#### 选择与门控

```
CompareScalar<float, CMPMODE::LT>(cmpMaskReg, x, 0.0f, mask)  // cmpMaskReg = (x < 0)
Select(sigmoidReg, expReg, sigmoidReg, cmpMaskReg)             // sigmoidReg = (x<0) ? resA : resB
                                                                // Select 写入源操作数合法（mish_dag.h:99 同款）
Mul(outFReg, sigmoidReg, y, mask)                              // outFReg = tanh(x) * y
```

### 3.3 寄存器增量总结

| 类型 | SILU 现有 | TANH 双分支 | 增量 |
|------|----------|------------|------|
| `RegTensor<float>` 数据寄存器 | 5 (negReg/expReg/addsReg/sigmoidReg/outFReg) | 5 (同名复用) | **0** |
| `MaskReg` | 4 (mask/mask1/mask2/mask3) | 5 (+cmpMaskReg) | **+1** |
| `RegTensor<float> weightReg` | 1 (无条件声明) | 1 (同) | 0 |

**双分支 tanh 只需新增 1 个 `MaskReg`（`cmpMaskReg`），数据寄存器零增量。**

### 3.4 寄存器声明与"隔离"问题

#### C++ 作用域限制

`if constexpr` 在 C++ 中仍然创建作用域。在 `if constexpr` 块内声明的变量，在该块外不可见：

```cpp
__VEC_SCOPE__ {
    if constexpr (M == ActMode::TANH) {
        MaskReg cmpMaskReg;  // 仅此块内可见
    }
    for (...) {
        if constexpr (M == ActMode::TANH) {
            // cmpMaskReg 不可见！编译错误
        }
    }
}
```

因此，`cmpMaskReg` 和 `weightReg` 都**无法通过 `if constexpr` 隔离声明**，除非将整个 for 循环体包裹在 `if constexpr` 中（导致 SILU/TANH 路径代码完全重复，不可接受）。

#### 实际处理方式

与现有 `weightReg`（无条件声明但仅 `TopkWeightsPrefetch` 使用）保持一致的模式：

```cpp
__VEC_SCOPE__
{
    // --- 共享寄存器声明（所有激活模式共用）---
    AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1, vregX2;
    AscendC::MicroAPI::RegTensor<float> vregX1F, vregX2F;
    AscendC::MicroAPI::RegTensor<float> negReg, expReg, addsReg, sigmoidReg, outFReg;
    AscendC::MicroAPI::RegTensor<float> weightReg;           // 现有：无条件声明
    AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg, zeroReg;
    AscendC::MicroAPI::MaskReg cmpMaskReg;                   // 新增：无条件声明
    AscendC::MicroAPI::MaskReg mask = ...;
    // ...

    for (...) {
        // ...
        if constexpr (M == ActMode::SILU) {
            // SILU 指令序列（不使用 cmpMaskReg）
        } else {
            // TANH 指令序列（使用 cmpMaskReg）
        }
        if constexpr (TopkWeightsPrefetch) {
            Mul(outFReg, outFReg, weightReg, mask);  // 使用 weightReg
        }
    }
}
```

- 声明无条件，但**指令使用被 `if constexpr` 隔离**：SILU 路径不编译 `cmpMaskReg` 相关指令，TANH 路径不编译 `weightReg` 相关指令（若 prefetch 关闭）。
- MicroAPI 后端对未使用的 `RegTensor`/`MaskReg` 声明会优化掉，不占用物理寄存器（与现有 `weightReg` 在非 prefetch 模式下的行为一致）。

---

## 4. 代码实现

### 4.1 枚举定义

在 `block_epilogue_swiglu_mx_quant.h` 文件头部（`SwigluQuantMsg` namespace 之后）新增：

```cpp
enum class ActMode : uint8_t {
    SILU = 0,  // x / (1 + e^(-x)) * y  （现有默认）
    TANH = 1,  // tanh(x) * y           （新增）
};
```

### 4.2 新增常量

在 `VFDoSwigluAndQuantForMX` 函数体内，现有 `scalarOne` / `negScalarOne` 旁新增：

```cpp
const float scalarOne = 1.0f;
const float negScalarOne = -1.0f;
const float scalarTwo = 2.0f;       // 新增：tanh 形式 A 用
const float negScalarTwo = -2.0f;   // 新增：tanh 形式 B 用
const float scalarZero = 0.0f;      // 新增：tanh CompareScalar 用
```

### 4.3 epilogue 类运行时成员

#### 4.3.1 Arguments 结构体新增字段

```cpp
struct Arguments {
    // ... 现有字段 ...
    float clampLimit{0.0f};
    uint8_t actMode{0};   // 新增：0=swiglu(silu), 1=tanh
    Arguments() = default;
};
```

#### 4.3.2 类成员变量

```cpp
float clampLimit_{0.0f};
uint8_t actMode_{0};   // 新增：运行时激活模式
```

#### 4.3.3 Init 读取

```cpp
clampLimit_ = params.clampLimit;
actMode_ = params.actMode;   // 新增
```

### 4.4 函数模板参数设计（关键）

#### 4.4.1 `VFDoSwigluAndQuantForMX` 新增函数模板参数

现有签名：

```cpp
template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc>
__aicore__ inline void BlockEpilogueSwigluMxQuant<...>::VFDoSwigluAndQuantForMX(...)
```

改为（末尾追加 `ActMode M`，带默认值 `SILU`）：

```cpp
template <SwigluQuantMsg::QuantMode quantMode, bool IsInterleavedSrc, ActMode M = ActMode::SILU>
__aicore__ inline void BlockEpilogueSwigluMxQuant<...>::VFDoSwigluAndQuantForMX(...)
```

`M` 是函数模板参数，在 `__VEC_SCOPE__` 内是编译期常量，满足 `if constexpr` 约束。

**注意**：epilogue 类模板参数**不变**，`ActMode` 不是类模板参数，只是内部函数的模板参数。这是本方案与"类模板参数方案"的根本区别。

#### 4.4.2 `VFDoSwigluForMX` 运行时分发

现有 `VFDoSwigluForMX` 直接调用 `VFDoSwigluAndQuantForMX`：

```cpp
VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, false>(...);
```

改为运行时 `if/else` 按 `actMode_` 分发：

```cpp
if (actMode_ == static_cast<uint8_t>(ActMode::TANH)) {
    VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, false, ActMode::TANH>(...);
} else {
    VFDoSwigluAndQuantForMX<SwigluQuantMsg::QuantMode::MX_PERGROUP_MODE, false, ActMode::SILU>(...);
}
```

此处的 `if/else` 是**运行时分支**，但：
- 不在 `__VEC_SCOPE__` 内（`VFDoSwigluForMX` 本身不在 `__VEC_SCOPE__` 内）
- 每个 tile 执行一次，不在向量循环内
- `actMode_` 在整个 kernel 生命周期恒定，分支预测 ~100% 命中

`VFDoSwigluForMX` 内有 `IsInterleaved_` 的 `if constexpr` 分支（interleaved / 非 interleaved），每个分支内再加 `actMode_` 的运行时 `if/else`，共 4 条路径（2×2），但运行时只走 1 条。

### 4.5 主循环激活段改造

现有主循环激活段（使用 `mask`）：

```cpp
// === 现有 SILU ===
Muls(negReg, vregX1F, negScalarOne, mask);   // -x
Exp(expReg, negReg, mask);                    // e^(-x)
Adds(addsReg, expReg, scalarOne, mask);       // e^(-x)+1
Div(sigmoidReg, vregX1F, addsReg, mask);      // silu(x) = x/(e^(-x)+1)
Mul(outFReg, sigmoidReg, vregX2F, mask);      // silu(x)*y
```

改为 `if constexpr` 分支：

```cpp
if constexpr (M == ActMode::SILU) {
    // SILU: x / (e^(-x) + 1) * y
    Muls(negReg, vregX1F, negScalarOne, mask);   // -x
    Exp(expReg, negReg, mask);                    // e^(-x)
    Adds(addsReg, expReg, scalarOne, mask);       // e^(-x)+1
    Div(sigmoidReg, vregX1F, addsReg, mask);      // silu(x)
    Mul(outFReg, sigmoidReg, vregX2F, mask);      // silu(x)*y
} else {
    // TANH 双分支: tanh(x) * y
    // --- 分支 A: (e^(2x)-1)/(e^(2x)+1), 供 x<0 使用 ---
    Muls(negReg, vregX1F, scalarTwo, mask);        // 2x
    Exp(expReg, negReg, mask);                      // e^(2x)
    Adds(addsReg, expReg, scalarOne, mask);         // denA = e^(2x)+1
    Adds(sigmoidReg, expReg, negScalarOne, mask);   // numA = e^(2x)-1
    Div(expReg, sigmoidReg, addsReg, mask);          // resA = (e^(2x)-1)/(e^(2x)+1), 覆写 expReg

    // --- 分支 B: (1-e^(-2x))/(1+e^(-2x)), 供 x>=0 使用 ---
    Muls(negReg, vregX1F, negScalarTwo, mask);      // -2x
    Exp(negReg, negReg, mask);                       // e^(-2x), 覆写 negReg
    Adds(addsReg, negReg, scalarOne, mask);          // denB = 1+e^(-2x), 覆写 denA
    Adds(sigmoidReg, negReg, negScalarOne, mask);    // e^(-2x)-1, 覆写 numA
    Muls(sigmoidReg, sigmoidReg, negScalarOne, mask); // numB = 1-e^(-2x)
    Div(sigmoidReg, sigmoidReg, addsReg, mask);       // resB = (1-e^(-2x))/(1+e^(-2x))

    // --- 选择: x<0 → resA(expReg), x>=0 → resB(sigmoidReg) ---
    CompareScalar<float, CMPMODE::LT>(cmpMaskReg, vregX1F, scalarZero, mask);
    Select(sigmoidReg, expReg, sigmoidReg, cmpMaskReg); // tanh(x), 覆写 resB

    // --- 门控 ---
    Mul(outFReg, sigmoidReg, vregX2F, mask);          // tanh(x)*y
}
```

### 4.6 尾部循环激活段改造

现有尾部循环激活段（使用 `mask1`），改为 `if constexpr` 分支（与主循环结构完全相同，仅 mask 换成 `mask1`）：

```cpp
if constexpr (M == ActMode::SILU) {
    Muls(negReg, vregX1F, negScalarOne, mask1);
    Exp(expReg, negReg, mask1);
    Adds(addsReg, expReg, scalarOne, mask1);
    Div(sigmoidReg, vregX1F, addsReg, mask1);
    Mul(outFReg, sigmoidReg, vregX2F, mask1);
} else {
    Muls(negReg, vregX1F, scalarTwo, mask1);
    Exp(expReg, negReg, mask1);
    Adds(addsReg, expReg, scalarOne, mask1);
    Adds(sigmoidReg, expReg, negScalarOne, mask1);
    Div(expReg, sigmoidReg, addsReg, mask1);

    Muls(negReg, vregX1F, negScalarTwo, mask1);
    Exp(negReg, negReg, mask1);
    Adds(addsReg, negReg, scalarOne, mask1);
    Adds(sigmoidReg, negReg, negScalarOne, mask1);
    Muls(sigmoidReg, sigmoidReg, negScalarOne, mask1);
    Div(sigmoidReg, sigmoidReg, addsReg, mask1);

    CompareScalar<float, CMPMODE::LT>(cmpMaskReg, vregX1F, scalarZero, mask1);
    Select(sigmoidReg, expReg, sigmoidReg, cmpMaskReg);
    Mul(outFReg, sigmoidReg, vregX2F, mask1);
}
```

### 4.7 指令序列完整对照

| 步骤 | SILU（5 条） | TANH 双分支（14 条） |
|------|------------|-------------------|
| 1 | `Muls(negReg, x, -1)` → -x | `Muls(negReg, x, 2)` → 2x |
| 2 | `Exp(expReg, negReg)` → e^(-x) | `Exp(expReg, negReg)` → e^(2x) |
| 3 | `Adds(addsReg, expReg, 1)` → e^(-x)+1 | `Adds(addsReg, expReg, 1)` → denA |
| 4 | `Div(sigmoidReg, x, addsReg)` → silu(x) | `Adds(sigmoidReg, expReg, -1)` → numA |
| 5 | `Mul(outFReg, sigmoidReg, y)` → silu*y | `Div(expReg, sigmoidReg, addsReg)` → resA |
| 6 | | `Muls(negReg, x, -2)` → -2x |
| 7 | | `Exp(negReg, negReg)` → e^(-2x) |
| 8 | | `Adds(addsReg, negReg, 1)` → denB |
| 9 | | `Adds(sigmoidReg, negReg, -1)` → e^(-2x)-1 |
| 10 | | `Muls(sigmoidReg, sigmoidReg, -1)` → numB |
| 11 | | `Div(sigmoidReg, sigmoidReg, addsReg)` → resB |
| 12 | | `CompareScalar<LT>(cmpMaskReg, x, 0)` → x<0? |
| 13 | | `Select(sigmoidReg, expReg, sigmoidReg, cmpMaskReg)` → tanh(x) |
| 14 | | `Mul(outFReg, sigmoidReg, y)` → tanh(x)*y |

TANH 比 SILU 多 9 条指令，但全部是向量原语，可与前后指令流水重叠，实际延迟增量远小于 9 周期。

### 4.8 clamp 处理

现有 clamp 代码对两个分支都保留：

```cpp
Mins(vregX1F, vregX1F, clampLimit_, mask);
Mins(vregX2F, vregX2F, clampLimit_, mask);
Maxs(vregX2F, vregX2F, -clampLimit_, mask);
```

- **SILU 路径**：clamp 行为不变（虽然默认 `FLT_MAX` 等于无 clamp，但不影响正确性）。
- **TANH 双分支路径**：双分支自身已数值稳定，clamp 是多余的但无害——`Mins(x, FLT_MAX)` 是 no-op，`Mins(x, 10)` 只会限制输入范围不影响 tanh 正确性。保留 clamp 代码可避免在 `if constexpr` 外再加分支，简化实现。

---

## 5. 接口透传

### 5.1 数据流

```
[host] activation 属性 ("swiglu"/"tanh")
    │
    ▼ tiling.cpp: 读取 + 映射
tilingData.actMode (uint8_t)
    │
    ▼ MegaMoe Init: epilogueOp_.Init({..., tilingData->actMode})
epilogue.actMode_ (uint8_t 成员)
    │
    ▼ VFDoSwigluForMX: 运行时 if/else 分发
VFDoSwigluAndQuantForMX<..., ActMode::SILU/TANH> (函数模板参数 M)
    │
    ▼ __VEC_SCOPE__ 内: if constexpr (M == ...)
SILU 或 TANH 指令序列
```

### 5.2 tilingData 字段（`mega_moe_tiling.h`）

在 `MegaMoeTilingData` 结构体末尾新增字段：

```cpp
struct MegaMoeTilingData {
    // ... 现有字段 ...
    int32_t topkWeightsPrefetch;
    uint32_t maxTilesPerExpert;
    uint8_t actMode;            // 新增：0=swiglu(silu), 1=tanh
};
```

### 5.3 host tiling（`mega_moe_tiling.cpp`）

#### 5.3.1 读取 activation 属性

`activation` 属性已声明（`mega_moe_def.cpp:170`，默认 `"swiglu"`），索引为 `attrActivationIndex = 9U`（`op_tiling/mega_moe.h:57`），但改造前 tiling 未消费。

改造内容（3 处）：

**a) 匿名 namespace 新增常量：**

```cpp
const static uint8_t ACT_MODE_SWIGLU = 0U;
const static uint8_t ACT_MODE_TANH = 1U;
```

**b) `CheckAttrPtrNullptr` 函数新增读取 + 校验：**

```cpp
auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));

OP_TILING_CHECK(activationPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "activation"),
                return ge::GRAPH_FAILED);
OP_TILING_CHECK(std::strcmp(activationPtr, "swiglu") != 0 && std::strcmp(activationPtr, "tanh") != 0,
                OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr,
                                          "should be 'swiglu' or 'tanh'"),
                return ge::GRAPH_FAILED);
```

使用 `GetAttrPointer<char>` + `std::strcmp`，与现有 `commAlg` 属性（同为 String 类型）的读取方式一致。

**c) `SetAttrParams` 函数新增映射写入：**

```cpp
auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
if (std::strcmp(activationPtr, "tanh") == 0) {
    tilingData->actMode = ACT_MODE_TANH;
} else {
    tilingData->actMode = ACT_MODE_SWIGLU;
}
```

**d) `PrintMegaMoeTilingData` 新增打印：**

```cpp
OP_LOGD(nodeName, "actMode is %u", tilingData->actMode);
```

#### 5.3.2 activation 属性默认值

`mega_moe_def.cpp:170` 已定义默认值 `"swiglu"`，不传 `activation` 属性时自动走 SILU 路径，向后兼容。

### 5.4 epilogue Init 调用点（`mega_moe.h` + `mega_moe_layered.h`）

共 4 处 Init 调用需追加 `actMode` 实参（与 `clampLimit` 并列传入 `Arguments`）：

| 文件 | 行号 | 调用对象 |
|------|------|---------|
| `mega_moe.h` | 321 | `epilogueOp_.Init(...)` |
| `mega_moe.h` | 1985 | `sharedEpilogueOp_.Init(...)` |
| `mega_moe_layered.h` | 322 | `epilogueOp_.Init(...)` |
| `mega_moe_layered.h` | 2025 | `sharedEpilogueOp_.Init(...)` |

改造示例（`mega_moe.h:321`）：

```cpp
// 改造前
epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
                  params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr,
                  params_.workspaceInfo.metaInfoPtr, tilingData->clampLimit});

// 改造后（末尾追加 tilingData->actMode）
epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
                  params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr,
                  params_.workspaceInfo.metaInfoPtr, tilingData->clampLimit, tilingData->actMode});
```

其余 3 处同理，追加 `tilingData->actMode` 或 `params_.tilingData->actMode`。

### 5.5 不改动的文件

| 文件 | 不改原因 |
|------|---------|
| `mega_moe_apt.cpp` | epilogue 类类型不依赖 `ActMode`，`MegaMoe` 模板参数不变，无需 if/else 实例化 |
| `mega_moe.h`（模板宏） | `TemplateMegaMoeTypeClass` / `TemplateMegaMoeTypeFunc` 不变 |
| `mega_moe.h`（BlockEpilogue 类型别名） | `BlockEpilogueSwigluMxQuant<...>` 参数列表不变 |
| `mega_moe_impl.h` | `GroupMatmulSwigluQuant` / `GroupMatmulSwigluQuantA8W4` 模板参数不变 |
| `mega_moe_tiling_key.h` | 不使用 TilingKey |
| `mega_moe_def.cpp` | `activation` 属性已声明，默认值 `"swiglu"` 无需修改 |
| `op_tiling/mega_moe.h` | `attrActivationIndex = 9U` 已就绪 |

---

## 6. 改动文件清单

| # | 文件 | 改动内容 | 改动量 |
|---|------|---------|--------|
| 1 | `block_epilogue_swiglu_mx_quant.h` | 新增 `ActMode` 枚举；`Arguments` 加 `actMode` 字段；`actMode_` 成员；`Init` 读取；`VFDoSwigluAndQuantForMX` 加函数模板参数 `ActMode M`（默认 SILU）；常量 `scalarTwo/negScalarTwo/scalarZero`；`cmpMaskReg` 声明；主循环 + 尾部循环 `if constexpr` 双分支；`VFDoSwigluForMX` 运行时 `if/else` 分发 | 中（核心） |
| 2 | `mega_moe_tiling.h` | `MegaMoeTilingData` 新增 `uint8_t actMode` 字段 | 小 |
| 3 | `mega_moe_tiling.cpp` | 匿名 namespace 加常量；`CheckAttrPtrNullptr` 读取 + 校验 `activation`；`SetAttrParams` 映射写入 `actMode`；打印日志 | 小 |
| 4 | `mega_moe.h` | 2 处 Init 调用追加 `tilingData->actMode`（`epilogueOp_` + `sharedEpilogueOp_`） | 小 |
| 5 | `mega_moe_layered.h` | 2 处 Init 调用追加 `tilingData->actMode`（同上） | 小 |

**不改动的文件**：`mega_moe_apt.cpp`、`mega_moe_impl.h`、`mega_moe_tiling_key.h`、`mega_moe_def.cpp`、`op_tiling/mega_moe.h`、`mega_moe.h`（模板宏和类型别名）、`mega_moe_layered.h`（模板宏和类型别名）。

---

## 7. 验证要点

### 7.1 精度验证

| 测试场景 | 输入 | 期望输出 | 验证点 |
|---------|------|---------|--------|
| tanh(0) * y | x=0, y=任意 | 0 | 零点正确性 |
| tanh(大正) * y | x=100, y=1 | ≈1 | 正向饱和，无 NaN |
| tanh(大负) * y | x=-100, y=1 | ≈-1 | 负向饱和，无 NaN |
| tanh(1) * y | x=1, y=1 | ≈0.7616 | 常规值精度 |
| tanh(-1) * y | x=-1, y=1 | ≈-0.7616 | 对称性 |
| SILU 回归 | x=任意, y=1 | silu(x) | 默认模式不回归 |
| 量化输出 | tanh(x)*y → BF16 → MX FP8/FP4 | 量化后数据 + scale | 量化流水不受激活切换影响 |

### 7.2 编译验证

- SILU 路径（`actMode=0`）：运行时 `if/else` 走 `else` 分支，调用 `VFDoSwigluAndQuantForMX<..., SILU>`，`if constexpr` 编译 SILU 指令序列
- TANH 路径（`actMode=1`）：运行时 `if/else` 走 `if` 分支，调用 `VFDoSwigluAndQuantForMX<..., TANH>`，`if constexpr` 编译 TANH 指令序列
- `__VEC_SCOPE__` 内无运行时 `if` 分支（`if constexpr` 条件 `M` 为函数模板参数，编译期常量）
- 两份 `VFDoSwigluAndQuantForMX` 模板实例（SILU/TANH）在同一编译单元内，1 份二进制

### 7.3 性能验证

- TANH 双分支比 SILU 多 9 条向量指令，但可与前后指令流水重叠
- `cmpMaskReg` 声明不影响 SILU 路径（`if constexpr` 消除使用指令，后端优化掉未用声明）
- 运行时 `if/else` 在 `VFDoSwigluForMX` 内，每个 tile 执行一次，不在向量循环内，`actMode_` 恒定，分支预测 ~100% 命中
- 量化段（`ComputeMaxExp` / `ComputeScale` / `ComputeDataForQuant*`）完全不受影响

### 7.4 接口验证

- 不传 `activation` 属性 → 默认 `"swiglu"` → `actMode=0` → SILU 路径（向后兼容）
- 传 `activation="tanh"` → `actMode=1` → TANH 路径
- 传 `activation="invalid"` → tiling 报错（校验拦截）

---

## 8. 注意事项

### 8.1 clamp 与 TANH 的关系

- TANH 双分支**完全不依赖 clamp**：两个分支的指数恒 ≤ 0，`e^(非正数)` 永远有界
- clamp 默认值为 `FLT_MAX`（`mega_moe_def.cpp:171`），对 TANH 是 no-op
- 若用户传了小 clamp（如 10），对 TANH 也无害（仅限制输入范围，不影响正确性）
- **结论**：clamp 代码保留不动，SILU 和 TANH 共用，无需按激活模式分支

### 8.2 `Select` 写入源操作数

`Select(sigmoidReg, expReg, sigmoidReg, cmpMaskReg)` 中 `sigmoidReg` 既是 dst 又是 src2。这在 MicroAPI 中合法——`mish_dag.h:99` 的 `Select(vregOutput, vregInputNumerator, vregOutput, cmpMaskReg)` 使用了相同模式。`Select` 语义为 `dst = mask ? src1 : src2`，写入 `src2` 本身等价于 `dst = mask ? src1 : dst`。

### 8.3 `cmpMaskReg` 声明位置

`cmpMaskReg` 必须在 for 循环之前声明（与 `weightReg` 同级），不能在 `if constexpr` 块内声明——因为 C++ 作用域规则限制，`if constexpr` 块内声明的变量在块外不可见。虽然声明是无条件的，但 `if constexpr` 保证了只有 TANH 路径才编译使用 `cmpMaskReg` 的指令，SILU 路径不会产生任何相关指令。这与现有 `weightReg` 的处理方式完全一致。

### 8.4 函数模板参数 vs 类模板参数

本方案的关键设计决策：`ActMode` 是 `VFDoSwigluAndQuantForMX` 的**函数模板参数**，而非 `BlockEpilogueSwigluMxQuant` 的**类模板参数**。

| 维度 | 类模板参数（未采用） | 函数模板参数（本方案） |
|------|-------------------|---------------------|
| epilogue 类型是否依赖 ActMode | 是 → `MegaMoe` 成员 `epilogueOp_` 类型依赖 → apt.cpp/impl 全链路透传 | 否 → epilogue 类型不变 |
| `if constexpr` 条件来源 | `ActMode_`（类模板参数） | `M`（函数模板参数） |
| 改动文件数 | 6-7 | 5 |
| apt.cpp / MegaMoe 类 / impl | 需改 | 不改 |

选择函数模板参数的根本原因：避免 `ActMode` 污染 epilogue 类型，从而避免从 apt.cpp 到 impl 的全链路模板参数透传。

### 8.5 运行时 `if/else` 的位置

运行时 `if/else` 分发位于 `VFDoSwigluForMX` 内，**不在 `__VEC_SCOPE__` 内**。`VFDoSwigluForMX` 的职责是准备 UB 地址并调用 `VFDoSwigluAndQuantForMX`，本身不在向量计算作用域内。运行时分支在此处执行：
- 每个 tile 执行一次（不在 dim0/dim1 循环内）
- `actMode_` 在整个 kernel 生命周期恒定
- 分支预测器命中率 ~100%，开销可忽略

### 8.6 interleaved 路径

`VFDoSwigluForMX` 内有 `if constexpr (IsInterleaved_)` 的编译期分支（interleaved / 非 interleaved），每个分支内再加 `actMode_` 的运行时 `if/else`。编译后共 4 条路径（2×2），但运行时只走 1 条。interleaved 路径的 TANH 分支与非 interleaved 完全一致，只是 UB 地址布局不同。

### 8.7 未来扩展

`ActMode` 使用 `enum class` 而非 `bool`，便于未来扩展其他激活函数（如 GELU、ReLU 等）。新增激活只需：

1. 在 `ActMode` 枚举中追加值
2. 在 `if constexpr` 链中追加 `else if constexpr` 分支
3. 在 `VFDoSwigluForMX` 的 `if/else` 中追加对应分发
4. 在 host tiling 中追加字符串映射

### 8.8 参考实现

| 参考文件 | 参考内容 |
|---------|---------|
| `ops-nn/activation/mish/op_kernel/arch35/mish_dag.h:81-100` | 双分支 + `CompareScalar<LT>` + `Select` 模式 |
| `ops-nn/activation/gelu_quant/op_kernel/arch35/gelu_quant_base_regbase.h:28-29` | 常量预乘优化思路（tanh 暂未用，但 `scalarTwo` 设计可借鉴） |
| `block_epilogue_swiglu_mx_quant.h`（现有 `TopkWeightsPrefetch`） | `__VEC_SCOPE__` 内 `if constexpr` 先例；`weightReg` 无条件声明 + `if constexpr` 隔离使用的现有模式 |
