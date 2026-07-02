# Flash Attention Score: Key 含 -inf 时将输出置零的修改方案

## 1. 背景与问题

当 Flash Attention 的输入 **Key 含有 `-inf`** 时（例如 KV-Cache 中未填充的位置被填为 `-inf`），整行 attention score 的计算链会产出 **NaN**，最终输出非 0 也非有效值，导致训练发散或推理错误。

### 1.1 NaN 传播路径

```
bmm1 (Q @ K^T):    Q × (-inf) = -inf  (或 NaN，若 Q 中有 0)
softmax:            max = -inf
                    exp(x - max) = exp(-inf - (-inf)) = exp(NaN) = NaN   ← P 变成 NaN
                    sum = NaN
bmm2 (P @ V):       NaN × V = NaN                                      ← P@V 变成 NaN
LastDiv:            NaN / sum = NaN                                    ← 最终输出为 NaN
```

核心矛盾在于 `exp(-inf - (-inf)) = exp(NaN) = NaN`，NaN 一旦产生就会污染后续所有计算，仅修复分母（sum）无法消除分子（P@V）中的 NaN。

### 1.2 期望行为

当某行的 rowmax 为 `-inf`（即整行被 mask / Key 全为 `-inf`）时，**该行 attention 输出应为 0**。

## 2. 现有机制分析

代码中已存在一套「无效行检测 + 置零」机制，但受限于三个条件无法在当前场景生效：

### 2.1 现有机制的三层守卫

| 层级 | 位置 | 条件 | 问题 |
|------|------|------|------|
| **编译期 (constexpr)** | 各调用点 | `implMode == AA_INVALID_LINE_HIGH_PRECISION` | bf16/fp16 默认 `implMode=AA_HIGH_PRECISION`，constexpr 块在编译期被完全消除 |
| **运行时 (runtime)** | 各调用点 | `tilingData->implMode == AA_INVALID_LINE_HIGH_PRECISION` | host 侧 tiling 未设置该值，运行时检查失败 |
| **比较模式 (EQ)** | SIMD 比较 | `max == -FLT_MAX (0xFF7FFFFF)` | 只匹配 `-FLT_MAX`，不匹配真正的 `-inf (0xFF800000)` |

### 2.2 现有机制的两个置零函数

1. **`SoftmaxSumUpdate`** (`vf_mul_sel_softmaxflashv2_cast_nz.h:586`)
   - 作用：检测 `max == -FLT_MAX` 的行，将其 `sum` 置为 `FLT_MAX`
   - 局限：**只修分母**，分子 P@V 已是 NaN，`NaN / FLT_MAX = NaN`

2. **`InvalidLineUpdate`** (`vf_flashupdate_new.h:454`) —— **关键函数**
   - 作用：在最终输出阶段（`Bmm2DataCopyOut`），检测 `max == -FLT_MAX` 的行，直接将整行输出替换为 `0`
   - 这是唯一能覆盖 NaN 的兜底机制
   - 局限：同样被三层守卫挡住

### 2.3 调用链路

```
Process()
  └─ softMaxCheckRes = true                          ← 守卫1+2 挡住
  └─ ProcessVec1 → ProcessVec1Dn/ProcessVec1Nd
       ├─ InvalidLineProcess                         ← 守卫1+2 挡住
       │    ├─ SoftmaxInvalidLineCheck               ← EQ 只匹配 -FLT_MAX
       │    └─ SoftmaxSumUpdate (修分母)             ← EQ 只匹配 -FLT_MAX
       └─ (P@V = NaN，分母修复无效)
  └─ ProcessVec2 → Bmm2DataCopyOut
       └─ InvalidLineUpdate (修分子=置零)            ← 守卫1+2 挡住，EQ 只匹配 -FLT_MAX
```

## 3. 修改方案

### 3.1 修改原则

- **始终自动启用**：去掉所有 `implMode` 相关的 constexpr + runtime 守卫，无条件执行检测
- **同时覆盖 -FLT_MAX 和 -inf**：用 `LE`（≤）替代 `EQ`（==），使所有 ≤ -FLT_MAX 的值都被识别为无效行
- **标量检测兼容 -inf**：位模式比较增加 `0xFF800000` 分支

### 3.2 修改总览

共 **5 个文件，10 处改动**，净减 27 行代码：

| # | 文件 | 函数/位置 | 改动类型 | 说明 |
|---|------|-----------|----------|------|
| 1 | `flash_attention_score_common_regbase.h:37` | 常量区 | 新增 | `NEGATIVE_INF_VALUE_FP32 = 0xFF800000` |
| 2 | `flash_attention_score_kernel_train.h:288` | `Process()` | 去守卫 | 无条件 `softMaxCheckRes = true` |
| 3 | `flash_attention_noquant_block_vec_base.h:598` | `ProcessVec1Dn` | 去守卫 | 无条件调用 `InvalidLineProcess` |
| 4 | `flash_attention_noquant_block_vec_base.h:1240` | `ProcessVec1Nd` | 去守卫 | 无条件调用 `InvalidLineProcess` |
| 5 | `flash_attention_noquant_block_vec_base.h:621` | `SoftmaxInvalidLineCheck` | 扩展检测 | 增加 `\|\| == NEGATIVE_INF_VALUE_FP32` |
| 6 | `flash_attention_noquant_block_vec_base.h:2218` | `GetExtremeValue` | 去守卫 | 无条件初始化 `positiveFloatScalar` |
| 7 | `flash_attention_noquant_block_vec_base.h:1918` | `Bmm2DataCopyOut` | 去守卫 | 无条件调用 `InvalidLineUpdate` |
| 8 | `vf_mul_sel_softmaxflashv2_cast_nz.h:601` | `SoftmaxSumUpdateVF` | EQ→LE | SIMD 分母修复覆盖 ≤-FLT_MAX |
| 9 | `vf_flashupdate_new.h:469` | `InvalidLineUpdateVF` | EQ→LE | SIMD 输出置零覆盖 ≤-FLT_MAX |
| 10 | `flash_attention_noquant_block_vec_base.h:623` | `SoftmaxInvalidLineCheck` | 修复UB | `reinterpret_cast` → `__builtin_memcpy` |

### 3.3 各改动详细说明

#### 改动 1: 新增 -inf 常量

**文件**: `attention/common/op_kernel/arch35/flash_attention_score_common_regbase.h`

```cpp
// 新增
constexpr uint32_t NEGATIVE_INF_VALUE_FP32 = 0xFF800000;  // -inf 的 IEEE 754 比特模式
```

供标量位模式比较使用（改动 5）。

#### 改动 2: 无条件开启检测开关

**文件**: `attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h`

```cpp
// 修改前
if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION ||
            IsSameType<INPUT_T, float>::value) {
    if (this->sharedParams.implMode ==
        static_cast<uint8_t>(ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION)) {
        this->constInfo.softMaxCheckRes = true;
    }
}

// 修改后
this->constInfo.softMaxCheckRes = true;
```

去掉 constexpr 编译期守卫和 runtime tiling 守卫，所有数据类型（bf16/fp16/fp32）均在 AIV 核上开启 `softMaxCheckRes` 检测标志。

#### 改动 3 & 4: 无条件调用 InvalidLineProcess

**文件**: `attention/common/op_kernel/arch35/flash_attention_noquant_block_vec_base.h`

`ProcessVec1Dn`（line 598）和 `ProcessVec1Nd`（line 1240）两处相同模式：

```cpp
// 修改前
if constexpr (implMode == ImplModeEnum::AA_INVALID_LINE_HIGH_PRECISION || ...) {
    if (this->tilingData->inputParamsRegbase.implMode == ...) {
        this->InvalidLineProcess(runInfo, constInfo, sumUb, maxUb);
    }
}

// 修改后
this->InvalidLineProcess(runInfo, constInfo, sumUb, maxUb);
```

DN（对齐）和 ND（非对齐）两条 VEC1 计算路径均在 softmax 更新后无条件执行无效行检测。

#### 改动 5: 标量检测兼容 -inf

**文件**: `attention/common/op_kernel/arch35/flash_attention_noquant_block_vec_base.h`

`SoftmaxInvalidLineCheck` 函数（line 621）：

```cpp
// 修改前
if (checkValue == negativeIntScalar) {

// 修改后
if (checkValue == negativeIntScalar || checkValue == NEGATIVE_INF_VALUE_FP32) {
```

标量逐行扫描 `maxUb` 时，同时匹配 `-FLT_MAX (0xFF7FFFFF)` 和 `-inf (0xFF800000)` 两种比特模式。

#### 改动 6: 无条件初始化 positiveFloatScalar

**文件**: `attention/common/op_kernel/arch35/flash_attention_noquant_block_vec_base.h`

`GetExtremeValue` 函数（line 2218）：

```cpp
// 修改前
negativeScalar = *((float *)&tmp1);
if constexpr (implMode == ... || IsSameType<INPUT_T, float>::value) {
    if (this->tilingData->...implMode == ...) {
        positiveScalar = *((float *)&tmp2);  // FLT_MAX
    }
}

// 修改后
negativeScalar = *((float *)&tmp1);
positiveScalar = *((float *)&tmp2);  // 无条件初始化
```

确保 `SoftmaxSumUpdate` 有正确的 `maxValue`（FLT_MAX）用于将无效行的 sum 置为大值。

#### 改动 7: 无条件调用 InvalidLineUpdate（关键）

**文件**: `attention/common/op_kernel/arch35/flash_attention_noquant_block_vec_base.h`

`Bmm2DataCopyOut` 函数（line 1918）：

```cpp
// 修改前
if constexpr (implMode == ... || IsSameType<INPUT_T, float>::value) {
    if (this->tilingData->...implMode == ...) {
        InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, ...);
    }
}

// 修改后
int64_t vec2MaxBufOffset = ComputeOffsetForSoftmax(runInfo, vec2S1Idx);
LocalTensor<float> maxTensor = softmaxMaxBuf[...].template Get<float>()[vec2MaxBufOffset];
InvalidLineUpdate<T, dTemplateAlign64>(vec2ResUb, vec2ResUb, maxTensor, runInfo.vec2S1RealSize,
    dSizeAligned64, this->negativeFloatScalar, 0.0);
```

**这是最关键的改动。** `InvalidLineUpdate` 在最终输出阶段，将 `max ≤ -FLT_MAX` 的行直接替换为 0，是唯一能消除 NaN 的兜底机制。

#### 改动 8 & 9: SIMD 比较 EQ → LE

**文件 8**: `vf/vf_mul_sel_softmaxflashv2_cast_nz.h` - `SoftmaxSumUpdateVF`（line 601）

**文件 9**: `vf/vf_flashupdate_new.h` - `InvalidLineUpdateVF`（line 469）

```cpp
// 修改前
Compares<T, CMPMODE::EQ>(preg_compare, vreg_max, minValue, preg_all);

// 修改后
Compares<T, CMPMODE::LE>(preg_compare, vreg_max, minValue, preg_all);
```

用 `≤`（LE）替代 `==`（EQ），使所有 `max ≤ -FLT_MAX` 的值（包括 `-FLT_MAX` 和 `-inf`）都被识别为无效行。`-inf < -FLT_MAX`，所以 `LE` 天然涵盖 `EQ`，向后兼容。

#### 改动 10: 修复 `SoftmaxInvalidLineCheck` 中的 reinterpret_cast UB

**文件**: `attention/common/op_kernel/arch35/flash_attention_noquant_block_vec_base.h`

`SoftmaxInvalidLineCheck` 函数（line 623）：

```cpp
// 修改前
T maxValue = maxUb.GetValue(i);
uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);

// 修改后
T maxValue = maxUb.GetValue(i);
uint32_t checkValue = 0;
__builtin_memcpy(&checkValue, &maxValue, sizeof(T));
```

**问题分析**：

原代码通过 `reinterpret_cast<uint32_t*>(&maxValue)` 将类型 `T` 的局部变量按 4 字节读取。当 `sizeof(T) < 4` 时（如 `T=half`，`sizeof=2`），会读取栈上相邻的未初始化内存，属于未定义行为（UB）。

**当前影响评估**：

从入口文件 `flash_attention_score_entry_regbase.h` 确认，所有实例化路径中模板参数 `T` 恒为 `float`（4 字节）：

| dtype | INPUT_T | T | OUTPUT_T |
|-------|---------|---|----------|
| fp32 | float | float | float |
| bf16 | bfloat16_t | float | bfloat16_t |
| fp16 | half | float | half |
| hifloat8 | hifloat8_t | float | bfloat16_t |

因此 `T=half` 的 UB 路径在当前代码中**实际不可达**。但该写法是脆弱的——若未来新增 `T=half` 的入口路径，将触发真正的 UB。

**修复方式**：

用 `__builtin_memcpy` 替代 `reinterpret_cast`，这是 C++ 标准中处理类型双关（type punning）的安全方式：
- 只拷贝 `sizeof(T)` 字节到 `uint32_t` 的低位，高位保持 0 初始化
- 对任何 `sizeof(T) <= 4` 的类型都安全
- 编译器会将 `__builtin_memcpy` 优化为等价的寄存器操作，零运行时开销

## 4. 修复后的完整计算链路

```
Key = -inf
  │
  ▼
bmm1: Q @ K^T → -inf
  │
  ▼
softmax flash V2:
  ├─ max = -inf
  ├─ exp(x - max) = exp(NaN) = NaN          ← P = NaN, sum = NaN
  └─ InvalidLineProcess (无条件执行)
       ├─ SoftmaxInvalidLineCheck:
       │    max == -FLT_MAX? YES 或 max == -inf? YES → 发现无效行
       ├─ softMaxCheckRes 保持 true
       └─ SoftmaxSumUpdate: max ≤ -FLT_MAX → sum = FLT_MAX  (修分母，但分子仍是 NaN)
  │
  ▼
bmm2: P @ V = NaN × V = NaN                  ← 分子仍是 NaN
  │
  ▼
Bmm2DataCopyOut → InvalidLineUpdate (无条件执行)
  ├─ max ≤ -FLT_MAX? YES
  └─ output = 0                               ← 最终兜底，NaN 被替换为 0 ✓
  │
  ▼
最终输出: 0
```

## 5. `AA_INVALID_LINE_HIGH_PRECISION` 原始触发条件

修改前，`implMode == AA_INVALID_LINE_HIGH_PRECISION` 通过以下三个来源设置，且受一个关键约束限制。

### 5.1 来源 1: 用户 API 显式传入

**文件**: `op_host/arch35/flash_attention_score_tiling_regbase.cpp:173-178`

```cpp
implMode = ImplMode::AA_HIGH_PRECISION;  // 默认值
if (attrs->GetAttrNum() > idx) {
    auto implModePtr = attrs->GetAttrPointer<uint8_t>(idx++);
    if (static_cast<ImplMode>(*implModePtr) == ImplMode::AA_INVALID_LINE_HIGH_PRECISION) {
        implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
    }
}
```

用户在调用 API 时通过 `impl_mode` 属性显式传 `2`（`AA_INVALID_LINE_HIGH_PRECISION`）。

### 5.2 来源 2: BAND sparse 模式自动启用

**文件**: `op_host/arch35/flash_attention_score_tiling_regbase.cpp:1347-1357` — `EnableBandInvalidLineImplMode()`

```cpp
if (preTokens < (s1Size - s2Size) || nextTokens < 0) {
    implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
}
```

当 **BAND sparse 模式** 且 `s1 > s2`（如 S1=1024, S2=512），某些 S1 行的 band 窗口完全落在 S2 范围之外，即**整行无有效 KV** 时自动启用。在以下 4 处调用：

| 调用位置 | 触发条件 |
|----------|----------|
| `flash_attention_score_tiling_regbase.cpp:1413` | `RIGHT_DOWN_CAUSAL` 且 `s1 != s2`，降级为 BAND |
| `flash_attention_score_tiling_regbase.cpp:1451` | `BAND` 且 `s1 != s2` |
| `flash_attention_score_tiling_regbase.cpp:1466` | `BAND` 且 `preTokens < 0` |
| `flash_attention_score_tiling_regbase.cpp:1484` | `BAND` 且 `nextTokens < 0` |

### 5.3 来源 3: PREFIX sparse 模式自动启用

**文件**: `op_host/arch35/flash_attention_score_tiling_regbase.cpp:1876-1878`

```cpp
if (s1Size > s2Size && prefixNData[i] == 0) {
    implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
}
```

当 **PREFIX 模式** 且 `s1 > s2` 且某 batch 的 `prefixN == 0` 时，该 batch 的上半部分 S1 行完全没有有效 KV。

varlen 路径（`op_host/arch35/flash_attention_score_tiling_varlen.cpp:528`）同理：`actualSeqLenQ > actualSeqLenKv` 且 `prefixN == 0`。

### 5.4 关键约束

**文件**: `op_host/arch35/flash_attention_score_tiling_regbase.cpp:1173`

```cpp
implMode = (hasAttenMask && inputDtypeBytes != DATA_TYPE_FP32) ? implMode : ImplMode::AA_HIGH_PRECISION;
```

即使上述三个来源的条件满足，**没有 attenMask 或 dtype 是 fp32 时，强制降级回 `AA_HIGH_PRECISION`**。这意味着原始机制**仅对「有 attenMask 的 bf16/fp16」生效**。

### 5.5 原始触发条件汇总

```
implMode = AA_INVALID_LINE_HIGH_PRECISION 当且仅当：
  (用户显式传入 implMode=2
   OR BAND模式且整行无有效KV
   OR PREFIX模式且s1>s2且prefixN=0)
  AND hasAttenMask == true
  AND dtype != fp32
```

## 6. 修改对原算子的影响评估

### 6.1 无效行检测现在对所有场景无条件执行

| 场景 | 修改前 | 修改后 | 影响 |
|------|--------|--------|------|
| BAND/PREFIX 自动启用（有 attenMask，bf16/fp16） | 执行检测 | 执行检测 | **无变化** |
| 用户显式传 implMode=2 | 执行检测 | 执行检测 | **无变化** |
| 无 attenMask 的普通场景 | **不执行** | 执行检测 | 见 6.2 分析 |
| fp32 场景 | **不执行** | 执行检测 | 见 6.3 分析 |

### 6.2 对「无 attenMask 的普通场景」的影响

修改后 `SoftmaxInvalidLineCheck` 会在每个 task 首个 s2 loop 执行一次逐行扫描：

```cpp
for (uint32_t i = 0; i < softmaxShapeInfo.srcM; i++) {
    T maxValue = maxUb.GetValue(i);
    uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
    if (checkValue == negativeIntScalar || checkValue == NEGATIVE_INF_VALUE_FP32) {
        isUpdateNeedCheck = true;
        break;
    }
}
```

- **正常输入（无 -inf/-FLT_MAX）**：首次扫描未命中 → `softMaxCheckRes = false` → 后续所有 loop 跳过。仅增加一次 m 次 scalar `GetValue` 的开销，可忽略。
- **`InvalidLineUpdate` 同理**：在 `Bmm2DataCopyOut` 中无条件执行，但 `LE` 比较对正常 max（远大于 -FLT_MAX）不会命中，Select 不改变原值。仅增加一次 SIMD LE+Select 的开销。

### 6.3 对 fp32 场景的影响

原始代码 fp32 走的是 `else` 分支（`IsSameType<INPUT_T, VEC2_RES_T>::value` 为 true），不经过 `InvalidLineUpdate`。修改后：

- `Bmm2DataCopyOut` 中的 `InvalidLineUpdate` 仍在 `!IsSameType<INPUT_T, VEC2_RES_T>::value` 分支内（line 1919），**fp32 仍然不进入此分支**，不受影响。
- 但 `ProcessVec1Dn`/`ProcessVec1Nd` 中的 `InvalidLineProcess` 现在**无条件执行**，fp32 也会走检测。由于 fp32 原本通过 `IsSameType<INPUT_T, float>::value` 进入检测，行为一致。

### 6.4 EQ → LE 的兼容性

`-inf (-∞) < -FLT_MAX`，所以 `LE`（≤）天然涵盖 `EQ`（==）。原本匹配 `-FLT_MAX` 的行继续匹配，新增匹配 `-inf` 的行。**完全向后兼容**。

### 6.5 性能影响总结

| 路径 | 新增开销 | 频率 |
|------|----------|------|
| `SoftmaxInvalidLineCheck` | m 次 scalar `GetValue` + 比较 | 每 task 首个 s2 loop 一次 |
| `SoftmaxSumUpdate`（仅无效行场景） | 1 次 SIMD LE + Select | 仅发现无效行后的每个 loop |
| `InvalidLineUpdate` | 1 次 SIMD LE + Select per row | 每个 `Bmm2DataCopyOut`（最后输出阶段） |

**结论：对原算子的功能无影响，对正常路径的性能影响可忽略（仅首次扫描开销），对异常路径（-inf/-FLT_MAX 行）的行为从「输出 NaN」修正为「输出 0」。**

## 7. 安全性分析

| 场景 | 行为 | 正确性 |
|------|------|--------|
| **正常输入（无 -inf）** | 首次 `SoftmaxInvalidLineCheck` 未发现无效行 → `softMaxCheckRes=false` → 后续全跳过 | 零行为变化 ✓ |
| **attenMask 产生 -FLT_MAX 行** | `LE` 涵盖 `EQ`，检测命中 → sum 置 FLT_MAX，输出置 0 | 向后兼容 ✓ |
| **Key 为 -inf 产生 -inf 行** | 新增 `0xFF800000` 检测命中 → sum 置 FLT_MAX，`InvalidLineUpdate` 置 0 | 修复目标 ✓ |
| **有效行 + 无效行混合** | 有效行 max > -FLT_MAX，不满足 `LE`，不误杀 | 不误杀 ✓ |

## 8. 编译验证

```bash
bash build.sh --pkg --ops=flash_attention_score --soc=ascend950 -j16
```

结果：`EXIT_CODE=0`，全部 9 个 kernel 变体（ascend950_0 ~ ascend950_8）构建成功，无编译错误。
