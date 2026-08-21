# PoolKeyIndexer 算子编译修复记录

## 概述

使用 `bash build.sh --pkg --soc=ascend910b --vendor_name=custom --ops=pool_key_indexer` 编译时出现多处编译错误，涉及 op_host（infershape、tiling、def）和 op_kernel（template tiling key、arch22/arch35 kernel）共 7 个文件、15 类问题。本文档记录每类问题的根因与修复方法。

## 问题总览

| # | 文件 | 问题 | 根因 |
|---|------|------|------|
| 1 | infershape.cpp | 宏参数数量不匹配 | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` 传 6 参，宏仅接受 4 参 |
| 2 | tiling.h | 常量重复定义 | `PKI_TOPK_6K` 在同一头文件中定义两次 |
| 3 | tiling.cpp | Shape 类型不匹配 | `GetInputShape()` 返回 `StorageShape*`，代码用了 `Shape*` |
| 4 | tiling.cpp | TilingData 字段访问错误 | `TILING_DATA_FIELD_DEF` 生成私有字段+getter/setter，代码直接访问公有字段 |
| 5 | tiling.cpp | Platform API 不存在 | 使用了旧版/错误的 API 名称和签名 |
| 6 | tiling.cpp | Workspace API 不存在 | `TilingContext` 无 `GetWorkspaceSize/SetWorkspaceSize` 方法 |
| 7 | tiling.cpp | 格式化字符串类型不匹配 | `%lu` 用于 `int64_t`（有符号 long） |
| 8 | template_tiling_key.h | 模板 Tiling Key 重复编号 | `ASCENDC_TPL_INT_DECL/SEL` 宏不存在，导致未展开文本泄漏到 `@@` 分隔值中 |
| 9 | arch22/kernel.h | 未声明标识符 | Init 参数名与函数体引用名不一致 |
| 10 | arch22/service_vector.h | 非常量左值引用无法绑定临时值 | `ExpandAndAppendIndices` 第一个参数为引用，调用方传入临时 `LocalTensor` |
| 11 | template_tiling_key.h | 宿主编译模式缺少语句终结符 | `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 调用末尾缺少 `;`，在 host 模式展开为 `static` 变量时编译失败 |
| 12 | tiling.cpp + def.cpp | Tiling 代码缺失 `GET_TPL_TILING_KEY` 调用 | tiling.cpp 未调用 `GET_TPL_TILING_KEY` + `SetTilingKey`，def.cpp 定义的属性值无法映射到模板 Tiling Key |
| 13 | tiling.h + template_tiling_key.h + tiling.cpp | quant_mode 值类型冲突与同名重定义 | def.cpp 属性 `Int(-1)` → tiling.h `int32_t(-1)` → TilingData `uint32_t` 时 `static_cast<uint32_t>(-1)=0xFFFFFFFF`，与模板值不匹配；且 `PKI_QUANT_NONE` 同名双重定义（`constexpr=-1` vs `#define`） |
| 14 | arch35/service_vector.h | 非常量左值引用无法绑定临时值 | arch35 的 `ExpandAndAppendIndices` 第一个参数为 `LocalTensor<int32_t>&`，调用方传入 `ReinterpretCast` 临时值（与 #10 arch22 同类问题） |
| 15 | template_tiling_key.h | mask 常量同名冲突 | `PKI_MASK_DEFAULT`/`PKI_MASK_CAUSAL` 在 tiling.h 为 `constexpr`、template_tiling_key.h 为 `#define`，`#define` 静默覆盖（与 #13 quant 同类问题） |

---

## 1. infershape.cpp — OP_LOGE_FOR_INVALID_VALUE_WITH_REASON 宏参数过多

**现象**

```
error: macro "OP_LOGE_FOR_INVALID_VALUE_WITH_REASON" passed 6 arguments, but takes just 4
```

**根因**

宏 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(entityName, paramName, incorrectValue, reason)` 仅接受 4 个参数，其中 `reason` 是纯字符串而非格式化字符串。代码在 `reason` 后追加了 `topk, poolSize` 两个格式化参数，导致宏展开失败。

**修复**

移除多余参数，将动态信息编码到 `incorrectValue` 或 `reason` 字符串中：

```cpp
// Before
OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("PoolKeyIndexer", "topk",
    std::to_string(topk).c_str(),
    "topk(%ld) must be divisible by pool_size(%ld)", topk, poolSize),

// After
OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("PoolKeyIndexer", "topk",
    std::to_string(topk).c_str(),
    "topk must be divisible by pool_size"),
```

---

## 2. tiling.h — PKI_TOPK_6K 重复定义

**现象**

```
error: redefinition of 'constexpr const uint32_t optiling::PKI_TOPK_6K'
```

**根因**

`PKI_TOPK_6K = 6144` 在 tiling.h 的 line 76 和 line 90 各定义了一次（Trunk 长度常量区和 TopK 阈值区有同名的值）。

**修复**

删除 line 90 处的重复定义，保留 line 76 处的定义。

---

## 3. tiling.cpp — gert::Shape* 与 gert::StorageShape* 类型不匹配

**现象**

```
error: cannot convert 'const gert::StorageShape*' to 'const gert::Shape*' in initialization
```

**根因**

在 `gert::TilingContext` 中，`GetInputShape()` / `GetOptionalInputShape()` / `GetOutputShape()` 均返回 `const gert::StorageShape*`，而非 `const gert::Shape*`。`StorageShape` 不直接提供 `GetDim()` / `GetDimNum()` / `GetShapeSize()` 方法，需通过 `GetStorageShape()` 获取内部 `gert::Shape` 引用后再调用。

> 注意：在 `gert::InferShapeContext`（infershape 代码）中，`GetInputShape()` 返回的是 `const gert::Shape*`，两者不可混淆。

**修复**

1. 将所有 `const gert::Shape *` 声明改为 `const gert::StorageShape *`（使用 replaceAll）
2. 在所有 `xxxShape->Get(Dim|DimNum|ShapeSize)(` 调用前插入 `GetStorageShape().`：

```
xxxShape->GetDim(i)          → xxxShape->GetStorageShape().GetDim(i)
xxxShape->GetDimNum()        → xxxShape->GetStorageShape().GetDimNum()
xxxShape->GetShapeSize()     → xxxShape->GetStorageShape().GetShapeSize()
```

使用 sed 批量替换：所有以 `Shape` 结尾的变量名 + `->Get(Dim|DimNum|ShapeSize)` 的模式。

---

## 4. tiling.cpp — TilingData 字段直接访问

**现象**

```
error: 'class optiling::PoolKeyIndexerTilingData' has no member named 's1BaseSize';
did you mean 'uint32_t optiling::PoolKeyIndexerTilingData::s1BaseSize_'?
(accessible via 'uint32_t optiling::PoolKeyIndexerTilingData::get_s1BaseSize()')
```

**根因**

`BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF(type, name)` 宏会自动生成：
- 私有成员 `name_`（带下划线后缀）
- 公有 getter `get_name()`
- 公有 setter `set_name(value)`

代码直接使用 `tilingData_.name = value` 访问公有字段，但该字段是私有的。

**修复**

```cpp
// 写入
tilingData_.s1BaseSize = PKI_S1_BASE_SIZE;     →  tilingData_.set_s1BaseSize(PKI_S1_BASE_SIZE);
// 读取
uint32_t s1 = tilingData_.s1BaseSize;           →  uint32_t s1 = tilingData_.get_s1BaseSize();
```

涉及 `SetTilingData`、`CalcTilingParams`、`CalcWorkspaceSize` 三个函数中所有 TilingData 字段的读写。

---

## 5. tiling.cpp — PlatformAscendC API 名称/签名错误

**现象**

```
error: 'class platform_ascendc::PlatformAscendC' has no member named 'GetNpuArch'; did you mean 'GetCurNpuArch'?
error: 'class platform_ascendc::PlatformAscendC' has no member named 'GetCoreNumPerAicore'; did you mean 'GetCoreNumAic'?
error: no matching function for call to 'PlatformAscendC::GetCoreMemSize(CoreMemType)'
error: 'CalcTschBlockDim' is not a member of 'platform_ascendc'
```

**根因**

使用了不存在或签名错误的 API。

**修复**

| 错误 API | 正确 API | 说明 |
|----------|----------|------|
| `ascendcPlatform.GetNpuArch()` | `ascendcPlatform.GetCurNpuArch()` | 返回 `NpuArch` 枚举 |
| `ascendcPlatform.GetCoreNumPerAicore(aiv, aic)` | `aiv = ascendcPlatform.GetCoreNumAiv();` `aic = ascendcPlatform.GetCoreNumAic();` | 分别获取，无 out-param 形式 |
| `ascendcPlatform.GetCoreMemType(type)` (1 参) | `ascendcPlatform.GetCoreMemSize(type, outVar)` (2 参) | out-param 风格 |
| `platform_ascendc::CalcTschBlockDim(aiv, aic, aiv)` (静态) | `ascendcPlatform.CalcTschBlockDim(aiv, aic, aiv)` (实例方法) | 需通过实例调用 |

---

## 6. tiling.cpp — Workspace API 不存在

**现象**

```
error: 'class gert::TilingContext' has no member named 'GetWorkspaceSize'; did you mean 'GetWorkspaceSizes'?
error: 'class gert::TilingContext' has no member named 'SetWorkspaceSize'; did you mean 'GetWorkspaceSizes'?
```

**根因**

`gert::TilingContext` 的 workspace API 是 `GetWorkspaceSizes(num)`，返回 `size_t*` 数组指针。不存在单值的 `GetWorkspaceSize` / `SetWorkspaceSize` 方法。

**修复**

```cpp
// Before
size_t curWsSize = 0;
context_->GetWorkspaceSize(curWsSize);
context_->SetWorkspaceSize(finalWsSize);

// After
size_t *workspaces = context_->GetWorkspaceSizes(1);
size_t curWsSize = (workspaces != nullptr) ? workspaces[0] : 0;
if (workspaces != nullptr) {
    workspaces[0] = finalWsSize;
}
```

---

## 7. tiling.cpp — 格式化字符串类型不匹配

**现象**

```
warning: format '%lu' expects argument of type 'long unsigned int', but argument 7 has type 'int64_t'
```

**根因**

`keyShape->GetStorageShape().GetDim(2)` 返回 `int64_t`（有符号 long），但格式化字符串用了 `%lu`（无符号 long）。

**修复**

```cpp
// Before
OP_LOGE(context_, "N2 mismatch: query N2=%u, key N2=%lu", info.n2Size, keyShape->GetDim(2));
// After
OP_LOGE(context_, "N2 mismatch: query N2=%u, key N2=%ld", info.n2Size, keyShape->GetDim(2));
```

---

## 8. template_tiling_key.h — ASCENDC_TPL_INT_DECL/SEL 宏不存在

**现象**

```
RuntimeError: There is duplicated number in ASCENDC_TPL_DECL_UINT LAYOUT_K!
Duplicated List: [0, 1, 2, 0, 3, 1, 0, 1].
```

**根因**

CANN 模板 Tiling Key 系统（`template_argument.h`）在 `ASCENDC_TPL_PRE` 模式下仅定义了以下宏：
- `ASCENDC_TPL_DTYPE_DECL/SEL`
- `ASCENDC_TPL_UINT_DECL/SEL`
- `ASCENDC_TPL_BOOL_DECL/SEL`
- `ASCENDC_TPL_FORMAT_DECL/SEL`
- 等等

**不存在** `ASCENDC_TPL_INT_DECL` 和 `ASCENDC_TPL_INT_SEL`。

当使用不存在的宏时，C 预处理器不会展开它，导致原始文本 `ASCENDC_TPL_INT_DECL(MASK_MODE, 0, 3)` 残留在 `@@` 分隔的预处理输出中。Python 解析器在按 `@@` 分割后，将残留文本中的数字（`0`、`3`）混入前一个参数 `LAYOUT_K` 的值列表中，产生重复编号 `[0, 1, 2, 0, 3, 1, 0, 1]`。

此外，`PKI_QUANT_NONE` 值为 `-1`，`UINT` 类型不支持负值，`extract_num` 的正则 `\d+` 会将 `-1` 中的 `1` 提取出来导致错误。

**修复**

1. 将 `ASCENDC_TPL_INT_DECL` → `ASCENDC_TPL_UINT_DECL`，并添加 `ASCENDC_TPL_UI_LIST` 标志和 `ASCENDC_TPL_4_BW` 位宽
2. 将 `ASCENDC_TPL_INT_SEL` → `ASCENDC_TPL_UINT_SEL`，并添加 `ASCENDC_TPL_UI_LIST` 标志
3. 将 `PKI_QUANT_NONE` 从 `-1` 改为 `2`（模板 Tiling Key 文件内的本地定义，不影响 host tiling 代码中的 `constexpr int32_t PKI_QUANT_NONE = -1`）

```cpp
// Before
ASCENDC_TPL_INT_DECL(MASK_MODE, PKI_MASK_DEFAULT, PKI_MASK_CAUSAL),
ASCENDC_TPL_INT_DECL(QUANT_MODE, PKI_QUANT_NONE, PKI_QUANT_FP8_PER_TOKEN, PKI_QUANT_MXFP8),

// After
ASCENDC_TPL_UINT_DECL(MASK_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                      PKI_MASK_DEFAULT, PKI_MASK_CAUSAL),
ASCENDC_TPL_UINT_DECL(QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                      PKI_QUANT_NONE, PKI_QUANT_FP8_PER_TOKEN, PKI_QUANT_MXFP8),
```

SEL 部分同理替换。

---

## 9. arch22/pool_key_indexer_kernel.h — 未声明标识符

**现象**

```
error: use of undeclared identifier 'sparseIndices'
error: use of undeclared identifier 'sparseValues'
error: use of undeclared identifier 'key'
```

**根因**

`Init` 函数的参数命名为 `sparse_indices`、`sparse_values`、`poolKey`（snake_case + 下划线），但函数体中引用了 `sparseIndices`、`sparseValues`、`key`（camelCase），名称不匹配。

**修复**

```cpp
// Before
indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
valueOutGm.SetGlobalBuffer((__gm__ float *)sparseValues);
keyGm.SetGlobalBuffer((__gm__ K_T *)key);

// After
indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparse_indices);
valueOutGm.SetGlobalBuffer((__gm__ float *)sparse_values);
keyGm.SetGlobalBuffer((__gm__ K_T *)poolKey);
```

---

## 10. arch22/pool_key_indexer_service_vector.h — 左值引用无法绑定临时值

**现象**

```
error: non-const lvalue reference to type 'LocalTensor<...>' cannot bind to a temporary of type 'LocalTensor<...>'
```

**根因**

`ExpandAndAppendIndices` 第一个参数 `poolIndices` 声明为 `LocalTensor<int32_t>&`（非常量左值引用）。但调用方传入的是临时值：
- `workLocal_[alignedPoolSize]` — 数组下标返回临时 `LocalTensor`
- `outIdxUb.template ReinterpretCast<int32_t>()` — `ReinterpretCast` 返回临时 `LocalTensor`

C++ 不允许非常量左值引用绑定到临时值。

函数内部仅通过 `poolIndices.GetValue(k)` 读取该参数（不修改），因此可以安全地改为按值传递。

**修复**

```cpp
// Before (declaration + definition)
__aicore__ inline void ExpandAndAppendIndices(LocalTensor<int32_t>& poolIndices, ...);

// After
__aicore__ inline void ExpandAndAppendIndices(LocalTensor<int32_t> poolIndices, ...);
```

`LocalTensor` 是轻量包装器（指针+偏移），按值传递无性能开销。

---

## 11. template_tiling_key.h — ASCENDC_TPL_ARGS_DECL / ASCENDC_TPL_SEL 末尾缺少分号

**现象**

```
error: expected ',' or ';' before 'namespace'
```

**根因**

`ASCENDC_TPL_ARGS_DECL` 和 `ASCENDC_TPL_SEL` 宏在 `ASCENDC_TPL_PRE` 模式下展开为 `@@` 标记（不需要分号），但在宿主编译模式（`#else` 分支）下展开为 `static` 变量定义：

```c
#define ASCENDC_TPL_ARGS_DECL(x, ...) static TilingDeclareParams g_tilingDeclareParams{ __VA_ARGS__ }
#define ASCENDC_TPL_SEL(...)          static TilingSelectParams g_tilingSelectParams{ __VA_ARGS__ }
```

`static` 变量定义需要以 `;` 结尾。当 tiling.cpp `#include` 模板 Tiling Key 头文件后，`ASCENDC_TPL_ARGS_DECL(...)` 展开为不带 `;` 的 `static` 变量定义，导致后续 `#include <numeric>` 等标准头文件解析失败。

参考 `sparse_flash_mla_template_tiling_key.h`，其 `ASCENDC_TPL_ARGS_DECL` 和 `ASCENDC_TPL_SEL` 调用末尾均带 `;`。

**修复**

在 `pool_key_indexer_template_tiling_key.h` 中为两个宏调用添加末尾 `;`：

```cpp
// Before
ASCENDC_TPL_ARGS_DECL(PoolKeyIndexer,
    ...
)
ASCENDC_TPL_SEL(
    ...
)

// After
ASCENDC_TPL_ARGS_DECL(PoolKeyIndexer,
    ...
);
ASCENDC_TPL_SEL(
    ...
);
```

同时将 tiling.cpp 中 `#include "../op_kernel/pool_key_indexer_template_tiling_key.h"` 移到标准头文件之后，避免头文件顺序问题。

---

## 12. tiling.cpp + def.cpp — 缺失 GET_TPL_TILING_KEY 调用

**现象**

编译通过，但运行时框架无法匹配到正确的 kernel 变体（所有模板 SEL 组合均不会被选中），因为 tiling 代码没有设置 Tiling Key。

**根因**

`def.cpp` 定义了算子的运行时属性（如 `quant_mode=-1`、`mask_mode=3`、`layout_q="BSND"` 等）。`template_tiling_key.h` 声明了 8 个模板参数（DT_Q、DT_K、DT_OUT、LAYOUT_Q、LAYOUT_K、MASK_MODE、QUANT_MODE、RETURN_VALUE）及多组 SEL 组合。框架需要 Tiling Key 来选择正确的编译产物。

但 `tiling.cpp` 的 `DoTiling` 函数既未调用 `GET_TPL_TILING_KEY` 计算键值，也未调用 `context_->SetTilingKey()` 设置键值。参考同类算子（sparse_flash_mla、lightning_indexer、mla_prolog 等），均在 tiling 代码中通过 `GET_TPL_TILING_KEY` 将运行时属性映射为模板 Tiling Key 并写入 context。

此外，`def.cpp` 中 `quant_mode` 默认值为 `-1`（用户接口值），而模板 Tiling Key 中 `PKI_QUANT_NONE` 已改为 `2`（UINT 不支持负值），需要在 tiling 代码中做值映射：`-1 → 2`。

**修复**

1. tiling.cpp 增加 `#include "../op_kernel/pool_key_indexer_template_tiling_key.h"`（传递引入 `GET_TPL_TILING_KEY` 宏）

2. `DoTiling` 函数中在 `SetBlockDim` 之前添加 Tiling Key 计算：

```cpp
// Map runtime attribute values to template tiling key values
uint32_t tplDtQ = 0;
if (info.queryDtype == ge::DT_FLOAT16) {
    tplDtQ = 1;       // PKI_TPL_FP16
} else if (info.queryDtype == ge::DT_BF16) {
    tplDtQ = 27;      // PKI_TPL_BF16
} else if (info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
    tplDtQ = 2;       // PKI_TPL_FP8
}
// quant_mode: -1 (no quant, def.cpp default) → 2 (PKI_QUANT_NONE in template)
uint32_t tplQuantMode = (info.quantMode == -1) ? 2 : static_cast<uint32_t>(info.quantMode);

uint64_t tilingKey = GET_TPL_TILING_KEY(
    tplDtQ, tplDtQ, 3U,                        // DT_Q, DT_K, DT_OUT
    static_cast<uint32_t>(info.layoutQ),        // LAYOUT_Q
    static_cast<uint32_t>(info.layoutK),        // LAYOUT_K
    static_cast<uint32_t>(info.maskMode),       // MASK_MODE
    tplQuantMode,                                // QUANT_MODE (mapped)
    static_cast<uint32_t>(info.returnValue ? 1 : 0));  // RETURN_VALUE
context_->SetTilingKey(tilingKey);
```

**值映射说明**

| 模板参数 | def.cpp 属性值 | 模板 Tiling Key 值 | 映射方式 |
|----------|---------------|-------------------|----------|
| DT_Q | `DT_FLOAT16` / `DT_BF16` / `DT_FLOAT8_E4M3FN` | 1 / 27 / 2 | if-else 分支 |
| DT_K | 同 DT_Q | 同 DT_Q | 直接复用 |
| DT_OUT | 固定 `DT_INT32` | 3 | 常量 |
| LAYOUT_Q | `"BSND"` / `"TND"` | 0 / 1 | `static_cast<uint32_t>(info.layoutQ)` |
| LAYOUT_K | `"BSND"` / `"TND"` / `"PA_BBND"` | 0 / 1 / 2 | `static_cast<uint32_t>(info.layoutK)` |
| MASK_MODE | `0` / `3` | 0 / 3 | `static_cast<uint32_t>(info.maskMode)` |
| QUANT_MODE | `-1` / `0` / `1` | `2` / `0` / `1` | 三元运算：`-1 → 2`，其余直传 |
| RETURN_VALUE | `false` / `true` | 0 / 1 | 三元运算 |

> `def.cpp` 中 `quant_mode` 属性默认值 `-1` 是用户接口值，无需修改。值映射在 tiling 代码中完成，保持用户 API 与模板内部编码解耦。

**def.cpp 本身无需修改**，但需在 tiling.cpp 中补充上述映射逻辑和 `SetTilingKey` 调用。

---

## 13. tiling.h + template_tiling_key.h + tiling.cpp — quant_mode 值类型冲突与同名重定义

**现象**

编译通过，但运行时 kernel 通过 TilingData 收到的 `quantMode` 为 `0xFFFFFFFF`（`static_cast<uint32_t>(-1)`），而非模板 Tiling Key 期望的 `2`，导致 kernel 无法正确识别"不量化"场景。

**根因**

`quant_mode` 在三个层次存在值/类型不一致：

| 层次 | 定义 | 类型 | 值 |
|------|------|------|----|
| def.cpp 属性 | `Attr("quant_mode").Int(-1)` | `int`（有符号） | `-1` |
| tiling.h TilingInfo | `int32_t quantMode = PKI_QUANT_NONE` | `int32_t`（有符号） | `-1` |
| tiling.h TilingData | `TILING_DATA_FIELD_DEF(uint32_t, quantMode)` | `uint32_t`（无符号） | — |
| tiling.cpp 赋值 | `set_quantMode(static_cast<uint32_t>(info.quantMode))` | `uint32_t` | `0xFFFFFFFF` |
| template_tiling_key.h | `#define PKI_QUANT_NONE 2` | 宏 | `2` |

**冲突一：值溢出**

`info.quantMode = -1`（来自 def.cpp 属性），经 `static_cast<uint32_t>(-1)` 后变为 `0xFFFFFFFF`。kernel 从 TilingData 读取到 `0xFFFFFFFF`，无法匹配模板值 `0` / `1` / `2` 中的任何一个。

**冲突二：同名重定义**

`PKI_QUANT_NONE` 被定义了两次：
- `tiling.h`: `constexpr int32_t PKI_QUANT_NONE = -1;`（host 侧常量）
- `template_tiling_key.h`: `#define PKI_QUANT_NONE 2`（kernel 模板侧宏）

tiling.cpp 同时 include 两个头文件后，`#define` 会静默覆盖后续代码中所有 `PKI_QUANT_NONE` 的引用，将 `-1` 替换为 `2`，可能导致逻辑错误。

**修复**

1. **消除同名冲突**：将 template_tiling_key.h 中的常量加 `TPL_` 前缀：

```cpp
// Before
#define PKI_QUANT_NONE 2
#define PKI_QUANT_FP8_PER_TOKEN 0
#define PKI_QUANT_MXFP8 1

// After
#define PKI_TPL_QUANT_NONE 2
#define PKI_TPL_QUANT_FP8_PER_TOKEN 0
#define PKI_TPL_QUANT_MXFP8 1
```

DECL 和 SEL 中的引用同步更新为 `PKI_TPL_QUANT_*`。

2. **值映射**：在 tiling.cpp 的 `SetTilingData` 和 `GET_TPL_TILING_KEY` 调用中，使用 `+1` 偏移将 host 负值映射为模板非负值：

```cpp
// SetTilingData 中
tilingData_.set_quantMode(static_cast<uint32_t>(info.quantMode + 1));

// GET_TPL_TILING_KEY 调用中
uint32_t tplQuantMode = static_cast<uint32_t>(info.quantMode + 1);
```

**映射关系总结**

```
用户接口 (def.cpp)     host 内部 (tiling.h)      TilingData / 模板 (template_tiling_key.h)
                     info.quantMode            tplQuantMode = quantMode + 1
─────────────────    ─────────────────────    ─────────────────────────────────
quant_mode = -1   →   -1                  →   0  (PKI_TPL_QUANT_NONE)
quant_mode = 0    →    0                  →   1  (PKI_TPL_QUANT_FP8_PER_TOKEN)
quant_mode = 1    →    1                  →   2  (PKI_TPL_QUANT_MXFP8)
```

采用 `+1` 偏移而非特判映射的优势：
- 映射统一：`tpl = host + 1`，无需对 `-1` 做特判分支
- 可逆：`host = tpl - 1`，kernel 侧如需还原原始值可直接减 1
- 模板值连续：`0, 1, 2`，编码紧凑无空洞

> **设计原则**：def.cpp 的属性值（`-1`）是用户接口约定，不应修改；tiling.h 的 `int32_t` 是 host 中间表示，保持与属性一致；TilingData 的 `uint32_t` 和模板 Tiling Key 的值在 tiling.cpp 中通过 `+1` 偏移统一映射，实现用户接口与 kernel 内部编码的解耦。

---

## 14. arch35/pool_key_indexer_service_vector.h — 非常量左值引用无法绑定临时值

**现象**

```
error: non-const lvalue reference to type 'LocalTensor<...>' cannot bind to a temporary of type 'LocalTensor<...>'
```

**根因**

与 #10（arch22）完全相同的问题。arch35 的 `ExpandAndAppendIndices` 第一个参数声明为 `LocalTensor<int32_t>&`（非常量左值引用），但调用方（line 662）传入 `indicesOutLocal_.ReinterpretCast<int32_t>()` 返回的临时值。C++ 不允许非常量左值引用绑定到临时值。

arch22 已修复此问题，但 arch35 遗漏。

**修复**

```cpp
// Before (declaration line 72 + definition line 305)
__aicore__ inline void ExpandAndAppendIndices(LocalTensor<int32_t>& poolIndices, ...);

// After
__aicore__ inline void ExpandAndAppendIndices(LocalTensor<int32_t> poolIndices, ...);
```

函数内部仅通过 `poolIndices.GetValue(k)` 读取（不修改），按值传递安全。

---

## 15. template_tiling_key.h — mask 常量同名冲突

**现象**

编译通过，但存在 `PKI_MASK_DEFAULT`/`PKI_MASK_CAUSAL` 同名双重定义隐患。

**根因**

与 #13（quant_mode）相同的同名冲突模式：

| 文件 | 定义 | 类型 | 值 |
|------|------|------|----|
| tiling.h | `constexpr int32_t PKI_MASK_DEFAULT = 0` | C++ 变量 | 0 |
| template_tiling_key.h | `#define PKI_MASK_DEFAULT 0` | 预处理宏 | 0 |

tiling.cpp 同时 include 两个头文件后，`#define` 静默覆盖后续代码中所有 `PKI_MASK_DEFAULT` 引用。虽然 mask 值本身已是非负（0 和 3），不存在 quant_mode 那样的类型溢出问题，但同名冲突仍是不良实践。

**修复**

将 template_tiling_key.h 中的常量加 `TPL_` 前缀，与 quant 常量保持一致：

```cpp
// Before
#define PKI_MASK_DEFAULT 0
#define PKI_MASK_CAUSAL 3

// After
#define PKI_TPL_MASK_DEFAULT 0
#define PKI_TPL_MASK_CAUSAL 3
```

DECL 和 SEL 中的引用同步更新。mask 值本身已非负，无需 `+1` 偏移，`static_cast<uint32_t>(info.maskMode)` 直接使用即可。

---

## 修复文件清单

| 文件 | 修改内容 |
|------|----------|
| `op_host/pool_key_indexer_infershape.cpp` | 宏参数从 6 改为 4 |
| `op_host/pool_key_indexer_tiling.h` | 删除重复 `PKI_TOPK_6K`；quant 常量加注释 |
| `op_host/pool_key_indexer_tiling.cpp` | Shape→StorageShape + GetStorageShape() + set_/get_ + Platform API + Workspace API + 格式化字符串 + GET_TPL_TILING_KEY + SetTilingKey + quant_mode +1 映射 |
| `op_kernel/pool_key_indexer_template_tiling_key.h` | INT_DECL→UINT_DECL + UI_LIST + PKI_QUANT→PKI_TPL_QUANT(+1偏移) + PKI_MASK→PKI_TPL_MASK + DECL/SEL 末尾加分号 |
| `op_kernel/arch22/pool_key_indexer_kernel.h` | 参数名 snake_case 对齐 |
| `op_kernel/arch22/pool_key_indexer_service_vector.h` | `ExpandAndAppendIndices` 引用参数改为值传递 |
| `op_kernel/arch35/pool_key_indexer_service_vector.h` | `ExpandAndAppendIndices` 引用参数改为值传递（与 arch22 一致） |
| `op_host/pool_key_indexer_def.cpp` | 无需修改（属性值 -1 通过 tiling.cpp +1 映射为模板值） |

## 参考模式来源

修复中参考了同仓库以下算子的正确实现：
- `attention/sparse_flash_mla` — `KV_LAYOUT_T` 使用 BSND/TND/PA_BBND 三值的 `ASCENDC_TPL_UINT_DECL` 正确用法
- `attention/flash_attention_score` — `ASCENDC_TPL_UINT_DECL` 多值列表 + Platform API 用法
- `moe/moe_token_permute` — `tilingData.set_field()` 和 `GetWorkspaceSizes(1)` 用法
- `ffn/ffn` — `GetCoreNumAiv/Aic`、`GetCoreMemSize` 2 参、`CalcTschBlockDim` 实例方法
- `common/src/op_host/tiling_util.cpp` — `GetCurNpuArch()` 正确用法
- `attention/sparse_flash_mla/op_host/sparse_flash_mla_tiling.cpp` — `GET_TPL_TILING_KEY` + `SetTilingKey` 调用模式 + `#include "../op_kernel/*_template_tiling_key.h"` 用法
- `attention/lightning_indexer/op_host/lightning_indexer_tiling.cpp` — dtype 到模板值的 if-else 映射模式
- `attention/quant_compressor/op_host/arch35/quant_compressor_tiling.cpp` — quant_mode 属性直传到 GET_TPL_TILING_KEY 的 Strategy A 用法

---
---

# 运行时崩溃修复记录（EZ9999 MTE 非法 GM 地址）

> 2026-08-20 排查修复。在 910B3 + CANN 9.1.0 环境，上述编译问题修复后，算子所有用例运行即崩溃（EZ9999 `MTE accesses an invalid GM address` / `Illegal instruction`），此前 pytest 12 用例 0 通过。本文记录定位过程与最终根因。

## 问题现象

- 任意 shape（BSND/TND、任意 pool_size/topk/mask_mode）调用 `pool_key_indexer` 均异步崩溃，错误码 507015（aicore exception）
- 崩溃点不固定：有时报 fftsplus aivector error（AIV 核），有时报 aicore error（AIC 核），甚至空闲 AIV 核也报错
- 同一二进制在 mssanitizer memcheck/synccheck 拦截模式下可完整跑通、无任何越界报告（假阴性）
- 出错 pc 随代码插入（printf）而漂移，说明报错 pc 是异常陷入点而非出错指令本身

## 排除项（走过的弯路，供复用）

| 假设 | 验证方法 | 结论 |
|------|----------|------|
| 输入 tensor 不在同一 NPU 设备 | 实测全在 npu:0；故意制造不匹配对比错误签名（不匹配时 AIC+AIV 同时报错且 pc 不固定，签名不同） | 排除 |
| 核间同步 flag id 冲突（syncC1V1/syncV1C1 双向共用 id 0） | 对照 LIV2：其同样双向共用 0，是 FIA_SYNC_MODE2 模式下的正确设计 | 排除 |
| vec1Param 初始化写缺 MTE3_V 等待 | 对齐 LIV2 补 SetWaitFlag 重新编译 | 无效 |
| kernel 与 host 代码不同步 | diff 仓库源码与 vendor 安装包，一致 | 排除 |
| 环境问题 | 同机同管线编译安装 LIV2（同源架构）A/B 测试：LIV2 通过、PKI 崩溃 | 排除环境，锁定 PKI 特有 |

## 定位过程（关键步骤）

1. **A/B 对照**：将 `lightning_indexer_v2`（LIV2，与 PKI 同架构：MIX AIC+AIV、同样的 workspace 三段布局 mm1Res|vec1Res|vec1Param、同样的 LD 流程）编入同一 vendor 包测试 —— LIV2 全过，PKI 全崩，排除环境。

2. **workspace 第一轮修复（必要但不足）**：发现 host `CalcWorkspaceSize` 的公式与 kernel 实际布局完全不符（host 按旧布局只算出 409600 字节；kernel 按自身公式需要 ~10.5MB）。重写为镜像 kernel 公式后仍崩，但推进了排除面。

3. **编译宏二分**：在 ProcessBaseBlock/InitBuffers/ProcessVec 各阶段插入 `PKI_BISECT_SKIP_*` 编译宏：
   - 屏蔽全部计算（ComputeMm1+ProcessVec）仍崩 → 与计算逻辑无关
   - 屏蔽 InitBuffers 初始化段 → 通过
   - 三段单独独活（CopyIn 链 / Sort 链 / CopyOut 链）都崩 → 存在公共毒药
   - AIC 侧单独跑崩、AIV 侧单独跑也崩 → 双侧独立触发

4. **mssanitizer 决定性证据**：对"SKIP_MM1 + 全量 ProcessVec"构建跑 memcheck，终于复现并抓到 illegal read/write。其 `MEMORY_RECORD` 记录了运行时真实分配：
   ```
   MALLOC dstAddr:0x12c041400000 memSize:23068672 (22MB)   ← RT 实际分配
   MALLOC ... memInfoDesc:input_13 memSize:21053440        ← tiling 申请的 21MB
   ```
   而 kernel 内 `GetUserWorkspace()` 返回 `0x12c042400000`（= 基址 + **16MB**），首个非法访问地址 `0x12c042a00000` 与 22MB 分配边界**精确重合**。

## 根因

**host tiling 的 workspace 总量漏加了 16MB 系统保留区。**

- arch22（`__NPU_ARCH__ == 2201`）上 `GetUserWorkspace()` 实现为 `workspace + RESERVED_WORKSPACE`，其中 `RESERVED_WORKSPACE = 16MB`（`kernel_utils_constants.h:274`），即 kernel 实际可用的 user workspace 从总分配的 +16MB 处开始
- LIV2 的 `DoTiling` 首行就是 `workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize()`（= 该 16MB）再加上自身区域
- PKI 的 `CalcWorkspaceSize` 只计算了自身区域，运行时按 tiling 尺寸分配（取整 22MB），kernel 从 +16MB 起使用 10.5MB → 实际只剩 ~6MB，各核按 `aiCoreIdx * 区域大小` 偏移写入时越界
- 原始 400KB 版本更是从第一条 AIC fixp 写（mm1ResGm）就开始大幅越界 —— 这正是最初 `MTE accesses an invalid GM address` 的直接来源
- 崩溃点不固定 / 空闲核也崩 / pc 漂移 / mssanitizer 假阴性等现象均为越界写踩踏 16MB 保留区（系统区）导致的时序性副作用

## 修复

`op_host/pool_key_indexer_tiling.cpp` 两处：

1. `CalcWorkspaceSize` 重写为镜像 kernel 侧布局（`op_kernel/arch22/pool_key_indexer_kernel.h` Init）：

   ```
   | mm1Res（score 双缓冲） | vec1Res（LD 中间结果） | vec1Param（LD 参数） |
   scoreGm   = aicNum × 2 × mBaseSizeAlign × 512 × 4B
   ldScore   = aicNum × s1BaseSize × 2 × 2 × BASE_TOPK(2048) × 4B
   ldParam   = aicNum × s1BaseSize × 2 × LD_PARAM_NUM(16) × 8B
   ```
   其中 `s1BaseSize = sparseCount > 2048 ? 8192/sparseCount*2 : 8`，
   `mBaseSizeAlign = Align(s1BaseSize × gSize, 16)`。核数取 `max(aicNum, aivNum)` 以覆盖 kernel 侧 `GetBlockNum()` 的取值。

2. `DoTiling` 中补上系统保留区（关键一行）：

   ```cpp
   // kernel 侧 GetUserWorkspace() 会跳过系统保留区（arch22 为 16MB，
   // 即 GetLibApiWorkSpaceSize()），必须计入总 workspace，否则 kernel 实际
   // 可用空间不足导致 GM 越界（参照 lightning_indexer_v2_tiling.cpp）
   workspaceSize += ascendcPlatform.GetLibApiWorkSpaceSize();
   ```

修复后 plog 验证：`workspace: ... total=37801984`（16MB + 21MB），与 kernel 需求匹配。

## 测试脚本同步修复

`tests/pytest/simple_test.py` 用例 2（TND）：`actual_seq_k` 误传**累计 token 数** `[64, 128]`，host 校验（`tiling.cpp:710`）要求 TND 下 last 必须等于 pool_key 的 T 维（即**累计 pool 数**）。修正为：

```python
actual_seq_k = torch.tensor([k // 16 for k in k_lens], dtype=torch.int64).cumsum(0).npu()  # [4, 8]
```

## 验证结果

| 项目 | 结果 |
|------|------|
| `simple_test.py` 两用例 | 全部 PASSED（历史 0 通过） |
| 9 组 shape 变体（1核/8核/64核、S2=4096 多 S2 块、topk=2048/4096、B=2、pool16+causal、G=1） | 9/9 通过 |
| 重复运行 ×3 | 稳定通过 |
| LIV2 同包回归 | 无影响 |
| 官方 pytest `test_pool_key_indexer.py` | 9/12 通过（历史 0/12） |

## 遗留事项（已全部解决）

3 个 TND 用例失败在测试侧 `actual_seq_k` 语义错误（与 simple_test.py 同类）：`test_pool_key_indexer.py` 的 `_build_tnd_inputs` 误用 **token 长度** 构造 `actual_seq_k`，而 host 校验（TND 下 last 必须等于 pool_key 的 T 维）与 golden 内部（key 解包、causal mask 的 `L_orig = S2*pool_size + tail`）均按 **累计 pool 数** 解释。修正为：

```python
k_pool_lens = [kl // pool_size for kl in k_lens]
actual_seq_k = _make_actual_seq(k_pool_lens)   # 原： _make_actual_seq(k_lens)
```

修正后 pytest **12/12 全部通过**（修复前 0/12，workspace 修复后 9/12）。

## 经验教训

1. **MIX 类算子 workspace 总量 = `GetLibApiWorkSpaceSize()`（系统保留区，arch22=16MB）+ 算子自用区域**，二者缺一不可。host tiling 与 kernel `GetUserWorkspace()` 是"总量"与"+16MB 起点"的契约关系
2. host/kernel 的 workspace 布局公式必须逐字段镜像维护，任何一侧改动都要同步另一侧（本次 host 公式与 kernel 完全不符是第一层 bug，漏加 16MB 是第二层）
3. 越界踩踏系统保留区的故障表现极具迷惑性：崩溃核不固定、pc 漂移、拦截式检测工具假阴性。`mssanitizer` 的 `MEMORY_RECORD`（记录 RT 真实 MALLOC 边界）是对照"kernel 实际访问地址 vs 运行时分配边界"的最有效手段
4. 同源参考算子（LIV2）A/B 是排除环境问题的最快路径；编译宏二分（屏蔽/独活）适合"多嫌疑点全崩"的场景

---
---

# 精度修复记录（pool_size>1 索引错乱）

> 2026-08-20 二轮排查。测试侧引入"强制 indices 集合对比"后暴露 7 个 ps>1 用例失败（values 全对、indices 错乱）。经系统性 code review ps>1 输出链路 vs LIV2 逐行差异，定位并修复 4 处 bug（3 处 kernel + 1 处 golden 测试侧）。

## 问题现象

- 仅 `pool_size > 1` 失败（ps=1 的 5 个用例全过），BSND/TND 均中招
- values 输出全部正确（TopK 分数链路无损），仅 indices 错乱
- 错误模式高度规律：错误值落在 `pos = value + C (mod N)` 的固定偏移处，如 ps16/topk128 时 C=113/129/2，跨构建输出会漂移但模式稳定

## 定位过程（关键转折）

前期 7 轮盲改（流水同步 V→MTE3/S→V/V→S、idx 数据源改直读排序对、Extract dstIndex 布局、事件链）全部无效——输出逐位不变，证明那些代码本来就对。教训：**修改无效应立即回退并更换定位策略，而不是继续叠加假设**。

转折点是系统性 code review：LIV2 无 ps>1 展开逻辑，该分支为 PKI 新增代码，无参照实现。将输出链路所有偏移/步长公式列成表逐一审计，与"输出张量的真实行宽"对照后 4 处 bug 一次浮出。

## Bug 1：indices 输出行基址按 sparseCount 而非 outputLen 计算（kernel，根因）

**位置**：`op_kernel/arch22/pool_key_indexer_kernel.h` 的 `CalcRunInfo`（`indiceOutCoreOffset`）、`DealActSeqLenIsZero`（TND/BSND 清理偏移）、`ProcessInvalid`（totalOutputSize）三处同源错误。

**根因**：LIV2 的输出行宽恰为 `sparseCount`（topk），其公式 `prefixSum × kHeadNum × sparseCount + n2Idx × sparseCount` 正确。PKI 的 ps>1 输出行宽为 `outputLen = sparseCount×poolSize + poolSize − 1`（如 8×16+15=143），沿抄 LIV2 公式导致 batch>0 的行基址大幅错位、单行写入跨界覆盖相邻行。

**数学闭环**（TND, B=2, S1=32, sparseCount=8, outputLen=143, ps=16）：
- batch1 行基址实写 `32×8=256`，正确应为 `32×143=4576`
- 256 = 1×143 + **113** → 探针实测的 C=113 之谜精确解开
- batch1 s1=7 的写入落在行8 pos113..142 + 行9 pos0..17，与 dump 逐位吻合
- ps=1 全对的原因：`outputLen == sparseCount`，公式恰好成立

**修复**：三处统一引入
```cpp
uint32_t idxOutStride = (constInfo.poolSize > 1)
                            ? (constInfo.sparseCount * constInfo.poolSize + constInfo.poolSize - 1)
                            : constInfo.sparseCount;
```
（`ProcessVec` 的 `idxStride` 原本就写对了，可对照）

## Bug 2：RunInfo.poolTailK 从未赋值（kernel）

**位置**：`CalcRunInfo`。

**根因**：`RunInfo.poolTailK` 声明默认 0，全代码无赋值点（kernel 仅在 `GetS1S2ActualSeqLen` 里读了局部 `tailK` 参与 `actS2SizeOrig` 计算，未传入 runInfo）→ `ExpandAndAppendIndices` 的 `if (poolTailK > 0)` 永假，**尾块永不输出**。症状：causal 用例 NPU 有效数恒 128 而 golden 为 129+ 递增。

**修复**：
```cpp
runInfo.poolTailK = hasPoolTailK_ ? static_cast<int32_t>(poolTailKGm_.GetValue(tempLoopInfo.bIdx)) : 0;
```

## Bug 3：尾块按 8 对齐写越出 tail 容量（kernel）

**位置**：`ExpandAndAppendIndices` 尾段。

**根因**：原实现 `Duplicate(tokenIndices[tailPos], L_orig-poolTailK, Align(visibleTailK,8)) + Add(尾模板)`，对齐宽度会超出 tail 容量 `poolSize−1`。ps=8、tail=5 时写 8 个（多 3 个）；ps=16、tail=9 时写 16 个直接填满整行（143），症状为"整行有效 143"。

**修复**：先按 8 对齐**清空**整个尾区，再标量精确写 `visibleTailK` 个（`SetValue(L_orig-poolTailK+t)`），末尾补 `S→MTE3` 同步（标量写后由 MTE3 搬出）。

## Bug 4：golden 的 BSND 场景 L_orig 重复乘 pool_size（测试侧）

**位置**：`tests/atk/pool_key_indexer_golden.py` `forward` 内 `l_orig = curr_k * pool_size + cur_pool_tail_k`。

**根因**：TND 下 `actual_seq_k` 为累计 **pool 数**，`×pool_size` 还原 token 数正确；BSND 下无 `actual_seq_k`，`curr_k`（=k_seq）**本就是 token 数**，再乘 ps 得到 2053 而非 261，尾 token 算成 [2048..2052]（golden 错）而 NPU 的 [256..260] 才是对的。

**修复**：
```python
if actual_seq_k is not None:      # TND: 累计 pool 数 → 还原 token 数
    l_orig = curr_k * pool_size + cur_pool_tail_k
else:                             # BSND: k_seq 已是 token 数
    l_orig = curr_k + cur_pool_tail_k
```

## 验证结果

| 项目 | 结果 |
|------|------|
| 官方 pytest（含强制 indices 集合对比 + 归一化 values 独立通道） | **12/12 通过** |
| simple_test.py | ALL DONE |
| 稳定性重复 ×3 + 9 组 shape 矩阵（1/8/64 核、S2=4096、topk 2048/4096、B=2、ps16 causal、G=1） | 全过 |
| ps=1 场景回归 | 无影响 |

## 经验教训

1. **沿抄参考实现的公式时，必须核对其隐含前提**：LIV2 输出行宽 = sparseCount 恰好成立；PKI ps>1 行宽改变后，一切以 sparseCount 为 stride 的公式（行基址/清理/总量）都要重新推导
2. **"修改无效应立即回退换策略"**：7 轮盲改同步/数据源不如一张偏移公式审计表。对确定性数据错乱，先做"输出布局数学反推"（pos−value 偏移统计）再动手，C 值规律直接指向基址错位
3. 新增 struct 字段（如 RunInfo.poolTailK）若无赋值点，等于静默禁用对应功能——review 时对"声明未赋值"要专门扫一遍
4. 测试侧 golden 与算子语义出现分歧时，先用独立第三实现（手写 torch）仲裁谁对，本轮 Bug 4 正是靠 NPU 与手工推导一致、golden 单方出错而定性

---
---

# PA_BBND 场景修复记录（布局/寻址/语义）

> 2026-08-21 三轮排查。simple_test 与 pytest 增加 PA 场景覆盖后暴露：模板 SEL 缺失(EZ1001)、TND/PA 组合假阳性（异步报错未同步）、indices 集合与 golden 不符。最终按 LIV2 权威语义完成修复。

## Bug 1：模板 SEL 缺 TND/PA_BBND 组合（kernel）

**现象**：TND/PA_BBND 调用报 `EZ1001 not find tilingKey[553844993]`，但 simple_test 显示"PASSED"——**假阳性**：aclnn 异步报错未抛到主流程，输出的 indices 实为未写入的 empty 垃圾张量。

**修复**：
- `pool_key_indexer_template_tiling_key.h` 新增 2 条 SEL（FP16/BF16 的 TND/PA_BBND，mask=default+causal、val=0/1）
- simple_test.py 末尾补 `torch_npu.npu.synchronize()` + 输出有效性断言，堵住同类假阳性

## Bug 2：PA keyStride0=0 导致物理块寻址塌缩（host）

**根因**：`GetDynamicInputStride` 对连续 tensor 返回空 → `keyStride0=0` → `bt[blk]×0` 所有物理块基址塌缩为 0。对照实验（恒等 bt、全同物理块）实证：kernel 用了 bt 但块基址全 0。

**修复**（tiling.cpp）：
```cpp
if (!keyStridesVec_.empty() && keyStridesVec_[0] != 0) {
    info.keyStride0 = keyStridesVec_[0];              // 0轴非连续: 运行时 stride(LIV2 arch35 形态)
} else {
    info.keyStride0 = blockSize * n2Size * headDim;   // 连续: 按形状推导
}
```
比 LIV2（缺失即 0，依赖输入必须非连续才拿到 stride）更稳：连续输入同样可用。

## Bug 3：PA key 块内行语义——按 LIV2 权威语义重构（关键认知修正）

**弯路**：初次修复时误判"PA key 块内是 token 展开"，把 `KeyNd2NzForPA` 改为池级取数（`srcDValue=poolSize×headDim`、代表 token 定位）+ 测试 helper 做了 token 展开。该方案在自造约定下自洽并通过全部测试。

**纠正依据**（LIV2 golden `lightning_indexer_v2_golden.py:834`）：`key_bnsd` 的 S2 维直接是**池数**逐行填块——LIV2 权威语义为：

> **PA key `(blockNum, blockSize, N2, D)` 块内每行 = 1 个池（key 不做 token 展开）；`actS2SizeOrig = 池数×poolSize + tail` 的 token 语义仅用于 causal 可见性换算，与 key 搬运无关。**

**重构**（对齐 LIV2）：
- kernel `KeyNd2NzForPA`：回退池级取数，恢复 LIV2 原样（`srcDValue=headDim` 连续池行、L0 分型）；正当差异仅保留 keyStride0 寻址（=arch35 形态）
- 测试 helper `_pack_pa_key`/`_build_block_table`：改池粒度打包（每池一行、每 batch 独立连续填块、随机排布）；并修复两 helper 块数口径不一致导致的 bt 越界
- golden：`_pa_bbnd_to_bnsd` 池行直取；`actual_seq_k` 不再乘 pool_size；`curr_k` 统一池数语义（BSND 由 `k_seq//pool_size` 换算）；`L_orig = 池数×poolSize + tail`

**架构结论**（对照 LIV1/LIV2 arch22/arch35 四方审计）：
- `keyStride0` 寻址是 arch35（950）形态，支持 0 轴非连续（KV cache 融合场景）；arch22 硬算 `bs×N2×D` 只对连续成立
- 非 PA `KeyNd2Nz` 不需要 stride：host `CheckKeyContiguous` 对非 PA 全轴校验连续（非连续拒绝），PA 仅从轴 1 起校验（显式放行 0 轴非连续）——与两代 LIV2 能力边界一致
- LIV2 的 `PA_BBND + cmp_ratio>1` 组合按代码推演存在同款列/池错位隐患，其测试固定 `cmp_ratio=1` 未暴露（已分析，待反馈维护者）

## Bug 4：文档语义勘误（aclnnKpoolIndexer.md）

- 279 行：PA 的 actual_seq_k 语义由"当前 batch 的 token 数"更正为"当前 batch 的 **pool 数**（有效 token 数除以 pool_size 下取整，非前缀和，与 lightning_indexer_v2 的 seqused_k 语义一致）"
- 209 行：PA blockSize 语义由"一个 block 的 token 数"更正为"一个 block 的 **pool 数**（块内每行为一个 pool 的 key）"
- 278 行：维度约束同步改为 pool 数口径

## 验证

| 项目 | 结果 |
|------|------|
| pytest（12 原有 + 4 PA） | **16/16 通过** |
| simple_test.py（含 BSND/PA、TND/PA） | 4/4 PASSED，ALL DONE（含全局同步+有效性断言） |
| ps=1 回归（与 LIV2 语义退化一致） | 无影响 |

## 遗留

1. PA 0 轴非连续（blockFusion 融合 KV cache）测试用例未覆盖（keyStride0 运行时 stride 分支）
2. arch35（950）分支 PA 支持未验证、模板 SEL 未同步确认
3. LIV2 `PA+cmp_ratio>1` 隐患待反馈维护者

---

## 遗留问题记录：arch35（Ascend 950）编译失败（存量问题，非本次引入）

> 2026-08-21 排查 PA 场景 arch35 分支时发现。已用 `git stash` 验证：**HEAD 原样代码编译 950 同样失败**，与本次 PA/精度修复无关。

### 现象

`bash build.sh --pkg --soc=ascend950 --vendor_name=custom --ops=pool_key_indexer` 编译失败（exit=2）：

1. **fp8 后端错误（214 处，根因）**：
   ```
   fatal error: error in backend: Error: fp8e8m0, fp8e5m2, fp8e4m3, fp4e2m1x2,
   fp4e1m2x2, hifloat8, hifloat4x2 type only supports pointer operations,
   scalar float type semantics are not supported!
   bisheng: error: clang frontend command failed with exit code 70
   ```
   影响**所有**模板实例（含 FP16 主模板 0d4d6c），非仅 FP8 组合——arch35 vector 侧（疑为 VF radix/histogram TopK 路径，`op_kernel/arch35/vf/`）存在对任意 dtype 实例化都会生成 fp8 标量操作的代码，触发后端限制。.o 缺失导致 ld.lld 链接连锁失败。
2. **参数名笔误（已修）**：`arch35/pool_key_indexer_kernel.h:464` `keyGm.SetGlobalBuffer((__gm__ K_T *)key)` 中 `key` 未声明，实际参数名为 `poolKey`——本次已修复（950 专项修改之一）。

### 已完成的 arch35 侧工作

- ✅ 修复 `key` → `poolKey` 笔误（编译前置阻塞）
- ✅ 确认 `KeyNd2NzForPA`（arch35）与 LIV2 arch35 同构：`bt[blk]×keyStride0 + blkOffset×headDim` 池行连续寻址，无需修改
- ✅ 确认 `GetS1S2ActualSeqLen`/模板入口与 arch22 同构；模板 SEL 为 soc 无关声明，新增 TND/PA_BBND 组合自动对 950 生效
- ✅ host 侧 `keyStride0`（运行时 stride / 连续推导）为两 soc 共用，已生效

### 待办（需 arch35 维护者处理）

1. 定位并修复 arch35 vector（`vf/` 目录）的 fp8 标量语义问题——可参照报错 `exit code 70` + 二分法屏蔽 VF 段定位
2. 修好后补 950 的 PA 场景编译验证（本机 910B 无 950 运行环境，只能编译验证）
3. `arch35` 的 PA 运行时精度验证需在 950 设备进行（pytest 的 4 个 PA 用例可直接复用）

### 复现方式

```bash
cd /opt/gjw/ops-transformer
rm -rf build
bash build.sh --pkg --soc=ascend950 --vendor_name=custom --ops=pool_key_indexer
# 观察: grep "error in backend" build 日志 → 214 处 fp8 标量语义错误
```
