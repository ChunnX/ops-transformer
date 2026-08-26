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

## 遗留问题记录：arch35（Ascend 950）编译失败（存量问题，已于 2026-08-25 修复，见下节）

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
- ✅ 移植 arch22 的 4 处 poolSize>1 行宽修复（2026-08-25，arch22 侧 2026-08-24 修复时 arch35 漏改，详见下方"0 池 batch 残留脏索引"与"values 输出 batch>0 错位"两节）：
  1. `CalcRunInfo`：`indiceOutCoreOffset` 批次基址改按 `idxOutStride`(ps>1 时为 outputLen) 行距——原按 sparseCount，ps>1 且 batch>0 时主输出整体错位、跨界覆盖相邻 batch；
  2. `DealActSeqLenIsZero`：清理偏移改按 `idxOutStride`——原按 sparseCount，ps>1 时 0 池/0 长度 batch 清理写错地址（目标行残留脏索引 + 跨行踩写）；
  3. `CleanInvalidOutput`：values 偏移增加行宽换算 `(offset/idxStride)*sparseCount`——原直接复用 indices 偏移；
  4. `ProcessInvalid`：indices 总量改按 `idxOutStride`，values 按自身 `sparseCount` 总量独立切分清理——原复用 indices 的 baseSize/dealSize，ps>1 时越出 values 张量边界（OOB 写）。
  注：修改后 950 编译仍被既有 VF fp8 问题阻塞；已做 stash 对照验证（错误签名与未修改前一致、无新增错误、报错均为 backend 阶段即 frontend 解析已通过），待 VF 修复后补编译+运行时验证。

### 待办（需 arch35 维护者处理）

1. ~~定位并修复 arch35 vector（`vf/` 目录）的 fp8 标量语义问题~~ → 已修复（见下节，根因不在 vf/ 目录）
2. ~~修好后补 950 的 PA 场景编译验证~~ → 已通过（3 个 dtype bin 均产出）
3. `arch35` 的 PA 运行时精度验证需在 950 设备进行（pytest 的 4 个 PA 用例可直接复用）

### 复现方式

```bash
cd /opt/gjw/ops-transformer
rm -rf build
bash build.sh --pkg --soc=ascend950 --vendor_name=custom --ops=pool_key_indexer
# 观察: grep "error in backend" build 日志 → 214 处 fp8 标量语义错误
```

---

# arch35（Ascend 950）编译失败修复记录（fp8 标量语义，两层根因）

> 2026-08-25 修复。此前 214 处 `error in backend: fp8...scalar float type semantics are not supported`（exit code 70）导致 950 编译全量失败。实际根因不在 `vf/` TopK 路径，而是模板分发与 W_T 类型推导两层问题，共 2 处代码修改。

## 根因 1：dtype 分发用普通 `if` 而非 `if constexpr`（影响所有 bin）

**位置**：`op_kernel/pool_key_indexer.cpp` arch35 分支（`__CCE_AICORE__ == 310`）。

**机理**：构建系统按输入 dtype 生成 3 个 bin（bf16/fp16/fp8），通过 `-DORIG_DTYPE_QUERY=<dtype>` 区分（见 `build/binary/ascend950/src/pool_key_indexer/PoolKeyIndexer.py` 的 `get_dtype_fmt_options`）。`ORIG_DTYPE_QUERY` 是编译期常量，但普通 `if` **不阻止模板实例化**——即使条件为假，`INVOKE_PKI_OP_IMPL(PoolKeyIndexerKernel, fp8_e4m3fn_t, ...)` 分支仍会完整实例化。于是每个 bin（含纯 bf16/fp16 bin）都背上 fp8 kernel 实例，其中 vector 侧 weights 路径的 fp8 标量语义触发 bisheng 后端拒绝（参考 CANN 内置 `ascend_anti_quant` arch35 实现的注释：fp8 类型仅支持指针操作，`DataCopyPadExtParams<T>` 等含 T 类型标量成员/参数的结构会实例化 fp8 标量语义）。

**修复**：`if` → `if constexpr`，每个 bin 只实例化自身 dtype 分支。修复后 214 处错误降为 80 处（仅剩 fp8 bin 自身失败）。

**证据**：`kv_quant_sparse_flash_attention.cpp`（含 fp8 量化路径的同类算子）arch35 分发正是 `if constexpr (ORIG_DTYPE_QUERY == ...)` 写法。

## 根因 2：FP8 模式 W_T 误绑 fp8_e4m3fn_t（weights 实为 FP16）

**位置**：`op_kernel/arch35/pool_key_indexer_service_vector.h` / `arch22/..._service_vector.h` 的 `PkiTypeTraits`。

**机理**：`W_T = PkiTypeTraits<Q_T, void>::weightsType = Q_T`。FP8 模式下 Q_T=fp8_e4m3fn_t，W_T 随之变为 fp8_e4m3fn_t。但按 op_host `def.cpp` 的 dtype 约束与 issue.md 接口定义：**quant_mode=0/1 时 q/k=FP8_E4M3FN 而 weights=FP16**。这同时是功能错误（fp16 数据被按 fp8 解释）与编译错误（vector 侧 weights 搬运/加载路径 `DataCopyPadExtParams<W_T>`、`MulWeightAndReduceSum` 的 `LoadAlign<W_T, DIST_UNPACK_B16>`/`Cast<float, W_T>`/`CreateMask<W_T>` 全部以 fp8 类型实例化，触发标量 fp8 后端限制）。

**修复**：为 `PkiTypeTraits` 增加 fp8 特化，FP8 模式下 W_T 绑定 half：

```cpp
template <>
struct PkiTypeTraits<fp8_e4m3fn_t> {
    using weightsType = half;
};
```

arch22/arch35 两个头文件同步添加（arch22 虽不实际实例化 fp8，保持同构防误用；`fp8_e4m3fn_t` 在非 fp8 平台是 `uint8_t` 别名，特化始终合法）。Q/K 的 fp8 数据仅经 Cube 侧 matmul 消费（指针操作，950 支持），vector 侧自此无任何 fp8 类型实例。

## 修复文件清单

| 文件 | 修改 |
|------|------|
| `op_kernel/pool_key_indexer.cpp` | arch35 dtype 分发 `if` → `if constexpr`（附根因注释） |
| `op_kernel/arch35/pool_key_indexer_service_vector.h` | `PkiTypeTraits` 增加 `fp8_e4m3fn_t → half` 特化 |
| `op_kernel/arch22/pool_key_indexer_service_vector.h` | 同上（同构防误用） |

## 验证

| 项目 | 结果 |
|------|------|
| `--soc=ascend950 --pkg` | exit=0，0 处后端错误 |
| 950 产出 bin | bf16（2b73...）/ fp16（403a...）/ fp8（8510...）三个 .o+.json 齐全 |
| 自解压包 | `cann-ops-transformer-custom_linux-x86_64.run` 生成成功 |
| `--soc=ascend910b --pkg` 回归 | exit=0，无回归 |

## 遗留

1. arch35 PA 场景**运行时**精度/功能验证仍需 950 设备（本机 910B3 无 950 环境）
2. FP8 反量化路径（quantMode=0/1）在 kernel 侧是否完整实现（q_descale/k_descale 消费链路）未在本次编译修复中审计，需 950 环境实测
3. 遗留事项（前文）：arch35 indices 通道 outputLen 行宽同族问题（`indiceOutCoreOffset` 用 sparseCount 行距，service_vector 按 outputLen 寻址）仍未修，建议 950 实测时一并处理

---

---

# values 输出 batch>0 错位修复记录（valueOutGm 复用 indices 行宽偏移）

> 2026-08-24 四轮排查。测试侧弃用"归一化比对"改直比原始输出后，values 参考通道暴露 kernel 真实缺陷：indices 全对、values 在 poolSize>1 且 batch>0 时整体错位。属于 Bug 1（indices 行宽）的同族遗漏——当时只修了 indices 通道，values 通道漏改。

## 问题现象

- 仅 `return_value=True` 且 `pool_size>1` 且 `batch>1` 的用例 values 错位：B=2 时约 50% 行失配、B=3 时 33%、B=4 时 30%（恰为 batch0 占比），`max_abs_err` 高达数百
- 实测 batch1 起的 values 全为 0（写偏后被相邻清理覆盖/未写到）；indices 通道 42 用例全部 100% 集合匹配
- 旧"归一化比对"用 golden 分数重排重建 values 再比对，恰好掩盖了该缺陷（重建值来自 golden 而非算子原始输出）

## 根因

`valueOutGm` 写入偏移直接复用了 **indices 通道的行基址偏移**，而两个输出张量行宽不同：

| 输出 | 行宽 | 说明 |
|------|------|------|
| sparseIndices | `outputLen = sparseCount*poolSize + poolSize-1`（ps>1）| ps=1 时退化等于 sparseCount |
| sparseValues | 恒为 `sparseCount` | 与 ps 无关 |

arch22 三处同族错误：
1. `CalcRunInfo`：只算 `indiceOutCoreOffset`（outputLen 行距）并塞进 `runInfo.indiceOutOffset`；`RunInfo.valueOutOffset` 字段声明了但从未赋值。`ProcessVec` 直写路径两处 `valueOutGm[info.indiceOutOffset + cuS1Idx*sparseCount + ...]`——batch>0 时基址按 outputLen 行距累计，行号越深偏得越远
2. `ProcessLD`（跨核归并输出）：`DataCopyPad(valueOutGm[outOffset], ...)` 与 indices 共用 slot 8 偏移（outputLen 行距）
3. `ProcessInvalid`（无任务全量清理）：values 清理复用 indices 的 `baseSize/dealSize`（idxOutStride 总量），ps>1 时越出 values 张量边界

对照 arch35（950）实现确认正确形态：其 `CalcRunInfo` 早已独立维护 `valueOutCoreOffset`（sparseCount 行距）并下发 `runInfo.valueOutOffset`，kernel 侧 `valueOutGm[info.valueOutOffset + ...]`。

## 修复（arch22，910b 实测路径）

| 文件 | 修改 |
|------|------|
| `arch22/pool_key_indexer_kernel.h` | ① 新增成员 `valueOutCoreOffset`；② `CalcRunInfo` 按 sparseCount 行距独立计算并赋值 `runInfo.valueOutOffset`；③ `ProcessInvalid` 的 values 清理按自身总大小（`batchSize*qSeqSize*kHeadNum*sparseCount`）独立切分，不再复用 indices 的 baseSize/dealSize |
| `arch22/pool_key_indexer_service_vector.h` | ① `ProcessVec` 两处直写 `valueOutGm[...]` 改用 `info.valueOutOffset`；② LD 参数块新增 slot 12（`valueOutOffset + cuS1Idx*sparseCount`）；③ `ProcessLD` 读取 slot 12 为 `valueOutOffsetLd`，values 搬出改用该偏移 |

LD 链上各核对同一 s1 行写的 slot 12 值一致（同 batch 同 cuS1Idx），与 slot 8 的读法完全同构，故读取位置沿用既有逻辑。slot 12 与 slot 8 在同一 `tmpiBuff` 块内写入（`CopyOut` 16 个参数），头/尾槽两种写入形态均覆盖。

## 验证

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| pytest 42 用例（indices 门禁） | 42/42 通过 | 42/42 通过（无回归） |
| values 参考比对（30 个 val=True 用例） | 多 batch ps>1 用例 30%~50% 行失配，max_err 60~307 | **全部 100% 行匹配，max_abs_err=0.000000** |
| `_dbg_rawval.py`（bsnd_bf16_ps8 B=2 失配复现点） | batch1 row0 max_err=23.88 | 逐值一致 |
| `boundary_causal_test.py`（2 场景×2 布局×t=0..15 全扫） | — | ALL PASS（含 values 100%） |
| ps=1 回归（outputLen==sparseCount，两偏移恒等） | — | 无影响 |

## 遗留

1. arch35（950）values 通道本就正确（独立 `valueOutOffset`），但其 **indices** 通道存在同族行宽问题（`indiceOutCoreOffset` 用 sparseCount 行距，而 service_vector 按 `outputLen_` 行寻址）——因 arch35 存量编译失败（fp8 后端 214 处，见前文遗留）且本机无 950 环境，未改动，待 arch35 维护者修复编译后一并处理
2. 本轮仅重编 ascend910b 包并安装验证；arch35 相关待办见前文

## 经验教训

1. **同一算子的两个输出张量若有不同行宽，所有"行基址/行距/清理"公式必须各建各的**——修复 Bug 1 时只按 grep `indiceOutGm` 修 indices 通道，漏掉了 `valueOutGm` 共用同一偏移变量的三处；正确做法是以"偏移变量 × 消费点"二维矩阵审计
2. **结构体字段声明了但无赋值点 = 静默失效**（`RunInfo.valueOutOffset`），与 Bug 2（poolTailK）同款模式，review 时对"声明未赋值"要专门扫一遍
3. 比对框架引入"重建/归一化"后处理会系统性掩盖被测通道的真实缺陷——被验证的必须是算子的原始输出，后处理只能用于诊断不能用于判定

---

# 标杆/算子 缩放系数缺失修复记录（1/sqrt(headDim)）

## 问题现象

`tests/pytest/pool_key_indexer_reference.py` docstring 声明"不做 1/sqrt(headDim) 缩放"，与算子文档
`docs/aclnnKpoolIndexer.md` 计算公式冲突：

$$S = Q \otimes K_{pool}^T \odot \frac{1}{\sqrt{headDim}}$$

标杆与算子（非量化路径）均未做缩放，双方"一致地错"——indices 集合比对（TopK 次序对正缩放不变）
无法暴露，values 数值通道与文档规格不符。

## 修复方案

缩放系数为正数，与 ReLU、多头加权求和数学可交换，统一在**聚合后**施加，不影响 TopK 次序。
系数由 host 侧预计算、经 TilingData 下发（与本仓 attention 算子 softmaxScale/scaleValue 惯例一致）。

| 文件 | 修改 |
|------|------|
| `tests/pytest/pool_key_indexer_reference.py` | `_compute_batch` 点积乘 `1/sqrt(d)`；docstring 同步修正 |
| `tests/atk/pool_key_indexer_golden.py` | `_cal_atten_per_batch` matmul 后除 `sqrt(head_dim)`（同族缺陷同步修复） |
| `op_host/pool_key_indexer_tiling.h` | TilingData 新增 `TILING_DATA_FIELD_DEF(float, qkScale)` |
| `op_host/pool_key_indexer_tiling.cpp` | `SetTilingData`: `set_qkScale(1.0f / std::sqrt(info.headDim))` |
| `op_kernel/pool_key_indexer_common.h` | `ConstInfo` 新增 `float qkScale = 1.0f` |
| `op_kernel/arch22/pool_key_indexer_kernel.h` | `InitTilingData` 读取 `tilingData->qkScale` |
| `op_kernel/arch35/pool_key_indexer_kernel.h` | 同上 |
| `op_kernel/arch22/pool_key_indexer_service_vector.h` | `ProcessVec` 分数拷贝处 `Adds(...,0.0)` → `Muls(...,qkScale)`（零额外指令） |
| `op_kernel/arch35/vf/pool_key_indexer_vector1.h` | `MulWeightAndReduceSum` 增加 scale 参数，聚合后、转 sortable key 前乘缩放 |
| `op_kernel/arch35/pool_key_indexer_service_vector.h` | 调用点传入 `constInfo_.qkScale` |
| `tests/ut/op_kernel/test_pool_key_indexer.cpp` | `SetTilingData` 补填 `qkScale`（新增 tiling 字段需同步 UT，否则 UT 读到未初始化值） |

附带修复（本轮验证中发现的比对工具缺陷）：`tests/pytest/result_compare.py` 的 `compare_values`
原实现只对**未过 allclose 的失败行**计算 `row_err`，通过行直接 `continue`，导致全通过时
`max_abs_err` 恒为 0.0（统计假象）。已改为通过行也记录真实误差。

## 验证（ascend910b， 42/42 用例通过）

| 检查项 | 结果 |
|--------|------|
| indices 门禁 | 42/42（正缩放不改变 TopK 次序，无回归） |
| values 数值通道（fp16/bf16 × BSND/TND/PA × causal/non-causal） | 100% 行匹配，真实 max_abs_err=1e-6~3e-6（ULP 级，源自累加顺序与缩放施加位置的浮点舍入差异） |
| 独立探针（/tmp 探针脚本直比） | max_abs_err=1.9e-6, max_rel_err=1.0e-4 |
| boundary_causal_test.py / simple_test.py | ALL PASS / PASSED |

注：TilingData 布局变更（新增字段）后，op_host 与 op_kernel 必须同包重编重装，否则旧 kernel 读新
tiling 会取到越界/错位值。

---

# pytest 用例覆盖度分析与缺口登记（2026-08-24）

统计对象：`tests/pytest/test_pool_key_indexer.py`，42 个参数化用例（当时全部通过）。

## 已覆盖维度

| 维度 | 覆盖情况 |
|------|----------|
| layout 组合 | 4/4 全覆盖：BSND/BSND(23)、BSND/PA(6)、TND/TND(9)、TND/PA(4) |
| dtype | fp16(35)、bf16(7) |
| mask_mode | 0(17)、3(25) |
| return_value | True(30)、False(12) |
| pool_size | 1(退化)、8、16、32、64、128(上限) |
| topk | 16、64、128、256、512、2048(建议值)、3072/4096/8192(全部大值特殊档) |
| sparseCount | 1(退化)、2、4、8、16、32、64、128、256、512、3072(>2048 大分档路径) |
| n1 | 1(最小)~64(上限)全档 |
| batch | 1/2/3/4 + 不等长 list |
| pool_tail_k | 随机 / 0 / 15 / 127(ps-1) / 混合[0,4,7] |
| PA 特性 | 不等池数(-1 空槽+尾块 padding)、block_size 16/32/64、B=4、0 轴非连续 key(2 例) |
| kernel 关键分支 | splitMFlag(n1=64 且 sc≤2048)、actS1Size≤4 缓存排序、isSparseCountOver2K |

## 缺口登记（按风险排序）

### P0 — kernel 有专门分支但 pytest 从未执行

1. **BSND + actual_seq_q 部分有效**： 所有 BSND/BSND 用例 `actual_seq_q=None`。
   `needDealActS1LessThanS1` → `CleanInvalidOutput`（无效 S1 行输出 -1）路径零覆盖
   （arch22 `pool_key_indexer_service_vector.h` 的 BSND 无效 S1 清理段）。
2. **BSND/BSND + actual_seq_k 每 batch 池数不等**： 仅 PA 场景传了 actual_seq_k，
   密集布局的池截断寻址路径未验证。
3. **S1 > K + causal**： 无任何用例（仅 1 例 S1==K）。前段行 `cuRealAcSeq≤0` →
   全 -1 输出路径未覆盖。

### P1 — 组合缺口

4. **bf16 × BSND/PA_BBND**： 7 个 bf16 用例分布在 BSND/BSND、TND/TND、TND/PA，
   无 BSND/PA 组合。
5. **topk 特殊值 5120/6144/7168 未测**（其中 ps≥32 且 topk≥6144 有已知 arch22 崩陷缺陷，
   已在 TEST_CASES 注释登记为不执行，见下）。
6. **pool_size 非幂数**（2/3/4/5/127）： 只测了 1 和 2 的幂——非对齐 pool_size 走
   `Align(poolSize,8)` 掩码路径。

### P2 — 低风险

7. PA × ps=1（退化）、PA × 大 topk(>2048)。
8. actual_seq 为 0 的空 batch（`DealActSeqLenIsZero`）。
9. PA B=1、PA tail 确定性值。
10. topk=1（最小值）。

### N/A（当前硬件不可测）

- quantMode 0/1（FP8/mxFP8）： 仅 Ascend 950 支持，当前 910B3 环境无法覆盖。

### 已知缺陷（关联）

- arch22/910B： pool_size≥32 且 topk≥6144 组合触发 aicore exception（EE9999/507015），
  崩溃污染进程内后续用例，故不作为 xfail 纳入（TEST_CASES 内 NOTE 登记），kernel 修复后
  可补：`tnd_fp16_ps32_topk6144_causal`、`bsnd_fp16_ps64_topk8192_causal`。

---

---

# arch35（Ascend 950）运行时修复记录（2026-08-25，首次 950 上板验证）

> 环境勘误：**本机实为 Ascend950PR**（npu-smi 显示 8×950PR，`torch_npu.get_device_properties` = Ascend950PR_9579，28 cube/56 vector）。此前文档中"本机 910B3 无 950 环境"的记录已过时——arch35 修复首次可在本机做**运行时验证**，torch_extension 亦按 README 流程编译安装（python3.10 + build wheel + pip install，protobuf 依赖经华为镜像补装）。

## 问题总览

arch35 编译修复（见前文）后首次上板：pytest **6/42**（恰为 5 个 ps=1 + 1 个退化场景），
36 个 ps>1 用例全挂。经探针二分定位**四层独立缺陷**，全部修复后 **42/42**。

## Bug 1：arch35 展开路径 V/S 流水同步缺失（kernel，根因一）

**现象**：pool 集合含重复/越界序号（如 8 池场景选出"pool 9"）、每池第 16 lane 恒为垃圾值、
首调输出随 UB 残留内容漂移（重复调用趋稳）。

**根因**：arch35（950）上标量访存（`GetValue`/`SetValue`，S pipe）与向量操作（V pipe）、
MTE3 间**必须显式硬同步**，`PipeBarrier<PIPE_V>` 不保证 S 侧顺序。`ExpandAndAppendIndices` 中：
1. `poolIndices.GetValue(k)` 读 `indicesOutLocal_`，其最近写入方为向量操作
   （topkOp_ 结尾 `DataCopy` / `CreateVecIndex` / -1 `Duplicate` 填充）→ **缺 V→S 同步**，
   标量读到陈旧/垃圾 pool 序号；
2. `offsetTpl.SetValue(p,p)` 后向量 `Add` 读模板 → **缺 S→V 同步**，部分 lane 读陈旧模板
   （实测每池末 lane 恒加同一垃圾值）；
3. 尾块 `SetValue` 后由 MTE3 `DataCopyPad` 搬出 → **缺 S→MTE3 同步**。

同仓 arch35 权威参照：`bsa_select_block_mask/op_kernel/arch35/bsa_radix_topk_service.h` 的
`VToSSync/SToVSync/SToMTE3Sync`（`FetchEventID(HardEvent::V_S/S_V/S_MTE3)` + Set/WaitFlag）。

**修复**（`arch35/pool_key_indexer_service_vector.h`）：新增三个同步辅助函数并按上述
三处补齐。

## Bug 2：arch35 展开轮数未截断 + 尾块对齐写越界（kernel，根因二，arch22 同族存量）

与 arch22 已修复的 Bug 1/Bug 3 完全同族（沿抄 LIV2 公式未核对行宽前提）：
1. 展开循环 `k < validS2Len` 未按 `sparseCount` 截断——可见池多于选中池时越写
   outputLen 之外污染尾区/相邻行（改为 `Min(validS2Len, sparseCount)`，对齐 arch22）；
2. 尾块按 `Align(visibleTailK,8)` 对齐写会越出尾区容量（ps-1）——改为"对齐清空尾区 +
   标量精确写 visibleTailK 个"（对齐 arch22 修复），且清空 `Duplicate` 与标量写之间需
   V→S 同步（迟到的 V 写会覆盖标量写，实测尾区全 -1）。

## Bug 3：host tiling TND/PA 的 s2Size=0（host，根因三，本 bug 仅 arch35 显形）

**定位**：BSND/TND 同一张量 A/B 对照（BSND 32/32、TND 1/32）+ tiling 调试日志，抓到
`TND: s2Size=0`。

**根因**：tiling 的 TND/PA key 分支写 `info.s2Size = 0`（注释"determined from
actual_seq_k"），但**无任何后续赋值**。arch22 不受影响（kSeqSize 仅用作缺省 seq len 与
BSND 前缀），arch35 却用 `kSeqSize` 作 **scoreGm 行距与每核区域大小**：

```
kSeqSizeAlign = Align(kSeqSize, s2BaseSize)  →  0
scoreGm 行距 = 0   →  所有 S1 行的分数互相覆盖（同一 gS1 块 4 行共享同一分数）
singleCoreScoreSize = 0  →  各核 score 区域重叠
```

症状：同一 gS1 块内所有行输出同一 pool 集合（TND/PA 全挂、BSND 全过的分水岭）。

**修复**（参照 LIV2 `GetS2Size`/`GetS2SizeForPageAttention` 权威语义）：
- TND：`s2Size = key.shape.dim(0)`（T2 = 累计 pool 总数）；
- PA：`s2Size = maxBlockNumPerBatch × blockSize`（block_table.dim(1) × key.blockSize，
  每 batch 最大 pool 数上界）。

## Bug 4：host workspace 未覆盖 arch35 score 区域（host，随 Bug 3 修复暴露）

arch35 kernel score 区域 = `GetBlockNum()(=aicNum) × s1BaseSize(4|2) × Align(s2Size,128) × 4B`，
与 arch22 布局公式完全不同（arch35 无 LD 区域）。原 host 公式仅按 arch22 布局计算，
`CalcWorkspaceSize` 取 `max(arch22 公式, arch35 公式)` 覆盖两代（TND/PA 的 s2Size 为
累计池数/块槽上界，可能大于 arch22 场景需求）。

## 附带修复：FP8 反量化链路 fail-fast 守卫（host）

审计确认 kernel 侧（arch35）**完全不消费 q_descale/k_descale**（Mmad 直接用 fp8 裸数据，
缺 dq/dk 缩放），quant_mode=0/1 调用会静默产出错误分数。在 `ParseAndCheckParams` 增加
显式拒绝（kernel 侧反量化实现并经验证前不放开），避免"返回错误结果"而非报错。

## 修复文件清单

| 文件 | 修改 |
|------|------|
| `op_kernel/arch35/pool_key_indexer_kernel.h` | `CalcRunInfo`/`DealActSeqLenIsZero`/`ProcessInvalid` 的 indices 行宽改 outputLen（outputLen 行距）、values 独立行距/切分清理（arch22 同族修复，见前轮记录） |
| `op_kernel/arch35/pool_key_indexer_service_vector.h` | `CleanInvalidOutput` values 偏移换算；`ExpandAndAppendIndices` 重写：V→S/S→V/S→MTE3/V→S(尾块) 四处同步 + 展开轮数 Min 截断 + 尾块精确写；新增 `VToSSync/SToVSync/SToMTE3Sync` |
| `op_host/pool_key_indexer_tiling.cpp` | TND/PA `s2Size` 正确赋值（T2 / maxBlockNum×blockSize）；workspace 取两代公式 max；FP8 fail-fast 守卫 |
| `op_kernel/arch35/pool_key_indexer_service_vector.h` + `arch22/...` | `PkiTypeTraits<fp8_e4m3fn_t>` 特化 → weights 绑 half（编译期修复，见前轮） |

## 验证（Ascend950PR，本机实测）

| 项目 | 结果 |
|------|------|
| pytest 42 用例 | **42/42 通过**（修复前 6/42） |
| 稳定性重复 ×3 | 42/42 ×3 |
| `boundary_causal_test.py`（2 场景×2 布局×t=0..15 全扫） | ALL PASS |
| `simple_test.py` | PASSED |
| 探针矩阵（ps∈{1,8,16} × S2∈{8..300} × topk） | 全部通过 |
| BSND/TND 同张量 A/B | 双 32/32 |
| `--soc=ascend910b --pkg` 编译回归 | exit=0 |
| `--soc=ascend950 --pkg` 编译 | exit=0 |

## 遗留

1. FP8 反量化（quantMode=0/1）kernel 实现仍缺失（host 已 fail-fast 拒绝，不会静默出错）；
   实现时需在 cube 侧或 Mmad 前挂 dequant，参照 CANN `ascend_anti_quant` arch35 的
   regbase 指针路径规避 fp8 标量后端限制
2. PA 0 轴非连续（keyStride0 运行时 stride 分支）运行时验证未覆盖（pytest 无该 dtype 场景组合）
3. 此前登记的 arch22 ps≥32 且 topk≥6144 崩陷缺陷不变（950 上 42 例不含该组合）

## 经验教训

1. **"本机环境"记录会过时**：文档长期记载"本机 910B3 无 950"，实际机器已换 8×950PR——
   修复前先用 `npu-smi info`/`get_device_properties` 复核环境，勿信历史记录
2. **V/S 流水同步是 arch35 与 arch22 的本质差异**：arch22 上 `PipeBarrier<PIPE_V>` 后跟标量
   读写能工作纯属时序侥幸；跨代移植（arch22→arch35）时所有 `GetValue/SetValue` 与向量/MTE
   交错点必须逐个补 `HardEvent::V_S/S_V/S_MTE3` 硬同步（仓内权威参照 bsa_select_block_mask）
3. **"占位 0 待运行时确定"的 tiling 字段=定时炸弹**：s2Size=0 在 arch22 无害（没人用），
   arch35 新增消费点后爆雷——新增 tiling 字段消费侧时，必须反查所有 layout 分支的该字段
   赋值路径
4. A/B 对照（同一物理张量、两种 layout 属性调用）是隔离"数据问题 vs 路径问题"的最快手段；
   tiling 调试日志（临时 OP_LOGI + ASCEND_SLOG_PRINT_TO_STDOUT=1）一击定位 s2Size=0

---
---

# 已知缺陷场景测试纳入与修复记录（缺陷 C/D/E，arch35/Ascend 950PR）

> 2026-08-26 排查修复。环境：8×Ascend950PR（arch35，28 cube/56 vector），CANN 9.1.0-beta.3。
> 背景：`test_pool_key_indexer_general.py` 曾将三类崩溃组合主动排除出测试（2026-08-24
> 在 arch22/910B3 实测，EE9999/507015 且崩溃污染进程内后续用例）。本次按缺陷家族
> 逐个重新纳入（新增 `test_pool_key_indexer_known_defects.py` + `known_defect_cases.py`
> + `_defect_case_runner.py`），在 arch35 上逐个复现、定位与修复；arch22 侧由对应
> 责任人处理（文末附修改建议）。

## 测试架构：进程隔离

EE9999/aicore exception 会污染进程内后续用例（异步报错、进程级 device context 失效），
historically 因此连 xfail 都未纳入。本次方案：

- `known_defect_cases.py`：缺陷用例注册表，三家族 26 例（C:pool_size 非 8 倍数 11 例、
  D:topk 5120/6144/7168 9 例、E:ps≥32 且 topk≥6144 5 例，每家族含 canary 对照例）
- `_defect_case_runner.py`：单用例 runner，独立子进程执行；输出协议
  `DEFECT_CASE_RESULT: PASS|FAIL|CRASH <id> <detail>`，退出码 0/1/2
- `test_pool_key_indexer_known_defects.py`：pytest 封装，`subprocess.run` 逐例拉起
  （600s 超时兜底），按协议行+退出码判定；子进程被信号杀死/无协议行亦兜底判 CRASH

崩溃只污染子进程，套件可持续执行并给出逐例判定。比对策略与主套件一致
（indices 逐行有效索引集合比对 100% 门禁 + values 参考日志）。

## 缺陷 C：pool_size 非 8 的倍数崩溃（arch35 复现并修复）

### 复现（修复前，arch35）

11 例全部 CRASH（507015），与 arch22 表现一致：ps∈{2,3,4,5,7,12,100,127} ×
BSND/TND/PA 全崩；canary（ps=8 同 shape）通过。

### 定位：判别实验锁定向量操作数非 32B 对齐

**假设**：arch35 向量指令（Duplicate/Add 等）要求操作数地址 32B（int32 即 8 元素）
对齐；`ExpandAndAppendIndices`（`arch35/pool_key_indexer_service_vector.h`）中三处
向量写的 dst 偏移在 ps%8≠0 时非 8 元素对齐：

| 写点 | 偏移 | ps%8≠0 时 |
|------|------|-----------|
| 展开轮 Duplicate/Add | `outOff = k*poolSize`（k≥1） | 非对齐 |
| 池不足补 -1 Duplicate | `validExpand = validS2Len*poolSize` | 非对齐 |
| 尾区清空 Duplicate | `tailPos = topk = sparseCount*poolSize` | 可能非对齐 |

**判别实验**（BSND, s2pools=64, nocausal，逐个独立进程）：

| 用例 | 形态 | 预测 | 实测 |
|------|------|------|------|
| E: ps=16 | 对照（全对齐） | 不崩 | 不崩, indices 对 |
| A: ps=5, topk=5(sc=1), tail=0 | 唯一无非对齐写形态（仅 offset 0 对齐写） | 不崩 | **不崩**（但 indices 错，见下） |
| B: ps=5, topk=5, tail=3 | 尾区 Duplicate 在 offset 5 | 崩 | **崩** |
| C: ps=5, topk=10(sc=2), tail=0 | 第 2 轮展开在 offset 5 | 崩 | **崩** |
| D: ps=5, topk=40(sc=8), tail=0 | 多处非对齐 | 崩 | **崩** |

预测全部命中 → 根因确证。ps=1 例外成立的原因：`ProcessTopK` 仅在 `poolSize_>1`
时调用展开；ps%8==0 时上述偏移天然 8 对齐——原向量路径本就隐含"ps 为 8 倍数"前提。

**附带发现**（探针 A）：即使无崩溃，向量路径按 `Align(poolSize,8)` 宽度写会越出
pool 边界（ps=5 写 8 个），无尾块清理时 [topk, topk+Align(ps,8)-ps) 残留
`base+p` 垃圾索引（正值、非 -1），集合比对失败——与崩溃同源的第二类缺陷。

### 修复（`op_kernel/arch35/pool_key_indexer_service_vector.h`）

`ExpandAndAppendIndices` 增加非对齐 ps 的标量退化路径，`poolSize % 8 != 0` 时：

1. 向量铺 -1：`Duplicate(tokenIndices, -1, alignedTotalOut)`（offset 0，恒对齐）
2. `VToSSync()`：V 写（-1 填充 + poolIndices 的 TopK 向量写）对标量读写可见
3. 标量精确写展开：`tokenIndices.SetValue(k*ps + p, base + p)`（逐 token，无对齐
   要求，天然精确到 pool 边界——同时消除垃圾残留问题）
4. 标量精确写尾块 `visibleTailK` 个（visibleTailK 计算提取为两路径共用）
5. `SToMTE3Sync()`：S 写由 MTE3(DataCopyPad) 读出前的硬同步（外层 ProcessTopK
   的 V_MTE3 等待不覆盖 S 写）

`poolSize % 8 == 0` 主流路径（ps=8/16/.../128）保持原向量实现不变（性能无损）。
标量路径仅作用于非 8 倍数边缘场景（≤ topk 个标量写/行），且规格 topk 值集与
ps%8≠0 组合下 sparseCount 恒 ≤2048，不触碰 isSparseCountOver2K 路径。

### 验证

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| 缺陷 C 家族 11 例（runner 直跑） | 10 CRASH + canary PASS | **11/11 PASS** |
| `test_pool_key_indexer_known_defects.py` 全套件 | — | **26/26 passed**（含 D/E 家族与清单校验） |
| 定点 42 例 + 泛化 60 例 + 覆盖率收口回归 | 103 passed | **103 passed**（无回归） |
| `boundary_causal_test.py`（ps=16 因果边界 t=0..15 全扫） | ALL PASS | ALL PASS |
| `--soc=ascend950 --pkg` 编译 | — | exit=0，3 个 dtype bin 齐全 |

## 缺陷 D：topk ∈ {5120, 6144, 7168}（arch35 不复现）

9 例（含 arch22 实测崩溃点 ps=16/topk=5120×s2p∈{384,512}、正常点 s2p=640 对照、
ps=1/ps=8 对照、TND 形态、topk=6144/7168）在 arch35 **全部 PASS，修复前后一致**
（本缺陷无需 arch35 修复）。

原因分析：arch22 的 TopK 走 LD 跨核归并路径（vec1Res/vec1Param workspace 三段布局、
`ProcessLD` 按 BASE_TOPK=2048 分档），崩溃与其 LD 区域 sizing/分档逻辑相关；arch35
为 VF 直通路径（`vf/pool_key_indexer_topk.h` LITopk 直方图 radix TopK，scoreGm 单段），
无 LD 路径，不存在该缺陷面。用例保留作 arch35 回归保护。

## 缺陷 E：pool_size≥32 且 topk≥6144（arch35 不复现）

5 例（含原始崩溃形态 TND ps32/topk6144 k=16384、BSND ps64/topk8192 k=16384，及
ps32/6144、ps64/8192、ps128/8192 变体）在 arch35 **全部 PASS，修复前后一致**。
同缺陷 D，属 arch22 LD 路径特有。用例保留作 arch35 回归保护。

## 修复文件清单

| 文件 | 修改 |
|------|------|
| `op_kernel/arch35/pool_key_indexer_service_vector.h` | `ExpandAndAppendIndices`：visibleTailK 计算前移共用；新增 ps%8≠0 标量精确写退化路径（Duplicate 铺 -1 + VToSSync + 标量展开/尾块 + SToMTE3Sync）；ps%8==0 向量路径不变 |
| `tests/pytest/known_defect_cases.py` | 新增：缺陷 C/D/E 三家族 26 例注册表（含 canary 对照与 arch22 崩溃形态复刻） |
| `tests/pytest/_defect_case_runner.py` | 新增：子进程单用例 runner（输入构造 BSND/TND/PA + golden 比对 + PASS/FAIL/CRASH 输出协议） |
| `tests/pytest/test_pool_key_indexer_known_defects.py` | 新增：pytest 封装（subprocess 逐例拉起、超时兜底、用例清单校验） |
| `tests/pytest/test_pool_key_indexer.py` | NOTE/docstring 更新：缺陷 E 的 arch35 复测结论与用例去向 |
| `tests/pytest/test_pool_key_indexer_general.py` | docstring C/D/E/H 项与分析报告打印更新为 arch35 现状 |

## arch22（910B）修改建议（待责任人处理，本次未改动）

1. **缺陷 C 同款根因确认**：`arch22/pool_key_indexer_service_vector.h`
   `ExpandAndAppendIndices` 的 302-306 行（展开轮 Duplicate/Add 于 `tokenIndices[outOff]`，
   outOff=k*poolSize）、313 行（`tokenIndices[validExpand]` 补 -1）、333 行
   （`tokenIndices[tailPos]` 尾区清空）与 arch35 修复前逐字同构——arch22 的
   ps∈[2,7]/非 8 倍数崩溃应同为向量操作数非 32B 对齐。建议按本次 arch35 修复
   同模式移植标量退化路径，差异点：arch22 无需 VToSSync（其 V→S 顺序由时序保证，
   与既有 tail 标量写一致仅 PipeBarrier 即可）、S→MTE3 沿用其
   `SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3)` 惯用法
2. **缺陷 D/E 建议排查方向**（arch35 无 LD 路径，无参照，仅给线索）：
   崩溃面与 s2pool 相对 `topkCountAlign256_=Align(sparseCount,256)` 的位置相关
   （实测 s2p∈{384,512} 崩、640 正常，ps=16/topk=5120 时 sparseCount=320、
   Align256=512）；建议在 910B 上审计 `ProcessLD`/`CalcWorkspaceSize` 的 LD 区域
   在 `validS2Len < topkCountAlign256_` 与 `topkCount_ > trunkLen_` 分档组合下的
   越界（对照 mssanitizer MEMORY_RECORD 边界法，见前文 workspace 排查记录）；
   `tiling.cpp` 的 `GetTrunkLen`（按 sparseCount 分档 16K/12K/8K/4K/2K）与
   arch22 kernel 侧实际消费的 trunkLen 是否镜像一致也需复核
3. 修复后可直接复用 `test_pool_key_indexer_known_defects.py` 三件套在 910B 上回归
   （runner 支持 `--device` 参数；用例注册表与平台无关）

## 遗留

1. values 参考通道在 950 上存在数值差异（集合比对下 valid count 一致、max_abs_err
   量级 1e-3~8，rtol=1e-3 门限下多行失配；基线亦如此，非本次缺陷修复引入）。
   疑与 arch35 分数通道累加顺序/sortable key 转换精度相关，建议后续独立排查
   （比对策略本就不以 values 为门禁，不影响精度结论）
2. 缺陷 C 修复后，泛化随机层的 `POOL_SIZES_VALID` 采样包络保持 8 倍数不变
   （基准稳定性），非 8 倍数 pool_size 由 known_defects 套件覆盖；后续可考虑
   将非 8 倍数纳入随机采样扩大泛化面
3. FP8 反量化（quantMode=0/1）kernel 实现仍缺失（host fail-fast，见前文）

## 经验教训

1. **"实测仅 {1}∪{8 的倍数} 可用"这类现象背后往往是硬性对齐约束**：8 元素=32B
   正是 int32 向量操作数对齐粒度。遇到"参数值域按 8 的倍数可用"的崩溃面，
   第一反应应查向量指令操作数地址对齐，而非业务逻辑
2. **判别实验设计**：构造"假设成立时唯一不触发问题"的形态（sc=1+tail=0 →
   仅 offset 0 对齐写）与"假设成立时必触发"的形态（tail 在非对齐 offset），
   一次实验同时拿到正反两面证据，比逐点注释代码二分更快
3. **进程隔离是把"崩溃型缺陷"纳入 CI 的唯一可行路径**：xfail 无法处理进程级
   污染（EE9999 毒化后续用例），子进程 + 输出协议 + 退出码三重判定既保住逐例
   报告又防雪崩
4. 崩溃与数据错误可能同源（非对齐写崩溃 + 对齐宽度越界垃圾残留）：修复时按
   根因统一处理（标量精确写一举消除两类问题），不要分开打补丁

---

# 展开模板一次性构建优化（ps%8==0 主路径去逐行标量）

> 2026-08-26 实施。缺陷 C 修复评审时识别的性能点：ps%8==0 主路径的展开主体
> 已是 SIMD（每池 Duplicate 广播 + Add 加模板），但模板 `offsetTpl`(0..ps-1)
> 此前在 `ExpandAndAppendIndices` 内**每行**用 ps 次 `SetValue` 标量重建，
> 并伴随一次 `SToVSync` 硬同步。

## 修改（`op_kernel/arch35/pool_key_indexer_service_vector.h`）

1. `InitBuffers`：`poolSize_>1` 时模板经 `CreateVecIndex(workLocal_, 0, ps)`
   一次性构建并常驻 `workLocal_` 头部（kernel 生命周期内不变，ps 是 tiling 常量）
2. `ExpandAndAppendIndices` 向量路径：删除每行模板标量重建与 `SToVSync`，
   直接消费常驻模板

依据：`CreateVecIndex` Level-2(count 模式) 在 dav_3510 支持 int32、V pipe 写出
（`ProcessTopK` 既有用法 605/677 行同款）；模板写(InitBuffers)与消费(Add)同在
V pipe，中间隔多次向量操作与跨核同步，同 pipe 有序性足够（与既有
CreateVecIndex→PipeBarrier→向量消费的依赖模式一致）。ps%8!=0 标量退化路径
不使用模板，不受影响；workLocal_ 除模板外无其他写方，常驻安全。

## 性能（Atlas 950PR，BSND/fp16/causal/val=True，n=30 中位，共享机器）

| 配置 | 优化前 | 优化后 | 中位提升 |
|------|--------|--------|----------|
| ps=16, topk=2048, S1=64, S2pool=512 | 0.087 ms | 0.070 ms | ~20% |
| ps=64, topk=2048, S1=32, S2pool=128 | 0.081 ms | 0.062 ms | ~23% |
| ps=128, topk=8192, S1=16, S2pool=128 | 0.084 ms | 0.060 ms | ~29% |

（min 值同步下降 0.077→0.052~0.062，非负载噪声；探针：
`tests/pytest/_perf_probe.py`）

## 残留标量与后续优化方向

向量路径每行仍残留：① 逐池 `GetValue`（sparseCount 次，Duplicate 广播源的
固有成本）② 尾块标量写（≤ps-1 次）。二者与 ps%8!=0 路径的完全向量化可由
Gather 方案统一解决：`out[i] = i + ps*(poolIdx[i/ps] - i/ps)`，逐行
CreateVecIndex(iota) + Gather(sel, poolIdx, kTpl) + Sub/Muls/Add 共 ~5 条向量
指令（kTpl=i/ps 模板同样可一次性构建；Gather Level-2 已确认 dav_3510 支持
int32）。代价为 +1 个 outputLen 长度 UB 模板（≤32KB）。待非 8 倍数 ps 成为
真实业务配置或逐池 GetValue 成为瓶颈时实施。

## 回归

| 项目 | 结果 |
|------|------|
| 定点 42 + 泛化 60 + 覆盖率收口 | 103 passed（耗时 40s→15.6s） |
| known_defects 26 例（ps%8==0 向量路径 + ps%8!=0 标量路径双覆盖） | 26 passed |
| boundary_causal_test.py（ps=16, t=0..15 全扫） | ALL PASS |

## arch22 同步建议

`arch22/pool_key_indexer_service_vector.h` 的同款逐行模板构建（~291-294 行）
可按此模式同步（arch22 的 CreateVecIndex 用法已在 ProcessVec/ProcessLD 存在，
确认其在 910B 的 pipe 语义后移植；arch22 无需 VToSSync 的前提不变）。

---

# GE 图模式（TorchAir）打通与 arch22 尾块流水线竞态修复（2026-08-26）

> 2026-08-26 实施。目标：使 pool_key_indexer 支持 TorchAir（GE 图）模式调用，
> 并修复排障过程中发现的 arch22 kernel 尾块追加流水线竞态。全程 Atlas A2
> 910B3 验证。

## 一、GE 图模式打通（3 个缺失环节）

### 环节 1：TilingParse 未注册 → 图编译必挂

**现象**：torchair `torch.compile(fullgraph=True)` 图编译失败，表层报错被吞
（`Tiling func of op_type PoolKeyIndexer failed`），真实报错需
`ASCEND_SLOG_PRINT_TO_STDOUT=1` 捞取：

```
[auto_tiling_rt2.cc:120][OP_PROTO][ParseAutoTilingRun] OpName:[PoolKeyIndexer]
"compile info not contain [_pattern]"
```

**根因**：图编译期 FE（`TbeOpTilingPyInterfaceNew` → `ParseAutoTilingRun`）
需从算子 compile info JSON 解析 `_pattern` 等字段，该 JSON 只有注册了
`TilingParse` 回调才会由 te 编译器生成。此前 `IMPL_OP_OPTILING(PoolKeyIndexer)`
只注册了 `.Tiling()`，eager/aclnn 路径不走此解析所以从未暴露。

**修复**（`op_host/pool_key_indexer_tiling.h/.cpp`，参照仓内
`grouped_matmul_finalize_routing_tiling.cpp:118` 范式）：

1. `tiling.h`：填充 `PoolKeyIndexerCompileInfo`（socVersion/npuArch/aicNum/
   aivNum/ubSize/l1Size），include `tiling_parse_context.h`
2. `tiling.cpp`：新增 `TilingPrepareForPoolKeyIndexer` 回调（写平台信息进
   CompileInfo），注册改为 `.Tiling(...).TilingParse<PoolKeyIndexerCompileInfo>(...)`

### 环节 2：图模式下 tiling 值校验硬报错 → TND/PA 编译失败

**现象**：打通环节 1 后，BSND 图模式可编译，但 TND/PA 布局图编译报
`"TND: actual_seq_q data is null, tiling sink not supported"`。

**根因**：图模式下 `actual_seq_q/k/pool_tail_k` 是 device tensor，tiling 期
（编译期）数值不可见（`GetOptionalInputTensor()->GetData<int64_t>()` 返回
nullptr）。原 tiling 对 data==null 硬报错（假定走 tiling sink，本算子未实现）。
kernel 侧本就运行期从 GM 读值（`GetActualSeqLen`/`poolTailKGm_.GetValue`），
host 侧值校验只是防线，跳过不影响功能正确性。

**修复**（`tiling.cpp` TND/PA 值校验分支）：data==null 时降级为跳过值校验
（打 INFO 日志），B 值改由 shape（元素数）推导；eager 路径 data 非 null 时
保持原校验不变。

### 环节 3：converter 缺 key_stride0 → PA 非连续图模式失效

**修复**（`torch_extension/.../graph_convert/graph_convert_pool_key_indexer.py`）：
converter 补 `key_stride0` attr 下传（转换期从 tensor 元数据读 stride(0)，
读不到回退 -1），与 eager 路径 csrc 直读 `at::Tensor::stride(0)` 对齐；属性
顺序按 op_def 注册序（key_stride0 为第 8 个属性）。

## 二、arch22 尾块追加流水线竞态（mask=0 且池数<sparseCount 时 tail 丢失）

### 现象与定位

GE 泛化测试 60 例中 5 例失败（eager 同参数全过）。DIAG 日志证明 eager 与
图模式 tiling 完全一致（tilingKey=196865/aic=20），失败精确锁定在唯一让
"-1 清理分支（validExpand<topk）+ 尾块写入分支"同行先后执行的组合：

| 组合 | 清理分支 | 尾块写入 | GE 图模式 |
|------|:---:|:---:|:---:|
| mask=3（大 topk） | 执行 | 不执行 | ✓ |
| mask=3（小 topk，pools≥sparse） | 不执行 | 执行 | ✓ |
| mask=0 + pools≥sparse | 不执行 | 执行 | ✓ |
| **mask=0 + pools<sparse** | **执行** | **执行** | **✗（~75% 行 tail 丢失）** |

排除项（均有实验证据）：tiling 不同（证伪）、kernel bin 不同（证伪）、
图模式读不到 pool_tail_k 值（mask=3+topk=64 实验 visibleTailK 正确，证伪）。

### 根因：arch22 缺失 arch35 已有的同步纪律

`arch22/pool_key_indexer_service_vector.h` 的 `ExpandAndAppendIndices`
（对照 arch35 同函数 :396-407，**arch35 注释明言"否则迟到的 V 写会覆盖标量
写(实测尾区全 -1)"——同症状已修复过，arch22 未同步**）：

1. **V→S 竞态**：尾区 `Duplicate(-1)`（V 流水异步）与紧随的标量 `SetValue`
   写尾 token 之间仅 `PipeBarrier<PIPE_V>`（只保证 V 内部有序）。S 写先发射、
   V 写后落地时尾值被 -1 覆盖。
2. **行间 MTE3→V 竞态**：行循环每行 `DataCopyPad`（MTE3 读 expandOutLocal_）
   后仅惰性 `SetWaitFlag<MTE3_V>` 登记（不构成等待）；下一行 Expand 开头的
   `Duplicate(-1)`（V 写）覆写同一 UB。arch35 行循环头有显式
   `WaitFlag<MTE3_V>`（TOPK_MTE3_V_EVENT），arch22 没有。

eager 与 GE 只是**不同 launch 时序执行同一 kernel**：aclnn 单算子下发与整图
紧凑下发的 GM 访问延迟/跨核握手时序不同，恰好把 S 发射推进/推出 V/S 交叠
窗口——eager 被时序掩盖，GE 稳定踩中（约 96/128 行，run 间波动是时序竞态
签名）。

### 修复（对齐 arch35 同步纪律，`arch22/pool_key_indexer_service_vector.h`）

| # | 位置 | 修复 |
|---|------|------|
| 1 | 展开段 | `poolIndices.GetValue`（S 读）前加 `VToSSync`；offsetTpl 标量写后加 `SToVSync` |
| 2 | 尾块段 | 尾区 `Duplicate(-1)` 与标量 `SetValue` 之间加 `VToSSync`（tail 丢失主因）；尾部惰性 `SetWaitFlag<S_MTE3>` 升级为 `SToMTE3Sync` 真等待 |
| 3 | 行循环 3 个调用点（ProcessVec 1 处 + ProcessLD 2 处） | 每行 Expand 前加 MTE3→V 真等待（SetFlag/WaitFlag 自旋对，arch35 CleanInvalidOutput 同款写法） |

新增 `VToSSync/SToVSync/SToMTE3Sync` helpers（FetchEventID 动态分配）。
同步只增不减，eager 行为不受影响（回归证实）。

### 排障环境陷阱：GE 编译缓存

排障中途"修复无效"（改 kernel 后 GE 行为不变、错误行集合恒定）的元凶是
**GE 图编译命中 `/root/atc_data/kernel_cache/Ascend910B3/` 旧编译缓存**
（te_poolkeyindexer_* 共 216 条），重建 vendor 包不影响该缓存。清缓存后
竞态组合立即 20/20 全对。

**经验**：后续修改 pool_key_indexer kernel 并验证 GE 图模式时，必须先执行

```bash
rm -rf /root/atc_data/kernel_cache/Ascend910B3/te_poolkeyindexer_*
```

否则图模式跑的是旧 bin，结论无效。

## 三、新增测试

`tests/pytest/test_pool_key_indexer_ge_graph.py`（新增文件，GE 图模式定点）：

- `test_ge_graph_bsnd_baseline`：BSND+causal+pooling 基线（图模式原已可用）
- `test_ge_graph_tnd`：TND（值校验跳过路径，原编译失败）
- `test_ge_graph_bsnd_pa` / `test_ge_graph_tnd_pa`：PA_BBND（含不等池数/-1 空槽）
- `test_ge_graph_bsnd_pa_key_noncontig`：PA key 0 轴非连续（key_stride0 属性路径）
- `test_ge_graph_bsnd_mask0_pools_lt_sparse_race`：**竞态回归**（精确失败组合
  mask=0/pools=4<sparse=14/非零 tail），编译图后重复调用 20 次逐次比对
- `test_ge_graph_tnd_mask0_pools_lt_sparse_race`：竞态回归 TND 变体（顺带覆盖
  前缀和 device tensor 值校验跳过路径），同样 20 次压测

`tests/pytest/test_pool_key_indexer_general_ge_graph.py`（新增文件，GE 泛化）：
复用 eager 泛化的全部 60 例配置（ALL_CASES 确定性生成）+ 覆盖率汇总断言，
执行后端替换为 torchair 编译路径；支持 `PKI_GE_GENERAL_MAX=N` 冒烟子集。

## 四、验证结果（Atlas A2 910B3）

| 项目 | 结果 |
|------|------|
| 竞态组合压力（GE，同图重复调用 20 次） | 0 bad / 20 runs（修复前 96/128 行稳定失败） |
| GE 图模式定点（7 例，含 2 竞态回归） | 7 passed |
| eager 定点 + 泛化 + 覆盖率收口 | 103 passed |
| GE 泛化（60 例 + 汇总） | 61 passed（修复前 56/61） |
| eager 竞态组合回归（3 次） | 0 bad |

## 五、回退记录（排障过程中的错误方向，留档避免重蹈）

排障中期曾按"REQUIRED+ValueDepend 触发 GE placeholder 替换"的推断修改
`op_def`（去掉 pool_tail_k 的 ValueDepend）及配套 csrc/ATK/文档，引入 eager
segfault 回归。该推断被 mask=3+topk=64 实验**直接证伪**（图模式
visibleTailK 计算正确 → kernel 能读到真实 ptk 值）。已全部回退
（`git checkout` def.cpp/csrc/kernel，ATK/文档手工恢复），真实根因为上述
kernel 流水线竞态。教训：**placeholder 假设无直接证据时不应改 op_def 这种
全局影响面最大的文件；应先做读值判别实验。**
