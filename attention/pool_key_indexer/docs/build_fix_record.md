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
