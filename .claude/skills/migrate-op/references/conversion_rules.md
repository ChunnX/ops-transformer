# op_host 通用适配规则

**定位**：适用于**所有算子**（简单和复杂）的 op_host 通用转换规则。
转换目标：将 CANN 开源仓库的算子代码适配为 NPUKernelBench 单算子工程格式，仅支持 ascend910b/ascend910_93。

复杂算子还有额外的专属规则（CompileInfo 保留、框架 stub、跨算子依赖等），详见 `complex_op_host.md`。

---

## 1. 头文件处理

### 1.1 删除（单算子工程不可用）

```
util/math_util.h
log/log.h
platform/platform_info.h
platform/platform_infos_def.h
tiling_base/tiling_util.h
```

### 1.2 保留

| 头文件 | 依赖说明 |
|:---|:---|
| `register/tilingdata_base.h` | TilingData 基类 |
| `register/op_def_registry.h` | `ge::graphStatus`、`OpDef` 依赖此头文件，缺失会编译错误 |
| `register/op_impl_registry.h` | `IMPL_OP_OPTILING`、`gert::TilingContext` 依赖此头文件 |
| `tiling/platform/platform_ascendc.h` | `PlatformAscendC` 类定义，CompileInfo 替换必需 |
| `graph/utils/type_utils.h` | 类型工具 |

### 1.3 补全检查宏

若 tiling 文件使用了检查宏，在文件顶部添加：

```cpp
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        std::printf("nullptr error!"); \
        return ge::GRAPH_FAILED; \
    }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)
```

日志宏定义（`OP_LOGD`/`OP_LOGE` 等）见第7节。

---

## 2. tiling.cpp 必需头文件三件套

tiling.cpp 必须包含以下头文件（按顺序）：

```cpp
#include "<opname>_tiling.h"                     // TilingData 定义
#include "register/op_def_registry.h"             // ge::graphStatus, OpDef
#include "register/op_impl_registry.h"            // IMPL_OP_OPTILING, gert::TilingContext
#include "tiling/platform/platform_ascendc.h"     // PlatformAscendC (CompileInfo 替换)
```

缺失任何一个会导致 `gert has not been declared`、`graphStatus does not name a type`、`IMPL_OP_OPTILING undeclared` 等错误。

---

## 3. 非目标平台清理

仅保留 ascend910b/ascend910_93，删除所有其他平台代码。

### 3.1 _def.cpp

- 删除非 ascend910b/ascend910_93 的 `OpAICoreConfig` 及其 `.Input()`/`.Output()`
- 删除对应的 `.AddConfig("ascend950"/"ascend310p"/"ascend310b"/"kirinx90"/...)`
- 删除 `.ExtendCfgInfo(...)`、`.DynamicCompileStaticFlag(...)`、`.DynamicFormatFlag(...)`、`.DynamicRankSupportFlag(...)`、`.DynamicShapeSupportFlag(...)`、`.NeedCheckSupportFlag(...)` 等框架专用标志

### 3.2 _tiling.cpp

- 删除 `isAscend950`、`IsRegbaseSocVersion`、`SOC_VERSION_310P` 等平台检查及分支
- 删除 `DCACHE_SIZE`、`REGBASE_CCEC_CACHE_SIZE` 相关逻辑
- 删除 `socVersionType` 成员变量及其所有条件分支（如 `if (socVersionType == SOC_VERSION_310P)`），仅保留 910B 路径
- 删除专属其他平台的 tiling 函数（如 `KerneTiling310p`）及其调用点
- `GetNeedCoreNum` 等函数中的平台条件索引（如 `socVersionType == SOC_VERSION_310P ? output_shapes[1] : output_shapes[3]`），删除条件判断，直接使用 910B 路径值

### 3.3 op_kernel

- 删除 `#if __CCE_AICORE__ == 200` 非 220 分支（含 `arch35/` 和 `*Simt` 命名空间），仅保留 `__CCE_AICORE__ == 220` 路径
- 删除 `*_apt.cpp` 其他芯片入口文件
- 含 `__CCE_AICORE__ == 300` 条件编译的，保留 220 路径，移除非目标架构分支

---

## 4. CompileInfo 与 TilingParse 处理

msopgen 单算子工程**原生支持** `TilingParse<XxxCompileInfo>` 注册机制（通过 `tiling_api` 库）。是否保留 CompileInfo 取决于其内容复杂度，而非算子本身的复杂度。

复杂算子有额外的 CompileInfo 保留规则（TilingBaseClass 继承链、框架内部 stub 等），详见 `complex_op_host.md` 第2节。

### 4.1 决策树

```
CompileInfo 包含什么？
├── 空结构体或仅 totalCoreNum / ubSize / aivCoreNum 等基础参数
│   ├── 可替换为 PlatformAscendC（方案 A，减少代码量）→ 见 4.2
│   └── 也可保留原样（方案 B，减少改动量）→ 见 4.3
│
├── 含 cube 硬件参数（l1Size, l0ASize, l0BSize, l0CSize, l2Size 等）
│   或含平台特性探测（supportL0c2out, supportL12BtBf16 等）
│   └── 必须保留 CompileInfo + TilingParse → 见 4.3
│
└── 含 bank/cache 相关字段（由 TilingPrepareForOpCache 填充）
    └── 保留 CompileInfo，但 stub OpCache 调用 → 见 `complex_op_host.md` 第3节
```

### 4.2 方案 A：替换为 PlatformAscendC

适用于 CompileInfo 仅含基础参数且希望减少代码量的场景。

1. 删除 CompileInfo 结构体定义（`_tiling.h`）
2. 删除 TilingPrepare 函数
3. 硬件参数获取替换：

```cpp
// 替换前
auto compileInfo = static_cast<const XxxCompileInfo*>(tilingContext_->GetCompileInfo());
coreNum = compileInfo->totalCoreNum;
ubSize = compileInfo->ubSizePlatForm;
workspaceSize = compileInfo->workspaceSize;

// 替换后
auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
auto coreNum = ascendcPlatform.GetCoreNum();
uint64_t ubSize = 0;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
```

4. 简化注册（见第5节）

注意事项：
- 删除所有 `context->GetCompiledInfo<XxxCompileInfo>()` 调用
- 若 CompileInfo 含 `socVersionType`：删除成员变量、TilingPrepare 赋值（含 `GetCurNpuArch()` 判断）、所有条件分支
- 若 `compileInfo->totalCoreNum` 多平台用不同 API（如 910B 用 `GetCoreNumAic()`、310p 用 `GetCoreNumAiv()`），统一替换为 `GetCoreNumAiv()`
- 多处引用 CompileInfo 时（如 `RunBigKernelTiling` 和 `GetNeedCoreNum`），每处独立替换为 PlatformAscendC 实例

### 4.3 方案 B：保留 CompileInfo + TilingParse

适用于希望最小化改动量，或 CompileInfo 含 cube 硬件参数的场景。

msopgen 的 `cust_optiling` 链接 `tiling_api` 库，`IMPL_OP_OPTILING(Xxx).TilingParse<CompileInfo>(...)` 注册机制完全可用。

**保留步骤**：

1. **保留** CompileInfo 结构体定义（在 `_tiling.h` 中）
2. **保留** TilingParse 注册：`.TilingParse<XxxCompileInfo>(TilingPrepareForXxx)`
3. **改造** TilingPrepare 函数：
   - 保留 `PlatformAscendC` 的硬件参数填充路径
   - 删除 `socVersionType` 相关条件分支（如 310p 判断）
   - 删除 `TilingPrepareForOpCache(context)` 调用（如有）

**两种 TilingParse 模式**：

模式一：空壳 TilingParse（CompileInfo 为空结构体，平台信息在 TilingFunc 里取）

```cpp
// _tiling.h
struct MyOpCompileInfo {};

// _tiling.cpp
static ge::graphStatus TilingPrepare4MyOp(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MyOp)
    .Tiling(TilingFunc)
    .TilingParse<MyOpCompileInfo>(TilingPrepare4MyOp);
```

模式二：缓存平台参数（TilingParse 里取一次，TilingFunc 里通过 compileInfo 直接用）

```cpp
// _tiling.h
struct MyOpCompileInfo {
    uint64_t ubSize{0};
    uint64_t coreNum{0};
    uint64_t aivCoreNum{0};
    uint64_t l1Size{0};
};

// _tiling.cpp
static ge::graphStatus TilingPrepare4MyOp(gert::TilingParseContext* context) {
    auto compileInfoPtr = context->GetCompiledInfo<MyOpCompileInfo>();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    compileInfoPtr->aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}
```

**TilingFunc 中使用 compileInfo 的兼容写法**（同时支持有/无 compileInfo 两种路径）：

```cpp
// 部分开源算子（如 ForeachAbs）的兼容写法：优先从 compileInfo 取，fallback 到 PlatformAscendC
auto platformInfo = context->GetPlatformInfo();
auto compileInfoPtr = (const MyOpCompileInfo*)(context->GetCompileInfo());
uint64_t ubSize = 0;
uint32_t coreNum = 0;
if (platformInfo != nullptr) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNumAiv();
} else {
    ubSize = compileInfoPtr->ubSize;
    coreNum = compileInfoPtr->aivCoreNum;
}
```

### 4.4 构建系统无需额外配置

无论选择方案 A 还是方案 B，**默认 msopgen 的 op_host CMakeLists.txt 无需修改**：
- `aux_source_directory` 自动收集所有 `.cpp` 编译进 `cust_optiling`
- `cust_optiling` 已链接 `tiling_api`（提供 `IMPL_OP_OPTILING` 和 `TilingParse` 注册机制）
- TilingParse 和 CompileInfo 是用户代码，直接编译进 `cust_optiling.so`，不依赖 `op_build` 工具生成

---

## 5. 注册逻辑适配

### 5.1 方案 A：替换 CompileInfo 时（对应第4.2节）

1. 删除整个 `TilingPrepareForXxx(gert::TilingParseContext* context)` 函数
2. IMPL_OP_OPTILING 简化：

```cpp
// 替换前
IMPL_OP_OPTILING(Xxx).InputsDataDependency({...}).Tiling(TilingXxx).TilingParse<XxxCompileInfo>(TilingPrepareForXxx);
// 替换后
IMPL_OP_OPTILING(Xxx).Tiling(TilingXxx);
```

### 5.2 方案 B：保留 CompileInfo 时（对应第4.3节）

保持原始注册不变：

```cpp
// 保持不变
IMPL_OP_OPTILING(Xxx)
    .InputsDataDependency({...})
    .Tiling(TilingXxx)
    .TilingParse<XxxCompileInfo>(TilingPrepareForXxx);
```

仅需清理 TilingPrepareForXxx 函数内部的非目标平台分支和框架内部调用（如 `TilingPrepareForOpCache`）。

### 5.3 通用规则

- 保持独立文件结构（`<opname>_def.cpp` + `<opname>_tiling.cpp`）
- 删除 `.ExtendCfgInfo(...)`、`.DynamicCompileStaticFlag(...)` 等框架专用配置标志

---

## 6. 多文件 tiling 类合并

若源码 tiling 拆分为多个文件（如 `*_tiling.cpp` + `*_last_dim_tiling.h`），合并为单个 tiling.cpp 时：
- 将所有 tiling 类定义内联到同一个 .cpp 文件中
- 删除 CompileInfo 相关的独立头文件
- 确保 `Ops::Base::CeilAlign` 等工具函数有本地替代（命名空间或 inline 函数）
- 若源码使用了 `platform/platform_infos_def.h`，删除该引用，改用 `tiling/platform/platform_ascendc.h`

---

## 7. 日志宏定义

源码中 `OP_LOGD`/`OP_LOGE` 存在两种调用方式：
- 双参数：`OP_LOGD(nodeName, "fmt %d", val)` — 多数 tiling 函数内
- 单参数：`OP_LOGD("message")` — 部分工具函数内

**规则**：使用变参宏以兼容两种调用：

```cpp
#define OP_LOGD(...) do { std::printf(__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(...) do { std::printf(__VA_ARGS__); std::printf("\n"); } while(0)
```

---

## 8. op_kernel 中不需要 include tiling.h

op_kernel 文件中使用 `XxxTilingData` 类型（如 Init 函数参数），但**不需要**手动 `#include "xxx_tiling.h"`。构建系统会自动从 op_host 的 `REGISTER_TILING_DATA_CLASS` 宏生成 tiling 数据定义，并在内核编译时自动包含。

手动 include 反而会导致 `'xxx_tiling.h' file not found` 编译错误（因为 op_kernel 和 op_host 的 include 路径不同）。

---

## 常见编译错误速查

| 错误信息 | 原因 | 修复 |
|:---|:---|:---|
| `gert has not been declared` | 缺少 `register/op_impl_registry.h` | 添加三件套头文件 |
| `graphStatus does not name a type` | 缺少 `register/op_def_registry.h` | 添加三件套头文件 |
| `IMPL_OP_OPTILING undeclared` | 缺少 `register/op_impl_registry.h` | 添加三件套头文件 |
| `XxxCompileInfo undeclared` | 未清理 CompileInfo 引用 | 执行第 4 节替换 |
| `arch35 namespace not found` | 未清理非目标平台分支 | 执行第 3.3 节清理 |

---

## 转换校验清单

简单算子转换完成后逐项自查（复杂算子额外检查项见 `complex_op_host.md` 第9节）：

- [ ] 无非目标平台代码残留（950/310p/kirin 等）
- [ ] CompileInfo 已按第4节决策树处理（替换或保留，无遗漏残留）
- [ ] 若保留 CompileInfo：TilingPrepare 函数无非目标平台分支、无框架内部调用（OpCache 等）
- [ ] 头文件/宏定义完整（三件套：`<opname>_tiling.h` + `op_def_registry.h` + `op_impl_registry.h`）
- [ ] Tiling 注册：answer 用 `IMPL_OP_OPTILING`（question 骨架的 `SetTiling` 注册在步骤5生成时验证）

---

## 9. NC1HWC0 格式算子适配

**判断条件**：`_def.cpp` 中 Input/Output Format 为 `FORMAT_NC1HWC0`。

### 9.1 _def.cpp 格式修改

将 `FORMAT_NC1HWC0` 替换为 `FORMAT_ND`：

```cpp
// 替换前
.Input({x, ge::TensorDesc(ge::Shape(x_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT)})
.Output({y, ge::TensorDesc(ge::Shape(y_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT)})

// 替换后
.Input({x, ge::TensorDesc(ge::Shape(x_shape), ge::FORMAT_ND, ge::DT_FLOAT)})
.Output({y, ge::TensorDesc(ge::Shape(y_shape), ge::FORMAT_ND, ge::DT_FLOAT)})
```

### 9.2 question 骨架格式修改

question 骨架中的 `_def.cpp` 也需同步将 `FORMAT_NC1HWC0` 替换为 `FORMAT_ND`。

### 9.3 Tiling 代码清理

删除 `FORMAT_NC1HWC0` 格式检查相关逻辑。

### 9.4 Python 侧转换

Python 侧的 reshape/permute 转换规则（`module.py` 的 `ModelNew.forward()` 中）见 `validation_guide.md` 第8.1节。

---

## 变更记录

- 2026-04-24: 重构第4节（CompileInfo 与 TilingParse 处理）——新增决策树区分方案A（替换）和方案B（保留）；新增第4.3节保留 CompileInfo + TilingParse 的两种模式（空壳/缓存平台参数）；新增第4.4节构建系统说明；重构第5节注册逻辑适配——区分方案A/B的注册处理方式；更新转换校验清单
- 2026-04-23: 新增第9节 NC1HWC0 格式算子适配（从 `validation_guide.md` 第8.1节拆出 _def.cpp 和 question 骨架部分）
- 2026-04-23: 重构为通用适配规则——移除第9-12节（复杂算子专属内容已迁移至 `complex_op_host.md`）；明确文档定位为"适用于所有算子的通用规则"
- 2026-04-22: 新增转换校验清单（从 SKILL.md 移入）；第1.3节日志宏定义移至第7节，改为引用
- 2026-04-21: 新增第9-12节（CompileInfo 保留、框架 stub、跨算子 include 改写、模板化 tiling key），支持复杂算子迁移
- 2026-04-09: 新增第7节日志宏变参规则、第8节 kernel 不需要 include tiling.h 规则、速查表新增 OP_LOGD 错误条目
