# Tiling 系统完整参考

Tiling 是 AscendC 算子的核心机制。本文覆盖从概念到具体代码的每个层次，包括基础 gert API、Dynamic Input/ScalarList、CompileInfo、OpAICoreConfig、注册方式等。

## 1. 什么是 Tiling

在 AscendC 中，tiling 是一个**host 侧（CPU）的预处理阶段**，在 NPU 执行 kernel 之前运行。它将计算工作负载分解为适合 NPU 片上内存层次结构（UB — Unified Buffer、L1、L0A、L0B、L0C）的块。

### 为什么需要 Tiling

- NPU 的 UB（片上缓存）大小有限（通常约 192KB）
- 输入张量通常远大于 UB，需要分块处理
- Tiling 函数计算每个块的大小、分配方式、核心分配等参数
- Kernel（device 侧）根据这些参数在 UB 上分块循环处理数据

### 执行流程

> **两种调用路径：** CANN 支持 graph 编译路径（先 TilingParse → 后 TilingFunc）和 aclnn 直调路径（直接 TilingFunc）。**NPUKernelBench 通过 pybind → aclnn 调用，走的是 aclnn 直调路径，TilingParse 不会被调用。**

```
用户调用 aclnnOp(x, y, z, attrs)
    ↓
[Host / CPU]
  1. InferShape: 推导输出 shape
  2. TilingFunc(context): 计算分块参数
     - 获取平台信息：platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
     - 获取输入 shape、dtype
     - 计算分块策略
     - 设置 SetBlockDim(num_cores)
     - 设置 SetTilingKey(key)
     - SaveToBuffer(tiling_data)
     - 设置 workspace 大小
  3. 将 tiling data 传给 kernel launcher
    ↓
[Device / NPU]
  4. kernel 入口接收 GM_ADDR ..., GM_ADDR tiling
  5. GET_TILING_DATA(tiling_data, tiling) 解包参数
  6. 使用 tiling 参数在 UB 上分块循环处理数据
```

## 2. TilingData — Host/Kernel ABI

`TilingData` 是在 host 和 kernel 之间传递参数的**二进制结构体**。它是迁移时**绝对不能破坏**的 ABI 契约。

### Host 侧定义（`_tiling.h`）

```cpp
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeluTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
    TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);
    TILING_DATA_FIELD_DEF(uint32_t, core_size);
    TILING_DATA_FIELD_DEF(uint32_t, core_remain);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gelu, GeluTilingData)
}  // namespace optiling
```

**`REGISTER_TILING_DATA_CLASS` 的第一个参数必须与 `OP_ADD(OpName)` 中的 `OpName` 一致**（即任务目录的 `OpName`）。

### 字段类型

| 宏 | 含义 |
|----|------|
| `TILING_DATA_FIELD_DEF(type, name)` | 标量字段 |
| `TILING_DATA_FIELD_DEF_ARR(type, size, name)` | 固定长度数组 |
| `TILING_DATA_FIELD_DEF_STRUCT(type, name)` | 嵌套结构体 |

常用类型：`uint32_t`、`int64_t`、`float`

### Kernel 侧读取

```cpp
extern "C" __global__ __aicore__ void gelu(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(GeluTilingData, tiling_data);
    // 宏展开后：从 tiling 指针读取二进制数据到 tiling_data 结构体

    uint32_t totalLength = tiling_data.totalLength;
    uint32_t tileNum = tiling_data.tileNum;
    // ...
}
```

**`GET_TILING_DATA` 宏由 `tiling_data_def_build.py` 在编译期生成**，它从 `_tiling.h` 解析字段定义，生成 `#pragma pack(1)` 结构体和 `Init*` 宏。

### ABI 规则

1. **Host 和 kernel 使用相同的字段类型、顺序和数组长度**
2. 不要为了"清理"而重新排列或重命名字段
3. 如果 host 修改了 TilingData 结构但 kernel 没有同步 → 运行时错位（输出全部为 0 或崩溃）
4. 如果 kernel 使用了 host 未设置的 tiling key → 命中空分支（输出全部为 0）

## 3. Tiling 函数 — gert::TilingContext

### 两种获取输入信息的方式

```cpp
// 方式 1: GetInputShape + GetInputDesc（分别获取 shape 和 dtype）
uint64_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
auto dt = context->GetInputDesc(0)->GetDataType();

// 方式 2: GetInputTensor（一次调用获取 tensor 的 shape 和 dtype）
uint64_t totalLength = context->GetInputTensor(0)->GetShapeSize();
auto dt = context->GetInputTensor(0)->GetDataType();
```

两种方式等效，选择已经在源仓中使用的方式即可，减少不必要的改动。

### 所有 gert tiling 函数的标准签名

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 1. 空指针检查
    if (context == nullptr || context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 2. 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) { coreNum = 1; }

    // 3. 获取输入张量信息
    uint64_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    auto dt = context->GetInputTensor(0)->GetDataType();

    // 4. 计算 tiling 参数
    // ...

    // 5. 设置 block dim（启动多少 AI 核心）
    context->SetBlockDim(coreNum);

    // 6. 设置 tiling key（多实现分支时）
    context->SetTilingKey(1);

    // 7. 将 tiling data 写入缓冲区
    MyTilingData tiling;
    tiling.totalLength = totalLength;
    // ... 设置其他字段 ...
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 8. 设置 workspace 大小
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;  // 0 或实际需要的字节数
    }

    return ge::GRAPH_SUCCESS;
}
```

### gert::TilingContext 常用 API 速查表

| 操作 | 代码 |
|------|------|
| 平台信息 | `platform_ascendc::PlatformAscendC(context->GetPlatformInfo())` |
| UB 大小 | `platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize)` |
| AIV 核心数 | `platform.GetCoreNumAiv()` |
| 通用核心数 | `platform.GetCoreNum()` |
| 输入 shape | `context->GetInputShape(i)->GetStorageShape()` |
| 输入张量（含 dtype） | `context->GetInputTensor(i)->GetShapeSize()` / `GetDataType()` |
| 动态输入第 j 个 | `context->GetDynamicInputTensor(dynamicInputIndex, j)` |
| 可选输入 | `context->GetOptionalInputTensor(i)` |
| 数据类型 | `context->GetInputDesc(i)->GetDataType()` |
| 属性（Attr） | `context->GetAttrs()->GetAttrPointer<T>(idx)`、`GetInt(idx)`、`GetListInt(idx)` |
| 设置 block dim | `context->SetBlockDim(coreNum)` |
| 设置 tiling key | `context->SetTilingKey(key)` |
| workspace | `size_t* ws = context->GetWorkspaceSizes(1); ws[0] = size;` |
| 写入 tiling | `tiling.SaveToBuffer(data, cap)` + `SetDataSize(tiling.GetDataSize())` |
| ~~获取 CompileInfo~~ | **禁止使用**。aclnn 直调路径下 `GetCompileInfo<T>()` 始终返回 nullptr。用 `PlatformAscendC(context->GetPlatformInfo())` 直接获取平台信息 |
| ~~TilingParse / GetCompiledInfo~~ | **禁止使用**。aclnn 直调路径下 TilingParse 不会被调用。迁移时删除所有 TilingParse 相关代码 |

### 空指针检查

以下指针可能为空，tiling 函数必须判空：

- `context` 本身
- `context->GetPlatformInfo()` → 复杂算子可能 fallback 到 CompileInfo
- `context->GetInputShape(i)` / `context->GetInputDesc(i)` → 动态输入可能为空
- `context->GetOptionalInputTensor(i)` → 可选输入可能为空
- `context->GetWorkspaceSizes(1)` → 返回值可能为空
- `context->GetAttrs()->GetAttrPointer<T>(idx)` → OPTIONAL 属性可能为空

## 4. 动态输入（Dynamic Input）

当前面有 `ParamType(DYNAMIC)` 的输入时，使用 `GetDynamicInputTensor` 而非 `GetInputTensor`：

```cpp
// 遍历动态输入列表
uint32_t tensorCount = 0;
uint64_t totalCount = 0;
for (uint32_t i = 0; i < MAX_TENSOR_COUNT; ++i) {
    auto tensor = context->GetDynamicInputTensor(0, i);  // 0 = 第 0 个 DYNAMIC 输入
    if (tensor == nullptr) {
        break;  // 列表结束
    }
    totalCount += tensor->GetShapeSize();
    ++tensorCount;
}
```

对应的 OpDef 声明：

```cpp
this->Input("x")
    .ParamType(DYNAMIC)  // 接受不定数量的张量
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
```

### ScalarList 输入

```cpp
this->Input("scalars")
    .ScalarList()           // 标量列表
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_INT64})
    .Format(formats)
    .UnknownShapeFormat(formats)
    .AutoContiguous();
```

动态输入和 ScalarList 常见于 Foreach 类算子（对每个张量施加相同的操作）。迁移时 OpDef 不能简化为固定输入，tiling 中必须用 `GetDynamicInputTensor(dynamicInputIndex, j)`。

## 5. 从旧接口迁移到 gert

如果你的源算子使用旧接口（`TeOpParas`、`OpRunInfo`、`nlohmann::json`），必须重写为 `gert::TilingContext*`。

### 对照表

| 旧接口 | gert/msopgen 等效 |
|--------|-------------------|
| `para.inputs[i].tensor.shape` | `context->GetInputShape(i)->GetStorageShape()` |
| `para.inputs[i].tensor.dtype` | `context->GetInputDesc(i)->GetDataType()` |
| `para.op_params["attr"]` | `context->GetAttrs()->GetAttrPointer<T>(idx)` |
| JSON `compileInfo["ub_size"]` | `PlatformAscendC(...).GetCoreMemSize(...)` 或 `CompileInfo` |
| JSON `compileInfo["core_num"]` | `platform.GetCoreNumAiv()` / `GetCoreNum()` |
| `runInfo.SetBlockDim(n)` | `context->SetBlockDim(n)` |
| `runInfo.AddTilingData(&buf, len)` | `SaveToBuffer(...)` + `SetDataSize(...)` |
| `runInfo.SetWorkspace(size)` | `context->GetWorkspaceSizes(1)[0] = size` |
| `return true` | `return ge::GRAPH_SUCCESS` |
| `return false` | `return ge::GRAPH_FAILED` |

### 实际改写示例

**改前（旧接口）：**
```cpp
using json = nlohmann::json;
using TeOpParas = te::OpParas;
using OpRunInfo = te::OpRunInfo;

bool TilingFunc(const TeOpParas& para, OpRunInfo& runInfo) {
    uint64_t totalLength = para.inputs[0].tensor.shape.GetShapeSize();
    auto compileInfo = json::parse(para.compile_info);
    uint64_t ubSize = compileInfo["ub_size"];
    uint32_t coreNum = compileInfo["core_num"];

    // 计算 tiling 参数 ...

    runInfo.SetBlockDim(coreNum);
    runInfo.AddTilingData(&tilingData, tilingSize);
    runInfo.SetWorkspace(workspaceSize);
    return true;
}
```

**改后（gert）：**
```cpp
#include "tiling/platform/platform_ascendc.h"

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) { coreNum = 1; }

    // 同样的 tiling 计算逻辑 ...

    MyTilingData tiling;
    // ... 设置字段 ...
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(coreNum);

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = workspaceSize;
    }

    return ge::GRAPH_SUCCESS;
}
```

## 6. Tiling Key

Tiling key 允许同一个 kernel 有不同的执行路径（如针对不同 dtype、不同 shape 规模）。

### Host 侧设置 key

```cpp
context->SetTilingKey(key);  // key 是整数，如 0, 1, 2, 3
```

### Kernel 侧检查 key

```cpp
if (TILING_KEY_IS(1)) {
    // 路径 1
} else if (TILING_KEY_IS(2)) {
    // 路径 2
} else {
    // 默认路径
}
```

或使用 `GetTilingKey()` 获取 key 值。

### 迁移规则

1. 保留源仓中 tiling key 的设置逻辑和对应的 kernel 路径
2. 删除非目标架构的 kernel 路径时，确保对应的 tiling key 不会被 host 设置
3. 如果某个 tiling key 对应的 kernel 路径在 `arch35/` 等非目标目录中 → host 不要再设置该 key

## 7. 两种 Tiling 注册方式

### 方式一：简单 `.SetTiling()`（模式 1、2a）

```cpp
// 在 OpDef 构造函数中
this->AICore().SetTiling(optiling::TilingFunc);
```

TilingFunc 是一个普通的 gert tiling 函数。OpDef 和 tiling 实现可以在同一个 `.cpp` 文件中。

### 方式二：`IMPL_OP_OPTILING`（模式 2b、3）

当算子使用 `IMPL_OP_OPTILING` 注册 tiling 时，`_def.cpp` 中**不能**写 `.SetTiling()`。

**简单 IMPL_OP_OPTILING（模式 2b — 无多策略 registry）：**

```cpp
// 在 _tiling.cpp 中
IMPL_OP_OPTILING(MyOp)
    .Tiling(TilingFunc);
```

**复杂 IMPL_OP_OPTILING（模式 3 — 多策略 registry）：**

> **注意：aclnn 直调路径下 TilingParse 不会被调用。** 迁移时必须删除 `TilingPrepare` 和 `.TilingParse<...>(...)`。各 tiling 策略类中用到 `GetCompileInfo()` 的代码改为 `PlatformAscendC(context->GetPlatformInfo())` 直接获取。

```cpp
// 迁移后：删除 TilingParse，直接用 PlatformAscendC
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {
ge::graphStatus Tiling4MyOp(gert::TilingContext* context)
{
    return OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

// TilingParse 已删除 — aclnn 路径下不会被调用

REGISTER_OPS_TILING_TEMPLATE(MyOp, StrategyATiling, 50000);
REGISTER_OPS_TILING_TEMPLATE(MyOp, StrategyBTiling, 60000);

IMPL_OP_OPTILING(MyOp)
    .Tiling(Tiling4MyOp);
    // 注意：已删除 .TilingParse<MyOpCompileInfo>(...)
}  // namespace optiling
```

## 8. CompileInfo 模式 → 迁移时必须移除

源仓中部分算子使用 `CompileInfo` + `TilingParse` 模式，在编译期通过 `TilingParse` 获取平台信息，然后在 `TilingFunc` 中通过 `GetCompileInfo<T>()` 读取。

```cpp
// 源仓模式（在 aclnn 路径下不可用）
namespace optiling {
struct MyOpCompileInfo {
    uint64_t ubSize = 0;
    uint32_t aivCoreNum = 0;
};
}  // namespace optiling
```

### 为什么不能保留

**NPUKernelBench 通过 pybind → aclnn 调用算子，走的是 aclnn 直调路径（非 graph 编译路径）。在此路径下，`TilingParse` 回调不会被调用，`context->GetCompileInfo<T>()` 始终返回 nullptr。**

### 迁移方法：直接用 PlatformAscendC 替代

移除 `CompileInfo` 结构体、`TilingParse` 函数和 `.TilingParse<CompileInfo>(...)` 注册。在 `TilingFunc` 中直接从 runtime context 获取平台信息：

```cpp
// 迁移后：直接从 runtime context 获取
#include "tiling/platform/platform_ascendc.h"

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivCoreNum = platform.GetCoreNumAiv();
    // ... 直接使用 ubSize 和 aivCoreNum ...
}
```

**迁移步骤：**
1. 找到 `TilingFunc` 中所有 `context->GetCompileInfo<T>()` 调用，删除
2. 用 `platform_ascendc::PlatformAscendC(context->GetPlatformInfo())` 直接获取平台信息
3. 删除 `TilingParse` 函数定义
4. 删除 `IMPL_OP_OPTILING` 中的 `.TilingParse<CompileInfo>(...)` 注册
5. 删除 `CompileInfo` 结构体定义（如果只用于 tiling 传参）

## 9. 关于 REGISTER_TILING_TEMPLATE 宏的说明

CANN SDK 提供的 `REGISTER_TILING_TEMPLATE` 第一个参数是**字符串**（带引号）：
```cpp
REGISTER_TILING_TEMPLATE("OpName", ClassName, priority);
```

本地化的 `tiling_base/tiling_templates_registry.h` 中通常定义了 `REGISTER_OPS_TILING_TEMPLATE`，第一个参数是**裸符号**（不带引号，内部 stringify）：
```cpp
REGISTER_OPS_TILING_TEMPLATE(OpName, ClassName, priority);
```

迁移时使用哪种取决于你复制的 `tiling_base/` 来源。**不要混用**：如果用了本地化头中的宏就用裸符号，如果是 CANN SDK 的宏就用字符串。

## 10. OpAICoreConfig 高级特性

某些算子使用 `OpAICoreConfig`（由 `.AddConfig()` 返回）而不是简单的 `.AddConfig("ascend910b")`：

```cpp
auto cfg = this->AICore().AddConfig("ascend910b");
cfg.AutoContiguous()                        // 自动处理不连续张量
   .DynamicCompileStaticFlag(true)           // 动态编译静态化
   .DynamicRankSupportFlag(true)             // 支持动态 rank
   .DynamicShapeSupportFlag(true)            // 支持动态 shape
   .ExtendCfgInfo("opFile.value", "kernel_file_name");  // 指定 kernel 文件名
```

这些特性在源仓中如果有，迁移时保留。如果源仓没有，不要自行添加。

## 11. Tiling 迁移检查清单

- [ ] TilingData 字段类型、顺序、数组长度与源仓完全一致
- [ ] `REGISTER_TILING_DATA_CLASS(OpName, ...)` 的 OpName 与 `OP_ADD(OpName)` 一致
- [ ] kernel 侧 `GET_TILING_DATA` 后读取的字段名/顺序与 host `_tiling.h` 一致
- [ ] tiling 注册方式正确（简单用 `.SetTiling()`，复杂用 `IMPL_OP_OPTILING`，两者不同时存在）
- [ ] 旧接口（`TeOpParas`、`nlohmann`）已完全替换为 `gert::TilingContext*`
- [ ] 已设置 `SetBlockDim()`，值 > 0
- [ ] 已设置 workspace，且对 workspace 指针判空
- [ ] 如果 kernel 使用 `TILING_KEY_IS`，host 设置了对应的 `SetTilingKey`
- [ ] 已删除所有 `TilingParse` 函数和 `.TilingParse<CompileInfo>(...)` 注册（aclnn 直调路径不会调用）
- [ ] 已删除所有 `GetCompileInfo<T>()` 调用，改为 `PlatformAscendC(context->GetPlatformInfo())` 直接获取
- [ ] Dynamic Input 使用 `GetDynamicInputTensor`，OpDef 中有 `ParamType(DYNAMIC)`
- [ ] ScalarList 输入正确声明
- [ ] 来自开源仓公共 include 的 tiling 依赖已本地化
- [ ] 非目标芯片配置已删除
