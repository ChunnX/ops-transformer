# op_host Tiling 迁移指南

本文用于把 CANN 开源仓算子的 `op_host` 侧代码迁移到 NPUKernelBench 的 msopgen 单算子工程。范围包括 `OpDef`、InferShape、TilingData、tiling 入口、复杂 tiling 公共库和 CMake；`op_kernel` 架构分支清理按 skill 主流程处理。

迁移原则只有一个：保留原算子的 tiling 算法和 kernel ABI，只替换工程入口、注册方式、依赖和目标芯片配置。

## 1. 先理解差异

| 维度 | 开源仓 | NPUKernelBench/msopgen |
|---|---|---|
| 工程组织 | 仓库根 CMake 统一编译，各算子可依赖公共模块、公共 target 和芯片配置目录 | `answer/{sample}/` 被复制到独立 msopgen 工程中编译，算子依赖必须本地可见 |
| OpDef | 新算子通常已有 `op_host/*_def.cpp`，老算子可能从 proto/README/op_api 推导 | 必须提供 `OpDef` + `OP_ADD(Op)`，并能被 `opbuild(OPS_SRC ...)` 处理 |
| Tiling 注册 | 简单算子可用 `.SetTiling(...)`；复杂算子常用仓库公共 `REGISTER_TILING_TEMPLATE` | 简单算子继续 `.SetTiling(...)`；复杂算子本地化 registry 后用 `IMPL_OP_OPTILING` |
| 公共依赖 | 可 include `../../xxx_utils`、`tiling/tiling_base.h`、`log/ops_log.h`、`error/ops_error.h` | 依赖要复制到本算子 `op_host/`，或替换为 CANN/msopgen 公开头 |
| CMake | 常见 `add_modules_sources(...)`、`DEPENDENCIES xxx_utils`、`TILING_DIR arch35` | 使用 msopgen CMake；有子目录/手写 op_api/多 cpp 时写本地 `op_host/CMakeLists.txt` |
| 芯片配置 | 可能包含 `ascend950`、`kirinx90`、`kirin9030`、regbase/arch35 | 目标为 CANN 8.5.0 + 910B/910_93，通常只保留 `ascend910b` 和 `ascend910_93` |

当前工程默认 `kernel_only_mode: True`。编译前会先复制 `libs/common_include/msopgen`，再把 `answer/{sample}/` 覆盖到工程根目录，因此：

- `answer/{sample}/CMakeLists.txt` 会覆盖 msopgen 根 CMake；若存在，必须是 msopgen 可用的根 CMake。
- `answer/{sample}/op_host/CMakeLists.txt` 会覆盖 msopgen 默认 host CMake。
- 默认 `op_host/CMakeLists.txt` 只用 `aux_source_directory` 收集 `op_host` 当前目录 `.cpp`，不会递归收集子目录。

## 2. 迁移前分类

先在原始算子目录运行：

```bash
rg -n "gert::TilingContext|TilingParseContext|TeOpParas|OpRunInfo|nlohmann|REGISTER_TILING_TEMPLATE|IMPL_OP_OPTILING|SetTiling" op_host
```

按结果选择迁移路径：

| 形态 | 判断特征 | 迁移路径 |
|---|---|---|
| 简单 gert 算子 | `TilingFunc(gert::TilingContext*)`，无多策略 registry | 保留 tiling 函数，补齐 `OpDef`，用 `.SetTiling(optiling::TilingFunc)` |
| 分文件 gert 算子 | 有 `*_def.cpp`、`*_tiling.cpp`、`*_infershape.cpp`，无 `TilingBaseClass` | 分文件保留，裁剪不可用依赖和芯片配置；根据源码注册方式选择 `.SetTiling` 或 `IMPL_OP_OPTILING` |
| 复杂多策略算子 | 有 `TilingBaseClass`、`REGISTER_TILING_TEMPLATE`、多个 `*_tiling*.cpp` | 本地化 `common/` 和 `tiling_base/`，改用 `REGISTER_OPS_TILING_TEMPLATE` + `IMPL_OP_OPTILING` |
| 旧接口算子 | 出现 `TeOpParas`、`OpRunInfo`、`ge::Operator`、`nlohmann::json` | 重写为 `gert::TilingContext*`，平台/shape/attr/workspace 全部从 context 获取 |
| 只有 arch35/apt | 主逻辑在 `arch35/`、`*_apt.cpp`，OpDef 只配 `ascend950` | 不直接迁移该路径；优先找 910B/membase 源码，否则需要重新实现入口和 tiling |

目标目录通常是：

```text
answer/0/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt              # 只有多文件/子目录场景需要
│   ├── {kernel}_def.cpp            # 可与 tiling 合并
│   ├── {kernel}_tiling.h
│   ├── {kernel}_tiling.cpp         # 可与 def 合并
│   ├── {kernel}_infershape.cpp     # 需要时保留
│   ├── common/                     # 本地化日志/工具
│   └── tiling_base/                # 多策略 registry 才需要
└── op_kernel/
    └── ...
```

## 3. OpDef 迁移

`_def.cpp` 是 msopgen `opbuild` 的输入。开源仓已有 `*_def.cpp` 时优先复制后裁剪；没有时再从 README、proto、op_api、测试用例恢复定义。

最小模板：

```cpp
#include "register/op_def_registry.h"

namespace ops {
class MyOp : public OpDef {
public:
    explicit MyOp(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore().SetTiling(optiling::TilingFunc);  // 仅简单算子使用
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(MyOp);
}  // namespace ops
```

核对要点：

- `OP_ADD(OpName)`、TilingData 注册名、kernel 注册名、pybind 调用名和任务目录的 `OpName` 要一致。
- Input/Output 名称和顺序以原 OpDef/proto/op_api 语义为准，同时要和 kernel `GM_ADDR` 参数顺序对应。
- `.DataType({...})`、`.Format({...})`、`.UnknownShapeFormat({...})` 的元素数量必须一致。
- 动态输入保留 `.ParamType(DYNAMIC)`；可选输入保留 `.ParamType(OPTIONAL)`；标量/列表按原定义保留 `.Scalar()`、`.ScalarList()`、`.ListTensorType()` 等。
- 删除 Kirin、ascend950-only、arch35/regbase 专用配置。保留 `OpAICoreConfig` 时确认没有 `.ExtendCfgInfo("opFile.value", "..._apt")` 指向非目标 kernel。
- 使用 `IMPL_OP_OPTILING` 的复杂算子，`_def.cpp` 中不要再写 `.SetTiling(...)`。

## 4. TilingData 与 CompileInfo

TilingData 宏本身兼容 msopgen：

```cpp
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MyOpTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, shapeInfo);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MyOp, MyOpTilingData)
}  // namespace optiling
```

不要把 `_tiling.h` 当作“完全无需修改”的文件。迁移时只做工程必要改动，并检查：

- `REGISTER_TILING_DATA_CLASS(MyOp, MyOpTilingData)` 的 op 名和 `OP_ADD(MyOp)` 一致。
- 字段名、类型、顺序和 kernel 侧 `GET_TILING_DATA` 后读取的字段一致。
- 原头文件的仓库相对 include 已本地化或替换。
- 字段大小和类型不要为了清理而改变，否则会破坏 tiling buffer ABI。

如果 tiling 类中读取 compile info，例如：

```cpp
auto compileInfo = context_->GetCompileInfo<MyOpCompileInfo>();
// 或
auto compileInfo = reinterpret_cast<const MyOpCompileInfo*>(context_->GetCompileInfo());
```

就必须保留对应结构，并在 `TilingParse` 中写入同一个类型：

```cpp
namespace optiling {
struct MyOpCompileInfo {
    uint64_t ubSize = 0;
    uint32_t aivCoreNum = 0;
    int32_t socVersion = 0;
};
}  // namespace optiling
```

## 5. Tiling 实现

### 5.1 gert 常用 API

| 操作 | 写法 |
|---|---|
| 平台信息 | `platform_ascendc::PlatformAscendC(context->GetPlatformInfo())` |
| UB 大小 | `platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize)` |
| AIV 核数 | `platform.GetCoreNumAiv()` |
| 通用核数 | `platform.GetCoreNum()` |
| 输入 shape | `context->GetInputShape(i)->GetStorageShape()` |
| tensor shape/dtype | `context->GetInputTensor(i)->GetStorageShape()` / `GetDataType()` |
| 动态输入第 j 个 | `context->GetDynamicInputTensor(inputIndex, j)` |
| 可选输入 | `context->GetOptionalInputTensor(i)` |
| dtype | `context->GetInputDesc(i)->GetDataType()` |
| attr | `context->GetAttrs()->GetAttrPointer<T>(idx)`、`GetInt(idx)`、`GetListInt(idx)` |
| block dim | `context->SetBlockDim(coreNum)` |
| tiling key | `context->SetTilingKey(key)` |
| workspace | `size_t* ws = context->GetWorkspaceSizes(1); ws[0] = size;` |
| 写 tiling buffer | `tiling.SaveToBuffer(...)` + `context->GetRawTilingData()->SetDataSize(tiling.GetDataSize())` |

旧接口改写时按下表替换：

| 旧接口 | gert/msopgen |
|---|---|
| `para.inputs[i].tensor.shape` | `context->GetInputShape(i)->GetStorageShape()` |
| `para.inputs[i].tensor.dtype` | `context->GetInputDesc(i)->GetDataType()` |
| `para.op_params["attr"]` | `context->GetAttrs()->GetAttrPointer<T>(idx)` |
| JSON `compileInfo["ub_size"]` | `PlatformAscendC(...).GetCoreMemSize(...)` 或 `CompileInfo` |
| JSON `compileInfo["core_num"]` | `GetCoreNumAiv()` / `GetCoreNum()` |
| `runInfo.SetBlockDim(n)` | `context->SetBlockDim(n)` |
| `runInfo.AddTilingData(&buf, len)` | `SaveToBuffer(...)` + `SetDataSize(...)` |
| `runInfo.SetWorkspace(size)` | `context->GetWorkspaceSizes(1)[0] = size` |
| `return true` | `return ge::GRAPH_SUCCESS` |

空指针处理要集中在边界处：`GetPlatformInfo()` 可能为空，复杂算子通常 fallback 到 `GetCompileInfo()`；`GetWorkspaceSizes(1)`、可选输入和 attr 指针都要判空。

### 5.2 简单 `.SetTiling`

简单算子可把 TilingFunc、InferShape 和 OpDef 放在同一个 `.cpp`，也可以分文件。关键是 `.SetTiling(optiling::TilingFunc)` 只能注册一个可见的 gert tiling 函数。

```cpp
#include "my_op_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    MyOpTilingData tiling;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    uint64_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    tiling.set_totalLength(static_cast<uint32_t>(totalLength));

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
```

### 5.3 复杂 `IMPL_OP_OPTILING`

复杂算子通过 registry 选择 tiling class。迁移后 `_def.cpp` 只定义 OpDef，不写 `.SetTiling()`；入口和 parse 放在 `_tiling.cpp`：

```cpp
#include "my_op_tiling.h"
#include "my_strategy_a_tiling.h"
#include "my_strategy_b_tiling.h"
#include "common/log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
ge::graphStatus Tiling4MyOp(gert::TilingContext* context)
{
    return OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMyOp(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<MyOpCompileInfo>();
    if (compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    compileInfo->aivCoreNum = platform.GetCoreNumAiv();
    compileInfo->socVersion = static_cast<int32_t>(platform.GetSocVersion());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MyOp, MyStrategyATiling, 50000);
REGISTER_OPS_TILING_TEMPLATE(MyOp, MyStrategyBTiling, 60000);

IMPL_OP_OPTILING(MyOp)
    .Tiling(Tiling4MyOp)
    .TilingParse<MyOpCompileInfo>(TilingPrepareForMyOp);
}  // namespace optiling
```

如果原代码已有 `TilingPrepareFor...`，优先保留原逻辑，只替换不可用依赖和非目标芯片分支。

## 6. InferShape / InferDataType

输出 shape 简单等于输入时可以内联；多输出、动态输出、shape 依赖 attr、输出 dtype 不等于第一个输入 dtype、可选输入影响输出时，单独写 `*_infershape.cpp` 更清晰。

```cpp
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape4MyOp(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    gert::Shape* y = context->GetOutputShape(0);
    if (x == nullptr || y == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y = *x;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MyOp(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MyOp).InferShape(InferShape4MyOp).InferDataType(InferDataType4MyOp);
}  // namespace ops
```

## 7. 动态输入、ScalarList 和可选输入

Foreach 类算子常见动态 tensor list + scalar/scalar list。OpDef 不能简化成固定输入：

```cpp
namespace ops {
class ForeachLikeOp : public OpDef {
public:
    explicit ForeachLikeOp(const char* name) : OpDef(name)
    {
        std::vector<ge::DataType> tensorDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16};
        std::vector<ge::Format> formats(tensorDtypes.size(), ge::FORMAT_ND);
        std::vector<ge::DataType> scalarDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT};

        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType(tensorDtypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("scalars")
            .ScalarList()
            .ParamType(REQUIRED)
            .DataType(scalarDtypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType(tensorDtypes)
            .Format(formats)
            .UnknownShapeFormat(formats)
            .AutoContiguous();
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(ForeachLikeOp);
}  // namespace ops
```

遍历动态输入时用动态输入序号，而不是普通输入序号：

```cpp
uint32_t tensorCount = 0;
uint64_t totalCount = 0;
for (uint32_t i = 0; i < MAX_TENSOR_CONT; ++i) {
    auto tensor = context->GetDynamicInputTensor(0, i);
    if (tensor == nullptr) {
        break;
    }
    totalCount += tensor->GetStorageShape().GetShapeSize();
    ++tensorCount;
}
```

可选输入从 OpDef 到 tiling/infershape 都要保持一致：

```cpp
this->Input("bias")
    .ParamType(OPTIONAL)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
```

```cpp
const gert::Tensor* bias = context->GetOptionalInputTensor(BIAS_INDEX);
bool hasBias = (bias != nullptr && bias->GetShapeSize() > 0);
```

ScalarList 的 dtype 列表经常不是 tensor dtype 原样复制；开源仓若通过 `foreach_utils` 转换，迁移时应本地展开，避免继续依赖公共仓库模块。

## 8. 复杂公共库本地化

当原代码依赖 `tiling/tiling_base.h`、`tiling/tiling_templates_registry.h`、`log/ops_log.h`、`error/ops_error.h` 时，复制本仓同类样例中的 `common/` 和 `tiling_base/` 到当前算子，例如：

- `tasks/level3/Posembedding/RotaryPositionEmbedding/answer/0/op_host/common/log.h`
- `tasks/level3/Posembedding/RotaryPositionEmbedding/answer/0/op_host/tiling_base/`
- `tasks/level3/Moe/MoeFinalizeRoutingV2/answer/0/op_host/common/`

核心替换：

| 原开源仓 | 本地化后 |
|---|---|
| `namespace optiling { class TilingBaseClass ... }` | 通常改为 `namespace OpTiling`，避免和算子自身 `optiling` 冲突 |
| `REGISTER_TILING_TEMPLATE("MyOp", Class, priority)` | `REGISTER_OPS_TILING_TEMPLATE(MyOp, Class, priority)` |
| `optiling::TilingRegistry::GetInstance()` | `OpTiling::TilingRegistry::GetInstance()` 或样例中的 `Ops::Transformer::OpTiling::...` |
| `OPS_ERR_IF`、`OPS_LOG_D`、`OP_TILING_CHECK` | 本地 `common/log.h` 中的 `OP_CHECK_IF`、`OP_LOGD/E/I` |
| `log/ops_log.h`、`error/ops_error.h` | `common/log.h` |

如果原样例按 SocVersion 注册策略，要同时复制 `TilingRegistryNew`、`RegisterNew` 和对应宏，不要只复制旧版 registry。

简化版 `common/log.h` 只适合宏较少的算子：

```cpp
#pragma once
#include <cstdio>

#define OP_LOGD(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while (0)
#define OP_LOGI(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while (0)
#define OP_LOGE(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while (0)

#define unlikely(x) __builtin_expect(!!(x), 0)
#define OP_CHECK_IF(condition, log, returnExpr) \
    do {                                        \
        if (unlikely(condition)) {              \
            log;                                \
            returnExpr;                         \
        }                                       \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    do {                                         \
        if ((ptr) == nullptr) {                  \
            OP_LOGE(context, "nullptr error");   \
            return ge::GRAPH_FAILED;             \
        }                                        \
    } while (0)
```

宏较多时直接复制已迁移同类算子的 `common/log.h`，不要继续 include 开源仓日志头。

## 9. CMake 迁移

原开源仓 CMake 中的下列写法不能直接保留：

```cmake
add_modules_sources(...)
add_onnx_plugin_sources()
DEPENDENCIES foreach_utils
TILING_DIR arch35
COMPUTE_UNIT ascend950
```

选择 CMake 的规则：

| 场景 | 做法 |
|---|---|
| 只有 `op_host/*.cpp` 当前目录源码 | 可用 msopgen 默认 `op_host/CMakeLists.txt` |
| 有 `op_host/common/*.cpp`、`tiling_base/*.cpp`、`op_api/*.cpp` | 写自定义 `op_host/CMakeLists.txt` |
| 手写 `aclnn_*.cpp` | CMake 中显式选择本地 `aclnn_*.cpp`，并 link `opapi` |
| 只有头文件子目录 | 默认 CMake 可能够用，但 include 路径要确认 |
| 使用 `IMPL_OP_OPTILING` 分文件 | 确认 `_tiling.cpp` 被编进 `cust_optiling`，`*_def.cpp` 被 `opbuild` 使用 |

多文件 `op_host/CMakeLists.txt` 的核心是分离 `DEF_CPP`、`TILING_SRC` 和 `INFERSHAPE_SRC`：

```cmake
file(GLOB_RECURSE OPS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE DEF_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*def.cpp")
file(GLOB_RECURSE TILING_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*tiling*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*tiling*.h")
file(GLOB_RECURSE INFERSHAPE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*infershape.cpp")

opbuild(OPS_SRC ${DEF_CPP}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_proto/*.cc)
add_library(cust_op_proto SHARED
    $<$<TARGET_EXISTS:group_proto_src>:${group_proto_src}>
    ${INFERSHAPE_SRC}
    ${ASCEND_AUTOGEN_PATH}/op_proto.cc
)
target_include_directories(cust_op_proto PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_link_libraries(cust_op_proto PRIVATE
    intf_pub exe_graph register tiling_api
    -Wl,--whole-archive rt2_registry -Wl,--no-whole-archive
)

add_library(cust_optiling SHARED ${TILING_SRC})
target_include_directories(cust_optiling PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_link_libraries(cust_optiling PRIVATE
    nnopbase intf_pub exe_graph register tiling_api ops_base
    -Wl,--whole-archive rt2_registry -Wl,--no-whole-archive
)
```

这个片段只展示 proto/tiling 两个核心库的选源和 include/link 规则，不是完整可直接替换的文件。完整 `op_host/CMakeLists.txt` 还要保留 `cust_opapi`、`optiling_compat`、install 等 msopgen 打包逻辑；实际迁移优先参考本仓已通过编译的同类样例。

多文件 tiling 算子不要只满足编译通过，还要确认打包产物包含运行期 tiling 库：

```text
packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}/libcust_opmaster_rt2.0.so
packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/liboptiling.so
packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}/libcust_opsproto_rt2.0.so
packages/vendors/${vendor_name}/op_proto/inc/op_proto.h
```

若 `_CPack_Packages/.../packages/vendors/...` 下缺少这些文件，precision 可能会加载系统内置 tiling 或旧 tiling，表现为 `compileInfoPtr is nullptr`、未命中本地日志或同名内置算子的错误栈。自定义 CMake 中应保留 `cust_op_proto`、`cust_optiling`、`optiling_compat` 和对应 install 规则。

## 10. 迁移检查清单

- [ ] 原开源仓 CMake 已替换为 msopgen 可用 CMake，没有 `add_modules_sources` 等仓库级函数。
- [ ] `_def.cpp` 有 `OP_ADD(OpName)`，Input/Output/Attr 名称、顺序、dtype、format 和原语义一致。
- [ ] 目标芯片至少包含 `ascend910b`，通常包含 `ascend910_93`，未保留 Kirin/ascend950-only/arch35 专用配置。
- [ ] 简单算子只用 `.SetTiling()`；复杂算子只用 `IMPL_OP_OPTILING`。
- [ ] `TilingParse<CompileInfo>()` 与 `context->GetCompileInfo()` 成对存在。
- [ ] `_tiling.h` 的 `REGISTER_TILING_DATA_CLASS` op 名与 `OP_ADD` 一致。
- [ ] TilingData 字段、类型、顺序与 kernel 侧读取一致。
- [ ] 已设置 `SetBlockDim()`；需要多 kernel 分支时已设置 `SetTilingKey()`。
- [ ] 已设置 workspace，且对 workspace 指针判空。
- [ ] 所有开源仓相对 include 已本地化或替换。
- [ ] `op_host` 子目录源码能被 CMake 收集。
- [ ] 没有引用 `nlohmann/json.hpp`、`log/ops_log.h`、`error/ops_error.h`、`../../xxx_utils` 等不可用依赖。
- [ ] InferShape/InferDataType 覆盖动态 shape、多输出、可选输入和 dtype 变化。
