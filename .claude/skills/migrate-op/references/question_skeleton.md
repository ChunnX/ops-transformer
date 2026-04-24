# 题目骨架代码模板

## question/op_host/<opname>.cpp

```cpp
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ops {
class <OpName> : public OpDef {
public:
    explicit <OpName>(const char* name) : OpDef(name)
    {
        // 从 answer 复制 Input/Output 定义，仅保留 ascend910b/ascend910_93
        // 删除 aicoreConfig、TilingParse、ExtendCfgInfo
        // 如果 answer 中有 SetInferShape，也要包含

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(<OpName>);
}
```

## question/op_kernel/<opname>.cpp

```cpp
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void <opname>(GM_ADDR input1, ..., GM_ADDR output1, ..., GM_ADDR workspace, GM_ADDR tiling)
{
}
```

**注意：** kernel 函数签名参数顺序（输入、输出、workspace、tiling）必须与 answer 内核**完全匹配**。
