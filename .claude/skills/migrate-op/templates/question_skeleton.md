# 题目骨架模板

## question/op_host/<kernel_name>.cpp

```cpp
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    context->SetBlockDim(coreNum == 0 ? 1 : coreNum);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class <OpName> : public OpDef {
public:
    explicit <OpName>(const char *name) : OpDef(name)
    {
        // Copy Input/Output/Attr declarations from answer.
        // Keep ascend910b and ascend910_93 configs.
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(<OpName>);
}  // namespace ops
```

## question/op_kernel/<kernel_name>.cpp

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void <kernel_name>(
    GM_ADDR input,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
}
```

Kernel 函数名和 `GM_ADDR` 参数顺序必须与 `answer/0/op_kernel` 完全一致。
