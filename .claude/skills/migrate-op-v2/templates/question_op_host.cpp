#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// 占位 tiling 函数：给 LLM 作为骨架输入，返回成功并设置最小 block dim
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // TODO: 实现完整的分块参数计算
    // 1. 获取平台信息
    //    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    //    uint64_t ubSize;
    //    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // 2. 获取输入 tensor shape 和 dtype
    //    context->GetInputTensor(0)->GetShapeSize() / GetDataType()
    // 3. 计算分块参数
    // 4. 设置 block dim
    // 5. SaveToBuffer + SetDataSize
    // 6. 设置 workspace

    context->SetBlockDim(1);

    // TODO: 用实际 TilingData 结构体填充并写入
    // MyOpTilingData tiling;
    // tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
    //                     context->GetRawTilingData()->GetCapacity());
    // context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

// OpDef 骨架：Input/Output 名称和类型需要根据实际算子填充
class <OpName> : public OpDef {
public:
    explicit <OpName>(const char* name) : OpDef(name)
    {
        // TODO: 根据 answer 实现填充正确的 Input/Output/Attr 定义
        // 示例：
        // this->Input("x")
        //     .ParamType(REQUIRED)
        //     .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
        //     .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // this->Output("y")
        //     .ParamType(REQUIRED)
        //     .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
        //     .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 仅在模式1/2a（使用 .SetTiling() 注册 tiling）时取消下面这行的注释。
        // 如果 answer 使用 IMPL_OP_OPTILING（模式2b/3），保持注释状态——tiling 由 _tiling.cpp 注册。
        // this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(<OpName>);

}  // namespace ops
