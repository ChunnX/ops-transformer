#include "data_copy_example.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling{
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto nodeName = context->GetNodeName();
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t UbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, UbSize);
    std::cout << "------coreNum: " << coreNum << " UbSize: " << UbSize << std::endl;

    DataCopyExampleTilingData tiling;
    // 从context中获取输入数据[T, N, D]
    auto xShapePtr = context->GetInputShape(0);
    auto xShape = xShapePtr->GetStorageShape();
    auto xDtypePtr = context->GetInputDesc(0);
    auto xDtype = xDtypePtr->GetDataType();
    if (xDtype != ge::DT_FLOAT) {
        std::cout << "datatype of input must be in [float32]" << std::endl;
        return ge::GRAPH_FAILED;
    }
    if(xShape.GetDimNum() != 3){
        std::cout << "check xshape failed, the dim of x1 not equal 3" << std::endl;
        return ge::GRAPH_FAILED;
    }
    auto shapeT = xShape.GetDim(0);
    auto shapeN = xShape.GetDim(1);
    auto shapeD = xShape.GetDim(2);
    if (shapeT >= 128*1024) {
        std::cout << "check shapeT failed, T must < 128*1024" << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (shapeN >= 100) {
        std::cout << "check shapeN failed, N must < 100" << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (shapeD > 256) {
        std::cout << "check shapeD failed, D must <= 256" << std::endl;
        return ge::GRAPH_FAILED;
    }

    // 从context中获取act_seq_len
    auto yShapePtr = context->GetInputShape(1);
    auto yShape = yShapePtr->GetStorageShape();
    if(yShape.GetDimNum() != 1) {
        std::cout << "check yshape failed, the dim of x1 not equal 1" << std::endl;
        return ge::GRAPH_FAILED;
    }
    auto B = yShape.GetDim(0) - 1;
    if (B < 1) {
        std::cout << "check B failed, D must >= 1" << std::endl;
        return ge::GRAPH_FAILED;
    }
    // tiling切分：每个B的T切24，ND合轴后切1024
    auto shapeTAlign = 24;
    auto shapeNDAlign = 1024;

    tiling.set_shapeT(shapeT);
    tiling.set_shapeN(shapeN);
    tiling.set_shapeD(shapeD);
    tiling.set_B(B);
    tiling.set_shapeTAlign(shapeTAlign);
    tiling.set_shapeNDAlign(shapeNDAlign);
    tiling.set_useCoreNum(coreNum);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1); // c和v通信，中间存储数据
    currentWorkspace[0] = 16 * 1024 * 1024;

    // 打印校验
    std::cout << "<<<<<< host tiling results >>>>>>" << std::endl;
    std::cout << "shapeT " << shapeT << std::endl;
    std::cout << "shapeN " << shapeN << std::endl;
    std::cout << "shapeD " << shapeD << std::endl;
    std::cout << "B " << B << std::endl;
    std::cout << "shapeTAlign " << shapeTAlign << std::endl;
    std::cout << "shapeNDAlign " << shapeNDAlign << std::endl;
    std::cout << "useCoreNum " << coreNum << std::endl;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto xShape = context->GetInputShape(0);
    auto shapeT = xShape->GetDim(0);
    auto shapeN = xShape->GetDim(1);
    auto shapeD = xShape->GetDim(2);
    
    gert::Shape* z_Shape = context->GetOutputShape(0);
    z_Shape->SetDimNum(3);
    z_Shape->SetDim(0, shapeT);
    z_Shape->SetDim(1, shapeN);
    z_Shape->SetDim(2, shapeD);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType xDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDataType);
    return GRAPH_SUCCESS;
}
}

namespace ops{
class DataCopyExample : public OpDef {
public:
    explicit DataCopyExample(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(DataCopyExample);
}