#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "diag_v2_tiling.h"
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;

inline bool IsUnknownShape(const gert::Shape *check_shape)
{
    for (size_t i = 0; i < check_shape->GetDimNum(); i++) {
        if (check_shape->GetDim(i) == UNKNOWN_DIM_VALUE_) {
            return true;
        }
    }
    return false;
}

static inline bool IsUnknownRank(const gert::Shape* check_shape) {
  return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

namespace optiling {

static const int32_t SIZE_2 = 2;
static const int32_t LENGTH_1024 = 1024;

static const int32_t B16_INPUT_LENGTH = 16;

static ge::graphStatus DiagFlatSetTilingData(gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    std::cout << "*******************START*******************" << std::endl;
    std::cout << "totalCoreNum = " << tilingData.get_totalCoreNum() << std::endl;
    std::cout << "usedCoreNum = " << tilingData.get_usedCoreNum() << std::endl;
    std::cout << "normalCoreHandleNum = " << tilingData.get_normalCoreHandleNum() << std::endl;
    std::cout << "lastCoreHandleNum = " << tilingData.get_lastCoreHandleNum() << std::endl;
    std::cout << "inputNum = " << tilingData.get_inputNum() << std::endl;
    std::cout << "diagonal = " << tilingData.get_diagonal() << std::endl;
    std::cout << "ubInputSize = " << tilingData.get_ubInputSize() << std::endl;
    std::cout << "ubOutputSize = " << tilingData.get_ubOutputSize() << std::endl;
    std::cout << "workspaceSize = " << tilingData.get_workspaceSize() << std::endl;
    std::cout << "tilingKey = " << tilingData.get_tilingKey() << std::endl;
    std::cout << "*******************END*******************" << std::endl;

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcScalarTiling(const gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    tilingData.set_totalCoreNum(tilingData.get_totalCoreNum());
    tilingData.set_usedCoreNum(1);
    tilingData.set_normalCoreHandleNum(tilingData.get_inputNum());
    tilingData.set_lastCoreHandleNum(0);

    tilingData.set_tilingKey(SCALAR_UNEQUAL_SIZE16);
    return ge::GRAPH_SUCCESS;
}

static void PrintTilingData(DiagV2TilingData& tilingData)
{
    OP_LOGI("---[DiagFlat]---", "normalCoreHandleNum: %ld", tilingData.get_normalCoreHandleNum());
    OP_LOGI("---[DiagFlat]---", "lastCoreHandleNum: %ld", tilingData.get_lastCoreHandleNum());
    OP_LOGI("---[DiagFlat]---", "totalCoreNum: %ld", tilingData.get_totalCoreNum());
    OP_LOGI("---[DiagFlat]---", "usedCoreNum: %ld", tilingData.get_usedCoreNum());
    OP_LOGI("---[DiagFlat]---", "inputNum: %ld", tilingData.get_inputNum());
    OP_LOGI("---[DiagFlat]---", "diagonal: %ld", tilingData.get_diagonal());
    OP_LOGI("---[DiagFlat]---", "ubInputSize: %ld", tilingData.get_ubInputSize());
    OP_LOGI("---[DiagFlat]---", "ubOutputSize: %ld", tilingData.get_ubOutputSize());
    OP_LOGI("---[DiagFlat]---", "workspaceSize: %ld", tilingData.get_workspaceSize());
    OP_LOGI("---[DiagFlat]---", "tilingKey: %ld", tilingData.get_tilingKey());
}

ge::graphStatus TilingDiagFlat(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingDiagFlat running begin");
    auto input = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    auto inputShape = input->GetStorageShape();
    DiagV2TilingData tilingData;
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, diagonalPtr);
    const int64_t diagonal = *diagonalPtr;
    tilingData.set_diagonal(diagonal);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto totalCoreNum = ascendcPlatform.GetCoreNum();
    tilingData.set_totalCoreNum(totalCoreNum);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto dataType = input->GetDataType();
    auto auxMatrix = SCALAR_THRESHOLD_NUM * SCALAR_THRESHOLD_NUM * sizeof(dataType);
    auto outputDataSize = SCALAR_THRESHOLD_NUM * SCALAR_THRESHOLD_NUM * sizeof(dataType);
    auto inputDataSize = ubSize - auxMatrix - outputDataSize;
    tilingData.set_ubInputSize(inputDataSize);
    tilingData.set_ubOutputSize(outputDataSize);
    tilingData.set_inputNum(inputShape.GetShapeSize());
    
    if (CalcScalarTiling(context, tilingData) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalcScalarTiling fail.");
        return ge::GRAPH_FAILED;
    }
    
    size_t userSize = tilingData.get_totalCoreNum() * 32;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userSize + sysWorkspaceSize;
    tilingData.set_workspaceSize(currentWorkspace[0]);
    if (DiagFlatSetTilingData(context, tilingData) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "DiagFlatSetTilingData set tiling data fail.");
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(tilingData.get_totalCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    context->SetScheduleMode(1); 
    PrintTilingData(tilingData);
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus processDiagFlat(gert::TilingContext* context)
{
    return TilingDiagFlat(context);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    OP_LOGD(context, "Tiling4DiagV2 running begin");
    auto inputShapePtr = context->GetInputShape(0);
    auto inputShape = inputShapePtr->GetStorageShape();
    if (inputShape.GetDimNum() != SIZE_2 && inputShape.GetDimNum() != 1) {
        OP_LOGE(
            context, "Tiling4DiagV2 get input shape dim(=%zu) is not 1 or 2, please check.", inputShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    return processDiagFlat(context);
}
}

namespace ge {

static constexpr size_t DiagV2_IN_X_IDX = 0;
static constexpr size_t DiagV2_OUT_Y_IDX = 0;
static constexpr int64_t INT_DATA_2 = 2;
constexpr size_t kAxisAttrIdx = 0U;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t NEG_TWO = -2;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do DiagV2Infershape.");
    const gert::Shape* input_x_shape = context->GetInputShape(DiagV2_IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_x_shape);

    size_t x_dim_num = input_x_shape->GetDimNum();
    if (x_dim_num > DIM_TWO) {
        OP_LOGD(context, "The dim number of input should be less than 3.");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto pkAxisAttrIdx = attrs->GetInt(kAxisAttrIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, pkAxisAttrIdx);
    int64_t diagonal = *pkAxisAttrIdx;

    gert::Shape* output_y_shape = context->GetOutputShape(DiagV2_OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_y_shape);

    if (IsUnknownRank(input_x_shape)) {
        output_y_shape->SetDimNum(1);
        output_y_shape->SetDim(0, NEG_TWO);
    } else if (IsUnknownShape(input_x_shape)) {
        if (x_dim_num < DIM_TWO) {
            output_y_shape->SetDimNum(DIM_TWO);
            output_y_shape->SetDim(0, NEG_ONE);
            output_y_shape->SetDim(1, NEG_ONE);
        } else {
            output_y_shape->SetDimNum(1);
            output_y_shape->SetDim(0, NEG_ONE);
        }
    } else if (x_dim_num < DIM_TWO) { 
        output_y_shape->SetDimNum(0);
        auto total_element_num = input_x_shape->GetShapeSize();
        for (int64_t k = 0; k < INT_DATA_2; k++) {
            output_y_shape->AppendDim(total_element_num + std::abs(diagonal));
        }
    } else { 
        output_y_shape->SetDimNum(1);
        if (diagonal > 0) {
            output_y_shape->SetDim(0, std::min(input_x_shape->GetDim(0), input_x_shape->GetDim(1) - diagonal));
            if (diagonal >= input_x_shape->GetDim(1)) {
                output_y_shape->SetDim(0, 0);
            }
        } else {
            output_y_shape->SetDim(0, std::min(input_x_shape->GetDim(0) + diagonal, input_x_shape->GetDim(1)));
            if (-diagonal >= input_x_shape->GetDim(0)) {
                output_y_shape->SetDim(0, 0);
            }
        }
    }
    OP_LOGD(context, "End to do DiagV2Infershape.");

    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class DiagV2 : public OpDef {
public:
    explicit DiagV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,
                 ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_DOUBLE, ge::DT_BOOL})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64,
                 ge::DT_UINT64, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_DOUBLE, ge::DT_BOOL})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND});
        this->Attr("diagonal").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(DiagV2);
}