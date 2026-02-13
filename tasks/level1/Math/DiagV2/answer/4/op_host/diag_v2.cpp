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

template <typename T1, typename T2>
T1 CeilDiv(T1 a, T2 b) {
    a = int64_t(a);
    b = int64_t(b);
    return T1(b == 0 ? a : (a + b - 1) / b);
};

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
static const int32_t SIZE_8 = 8;
static const int32_t SIZE_16 = 16;
static const int32_t LENGTH_1024 = 1024;
static const int32_t LENGTH_64 = 64;
static const int32_t BLOCK_SIZE = 32L;

inline static ge::graphStatus DiagV2SetTilingData(gert::TilingContext* context, DiagV2TilingData& tilingData)
{
    auto rawTilingData = context->GetRawTilingData();
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CalcAuxMatrixTiling(const int32_t typeSize, DiagV2TilingData& tilingData)
{
    switch (typeSize) {
        case SIZE_8:
            tilingData.set_tilingKey(static_cast<int64_t>(DiagV2TilingKey::ASSIST_SIZE_8));
            tilingData.set_matrixRowLength(LENGTH_64);
            break;
        default:
            break;
    }

    return ge::GRAPH_SUCCESS;
}

static inline int64_t CalcLeastNumPerCore(const int32_t typeSize, const gert::TilingContext* context)
{
    if (typeSize <= 0) {
        OP_LOGE(context, "Tiling4DiagV2 typeSize is invalid %d, please check.");
        return -1;
    }
    int64_t leastNumPerCore = BLOCK_SIZE / typeSize;
    return (leastNumPerCore > 0) ? leastNumPerCore : 1;
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

    DiagV2TilingData tilingData;
    tilingData.set_xHeight(inputShape.GetDim(0));
    tilingData.set_xWidth(inputShape.GetDim(1));
    auto attrs = context->GetAttrs();
    const int64_t* diagonalPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t diagonal = *diagonalPtr;
    if ((diagonal >= 0 && diagonal > tilingData.get_xWidth()) ||
        (diagonal < 0 && std::abs(diagonal) > tilingData.get_xHeight())) {
            OP_LOGE(
            context, "Tiling4DiagV2 attr diagonal(=%ld) is wrong, please check. w=%ld, h=%ld", diagonal,
            tilingData.get_xWidth(), tilingData.get_xHeight());
            return ge::GRAPH_FAILED;
    }
    tilingData.set_gmOffset(diagonal > 0 ? diagonal : std::abs(diagonal) * tilingData.get_xWidth());
    tilingData.set_numOut(
        (diagonal >= 0) ? std::min((tilingData.get_xWidth() - diagonal), tilingData.get_xHeight()) :
                          std::min((tilingData.get_xHeight() + diagonal), tilingData.get_xWidth()));
    auto inputDesc = context->GetInputDesc(0);
    auto dataType = inputDesc->GetDataType();
    const int32_t typeSize = ge::GetSizeByDataType(dataType);
    int64_t leastNumPerCore = CalcLeastNumPerCore(typeSize, context);
    if (leastNumPerCore <= 0) {
        OP_LOGE(context, "Tiling4DiagV2 leastNumPerCore is invalid, please check.");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    int64_t tmpRealCoreNum = tilingData.get_numOut() < coreNum ? tilingData.get_numOut() : coreNum;
    int64_t tmpCorePerNum = CeilDiv(tilingData.get_numOut(), tmpRealCoreNum);
    tilingData.set_numPerCore(CeilDiv(tmpCorePerNum, leastNumPerCore) * leastNumPerCore);
    tilingData.set_realCoreNum(CeilDiv(tilingData.get_numOut(), tilingData.get_numPerCore()));
    tilingData.set_tailNum(tilingData.get_numOut() - (tilingData.get_realCoreNum() - 1) * tilingData.get_numPerCore());
    if (CalcAuxMatrixTiling(typeSize, tilingData) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "CalcAuxMatrixTiling fail.");
        return ge::GRAPH_FAILED;
    }
    if (DiagV2SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "DiagV2SetTilingData set tiling data fail.");
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    OP_LOGD(
        context,
        "tilingData is xWidth:%ld, xHeight:%ld, gmOffset:%ld, numOut:%ld, realCoreNum:%ld, numPerCore:%ld, tailNum:%ld,\
                              tilingKey:%ld, matrixRowLength:%ld",
        tilingData.get_xWidth(), tilingData.get_xHeight(), tilingData.get_gmOffset(), tilingData.get_numOut(),
        tilingData.get_realCoreNum(), tilingData.get_numPerCore(), tilingData.get_tailNum(), tilingData.get_tilingKey(),
        tilingData.get_matrixRowLength());

    return ge::GRAPH_SUCCESS;
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