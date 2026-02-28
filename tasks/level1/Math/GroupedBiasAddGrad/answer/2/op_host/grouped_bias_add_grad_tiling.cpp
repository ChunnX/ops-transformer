#include <sstream>
#include "register/op_impl_registry.h"
#include "grouped_bias_add_grad_tiling_base.h"
#include "grouped_bias_add_grad_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace groupedbiasaddgrad {

static const uint32_t ATTR_GROUP_IDX_TYPE_INDEX = 0;

#define OP_LOGE(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGI(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)

template <typename T>
inline auto CeilDiv(T num, T div) -> T
{
    return div == 0 ? 0 : (num + div - 1) / div;
}

ge::graphStatus GroupedBiasAddGradTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] GetPlatformInfo start running.");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

    baseInfoOp_.vectorCoreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(
        (baseInfoOp_.vectorCoreNum <= 0),
        OP_LOGE(nodeName_, "GroupedBiasAddGradTiling get num of vector core is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, baseInfoOp_.ubSize);
    OP_CHECK_IF(
        (baseInfoOp_.ubSize <= 0),
        OP_LOGE(nodeName_, "GroupedBiasAddGradTiling get ub size is less than or equal to 0."),
        return ge::GRAPH_FAILED);

    baseInfoOp_.ubSize -= RESERVED_UB_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::GetInputInfo()
{
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] GetInputInfo start running.");
    auto gradYInputDesc = context_->GetInputDesc(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYInputDesc);
    baseInfoOp_.gradYInputDtype = gradYInputDesc->GetDataType();
    OP_CHECK_IF(
        (baseInfoOp_.gradYInputDtype != ge::DT_FLOAT && baseInfoOp_.gradYInputDtype != ge::DT_FLOAT16 &&
         baseInfoOp_.gradYInputDtype != ge::DT_BF16),
        OP_LOGE(nodeName_, "the dtype of input grad_y should be one of FP32/FP16/BF16."), return ge::GRAPH_FAILED);

    auto gradYInputShapePtr = context_->GetInputShape(GRAD_Y_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYInputShapePtr);
    auto gradYInputShape = gradYInputShapePtr->GetStorageShape();
    baseInfoOp_.gradYDimNum = gradYInputShape.GetDimNum();

    auto groupIdxInputShapePtr = context_->GetOptionalInputShape(GROUP_IDX_INPUT_INDEX);
    if (groupIdxInputShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    } else {
        OP_CHECK_IF(
            (baseInfoOp_.gradYDimNum != TWO_NUM || groupIdxInputShapePtr->GetStorageShape().GetDimNum() != 1),
            OP_LOGE(
                nodeName_,
                "the input grad_y should be 2D tensor when group_idx is not null. and group_idx must be 1D tensor."),
            return ge::GRAPH_FAILED);
        baseInfoOp_.existGroupIdx = 1;
        baseInfoOp_.dimGB = gradYInputShape.GetDim(0);
        baseInfoOp_.dimG = groupIdxInputShapePtr->GetStorageShape().GetShapeSize();
        baseInfoOp_.dimH = gradYInputShape.GetDim(1);
        auto groupIdxInputDesc = context_->GetInputDesc(GROUP_IDX_INPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, groupIdxInputDesc);
        baseInfoOp_.groupIdxInputDtype = groupIdxInputDesc->GetDataType();
        OP_CHECK_IF(
            (baseInfoOp_.groupIdxInputDtype != ge::DT_INT32 && baseInfoOp_.groupIdxInputDtype != ge::DT_INT64),
            OP_LOGE(nodeName_, "the dtype of input group_idx should be INT32 or INT64."), return ge::GRAPH_FAILED);
    }

    if (baseInfoOp_.existGroupIdx == 1 && baseInfoOp_.dimG > INPUT_MAX_GROUP) {
        OP_LOGE(
            nodeName_, "input group_idx shape not support more than %ld, but got %ld.", INPUT_MAX_GROUP,
            baseInfoOp_.dimG);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::GetAttrInfo()
{
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] GetAttrInfo start running.");
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrGroupIdxType = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_IDX_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrGroupIdxType);

    baseInfoOp_.groupIdxType = static_cast<int32_t>(*attrGroupIdxType);
    OP_CHECK_IF(
        (baseInfoOp_.groupIdxType != 0 && baseInfoOp_.groupIdxType != 1),
        OP_LOGE(nodeName_, "the value of group_idx_type should be 0 or 1, but got %d", baseInfoOp_.groupIdxType),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::CheckOutput()
{
    auto gradBiasOutputShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradBiasOutputShapePtr);
    auto gradBiasOutputShape = gradBiasOutputShapePtr->GetStorageShape();
    OP_CHECK_IF(
        (gradBiasOutputShape.GetDimNum() != TWO_NUM),
        OP_LOGE(nodeName_, "the dim of grad_bias should be 2, but got %zu.", gradBiasOutputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    auto gradBiasdim0 = gradBiasOutputShape.GetDim(0);
    auto gradBiasdim1 = gradBiasOutputShape.GetDim(1);
    OP_CHECK_IF(
        ((gradBiasdim0 != baseInfoOp_.dimG) || (gradBiasdim1 != baseInfoOp_.dimH)),
        OP_LOGE(
            nodeName_, "the shape of grad_bias should be [%ld, %ld], bug got [%ld, %ld].", baseInfoOp_.dimG,
            baseInfoOp_.dimH, gradBiasdim0, gradBiasdim1),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::DoSplitTiling()
{
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] DoUnequalCTiling start running.");
    int64_t coreNum = baseInfoOp_.vectorCoreNum;
    int64_t totalSplitNum = CeilDiv(baseInfoOp_.dimH, splitCoreOp_.baseH) * baseInfoOp_.dimG;
    splitCoreOp_.usedCoreNum = totalSplitNum > coreNum ? coreNum : totalSplitNum;
    splitCoreOp_.normalCoreProcessNum = CeilDiv(totalSplitNum, splitCoreOp_.usedCoreNum);
    splitCoreOp_.tailCoreProcessNum = splitCoreOp_.normalCoreProcessNum - 1;
    int64_t tailCoreNum = splitCoreOp_.normalCoreProcessNum * splitCoreOp_.usedCoreNum - totalSplitNum;
    splitCoreOp_.normalCoreNum = splitCoreOp_.usedCoreNum - tailCoreNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::DoUnequalCPerformanceTiling()
{
    OP_LOGD(nodeName_, "[GroupedBiasAddGrad] DoUnequalCTiling start running.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedBiasAddGradTiling::DoTiling()
{
    splitCoreOp_.baseH = H_BASE_SIZE / sizeof(float);
    int64_t avilableUbsize = baseInfoOp_.ubSize - H_BASE_SIZE * BUFFER_NUM - UB_GROUP_SUM_NUM * H_BASE_SIZE;
    splitCoreOp_.baseC = avilableUbsize / ACTIVE_NODES_NUM / sizeof(float) / splitCoreOp_.baseH;
    DoSplitTiling();
    
    int64_t unitLength = baseInfoOp_.groupIdxInputDtype == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int64_t);
    {
        int64_t groupIdxAlign = (baseInfoOp_.dimG * unitLength + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        avilableUbsize = avilableUbsize - groupIdxAlign - groupIdxAlign;
        splitCoreOp_.baseC = avilableUbsize / ACTIVE_NODES_NUM / sizeof(float) / splitCoreOp_.baseH;
        splitCoreOp_.wsUnitNum = CeilDiv(baseInfoOp_.dimGB, splitCoreOp_.baseC);
    }
    return ge::GRAPH_SUCCESS;
}

void GroupedBiasAddGradTiling::SaveToTilingData()
{
    tilingData_.set_usedCoreNum(splitCoreOp_.usedCoreNum);
    tilingData_.set_normalCoreNum(splitCoreOp_.normalCoreNum);
    tilingData_.set_normalCoreProcessNum(splitCoreOp_.normalCoreProcessNum);
    tilingData_.set_tailCoreProcessNum(splitCoreOp_.tailCoreProcessNum);
    tilingData_.set_wsUnitNum(splitCoreOp_.wsUnitNum);
    tilingData_.set_dimG(baseInfoOp_.dimG);
    tilingData_.set_dimC(baseInfoOp_.dimC);
    tilingData_.set_dimH(baseInfoOp_.dimH);
    tilingData_.set_dimGB(baseInfoOp_.dimGB);
    tilingData_.set_baseH(splitCoreOp_.baseH);
    tilingData_.set_baseC(splitCoreOp_.baseC);
    tilingData_.set_loopCNum(splitCoreOp_.loopCNum);
    tilingData_.set_groupIdxType(baseInfoOp_.groupIdxType);
}

ge::graphStatus GroupedBiasAddGradTiling::PostTiling()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t workspaceSize = WORKSPACE_BASE_CAL;
    workspaces[0] = workspaceSize;
    workspaces[0] += splitCoreOp_.wsUnitNum * H_BASE_SIZE * splitCoreOp_.usedCoreNum;

    SaveToTilingData();

    context_->SetBlockDim(splitCoreOp_.usedCoreNum);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedBiasAddGradTiling::ComputeTiling(const std::vector<uint32_t>& args) const
{
    uint64_t result = 1000000UL;
    uint64_t startValue = 1;
    constexpr uint64_t incrementCoefficient = 10;
    for (auto it = args.rbegin(); it != args.rend(); ++it) {
        result += *it * startValue;
        startValue *= incrementCoefficient;
    }
    return result;
}

uint64_t GroupedBiasAddGradTiling::GetTilingKey() const
{
    DtypeEnum inDtype = DtypeEnum::FLOAT16;
    if (baseInfoOp_.gradYInputDtype == ge::DT_BF16) {
        inDtype = DtypeEnum::BFLOAT16;
    }
    if (baseInfoOp_.gradYInputDtype == ge::DT_FLOAT) {
        inDtype = DtypeEnum::FLOAT32;
    }

    uint32_t useUBSum = 0;
    if (splitCoreOp_.wsUnitNum <= UB_GROUP_SUM_NUM) {
        useUBSum = 1;
    }

    uint32_t groupIdxDtype = baseInfoOp_.groupIdxInputDtype == ge::DT_INT32 ? 0 : 1;
    auto tilingKey = ComputeTiling(
        {static_cast<uint32_t>(groupIdxDtype), baseInfoOp_.performance, useUBSum, baseInfoOp_.existGroupIdx,
         static_cast<uint32_t>(inDtype)});
    OP_LOGI(nodeName_, "[GroupedBiasAddGrad] GetTilingKey [%lu].", tilingKey);
    return tilingKey;
}

void GroupedBiasAddGradTiling::DumpTilingInfo() const
{
    std::ostringstream info;
    info << "baseInfoOp_.vectorCoreNum: " << baseInfoOp_.vectorCoreNum << std::endl;
    info << "baseInfoOp_.gradYDimNum: " << baseInfoOp_.gradYDimNum << std::endl;
    info << "baseInfoOp_.dimG: " << baseInfoOp_.dimG << std::endl;
    info << "baseInfoOp_.dimC: " << baseInfoOp_.dimC << std::endl;
    info << "baseInfoOp_.dimH: " << baseInfoOp_.dimH << std::endl;
    info << "baseInfoOp_.dimGB: " << baseInfoOp_.dimGB << std::endl;
    info << "baseInfoOp_.gradYInputDtype: " << baseInfoOp_.gradYInputDtype << std::endl;
    info << "baseInfoOp_.existGroupIdx: " << baseInfoOp_.existGroupIdx << std::endl;
    info << "baseInfoOp_.performance: " << baseInfoOp_.performance << std::endl;
    info << "baseInfoOp_.groupIdxType: " << baseInfoOp_.groupIdxType << std::endl;

    info << "splitCoreOp_.usedCoreNum: " << splitCoreOp_.usedCoreNum << std::endl;
    info << "splitCoreOp_.normalCoreNum: " << splitCoreOp_.normalCoreNum << std::endl;
    info << "splitCoreOp_.normalCoreProcessNum: " << splitCoreOp_.normalCoreProcessNum << std::endl;
    info << "splitCoreOp_.tailCoreProcessNum: " << splitCoreOp_.tailCoreProcessNum << std::endl;
    info << "splitCoreOp_.baseH: " << splitCoreOp_.baseH << std::endl;
    info << "splitCoreOp_.baseC: " << splitCoreOp_.baseC << std::endl;
    info << "splitCoreOp_.loopCNum: " << splitCoreOp_.loopCNum << std::endl;
    info << "splitCoreOp_.wsUnitNum: " << splitCoreOp_.wsUnitNum << std::endl;

    OP_LOGI(nodeName_, "%s", info.str().c_str());

    std::cout << "*******************START*******************" << std::endl;
    std::cout << "usedCoreNum = " << splitCoreOp_.usedCoreNum << std::endl;
    std::cout << "normalCoreNum = " << splitCoreOp_.normalCoreNum << std::endl;
    std::cout << "normalCoreProcessNum = " << splitCoreOp_.normalCoreProcessNum << std::endl;
    std::cout << "tailCoreProcessNum = " << splitCoreOp_.tailCoreProcessNum << std::endl;
    std::cout << "wsUnitNum = " << splitCoreOp_.wsUnitNum << std::endl;
    std::cout << "dimG = " << baseInfoOp_.dimG << std::endl;
    std::cout << "dimC = " << baseInfoOp_.dimC << std::endl;
    std::cout << "dimH = " << baseInfoOp_.dimH << std::endl;
    std::cout << "dimGB = " << baseInfoOp_.dimGB << std::endl;
    std::cout << "baseH = " << splitCoreOp_.baseH << std::endl;
    std::cout << "baseC = " << splitCoreOp_.baseC << std::endl;
    std::cout << "loopCNum = " << splitCoreOp_.loopCNum << std::endl;
    std::cout << "groupIdxType = " << baseInfoOp_.groupIdxType << std::endl;
    std::cout << "*******************END*******************" << std::endl;
}

ge::graphStatus GroupedBiasAddGradTiling::RunGroupedBiasAddGradTiling()
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ret = GetInputInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetAttrInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = DoTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGroupedBiasAddGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(
        context == nullptr, OP_LOGE("GroupedBiasAddGrad", "context should not be nullptr."), return ge::GRAPH_FAILED);

    GroupedBiasAddGradTiling tiling(context);
    return tiling.RunGroupedBiasAddGradTiling();
}

ge::graphStatus TilingPrepareForGroupedBiasAddGrad(gert::TilingParseContext* context)
{
    OP_CHECK_IF(
        context == nullptr, OP_LOGE("GroupedBiasAddGrad", "context should not be nullptr."), return ge::GRAPH_FAILED);
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "TilingPrepareForGroupedBiasAddGrad start.");

    auto compileInfo = context->GetCompiledInfo<GroupedBiasAddGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(nodeName, "Failed to get core number."), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(nodeName, "Failed to get ub size."), return ge::GRAPH_FAILED);

    OP_LOGD(nodeName, "TilingPrepareForGroupedBiasAddGrad end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedBiasAddGrad)
    .Tiling(TilingForGroupedBiasAddGrad)
    .TilingParse<GroupedBiasAddGradCompileInfo>(TilingPrepareForGroupedBiasAddGrad);
} // namespace groupedbiasaddgrad
} // namespace optiling