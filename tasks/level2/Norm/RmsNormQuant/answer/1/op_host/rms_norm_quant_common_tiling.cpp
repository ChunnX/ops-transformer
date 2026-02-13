/*!
 * \file rms_norm_quant_common_tiling.cpp
 * \brief
 */
#include <climits>
#include <sstream>
#include "common/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "rms_norm_quant_common_tiling.h"

namespace optiling {
// const defination
static constexpr uint32_t BYTE_32 = 32;

static constexpr uint32_t SLICE_SIZE = 8192;
static constexpr uint32_t SLICE_SIZE_WITH_BIAS = 8192;
static constexpr uint32_t FP16_PER_BLOCK = 16;
static constexpr size_t WORKAPCE_RESERVE = 16 * 1024 * 1024;
static constexpr uint32_t BIT_OFFSET_SIX = 6;
constexpr uint32_t DST_TYPE_INDEX = 3;
constexpr uint32_t CONST_ROUR = 8;

enum class PrePostType : uint32_t
{
    Norm = 0,
};

// function defination
uint32_t CalcNormTilingKey(NormCommonTilingData1* tilingDataPtr, ge::DataType dtype, bool useBeta)
{
    uint32_t tilingKey = static_cast<uint32_t>(tilingDataPtr->get_gemmaMode() ? 1 : 0);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(useBeta ? 1 : 0);
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(tilingDataPtr->get_highPrecisionMode() ? 1 : 0);
    bool useSlice = tilingDataPtr->get_numCol() > tilingDataPtr->get_sliceSize();
    tilingKey = (tilingKey << 1) + static_cast<uint32_t>(useSlice ? 1 : 0);
    tilingKey = (tilingKey << BIT_OFFSET_SIX) + static_cast<uint32_t>(dtype);
    return tilingKey;
};

static ge::graphStatus PrepareForRmsNormCommonTiling(gert::TilingParseContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNumAiv <= 0), OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF(ubSizePlatForm <= 0, OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQuantAttrs([[maybe_unused]]gert::TilingContext* context, NormCommonTilingData1* tilingDataPtr)
{
    const char* quantMinFlagPtr = std::getenv("ASDOPS_QUANT_MIN_NEG_127");
    if (quantMinFlagPtr != nullptr && strcmp(quantMinFlagPtr, "1") == 0) {
        tilingDataPtr->set_quantMin(-127); // set int8 min to -127
    } else {
        tilingDataPtr->set_quantMin(std::numeric_limits<int8_t>::min());
    }
    tilingDataPtr->set_scale(1.0f);
    tilingDataPtr->set_offset(0);
    return ge::GRAPH_SUCCESS;
}

void LogTilingData(const std::string& opName, NormCommonTilingData1* tilingDataPtr)
{
    std::stringstream tilingDataLogStringSS;
    tilingDataLogStringSS << opName << " Tiling Data: "
                          << " numCore " << tilingDataPtr->get_numCore() << " numCol " << tilingDataPtr->get_numCol()
                          << " numRow " << tilingDataPtr->get_numRow() << " avgFactor "
                          << tilingDataPtr->get_avgFactor() << " epsilon " << tilingDataPtr->get_epsilon()
                          << " sliceSize " << tilingDataPtr->get_sliceSize() << " precisionMode "
                          << tilingDataPtr->get_highPrecisionMode();
    OP_LOGD(opName, "%s", tilingDataLogStringSS.str().c_str());
}

inline static int64_t RoundUp(int64_t a, int64_t b)
{
    return Ops::Base::CeilDiv(a, b) * b;
}

template <bool EN_QUANT>
static uint32_t CalcMaxSliceSize(PrePostType prePostType, bool enBias, const uint32_t ubSize)
{
    if (prePostType == PrePostType::Norm) {
        if (EN_QUANT) {
            return SLICE_SIZE;
        }
    }
    return SLICE_SIZE;
}

template <bool EN_QUANT>
static ge::graphStatus NormCommonTiling(gert::TilingContext* context, PrePostType prePostType)
{
    NormCommonTilingData1 tilingData;

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNumAiv <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF(ubSizePlatForm <= 0, OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);

    std::string opName = context->GetNodeName();
    gert::Shape xShape = context->GetInputShape(static_cast<int>(NormInputIndex::X))->GetOriginShape();

    int64_t numRow = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; ++i) {
        uint32_t dim = xShape.GetDim(i);
        OP_CHECK_IF(dim <= 0, OP_LOGD(opName, "In tensor X dim %zu is invalid", i), return ge::GRAPH_PARAM_OUT_OF_RANGE);
        OP_CHECK_IF(
            numRow > std::numeric_limits<uint32_t>::max() / dim,
            OP_LOGD(opName, "At dim %zu, tmpNumRow is overflowed", i), return ge::GRAPH_FAILED);
        numRow *= dim;
    }
    tilingData.set_numRow(static_cast<uint32_t>(numRow));
    uint32_t numCol = xShape.GetDim(xShape.GetDimNum() - 1);
    OP_CHECK_IF(
        numCol > std::numeric_limits<uint32_t>::max(), OP_LOGD(opName, "numCol is overflowed"),
        return ge::GRAPH_PARAM_OUT_OF_RANGE);
    tilingData.set_numCol(numCol);
    tilingData.set_numCore(
        Ops::Base::CeilDiv(tilingData.get_numRow(), Ops::Base::CeilDiv(tilingData.get_numRow(), coreNumAiv)));
    OP_CHECK_IF(
        tilingData.get_numCore() > std::numeric_limits<uint32_t>::max() - tilingData.get_numRow(),
        OP_LOGD(opName, "numRow + numCore is overflowed"), return ge::GRAPH_PARAM_OUT_OF_RANGE);
    auto betaTensorDesc = context->GetInputDesc(static_cast<int>(NormInputIndex::BETA));
    bool enableBeta = betaTensorDesc != nullptr;
    if (enableBeta) {
        enableBeta = context->GetInputShape(static_cast<int>(NormInputIndex::BETA))->GetOriginShape().GetDimNum() != 0;
    }

    auto yDtype = context->GetOutputDesc(static_cast<int>(NormOutputIndex::Y))->GetDataType();
    if (yDtype == ge::DT_INT4 && numCol % static_cast<int32_t>(CONST_ROUR) != 0) {
      OP_LOGE(context->GetNodeName(), "if y datatype is int4, the last dim(%u) of x should be divisible by 8.", numCol);
      return ge::GRAPH_FAILED;
    }

    const uint32_t maxSliceSize = CalcMaxSliceSize<EN_QUANT>(prePostType, enableBeta, ubSize);
    tilingData.set_sliceSize(maxSliceSize);
    OP_CHECK_IF(
        tilingData.get_numCol() > std::numeric_limits<uint32_t>::max() - tilingData.get_sliceSize(),
        OP_LOGD(opName, "numCol + sliceSize is overflowed"), return ge::GRAPH_PARAM_OUT_OF_RANGE);
    OP_CHECK_IF(numCol <= 0, OP_LOGE(opName, "The numCol size is neg: %ud.", numCol), return ge::GRAPH_PARAM_INVALID);
    float avgFactor = static_cast<float>(1.0 / numCol);
    tilingData.set_avgFactor(avgFactor);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const float* epsilon = attrs->GetFloat(static_cast<int>(NormAttrIndex::EPSILON));
    OP_CHECK_IF(*epsilon <= 0, OP_LOGD(opName, "epsilon should be positive"), return ge::GRAPH_PARAM_INVALID);
    tilingData.set_epsilon(*epsilon);
    const bool* highPrecisionMode = attrs->GetBool(static_cast<int>(NormAttrIndex::HIGH_PRECISION_MODE));

    tilingData.set_highPrecisionMode(highPrecisionMode && *highPrecisionMode);
    const bool* gemmaMode = attrs->GetBool(static_cast<int>(NormAttrIndex::GEMMA_MODE));
    tilingData.set_gemmaMode(gemmaMode && *gemmaMode);

    auto dstType = static_cast<int64_t>(ge::DT_INT8);
    auto dstTypePtr = attrs->GetInt(static_cast<int>(NormAttrIndex::DST_TYPE));
    if (dstTypePtr != nullptr) {
        dstType = static_cast<int64_t>(*dstTypePtr);
    }
    OP_CHECK_IF((dstType != ge::DT_INT8 && dstType != ge::DT_INT4), OP_LOGE(opName, "dstType can only be 2 or 29."), return ge::GRAPH_PARAM_INVALID);
    tilingData.set_dstType(dstType);

    if (EN_QUANT) {
        GetQuantAttrs(context, &tilingData);
    }

    ge::DataType dtype = context->GetInputDesc(static_cast<int>(NormInputIndex::X))->GetDataType();

    uint32_t tilingKey = CalcNormTilingKey(&tilingData, dtype, enableBeta);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + WORKAPCE_RESERVE;

    OP_LOGD(opName, "tiling key is %d", tilingKey);
    context->SetBlockDim(tilingData.get_numCore());
    context->SetTilingKey(tilingKey);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    LogTilingData(opName, &tilingData);
    return ge::GRAPH_SUCCESS;
}

template <bool EN_QUANT>
static ge::graphStatus RmsNormTilingFunc(gert::TilingContext* context)
{
    std::string opName = context->GetNodeName();
    auto ret = NormCommonTiling<EN_QUANT>(context, PrePostType::Norm);
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGD(opName, "RmsNormTilingFunc running error!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNormQuant)
    .Tiling(RmsNormTilingFunc<true>)
    .TilingParse<NormCommonCompileInfo>(PrepareForRmsNormCommonTiling);
} // namespace optiling