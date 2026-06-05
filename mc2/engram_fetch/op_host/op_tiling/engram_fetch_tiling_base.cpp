#include "engram_fetch_tiling_base.h"
#include "mc2_log.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/moe_tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"

using namespace Ops::Transformer::OpTiling;

namespace optiling {

static constexpr size_t ENGRAM_FETCH_INPUT_BUFFERHANDLE_INDEX = 0;
static constexpr size_t ENGRAM_FETCH_INPUT_INDICES_INDEX = 1;

static constexpr size_t ENGRAM_FETCH_ATTR_NUM_ENTRIES_PER_RANK_INDEX = 0;
static constexpr size_t ENGRAM_FETCH_ATTR_HIDDEN_SIZE_INDEX = 1;
static constexpr size_t ENGRAM_FETCH_ATTR_GROUP_INDEX = 2;
static constexpr size_t ENGRAM_FETCH_ATTR_OUTPUT_DTYPE_INDEX = 3;

ge::graphStatus EngramFetchTilingBase::CheckRequiredAttrValue()
{
    const auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, attrs, ge::GRAPH_FAILED);

    const int64_t *numEntriesPerRank = attrs->GetInt(ENGRAM_FETCH_ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    OP_CHECK_IF(numEntriesPerRank == nullptr || *numEntriesPerRank <= 0,
                OP_LOGE(nodeName_, "num_entries_per_rank attr invalid"), return ge::GRAPH_FAILED);

    const int64_t *hiddenSize = attrs->GetInt(ENGRAM_FETCH_ATTR_HIDDEN_SIZE_INDEX);
    OP_CHECK_IF(hiddenSize == nullptr || *hiddenSize <= 0, OP_LOGE(nodeName_, "hidden_size attr invalid"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramFetchTilingBase::GetAttrAndSetTilingData()
{
    const auto attrs = context_->GetAttrs();

    tilingData_->engramFetchInfo.numEntriesPerRank =
        static_cast<uint32_t>(*attrs->GetInt(ENGRAM_FETCH_ATTR_NUM_ENTRIES_PER_RANK_INDEX));
    tilingData_->engramFetchInfo.hiddenDim = static_cast<uint32_t>(*attrs->GetInt(ENGRAM_FETCH_ATTR_HIDDEN_SIZE_INDEX));

    const char *groupPtr = attrs->GetAttrPointer<char>(ENGRAM_FETCH_ATTR_GROUP_INDEX);
    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE(nodeName_, "group attr is null"), return ge::GRAPH_FAILED);
    group_ = std::string(groupPtr);

    const int64_t *outputDtype = attrs->GetInt(ENGRAM_FETCH_ATTR_OUTPUT_DTYPE_INDEX);
    tilingData_->engramFetchInfo.outputDtype =
        (outputDtype != nullptr) ? static_cast<int32_t>(*outputDtype) : static_cast<int32_t>(ge::DT_BF16);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramFetchTilingBase::CheckInputTensorShape()
{
    const gert::StorageShape *indicesShape = context_->GetInputShape(ENGRAM_FETCH_INPUT_INDICES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, indicesShape, ge::GRAPH_FAILED);
    OP_CHECK_IF(indicesShape->GetStorageShape().GetDimNum() != 1,
                OP_LOGE(nodeName_, "indices must be 1D, got %lu dims", indicesShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramFetchTilingBase::CheckInputTensorDataType()
{
    const auto indicesDesc = context_->GetInputDesc(ENGRAM_FETCH_INPUT_INDICES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context_, indicesDesc, ge::GRAPH_FAILED);
    ge::DataType indicesDtype = indicesDesc->GetDataType();
    OP_CHECK_IF(indicesDtype != ge::DT_INT32,
                OP_LOGE(nodeName_, "indices dtype must be INT32, got %d", static_cast<int>(indicesDtype)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramFetchTilingBase::CheckTensorShapeSizeAndSetTilingData()
{
    const gert::StorageShape *indicesShape = context_->GetInputShape(ENGRAM_FETCH_INPUT_INDICES_INDEX);

    uint32_t numTokens = static_cast<uint32_t>(indicesShape->GetStorageShape().GetDim(0));
    int32_t outputDtype = tilingData_->engramFetchInfo.outputDtype;
    uint32_t elemBytes = 0;
    if (outputDtype == static_cast<int32_t>(ge::DT_BF16)) {
        elemBytes = 2;
    } else if (outputDtype == static_cast<int32_t>(ge::DT_FLOAT16)) {
        elemBytes = 2;
    } else if (outputDtype == static_cast<int32_t>(ge::DT_FLOAT)) {
        elemBytes = 4;
    } else {
        OP_LOGE(nodeName_, "unsupported output_dtype %d", outputDtype);
        return ge::GRAPH_FAILED;
    }

    uint32_t hiddenDim = tilingData_->engramFetchInfo.hiddenDim;

    tilingData_->engramFetchInfo.numTokens = numTokens;
    tilingData_->engramFetchInfo.elemBytes = elemBytes;
    tilingData_->engramFetchInfo.hiddenBytes = hiddenDim * elemBytes;

    return ge::GRAPH_SUCCESS;
}

uint64_t EngramFetchTilingBase::GetTilingKey() const
{
    uint64_t key = 0;
    key += static_cast<uint64_t>(tilingData_->engramFetchInfo.outputDtype);
    key += mc2tiling::MC2_TILINGKEY_OFFSET;
    return key;
}

void EngramFetchTilingBase::SetTilingKey()
{
    context_->SetTilingKey(GetTilingKey());
}

void EngramFetchTilingBase::SetPlatformInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(1);
    tilingData_->engramFetchInfo.aivNum = aivNum;
}

ge::graphStatus EngramFetchTilingBase::SetWorkspace()
{
    size_t *workspace = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspace[0] = 16UL * 1024UL * 1024UL;
    return ge::GRAPH_SUCCESS;
}

void EngramFetchTilingBase::SetHcommCfg()
{
    uint32_t opType = 8U;
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    uint8_t aivEngineValue = mc2tiling::AIV_ENGINE;

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group_, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(aivEngineValue);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling);
}

ge::graphStatus EngramFetchTilingBase::EngramFetchTilingFuncImpl()
{
    OP_CHECK_IF(CheckRequiredAttrValue() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "CheckRequiredAttrValue failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputTensorShape() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "CheckInputTensorShape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputTensorDataType() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "CheckInputTensorDataType failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrAndSetTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName_, "GetAttrAndSetTilingData failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorShapeSizeAndSetTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(nodeName_, "CheckTensorShapeSizeAndSetTilingData failed"), return ge::GRAPH_FAILED);

    SetHcommCfg();
    SetTilingKey();
    SetPlatformInfo();

    tilingData_->engramFetchInfo.tagValue = 1;

    return SetWorkspace();
}

IMPL_OP_OPTILING(EngramFetch).Tiling([](gert::TilingContext *context) -> ge::graphStatus {
    return TilingRegistry::GetInstance().DoTilingImpl(context);
});

} // namespace optiling