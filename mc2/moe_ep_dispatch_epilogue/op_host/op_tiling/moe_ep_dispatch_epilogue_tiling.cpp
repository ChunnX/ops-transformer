#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/moe_ep_dispatch_epilogue_tiling_data.h"
#include "../../op_kernel/moe_ep_dispatch_epilogue_tiling_key.h"
#include "../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling_data.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr uint32_t DST_BUFFER_SLOT_IDX_INDEX = 0U;
constexpr uint32_t RECV_SRC_META_INTERNAL_INDEX = 1U;
constexpr uint32_t NUM_RECV_PER_RANK_HP_INDEX = 2U;
constexpr uint32_t NUM_RECV_PER_EXPERT_HP_INDEX = 3U;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 3;
constexpr uint32_t ATTR_NUM_MAX_TPR_INDEX = 4;
constexpr uint32_t ATTR_EXPERT_ALIGNMENT_INDEX = 5;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_EP_WORLD_SIZE = 288;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;

static void PrintTilingDataInfo(const char *nodeName, const MoeEpDispatchEpilogueInfo &info)
{
    OPS_LOG_D(nodeName, "epWorldSize=%u, epRankId=%u, numExperts=%u, numLocalExperts=%u",
              info.cfg.epWorldSize, info.cfg.epRankId, info.cfg.numExperts, info.cfg.numLocalExperts);
    OPS_LOG_D(nodeName, "hidden=%u, topK=%u, numMaxTokensPerRank=%u, perSlotBytes=%u",
              info.cfg.hidden, info.cfg.topK, info.cfg.numMaxTokensPerRank, info.cfg.perSlotBytes);
    OPS_LOG_D(nodeName, "aAlloc=%u, numLocalExperts=%u, aivNum=%u", info.aAlloc, info.numLocalExperts, info.aivNum);
    OPS_LOG_D(nodeName, "windowBaseAddr=%lu, numRecvPerRankAddr=%lu, numRecvPerExpertAddr=%lu",
              info.windowBaseAddr, info.numRecvPerRankAddr, info.numRecvPerExpertAddr);
}

static ge::graphStatus MoeEpDispatchEpilogueTilingFuncImpl(gert::TilingContext &context)
{
    context.SetScheduleMode(1U);
    const char *nodeName = context.GetNodeName();
    OPS_ERR_IF(nodeName == nullptr, OPS_LOG_E("unKnownNodeName", "nodeName is nullptr."),
               return ge::GRAPH_FAILED);

    MoeEpDispatchEpilogueTilingData *tilingData = context.GetTilingData<MoeEpDispatchEpilogueTilingData>();
    OPS_ERR_IF(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."),
               return ge::GRAPH_FAILED);

    auto attrs = context.GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);
    auto expertAlignmentPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_ALIGNMENT_INDEX);

    int64_t epWorldSize = *epWorldSizePtr;
    MoeEpDispatchEpilogueInfo &info = tilingData->moeEpDispatchEpilogueInfo;
    info.cfg.epWorldSize = static_cast<uint32_t>(epWorldSize);
    info.cfg.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    info.cfg.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    info.cfg.numLocalExperts = static_cast<uint32_t>(*numExpertsPtr / epWorldSize);
    info.cfg.numMaxTokensPerRank = static_cast<uint32_t>(*nmtPtr);
    info.cfg.expertAlignment = static_cast<uint32_t>(*expertAlignmentPtr);
    info.numLocalExperts = info.cfg.numLocalExperts;

    auto recvXDesc = context.GetOutputDesc(0);
    OPS_ERR_IF(recvXDesc == nullptr, OPS_LOG_E(nodeName, "recvX output desc is null."),
               return ge::GRAPH_FAILED);
    info.aAlloc = static_cast<uint32_t>(recvXDesc->GetStorageShape().GetDim(0));
    info.cfg.hidden = static_cast<uint32_t>(recvXDesc->GetStorageShape().GetDim(1));
    info.cfg.perSlotBytes = static_cast<uint32_t>(
        ((static_cast<uint64_t>(info.cfg.hidden) * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN
        + 80UL + 16UL);

    std::string groupEp(groupEpPtr);
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);
    OPS_ERR_IF(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
               OPS_LOG_E(nodeName, "GetTiling mc2InitTiling failed."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
               OPS_LOG_E(nodeName, "GetTiling mc2CcTiling failed."), return ge::GRAPH_FAILED);

    size_t *workSpaces = context.GetWorkspaceSizes(1);
    OPS_ERR_IF(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."),
               return ge::GRAPH_FAILED);
    uint64_t prefixSumBytes = static_cast<uint64_t>(epWorldSize + 1) * sizeof(int32_t);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + prefixSumBytes;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context.GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context.SetBlockDim(blockDim);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    uint64_t tilingKey = GET_TPL_TILING_KEY(TILINGKEY_TPL_A5);
    context.SetTilingKey(tilingKey);
    OPS_LOG_D(nodeName, "tilingKey=%lu, blockDim=%u, aivNum=%u", tilingKey, blockDim, aivNum);

    PrintTilingDataInfo(nodeName, info);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeEpDispatchEpilogueTilingFunc(gert::TilingContext *context)
{
    return MoeEpDispatchEpilogueTilingFuncImpl(*context);
}

struct MoeEpDispatchEpilogueCompileInfo {};
ge::graphStatus TilingParseForMoeEpDispatchEpilogue(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpDispatchEpilogue)
    .Tiling(MoeEpDispatchEpilogueTilingFunc)
    .TilingParse<MoeEpDispatchEpilogueCompileInfo>(TilingParseForMoeEpDispatchEpilogue);
}  // namespace optiling