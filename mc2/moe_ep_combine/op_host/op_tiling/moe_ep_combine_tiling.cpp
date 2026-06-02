#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/moe_ep_combine_tiling.h"
#include "../../op_kernel/moe_ep_combine_tiling_key.h"
#include "../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t DST_BUFFER_SLOT_IDX_INDEX = 1U;
constexpr uint32_t RECV_SRC_METADATA_INDEX = 2U;
constexpr uint32_t NUM_RECV_PER_RANK_INDEX = 3U;
constexpr uint32_t TOPK_WEIGHTS_INDEX = 4U;
constexpr uint32_t BIAS_0_INDEX = 5U;
constexpr uint32_t BIAS_1_INDEX = 6U;

constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_MAX_TPR_INDEX = 3;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 4;

constexpr uint32_t TWO_DIMS = 2;
constexpr int64_t MAX_EP_WORLD_SIZE = 288;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 7168;

static void PrintTilingDataInfo(const char *nodeName, const MoeEpCombineInfo &info)
{
    OPS_LOG_D(nodeName, "epWorldSize=%u, epRankId=%u, numExperts=%u, numLocalExperts=%u",
              info.cfg.epWorldSize, info.cfg.epRankId, info.cfg.numExperts, info.cfg.numLocalExperts);
    OPS_LOG_D(nodeName, "numTokens=%u, hidden=%u, topK=%u, numMaxTokensPerRank=%u",
              info.cfg.numTokens, info.cfg.hidden, info.cfg.topK, info.cfg.numMaxTokensPerRank);
    OPS_LOG_D(nodeName, "perSlotBytes=%u, hasTopkWeights=%u, aivNum=%u",
              info.cfg.perSlotBytes, info.hasTopkWeights, info.aivNum);
    OPS_LOG_D(nodeName, "totalWinSizeEp=%lu, totalUbSize=%lu", info.totalWinSizeEp, info.totalUbSize);
}

static ge::graphStatus MoeEpCombineTilingFunc(gert::TilingContext *context)
{
    context->SetScheduleMode(1U);
    const char *nodeName = context->GetNodeName();
    OPS_ERR_IF(nodeName == nullptr, OPS_LOG_E("unKnownNodeName", "nodeName is nullptr."),
               return ge::GRAPH_FAILED);

    MoeEpCombineTilingData *tilingData = context->GetTilingData<MoeEpCombineTilingData>();
    OPS_ERR_IF(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_LOG_I(nodeName, "Enter MoeEpCombine tiling func.");

    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);

    int64_t epWorldSize = *epWorldSizePtr;
    MoeEpCombineInfo &info = tilingData->moeEpCombineInfo;
    info.cfg.epWorldSize = static_cast<uint32_t>(epWorldSize);
    info.cfg.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    info.cfg.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    info.cfg.numLocalExperts = static_cast<uint32_t>(*numExpertsPtr / epWorldSize);
    info.cfg.numMaxTokensPerRank = static_cast<uint32_t>(*nmtPtr);

    bool hasTopkWeights = (context->GetInputDesc(TOPK_WEIGHTS_INDEX) != nullptr);
    info.hasTopkWeights = hasTopkWeights ? 1 : 0;

    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX);
    OPS_ERR_IF(xShape == nullptr, OPS_LOG_E(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xShape->GetStorageShape().GetDimNum() != TWO_DIMS,
               OPS_LOG_E(nodeName, "xShape dims must be 2."), return ge::GRAPH_FAILED);
    info.cfg.numTokens = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(0));
    info.cfg.hidden = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(1));
    info.cfg.topK = static_cast<uint32_t>(context->GetInputShape(DST_BUFFER_SLOT_IDX_INDEX)
                                          ->GetStorageShape().GetDim(1));
    info.cfg.perSlotBytes = static_cast<uint32_t>(
        ((static_cast<uint64_t>(info.cfg.hidden) * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN
        + 80UL + 16UL);

    uint64_t maxWindowSize = static_cast<uint64_t>(*cclBufferSizePtr);
    uint32_t combineTopk = std::min(static_cast<uint32_t>(epWorldSize), info.cfg.topK);
    uint64_t winNeedCombine = static_cast<uint64_t>(combineTopk) * static_cast<uint64_t>(*nmtPtr)
                              * static_cast<uint64_t>(info.cfg.perSlotBytes);
    uint64_t winNeed = static_cast<uint64_t>(epWorldSize) * static_cast<uint64_t>(*nmtPtr)
                       * static_cast<uint64_t>(info.cfg.perSlotBytes);
    uint64_t winNeedTotal = (winNeed > winNeedCombine) ? winNeed : winNeedCombine;
    OPS_ERR_IF(winNeedTotal > maxWindowSize,
               OPS_LOG_E(nodeName, "HCCL_BUFFSIZE too small, need %luMB.", (winNeedTotal / MB_SIZE) + 1UL),
               return ge::GRAPH_FAILED);
    info.totalWinSizeEp = maxWindowSize;

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."),
               return ge::GRAPH_FAILED);
    uint64_t stashBytes = static_cast<uint64_t>(info.cfg.numTokens) * static_cast<uint64_t>(info.cfg.topK)
                          * static_cast<uint64_t>(info.cfg.perSlotBytes);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + stashBytes;

    uint32_t tplHasTopkWeights = hasTopkWeights ? 1 : 0;
    uint64_t tilingKey = GET_TPL_TILING_KEY(tplHasTopkWeights, TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OPS_LOG_D(nodeName, "tilingKey=%lu, blockDim=%u, aivNum=%u", tilingKey, blockDim, aivNum);
    PrintTilingDataInfo(nodeName, info);
    return ge::GRAPH_SUCCESS;
}

struct MoeEpCombineCompileInfo {};
ge::graphStatus TilingParseForMoeEpCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpCombine)
    .Tiling(MoeEpCombineTilingFunc)
    .TilingParse<MoeEpCombineCompileInfo>(TilingParseForMoeEpCombine);
}  // namespace optiling