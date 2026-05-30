#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/moe_ep_dispatch_tiling_data.h"
#include "../../op_kernel/moe_ep_dispatch_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t TOPK_IDX_INDEX = 1U;
constexpr uint32_t TOPK_WEIGHTS_INDEX = 2U;
constexpr uint32_t CACHED_NUM_PER_EXPERT_INDEX = 3U;
constexpr uint32_t CACHED_SLOT_IDX_INDEX = 4U;
constexpr uint32_t CACHED_SRC_META_INDEX = 5U;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 3;
constexpr uint32_t ATTR_NUM_MAX_TPR_INDEX = 4;
constexpr uint32_t ATTR_EXPERT_ALIGNMENT_INDEX = 5;
constexpr uint32_t ATTR_DO_CPU_SYNC_INDEX = 6;

constexpr uint32_t TWO_DIMS = 2;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_EP_WORLD_SIZE = 288;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 7168;
constexpr int64_t K_MAX = 16;
constexpr int64_t BS_UPPER_BOUND = 8192;

constexpr uint64_t INIT_TILINGKEY = 10000;

static void PrintTilingDataInfo(const char *nodeName, const MoeEpDispatchInfo &info)
{
    OPS_LOG_D(nodeName, "epWorldSize is %u.", info.cfg.epWorldSize);
    OPS_LOG_D(nodeName, "epRankId is %u.", info.cfg.epRankId);
    OPS_LOG_D(nodeName, "numExperts is %u.", info.cfg.numExperts);
    OPS_LOG_D(nodeName, "numLocalExperts is %u.", info.cfg.numLocalExperts);
    OPS_LOG_D(nodeName, "numTokens is %u.", info.cfg.numTokens);
    OPS_LOG_D(nodeName, "hidden is %u.", info.cfg.hidden);
    OPS_LOG_D(nodeName, "topK is %u.", info.cfg.topK);
    OPS_LOG_D(nodeName, "numMaxTokensPerRank is %u.", info.cfg.numMaxTokensPerRank);
    OPS_LOG_D(nodeName, "perSlotBytes is %u.", info.cfg.perSlotBytes);
    OPS_LOG_D(nodeName, "expertAlignment is %u.", info.cfg.expertAlignment);
    OPS_LOG_D(nodeName, "doCpuSync is %u.", info.doCpuSync);
    OPS_LOG_D(nodeName, "isCached is %u.", info.isCached);
    OPS_LOG_D(nodeName, "networkMode is %u.", info.networkMode);
    OPS_LOG_D(nodeName, "numScaleupRanks is %u.", info.numScaleupRanks);
    OPS_LOG_D(nodeName, "numScaleoutRanks is %u.", info.numScaleoutRanks);
    OPS_LOG_D(nodeName, "numAivStage1 is %u.", info.numAivStage1);
    OPS_LOG_D(nodeName, "numAivStage2 is %u.", info.numAivStage2);
    OPS_LOG_D(nodeName, "aivNum is %u.", info.aivNum);
    OPS_LOG_D(nodeName, "hostPinnedCounterAddr is %lu.", info.hostPinnedCounterAddr);
    OPS_LOG_D(nodeName, "totalWinSizeEp is %lu.", info.totalWinSizeEp);
    OPS_LOG_D(nodeName, "totalUbSize is %lu.", info.totalUbSize);
}

static bool CheckInputTensorShape(const gert::TilingContext &context, const char *nodeName,
                                   MoeEpDispatchInfo &info)
{
    const gert::StorageShape *xStorageShape = context.GetInputShape(X_INDEX);
    OPS_ERR_IF(xStorageShape == nullptr, OPS_LOG_E(nodeName, "xShape is null."), return false);
    OPS_ERR_IF(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
               OPS_LOG_E(nodeName, "xShape dims must be 2, but current dim num is %lu.",
                         xStorageShape->GetStorageShape().GetDimNum()),
               return false);

    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OPS_ERR_IF((xDim0 <= 0) || (xDim0 > BS_UPPER_BOUND),
               OPS_LOG_E(nodeName, "xDim0(numTokens) is invalid, should be in [1, %ld], but got %ld.",
                         BS_UPPER_BOUND, xDim0),
               return false);
    OPS_ERR_IF((xDim1 < H_MIN) || (xDim1 > H_MAX),
               OPS_LOG_E(nodeName, "xDim1(hidden) is invalid, should be in [%ld, %ld], but got %ld.",
                         H_MIN, H_MAX, xDim1),
               return false);
    info.numTokens = static_cast<uint32_t>(xDim0);
    info.hidden = static_cast<uint32_t>(xDim1);

    const gert::StorageShape *topkIdxStorageShape = context.GetInputShape(TOPK_IDX_INDEX);
    OPS_ERR_IF(topkIdxStorageShape == nullptr, OPS_LOG_E(nodeName, "topkIdxShape is null."), return false);
    OPS_ERR_IF(topkIdxStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
               OPS_LOG_E(nodeName, "topkIdxShape dims must be 2."), return false);
    const int64_t topkDim0 = topkIdxStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkDim1 = topkIdxStorageShape->GetStorageShape().GetDim(1);
    OPS_ERR_IF(xDim0 != topkDim0,
               OPS_LOG_E(nodeName, "topkIdx dim0 must equal x dim0, got %ld vs %ld.", topkDim0, xDim0),
               return false);
    OPS_ERR_IF((topkDim1 <= 0) || (topkDim1 > K_MAX),
               OPS_LOG_E(nodeName, "topkIdx dim1(topK) is invalid, should be in (0, %ld], got %ld.",
                         K_MAX, topkDim1),
               return false);
    info.topK = static_cast<uint32_t>(topkDim1);

    return true;
}

static ge::graphStatus GetAttrsAndFillCommonCfg(const gert::TilingContext &context, const char *nodeName,
                                                  MoeEpDispatchInfo &info, std::string &groupEp)
{
    auto attrs = context.GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);
    auto expertAlignmentPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_ALIGNMENT_INDEX);
    auto doCpuSyncPtr = attrs->GetAttrPointer<bool>(ATTR_DO_CPU_SYNC_INDEX);

    OPS_ERR_IF((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0),
               OPS_LOG_E(nodeName, "groupEp is invalid."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(epWorldSizePtr == nullptr, OPS_LOG_E(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(epRankIdPtr == nullptr, OPS_LOG_E(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(numExpertsPtr == nullptr, OPS_LOG_E(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(nmtPtr == nullptr, OPS_LOG_E(nodeName, "numMaxTokensPerRankPtr is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertAlignmentPtr == nullptr, OPS_LOG_E(nodeName, "expertAlignmentPtr is null."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(doCpuSyncPtr == nullptr, OPS_LOG_E(nodeName, "doCpuSyncPtr is null."), return ge::GRAPH_FAILED);

    int64_t epWorldSize = *epWorldSizePtr;
    OPS_ERR_IF((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
               OPS_LOG_E(nodeName, "epWorldSize is invalid, should be in [%ld, %ld], got %ld.",
                         MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
               OPS_LOG_E(nodeName, "epRankId is invalid, should be in [0, %ld), got %ld.",
                         epWorldSize, *epRankIdPtr),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((*numExpertsPtr <= 0) || (*numExpertsPtr % epWorldSize != 0),
               OPS_LOG_E(nodeName, "numExperts must be positive and divisible by epWorldSize, got %ld.",
                         *numExpertsPtr),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF((*nmtPtr <= 0), OPS_LOG_E(nodeName, "numMaxTokensPerRank must be positive, got %ld.", *nmtPtr),
               return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    info.cfg.epWorldSize = static_cast<uint32_t>(epWorldSize);
    info.cfg.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    info.cfg.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    info.cfg.numLocalExperts = static_cast<uint32_t>(*numExpertsPtr / epWorldSize);
    info.cfg.numMaxTokensPerRank = static_cast<uint32_t>(*nmtPtr);
    info.cfg.expertAlignment = static_cast<uint32_t>(*expertAlignmentPtr);

    bool cached = (context.GetInputDesc(CACHED_NUM_PER_EXPERT_INDEX) != nullptr);
    bool doCpuSync = *doCpuSyncPtr;
    info.doCpuSync = (doCpuSync && !cached) ? 1 : 0;
    info.isCached = cached ? 1 : 0;

    info.cfg.perSlotBytes = static_cast<uint32_t>(
        ((static_cast<uint64_t>(info.cfg.hidden) * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN
        + 80UL + 16UL);
    OPS_LOG_D(nodeName, "perSlotBytes = %u (hidden=%u)", info.cfg.perSlotBytes, info.cfg.hidden);

    return ge::GRAPH_SUCCESS;
}

static uint64_t CalcDispatchWorkspace(const MoeEpDispatchInfo &info)
{
    uint64_t counterBytes = static_cast<uint64_t>(info.cfg.epWorldSize) * sizeof(int32_t);
    uint64_t globalABytes = sizeof(int32_t);
    uint64_t stashBytes = static_cast<uint64_t>(info.cfg.numTokens) * static_cast<uint64_t>(info.cfg.topK)
                          * static_cast<uint64_t>(info.cfg.perSlotBytes);
    return SYSTEM_NEED_WORKSPACE + counterBytes + globalABytes + stashBytes;
}

static ge::graphStatus CheckWinSize(const gert::TilingContext &context, MoeEpDispatchInfo &info,
                                     const char *nodeName)
{
    uint64_t maxWindowSize = 512UL * 1024UL * 1024UL;
    auto attrs = context.GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    auto ret = mc2tiling::GetCclBufferSize(groupEpPtr, &maxWindowSize, nodeName);
    if (ret != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(nodeName, "Can't get HCCL_BUFFSIZE, use default %luMB.", maxWindowSize / MB_SIZE);
    }
    uint64_t epWorldSize = static_cast<uint64_t>(info.cfg.epWorldSize);
    uint64_t nmt = static_cast<uint64_t>(info.cfg.numMaxTokensPerRank);
    uint64_t perSlotBytes = static_cast<uint64_t>(info.cfg.perSlotBytes);

    uint64_t winNeedDirect = epWorldSize * nmt * perSlotBytes;
    uint64_t winNeed = winNeedDirect;
    if (epWorldSize > 128) {
        uint64_t numScaleup = 128UL;
        uint64_t numScaleout = (epWorldSize + 127UL) / 128UL;
        uint64_t scaleupRecv = numScaleup * (numScaleout * nmt) * perSlotBytes;
        uint64_t scaleoutSend = 1UL * nmt * perSlotBytes;
        uint64_t scaleoutRecv = numScaleout * (nmt + 32UL) * perSlotBytes;
        winNeed = scaleupRecv + scaleoutSend + scaleoutRecv;
    }
    OPS_ERR_IF(winNeed > maxWindowSize,
               OPS_LOG_E(nodeName,
                         "HCCL_BUFFSIZE is too SMALL, need %luMB, available %luMB.",
                         (winNeed / MB_SIZE) + 1UL, maxWindowSize / MB_SIZE),
               return ge::GRAPH_FAILED);
    info.totalWinSizeEp = maxWindowSize;
    OPS_LOG_D(nodeName, "windowSize = %lu", maxWindowSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext &context, MoeEpDispatchTilingData &tiling,
                                    const std::string groupEp)
{
    const char *nodeName = context.GetNodeName();
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);
    OPS_ERR_IF(mc2CcTilingConfig.GetTiling(tiling.mc2InitTiling) != 0,
               OPS_LOG_E(nodeName, "GetTiling mc2InitTiling failed."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(mc2CcTilingConfig.GetTiling(tiling.mc2CcTiling) != 0,
               OPS_LOG_E(nodeName, "GetTiling mc2CcTiling failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext &context, const MoeEpDispatchInfo &info,
                                     const char *nodeName)
{
    size_t *workSpaces = context.GetWorkspaceSizes(1);
    OPS_ERR_IF(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = CalcDispatchWorkspace(info);
    return ge::GRAPH_SUCCESS;
}

static uint64_t CalTilingKey(const uint32_t doCpuSync, const uint32_t isCached,
                              const uint32_t networkMode)
{
    uint32_t tplDoCpuSync = doCpuSync ? 1 : 0;
    uint32_t tplIsCached = isCached ? 1 : 0;
    uint32_t tplNetworkMode = networkMode ? TILINGKEY_HYBRID : TILINGKEY_DIRECT;
    return GET_TPL_TILING_KEY(tplDoCpuSync, tplIsCached, tplNetworkMode, TILINGKEY_TPL_A5);
}

static ge::graphStatus MoeEpDispatchTilingFuncImpl(gert::TilingContext &context)
{
    context.SetScheduleMode(1U);

    const char *nodeName = context.GetNodeName();
    OPS_ERR_IF(nodeName == nullptr, OPS_LOG_E("unKnownNodeName", "nodeName is nullptr."),
               return ge::GRAPH_FAILED);

    MoeEpDispatchTilingData *tilingData = context.GetTilingData<MoeEpDispatchTilingData>();
    OPS_ERR_IF(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_LOG_I(nodeName, "Enter MoeEpDispatch tiling func.");

    std::string groupEp = "";
    MoeEpDispatchInfo &info = tilingData->moeEpDispatchInfo;

    OPS_ERR_IF(GetAttrsAndFillCommonCfg(context, nodeName, info, groupEp) != ge::GRAPH_SUCCESS,
               OPS_LOG_E(nodeName, "Get attrs and fill common cfg failed."), return ge::GRAPH_FAILED);

    OPS_ERR_IF(!CheckInputTensorShape(context, nodeName, info),
               OPS_LOG_E(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);

    uint32_t epWorldSize = info.cfg.epWorldSize;
    info.networkMode = (epWorldSize > 128) ? 1 : 0;
    if (epWorldSize > 128) {
        info.numScaleupRanks = 128;
        info.numScaleoutRanks = (epWorldSize + 127) / 128;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context.GetPlatformInfo());
        uint32_t totalAiv = ascendcPlatform.GetCoreNumAiv();
        info.numAivStage1 = totalAiv * 5 / 8;
        info.numAivStage2 = totalAiv * 2 / 8;
    } else {
        info.numScaleupRanks = epWorldSize;
        info.numScaleoutRanks = 1;
        info.numAivStage1 = 0;
        info.numAivStage2 = 0;
    }

    OPS_ERR_IF(CheckWinSize(context, info, nodeName) != ge::GRAPH_SUCCESS,
               OPS_LOG_E(nodeName, "Check window size failed."), return ge::GRAPH_FAILED);

    OPS_ERR_IF(SetWorkSpace(context, info, nodeName) != ge::GRAPH_SUCCESS,
               OPS_LOG_E(nodeName, "Set workspace failed."), return ge::GRAPH_FAILED);

    OPS_ERR_IF(SetHcommCfg(context, *tilingData, groupEp) != ge::GRAPH_SUCCESS,
               OPS_LOG_E(nodeName, "SetHcommCfg failed."), return ge::GRAPH_FAILED);

    uint64_t tilingKey = CalTilingKey(info.doCpuSync, info.isCached, info.networkMode);
    OPS_LOG_D(nodeName, "tilingKey is %lu", tilingKey);
    context.SetTilingKey(tilingKey);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context.GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context.SetBlockDim(blockDim);

    info.aivNum = aivNum;
    info.totalUbSize = ubSize;
    OPS_LOG_D(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, info);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeEpDispatchTilingFunc(gert::TilingContext *context)
{
    return MoeEpDispatchTilingFuncImpl(*context);
}

struct MoeEpDispatchCompileInfo {};
ge::graphStatus TilingParseForMoeEpDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpDispatch)
    .Tiling(MoeEpDispatchTilingFunc)
    .TilingParse<MoeEpDispatchCompileInfo>(TilingParseForMoeEpDispatch);
}  // namespace optiling