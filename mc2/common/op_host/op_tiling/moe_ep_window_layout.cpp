#include "moe_ep_window_layout.h"
#include "mc2_log.h"

uint64_t AlignMoeEpUb(uint64_t value)
{
    return (value + MOE_EP_UB_ALIGN - 1UL) / MOE_EP_UB_ALIGN * MOE_EP_UB_ALIGN;
}

uint64_t AlignMoeEpWin(uint64_t value)
{
    return (value + MOE_EP_WIN_ALIGN - 1UL) / MOE_EP_WIN_ALIGN * MOE_EP_WIN_ALIGN;
}

ge::graphStatus ResolveMoeEpTopology(uint32_t epWorldSize, int64_t requestedNetworkMode,
                                     int64_t rankNumPerServer, MoeEpTopology &topology)
{
    if (rankNumPerServer <= 0 || epWorldSize == 0 || epWorldSize % static_cast<uint32_t>(rankNumPerServer) != 0) {
        return ge::GRAPH_FAILED;
    }
    topology.rankNumPerServer = static_cast<uint32_t>(rankNumPerServer);
    topology.serverNum = epWorldSize / topology.rankNumPerServer;
    topology.networkMode = (requestedNetworkMode == static_cast<int64_t>(MOE_EP_NETWORK_HYBRID) &&
                            topology.serverNum > 1U)
                               ? MOE_EP_NETWORK_HYBRID
                               : MOE_EP_NETWORK_DIRECT;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcMoeEpWindowLayout(const MoeEpWindowLayoutParams &params, MoeEpWindowLayout &layout)
{
    uint64_t E = params.epWorldSize;
    uint64_t L = params.localExpertNum;
    uint64_t N = params.nmt;
    uint64_t K = params.topK;
    uint64_t H = params.hidden;
    uint64_t A = MOE_EP_WIN_ALIGN;
    uint64_t S = params.serverNum;

    layout.metadataOffset = 0UL;
    layout.metadataSize = MOE_EP_METADATA_SIZE;
    layout.perCoreDiagOffset = MOE_EP_METADATA_SIZE;
    layout.perCoreDiagSize = MOE_EP_PER_CORE_DIAG_SIZE;

    uint64_t P = MOE_EP_FIXED_PREFIX;

    uint64_t countNotifyOffset = P;
    uint64_t countNotifySize = E * A;

    uint64_t expertCountOffset = countNotifyOffset + countNotifySize;
    uint64_t expertCountSize = E * AlignMoeEpWin(L * MOE_EP_METADATA_DTYPE_SIZE);

    layout.cntWinStateOffset = countNotifyOffset;
    layout.cntWinStateSize = countNotifySize + expertCountSize;

    layout.slotWinStateOffset = expertCountOffset + expertCountSize;
    layout.dispatchSlotStateSize = ((N + MOE_EP_NOTIFY_TOKEN_LIMIT - 1UL) / MOE_EP_NOTIFY_TOKEN_LIMIT) * E * A;
    layout.dispatchNotifyCount = static_cast<uint32_t>((N + MOE_EP_NOTIFY_TOKEN_LIMIT - 1UL) / MOE_EP_NOTIFY_TOKEN_LIMIT);

    uint64_t combineTokenStateOffset = layout.slotWinStateOffset + layout.dispatchSlotStateSize;
    uint64_t combineTokenStateSize = N * K * A;
    uint64_t combineStateTailOffset = combineTokenStateOffset + combineTokenStateSize;
    uint64_t combineStateTailSize = E * A;

    layout.combineStateWinOffset = combineTokenStateOffset;
    layout.combineStateWinSize = combineTokenStateSize + combineStateTailSize;

    uint64_t dataBase = combineStateTailOffset + combineStateTailSize;

    uint64_t hiddenAlign = AlignMoeEpUb(H * MOE_EP_MAX_OUT_DTYPE_SIZE);
    uint64_t topKAlign = AlignMoeEpUb(K * MOE_EP_METADATA_DTYPE_SIZE);
    layout.dispatchReservedPerSlotBytes = AlignMoeEpWin(hiddenAlign + topKAlign * 2UL + MOE_EP_UB_ALIGN);
    layout.scaleoutReservedPerSlotBytes = AlignMoeEpWin(layout.dispatchReservedPerSlotBytes + K * sizeof(int32_t));
    layout.combineReservedPerSlotBytes = AlignMoeEpWin(hiddenAlign + MOE_EP_UB_ALIGN);

    uint64_t combineDataSize = N * K * layout.combineReservedPerSlotBytes;

    if (params.networkMode == MOE_EP_NETWORK_HYBRID) {
        layout.scaleoutRecvDataSize = S * N * layout.scaleoutReservedPerSlotBytes;
        layout.dispatchRecvDataSize = E * N * layout.dispatchReservedPerSlotBytes;
        layout.scaleoutRecvStatusSize = AlignMoeEpWin(S * N * A);

        layout.scaleoutRecvDataOffset = dataBase;
        uint64_t scaleupFinalRecvDataOffset = layout.scaleoutRecvDataOffset + layout.scaleoutRecvDataSize;
        layout.scaleoutRecvStatusOffset = scaleupFinalRecvDataOffset + layout.dispatchRecvDataSize;
        layout.combineDataWinOffset = layout.scaleoutRecvStatusOffset + layout.scaleoutRecvStatusSize;
        layout.payloadStashWinOffset = layout.combineDataWinOffset + combineDataSize;
        layout.payloadStashWinSize = N * layout.scaleoutReservedPerSlotBytes;
        layout.winDataOffset = scaleupFinalRecvDataOffset;
        layout.combineDataSize = combineDataSize;
        layout.requiredBytes = layout.payloadStashWinOffset + layout.payloadStashWinSize;
    } else {
        layout.dispatchRecvDataSize = E * N * layout.dispatchReservedPerSlotBytes;
        layout.scaleoutRecvDataOffset = 0UL;
        layout.scaleoutRecvDataSize = 0UL;
        layout.scaleoutRecvStatusOffset = 0UL;
        layout.scaleoutRecvStatusSize = 0UL;
        layout.winDataOffset = dataBase;
        layout.combineDataWinOffset = dataBase + layout.dispatchRecvDataSize;
        layout.combineDataSize = combineDataSize;
        layout.payloadStashWinOffset = layout.combineDataWinOffset + combineDataSize;
        layout.payloadStashWinSize = layout.dispatchRecvDataSize;
        layout.requiredBytes = layout.payloadStashWinOffset + layout.payloadStashWinSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMoeEpWindowCapacity(uint64_t requiredBytes, uint64_t cclBufferSize, const char *nodeName)
{
    constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
    if (requiredBytes > cclBufferSize) {
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, need %luMB, available %luMB.",
                (requiredBytes / MB_SIZE) + 1UL, cclBufferSize / MB_SIZE);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FillMoeEpDumpMetadata(const MoeEpWindowLayoutParams &params, const MoeEpWindowLayout &layout,
                           uint32_t aivNum, MoeEpDumpMetadata &metadata)
{
    metadata.layoutVersion = MOE_EP_DUMP_LAYOUT_VERSION;
    metadata.nmt = params.nmt;
    metadata.topK = params.topK;
    metadata.hidden = params.hidden;
    metadata.epWorldSize = params.epWorldSize;
    metadata.localExpertNum = params.localExpertNum;
    metadata.aivNum = aivNum;
    metadata.networkMode = params.networkMode;
    metadata.serverNum = params.serverNum;
    metadata.rankNumPerServer = params.rankNumPerServer;

    uint64_t E = params.epWorldSize;
    uint64_t L = params.localExpertNum;
    uint64_t N = params.nmt;
    uint64_t K = params.topK;
    uint64_t A = MOE_EP_WIN_ALIGN;
    uint64_t S = params.serverNum;

    metadata.regions[MOE_EP_DUMP_REGION_PER_CORE_DIAG].offset = layout.perCoreDiagOffset;
    metadata.regions[MOE_EP_DUMP_REGION_PER_CORE_DIAG].size = layout.perCoreDiagSize;

    metadata.regions[MOE_EP_DUMP_REGION_COUNT_NOTIFY].offset = layout.cntWinStateOffset;
    metadata.regions[MOE_EP_DUMP_REGION_COUNT_NOTIFY].size = E * A;

    metadata.regions[MOE_EP_DUMP_REGION_EXPERT_COUNT].offset = layout.cntWinStateOffset + E * A;
    metadata.regions[MOE_EP_DUMP_REGION_EXPERT_COUNT].size = E * AlignMoeEpWin(L * MOE_EP_METADATA_DTYPE_SIZE);

    metadata.regions[MOE_EP_DUMP_REGION_DISPATCH_SLOT_STATE].offset = layout.slotWinStateOffset;
    metadata.regions[MOE_EP_DUMP_REGION_DISPATCH_SLOT_STATE].size = layout.dispatchSlotStateSize;

    metadata.regions[MOE_EP_DUMP_REGION_COMBINE_TOKEN_STATE].offset = layout.combineStateWinOffset;
    metadata.regions[MOE_EP_DUMP_REGION_COMBINE_TOKEN_STATE].size = N * K * A;

    metadata.regions[MOE_EP_DUMP_REGION_COMBINE_STATE_TAIL].offset =
        layout.combineStateWinOffset + N * K * A;
    metadata.regions[MOE_EP_DUMP_REGION_COMBINE_STATE_TAIL].size = E * A;

    if (params.networkMode == MOE_EP_NETWORK_HYBRID) {
        metadata.regions[MOE_EP_DUMP_REGION_HYBRID_SCALEOUT_STATUS].offset = layout.scaleoutRecvStatusOffset;
        metadata.regions[MOE_EP_DUMP_REGION_HYBRID_SCALEOUT_STATUS].size = AlignMoeEpWin(S * N * A);
    } else {
        metadata.regions[MOE_EP_DUMP_REGION_HYBRID_SCALEOUT_STATUS].offset = 0UL;
        metadata.regions[MOE_EP_DUMP_REGION_HYBRID_SCALEOUT_STATUS].size = 0UL;
    }
}
