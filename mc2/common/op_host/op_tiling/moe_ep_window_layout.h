#ifndef MOE_EP_WINDOW_LAYOUT_H
#define MOE_EP_WINDOW_LAYOUT_H

#include <cstdint>
#include "graph/ge_error_codes.h"
#include "../../op_kernel/moe_ep_exception_dump_defs.h"

struct MoeEpWindowLayoutParams {
    uint32_t epWorldSize;
    uint32_t localExpertNum;
    uint32_t nmt;
    uint32_t topK;
    uint32_t hidden;
    uint32_t networkMode;
    uint32_t rankNumPerServer;
    uint32_t serverNum;
};

struct MoeEpWindowLayout {
    uint32_t dispatchNotifyCount;

    uint64_t metadataOffset;
    uint64_t metadataSize;
    uint64_t perCoreDiagOffset;
    uint64_t perCoreDiagSize;

    uint64_t cntWinStateOffset;
    uint64_t cntWinStateSize;
    uint64_t slotWinStateOffset;
    uint64_t dispatchSlotStateSize;
    uint64_t combineStateWinOffset;
    uint64_t combineStateWinSize;

    uint64_t scaleoutRecvDataOffset;
    uint64_t scaleoutRecvDataSize;
    uint64_t winDataOffset;
    uint64_t dispatchRecvDataSize;
    uint64_t scaleoutRecvStatusOffset;
    uint64_t scaleoutRecvStatusSize;
    uint64_t combineDataWinOffset;
    uint64_t combineDataSize;
    uint64_t payloadStashWinOffset;
    uint64_t payloadStashWinSize;

    uint64_t dispatchReservedPerSlotBytes;
    uint64_t scaleoutReservedPerSlotBytes;
    uint64_t combineReservedPerSlotBytes;
    uint64_t requiredBytes;
};

struct MoeEpTopology {
    uint32_t networkMode;
    uint32_t rankNumPerServer;
    uint32_t serverNum;
};

uint64_t AlignMoeEpUb(uint64_t value);
uint64_t AlignMoeEpWin(uint64_t value);

ge::graphStatus ResolveMoeEpTopology(uint32_t epWorldSize, int64_t requestedNetworkMode,
                                     int64_t rankNumPerServer, MoeEpTopology &topology);

ge::graphStatus CalcMoeEpWindowLayout(const MoeEpWindowLayoutParams &params, MoeEpWindowLayout &layout);

ge::graphStatus CheckMoeEpWindowCapacity(uint64_t requiredBytes, uint64_t cclBufferSize, const char *nodeName);

void FillMoeEpDumpMetadata(const MoeEpWindowLayoutParams &params, const MoeEpWindowLayout &layout,
                           uint32_t aivNum, MoeEpDumpMetadata &metadata);

#endif
