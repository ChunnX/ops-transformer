/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_base.h
 * \brief
 */

#ifndef MEGA_MOE_BASE_H
#define MEGA_MOE_BASE_H

#include "lib/std/tuple.h"
#include "mega_moe_tiling.h"
#include "op_kernel/math_util.h"
#include "mega_moe_workspace_info.h"

struct Mc2MoeContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0; // 通信API所需的地址
    uint64_t epHcclBuffer[1024] = {};
    uint64_t hcommHandle[1024] = {};   // 支持ROCE或者URMA
    uint64_t urmacclBuffer[1024] = {};  // URMA通信buffer
};

struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    __gm__ int32_t* groupFlagList;
    __gm__ int32_t* groupFlagList2;
    GM_ADDR gmm2OutGlobal;
};

struct PeermemInfo {
    GM_ADDR rankSyncInWorldPtr;
    GM_ADDR maskRecvPtr;             // 源卡算好的 send-mask 接收区, 布局 [localExpert][srcRank]
    GM_ADDR quantTokenScalePtr;      // 量化结果，包括data+scale
    GM_ADDR combineSendPtr;
    uint64_t URMACombineWinPtr;
    uint64_t URMAmaskRecvPtr;
    uint64_t URMADispatchL1RecvPtr;   // L1 dispatch 接收区在 URMA 窗口内的偏移 (相对 rankSyncInWorldPtr)
    Hcomm<COMM_PROTOCOL_UBC_CTP> *hcomm;
    __gm__ Mc2MoeContext* mc2Context;
    __aicore__ inline PeermemInfo() = default;
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData *tilingData, uint32_t elemsPerByte = 1)
    {
        rankSyncInWorldPtr = base;
        int64_t offset = PEERMEM_DATA_OFFSET;
        maskRecvPtr = base + offset;
        int64_t sendTotalNum = static_cast<int64_t>(tilingData->bs) * tilingData->topK;
        int64_t compareCount = Ops::Base::CeilAlign(sendTotalNum * (int64_t)sizeof(int32_t), (int64_t)ALIGN_256) /
            (int64_t)sizeof(int32_t);
        int64_t maskAlignSize = Ops::Base::CeilAlign(compareCount / 8, (int64_t)ALIGN_32);
        int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32;
        offset += Ops::Base::CeilAlign(
            (int64_t)tilingData->expertPerRank * tilingData->epWorldSize * maskSlotSize, (int64_t)ALIGN_512);
        
        quantTokenScalePtr = base + offset;
        uint32_t mxScaleNum = Ops::Base::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32));
        uint32_t dataBytes = Ops::Base::CeilAlign(tilingData->h / elemsPerByte,
            static_cast<uint32_t>(ALIGN_256)) * sizeof(int8_t);
        uint32_t scaleBytes = mxScaleNum * sizeof(int8_t);
        uint32_t tokenBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
        offset += Ops::Base::CeilAlign((int64_t)(tilingData->bs * tokenBytes * sizeof(int8_t)), (int64_t)ALIGN_512);
        combineSendPtr = base + offset;

        URMAmaskRecvPtr = 0;
        URMACombineWinPtr = Ops::Base::CeilAlign(
            (int64_t)(tilingData->expertPerRank * tilingData->epWorldSize * maskSlotSize), (int64_t)ALIGN_512);

        // L1 dispatch 接收区: 紧接 combine 区之后
        // 每个 source server 一个 section, 每 section = bs * slotSize
        //   slotSize = tokendata + scaledata + 32B, 再 512B 对齐 (与 kernel 端 l1SlotSize_ 一致)
        // 共 totalServers 个 section (含本 server, 本 server section 不使用)
        int64_t slotSize = static_cast<int64_t>(Ops::Base::CeilAlign(
            static_cast<int64_t>(tokenBytes) + static_cast<int64_t>(ALIGN_32), ALIGN_512));
        int64_t l1RecvPerServer = Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData->bs) * slotSize, ALIGN_512);
        int64_t totalServers = tilingData->epWorldSize; // 最坏 case: rankSizePerServer=1
        URMADispatchL1RecvPtr = URMACombineWinPtr +
            Ops::Base::CeilAlign(
                (int64_t)(tilingData->bs) * (tokenBytes + sizeof(int32_t)) * tilingData->expertPerRank * tilingData->epWorldSize,
                (int64_t)ALIGN_512) + // combineSend 区实际大小 (用简化公式)
            Ops::Base::CeilAlign(totalServers * l1RecvPerServer, ALIGN_512);
    }
};

struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR expertIdxGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR bScaleGmAddr;
    GM_ADDR b2GmAddr;
    GM_ADDR b2ScaleGmAddr;
    GM_ADDR probsGmAddr;
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    int32_t rankId;
    GM_ADDR epGmAddr;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;
    MegaMoeTilingData *tilingData;
};

enum class AddrUpdateMode : int32_t {
    kGmm1,
    kGmm2
};

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void EndSync(int32_t& vecSetSyncCom)
{
    if (vecSetSyncCom == 0) {
        return;
    }
    if constexpr(g_coreType == AIC) {
        WaitForVector();
    }
}

__aicore__ inline void EndGMM2Sync(int32_t& vecSetSyncCom, uint16_t gmm2PingPongIdx)
{
    if constexpr(g_coreType == AIV) {
        return;
    }
    if (vecSetSyncCom <= 0) {
        return;
    } else if (vecSetSyncCom == 1) {
        WaitForVector();
    } else {
        WaitForVector(gmm2PingPongIdx);
        WaitForVector(1 - gmm2PingPongIdx);
    }
}

__aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen,
    int32_t& coreOffset, int32_t align = ALIGN_32)
{
    int32_t coreIdx = GetBlockIdx();
    int32_t coreNum = GetBlockNum() * 2; // 取到vec核总数
    int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
    int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
    coreLen = lenPerCoreAlign;
    coreOffset = coreIdx * lenPerCoreAlign;
    if (coreOffset + coreLen >= totalLen) {
        coreLen = totalLen - coreOffset;
    }
    if (coreOffset >= totalLen) {
        coreLen = 0;
    }
}

__aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        if (ReadGmByPassDCache(sig_addr) == cmp_val) {
            return;
        }
    } while (true);
}

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];
__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}


__aicore__ inline uint64_t GetUrmaCommHandle(__gm__ Mc2MoeContext* mc2Context_, uint32_t rankId, uint32_t epRankId)
{
    uint32_t index = rankId > epRankId ? rankId - 1 : rankId;
    return mc2Context_->hcommHandle[index];
}
 	 
// 获取指定 rank 的 urma 通信数据区地址（在远程 rank 通信窗口基址 + offset）。
// offset 由调用方根据具体数据布局计算（如 maskWinOffset_ / quantWinOffset_ 等）。
__aicore__ inline GM_ADDR GetUrmaAddrByRankId(__gm__ Mc2MoeContext* mc2Context_, uint32_t rankId)
{
    return (GM_ADDR)mc2Context_->urmacclBuffer[rankId];
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

template<AscendC::HardEvent event, int32_t eventId>
__aicore__ inline void SyncFuncStatic()
{
    // static_assert(eventId >= 0 && eventId <= 5, "SyncFuncStatic eventId must be [0, 5]!");
    AscendC::SetFlag<event>(static_cast<event_t>(eventId));
    AscendC::WaitFlag<event>(static_cast<event_t>(eventId));
}

#endif