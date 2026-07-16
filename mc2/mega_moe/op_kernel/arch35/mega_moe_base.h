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
#include "adv_api/hcomm/hcomm.h"

struct Mc2MoeContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0; // 通信API所需的地址
    uint64_t epHcclBuffer[HCCL_MAX_RANK_SIZE] = {};
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE] = {}; // 支持ROCE或者URMA
};

struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    GM_ADDR gmm1OutGlobal;
    GM_ADDR gmm2OutGlobal;
    __gm__ int32_t *swigluToGmm2Flag;
    __gm__ int32_t *dispatchToGmm1Flag;
};

struct PeermemInfo {
    GM_ADDR rankSyncInWorldPtr;
    GM_ADDR maskRecvPtr;        // 源卡算好的 send-mask 接收区, 布局 [localExpert][srcRank]
    GM_ADDR quantTokenScalePtr; // 量化结果，包括data+scale
    GM_ADDR dispatchRecivePtr;  // dispatch 1级通信区,
    GM_ADDR combineSendPtr;
    __aicore__ inline PeermemInfo() = default;
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData *tilingData, uint32_t elemsPerByte = 1,
                                  uint32_t serverNum = 1)
    {
        rankSyncInWorldPtr = base;
        int64_t offset = PEERMEM_DATA_OFFSET;
        maskRecvPtr = base + offset;
        // 每张卡为自己的 expertPerRank 个专家、各 worldSize 个源卡保留一份 [mask | count] 槽位。
        // 槽位 = bit-packed mask (CeilAlign(compareCount/8,32)) + 32B count(源卡 SendMaskCal 同步算好),
        // 与 mega_moe.h FirstBuffInit 的 maskSlotSize_ 一致; 接收端直接读 count, 不再 GatherMask 计数。
        int64_t sendTotalNum = static_cast<int64_t>(tilingData->bs) * tilingData->topK;
        int64_t compareCount = Ops::Base::CeilAlign(sendTotalNum * (int64_t)sizeof(int32_t), (int64_t)ALIGN_256) /
                               (int64_t)sizeof(int32_t);
        int64_t maskAlignSize = Ops::Base::CeilAlign(compareCount / 8, (int64_t)ALIGN_32);
        int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32; // mask + 32B count
        // 整个 mask 区按 512 对齐, 保证后续 quantTokenScalePtr 仍 512 对齐(CopyGMToGMPerToken 用普通 DataCopy 读)。
        offset += Ops::Base::CeilAlign((int64_t)tilingData->expertPerRank * tilingData->epWorldSize * maskSlotSize,
                                       (int64_t)ALIGN_512);

        if (tilingData->topoType == TOPO_TYPE_MTE) {
            quantTokenScalePtr = base + offset;
            uint32_t mxScaleNum = Ops::Base::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32));
            uint32_t dataBytes =
                Ops::Base::CeilAlign(tilingData->h / elemsPerByte, static_cast<uint32_t>(ALIGN_256)) * sizeof(int8_t);
            uint32_t scaleBytes = mxScaleNum * sizeof(int8_t);
            uint32_t tokenBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
            offset += Ops::Base::CeilAlign(
                (int64_t)(static_cast<int64_t>(tilingData->bs) * tokenBytes * sizeof(int8_t)), (int64_t)ALIGN_512);
        } else {
            dispatchRecivePtr = base + offset;
            uint32_t mxScaleNum = Ops::Base::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32));
            uint32_t dataBytes =
                Ops::Base::CeilAlign(tilingData->h / elemsPerByte, static_cast<uint32_t>(ALIGN_256)) * sizeof(int8_t);
            uint32_t scaleBytes = mxScaleNum * sizeof(int8_t);
            uint32_t tokenBytes = Ops::Base::CeilAlign(
                static_cast<int64_t>(Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32)) +
                                     ALIGN_32),
                static_cast<int64_t>(ALIGN_512));
            offset += Ops::Base::CeilAlign((int64_t)(static_cast<int64_t>(tilingData->bs) * tokenBytes *
                                                     sizeof(int8_t) * static_cast<int64_t>(serverNum)),
                                           (int64_t)ALIGN_512);
        }
        combineSendPtr = base + offset;
    }
};

struct CombineCommParams {
    uint32_t rankId;
    Hcomm<COMM_PROTOCOL_UBC_CTP> *hcomm;
    __gm__ Mc2MoeContext *mc2Context;
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
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;
    MegaMoeTilingData *tilingData;
    CombineCommParams combineCommParams;
};

enum class AddrUpdateMode : int32_t {
    GMM1,
    GMM2
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

__aicore__ inline void NotifyAiv1GmTileReady(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_EPILOGUE_FLAG + FLAG_ID_MAX_PER_V + value);
}

__aicore__ inline void WaitForAicGmTileReady(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_MTE2>(AIC_SYNC_AIV_EPILOGUE_FLAG + value);
}

// AIV1 acknowledges that it has accepted the AIC-to-AIV1 GM tile notification.
// On DAV_3510, an AIV1 flag ID maps to the AIC flag ID plus FLAG_ID_MAX_PER_V.
__aicore__ inline void NotifyAicGmTileAccepted()
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE2>(AIV1_SYNC_AIC_EPILOGUE_ACK_FLAG);
}

__aicore__ inline void WaitForAiv1GmTileAccepted()
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV1_SYNC_AIC_EPILOGUE_ACK_FLAG + FLAG_ID_MAX_PER_V);
}

__aicore__ inline void EndSync(int32_t &vecSetSyncCom)
{
    if (vecSetSyncCom == 0) {
        return;
    }
    if constexpr (g_coreType == AIC) {
        WaitForVector();
    }
}

__aicore__ inline void EndGMM2Sync(int32_t &vecSetSyncCom, uint16_t gmm2PingPongIdx)
{
    if constexpr (g_coreType == AIV) {
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

__aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen, int32_t &coreOffset, int32_t align = ALIGN_32)
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

__aicore__ inline uint64_t GetUrmaCommHandle(__gm__ Mc2MoeContext *mc2Context_, uint32_t rankId, uint32_t epRankId)
{
    uint32_t index = rankId > epRankId ? rankId - 1 : rankId;
    return mc2Context_->hcommHandle[index];
}


inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];
__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

template <AscendC::HardEvent event, int32_t eventId>
__aicore__ inline void SyncFuncStatic()
{
    // static_assert(eventId >= 0 && eventId <= 5, "SyncFuncStatic eventId must be [0, 5]!");
    AscendC::SetFlag<event>(static_cast<event_t>(eventId));
    AscendC::WaitFlag<event>(static_cast<event_t>(eventId));
}

// MegaMoe batch sizing constants. DAV_3510 的 AIV 可用 UB 上限按 248KB 计算。
//
// Unpermute 保留 1 个累加/输出 buffer，并支持在 EVENT_ID0~EVENT_ID5 上运行 2~6 个输入 buffer。
// 实际 UB 仅按运行时选中的输入 buffer 数量分配；该上限不表示固定预留 6 个 buffer。
//
// 最低双 buffer 是基准 batch 的成立前提。按最坏合法规格 k=8192、BF16 topK weight、combine quant 计算：
//   1 个累加槽 + 2 个输入槽 = 3 * (8192 * 2B + 8192 * 4B) = 144KB；
//   8192 个 weight item       = 8192 * (4B + 2B)              = 48KB；
//   BF16/FP32 scale scratch   = 1KB + 4KB                     = 5KB；
//   合计约 197KB，小于 248KB 可用 UB。
// 若修改 MAX_H、weight batch 基准、scale scratch 布局或最低 buffer 数，必须重新验证该前提。
constexpr int32_t MIN_UNPERMUTE_INPUT_BUFFER_COUNT = 2;
constexpr int32_t MAX_UNPERMUTE_INPUT_BUFFER_COUNT = 6;
// 延续原 Unpermute 的 weight batch 基准：tokensPerBatch * topK 不超过 1024 * 8 个 weight item。
// 先按该基准确定 batch，再用剩余 UB 反推输入 buffer 数量；上述最坏规格已保证基准 batch 至少容纳
// 2 个输入 buffer，因此运行时可以直接将反推结果钳位到 MIN_UNPERMUTE_INPUT_BUFFER_COUNT。
constexpr int32_t UNPERMUTE_BASE_TOPK = 8;
constexpr int32_t UNPERMUTE_BASE_TOKENS_PER_BATCH = 1024;
constexpr int32_t UNPERMUTE_WEIGHT_ITEMS_PER_BATCH = UNPERMUTE_BASE_TOPK * UNPERMUTE_BASE_TOKENS_PER_BATCH;
// 扩大 Unpermute weight batch 时，为 FP32 weight 和可选的 BF16 中转 buffer 各预留一个 32B 对齐块。
// 统一预留 64B 可覆盖两个独立 buffer 各最多 31B 的 padding，使两种 weight 路径共享同一套反推逻辑。
constexpr uint32_t UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES = 2 * ALIGN_32;

// DeQuantMxFp8 的 scale scratch 展开倍数。E8M0 scale 转成 BF16 时，DIST_INTLV_B16 将每个 scale
// 交织复制为 2 个 BF16；随后 DIST_INTLV_B32 再将每个 BF16 scale 交织复制为 2 个 FP32。因此，
// BF16 scratch 需要 scaleNum * 2 个元素，FP32 scratch 需要 scaleNum * 4 个元素。这两个倍数只由
// DeQuantMxFp8 的 scale 布局决定，与 Unpermute 的输入 buffer 数量无关；若修改该布局，需同步更新。
constexpr uint32_t DEQUANT_BF16_SCALE_EXPANSION = 2U;
constexpr uint32_t DEQUANT_FP32_SCALE_EXPANSION = 4U;

// Recv route batch 上限只由 DispatchBuffInit 的 UB 布局约束。Brecv 表示每个 recv batch 的 route item 数，
// W=worldSize，E=expertPerRank，D=dispatch buffer 数：
//   recvVariableBytes(Brecv) = Brecv / 8 + 2 * Brecv * sizeof(int32_t) = 65 * Brecv / 8；
//   recvFixedBytes = 32 + Align32(W * E * 4) + W * 32 + D * tokenScaleBytes + Align32(E * 4) + D * 32；
//   recvFixedBytes + recvVariableBytes(Brecv) <= 248KB。
// 12288 按 k=8192 和 recv fixedBytes 的最坏合法参数验证。Brecv 须向下对齐到 ALIGN_256；若修改
// DispatchBuffInit、dispatch buffer 数量上限或 k/worldSize/expert 上限，只需重新计算 recv 上限。
// CopyGMToGMPerToken 的 copyTmp、triple 和 event ID 三类槽位一一对应，统一使用 2~6 个 buffer。
constexpr int32_t MIN_DISPATCH_BUFFER_COUNT = 2;
constexpr int32_t MAX_DISPATCH_BUFFER_COUNT = 6;
constexpr int32_t MAX_RECV_ROUTE_ITEMS_PER_BATCH = 12288;

// Send route batch 基准用于先选择 mask buffer 数量，再固定 buffer 数反推最终 batch。Bsend 表示基准 batch
// 的 route item 数，N 表示 mask buffer 数：
//   sendVariableBytes(Bsend, N) = 2 * Bsend * sizeof(int32_t) + N * (Bsend / 8 + 32)；
//   sendFixedBytes = resetTensorBytes + 2048 + 2 * xOutTensorBytes + 2 * xInTensorBytes + sendCntAccBytes；
//   sendFixedBytes + sendVariableBytes(Bsend, N) <= 248KB。
//
// 12288 的成立前提按最坏合法规格 k=8192、moeExpertNum=2048、N=6 保守验证：
//   resetTensor                 <= 8KB；
//   mxTempTensor                = 2KB；
//   2 个 xOutTensor             = 2 * (8192B data + 256B scale) = 16.5KB；
//   2 个 xInTensor              = 2 * 8192 * sizeof(bfloat16_t) = 32KB；
//   sendCntAccTensor            <= Align32(2048 * sizeof(int32_t)) = 8KB；
//   2 个 route int32 tensor     = 2 * 12288 * sizeof(int32_t) = 96KB；
//   6 个 mask buffer            = 6 * (12288 / 8 + 32) = 9408B；
//   合计 175808B，约 172KB，小于 248KB 可用 UB。
// 因此基准 batch 在所有合法规格下至少支持双 buffer，且最坏规格也能支持当前上限 6 个 buffer。
// 若修改 MAX_H、MAX_MOE_EXPERT_NUM、DISPATCH_RESET_BATCH、Quant buffer 布局或 mask buffer 数量上限，
// 必须重新验证该前提。
// 最终 batch 由 CalcSendMaskBufferConfig 使用剩余 UB 扩大，并向下对齐到 ALIGN_256。
//
// SendMaskCal 使用 MTE3_V 的 EVENT_ID0~EVENT_ID5 管理 mask push ring。最低保留双 buffer 兼容原有
// 流水，最多 6 个 buffer；实际数量由 CalcSendMaskBufferConfig 按基准 batch 和剩余 UB 反推。
constexpr int32_t MIN_SEND_MASK_BUFFER_COUNT = 2;
constexpr int32_t MAX_SEND_MASK_BUFFER_COUNT = 6;
constexpr int32_t BASE_SEND_ROUTE_ITEMS_PER_BATCH = 12288;

// Send/Recv 均按 ALIGN_256 对齐，以满足 mask 的 32B 对齐和 DAV_3510 CompareScalar 的 256B 输入长度要求。
// recv 上限与 send 基准当前数值相同只是各自预算计算后的结果，彼此独立，不要求保持相等。

// 大 BS dispatch 分块常量
// DISPATCH_RESET_BATCH: ResetFlagList 分批清零的粒度（int32 个数）。
//   2048 * sizeof(int32_t) = 8KB；本核 flag 份额超此值时复用同一片零 UB 分批推 GM，使 reset UB 与 BS 解耦。
//   若调整 resetTensor_ 的 UB 预算或元素类型，需要按 targetResetUbBytes / sizeof(elementType) 重新计算。
constexpr int32_t DISPATCH_RESET_BATCH = 2048;

#endif
