/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TOKEN_COMBINE_H
#define MEGA_MOE_TOKEN_COMBINE_H

#include "kernel_operator.h"
#include "../mega_moe_base.h"
#include "../mega_moe_impl_base.h"
#include "../mega_moe_job_context.h"
#include "../mega_moe_utils.h"
#include "../aiv_compute/mega_moe_mxfp8_quantization.h"
#if __has_include("../../../common/mc2_kernel_utils.h")
#include "../../../common/mc2_kernel_utils.h"
#include "../../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

using namespace AscendC;

namespace MegaMoeCombineImpl {

// 为 tile 内每个 token 行发送一段有效的 GMM2 tile 数据。
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(uint32_t nLoc, uint32_t n, LocalTensor<int32_t> &metaInfoTensor,
                                     LocalTensor<ElementMMadOut2> &l0cOutUbGMM2, BlockShape &actualBlockShape,
                                     uint32_t ubTileN, const Params &params)
{
    // 调用方在进入该数据操作前，保证批量加载的 metadata 对 Scalar 可见。
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        uint32_t toRankId = metaInfoTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = metaInfoTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = metaInfoTensor.GetValue(tileIdx * 8 + 2);
        gmRemoteD.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementMMadOut2 *>(GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset], l0cOutUbGMM2[tileIdx * ubTileN], ub2GmParams);
    }
}

// 发送一行完整的 Combine 数据，BF16 与 FP8 记录复用相同的定长搬运流程。
template <typename Element>
__aicore__ inline void SendCombineTokenRow(uint32_t rowElements, uint64_t gmRemoteBaseOffset,
                                           LocalTensor<int32_t> &metaInfoTensor, LocalTensor<Element> &rowTensor,
                                           const Params &params)
{
    uint32_t toRankId = metaInfoTensor.GetValue(RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(TOPK_INDEX);

    AscendC::GlobalTensor<Element> gmRemoteD;
    gmRemoteD.SetGlobalBuffer(
        reinterpret_cast<__gm__ Element *>(GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));
    uint64_t gmDstRowOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * rowElements;

    constexpr uint32_t TRANSFER_BYTES = 512U;
    constexpr uint32_t TILE_ELEMENTS = TRANSFER_BYTES / sizeof(Element);
    AscendC::DataCopyExtParams ub2GmParams{1U, 0U, 0U, 0U, 0U};
    for (uint32_t elementOffset = 0U; elementOffset < rowElements; elementOffset += TILE_ELEMENTS) {
        uint32_t remainingElements = rowElements - elementOffset;
        uint32_t currentElements = remainingElements < TILE_ELEMENTS ? remainingElements : TILE_ELEMENTS;
        ub2GmParams.blockLen = currentElements * sizeof(Element);
        AscendC::DataCopyPad(gmRemoteD[gmDstRowOffset + elementOffset], rowTensor[elementOffset], ub2GmParams);
    }
}

// 通过本地 MTE 路径或远端 URMA 路径发送一个 layered 模板 token。
template <typename DataType, bool IsQuantized = true>
__aicore__ inline void CombineSendTokenToRemote(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                                uint32_t groupIdx, uint32_t rankId,
                                                LocalTensor<int32_t> &metaInfoTensor, LocalTensor<DataType> &ubQuant,
                                                const Params &params, GM_ADDR localSrcPtr)
{
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID3>();
    int64_t quantTokenSize = IsQuantized ? (n + nScale) : n;
    uint32_t toRankId = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOPK_INDEX);

    AscendC::GlobalTensor<DataType> gmLocalD;
    uint64_t gmRemoteOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    GM_ADDR srcAddr = localSrcPtr;

    if (toRankId == rankId) {
        srcAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
    }

    gmLocalD.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(srcAddr));
    uint64_t dstBaseOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * quantTokenSize;
    AscendC::DataCopyExtParams singleCopyParams{1, static_cast<uint32_t>(quantTokenSize * sizeof(DataType)), 0, 0, 0};

    if constexpr (!IsQuantized) {
        DataCopyPadExtParams<DataType> copyPadParams{false, 0U, 0U, 0U};
        if (toRankId == rankId) {
            AscendC::GlobalTensor<DataType> gmm2OutGm;
            gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(localSrcPtr));
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
            AscendC::DataCopyPad(ubQuant, gmm2OutGm, singleCopyParams, copyPadParams);
            SyncFuncStatic<AscendC::HardEvent::MTE2_MTE3, SYNC_EVENT_ID4>();
        }
    }

    if (IsQuantized || toRankId == rankId) {
        uint64_t dstOffset = toRankId == rankId ? dstBaseOffset : 0;
        AscendC::DataCopyPad(gmLocalD[dstOffset], ubQuant, singleCopyParams);
        SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID4>();
    }

    if (toRankId != rankId) {
        uint64_t channelHandle = GetUrmaCommHandle(params.combineCommParams.mc2Context, toRankId, rankId);
        GM_ADDR remoteAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset) + dstBaseOffset * sizeof(DataType);
        params.combineCommParams.hcomm->WriteNbi(channelHandle, remoteAddr, srcAddr,
                                                 quantTokenSize * sizeof(DataType));
    }
}

// 将一条量化 token 记录发送到 metadata 指定的 rank 和目标行。
template <typename QuantOutType>
__aicore__ inline void CombineQuantizedTokens(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                              uint32_t groupIdx, uint32_t rankId,
                                              LocalTensor<int32_t> &metaInfoTensor,
                                              LocalTensor<QuantOutType> &ubQuant, const Params &params,
                                              uint32_t quantTokenSizeBytes)
{
    uint32_t toRankId = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + RANK_ID);
    uint32_t tokenIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOKEN_ID);
    uint32_t topkIdx = metaInfoTensor.GetValue(batchStart * META_INFO_SIZE + TOPK_INDEX);

    AscendC::GlobalTensor<QuantOutType> gmRemoteD;
    uint64_t gmRemoteOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    __gm__ void *dstPeermemPtr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset);
    gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(dstPeermemPtr));

    uint64_t dstBaseOffset =
        (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * quantTokenSizeBytes;
    AscendC::DataCopyExtParams singleCopyParams{1, quantTokenSizeBytes, 0, 0, 0};
    AscendC::DataCopyPad(gmRemoteD[dstBaseOffset], ubQuant, singleCopyParams);
}

// 读取并按需量化一组普通/layered Combine token，然后发送到目标 rank。
template <uint8_t QuantMode, typename T, bool IsLayered = false, bool IsQuantized = true>
__aicore__ inline void CombineTokenGroup(uint32_t tokenStart, uint32_t tokenCount, uint32_t n, uint32_t groupIdx,
                                         uint32_t rankId, GM_ADDR gmm2OutAddr, const Params &params,
                                         LocalTensor<int32_t> &metaInfoTensor, int64_t ubTensorSize, int64_t offset,
                                         uint32_t quantTokenSizeBytes)
{
    LocalTensor<T> combineUbTensor(TPosition::VECIN, offset, ubTensorSize);
    offset += ubTensorSize * sizeof(T);

    uint32_t nScale = Ops::Base::CeilDiv(n, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t mxScaleNum = Align2(nScale);
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, static_cast<uint32_t>(ALIGN_32));
    uint32_t floatTempSize = Align32(mxScaleNum) + mxScaleNum / 2;
    LocalTensor<float> floatTemp = LocalTensor<float>(TPosition::VECIN, offset, floatTempSize);

    GlobalTensor<T> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gmm2OutAddr));
    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;

    uint32_t singleTokenElems = (nAlign32 * sizeof(T) + quantTokenSizeBytes) / sizeof(T);
    DataCopyPadExtParams<T> copyPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyExtParams gm2UbParams{static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};

    for (uint32_t i = 0; i < tokenCount; i++) {
        uint32_t pingPong = i % 2;
        LocalTensor<T> ubBf16 = combineUbTensor[pingPong * singleTokenElems];
        LocalTensor<T> ubQuantData = ubBf16[nAlign32];

        if constexpr (IsQuantized) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
            AscendC::DataCopyPad(ubBf16, gmm2OutGm[(tokenStart + i) * n], gm2UbParams, copyPadParams);
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID4>();
            QuantMxFp8<QuantMode, T>(ubQuantData, ubBf16, floatTemp, n);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID5>();
        }

        using SendType = typename std::conditional<IsQuantized, Fp8Type, T>::type;
        LocalTensor<SendType> ubQuantSend = ubQuantData.template ReinterpretCast<SendType>();
        if constexpr (IsLayered) {
            if constexpr (!IsQuantized) {
                ubQuantSend = ubBf16;
            }
            GM_ADDR localSrcPtr = gmm2OutAddr + (tokenStart + i) * n * sizeof(T);
            CombineSendTokenToRemote<SendType, IsQuantized>(i, 1, n, nScale, groupIdx, rankId,
                                                            metaInfoTensor, ubQuantSend, params, localSrcPtr);
        } else {
            CombineQuantizedTokens<SendType>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor, ubQuantSend,
                                             params, quantTokenSizeBytes);
        }
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
}

} // namespace MegaMoeCombineImpl

namespace MegaMoeImpl {

struct QuantCombineBufferConfig {
    int64_t combineUbElementCount;
    uint32_t quantTokenSizeBytes;
};

struct QuantCombineContext {
    AivJobContext job;
    uint32_t tokenHiddenDim;
    uint32_t rankId;
    uint32_t gmm2NTilesPerGroup;
    bool participates;
};

struct QuantCombineTokenRange {
    uint32_t tokenStart;
    uint32_t tokenCount;
    uint64_t metaInfoTokenOffset;
    uint32_t expertIdx;
};

// 加载当前任务负责的 metadata，并完成一段连续 token 的量化 Combine 发送。
template <uint8_t CombineMode>
__aicore__ inline void CombineQuantizedTokenRange(
    const QuantCombineContext &context, const QuantCombineBufferConfig &bufferConfig,
    const Params &params, const GMMAddrInfo &gmmAddrInfo, const QuantCombineTokenRange &tokenRange)
{
    AscendC::SetCtrlSpr<60, 60>(0);
    int64_t ubOffset = 0;
    LocalTensor<int32_t> metaInfoTensor(TPosition::VECIN, ubOffset,
                                        tokenRange.tokenCount * META_INFO_SIZE);
    ubOffset += tokenRange.tokenCount * META_INFO_SIZE * sizeof(int32_t);
    GlobalTensor<int32_t> metaInfoGm;
    metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.metaInfoPtr + tokenRange.metaInfoTokenOffset * META_INFO_SIZE * sizeof(int32_t)));
    DataCopy(metaInfoTensor, metaInfoGm, tokenRange.tokenCount * META_INFO_SIZE);
    PipeBarrier<PIPE_MTE2>();
    MegaMoeCombineImpl::CombineTokenGroup<CombineMode, bfloat16_t>(
        tokenRange.tokenStart, tokenRange.tokenCount, context.tokenHiddenDim, tokenRange.expertIdx,
        context.rankId, gmmAddrInfo.gmm2OutGlobal, params, metaInfoTensor,
        bufferConfig.combineUbElementCount, ubOffset, bufferConfig.quantTokenSizeBytes);
}

struct WaveCombineContext {
    AivJobContext job;
    uint32_t tokenHiddenDim;
    bool participates;
};

struct WaveCombineArgs {
    GM_ADDR metaInfoPtr;
};

struct WaveCombineBufferConfig {
    uint32_t rowBytes = 0;
    uint32_t rowStrideBytes = 0;
    uint32_t quantRowElements = 0;
    uint32_t quantRowStorageBytes = 0;
    uint32_t slotStrideBytes = 0;
    uint32_t quantTempElements = 0;
};

struct WaveCombineScratch {
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<bfloat16_t> rowBufferTensor;
    LocalTensor<float> quantTempTensor;
};

constexpr uint32_t WAVE_COMBINE_NO_QUANT_ROW_BUFFER_NUM = 6U;
constexpr uint32_t WAVE_COMBINE_QUANT_ROW_BUFFER_NUM = 2U;
constexpr uint32_t WAVE_COMBINE_UB_BASE = 64U * 1024U;
constexpr uint32_t WAVE_COMBINE_META_INFO_TOKEN_CAPACITY = 1536U;

// 构造 Combine 与 Unpermute 共用的量化 token 记录布局。
__aicore__ inline QuantCombineBufferConfig CreateQuantCombineBufferConfig(uint32_t tokenHiddenDim)
{
    uint32_t nAlign32 = Ops::Base::CeilAlign(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
    uint32_t nScale = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
    uint32_t tokenStorageBytes = Ops::Base::CeilAlign(tokenHiddenDim, static_cast<uint32_t>(ALIGN_256));
    uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
    uint32_t quantTokenSizeBytes =
        Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
    uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
    return {static_cast<int64_t>(singleTokenBytes * DOUBLE_BUFFER / sizeof(bfloat16_t)), quantTokenSizeBytes};
}

// 原型：MegaMoeWave::CombineBuffInit。构造 wave Combine 行环形 buffer 和 metadata 视图。
template <uint8_t CombineMode>
__aicore__ inline WaveCombineBufferConfig InitWaveCombineBuffers(const WaveCombineContext &context,
                                                                 WaveCombineScratch &scratch)
{
    WaveCombineBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }
    if (!context.participates) {
        return bufferConfig;
    }
    bufferConfig.rowBytes = context.tokenHiddenDim * sizeof(bfloat16_t);
    bufferConfig.rowStrideBytes = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(bufferConfig.rowBytes), static_cast<uint64_t>(ALIGN_32)));
    bufferConfig.slotStrideBytes = bufferConfig.rowStrideBytes;
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        uint32_t nScale = Ops::Base::CeilDiv(context.tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        uint32_t tokenStorageBytes =
            Ops::Base::CeilAlign(context.tokenHiddenDim, static_cast<uint32_t>(ALIGN_256));
        uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
        bufferConfig.quantRowStorageBytes =
            Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
        bufferConfig.quantRowElements = bufferConfig.quantRowStorageBytes;
        bufferConfig.slotStrideBytes += bufferConfig.quantRowStorageBytes;
        bufferConfig.quantTempElements =
            Ops::Base::CeilAlign(storedScaleBytes, static_cast<uint32_t>(ALIGN_32)) + storedScaleBytes / 2U;
    }
    constexpr uint32_t rowBufferCount =
        CombineMode == COMBINE_NO_QUANT ? WAVE_COMBINE_NO_QUANT_ROW_BUFFER_NUM :
                                         WAVE_COMBINE_QUANT_ROW_BUFFER_NUM;
    uint32_t rowRingBytes = rowBufferCount * bufferConfig.slotStrideBytes;
    scratch.rowBufferTensor = LocalTensor<bfloat16_t>(
        TPosition::VECIN, WAVE_COMBINE_UB_BASE, rowRingBytes / sizeof(bfloat16_t));
    scratch.metaInfoTensor = LocalTensor<int32_t>(
        TPosition::VECCALC, META_INFO_TENSOR_ADDR, WAVE_COMBINE_META_INFO_TOKEN_CAPACITY * META_INFO_SIZE);
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        scratch.quantTempTensor = LocalTensor<float>(
            TPosition::VECIN, WAVE_COMBINE_UB_BASE + rowRingBytes, bufferConfig.quantTempElements);
    }
    return bufferConfig;
}

// 返回一个逻辑 wave Combine 任务负责的连续 token 范围。
__aicore__ inline TokenRange GetWaveCombineOwnedRange(const WaveCombineContext &context, uint32_t tokenCount)
{
    if (!context.participates) {
        return {};
    }
    return GetBalancedTokenRange(tokenCount, context.job.jobIndex, context.job.totalJobs);
}

// 回收当前 wave 的 Combine 发送在行环形 buffer 上产生的事件。
template <uint8_t CombineMode>
__aicore__ inline void DrainWaveCombineRowRing(uint32_t issuedRowCount)
{
    constexpr uint32_t rowBufferCount =
        CombineMode == COMBINE_NO_QUANT ? WAVE_COMBINE_NO_QUANT_ROW_BUFFER_NUM :
                                         WAVE_COMBINE_QUANT_ROW_BUFFER_NUM;
    uint32_t activeSlotCount = issuedRowCount < rowBufferCount ? issuedRowCount : rowBufferCount;
    if (activeSlotCount > 0U) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if (activeSlotCount > 1U) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        if (activeSlotCount > 2U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
        if (activeSlotCount > 3U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        }
        if (activeSlotCount > 4U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        }
        if (activeSlotCount > 5U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        }
    }
}

// 加载一段连续 wave Combine token 的 metadata。
__aicore__ inline void PreloadWaveCombineMetaInfo(const WaveCombineArgs &args, WaveCombineScratch &scratch,
                                                  uint64_t gmTokenOffset, uint32_t tokenCount,
                                                  uint32_t ubTokenOffset)
{
    if (tokenCount == 0U) {
        return;
    }
    LocalTensor<int32_t> metaInfoUb = scratch.metaInfoTensor[ubTokenOffset * META_INFO_SIZE];
    GlobalTensor<int32_t> metaInfoGm;
    metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(args.metaInfoPtr));
    DataCopy(metaInfoUb, metaInfoGm[gmTokenOffset * META_INFO_SIZE], tokenCount * META_INFO_SIZE);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

// 搬运并按需量化一个 wave Combine token 后发送；函数内部不包含 GMM2-ready 依赖。
template <uint8_t CombineMode, bool IsBufferReuse>
__aicore__ inline void SendWaveCombineToken(const WaveCombineContext &context,
                                            const WaveCombineBufferConfig &bufferConfig,
                                            WaveCombineScratch &scratch, const Params &params,
                                            GlobalTensor<bfloat16_t> &gmm2OutGm, uint64_t gmRemoteBaseOffset,
                                            uint32_t tokenLocal, LocalTensor<int32_t> &tokenMetaInfo, uint32_t slot)
{
    TEventID eventId =
        static_cast<TEventID>(static_cast<int32_t>(EVENT_ID0) + static_cast<int32_t>(slot));
    if constexpr (IsBufferReuse) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    uint32_t slotElementOffset = slot * bufferConfig.slotStrideBytes / sizeof(bfloat16_t);
    LocalTensor<bfloat16_t> rowUb = scratch.rowBufferTensor[slotElementOffset];
    DataCopyExtParams gm2UbParams{1U, bufferConfig.rowBytes, 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> gm2UbPad{false, 0U, 0U, 0U};
    DataCopyPad(rowUb, gmm2OutGm[static_cast<uint64_t>(tokenLocal) * context.tokenHiddenDim],
                gm2UbParams, gm2UbPad);
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        MegaMoeCombineImpl::SendCombineTokenRow<bfloat16_t>(
            context.tokenHiddenDim, gmRemoteBaseOffset, tokenMetaInfo, rowUb, params);
    } else {
        LocalTensor<bfloat16_t> quantUb =
            scratch.rowBufferTensor[slotElementOffset + bufferConfig.rowStrideBytes / sizeof(bfloat16_t)];
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        MegaMoeCombineImpl::QuantMxFp8<CombineMode, bfloat16_t>(
            quantUb, rowUb, scratch.quantTempTensor, context.tokenHiddenDim);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT,
                                                  fp8_e4m3fn_t, fp8_e5m2_t>::type;
        LocalTensor<Fp8Type> quantSendUb = quantUb.template ReinterpretCast<Fp8Type>();
        MegaMoeCombineImpl::SendCombineTokenRow<Fp8Type>(
            bufferConfig.quantRowElements, gmRemoteBaseOffset, tokenMetaInfo, quantSendUb, params);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

// 原型：MegaMoeWave::ProcessCombineGm。发送一个逻辑 AIV 任务负责的已就绪 token 范围。
template <uint8_t CombineMode>
__aicore__ inline void CombineWaveTokenRange(const WaveCombineContext &context,
                                             const WaveCombineBufferConfig &bufferConfig,
                                             WaveCombineScratch &scratch, const Params &params,
                                             GM_ADDR gmm2OutGlobal, uint32_t tokenStart, uint32_t tokenCount,
                                             uint32_t metaInfoUbTokenOffset, uint32_t &rowSequence)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!context.participates || tokenCount == 0U) {
        return;
    }
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        AscendC::SetCtrlSpr<60, 60>(0);
    }
    GlobalTensor<bfloat16_t> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gmm2OutGlobal));
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;
    constexpr uint32_t rowBufferCount =
        CombineMode == COMBINE_NO_QUANT ? WAVE_COMBINE_NO_QUANT_ROW_BUFFER_NUM :
                                         WAVE_COMBINE_QUANT_ROW_BUFFER_NUM;
    uint32_t tokenIdxInSlice = 0U;
    uint32_t firstUseCount = rowSequence < rowBufferCount ? rowBufferCount - rowSequence : 0U;
    firstUseCount = firstUseCount < tokenCount ? firstUseCount : tokenCount;
    for (; tokenIdxInSlice < firstUseCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        LocalTensor<int32_t> tokenMetaInfo =
            scratch.metaInfoTensor[(metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE];
        SendWaveCombineToken<CombineMode, false>(context, bufferConfig, scratch, params, gmm2OutGm,
                                                 gmRemoteBaseOffset, tokenStart + tokenIdxInSlice,
                                                 tokenMetaInfo, slot);
    }
    for (; tokenIdxInSlice < tokenCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        LocalTensor<int32_t> tokenMetaInfo =
            scratch.metaInfoTensor[(metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE];
        SendWaveCombineToken<CombineMode, true>(context, bufferConfig, scratch, params, gmm2OutGm,
                                                gmRemoteBaseOffset, tokenStart + tokenIdxInSlice,
                                                tokenMetaInfo, slot);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_COMBINE_H
