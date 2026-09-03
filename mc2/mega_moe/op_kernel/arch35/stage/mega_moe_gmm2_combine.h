/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM2_COMBINE_H
#define MEGA_MOE_GMM2_COMBINE_H

#include <type_traits>

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "adv_api/reduce/reduce.h"
#include "apace/kernel/matmul/grouped_matmul/run_gmm2.h"
#include "../common/mega_moe_utils.h"
#include "../common/mega_moe_mxfp8_utils.h"
#include "../moe_gmm_sync_policy.h"
#if __has_include("../../../common/mc2_kernel_utils.h")
#include "../../../common/mc2_kernel_utils.h"
#include "../../../common/quantize_functions.h"
#else
#include "../../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../../common/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;
using namespace Apace::Gemm::Kernel;

namespace CombineImpl {

// Combine 发送路由：dispatch 阶段写入 metaInfo 的 (目标rank, 原token行, topk槽) 三元组。
struct CombineTokenRoute {
    uint32_t dstRankId;
    uint32_t tokenIdx;
    uint32_t topkIdx;
};

__aicore__ inline CombineTokenRoute LoadCombineTokenRoute(const LocalTensor<int32_t> &metaInfoTensor,
                                                          uint32_t recordIdx)
{
    uint32_t recordBase = recordIdx * META_INFO_SIZE;
    return {static_cast<uint32_t>(metaInfoTensor.GetValue(recordBase + RANK_ID)),
            static_cast<uint32_t>(metaInfoTensor.GetValue(recordBase + TOKEN_ID)),
            static_cast<uint32_t>(metaInfoTensor.GetValue(recordBase + TOPK_INDEX))};
}

// 目标卡 combine 接收区按 (token, topk) 展开的紧凑行号。
__aicore__ inline uint64_t GetCombineDstRowIndex(const CombineTokenRoute &route, const Params &params)
{
    return static_cast<uint64_t>(route.tokenIdx) * params.tilingData->topK + route.topkIdx;
}

// 为 tile 内每个 token 行发送一段有效的 GMM2 tile 数据。
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(uint32_t nLoc, uint32_t n, LocalTensor<int32_t> &metaInfoTensor,
                                     LocalTensor<ElementMMadOut2> &l0cOutUbGMM2, BlockShape &actualBlockShape,
                                     uint32_t ubTileN, const Params &params)
{
    // 调用方在进入该数据操作前，保证批量加载的 metadata 对 Scalar 可见。
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
    uint64_t gmRemoteBaseOffset =
        static_cast<uint64_t>(params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr);
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        CombineTokenRoute route = LoadCombineTokenRoute(metaInfoTensor, static_cast<uint32_t>(tileIdx));
        gmRemoteD.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementMMadOut2 *>(GetRankWinAddrWithOffset(route.dstRankId, gmRemoteBaseOffset)));
        uint64_t gmDstOffset = GetCombineDstRowIndex(route, params) * n + nLoc;
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset], l0cOutUbGMM2[tileIdx * ubTileN], ub2GmParams);
    }
}

// 发送一行完整的 Combine 数据，BF16 与 FP8 记录复用相同的定长搬运流程。
template <typename Element>
__aicore__ inline void SendCombineTokenRow(uint32_t rowElements, uint64_t gmRemoteBaseOffset,
                                           LocalTensor<int32_t> &metaInfoTensor, LocalTensor<Element> &rowTensor,
                                           const Params &params)
{
    CombineTokenRoute route = LoadCombineTokenRoute(metaInfoTensor, 0U);

    GlobalTensor<Element> gmRemoteD;
    gmRemoteD.SetGlobalBuffer(
        reinterpret_cast<__gm__ Element *>(GetRankWinAddrWithOffset(route.dstRankId, gmRemoteBaseOffset)));
    uint64_t gmDstRowOffset = GetCombineDstRowIndex(route, params) * rowElements;

    DataCopyExtParams ub2GmParams{1U, static_cast<uint32_t>(rowElements * sizeof(Element)), 0U, 0U, 0U};
    DataCopyPad(gmRemoteD[gmDstRowOffset], rowTensor, ub2GmParams);
}

// 通过本地 MTE 路径或远端 URMA 路径发送一个 layered 模板 token。
template <typename DataType, bool IsQuantized = true>
__aicore__ inline void CombineSendTokenToRemote(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                                uint32_t groupIdx, uint32_t rankId,
                                                LocalTensor<int32_t> &metaInfoTensor, LocalTensor<DataType> &ubQuant,
                                                const Params &params, GM_ADDR localSrcPtr)
{
#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
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
    uint64_t dstBaseOffset = (static_cast<uint64_t>(tokenIdx) * params.tilingData->topK + topkIdx) * quantTokenSize;
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
        params.combineCommParams.hcomm->WriteNbi(channelHandle, remoteAddr, srcAddr, quantTokenSize * sizeof(DataType));
    }
#endif
}

// 将一条量化 token 记录发送到 metadata 指定的 rank 和目标行。
template <typename QuantOutType>
__aicore__ inline void CombineQuantizedTokens(uint32_t batchStart, uint32_t curRows, uint32_t n, uint32_t nScale,
                                              uint32_t groupIdx, uint32_t rankId, LocalTensor<int32_t> &metaInfoTensor,
                                              LocalTensor<QuantOutType> &ubQuant, const Params &params,
                                              uint32_t quantTokenSizeBytes)
{
    CombineTokenRoute route = LoadCombineTokenRoute(metaInfoTensor, batchStart);

    AscendC::GlobalTensor<QuantOutType> gmRemoteD;
    uint64_t gmRemoteOffset =
        static_cast<uint64_t>(params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr);
    __gm__ void *dstPeermemPtr = GetRankWinAddrWithOffset(route.dstRankId, gmRemoteOffset);
    gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(dstPeermemPtr));

    uint64_t dstBaseOffset = GetCombineDstRowIndex(route, params) * quantTokenSizeBytes;
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
    uint32_t mxScaleNum = Ops::Base::CeilAlign(nScale, 2U);
    uint32_t nAlign32 = Ops::Base::CeilAlign(n, static_cast<uint32_t>(ALIGN_32));
    uint32_t floatTempSize = Ops::Base::CeilAlign(mxScaleNum, static_cast<uint32_t>(ALIGN_32)) + mxScaleNum / 2;
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
            Mxfp8::QuantMxFp8<QuantMode, T>(ubQuantData, ubBf16, floatTemp, n);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID5>();
        }

        using SendType = typename std::conditional<IsQuantized, Fp8Type, T>::type;
        LocalTensor<SendType> ubQuantSend = ubQuantData.template ReinterpretCast<SendType>();
        if constexpr (IsLayered) {
            if constexpr (!IsQuantized) {
                ubQuantSend = ubBf16;
            }
            GM_ADDR localSrcPtr = gmm2OutAddr + (tokenStart + i) * n * sizeof(T);
            CombineSendTokenToRemote<SendType, IsQuantized>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor,
                                                            ubQuantSend, params, localSrcPtr);
        } else {
            CombineQuantizedTokens<SendType>(i, 1, n, nScale, groupIdx, rankId, metaInfoTensor, ubQuantSend, params,
                                             quantTokenSizeBytes);
        }
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
}

} // namespace CombineImpl

constexpr uint32_t META_INFO_TENSOR_ADDR = 200U * 1024U;
constexpr int32_t MAX_AICORE_NUM = 36;

struct QuantTokenBufferConfig {
    uint32_t quantTokenSizeBytes;
};

struct WaveCombineBufferConfig {
    uint32_t rowBytes = 0;
    uint32_t rowStrideBytes = 0;
    uint32_t quantRowElements = 0;
    uint32_t quantRowStorageBytes = 0;
    uint32_t slotStrideBytes = 0;
    uint32_t quantTempElements = 0;
    uint32_t rowBufferCount = 0;
};

struct WaveCombineScratch {
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<bfloat16_t> rowBufferTensor;
    LocalTensor<float> quantTempTensor;
};

constexpr uint32_t WAVE_COMBINE_MIN_ROW_BUFFER_COUNT = 2U;
constexpr uint32_t WAVE_COMBINE_STEADY_ROW_BUFFER_COUNT = 1U;
constexpr uint32_t WAVE_COMBINE_MAX_ROW_BUFFER_COUNT = 6U;
constexpr uint32_t WAVE_COMBINE_UB_BASE = 64U * 1024U;
// [64 KiB, 184 KiB) is dedicated to the Combine row ring and quant scratch.
// The ready scan starts at 184 KiB, so adaptive row buffers must stay below it.
constexpr uint32_t WAVE_COMBINE_UB_LIMIT = 184U * 1024U;
constexpr uint32_t WAVE_COMBINE_META_INFO_TOKEN_CAPACITY = 1536U;

// 非 layered 路径统一使用的 Combine 行环。常规 Wave 只有 AIV1 建立并使用这些 UB 视图；
// 最后一轮不再与 GMM1/SwiGLU 并行时，调用方才为 AIV0 补充初始化。
template <uint8_t CombineMode, bool IncludeAiv0 = false, uint32_t MaxRowBufferCount = WAVE_COMBINE_MAX_ROW_BUFFER_COUNT>
__aicore__ inline WaveCombineBufferConfig InitWaveCombineBuffers(const MoeStageCommonConfig &common,
                                                                 WaveCombineScratch &scratch)
{
    static_assert(MaxRowBufferCount > 0U && MaxRowBufferCount <= WAVE_COMBINE_MAX_ROW_BUFFER_COUNT,
                  "invalid Combine row buffer count");
    WaveCombineBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }
    if (!IncludeAiv0 && GetSubBlockIdx() != 1U) {
        return bufferConfig;
    }
    bufferConfig.rowBytes = common.tokenHiddenDim * sizeof(bfloat16_t);
    bufferConfig.rowStrideBytes = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(bufferConfig.rowBytes), static_cast<uint64_t>(ALIGN_32)));
    bufferConfig.slotStrideBytes = bufferConfig.rowStrideBytes;
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        uint32_t nScale = Ops::Base::CeilDiv(common.tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        uint32_t tokenStorageBytes = Ops::Base::CeilAlign(common.tokenHiddenDim, static_cast<uint32_t>(ALIGN_256));
        uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
        bufferConfig.quantRowStorageBytes =
            Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
        bufferConfig.quantRowElements = bufferConfig.quantRowStorageBytes;
        bufferConfig.slotStrideBytes += bufferConfig.quantRowStorageBytes;
        bufferConfig.quantTempElements =
            Ops::Base::CeilAlign(storedScaleBytes, static_cast<uint32_t>(ALIGN_32)) + storedScaleBytes / 2U;
    }
    bufferConfig.rowBufferCount = MaxRowBufferCount;
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        uint32_t quantTempBytes = bufferConfig.quantTempElements * sizeof(float);
        uint32_t rowRingBudgetBytes = WAVE_COMBINE_UB_LIMIT - WAVE_COMBINE_UB_BASE - quantTempBytes;
        constexpr uint32_t minRowBufferCount = MaxRowBufferCount < WAVE_COMBINE_MIN_ROW_BUFFER_COUNT ?
                                                   MaxRowBufferCount :
                                                   WAVE_COMBINE_MIN_ROW_BUFFER_COUNT;
        bufferConfig.rowBufferCount = rowRingBudgetBytes / bufferConfig.slotStrideBytes;
        bufferConfig.rowBufferCount =
            bufferConfig.rowBufferCount < minRowBufferCount ? minRowBufferCount : bufferConfig.rowBufferCount;
        bufferConfig.rowBufferCount =
            bufferConfig.rowBufferCount > MaxRowBufferCount ? MaxRowBufferCount : bufferConfig.rowBufferCount;
    }
    uint32_t rowRingBytes = bufferConfig.rowBufferCount * bufferConfig.slotStrideBytes;
    scratch.rowBufferTensor =
        LocalTensor<bfloat16_t>(TPosition::VECIN, WAVE_COMBINE_UB_BASE, rowRingBytes / sizeof(bfloat16_t));
    scratch.metaInfoTensor = LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR,
                                                  WAVE_COMBINE_META_INFO_TOKEN_CAPACITY * META_INFO_SIZE);
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        scratch.quantTempTensor =
            LocalTensor<float>(TPosition::VECIN, WAVE_COMBINE_UB_BASE + rowRingBytes, bufferConfig.quantTempElements);
    }
    return bufferConfig;
}

// 默认由 AIV1 复用既有配置，仅为 AIV0 绑定最终 Combine 的 UB 视图。steady Wave 使用较小行环，
// 调用方排空旧行环后通过 ReinitializeAllAiv 让两个 AIV 都恢复最终 Wave 的最大容量。
template <uint8_t CombineMode, bool ReinitializeAllAiv = false>
__aicore__ inline WaveCombineBufferConfig PrepareFinalWaveCombineBuffers(
    const MoeStageCommonConfig &common, const WaveCombineBufferConfig &currentBufferConfig, WaveCombineScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return currentBufferConfig;
    }
    if (ReinitializeAllAiv || GetSubBlockIdx() == 0U) {
        return InitWaveCombineBuffers<CombineMode, true>(common, scratch);
    }
    return currentBufferConfig;
}

// 构造量化 Unpermute 输入记录的单 token 存储布局。
__aicore__ inline QuantTokenBufferConfig CreateQuantTokenBufferConfig(uint32_t tokenHiddenDim)
{
    uint32_t nScale = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
    uint32_t tokenStorageBytes = Ops::Base::CeilAlign(tokenHiddenDim, static_cast<uint32_t>(ALIGN_256));
    uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
    uint32_t quantTokenSizeBytes =
        Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
    return {quantTokenSizeBytes};
}

template <bool UseAllAivCores>
__aicore__ inline bool IsWaveCombineWorker()
{
    return UseAllAivCores || GetSubBlockIdx() == 1U;
}

// 常规 Wave 只使用每个 AIC 配对的 AIV1；最后一轮把同组 AIV0/AIV1 展开成两个独立任务。
template <bool UseAllAivCores>
__aicore__ inline AivJobContext GetWaveCombineJob(const AivJobContext &job)
{
    if constexpr (UseAllAivCores) {
        constexpr uint32_t AIV_CORE_COUNT_PER_AIC = 2U;
        uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
        return {job.jobIndex * AIV_CORE_COUNT_PER_AIC + subBlockIdx, job.totalJobs * AIV_CORE_COUNT_PER_AIC};
    }
    return job;
}

// 按专家累计 token 行数滚动物理 AIV 起点，每个 AIV 在当前专家内仍负责一个连续均衡区间。
template <bool UseAllAivCores = false>
__aicore__ inline WorkRange GetWaveCombineOwnedRange(const AivJobContext &job, uint32_t tokenCount,
                                                     uint64_t expertRowPrefix)
{
    if constexpr (g_coreType == AIC) {
        return {};
    }
    if (!IsWaveCombineWorker<UseAllAivCores>()) {
        return {};
    }
    AivJobContext combineJob = GetWaveCombineJob<UseAllAivCores>(job);
    if (combineJob.totalJobs == 0U || combineJob.jobIndex >= combineJob.totalJobs) {
        return {};
    }
    uint32_t firstOwnerJob = expertRowPrefix % combineJob.totalJobs;
    uint32_t logicalJobIndex = combineJob.jobIndex >= firstOwnerJob ?
                                   combineJob.jobIndex - firstOwnerJob :
                                   combineJob.jobIndex + combineJob.totalJobs - firstOwnerJob;
    return GetBalancedTokenRange(tokenCount, logicalJobIndex, combineJob.totalJobs);
}

// 等待并消费当前 Wave 的 Combine 发送在行环形 buffer 上产生的全部完成事件，使该环不再有在途 slot。
__aicore__ inline void DrainCombineRowBuffers(uint32_t &issuedRowCount, uint32_t rowBufferCount)
{
    uint32_t activeSlotCount = issuedRowCount < rowBufferCount ? issuedRowCount : rowBufferCount;
    for (uint32_t slot = 0U; slot < activeSlotCount; ++slot) {
        WaitFlag<HardEvent::MTE3_MTE2>(
            static_cast<TEventID>(static_cast<int32_t>(EVENT_ID0) + static_cast<int32_t>(slot)));
    }
    /*
     * Drain 会消费所有在途事件，下一段必须从空闲的 slot 0 重新开始；否则该段会把已消费的事件
     * 误判为 buffer 复用依赖并再次等待。W4 的计数跨 Wave 存活，A8W8 则在每次 Combine 调用时
     * 创建新计数；在这里统一重置，可以保证两种生命周期都遵守相同的行环起始状态。
     */
    issuedRowCount = 0U;
}

// 加载一段连续 wave Combine token 的 metadata。
__aicore__ inline void PreloadWaveCombineMetaInfo(const Params &params, WaveCombineScratch &scratch,
                                                  uint64_t gmTokenOffset, uint32_t tokenCount, uint32_t ubTokenOffset)
{
    if (tokenCount == 0U) {
        return;
    }
    LocalTensor<int32_t> metaInfoUb = scratch.metaInfoTensor[ubTokenOffset * META_INFO_SIZE];
    GlobalTensor<int32_t> metaInfoGm;
    metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr));
    DataCopy(metaInfoUb, metaInfoGm[gmTokenOffset * META_INFO_SIZE], tokenCount * META_INFO_SIZE);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

// 搬运并按需量化一个 wave Combine token 后发送；函数内部不包含 GMM2-ready 依赖。
template <uint8_t CombineMode, bool IsBufferReuse>
__aicore__ inline void SendWaveCombineToken(const MoeStageCommonConfig &common,
                                            const WaveCombineBufferConfig &bufferConfig, WaveCombineScratch &scratch,
                                            const Params &params, GlobalTensor<bfloat16_t> &gmm2OutGm,
                                            uint64_t gmRemoteBaseOffset, uint32_t tokenLocal,
                                            LocalTensor<int32_t> &tokenMetaInfo, uint32_t slot)
{
    TEventID eventId = static_cast<TEventID>(static_cast<int32_t>(EVENT_ID0) + static_cast<int32_t>(slot));
    if constexpr (IsBufferReuse) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    uint32_t slotElementOffset = slot * bufferConfig.slotStrideBytes / sizeof(bfloat16_t);
    LocalTensor<bfloat16_t> rowUb = scratch.rowBufferTensor[slotElementOffset];
    DataCopyExtParams gm2UbParams{1U, bufferConfig.rowBytes, 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> gm2UbPad{false, 0U, 0U, 0U};
    DataCopyPad(rowUb, gmm2OutGm[static_cast<uint64_t>(tokenLocal) * common.tokenHiddenDim], gm2UbParams, gm2UbPad);
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        CombineImpl::SendCombineTokenRow<bfloat16_t>(common.tokenHiddenDim, gmRemoteBaseOffset, tokenMetaInfo, rowUb,
                                                     params);
    } else {
        LocalTensor<bfloat16_t> quantUb =
            scratch.rowBufferTensor[slotElementOffset + bufferConfig.rowStrideBytes / sizeof(bfloat16_t)];
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        Mxfp8::QuantMxFp8<CombineMode, bfloat16_t>(quantUb, rowUb, scratch.quantTempTensor, common.tokenHiddenDim);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
        LocalTensor<Fp8Type> quantSendUb = quantUb.template ReinterpretCast<Fp8Type>();
        CombineImpl::SendCombineTokenRow<Fp8Type>(bufferConfig.quantRowElements, gmRemoteBaseOffset, tokenMetaInfo,
                                                  quantSendUb, params);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

// 使用行级流水发送当前逻辑 AIV 任务负责的已就绪 token 区间。
template <uint8_t CombineMode, bool UseAllAivCores = false>
__aicore__ inline void CombineWaveTokenRange(const MoeStageCommonConfig &common,
                                             const WaveCombineBufferConfig &bufferConfig, WaveCombineScratch &scratch,
                                             const Params &params, GM_ADDR gmm2OutGlobal, uint32_t tokenStart,
                                             uint32_t tokenCount, uint32_t metaInfoUbTokenOffset, uint32_t &rowSequence)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!IsWaveCombineWorker<UseAllAivCores>() || tokenCount == 0U) {
        return;
    }
    GlobalTensor<bfloat16_t> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gmm2OutGlobal));
    uint64_t gmRemoteBaseOffset =
        static_cast<uint64_t>(params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr);
    uint32_t rowBufferCount = bufferConfig.rowBufferCount;
    uint32_t tokenIdxInSlice = 0U;
    uint32_t firstUseCount = rowSequence < rowBufferCount ? rowBufferCount - rowSequence : 0U;
    firstUseCount = firstUseCount < tokenCount ? firstUseCount : tokenCount;
    for (; tokenIdxInSlice < firstUseCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        LocalTensor<int32_t> tokenMetaInfo =
            scratch.metaInfoTensor[(metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE];
        SendWaveCombineToken<CombineMode, false>(common, bufferConfig, scratch, params, gmm2OutGm, gmRemoteBaseOffset,
                                                 tokenStart + tokenIdxInSlice, tokenMetaInfo, slot);
    }
    for (; tokenIdxInSlice < tokenCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % rowBufferCount;
        LocalTensor<int32_t> tokenMetaInfo =
            scratch.metaInfoTensor[(metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE];
        SendWaveCombineToken<CombineMode, true>(common, bufferConfig, scratch, params, gmm2OutGm, gmRemoteBaseOffset,
                                                tokenStart + tokenIdxInSlice, tokenMetaInfo, slot);
    }
}

// 等待当前 GMM2 tile 对应的 SwiGLU 输入就绪。


// 紧凑 token 资源按累计行偏移寻址，专家固定资源按 expertIdx 寻址。
template <typename WeightType, typename ActivationOutType, typename QuantScaleType, bool ConfigureGmm2Output,
          bool ConfigureCombineCounter>
__aicore__ inline void UpdateMoeExpertGmm2GlobalBuffer(const GmmExecutionConfig &gmmConfig,
                                                       const MoeSyncWorkspaceLayout &syncLayout,
                                                       const WorkspaceInfo &workspace,
                                                       const ExpertWeightTensorListAddrs &weights,
                                                       GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                       uint32_t rowOffsetInExpert = 0U)
{
    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t activationOutputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    uint64_t gmm1OutputDim = static_cast<uint64_t>(Get<N_VALUE>(state.problemShape));
    uint64_t tokenHiddenDim = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    uint64_t globalTokenStartIndex = static_cast<uint64_t>(state.globalTokenStartIndex) + rowOffsetInExpert;
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    uint64_t activationOutputWidth = gmm1OutputDim / ACTIVATION_N_HALF;
    uint64_t activationScaleWidth =
        Ops::Base::CeilDiv(activationOutputWidth, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if constexpr (ConfigureGmm2Output) {
        gmmAddrInfo.tensorAddrs.gmm2OutGlobal =
            workspace.gmm2MmadResPtr + globalTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
    }
    gmmAddrInfo.metaInfoGlobal = workspace.metaInfoPtr + globalTokenStartIndex * META_INFO_SIZE * sizeof(int32_t);
    gmmAddrInfo.tensorAddrs.aGlobal = workspace.activationQuantDataPtr + globalTokenStartIndex * activationOutputWidth /
                                                                 activationOutputElementsPerByte *
                                                                 sizeof(ActivationOutType);
    gmmAddrInfo.tensorAddrs.aScaleGlobal =
        workspace.activationQuantScalePtr + globalTokenStartIndex * activationScaleWidth * sizeof(QuantScaleType);
    gmmAddrInfo.tensorAddrs.bGlobal = GetExpertWeightAddr<WeightType>(
        weights.weight2, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
        static_cast<uint64_t>(state.expertIdx) * tokenHiddenDim * activationOutputWidth / weightElementsPerByte);
    gmmAddrInfo.tensorAddrs.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
        static_cast<uint64_t>(state.expertIdx) * tokenHiddenDim * activationScaleWidth);
    if constexpr (ConfigureCombineCounter) {
        uint64_t syncSlotOffset = static_cast<uint64_t>(state.expertIdx) * syncLayout.combineSyncSlotCountPerExpert;
        gmmAddrInfo.gmm2CombineSyncCounter = reinterpret_cast<__gm__ int32_t *>(workspace.gmm2CombineSyncCounterPtr) +
                                             syncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
    }
    gmmAddrInfo.activationToGmm2Flag =
        reinterpret_cast<__gm__ int32_t *>(workspace.flagActivationToGmm2Ptr) +
        static_cast<uint64_t>(state.expertIdx) * syncLayout.activationFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag =
        reinterpret_cast<__gm__ int32_t *>(workspace.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(state.expertIdx) * syncLayout.dispatchFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
}

// Combine 只消费当前专家的 GMM2 输出；避免复用完整 GMM2 绑定接口时计算无关的输入、权重和 flag 地址。
__aicore__ inline void UpdateMoeExpertCombineGlobalBuffer(const WorkspaceInfo &workspace, GMMAddrInfo &gmmAddrInfo,
                                                          const ExpertLoopState &state)
{
    uint64_t tokenHiddenDim = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    uint64_t globalTokenStartIndex = static_cast<uint64_t>(state.globalTokenStartIndex);
    gmmAddrInfo.tensorAddrs.gmm2OutGlobal = workspace.gmm2MmadResPtr + globalTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
}

template <typename ActivationType, typename WeightType, typename QuantScaleType, uint32_t Gmm1TileM>
__aicore__ inline void UpdateSharedExpertGmm2GlobalBuffer(const MoeStageCommonConfig &commonConfig,
                                                          const GmmExecutionConfig &gmmConfig,
                                                          const WorkspaceInfo &workspace,
                                                          const ExpertWeightTensorListAddrs &weights,
                                                          GMMAddrInfo &gmmAddrInfo, uint32_t sharedExpertIdx)
{
    constexpr uint32_t activationElementsPerByte = PackedElementTraits<ActivationType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    uint64_t tokenNum = commonConfig.tokenNum;
    uint64_t tokenHiddenDim = commonConfig.tokenHiddenDim;
    uint64_t activationOutputWidth = commonConfig.gmm1OutputDim / ACTIVATION_N_HALF;
    uint64_t activationScaleWidth =
        Ops::Base::CeilDiv(activationOutputWidth, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t sharedExpertTokenStartIndex = static_cast<uint64_t>(sharedExpertIdx) * tokenNum;

    gmmAddrInfo.tensorAddrs.gmm2OutGlobal =
        workspace.sharedExpertResultPtr + sharedExpertTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
    gmmAddrInfo.tensorAddrs.aGlobal =
        workspace.sharedExpertActivationDataPtr +
        sharedExpertTokenStartIndex * activationOutputWidth / activationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.tensorAddrs.aScaleGlobal = workspace.sharedExpertActivationScalePtr +
                               sharedExpertTokenStartIndex * activationScaleWidth * sizeof(QuantScaleType);
    gmmAddrInfo.tensorAddrs.bGlobal = GetExpertWeightAddr<WeightType>(
        weights.weight2, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx,
        static_cast<uint64_t>(sharedExpertIdx) * tokenHiddenDim * activationOutputWidth / weightElementsPerByte);
    gmmAddrInfo.tensorAddrs.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx,
        static_cast<uint64_t>(sharedExpertIdx) * tokenHiddenDim * activationScaleWidth);
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(commonConfig.tokenNum, Gmm1TileM);
    gmmAddrInfo.sharedExpertGmm2TileCounter =
        reinterpret_cast<__gm__ int32_t *>(workspace.sharedExpertGmm2TileCounterPtr) +
        static_cast<uint64_t>(sharedExpertIdx) * tokenGroupCount * INT_CACHELINE;
}

// 供普通模板、Wave 模板和共享专家的 GMM2 阶段复用。
// 使用调用方传入的 block 任务执行 GMM2，并保持所选同步约定不变。
template <typename GenericElementA, typename A8W4ElementA, typename WeightType,
          typename QuantScaleType, bool EnableA8W4, bool EnableA4W4, uint32_t Gmm1TileM, bool TopkWeightsPrefetch,
          bool IsShared, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false>
__aicore__ inline void RunGmm2ByMode(const GmmExecutionConfig &gmmConfig, const GMMAddrInfo &gmmAddrInfo,
                                     const ProblemShape &problemShape, GmmRuntimeState &runtimeState,
                                     void *persistentBlockMmadContext = nullptr, bool allowWeightL2Bypass = false)
{
    // GMM2 等待上游（GMM1+Act）的通知粒度（原 FinalizeProblemConfig 语义：Prefetch 时减半）。
    uint32_t activationTileM = TopkWeightsPrefetch ? L1_TILE_M_128 : Gmm1TileM;
    if constexpr (EnableA8W4 || EnableA4W4) {
        Gmm2A8W4SyncPolicy<false, IsShared, IsWaveFlagGrained, false> sync(
            gmmAddrInfo, problemShape, activationTileM, gmmConfig.blockJob.totalJobs,
            static_cast<uint32_t>(Get<M_VALUE>(problemShape)), 0U);
        RunGmm2A8W4<A8W4ElementA, WeightType, bfloat16_t, QuantScaleType, QuantScaleType>(
            problemShape, gmmAddrInfo.tensorAddrs, runtimeState.startBlockIdx, gmmConfig.blockJob, sync);
    } else if (gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        Gmm2UbSyncPolicy<false, IsShared, IsWaveFlagGrained, IsGmm1Interleaved> sync(
            gmmAddrInfo, problemShape, activationTileM, gmmConfig.blockJob.totalJobs);
        RunGmm2Generic<GenericElementA, GenericElementA, bfloat16_t, QuantScaleType, QuantScaleType, true,
                        IsWaveFlagGrained>(problemShape, gmmAddrInfo.tensorAddrs, runtimeState.startBlockIdx,
                                           gmmConfig.blockJob, sync, persistentBlockMmadContext,
                                           allowWeightL2Bypass);
    } else {
        Gmm2UbSyncPolicy<false, IsShared, IsWaveFlagGrained, IsGmm1Interleaved> sync(
            gmmAddrInfo, problemShape, activationTileM, gmmConfig.blockJob.totalJobs);
        RunGmm2Generic<GenericElementA, GenericElementA, bfloat16_t, QuantScaleType, QuantScaleType, false,
                        IsWaveFlagGrained>(problemShape, gmmAddrInfo.tensorAddrs, runtimeState.startBlockIdx,
                                           gmmConfig.blockJob, sync, persistentBlockMmadContext,
                                           allowWeightL2Bypass);
    }
}

constexpr uint32_t WAVE_GMM2_READY_SCAN_UB_ADDR = WAVE_COMBINE_UB_LIMIT;
constexpr uint32_t WAVE_GMM2_READY_MAX_SCAN_BYTES = MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t) > ALIGN_512 ?
                                                        MAX_AICORE_NUM *INT_CACHELINE * sizeof(int32_t) :
                                                        ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR =
    WAVE_GMM2_READY_SCAN_UB_ADDR + ((WAVE_GMM2_READY_MAX_SCAN_BYTES + ALIGN_512 - 1U) / ALIGN_512) * ALIGN_512;
constexpr uint32_t WAVE_GMM2_READY_SUM_UB_ADDR = WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR + ALIGN_512;

__aicore__ inline uint64_t GetWaveGmm2ReadySlotStride(const AivJobContext &job)
{
    return static_cast<uint64_t>(job.totalJobs) * INT_CACHELINE;
}

// 每个 AIC 在 GMM2 写入可见后发布一个独占 cache line 的完成标记。
__aicore__ inline void NotifyWaveGmm2Ready(const AivJobContext &job, const Params &params, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm2ReadyPtr);
        uint64_t slotOffset = static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(job);
        AscendC::WriteGmByPassDCache(readyBase + slotOffset + static_cast<uint64_t>(job.jobIndex) * INT_CACHELINE,
                                     int32_t(1));
    }
}

// 一个 AIV Combine 任务等待全部 AIC 完成指定专家。
template <bool UseAllAivCores = false>
__aicore__ inline void WaitWaveGmm2Ready(const AivJobContext &job, const Params &params, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!IsWaveCombineWorker<UseAllAivCores>() || job.totalJobs == 0U) {
        return;
    }
    uint32_t readyElements = job.totalJobs * static_cast<uint32_t>(INT_CACHELINE);
    __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm2ReadyPtr);
    uint64_t readySlotOffset = static_cast<uint64_t>(slotIdx) * GetWaveGmm2ReadySlotStride(job);
    GlobalTensor<int32_t> readyGm;
    readyGm.SetGlobalBuffer(readyBase);
    LocalTensor<int32_t> scanUb(TPosition::VECCALC, WAVE_GMM2_READY_SCAN_UB_ADDR, readyElements);
    LocalTensor<uint8_t> reduceTmpUb(TPosition::VECCALC, WAVE_GMM2_READY_REDUCE_TMP_UB_ADDR, ALIGN_512);
    LocalTensor<int32_t> sumUb(TPosition::VECCALC, WAVE_GMM2_READY_SUM_UB_ADDR, INT_CACHELINE);
    const uint32_t readyShape[] = {1U, readyElements};
    while (true) {
        DataCopy(scanUb, readyGm[readySlotOffset], readyElements);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ReduceSum<int32_t, AscendC::Pattern::Reduce::AR, true>(sumUb, scanUb, reduceTmpUb, readyShape, true);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (sumUb.GetValue(0) >= static_cast<int32_t>(job.totalJobs)) {
            return;
        }
        int64_t waitStart = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - waitStart < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}

template <uint8_t CombineMode, bool UseAllAivCores = false>
__aicore__ inline void RunWaveExpertCombineStage(const MoeStageCommonConfig &common, const AivJobContext &job,
                                                 const WaveCombineBufferConfig &bufferConfig,
                                                 WaveCombineScratch &scratch, const Params &params,
                                                 const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                 uint32_t expertIdx, uint32_t &rowSequence)
{
    uint32_t currentExpertTokenNum = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    WorkRange currentCoreTokenRange = GetWaveCombineOwnedRange<UseAllAivCores>(
        job, currentExpertTokenNum, static_cast<uint64_t>(state.globalTokenStartIndex));
    if (currentCoreTokenRange.count != 0U) {
        WaitWaveGmm2Ready<UseAllAivCores>(job, params, expertIdx);
    }
    for (uint32_t processedTokenNum = 0U; processedTokenNum < currentCoreTokenRange.count;) {
        uint32_t remainingTokenNum = currentCoreTokenRange.count - processedTokenNum;
        uint32_t chunkTokenNum = remainingTokenNum < WAVE_COMBINE_META_INFO_TOKEN_CAPACITY ?
                                     remainingTokenNum :
                                     WAVE_COMBINE_META_INFO_TOKEN_CAPACITY;
        PreloadWaveCombineMetaInfo(
            params, scratch,
            static_cast<uint64_t>(state.globalTokenStartIndex) + currentCoreTokenRange.start + processedTokenNum,
            chunkTokenNum, 0U);
        CombineWaveTokenRange<CombineMode, UseAllAivCores>(
            common, bufferConfig, scratch, params, gmmAddrInfo.tensorAddrs.gmm2OutGlobal,
            currentCoreTokenRange.start + processedTokenNum, chunkTokenNum, 0U, rowSequence);
        processedTokenNum += chunkTokenNum;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_H
