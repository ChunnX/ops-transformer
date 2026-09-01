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
#include "../common/mega_moe_gmm_common.h"
#include "../common/mega_moe_utils.h"
#include "../common/mega_moe_mxfp8_utils.h"
#if __has_include("../../../common/mc2_kernel_utils.h")
#include "../../../common/mc2_kernel_utils.h"
#include "../../../common/quantize_functions.h"
#else
#include "../../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../../common/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

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
constexpr uint32_t WAVE_COMBINE_MAX_ROW_BUFFER_COUNT = 6U;
constexpr uint32_t WAVE_COMBINE_UB_BASE = 64U * 1024U;
// [64 KiB, 184 KiB) is dedicated to the Combine row ring and quant scratch.
// The ready scan starts at 184 KiB, so adaptive row buffers must stay below it.
constexpr uint32_t WAVE_COMBINE_UB_LIMIT = 184U * 1024U;
constexpr uint32_t WAVE_COMBINE_META_INFO_TOKEN_CAPACITY = 1536U;

// 非 layered 路径统一使用的 Combine 行环。常规 Wave 只有 AIV1 建立并使用这些 UB 视图；
// 最后一轮不再与 GMM1/SwiGLU 并行时，调用方才为 AIV0 补充初始化。
template <uint8_t CombineMode, bool IncludeAiv0 = false>
__aicore__ inline WaveCombineBufferConfig InitWaveCombineBuffers(const MoeStageCommonConfig &common,
                                                                 WaveCombineScratch &scratch)
{
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
    bufferConfig.rowBufferCount = WAVE_COMBINE_MAX_ROW_BUFFER_COUNT;
    if constexpr (CombineMode != COMBINE_NO_QUANT) {
        uint32_t quantTempBytes = bufferConfig.quantTempElements * sizeof(float);
        uint32_t rowRingBudgetBytes = WAVE_COMBINE_UB_LIMIT - WAVE_COMBINE_UB_BASE - quantTempBytes;
        bufferConfig.rowBufferCount = rowRingBudgetBytes / bufferConfig.slotStrideBytes;
        bufferConfig.rowBufferCount = bufferConfig.rowBufferCount < WAVE_COMBINE_MIN_ROW_BUFFER_COUNT ?
                                          WAVE_COMBINE_MIN_ROW_BUFFER_COUNT :
                                          bufferConfig.rowBufferCount;
        bufferConfig.rowBufferCount = bufferConfig.rowBufferCount > WAVE_COMBINE_MAX_ROW_BUFFER_COUNT ?
                                          WAVE_COMBINE_MAX_ROW_BUFFER_COUNT :
                                          bufferConfig.rowBufferCount;
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

// AIV1 复用既有配置；AIV0 仅在最终 Combine 到来时绑定同布局的私有 UB 视图。
template <uint8_t CombineMode>
__aicore__ inline WaveCombineBufferConfig PrepareFinalWaveCombineBuffers(
    const MoeStageCommonConfig &common, const WaveCombineBufferConfig &currentBufferConfig, WaveCombineScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return currentBufferConfig;
    }
    if (GetSubBlockIdx() == 0U) {
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

enum class WaveCombineAivSet : uint8_t {
    PAIRED_AIV1, // 稳态流水只使用每个 AIC 配对的 AIV1。
    ALL_AIV,     // 最后一轮无后续 Dispatch/Activation，可使用 AIV0 和 AIV1 收尾。
};

enum class Gmm2KernelKind : uint8_t {
    GENERIC, // A8W8 ND/NZ GMM2 scheduler。
    W4,      // A8W4/A4W4 共用的 W4 GMM2 scheduler。
};

// Combine 范围执行所需的只读公共资源。调用方只负责给出 [begin, end)；本结构不保存调度进度。
struct CombineRangeContext {
    const MoeStageCommonConfig &common;
    const GmmExecutionConfig &gmm;
    const AivJobContext &job;
    const BlockWorkspaceContext &countWorkspace;
    const MoeSyncWorkspaceLayout &syncLayout;
    const Params &params;
    const ExpertWeightTensorListAddrs &weights;
};

// 各物理核独立维护的 Combine 进度。pairwiseStartBlockIdx 用于重放与 AIC 相同的 GMM2 scheduler。
struct CombineRangeRuntime {
    uint32_t &pairwiseStartBlockIdx;
    int32_t &pairwiseTileSequence;
    uint32_t &rowSequence;
};

template <WaveCombineAivSet AivSet>
__aicore__ inline bool IsWaveCombineWorker()
{
    return AivSet == WaveCombineAivSet::ALL_AIV || GetSubBlockIdx() == 1U;
}

// 常规 Wave 只使用每个 AIC 配对的 AIV1；最后一轮把同组 AIV0/AIV1 展开成两个独立任务。
template <WaveCombineAivSet AivSet>
__aicore__ inline AivJobContext GetWaveCombineJob(const AivJobContext &job)
{
    if constexpr (AivSet == WaveCombineAivSet::ALL_AIV) {
        constexpr uint32_t AIV_CORE_COUNT_PER_AIC = 2U;
        uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
        return {job.jobIndex * AIV_CORE_COUNT_PER_AIC + subBlockIdx, job.totalJobs * AIV_CORE_COUNT_PER_AIC};
    }
    return job;
}

// 按专家累计 token 行数滚动物理 AIV 起点，每个 AIV 在当前专家内仍负责一个连续均衡区间。
template <WaveCombineAivSet AivSet = WaveCombineAivSet::PAIRED_AIV1>
__aicore__ inline WorkRange GetWaveCombineOwnedRange(const AivJobContext &job, uint32_t tokenCount,
                                                     uint64_t expertRowPrefix)
{
    if constexpr (g_coreType == AIC) {
        return {};
    }
    if (!IsWaveCombineWorker<AivSet>()) {
        return {};
    }
    AivJobContext combineJob = GetWaveCombineJob<AivSet>(job);
    if (combineJob.totalJobs == 0U || combineJob.jobIndex >= combineJob.totalJobs) {
        return {};
    }
    return GetRotatedBalancedTokenRange(tokenCount, combineJob.jobIndex, combineJob.totalJobs, expertRowPrefix);
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
template <uint8_t CombineMode, WaveCombineAivSet AivSet = WaveCombineAivSet::PAIRED_AIV1>
__aicore__ inline void CombineWaveTokenRange(const MoeStageCommonConfig &common,
                                             const WaveCombineBufferConfig &bufferConfig, WaveCombineScratch &scratch,
                                             const Params &params, GM_ADDR gmm2OutGlobal, uint32_t tokenStart,
                                             uint32_t tokenCount, uint32_t metaInfoUbTokenOffset, uint32_t &rowSequence)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!IsWaveCombineWorker<AivSet>() || tokenCount == 0U) {
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
template <bool IsShared, typename Config>
__aicore__ inline void WaitForGmm2InputReady(const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t mLoc)
{
    if constexpr (IsShared) {
        return;
    }
    if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
        uint32_t waveIdx = mLoc / L1_TILE_M_256;
        uint32_t waveStart = waveIdx * L1_TILE_M_256;
        uint32_t waveM = waveStart + L1_TILE_M_256 > config.m ? config.m - waveStart : L1_TILE_M_256;
        constexpr uint32_t sourceNFactor = Config::SOURCE_GMM1_INTERLEAVED ? ACTIVATION_N_HALF : 1U;
        uint32_t targetLoops =
            Ops::Base::CeilDiv(waveM, config.activationTileM) * Ops::Base::CeilDiv(config.k * sourceNFactor, L1_TILE_N);
        uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
        __gm__ int32_t *flagValueAddr = gmmAddrInfo.activationToGmm2Flag + flagOffset;
        WaitUntilGmFlagEquals(flagValueAddr, static_cast<int32_t>(targetLoops));
    } else {
        GmmKernel::BlockScheduler gmmBlockScheduler(
            {config.m, config.k, config.n},
            GmmKernel::BlockScheduler::Params{
                Te::MakeCoord(static_cast<int64_t>(config.activationTileM), static_cast<int64_t>(L1_TILE_N))});
        uint32_t targetLoops = gmmBlockScheduler.GetTileNum();
        WaitUntilGmFlagEquals(gmmAddrInfo.activationToGmm2Flag, static_cast<int32_t>(targetLoops));
    }
}

// Notifies all AIV consumers assigned to the completed GMM2 token group.
__aicore__ inline void NotifyCombineConsumersOfTileCompletion(uint32_t rowTileOffset,
                                                              const GroupSyncSlotLayout &slotLayout,
                                                              __gm__ int32_t *expertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    uint32_t firstSyncSlot = 0;
    uint32_t syncSlotCount = 0;
    GetGroupSyncSlotRange(tokenGroupIndex, slotLayout, firstSyncSlot, syncSlotCount);
    for (uint32_t syncSlot = firstSyncSlot; syncSlot < firstSyncSlot + syncSlotCount; ++syncSlot) {
        AscendC::AtomicAdd(GetCombineSyncCounterAddress(expertCounterBase, syncSlot), int32_t(1));
    }
}

/*
 * 每完成一个 GMM2 N tile，就向对应 expert-local 256-token group 的 Combine counter 发布一次通知。
 * FIX_S 保证通知发布前，该 tile 的 GMM2 输出已经写入 GM。
 */
__aicore__ inline void NotifyWaveCombineGroupCompletion(uint32_t rowTileOffset, __gm__ int32_t *expertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    AscendC::AtomicAdd(GetCombineSyncCounterAddress(expertCounterBase, tokenGroupIndex), int32_t(1));
}

// Notifies the AIV consumer assigned to the completed shared-expert token group.
__aicore__ inline void NotifySharedExpertTileCompletion(uint32_t rowTileOffset, __gm__ int32_t *sharedExpertCounterBase)
{
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);

    uint32_t tokenGroupIndex = rowTileOffset / COMBINE_TOKEN_GROUP_SIZE;
    AscendC::AtomicAdd(GetCombineSyncCounterAddress(sharedExpertCounterBase, tokenGroupIndex), int32_t(1));
}

namespace GmmKernel {

// 执行通用 GMM2 tile 循环。
template <uint8_t CombineQuantMode, typename BlockMmad, bool IsShared, bool IsLayered = false,
          Gmm2CombineSyncProtocol SyncProtocol = Gmm2CombineSyncProtocol::NONE, typename WorkSet, typename Config>
__aicore__ inline void Gmm2AicMmadGeneric(BlockMmad &blockMmad, WorkSet &workSet, const GMMAddrInfo &gmmAddrInfo,
                                          const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                          uint32_t rowOffsetInExpert = 0U, int32_t *gmTileSequence = nullptr)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr (IsLayered && !IsShared) {
        uint32_t logicalCoreCount = config.blockNum;
        if constexpr (!std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
            logicalCoreCount *= 2U;
        }
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        uint32_t kLoc = Get<K_VALUE>(blockCoord);

        // Slice only builds tensor views. Keep it ahead of the synchronization waits so the
        // current tile's address calculation is not inserted into the compute critical path.
        auto gmBlockA = workSet.gmA.Slice(Te::MakeCoord(mLoc, kLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<K_VALUE>(actualShape)));
        /* E8M0 scales are padded to an even count because GM->L1 moves 64-K scale pairs as b16. */
        auto gmBlockScaleA =
            workSet.gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        if constexpr (std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
        }

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            workSet.gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto gmBlockC = workSet.gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                          Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
        if constexpr (SyncProtocol == Gmm2CombineSyncProtocol::PAIRED_TILE_SEQUENCE && !IsShared) {
            // GMM2 tile 与配对 AIV1 的 Combine tile 一对一通知。
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++(*gmTileSequence));
        } else if constexpr (SyncProtocol == Gmm2CombineSyncProtocol::TOKEN_GROUP_COUNTER && !IsShared) {
            NotifyWaveCombineGroupCompletion(rowOffsetInExpert + mLoc, gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsLayered && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout, gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared && !IsLayered) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
    }
}

// 非量化 Combine 的 1:1 tile consumer。它与配对 AIC 重放完全相同的 scheduler，
// 每次只在本核序号到达后读取一个 [M tile, 256-N tile]，随即把该 tile 分段发送到目标 rank。
// UB 从 64 KiB 起步，避开 AIV1 Dispatch 跨 wave 保留在低区的 cumsum 状态。
template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename Config>
__aicore__ inline void Gmm2Aiv1EpilogueGeneric(Scheduler &scheduler, TensorC &l0cOutGm, const Params &params,
                                               const GMMAddrInfo &gmmAddrInfo, int32_t &gmTileSequence,
                                               const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        int32_t expectedReadySequence = gmTileSequence + 1;
        WaitUntilGmFlagAtLeast(gmmAddrInfo.gmmToEpilogueFlag, expectedReadySequence);
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        constexpr uint32_t tileUbBase = WAVE_COMBINE_UB_BASE;
        auto layoutTileUb = MakeLayoutC{}(config.tileM, L1_TILE_N);
        auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(tileUbBase), layoutTileUb);
        LocalTensor<ElementC> tileUb = LocalTensor<ElementC>(TPosition::VECIN, tileUbBase, config.tileM * L1_TILE_N);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUb, tensorBlockGm);

        int32_t lenTile = Get<M_VALUE>(actualShape);
        GlobalTensor<int32_t> metaInfoGm;
        LocalTensor<int32_t> metaInfoTensor =
            LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR, lenTile * META_INFO_SIZE);
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            gmmAddrInfo.metaInfoGlobal + static_cast<uint64_t>(mLoc) * META_INFO_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, lenTile * META_INFO_SIZE);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        CombineImpl::CombineTokens<ElementC, decltype(actualShape)>(nLoc, config.n, metaInfoTensor, tileUb, actualShape,
                                                                    L1_TILE_N, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        gmTileSequence = expectedReadySequence;
    }
}

// 执行 A8W4 GMM2 tile 循环。
template <bool IsShared, bool IsLayered, Gmm2CombineSyncProtocol SyncProtocol, typename BlockMmad, typename Scheduler,
          typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC, typename Config>
__aicore__ inline void Gmm2AicMmadA8W4(BlockMmad &blockMmad, Scheduler &scheduler, TensorA &gmA, TensorScaleA &gmScaleA,
                                       TensorScaleB &gmScaleB, TensorC &l0cOutGm, const GMMAddrInfo &gmmAddrInfo,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                       uint32_t expertTokenCount, uint32_t rowOffsetInExpert,
                                       int32_t *gmTileSequence = nullptr)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr (IsLayered && !IsShared) {
        uint32_t logicalCoreCount = config.blockNum;
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(expertTokenCount, logicalCoreCount);
    }
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);

        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        if constexpr (Config::IS_WAVE_FLAG_GRAINED) {
            uint32_t waveIdx = mLoc / L1_TILE_M_256;
            if (waveIdx != lastWaveWaited) {
                WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady<IsShared>(gmmAddrInfo, config, mLoc);
        }

        auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
        if constexpr (SyncProtocol == Gmm2CombineSyncProtocol::PAIRED_TILE_SEQUENCE && !IsShared) {
            // AIV1 与本 AIC 重放同一 scheduler。FIX 写回完成后发布本核单调序号；
            // W4 prologue 仍由 AIV0 独立推进，不参与该一对一完成链路。
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++(*gmTileSequence));
        } else if constexpr (IsLayered && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(rowOffsetInExpert + mLoc, groupSyncSlotLayout,
                                                   gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
    }
}

// 为一个 GMM2 逻辑 block 的 AIV0 路径展开一段 A8W4 权重。
template <typename BlockPrologue, typename Scheduler, typename TensorB, typename Config>
__aicore__ inline void Gmm2Aiv0PrologueA8W4(BlockPrologue &blockPrologue, Scheduler &scheduler, TensorB &gmB,
                                            const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        auto mL1Size = Get<M_VALUE>(actualShape);
        auto nL1Size = Get<N_VALUE>(actualShape);
        blockPrologue(gmB, mL1Size, config.k, nL1Size, nLoc, config.n, config.blockMmadTiling.l1Params.kL1);
    }
}

// 根据 GM 地址建立执行资源，并执行通用 GMM2 阶段。
template <uint8_t CombineQuantMode, typename BlockMmad, typename ElementC, bool IsLayered = false,
          bool IsShared = false, Gmm2CombineSyncProtocol SyncProtocol = Gmm2CombineSyncProtocol::NONE,
          typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecGeneric(Scheduler &scheduler, const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                       uint32_t startLoopIdx, uint32_t tileNum,
                                       BlockMmadContext<BlockMmad> *blockMmadContext, bool allowWeightL2Bypass,
                                       uint32_t rowOffsetInExpert = 0U, const Params *params = nullptr,
                                       int32_t *gmTileSequence = nullptr)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    if constexpr (Config::IS_WAVE_FLAG_GRAINED && g_coreType == AscendC::AIC) {
        SetWaveWeightL2CacheHint<KernelConfig::IS_WEIGHT_NZ, KernelConfig>(config, allowWeightL2Bypass, gmB, gmScaleB);
    }
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(gmmAddrInfo.gmm2OutGlobal)), layouts.c);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        if (blockMmadContext != nullptr) {
            InitBlockMmad(*blockMmadContext, config);
            Gmm2AicMmadGeneric<CombineQuantMode, BlockMmad, IsShared, IsLayered, SyncProtocol>(
                blockMmadContext->blockMmad, workSet, gmmAddrInfo, config, startLoopIdx, tileNum, rowOffsetInExpert,
                gmTileSequence);
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            InitBlockMmad(localBlockMmadContext, config);
            Gmm2AicMmadGeneric<CombineQuantMode, BlockMmad, IsShared, IsLayered, SyncProtocol>(
                localBlockMmadContext.blockMmad, workSet, gmmAddrInfo, config, startLoopIdx, tileNum, rowOffsetInExpert,
                gmTileSequence);
        }
    }
}

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM2 阶段。
template <typename BlockMmad, typename BlockPrologue, typename ElementC, bool IsShared, bool IsLayered,
          Gmm2CombineSyncProtocol SyncProtocol, typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecA8W4(Scheduler &scheduler, const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                    uint32_t startLoopIdx, uint32_t tileNum, uint32_t expertTokenCount,
                                    uint32_t rowOffsetInExpert, const Params *params = nullptr,
                                    int32_t *gmTileSequence = nullptr)
{
    using KernelConfig = typename Config::KernelConfig;
    using ElementA = typename KernelConfig::ElementAType;
    using ElementB = typename KernelConfig::ElementBType;
    using ElementMxScaleA = typename KernelConfig::ElementMxScaleAType;
    using ElementMxScaleB = typename KernelConfig::ElementMxScaleBType;
    using BiasType = typename KernelConfig::BiasType;

    auto layouts = KernelConfig::BuildLayouts(config);
    auto gmC = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementC *>(gmmAddrInfo.gmm2OutGlobal)), layouts.c);
    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementA *>(gmmAddrInfo.aGlobal)), layouts.a);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementB *>(gmmAddrInfo.bGlobal)), layouts.b);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleA *>(gmmAddrInfo.aScaleGlobal)),
        layouts.scaleA);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ElementMxScaleB *>(gmmAddrInfo.bScaleGlobal)),
        layouts.scaleB);
    auto gmBias =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(0UL)), layouts.bias);

    using WorkSetType = GroupMatmulWorkSet<Scheduler, decltype(gmA), decltype(gmB), decltype(gmScaleA),
                                           decltype(gmScaleB), decltype(gmBias), decltype(gmC)>;
    WorkSetType workSet{scheduler, gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC};

    if constexpr (g_coreType == AscendC::AIC) {
        BlockMmad blockMmad{};
        typename BlockMmad::BlockShape l0TileShape{config.blockMmadTiling.tileM, config.blockMmadTiling.tileN,
                                                   L0_TILE_K, 0};
        typename BlockMmad::ProblemShape matmulShape{config.m, config.outputN, config.k, 0};
        blockMmad.Init(matmulShape, l0TileShape, config.blockMmadTiling.l1Params);
        Gmm2AicMmadA8W4<IsShared, IsLayered, SyncProtocol, BlockMmad, decltype(workSet.scheduler),
                        decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                        std::remove_reference_t<decltype(workSet.gmC)>, Config>(
            blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, gmmAddrInfo,
            config, startLoopIdx, tileNum, expertTokenCount, rowOffsetInExpert, gmTileSequence);
    } else {
        if (GetSubBlockIdx() == 0U) {
            BlockPrologue blockPrologue;
            Gmm2Aiv0PrologueA8W4(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
        }
    }
}

} // namespace GmmKernel

// RunGmm2Generic：执行 Generic GMM2，支持量化和非量化 Combine 模式。
// =================================================================================================
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared = false, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false,
          Gmm2CombineSyncProtocol SyncProtocol = Gmm2CombineSyncProtocol::NONE>
__aicore__ inline void RunGmm2Generic(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                      const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                      const BlockJobContext &blockJob, void *blockMmadContext = nullptr,
                                      bool allowWeightL2Bypass = false, uint32_t rowOffsetInExpert = 0U,
                                      const Params *params = nullptr, int32_t *gmTileSequence = nullptr)
{
    static_assert(!IsShared || SyncProtocol == Gmm2CombineSyncProtocol::NONE,
                  "Shared expert GMM2 has its own completion counter");
    static_assert(!IsLayered || SyncProtocol == Gmm2CombineSyncProtocol::NONE,
                  "Layered GMM2 selects its synchronization through IsLayered");
    if constexpr (SyncProtocol == Gmm2CombineSyncProtocol::PAIRED_TILE_SEQUENCE && g_coreType == AscendC::AIV) {
        // AIV1 在独立 CombineTokenRange 中重放 scheduler；这里不能提前推进其 startBlockIdx。
        if (GetSubBlockIdx() == 1U) {
            return;
        }
    }
    using GmmConfig =
        GmmKernel::Config<false, CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                          IsWeightNZ, TopkWeightsPrefetch, IsShared, IsLayered, IsGmm1Interleaved, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if constexpr (CombineQuantMode == COMBINE_NO_QUANT && g_coreType == AscendC::AIV &&
                  SyncProtocol != Gmm2CombineSyncProtocol::PAIRED_TILE_SEQUENCE) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = GmmKernel::BlockMmadContext<BlockMmad>;
    auto *typedBlockMmadContext = reinterpret_cast<MmadContext *>(blockMmadContext);
    GmmKernel::Gmm2ExecGeneric<CombineQuantMode, BlockMmad, ElementC, IsLayered, IsShared, SyncProtocol>(
        scheduler, gmmAddrInfo, config, startLoopIdx, tileNum, typedBlockMmadContext, allowWeightL2Bypass,
        rowOffsetInExpert, params, gmTileSequence);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared = false, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false,
          Gmm2CombineSyncProtocol SyncProtocol = Gmm2CombineSyncProtocol::NONE>
__aicore__ inline void RunGmm2Generic(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                      const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                      void *blockMmadContext = nullptr, bool allowWeightL2Bypass = false,
                                      uint32_t rowOffsetInExpert = 0U, const Params *params = nullptr,
                                      int32_t *gmTileSequence = nullptr)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm2Generic<CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, IsWeightNZ,
                   IsLayered, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved, IsWaveFlagGrained,
                   SyncProtocol>(problemShape, gmmAddrInfo, startBlockIdx, blockJob, blockMmadContext,
                                 allowWeightL2Bypass, rowOffsetInExpert, params, gmTileSequence);
}

// RunGmm2A8W4：AIV0执行W4→W8 prologue，AIC执行GMM2；可选由配对AIV1逐tile Combine。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsLayered = false, bool IsWaveFlagGrained = false,
          Gmm2CombineSyncProtocol SyncProtocol = Gmm2CombineSyncProtocol::NONE>
__aicore__ inline void RunGmm2A8W4(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                   const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                   const BlockJobContext &blockJob, uint32_t expertTokenCount,
                                   uint32_t rowOffsetInExpert, const Params *params = nullptr,
                                   int32_t *gmTileSequence = nullptr)
{
    static_assert(!IsShared || SyncProtocol == Gmm2CombineSyncProtocol::NONE,
                  "Shared expert GMM2 has its own completion counter");
    static_assert(!IsLayered || SyncProtocol == Gmm2CombineSyncProtocol::NONE,
                  "Layered GMM2 selects its synchronization through IsLayered");
    if constexpr (SyncProtocol == Gmm2CombineSyncProtocol::PAIRED_TILE_SEQUENCE && g_coreType == AscendC::AIV) {
        // AIV1 在独立 CombineTokenRange 中重放 scheduler；AIV0 仍需执行 W4 prologue。
        if (GetSubBlockIdx() == 1U) {
            return;
        }
    }
    static_assert(std::is_same_v<ElementA, __fp8e4m3>, "Activation must be __fp8e4m3");
    static_assert(std::is_same_v<ElementB, __fp4e2m1x2>, "Weight must be __fp4e2m1x2");

    using GmmConfig = GmmKernel::Config<true, 0, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, false,
                                        TopkWeightsPrefetch, IsShared, IsLayered, false, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);

    using BlockMmad = typename GmmConfig::BlockMmad;
    using BlockPrologue = typename GmmConfig::BlockPrologue;
    GmmKernel::BlockScheduler scheduler({config.m, config.outputN, config.k},
                                        GmmKernel::BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx >= tileNum) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    GmmKernel::Gmm2ExecA8W4<BlockMmad, BlockPrologue, ElementC, IsShared, IsLayered, SyncProtocol,
                            GmmKernel::BlockScheduler, decltype(config)>(
        scheduler, gmmAddrInfo, config, startLoopIdx, tileNum, expertTokenCount, rowOffsetInExpert, params,
        gmTileSequence);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsLayered = false, bool IsWaveFlagGrained = false,
          Gmm2CombineSyncProtocol SyncProtocol = Gmm2CombineSyncProtocol::NONE>
__aicore__ inline void RunGmm2A8W4(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                   const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                   const Params *params = nullptr, int32_t *gmTileSequence = nullptr)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm2A8W4<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM, TopkWeightsPrefetch,
                 IsShared, IsLayered, IsWaveFlagGrained, SyncProtocol>(
        problemShape, gmmAddrInfo, startBlockIdx, blockJob, static_cast<uint32_t>(Get<M_VALUE>(problemShape)), 0U,
        params, gmTileSequence);
}

// 紧凑 token 资源按累计行偏移寻址，专家固定资源按 expertIdx 寻址。
template <typename WeightType, typename ActivationOutType, typename QuantScaleType>
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

    gmmAddrInfo.gmm2OutGlobal =
        workspace.gmm2MmadResPtr + globalTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
    gmmAddrInfo.metaInfoGlobal = workspace.metaInfoPtr + globalTokenStartIndex * META_INFO_SIZE * sizeof(int32_t);
    gmmAddrInfo.aGlobal = workspace.activationQuantDataPtr + globalTokenStartIndex * activationOutputWidth /
                                                                 activationOutputElementsPerByte *
                                                                 sizeof(ActivationOutType);
    gmmAddrInfo.aScaleGlobal =
        workspace.activationQuantScalePtr + globalTokenStartIndex * activationScaleWidth * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<WeightType>(
        weights.weight2, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
        static_cast<uint64_t>(state.expertIdx) * tokenHiddenDim * activationOutputWidth / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
        static_cast<uint64_t>(state.expertIdx) * tokenHiddenDim * activationScaleWidth);
    gmmAddrInfo.activationToGmm2Flag =
        reinterpret_cast<__gm__ int32_t *>(workspace.flagActivationToGmm2Ptr) +
        static_cast<uint64_t>(state.expertIdx) * syncLayout.activationFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag =
        reinterpret_cast<__gm__ int32_t *>(workspace.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(state.expertIdx) * syncLayout.dispatchFlagSlotCountPerExpert +
        static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
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

    gmmAddrInfo.gmm2OutGlobal =
        workspace.sharedExpertResultPtr + sharedExpertTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
    gmmAddrInfo.aGlobal =
        workspace.sharedExpertActivationDataPtr +
        sharedExpertTokenStartIndex * activationOutputWidth / activationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.sharedExpertActivationScalePtr +
                               sharedExpertTokenStartIndex * activationScaleWidth * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<WeightType>(
        weights.weight2, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx,
        static_cast<uint64_t>(sharedExpertIdx) * tokenHiddenDim * activationOutputWidth / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales2, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx,
        static_cast<uint64_t>(sharedExpertIdx) * tokenHiddenDim * activationScaleWidth);
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(commonConfig.tokenNum, Gmm1TileM);
    gmmAddrInfo.sharedExpertGmm2TileCounter =
        reinterpret_cast<__gm__ int32_t *>(workspace.sharedExpertGmm2TileCounterPtr) +
        static_cast<uint64_t>(sharedExpertIdx) * tokenGroupCount * INT_CACHELINE;
}

// 供普通模板、Wave 模板和共享专家的 GMM2 阶段复用。
// 使用调用方传入的 block 任务执行 GMM2，并保持所选同步约定不变。
template <uint8_t CombineMode, typename GenericElementA, typename A8W4ElementA, typename WeightType,
          typename QuantScaleType, bool EnableA8W4, bool EnableA4W4, uint32_t Gmm1TileM, bool TopkWeightsPrefetch,
          bool IsShared, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          Gmm2CombineSyncProtocol SyncProtocol = Gmm2CombineSyncProtocol::NONE>
__aicore__ inline void RunGmm2ByMode(const GmmExecutionConfig &gmmConfig, const GMMAddrInfo &gmmAddrInfo,
                                     const ProblemShape &problemShape, GmmRuntimeState &runtimeState,
                                     void *persistentBlockMmadContext = nullptr, bool allowWeightL2Bypass = false,
                                     uint32_t rowOffsetInExpert = 0U, const Params *params = nullptr,
                                     int32_t *gmTileSequence = nullptr)
{
    if constexpr (EnableA8W4 || EnableA4W4) {
        RunGmm2A8W4<A8W4ElementA, WeightType, bfloat16_t, QuantScaleType, QuantScaleType, Gmm1TileM,
                    TopkWeightsPrefetch, IsShared, false, IsWaveFlagGrained, SyncProtocol>(
            problemShape, gmmAddrInfo, runtimeState.startBlockIdx, gmmConfig.blockJob,
            static_cast<uint32_t>(Get<M_VALUE>(problemShape)), 0U, params, gmTileSequence);
    } else if (gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        RunGmm2Generic<CombineMode, GenericElementA, GenericElementA, bfloat16_t, QuantScaleType, QuantScaleType, true,
                       false, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved, IsWaveFlagGrained,
                       SyncProtocol>(
            problemShape, gmmAddrInfo, runtimeState.startBlockIdx, gmmConfig.blockJob, persistentBlockMmadContext,
            allowWeightL2Bypass, rowOffsetInExpert, params, gmTileSequence);
    } else {
        RunGmm2Generic<CombineMode, GenericElementA, GenericElementA, bfloat16_t, QuantScaleType, QuantScaleType, false,
                       false, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved, IsWaveFlagGrained,
                       SyncProtocol>(
            problemShape, gmmAddrInfo, runtimeState.startBlockIdx, gmmConfig.blockJob, persistentBlockMmadContext,
            allowWeightL2Bypass, rowOffsetInExpert, params, gmTileSequence);
    }
}

// AIV1 按与通用 GMM2 完全相同的 block scheduler 重放一个专家 slice，仅消费 AIC 已发布的 tile。
template <typename ElementA, typename QuantScaleType, bool IsWeightNZ, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch, bool IsGmm1Interleaved>
__aicore__ inline void ReplayGenericGmm2TilesForCombine(const ProblemShape &problemShape,
                                                        const BlockJobContext &blockJob,
                                                        const GMMAddrInfo &gmmAddrInfo, const Params &params,
                                                        uint32_t &startBlockIdx, int32_t &tileReadySequence)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }
    using GmmConfig = GmmKernel::Config<false, COMBINE_NO_QUANT, ElementA, ElementA, bfloat16_t, QuantScaleType,
                                        QuantScaleType, IsWeightNZ, TopkWeightsPrefetch, false, false,
                                        IsGmm1Interleaved, true>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);
    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx < tileNum) {
        using ProblemConfig = std::remove_reference_t<decltype(config)>;
        using KernelConfig = typename ProblemConfig::KernelConfig;
        auto layouts = KernelConfig::BuildLayouts(config);
        auto gmC = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ bfloat16_t *>(gmmAddrInfo.gmm2OutGlobal)),
            layouts.c);
        using MakeLayoutC = typename KernelConfig::MakeLayoutC;
        GmmKernel::Gmm2Aiv1EpilogueGeneric<bfloat16_t, MakeLayoutC>(
            scheduler, gmC, params, gmmAddrInfo, tileReadySequence, config, startLoopIdx, tileNum);
    }
    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

// AIV1 按与 W4 GMM2 完全相同的 block scheduler 重放一个专家 slice。
template <typename ElementA, typename WeightType, typename QuantScaleType, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void ReplayW4Gmm2TilesForCombine(const ProblemShape &problemShape,
                                                   const BlockJobContext &blockJob,
                                                   const GMMAddrInfo &gmmAddrInfo, const Params &params,
                                                   uint32_t &startBlockIdx, int32_t &tileReadySequence)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }
    using GmmConfig = GmmKernel::Config<true, 0, ElementA, WeightType, bfloat16_t, QuantScaleType, QuantScaleType,
                                        false, TopkWeightsPrefetch, false, false, false, true>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);
    GmmKernel::BlockScheduler scheduler({config.m, config.outputN, config.k},
                                        GmmKernel::BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();
    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    if (startLoopIdx < tileNum) {
        using ProblemConfig = std::remove_reference_t<decltype(config)>;
        using KernelConfig = typename ProblemConfig::KernelConfig;
        auto layouts = KernelConfig::BuildLayouts(config);
        auto gmC = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ bfloat16_t *>(gmmAddrInfo.gmm2OutGlobal)),
            layouts.c);
        using MakeLayoutC = typename KernelConfig::MakeLayoutC;
        GmmKernel::Gmm2Aiv1EpilogueGeneric<bfloat16_t, MakeLayoutC>(
            scheduler, gmC, params, gmmAddrInfo, tileReadySequence, config, startLoopIdx, tileNum);
    }
    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

// 根据模板的 GMM2 实现选择对应 scheduler；这里只做非量化 Combine 的 tile 重放。
template <typename ElementA, typename WeightType, typename QuantScaleType, Gmm2KernelKind KernelKind,
          uint32_t Gmm1TileM, bool TopkWeightsPrefetch,
          bool IsGmm1Interleaved>
__aicore__ inline void ReplayGmm2TilesForCombine(const GmmExecutionConfig &gmmConfig,
                                                 const ProblemShape &problemShape,
                                                 const GMMAddrInfo &gmmAddrInfo, const Params &params,
                                                 uint32_t &startBlockIdx, int32_t &tileReadySequence)
{
    if constexpr (KernelKind == Gmm2KernelKind::W4) {
        ReplayW4Gmm2TilesForCombine<ElementA, WeightType, QuantScaleType, Gmm1TileM, TopkWeightsPrefetch>(
            problemShape, gmmConfig.blockJob, gmmAddrInfo, params, startBlockIdx, tileReadySequence);
    } else if (gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        ReplayGenericGmm2TilesForCombine<ElementA, QuantScaleType, true, Gmm1TileM, TopkWeightsPrefetch,
                                         IsGmm1Interleaved>(
            problemShape, gmmConfig.blockJob, gmmAddrInfo, params, startBlockIdx, tileReadySequence);
    } else {
        ReplayGenericGmm2TilesForCombine<ElementA, QuantScaleType, false, Gmm1TileM, TopkWeightsPrefetch,
                                         IsGmm1Interleaved>(
            problemShape, gmmConfig.blockJob, gmmAddrInfo, params, startBlockIdx, tileReadySequence);
    }
}

// A8W8 量化 Wave 为当前专家绑定 256-token group 完成计数器；其他同步协议不需要该地址。
__aicore__ inline void BindMoeExpertGmm2GroupCounter(const MoeSyncWorkspaceLayout &syncLayout,
                                                     const WorkspaceInfo &workspace, uint32_t expertIdx,
                                                     GMMAddrInfo &gmmAddrInfo)
{
    uint64_t syncSlotOffset = static_cast<uint64_t>(expertIdx) * syncLayout.combineSyncSlotCountPerExpert;
    gmmAddrInfo.gmm2CombineSyncCounter = reinterpret_cast<__gm__ int32_t *>(workspace.gmm2CombineSyncCounterPtr) +
                                         syncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
}

// 将一个显式 token 范围展开为逐专家 GMM2 slice。非量化模式下 AIV1 不在这里推进 scheduler，
// 而是在 CombineTokenRange 中按相同范围重放；AIC/AIV0 仍按原顺序执行计算或 prologue。
template <typename ElementA, typename WeightType, typename QuantScaleType, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch, bool IsGmm1Interleaved = false>
__aicore__ inline void RunGmm2TokenRange(
    const MoeStageCommonConfig &common, const GmmExecutionConfig &gmmConfig,
    const BlockWorkspaceContext &countWorkspace, const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
    const ExpertWeightTensorListAddrs &weights, const ExpertTokenRange &range, ExpertLoopState &state,
    GMMAddrInfo &gmmAddrInfo,
    GmmRuntimeState &runtimeState, int32_t &tileReadySequence)
{
    if constexpr (g_coreType == AscendC::AIV) {
        if (GetSubBlockIdx() == 1U) {
            return;
        }
    }
    ExpertTokenPosition position = range.begin;
    const ExpertTokenPosition &rangeEnd = range.end;
    uint32_t gmm2TilesPerMGroup =
        Ops::Base::CeilDiv(common.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));
    while ((position.expertIdx != rangeEnd.expertIdx ||
            position.tokenIndexInExpert != rangeEnd.tokenIndexInExpert) &&
           position.expertIdx < common.moeExpertPerRank) {
        if (position.tokenIndexInExpert == 0U) {
            uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
                params.workspaceInfo.expertRevTokenNumsPtr, countWorkspace, common.moeExpertPerRank,
                position.expertIdx);
            UpdateExpertLoopState(state, position.expertIdx, expertTokenCount);
        }

        uint64_t expertRowCount = Get<M_VALUE>(state.problemShape);
        if (expertRowCount == 0U || position.tokenIndexInExpert >= expertRowCount) {
            ++position.expertIdx;
            position.tokenIndexInExpert = 0U;
            continue;
        }

        uint32_t sliceEndIndexInExpert = static_cast<uint32_t>(expertRowCount);
        if (position.expertIdx == rangeEnd.expertIdx) {
            sliceEndIndexInExpert = rangeEnd.tokenIndexInExpert;
        }
        uint32_t sliceTokenCount = sliceEndIndexInExpert - position.tokenIndexInExpert;
        uint32_t problemTileCount =
            GetMGroupCountForRows(sliceTokenCount, Gmm1TileM) * gmm2TilesPerMGroup;
        bool skipProblem = HandleWaveProblemWithoutWork(problemTileCount, gmmConfig.blockJob,
                                                        runtimeState.startBlockIdx);
        if (!skipProblem) {
            UpdateMoeExpertGmm2GlobalBuffer<WeightType, ElementA, QuantScaleType>(
                gmmConfig, syncLayout, params.workspaceInfo, weights, gmmAddrInfo, state,
                position.tokenIndexInExpert);
            gmmAddrInfo.gmmToEpilogueFlag =
                reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagGmmToEpiloguePtr) +
                static_cast<uint64_t>(gmmConfig.blockJob.jobIndex) * INT_CACHELINE;

            ProblemShape sliceProblemShape = state.problemShape;
            Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
            bool isWholeExpert =
                position.tokenIndexInExpert == 0U && static_cast<uint64_t>(sliceEndIndexInExpert) == expertRowCount;
            RunGmm2ByMode<COMBINE_NO_QUANT, ElementA, ElementA, WeightType, QuantScaleType, false, false, Gmm1TileM,
                          TopkWeightsPrefetch, false, IsGmm1Interleaved, true,
                          Gmm2CombineSyncProtocol::PAIRED_TILE_SEQUENCE>(
                gmmConfig, gmmAddrInfo, sliceProblemShape, runtimeState, nullptr, isWholeExpert,
                position.tokenIndexInExpert, &params, &tileReadySequence);
        }

        position.tokenIndexInExpert = sliceEndIndexInExpert;
        position.globalTokenIndex += sliceTokenCount;
        if (position.tokenIndexInExpert >= expertRowCount) {
            ++position.expertIdx;
            position.tokenIndexInExpert = 0U;
        }
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
template <WaveCombineAivSet AivSet = WaveCombineAivSet::PAIRED_AIV1>
__aicore__ inline void WaitWaveGmm2Ready(const AivJobContext &job, const Params &params, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (!IsWaveCombineWorker<AivSet>() || job.totalJobs == 0U) {
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

// 等待一个 expert-local 256-token group 的全部 GMM2 N tile 写入完成。
__aicore__ inline void WaitWaveCombineGroupReady(const MoeStageCommonConfig &common,
                                                 const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                                 uint32_t expertIdx, uint32_t tokenGroupIdx)
{
    uint64_t syncSlotIdx = static_cast<uint64_t>(expertIdx) * syncLayout.combineSyncSlotCountPerExpert +
                           static_cast<uint64_t>(tokenGroupIdx);
    __gm__ int32_t *counterAddr = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.gmm2CombineSyncCounterPtr) +
                                  syncSlotIdx * static_cast<uint64_t>(INT_CACHELINE);
    uint32_t gmm2NTileCount = Ops::Base::CeilDiv(common.tokenHiddenDim, L1_TILE_N);
    WaitUntilGmFlagEquals(counterAddr, static_cast<int32_t>(gmm2NTileCount));
}

// 消费本核在一个专家连续行段中的数据。A8W8 量化路径在读取每个 256-token group 前等待计数器；
// W4 量化路径已经在外层等待了整个专家，因此这里只执行行级 Combine。
template <uint8_t CombineMode, Gmm2CombineSyncProtocol SyncProtocol, WaveCombineAivSet AivSet>
__aicore__ inline void CombineReadyExpertRows(const MoeStageCommonConfig &common,
                                              const MoeSyncWorkspaceLayout &syncLayout,
                                              const WaveCombineBufferConfig &bufferConfig,
                                              WaveCombineScratch &scratch, const Params &params,
                                              uint32_t expertIdx, uint32_t expertLocalRowBegin,
                                              uint32_t globalRowBegin, uint32_t tokenCount,
                                              uint32_t &rowSequence)
{
    uint32_t processedTokenCount = 0U;
    while (processedTokenCount < tokenCount) {
        uint32_t currentExpertRow = expertLocalRowBegin + processedTokenCount;
        uint32_t tokenGroupIdx = currentExpertRow / COMBINE_TOKEN_GROUP_SIZE;
        uint32_t groupRowEnd = (tokenGroupIdx + 1U) * COMBINE_TOKEN_GROUP_SIZE;
        uint32_t rowsInCurrentGroup = groupRowEnd - currentExpertRow;
        uint32_t remainingTokenCount = tokenCount - processedTokenCount;
        if (rowsInCurrentGroup > remainingTokenCount) {
            rowsInCurrentGroup = remainingTokenCount;
        }
        if constexpr (SyncProtocol == Gmm2CombineSyncProtocol::TOKEN_GROUP_COUNTER) {
            WaitWaveCombineGroupReady(common, syncLayout, params, expertIdx, tokenGroupIdx);
        }

        for (uint32_t processedGroupRows = 0U; processedGroupRows < rowsInCurrentGroup;) {
            uint32_t remainingGroupRows = rowsInCurrentGroup - processedGroupRows;
            uint32_t chunkTokenCount = remainingGroupRows < WAVE_COMBINE_META_INFO_TOKEN_CAPACITY ?
                                           remainingGroupRows :
                                           WAVE_COMBINE_META_INFO_TOKEN_CAPACITY;
            uint32_t chunkGlobalRowBegin = globalRowBegin + processedTokenCount + processedGroupRows;
            PreloadWaveCombineMetaInfo(params, scratch, chunkGlobalRowBegin, chunkTokenCount, 0U);
            CombineWaveTokenRange<CombineMode, AivSet>(common, bufferConfig, scratch, params,
                                                       params.workspaceInfo.gmm2MmadResPtr, chunkGlobalRowBegin,
                                                       chunkTokenCount, 0U, rowSequence);
            processedGroupRows += chunkTokenCount;
        }
        processedTokenCount += rowsInCurrentGroup;
    }
}

// 非量化 Combine 按给定范围逐专家重放 GMM2 scheduler。范围边界决定地址和 M 大小，
// 函数不规划下一段、不执行 GMM2，也不负责最终 drain。
template <typename ElementA, typename WeightType, typename QuantScaleType, Gmm2KernelKind KernelKind,
          uint32_t Gmm1TileM, bool TopkWeightsPrefetch,
          bool IsGmm1Interleaved>
__aicore__ inline void CombinePairwiseTokenRange(const CombineRangeContext &context,
                                                 const ExpertTokenRange &range,
                                                 CombineRangeRuntime &runtime)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    ExpertTokenPosition position = range.begin;
    ExpertLoopState state = CreateExpertLoopState(context.common);
    state.expertIdx = range.begin.expertIdx;
    state.globalTokenStartIndex = static_cast<int64_t>(range.begin.globalTokenIndex) -
                                  static_cast<int64_t>(range.begin.tokenIndexInExpert);
    uint32_t gmm2TilesPerMGroup =
        Ops::Base::CeilDiv(context.common.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));
    while ((position.expertIdx != range.end.expertIdx ||
            position.tokenIndexInExpert != range.end.tokenIndexInExpert) &&
           position.expertIdx < context.common.moeExpertPerRank) {
        uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
            context.params.workspaceInfo.expertRevTokenNumsPtr, context.countWorkspace,
            context.common.moeExpertPerRank, position.expertIdx);
        UpdateExpertLoopState(state, position.expertIdx, expertTokenCount);
        if (expertTokenCount == 0U || position.tokenIndexInExpert >= expertTokenCount) {
            ++position.expertIdx;
            position.tokenIndexInExpert = 0U;
            continue;
        }

        uint32_t sliceEndIndexInExpert = expertTokenCount;
        if (position.expertIdx == range.end.expertIdx) {
            sliceEndIndexInExpert = range.end.tokenIndexInExpert;
        }
        uint32_t sliceTokenCount = sliceEndIndexInExpert - position.tokenIndexInExpert;
        uint32_t problemTileCount =
            GetMGroupCountForRows(sliceTokenCount, Gmm1TileM) * gmm2TilesPerMGroup;
        if (!HandleWaveProblemWithoutWork(problemTileCount, context.gmm.blockJob,
                                          runtime.pairwiseStartBlockIdx)) {
            GMMAddrInfo gmmAddrInfo{};
            UpdateMoeExpertGmm2GlobalBuffer<WeightType, ElementA, QuantScaleType>(
                context.gmm, context.syncLayout, context.params.workspaceInfo, context.weights,
                gmmAddrInfo, state, position.tokenIndexInExpert);
            gmmAddrInfo.gmmToEpilogueFlag =
                reinterpret_cast<__gm__ int32_t *>(context.params.workspaceInfo.flagGmmToEpiloguePtr) +
                static_cast<uint64_t>(context.gmm.blockJob.jobIndex) * INT_CACHELINE;
            ProblemShape sliceProblemShape = state.problemShape;
            Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
            ReplayGmm2TilesForCombine<ElementA, WeightType, QuantScaleType, KernelKind, Gmm1TileM,
                                      TopkWeightsPrefetch,
                                      IsGmm1Interleaved>(
                context.gmm, sliceProblemShape, gmmAddrInfo, context.params,
                runtime.pairwiseStartBlockIdx, runtime.pairwiseTileSequence);
        }

        position.tokenIndexInExpert = sliceEndIndexInExpert;
        position.globalTokenIndex += sliceTokenCount;
        if (position.tokenIndexInExpert >= expertTokenCount) {
            ++position.expertIdx;
            position.tokenIndexInExpert = 0U;
        }
    }
}

// 量化 Combine 按给定范围均衡到参与的 AIV，再按专家边界选择相应 ready 协议。
template <uint8_t CombineMode, Gmm2CombineSyncProtocol SyncProtocol,
          WaveCombineAivSet AivSet>
__aicore__ inline void CombineQuantTokenRange(const CombineRangeContext &context,
                                              const WaveCombineBufferConfig &bufferConfig,
                                              WaveCombineScratch &scratch, const ExpertTokenRange &range,
                                              uint32_t &rowSequence)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    uint32_t rangeRowCount = static_cast<uint32_t>(range.end.globalTokenIndex - range.begin.globalTokenIndex);
    WorkRange currentCoreTokenRange =
        GetWaveCombineOwnedRange<AivSet>(context.job, rangeRowCount, range.begin.globalTokenIndex);
    if (currentCoreTokenRange.count == 0U) {
        return;
    }

    uint32_t ownedWaveRowBegin = currentCoreTokenRange.start;
    uint32_t ownedWaveRowEnd = ownedWaveRowBegin + currentCoreTokenRange.count;
    uint32_t waveRowCursor = 0U;
    uint32_t expertIdx = range.begin.expertIdx;
    uint32_t expertLocalRowBegin = range.begin.tokenIndexInExpert;

    while (waveRowCursor < rangeRowCount && expertIdx < context.common.moeExpertPerRank) {
        uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
            context.params.workspaceInfo.expertRevTokenNumsPtr, context.countWorkspace,
            context.common.moeExpertPerRank, expertIdx);
        if (expertLocalRowBegin >= expertTokenCount) {
            ++expertIdx;
            expertLocalRowBegin = 0U;
            continue;
        }

        uint32_t expertRowsInWave = expertTokenCount - expertLocalRowBegin;
        uint32_t waveRowsRemaining = rangeRowCount - waveRowCursor;
        if (expertRowsInWave > waveRowsRemaining) {
            expertRowsInWave = waveRowsRemaining;
        }
        uint32_t expertWaveRowEnd = waveRowCursor + expertRowsInWave;
        uint32_t overlapWaveRowBegin = ownedWaveRowBegin > waveRowCursor ? ownedWaveRowBegin : waveRowCursor;
        uint32_t overlapWaveRowEnd = ownedWaveRowEnd < expertWaveRowEnd ? ownedWaveRowEnd : expertWaveRowEnd;
        if (overlapWaveRowBegin < overlapWaveRowEnd) {
            if constexpr (SyncProtocol == Gmm2CombineSyncProtocol::EXPERT_AIC_READY_FLAGS) {
                WaitWaveGmm2Ready<AivSet>(context.job, context.params, expertIdx);
            }
            uint32_t overlapExpertRowBegin = expertLocalRowBegin + overlapWaveRowBegin - waveRowCursor;
            uint32_t overlapGlobalRowBegin =
                static_cast<uint32_t>(range.begin.globalTokenIndex) + overlapWaveRowBegin;
            CombineReadyExpertRows<CombineMode, SyncProtocol, AivSet>(
                context.common, context.syncLayout, bufferConfig, scratch, context.params, expertIdx,
                overlapExpertRowBegin, overlapGlobalRowBegin, overlapWaveRowEnd - overlapWaveRowBegin,
                rowSequence);
        }

        waveRowCursor = expertWaveRowEnd;
        ++expertIdx;
        expertLocalRowBegin = 0U;
    }
}

/*
 * Combine 的唯一公开入口。调用方给出专家紧凑序列中的 [begin, end)，本函数只消费该范围：
 * - 非量化：AIV1 重放 GMM2 scheduler，逐 tile 等待并发送；
 * - A8W8 量化：逐 256-token group 等待 counter；
 * - W4 量化：逐专家等待全部 AIC ready flag。
 * buffer 初始化、最终 AIV 扩容和 row ring drain 均由调度层决定，不隐藏在本接口中。
 */
template <uint8_t CombineMode, typename ElementA, typename WeightType, typename QuantScaleType,
          Gmm2KernelKind KernelKind, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch, bool IsGmm1Interleaved = false,
          WaveCombineAivSet AivSet = WaveCombineAivSet::PAIRED_AIV1>
__aicore__ inline void CombineTokenRange(const CombineRangeContext &context,
                                         const WaveCombineBufferConfig &bufferConfig,
                                         WaveCombineScratch &scratch, const ExpertTokenRange &range,
                                         CombineRangeRuntime &runtime)
{
    if (range.end.globalTokenIndex <= range.begin.globalTokenIndex) {
        return;
    }
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        CombinePairwiseTokenRange<ElementA, WeightType, QuantScaleType, KernelKind, Gmm1TileM,
                                  TopkWeightsPrefetch,
                                  IsGmm1Interleaved>(context, range, runtime);
    } else if constexpr (KernelKind == Gmm2KernelKind::W4) {
        CombineQuantTokenRange<CombineMode, Gmm2CombineSyncProtocol::EXPERT_AIC_READY_FLAGS, AivSet>(
            context, bufferConfig, scratch, range, runtime.rowSequence);
    } else {
        CombineQuantTokenRange<CombineMode, Gmm2CombineSyncProtocol::TOKEN_GROUP_COUNTER, AivSet>(
            context, bufferConfig, scratch, range, runtime.rowSequence);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_H
