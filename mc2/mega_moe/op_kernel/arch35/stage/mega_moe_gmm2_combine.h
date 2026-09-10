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
#include "../../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../../common/op_kernel/quantize_functions.h"

namespace MegaMoeImpl {

using namespace AscendC;
#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;
#else
struct HcommBatchHandle {};
#endif

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
                                                const Params &params, GM_ADDR localSrcPtr,
                                                HcommBatchHandle &batchHandle)
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
        // 调用方已经按目标 rank 和固定归属核完成分组，此处直接向已绑定的批量句柄追加写 WQE。
        GM_ADDR remoteAddr = GetRankWinAddrWithOffset(toRankId, gmRemoteOffset) + dstBaseOffset * sizeof(DataType);
        // Combine 使用 Hcomm 默认 WQE 配置，不能复用 Dispatch 一级的无 CQE 配置。
        params.combineCommParams.hcomm->WriteNbi(batchHandle, remoteAddr, srcAddr, quantTokenSize * sizeof(DataType));
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

struct LayeredCombineBatchState {
    uint32_t dstRankId = 0U;
    HcommBatchHandle batchHandle{};
    uint32_t pendingTokenCount = 0U;
};

// 读取并按需量化一组普通/layered Combine token，然后发送到目标 rank。
template <uint8_t QuantMode, typename T, bool IsLayered = false, bool IsQuantized = true>
__aicore__ inline void CombineTokenGroup(uint32_t tokenStart, uint32_t tokenCount, uint32_t n, uint32_t groupIdx,
                                         uint32_t rankId, GM_ADDR gmm2OutAddr, const Params &params,
                                         LocalTensor<int32_t> &metaInfoTensor, int64_t ubTensorSize, int64_t offset,
                                         uint32_t quantTokenSizeBytes, LayeredCombineBatchState &batchState)
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
                                                            ubQuantSend, params, localSrcPtr, batchState.batchHandle);
#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
            if (batchState.dstRankId != rankId) {
                ++batchState.pendingTokenCount;
                if (batchState.pendingTokenCount == L1_TILE_M_256) {
                    params.combineCommParams.hcomm->BatchCommit(batchState.batchHandle);
                    batchState.pendingTokenCount = 0U;
                }
            }
#endif
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

/*
 * 非量化 Combine 的 GMM2 L0C->UB 直通：AIC 的 fixpipe 将 GMM2 tile 直接写入消费核 UB 的
 * tileUb 槽（C 张量落 UB 时 BlockMmad 自动走 L0C2UB，目的核为配对 V0），重放核跳过 GM
 * 读回，GM bf16 中继退役（本地 HBM 每 token 消 4H 往返）。就绪仍复用 gmmToEpilogueFlag
 * 序号；消费序号复用同一 AIC flag 槽（INT_CACHELINE 个 int32）的 word 1，随 flag 区
 * 每 launch 一次清零。GM 读回退役后 GM 写同步退役（双写不成立：消费读 UB 即需槽生命
 * 周期管理，GM 无界缓冲作用被架空）。
 *
 * 相位边界防护：直通 tile 槽 [64K,192K) 与 metaInfo 区 [200K,208K) 和 AIV0 的 GMM1
 * ping-pong 槽（[0,128K)/[128K,256K)）在 UB 上重叠，两个方向都需要显式排空——
 * word2（AIV0->AIC）：AIV0 排空本 wave epilogue 的 ping-pong 槽读后发布 wave 计数，
 * AIC 的首个直通 fixpipe 等它追平，防 GMM2 fixpipe 覆写进行中的 epilogue 读；
 * word1（AIV0->AIC）：AIC 离开直通阶段前等消费序号追平，防下一相位 GMM1 fixpipe
 * 覆写进行中的重放读。
 */
constexpr uint32_t GMM2_DIRECT_L0C_CONSUMED_SEQ_OFFSET = 1U;
constexpr uint32_t GMM2_DIRECT_L0C_EPILOGUE_DRAINED_SEQ_OFFSET = 2U;
// 直通要求 MMAD 每 tile 的生产时间稳定长于消费时间（深度 1 锁步零暴露）；K 低于该值时
// 生产不再慢于消费，回退 GM 中继避免拖累 GMM2。
constexpr uint32_t GMM2_DIRECT_L0C_MIN_K = 2048U;

// 量化直通（二期）在 AIV0 的槽外工作区：分组量化 band 与量化临时 scratch。
// [tile 槽末端 192K, ~202K) 不与 GMM2 tile 槽 [64K,192K) 冲突；落在 GMM1 ping-pong 槽
// [128K,256K) 内，由 word1（阶段出口排空）与 word2（阶段入口排空）相位护栏保护。
// 区间同时压过 metaInfo 区 [200K,248K) 的头部：AIV0 仅在量化直通下使用 band（此时
// CombineTokenRange/metaInfo 消费不在此核运行），两特性按 CombineQuantMode 互斥，改动
// 任一侧前须重审该互斥。
constexpr uint32_t GMM2_DIRECT_QUANT_BAND_ADDR =
    WAVE_COMBINE_UB_BASE + L1_TILE_M_256 * L1_TILE_N * static_cast<uint32_t>(sizeof(bfloat16_t));
constexpr uint32_t GMM2_DIRECT_QUANT_ROW_GROUP_COUNT = 32U;
// band 行距 288B = 256B fp8 data + 8B scale + 24B 垫整（QuantMxFp8 的输出布局，scale 段位于
// 行内 256B 偏移）：data 段暂存的 UB 源步进恰为 1 个 dataBlock(32B)；行尾 [256,288) 即本切片
// 32B 平面槽的有效内容（8B scale + 24B 残留，残留落自己槽内、散射侧紧凑时跳过），scale 暂存
// 直接以行尾 32B 为块源（源步进 8 个 dataBlock）——生产侧全程 V/MTE3，无标量中转。
constexpr uint32_t GMM2_DIRECT_QUANT_BAND_ROW_BYTES = 288U;
constexpr uint32_t GMM2_DIRECT_QUANT_BAND_BYTES = GMM2_DIRECT_QUANT_ROW_GROUP_COUNT * GMM2_DIRECT_QUANT_BAND_ROW_BYTES;
constexpr uint32_t GMM2_DIRECT_QUANT_TEMP_ADDR = GMM2_DIRECT_QUANT_BAND_ADDR + GMM2_DIRECT_QUANT_BAND_BYTES;
constexpr uint32_t GMM2_DIRECT_QUANT_TEMP_BYTES = 1024U;

// 编排层与 GmmKernel 层共用的直通激活判据（gmm2K 为 GMM2 的 K 维 = activation 宽 =
// gmm1OutputDim / ACTIVATION_N_HALF，与 BuildGmm2ProblemConfig 的 config.k 同源）。
// seqFlagPrecharged：存在 W4/A4W4 共享专家 GMM1 时，其完成序号由 AIV1 消费、word0 被
// 预充到 AIV0 本地计数之外（直通重放在 AIV0、序号域无法对齐），该混合配置回退 GM 中继。
// 非量化与两种 MXFP8 量化均支持直通：量化为槽内 tile 局部量化 + 双平面暂存（scale 按
// 切片独立落 32B 槽，规避 N 切片跨核散布下的凑组问题）。量化直通的定宽切片量化/落位与
// 紧凑公式（nScale = 8×nSlices）要求 tokenHiddenDim 为 L1_TILE_N 整数倍：末切片部分宽会
// 越界量化 UB 尾巴并越行写 data 平面，非整倍形状回退 GM 中继（非量化按 actualShape 处理
// 边界切片，无此约束）。主线 H（2048/4096/5120/7168）均满足，门控为防回归加固。
__aicore__ inline bool IsGmm2DirectL0cUbActive(int32_t combineQuantMode, uint32_t gmm2K, uint32_t tokenHiddenDim,
                                               bool seqFlagPrecharged)
{
    const bool tileConsumerMode = combineQuantMode == COMBINE_NO_QUANT || combineQuantMode == MXFP8_E4M3_COMM_QUANT ||
                                  combineQuantMode == MXFP8_E5M2_COMM_QUANT;
    const bool quantShapeAligned =
        combineQuantMode == COMBINE_NO_QUANT || tokenHiddenDim % static_cast<uint32_t>(L1_TILE_N) == 0U;
    return tileConsumerMode && quantShapeAligned && !seqFlagPrecharged && gmm2K >= GMM2_DIRECT_L0C_MIN_K;
}

// tile 序号通知（wave MoE 路径）且 K 足够大时启用 GMM2 L0C->UB 直通；AIC 与重放核用同一
// config 计算本判定，保证两侧路径选择一致。
template <uint8_t CombineQuantMode, bool NotifyCombineTileReady, typename Config>
__aicore__ inline bool UseGmm2DirectL0cUb(const Config &config, bool seqFlagPrecharged)
{
    if constexpr (NotifyCombineTileReady) {
        return IsGmm2DirectL0cUbActive(CombineQuantMode, config.k, config.n, seqFlagPrecharged);
    }
    return false;
}

// 每个 AIC 独占的 GMM2->Combine 序号 flag 槽（word0 就绪 / word1 消费 / word2 epilogue 排空）。
__aicore__ inline __gm__ int32_t *GetGmm2EpilogueFlagSlot(const Params &params, const BlockJobContext &blockJob)
{
    return reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagGmmToEpiloguePtr) +
           static_cast<uint64_t>(blockJob.jobIndex) * INT_CACHELINE;
}

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

// 按专家累计 token 行数滚动物理 AIV 起点，每个 AIV 在当前专家内仍负责一个连续均衡区间。
// 常规 Wave 只使用每个 AIC 配对的 AIV1；最终专家可把同组 AIV0/AIV1 展开成两个独立任务。
template <bool IncludeAiv0 = false>
__aicore__ inline WorkRange GetWaveCombineOwnedRange(const AivJobContext &job, uint32_t tokenCount,
                                                     uint64_t expertRowPrefix)
{
    if constexpr (g_coreType == AIC) {
        return {};
    }
    AivJobContext combineJob = job;
    if constexpr (IncludeAiv0) {
        constexpr uint32_t AIV_CORE_COUNT_PER_AIC = 2U;
        uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
        combineJob = {job.jobIndex * AIV_CORE_COUNT_PER_AIC + subBlockIdx, job.totalJobs * AIV_CORE_COUNT_PER_AIC};
    } else if (GetSubBlockIdx() != 1U) {
        return {};
    }
    if (combineJob.totalJobs == 0U || combineJob.jobIndex >= combineJob.totalJobs) {
        return {};
    }
    return GetRotatedBalancedWorkRange(tokenCount, combineJob.jobIndex, combineJob.totalJobs, expertRowPrefix);
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
template <uint8_t CombineMode>
__aicore__ inline void CombineWaveTokenRange(const MoeStageCommonConfig &common,
                                             const WaveCombineBufferConfig &bufferConfig, WaveCombineScratch &scratch,
                                             const Params &params, GM_ADDR gmm2OutGlobal, uint32_t tokenStart,
                                             uint32_t tokenCount, uint32_t metaInfoUbTokenOffset, uint32_t &rowSequence)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (tokenCount == 0U) {
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
template <typename Config>
__aicore__ inline void WaitForGmm2InputReady(const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t mLoc)
{
    if (gmmAddrInfo.activationToGmm2Flag == nullptr) {
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
          bool NotifyCombineTileReady = false, typename WorkSet, typename Config>
__aicore__ inline void Gmm2AicMmadGeneric(BlockMmad &blockMmad, WorkSet &workSet, const GMMAddrInfo &gmmAddrInfo,
                                          const Config &config, uint32_t startLoopIdx, uint32_t tileNum,
                                          uint32_t rowOffsetInExpert = 0U, int32_t *gmTileSequence = nullptr,
                                          bool seqFlagPrecharged = false)
{
    uint32_t lastWaveWaited = static_cast<uint32_t>(-1);
    GroupSyncSlotLayout groupSyncSlotLayout{};
    if constexpr ((CombineQuantMode != COMBINE_NO_QUANT || IsLayered) && !IsShared) {
        uint32_t logicalCoreCount = gmmAddrInfo.gmm2CombineLogicalCoreCount;
        if (logicalCoreCount == 0U) {
            logicalCoreCount = config.blockNum;
            if constexpr (!std::remove_reference_t<decltype(config)>::IS_WAVE_FLAG_GRAINED) {
                logicalCoreCount *= 2U;
            }
        }
        groupSyncSlotLayout = CalcGroupSyncSlotLayout(config.m, logicalCoreCount);
    }

    // L0C->UB 直通：C 张量改指向消费核 UB 的 tileUb 槽（与重放核 CombineTokenRange 的槽地址
    // 同约定），fixpipe 按 C 位置自动走 L0C2UB；GM 写同步退役。直通要求 gmmTileSequence 有效，
    // 由调用路径保证（NotifyCombineTileReady=true 仅出现在带序号的 wave 非量化 MoE 路径）。
    const bool directL0cUb = UseGmm2DirectL0cUb<CombineQuantMode, NotifyCombineTileReady>(config, seqFlagPrecharged);
    using DirectElementC = typename Config::KernelConfig::ElementCType;
    using DirectMakeLayoutC = typename Config::KernelConfig::MakeLayoutC;
    auto directUbCTensor =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, DirectElementC>(static_cast<int64_t>(WAVE_COMBINE_UB_BASE)),
                       DirectMakeLayoutC{}(config.tileM, L1_TILE_N));

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
                WaitForGmm2InputReady(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady(gmmAddrInfo, config, mLoc);
        }

        typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
                                                   Get<K_VALUE>(actualShape), 0};

        auto gmBlockB = workSet.gmB.Slice(Te::MakeCoord(kLoc, nLoc),
                                          Te::MakeShape(Get<K_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto gmBlockScaleB =
            workSet.gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        if (directL0cUb) {
            // 深度 1 背压：fixpipe 复写 UB 槽前，等重放核释放上一个 tile（MTE3 读离 UB 并发布
            // 消费序号）。主线规格下 MMAD 每 tile 比消费慢一个数量级以上，此处常态零等待。
            WaitUntilGmFlagAtLeast(gmmAddrInfo.gmmToEpilogueFlag + GMM2_DIRECT_L0C_CONSUMED_SEQ_OFFSET,
                                   *gmTileSequence);
            auto tensorBlockUb = directUbCTensor.Slice(
                Te::MakeCoord(0, 0), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, tensorBlockUb, singleShape);
        } else {
            auto gmBlockC = workSet.gmC.Slice(Te::MakeCoord(mLoc, nLoc),
                                              Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            blockMmad(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, workSet.gmBias, gmBlockC, singleShape);
        }
        if constexpr (NotifyCombineTileReady && !IsShared) {
            // GMM2 tile 与配对重放核的一对一通知。非量化恒发布（GM 回退重放也消费该序号）；
            // 量化仅直通时发布（GM 回退走专家粒度 gmm2Ready，无逐 tile 消费者），本地序号仍推进。
            if (directL0cUb || CombineQuantMode == COMBINE_NO_QUANT) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
                AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++(*gmTileSequence));
            } else {
                ++(*gmTileSequence);
            }
        } else if constexpr (IsLayered && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(mLoc, groupSyncSlotLayout, gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared && !IsLayered) {
            NotifySharedExpertTileCompletion(mLoc, gmmAddrInfo.sharedExpertGmm2TileCounter);
        }
    }
    if (directL0cUb) {
        // 相位边界排空：直通 tile 槽与 metaInfo 区同 AIV0 的 GMM1 ping-pong 槽重叠，AIC 离开
        // 本阶段（下一相位 GMM1 fixpipe 将复写 ping-pong 槽）前，须等重放核读离最后一个 tile。
        WaitUntilGmFlagAtLeast(gmmAddrInfo.gmmToEpilogueFlag + GMM2_DIRECT_L0C_CONSUMED_SEQ_OFFSET, *gmTileSequence);
    }
}

// 非量化 Combine 的统一入口。重放核直接使用 GMM2 已构造的 scheduler，逐 tile 等待配对 AIC 并消费。
// 当前 GMM2 problem 的起始地址和 M 大小共同限定 token 范围，无需再次遍历专家或重建 scheduler。
// UB 从 64 KiB 起步，避开 AIV1 Dispatch 跨 wave 保留在低区的 cumsum 状态。
// useDirectUb=true 为 L0C->UB 直通模式：tile 已由 AIC fixpipe 直投本核 UB 的 tileUb 槽，
// 跳过 GM 读回（目的行号由路由任意散射，发送仍逐 token 切片，消费开销与 GM 路径一致、被
// GMM2 计算 bound 裕量吸收）；就绪仍等 gmmToEpilogueFlag 序号，消费完发布消费序号供 AIC
// 背压（深度 1 槽复用）。
template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename Config>
__aicore__ inline void CombineTokenRange(Scheduler &scheduler, TensorC &l0cOutGm, const Params &params,
                                         const GMMAddrInfo &gmmAddrInfo, int32_t &gmTileSequence, const Config &config,
                                         uint32_t startLoopIdx, uint32_t tileNum, bool useDirectUb = false)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        int32_t expectedReadySequence = gmTileSequence + 1;
        WaitUntilGmFlagAtLeast(gmmAddrInfo.gmmToEpilogueFlag, expectedReadySequence);
        constexpr uint32_t tileUbBase = WAVE_COMBINE_UB_BASE;
        LocalTensor<ElementC> tileUb = LocalTensor<ElementC>(TPosition::VECIN, tileUbBase, config.tileM * L1_TILE_N);
        if (!useDirectUb) {
            auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                                Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
            auto layoutTileUb = MakeLayoutC{}(config.tileM, L1_TILE_N);
            auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(tileUbBase), layoutTileUb);
            auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
            AscendC::Te::Copy(copyGM2UB, tensorBlockUb, tensorBlockGm);
        }

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
        if (useDirectUb) {
            // 槽释放通知：MTE3 已读离本核 UB 槽（数据进入远端写通路），发布消费序号后 AIC 才能
            // 复写该槽。发布点与就绪序号同域（expectedReadySequence），AIC 侧按 tile 序号对齐。
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag + GMM2_DIRECT_L0C_CONSUMED_SEQ_OFFSET,
                                         expectedReadySequence);
        }
        gmTileSequence = expectedReadySequence;
    }
}

/*
 * 量化直通的槽内消费者（AIV0）：等 word0 后对 fixpipe 直投的 bf16 tile 做 tile 局部量化
 * （MXFP8 scale 为 32 元素组局部，与整行量化逐位等价），量化记录按双平面暂存 GM——data
 * 平面 [行 × H] 落切片 nLoc；scale 平面 [行 × nSlices×32] 按切片独立落 32B 槽。tile 的
 * N 切片按 block 跨核散布（loopIdx += blockNum），切片独立落位不依赖同核凑组；槽内 8B
 * 有效 + 24B 残留，记录所需的连续 scale 区由散射侧紧凑拼装（窗口记录布局不变，接收侧
 * unpermute 零改动）。暂存走默认 L2 路径（不加 bypass hint），散射侧重读命中 L2。暂存
 * MTE3 排空后发布 word1 释放槽位——散射与链路完全离槽，AIC 背压只覆盖量化+暂存（消费侧
 * 稳态速率含 V pass，k 门限实测后校准）。
 */
template <uint8_t CombineQuantMode, typename Scheduler, typename Config>
__aicore__ inline void Gmm2DirectQuantSlotStage(Scheduler &scheduler, const GMMAddrInfo &gmmAddrInfo,
                                                int32_t &gmTileSequence, const Config &config, uint32_t startLoopIdx,
                                                uint32_t tileNum, const Params &params)
{
    constexpr uint32_t rowGroupCount = GMM2_DIRECT_QUANT_ROW_GROUP_COUNT;
    const uint32_t hiddenDim = config.n;
    const uint32_t nSlices = Ops::Base::CeilDiv(hiddenDim, static_cast<uint32_t>(L1_TILE_N));
    const uint32_t scaleSlotRowBytes = nSlices * 32U;
    const uint64_t planeBase = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.workspaceInfo.gmm2MmadResPtr));
    const uint64_t expertRowBase =
        (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(gmmAddrInfo.gmm2OutGlobal)) - planeBase) /
        (2UL * static_cast<unsigned long>(hiddenDim));
    const uint64_t scalePlaneBase =
        planeBase + static_cast<uint64_t>(params.tilingData->maxOutputSize) * static_cast<uint64_t>(hiddenDim);

    LocalTensor<bfloat16_t> bandTensor(TPosition::VECIN, GMM2_DIRECT_QUANT_BAND_ADDR,
                                       GMM2_DIRECT_QUANT_BAND_BYTES / sizeof(bfloat16_t));
    LocalTensor<float> tempTensor(TPosition::VECCALC, GMM2_DIRECT_QUANT_TEMP_ADDR,
                                  GMM2_DIRECT_QUANT_TEMP_BYTES / sizeof(float));
    AscendC::GlobalTensor<uint8_t> dataPlaneGm;
    AscendC::GlobalTensor<uint8_t> scalePlaneGm;
    dataPlaneGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.workspaceInfo.gmm2MmadResPtr));
    scalePlaneGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(scalePlaneBase));

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);
        int32_t lenTile = Get<M_VALUE>(actualShape);

        int32_t expectedReadySequence = gmTileSequence + 1;
        WaitUntilGmFlagAtLeast(gmmAddrInfo.gmmToEpilogueFlag, expectedReadySequence);
        LocalTensor<bfloat16_t> tileUb(TPosition::VECIN, WAVE_COMBINE_UB_BASE, config.tileM * L1_TILE_N);
        const uint32_t sliceIdx = nLoc / L1_TILE_N;
        // 量化前关闭浮点溢出检查（60 = FLOAT_OVERFLOW_MODE_CTRL，GMM1 epilogue 量化同款）；
        // ProcessWave 已对全流程置 0，此处为局部自足的防御性重申。
        AscendC::SetCtrlSpr<60, 60>(0);
        for (int32_t groupStart = 0; groupStart < lenTile; groupStart += static_cast<int32_t>(rowGroupCount)) {
            const uint32_t rows = static_cast<uint32_t>(lenTile - groupStart) < rowGroupCount ?
                                      static_cast<uint32_t>(lenTile - groupStart) :
                                      rowGroupCount;
            for (uint32_t r = 0U; r < rows; ++r) {
                LocalTensor<bfloat16_t> inRow = tileUb[(static_cast<uint32_t>(groupStart) + r) * L1_TILE_N];
                LocalTensor<bfloat16_t> outRow =
                    bandTensor[r * (GMM2_DIRECT_QUANT_BAND_ROW_BYTES / sizeof(bfloat16_t))];
                Mxfp8::QuantMxFp8<CombineQuantMode, bfloat16_t>(outRow, inRow, tempTensor, L1_TILE_N);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            // data 平面：band 数据段（288B 行距）strided 暂存到全行长 H 的 fp8 行的 nLoc 切片。
            const uint64_t dataDstOffset =
                (expertRowBase + mLoc + static_cast<uint32_t>(groupStart)) * hiddenDim + nLoc;
            AscendC::DataCopyExtParams dataParams{static_cast<uint16_t>(rows), L1_TILE_N, 1U, hiddenDim - L1_TILE_N,
                                                  0U};
            AscendC::DataCopyPad(dataPlaneGm[dataDstOffset], bandTensor[0], dataParams);
            // scale：直接以 band 行尾 [256,288) 的 32B 为块源（8B 有效 + 24B 残留，落本切片自己
            // 的槽、散射侧紧凑时跳过残留），源步进 8 个 dataBlock——无标量中转与跨管线可见性问题。
            const uint64_t scaleDstOffset =
                (expertRowBase + mLoc + static_cast<uint32_t>(groupStart)) * scaleSlotRowBytes + sliceIdx * 32U;
            constexpr uint32_t bandScaleElemOffset = 256U / sizeof(bfloat16_t);
            constexpr uint32_t bandScaleSrcStrideBlocks = (GMM2_DIRECT_QUANT_BAND_ROW_BYTES - 32U) / 32U;
            AscendC::DataCopyExtParams scaleParams{static_cast<uint16_t>(rows), 32U, bandScaleSrcStrideBlocks,
                                                   scaleSlotRowBytes - 32U, 0U};
            AscendC::DataCopyPad(scalePlaneGm[scaleDstOffset], bandTensor[bandScaleElemOffset], scaleParams);
            // band/scale 暂存带复用：下一行组量化写 band 前等本组暂存读离。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        // 槽释放：暂存已 retire（默认 L2 路径），散射与链路离槽，AIC 背压到此处即止。
        AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag + GMM2_DIRECT_L0C_CONSUMED_SEQ_OFFSET,
                                     expectedReadySequence);
        gmTileSequence = expectedReadySequence;
    }
}

// 执行 A8W4 GMM2 tile 循环。
template <bool IsShared, bool IsLayered, bool NotifyCombineTileReady, typename BlockMmad, typename Scheduler,
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
        uint32_t logicalCoreCount = gmmAddrInfo.gmm2CombineLogicalCoreCount;
        if (logicalCoreCount == 0U) {
            logicalCoreCount = config.blockNum;
        }
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
                WaitForGmm2InputReady(gmmAddrInfo, config, mLoc);
                lastWaveWaited = waveIdx;
            }
        } else if (loopIdx == startLoopIdx) {
            WaitForGmm2InputReady(gmmAddrInfo, config, mLoc);
        }

        auto gmBlockA = gmA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.k));
        auto gmBlockScaleA =
            gmScaleA.Slice(Te::MakeCoord(mLoc, 0), Te::MakeShape(Get<M_VALUE>(actualShape), config.scaleK));

        auto gmBlockScaleB =
            gmScaleB.Slice(Te::MakeCoord(0, nLoc), Te::MakeShape(config.scaleK, Get<N_VALUE>(actualShape)));
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        blockMmad(gmBlockA, gmBlockScaleA, gmBlockScaleB, tensorBlockGm);
        if constexpr (NotifyCombineTileReady && !IsShared) {
            // AIV1 与本 AIC 重放同一 scheduler。FIX 写回完成后发布本核单调序号；
            // W4 prologue 仍由 AIV0 独立推进，不参与该一对一完成链路。
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WriteGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag, ++(*gmTileSequence));
        } else if constexpr (IsLayered && !IsShared) {
            NotifyCombineConsumersOfTileCompletion(rowOffsetInExpert + mLoc, groupSyncSlotLayout,
                                                   gmmAddrInfo.gmm2CombineSyncCounter);
        } else if constexpr (IsShared && !IsLayered) {
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
          bool IsShared = false, bool NotifyCombineTileReady = false, typename Scheduler, typename Config>
__aicore__ inline void Gmm2ExecGeneric(Scheduler &scheduler, const GMMAddrInfo &gmmAddrInfo, const Config &config,
                                       uint32_t startLoopIdx, uint32_t tileNum,
                                       BlockMmadContext<BlockMmad> *blockMmadContext, bool allowWeightL2Bypass,
                                       uint32_t rowOffsetInExpert = 0U, const Params *params = nullptr,
                                       int32_t *gmTileSequence = nullptr, bool seqFlagPrecharged = false)
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
            Gmm2AicMmadGeneric<CombineQuantMode, BlockMmad, IsShared, IsLayered, NotifyCombineTileReady>(
                blockMmadContext->blockMmad, workSet, gmmAddrInfo, config, startLoopIdx, tileNum, rowOffsetInExpert,
                gmTileSequence, seqFlagPrecharged);
        } else {
            BlockMmadContext<BlockMmad> localBlockMmadContext;
            InitBlockMmad(localBlockMmadContext, config);
            Gmm2AicMmadGeneric<CombineQuantMode, BlockMmad, IsShared, IsLayered, NotifyCombineTileReady>(
                localBlockMmadContext.blockMmad, workSet, gmmAddrInfo, config, startLoopIdx, tileNum, rowOffsetInExpert,
                gmTileSequence, seqFlagPrecharged);
        }
    } else if constexpr (NotifyCombineTileReady) {
        // 直通 tile 由 fixpipe 落配对 V0 的 UB（L0C2UB 默认目的核），消费者随之在 AIV0——
        // 非量化 GMM2 时段该核空闲，且 GM 序号协议不依赖消费核身份。两侧用同一 config 判定
        // 直通开关，AIC 与消费核的路径选择恒一致。
        const bool directL0cUb =
            UseGmm2DirectL0cUb<CombineQuantMode, NotifyCombineTileReady>(config, seqFlagPrecharged);
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            // GM 回退路径的重放核保持 AIV1（基线路径）。
            const uint32_t replaySubBlockIdx = directL0cUb ? 0U : 1U;
            if (GetSubBlockIdx() == replaySubBlockIdx) {
                using MakeLayoutC = typename KernelConfig::MakeLayoutC;
                CombineTokenRange<ElementC, MakeLayoutC>(workSet.scheduler, workSet.gmC, *params, gmmAddrInfo,
                                                         *gmTileSequence, config, startLoopIdx, tileNum, directL0cUb);
            }
        } else {
            // 量化直通：AIV0 槽内 tile 局部量化并暂存 GM 双平面（word1 到暂存为止）；散射由
            // AIV1 在专家粒度消费（gmm2Ready 门控，见 DirectQuantScatterExperts）。量化 GM
            // 回退无 tile 消费者（专家粒度 combine 走基线 ProcessCombineExperts）。
            if (directL0cUb && GetSubBlockIdx() == 0U) {
                Gmm2DirectQuantSlotStage<CombineQuantMode>(workSet.scheduler, gmmAddrInfo, *gmTileSequence, config,
                                                           startLoopIdx, tileNum, *params);
            }
        }
    }
}

// 根据 GM 地址建立执行资源，并执行 A8W4 GMM2 阶段。
template <typename BlockMmad, typename BlockPrologue, typename ElementC, bool IsShared, bool IsLayered,
          bool NotifyCombineTileReady, typename Scheduler, typename Config>
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
        Gmm2AicMmadA8W4<IsShared, IsLayered, NotifyCombineTileReady, BlockMmad, decltype(workSet.scheduler),
                        decltype(workSet.gmA), decltype(workSet.gmScaleA), decltype(workSet.gmScaleB),
                        std::remove_reference_t<decltype(workSet.gmC)>, Config>(
            blockMmad, workSet.scheduler, workSet.gmA, workSet.gmScaleA, workSet.gmScaleB, workSet.gmC, gmmAddrInfo,
            config, startLoopIdx, tileNum, expertTokenCount, rowOffsetInExpert, gmTileSequence);
    } else {
        if (GetSubBlockIdx() == 0U) {
            BlockPrologue blockPrologue;
            Gmm2Aiv0PrologueA8W4(blockPrologue, workSet.scheduler, workSet.gmB, config, startLoopIdx, tileNum);
        } else if constexpr (NotifyCombineTileReady) {
            using MakeLayoutC = typename KernelConfig::MakeLayoutC;
            CombineTokenRange<ElementC, MakeLayoutC>(workSet.scheduler, workSet.gmC, *params, gmmAddrInfo,
                                                     *gmTileSequence, config, startLoopIdx, tileNum);
        }
    }
}

} // namespace GmmKernel

// RunGmm2Generic：执行 Generic GMM2，支持量化和非量化 Combine 模式。
// =================================================================================================
template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared = false, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false>
__aicore__ inline void RunGmm2Generic(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                      const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                      const BlockJobContext &blockJob, void *blockMmadContext = nullptr,
                                      bool allowWeightL2Bypass = false, uint32_t rowOffsetInExpert = 0U,
                                      const Params *params = nullptr, int32_t *gmTileSequence = nullptr,
                                      bool seqFlagPrecharged = false)
{
    static_assert(!IsShared || !NotifyCombineTileReady, "Shared expert GMM2 has its own completion counter");
    static_assert(!IsLayered || !NotifyCombineTileReady, "Layered GMM2 selects its synchronization through IsLayered");
    using GmmConfig =
        GmmKernel::Config<false, CombineQuantMode, ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB,
                          IsWeightNZ, TopkWeightsPrefetch, IsShared, IsLayered, IsGmm1Interleaved, IsWaveFlagGrained>;
    auto config = GmmConfig::BuildGmm2ProblemConfig(problemShape, blockJob, Gmm1TileM);

    GmmKernel::BlockScheduler scheduler({config.m, config.schedulerN, config.k},
                                        GmmKernel::BlockScheduler::Params{Te::MakeCoord(
                                            static_cast<int64_t>(config.tileM), static_cast<int64_t>(L1_TILE_N))});
    uint32_t tileNum = scheduler.GetTileNum();

    if constexpr (CombineQuantMode == COMBINE_NO_QUANT && g_coreType == AscendC::AIV && !NotifyCombineTileReady) {
        startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
        return;
    }

    uint32_t startLoopIdx =
        (config.blockIdx < startBlockIdx ? config.blockIdx + config.blockNum : config.blockIdx) - startBlockIdx;
    using BlockMmad = typename GmmConfig::BlockMmad;
    using MmadContext = GmmKernel::BlockMmadContext<BlockMmad>;
    auto *typedBlockMmadContext = reinterpret_cast<MmadContext *>(blockMmadContext);
    GmmKernel::Gmm2ExecGeneric<CombineQuantMode, BlockMmad, ElementC, IsLayered, IsShared, NotifyCombineTileReady>(
        scheduler, gmmAddrInfo, config, startLoopIdx, tileNum, typedBlockMmadContext, allowWeightL2Bypass,
        rowOffsetInExpert, params, gmTileSequence, seqFlagPrecharged);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <uint8_t CombineQuantMode, typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA,
          typename ElementMxScaleB, bool IsWeightNZ = false, bool IsLayered = false, uint32_t Gmm1TileM = L1_TILE_M_256,
          bool TopkWeightsPrefetch = false, bool IsShared = false, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false>
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
                   NotifyCombineTileReady>(problemShape, gmmAddrInfo, startBlockIdx, blockJob, blockMmadContext,
                                           allowWeightL2Bypass, rowOffsetInExpert, params, gmTileSequence);
}

// RunGmm2A8W4：AIV0执行W4→W8 prologue，AIC执行GMM2；可选由配对AIV1逐tile Combine。
template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsLayered = false, bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false>
__aicore__ inline void RunGmm2A8W4(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                   const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                   const BlockJobContext &blockJob, uint32_t expertTokenCount,
                                   uint32_t rowOffsetInExpert, const Params *params = nullptr,
                                   int32_t *gmTileSequence = nullptr)
{
    static_assert(!IsShared || !NotifyCombineTileReady, "Shared expert GMM2 has its own completion counter");
    static_assert(!IsLayered || !NotifyCombineTileReady, "Layered GMM2 selects its synchronization through IsLayered");
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

    GmmKernel::Gmm2ExecA8W4<BlockMmad, BlockPrologue, ElementC, IsShared, IsLayered, NotifyCombineTileReady,
                            GmmKernel::BlockScheduler, decltype(config)>(scheduler, gmmAddrInfo, config, startLoopIdx,
                                                                         tileNum, expertTokenCount, rowOffsetInExpert,
                                                                         params, gmTileSequence);

    startBlockIdx = (startBlockIdx + tileNum) % config.blockNum;
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementMxScaleA, typename ElementMxScaleB,
          uint32_t Gmm1TileM = L1_TILE_M_256, bool TopkWeightsPrefetch = false, bool IsShared = false,
          bool IsLayered = false, bool IsWaveFlagGrained = false, bool NotifyCombineTileReady = false>
__aicore__ inline void RunGmm2A8W4(const AscendC::Shape<int64_t, int64_t, int64_t, int64_t> &problemShape,
                                   const GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx,
                                   const Params *params = nullptr, int32_t *gmTileSequence = nullptr)
{
    BlockJobContext blockJob{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation()),
                             static_cast<uint32_t>(GetBlockNum())};
    RunGmm2A8W4<ElementA, ElementB, ElementC, ElementMxScaleA, ElementMxScaleB, Gmm1TileM, TopkWeightsPrefetch,
                IsShared, IsLayered, IsWaveFlagGrained, NotifyCombineTileReady>(
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

    gmmAddrInfo.gmm2OutGlobal = workspace.gmm2MmadResPtr + globalTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
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

// Combine 只消费当前专家的 GMM2 输出；避免复用完整 GMM2 绑定接口时计算无关的输入、权重和 flag 地址。
__aicore__ inline void UpdateMoeExpertCombineGlobalBuffer(const WorkspaceInfo &workspace, GMMAddrInfo &gmmAddrInfo,
                                                          const ExpertLoopState &state)
{
    uint64_t tokenHiddenDim = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    uint64_t globalTokenStartIndex = static_cast<uint64_t>(state.globalTokenStartIndex);
    gmmAddrInfo.gmm2OutGlobal = workspace.gmm2MmadResPtr + globalTokenStartIndex * tokenHiddenDim * sizeof(bfloat16_t);
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
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    if (workspace.sharedActivationToGmm2Ptr != nullptr) {
        uint64_t sharedActivationFlagElementCount =
            static_cast<uint64_t>(CalcSharedActivationFlagElementsPerExpert(static_cast<int64_t>(tokenNum)));
        gmmAddrInfo.activationToGmm2Flag = reinterpret_cast<__gm__ int32_t *>(workspace.sharedActivationToGmm2Ptr) +
                                           static_cast<uint64_t>(sharedExpertIdx) * sharedActivationFlagElementCount;
    }
    gmmAddrInfo.sharedExpertGmm2TileCounter = nullptr;
    if (workspace.sharedExpertGmm2TileCounterPtr != nullptr) {
        uint32_t tokenGroupCount = Ops::Base::CeilDiv(commonConfig.tokenNum, Gmm1TileM);
        gmmAddrInfo.sharedExpertGmm2TileCounter =
            reinterpret_cast<__gm__ int32_t *>(workspace.sharedExpertGmm2TileCounterPtr) +
            static_cast<uint64_t>(sharedExpertIdx) * tokenGroupCount * INT_CACHELINE;
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
__aicore__ inline void WaitWaveGmm2Ready(const AivJobContext &job, const Params &params, uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (job.totalJobs == 0U) {
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

template <uint8_t CombineMode, bool IncludeAiv0 = false>
__aicore__ inline void RunWaveCombineStage(const MoeStageCommonConfig &common, const AivJobContext &job,
                                           const WaveCombineBufferConfig &bufferConfig, WaveCombineScratch &scratch,
                                           const Params &params, const GMMAddrInfo &gmmAddrInfo,
                                           const ExpertLoopState &state, uint32_t expertIdx, uint32_t &rowSequence)
{
    uint32_t currentExpertTokenNum = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    WorkRange currentCoreTokenRange = GetWaveCombineOwnedRange<IncludeAiv0>(
        job, currentExpertTokenNum, static_cast<uint64_t>(state.globalTokenStartIndex));
    if (currentCoreTokenRange.count != 0U) {
        WaitWaveGmm2Ready(job, params, expertIdx);
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
        CombineWaveTokenRange<CombineMode>(common, bufferConfig, scratch, params, gmmAddrInfo.gmm2OutGlobal,
                                           currentCoreTokenRange.start + processedTokenNum, chunkTokenNum, 0U,
                                           rowSequence);
        processedTokenNum += chunkTokenNum;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_H
