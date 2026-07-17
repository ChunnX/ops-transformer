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
 * \file mega_moe.h
 * \brief
 */

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"
#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

using namespace AscendC;

#ifndef MEGA_MOE_GMM2_GM_READY_SMOKE_TEST
#define MEGA_MOE_GMM2_GM_READY_SMOKE_TEST 0
#endif

#define MEGA_MOE_GMM2_READY_PER_EXPERT 0
#define MEGA_MOE_GMM2_READY_PER_BATCH 1
#ifndef MEGA_MOE_GMM2_READY_MODE
#define MEGA_MOE_GMM2_READY_MODE MEGA_MOE_GMM2_READY_PER_BATCH
#endif
#if MEGA_MOE_GMM2_READY_MODE != MEGA_MOE_GMM2_READY_PER_EXPERT && \
    MEGA_MOE_GMM2_READY_MODE != MEGA_MOE_GMM2_READY_PER_BATCH
#error "MEGA_MOE_GMM2_READY_MODE must be PER_EXPERT(0) or PER_BATCH(1)"
#endif

// Performance experiment: retain the final wave's full combine tail, but
// skip combine work in earlier PER_BATCH waves after preserving their GMM2
// ready dependency.  This is default-on so the requested A/B needs no extra
// compiler option.
#ifndef MEGA_MOE_GMM2_FINAL_BATCH_COMBINE_ONLY
#define MEGA_MOE_GMM2_FINAL_BATCH_COMBINE_ONLY 1
#endif

// Default-off NPU timeline diagnostic for the 16-local-expert target.  When
// enabled, expertTokenNumsOut is repurposed as sixteen uint32 words and must
// not be used for correctness or performance measurements.
#ifndef MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG
#define MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG 0
#endif

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t>;

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeTypeClass                                                                                       \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode,            \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegaMoeTypeFunc                                                                                        \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch,                 \
        IsGmm1Interleaved

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    template <int32_t QM>
    struct QuantTraits {
        using OutType = fp8_e4m3fn_t;
    };
    template <>
    struct QuantTraits<E5M2_QUANT> {
        using OutType = fp8_e5m2_t;
    };
    template <>
    struct QuantTraits<E2M1_QUANT> {
        using OutType = fp4x2_e2m1_t;
    };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType =
        typename std::conditional<Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    struct ExpertLoopState {
        TupleShape problemShape;
        BlockOffset baseOffset;
        // Rows before the current expert, kept per cursor for dispatch/GMM prefetch state split.
        uint32_t expertBeforeCnt = 0;
    };
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using DispatchBufferConfig = MegaMoeDispatchBufferConfig;

    __aicore__ inline DispatchBufferConfig DispatchBuffInit();
    __aicore__ inline SendMaskBufferConfig SendAndQuantBuffInit();
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void ResetGmm2CombineSyncCounters();
    __aicore__ inline void ResetGmm2NoQuantReadyCounters();
    __aicore__ inline void RunGmm2NoQuantReadySmokeTest();
    __aicore__ inline void RecordGmm2WaveTimeline(uint32_t slotIdx);
    __aicore__ inline void WriteGmm2WaveTimelineMetadata(uint32_t expertsPerWave);
    __aicore__ inline void SendMaskCal(const SendMaskBufferConfig &bufferConfig);
    __aicore__ inline void SendCntCal(int32_t localExpertId, uint64_t &sendCnt);
    __aicore__ inline void DispatchExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx,
                                          const DispatchBufferConfig &bufferConfig);
    __aicore__ inline uint64_t ReadExpertTokenCount(uint32_t expertIdx);
    __aicore__ inline void MetaInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId,
                                                  const DispatchBufferConfig &bufferConfig);
    template <AddrUpdateMode Mode>
    __aicore__ inline bool UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt = 0);
    __aicore__ inline bool UpdateSharedGroupParams(ExpertLoopState &state, uint32_t expertIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateSharedGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void Unpermute();
    __aicore__ inline UnpermuteBufferConfig UnpermuteBuffInit();
    __aicore__ inline void UnpermuteLoadWeights(int32_t coreOffset, int32_t batchTokenOffset, int32_t batchTokenCount,
                                                LocalTensor<bfloat16_t> &tempLocal);
    __aicore__ inline void UnpermuteProcessToken(int32_t tokenIdx, int32_t localIdx,
                                                 const GlobalTensor<bfloat16_t> &expandedX,
                                                 const UnpermuteBufferConfig &bufferConfig);
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void PublishGmm2NoQuantReady(uint32_t slotIdx);
    __aicore__ inline void WaitGmm2NoQuantReady(uint32_t slotIdx);
    __aicore__ inline uint64_t Gmm2NoQuantReadySlotStride() const;
    __aicore__ inline MegaMoeImpl::TokenRange GetCombineNoQuantOwnedRange(uint32_t tokenCount);
    __aicore__ inline TEventID CombineNoQuantRowEventId(uint32_t slot)
    {
        return static_cast<TEventID>(static_cast<int32_t>(EVENT_ID0) + static_cast<int32_t>(slot));
    }
    __aicore__ inline void InitCombineNoQuantRowRing();
    __aicore__ inline void DrainCombineNoQuantRowRing();
    __aicore__ inline void PreloadCombineNoQuantMetaInfo(uint64_t metaInfoGmTokenOffset, uint32_t tokenCount,
                                                        uint32_t metaInfoUbTokenOffset);
    __aicore__ inline void ProcessCombineNoQuantGm(GM_ADDR gmm2OutGlobal, uint32_t tokenStart, uint32_t tokenCount,
                                                   uint32_t metaInfoUbTokenOffset, uint32_t &rowSequence);
    __aicore__ inline void ProcessGmm2GmWaves(const DispatchBufferConfig &bufferConfig,
                                               int32_t &vecSetSyncCom);
    __aicore__ inline void ProcessCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
                                          uint32_t expertIdx);
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline auto DispatchCopyTmpTensor(int32_t bufferIdx) -> LocalTensor<ActivationType>;
    template <bool IsBufferReuse>
    __aicore__ inline void FetchTokenNLoadMetaInfo(int32_t bufferIdx, int32_t topkIndex, int32_t remoteRankIdx,
                                                   GlobalTensor<ActivationType> &remoteRankGlobalTensor,
                                                   uint32_t copyInNum);
    // 搬出一个 dispatch 槽：MTE3 写 token/scale/metaInfo 到 GM，并释放 buffer(MTE3_MTE2)与 metaInfo 槽(MTE3_S)
    __aicore__ inline void DispatchCopyMte3(int32_t bufferIdx, int32_t dstIdx,
                                            GlobalTensor<ActivationType> &tokenRevGlobalTensor,
                                            GlobalTensor<QuantScaleOutType> &scaleRevGlobalTensor,
                                            GlobalTensor<int32_t> &metaInfoGlobalTensor, int32_t copyStartIdx,
                                            int32_t copyIdx);
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx,
                                              int32_t copyNum, const DispatchBufferConfig &bufferConfig);
    __aicore__ inline void QuantProcessInRank();
    __aicore__ inline void SharedExpertCopyInput();
    __aicore__ inline void ProcessSharedExpertGmm1(const TupleShape &initShape, const BlockOffset &initOffset);
    __aicore__ inline void ProcessSharedExpertGmm2(const TupleShape &initShape, const BlockOffset &initOffset);
    __aicore__ inline void UnpermuteSharedExpert(int32_t tokenIdx, int32_t localIdx,
                                                 const UnpermuteBufferConfig &bufferConfig);
    __aicore__ inline void LoadTopkWeightsToUb(const LocalTensor<ActivationType> &xOutTensor, int32_t curentOffset,
                                               int32_t index, TEventID event);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithSwigluQuant(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                      uint32_t expertIdx, int32_t &vecSetSyncCom);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                  uint32_t expertIdx, int32_t &vecSetSyncCom);

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    Params params_{};

    GlobalTensor<int32_t> swigluToGmm2FlagGm_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> expertRevNumsGlobalTensor_;
    // A8W4 路径下 GroupMatmulSwigluQuant 会覆盖 V1 UB，导致 UB 上跨 expert 的状态
    // 无法保持。cumsumInfoGlobalTensor_ 作为 cumsum 数据的 GM 持久备份：
    // SendCntCal 中 Load(前序 expert 的前缀) → 计算 → Store；MetaInfoCalAndDispatch/ExpertTokenNumCopyOut 从 GM
    // 全量恢复。
    GlobalTensor<int32_t> cumsumInfoGlobalTensor_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t expertsPerBatch_ = 1;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint16_t gmm1PingPongIdx_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t blockNumPerRank_ = 2;
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t swigluFlagSlotsPerExpert_ = 0;
    int32_t maxWavesPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    uint32_t mxQuantScaleNumAlignPerToken_ = 0;
    uint32_t mxQuantTokenAlignBytes_ = 0;
    uint32_t mxQuantScaleAlignBytes_ = 0;
    uint32_t mxQuantTokenScaleAlignBytes_ = 0;
    uint32_t weightAlignBytes_ = 0;
    uint32_t ubBufferUsedAddr_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;   // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;  // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0; // quantTokenScalePtr 相对 win 基址的偏移
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）
    // 主线 shared-expert 特性成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;
    int64_t revTokenElemCnt_ = 0;
    int64_t revScaleElemCnt_ = 0;
    // 6-buffer 软流水(占满 EVENT_ID0..EVENT_ID5)的 UB 基址；槽视图由 base + bufferIdx*mxQuantTokenScaleAlignBytes_
    uint32_t copyTmpBaseAddr_ = 0;
    // ProcessCombine wave 流水参数：只依赖 k_(常量), InitCombineBuffers 算一次, 免每 expert 重算
    uint32_t combineNTilesPerGroup_ = 0;      // CeilDiv(k_, L1_TILE_N)
    uint32_t combineQuantTokenSizeBytes_ = 0; // CeilAlign(k_ + CeilDiv(k_,MXFP_SCALE_GROUP_NUM), ALIGN_32)

    // 大 BS route batch、ring buffer 和 reset batch 成员
    int32_t sendRouteItemsPerBatch_ = 0; // SendMaskCal 每个 batch 处理的 route item 数
    int32_t sendRouteBatchCount_ = 0;    // SendMaskCal 的 batch 总数
    int32_t recvRouteItemsPerBatch_ = 0; // MetaInfoCalAndDispatch 每个 batch 处理的 route item 数
    int32_t recvRouteBatchCount_ = 0;    // MetaInfoCalAndDispatch 的 batch 总数
    int32_t resetBatchElementCount_ = 0; // 每个 reset batch 清零的 int32 元素数（封顶到 DISPATCH_RESET_BATCH）

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    // ENABLE_A8W4: A8W8 路径（fp8 act + fp4 w1），GMM1 使用 A8W4 prologue（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4: A4W4 路径（fp4 act + fp4 weight），GMM2 复用 A8W4 prologue。
    //             a4w4 场景下 GMM1 走 generic a4w4、GMM2 走 a8w4，避免两段都用 a4w4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    // A8W4 has a separate GM-staged epilogue. Generic GMM1 can instead use
    // the prepared [gate, up] interleaved weight layout and two 128KB UB slots.
    static constexpr bool GMM1_INTERLEAVED = IsGmm1Interleaved && !ENABLE_A8W4;
    static constexpr bool GMM1_UB_PINGPONG = GMM1_INTERLEAVED;
    // The first GM rebuild is intentionally limited to the MTE class in this
    // file.  MegaMoeLayered has a different AIV1 sender schedule and keeps its
    // existing no-quant path until that schedule is redesigned independently.
    static constexpr bool ENABLE_GMM2_GM_REBUILD =
        !ENABLE_A8W4 && !ENABLE_A4W4 && CombineQuantMode == COMBINE_NO_QUANT;
    static constexpr uint32_t COMBINE_ROW_BUFFER_NUM = 6U;
    static constexpr uint32_t COMBINE_UB_USABLE_BYTES = 248U * 1024U;
    // GMM2-ready control scratch is live only after Dispatch(B+1) has drained
    // and before the six combine row buffers start. Keep it below the meta-info
    // arena and out of the row-buffer range, including the H=8K maximum.
    static constexpr uint32_t GMM2_READY_RELEASE_UB_ADDR = 160U * 1024U;
    static constexpr uint32_t GMM2_READY_SCAN_UB_ADDR = 184U * 1024U;
    static constexpr uint32_t GMM2_READY_MAX_SCAN_BYTES =
        MAX_AICORE_NUM * MEGA_MOE_GMM2_READY_CORE_STRIDE * sizeof(int32_t) > ALIGN_512 ?
            MAX_AICORE_NUM * MEGA_MOE_GMM2_READY_CORE_STRIDE * sizeof(int32_t) :
            ALIGN_512;
    static constexpr uint32_t GMM2_READY_PARTIAL_UB_ADDR =
        GMM2_READY_SCAN_UB_ADDR +
        ((GMM2_READY_MAX_SCAN_BYTES + ALIGN_512 - 1U) / ALIGN_512) * ALIGN_512;
    static constexpr uint32_t GMM2_READY_SUM_UB_ADDR = GMM2_READY_PARTIAL_UB_ADDR + ALIGN_512;
    static_assert(GMM2_READY_RELEASE_UB_ADDR + MAX_AICORE_NUM * ALIGN_512 <= GMM2_READY_SCAN_UB_ADDR,
                  "GMM2 ready release and scan scratch overlap");
    static_assert(GMM2_READY_SUM_UB_ADDR + ALIGN_512 <= META_INFO_TENSOR_ADDR,
                  "GMM2 ready scan scratch overlaps meta-info arena");
    static constexpr uint32_t COMBINE_META_INFO_TOKEN_CAPACITY =
        (COMBINE_UB_USABLE_BYTES - META_INFO_TENSOR_ADDR) / (META_INFO_SIZE * sizeof(int32_t));
    static_assert(COMBINE_META_INFO_TOKEN_CAPACITY > 0U, "combine meta-info UB must hold at least one token");
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<int32_t> sendCntTensor_;       // SendCntCal stride 只读 worldsize 个 count
    LocalTensor<uint8_t> maskBatchTensor_;     // MetaInfoCalAndDispatch 当前 batch 的 mask 切片
    LocalTensor<uint32_t> maskBatchU32Tensor_; // maskBatchTensor_ 的 u32 视图，供 GatherMask
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<int32_t> metaInfoTensor_;
    LocalTensor<bfloat16_t> xInTensor1_;
    LocalTensor<bfloat16_t> xInTensor2_;
    LocalTensor<ActivationType> xOutTensor1_;
    LocalTensor<ActivationType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<uint8_t> sendMaskTensor_;      // SendMaskCal 源卡生成并推送 [mask|count] 的动态 ring buffer
    LocalTensor<int32_t> sendGatherOutTensor_; // SendMaskCal GatherMask 计 count 的废弃输出 scratch
    LocalTensor<int32_t> sendCntAccTensor_;    // SendMaskCal per-expert 跨 batch count 累加器
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor_; // Unpermute bf16 weight 搬运中转

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，SwigluQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using SwigluQuantOutType = typename std::conditional<(QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    // SwigluQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    static constexpr uint32_t GMM1_TILE_M = MegaMoeImpl::L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M =
        TopkWeightsPrefetch ? MegaMoeImpl::L1_TILE_M_128 : MegaMoeImpl::L1_TILE_M_256;

    using BlockEpilogue =
        BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   EPILOGUE_TILE_M, MegaMoeImpl::L1_TILE_N, TopkWeightsPrefetch, GMM1_INTERLEAVED>;
    using SharedBlockEpilogue =
        BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   MegaMoeImpl::L1_TILE_M_256, MegaMoeImpl::L1_TILE_N, false, GMM1_INTERLEAVED>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
};

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    aicNum_ = tilingData->aicNum;
    topK_ = tilingData->topK;
    sendTotalNum_ = static_cast<uint64_t>(m_) * topK_;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    expertsPerBatch_ = tilingData->expertsPerBatch == 0U ? 1U : tilingData->expertsPerBatch;
    if (expertsPerBatch_ > moeExpertPerRank_) {
        expertsPerBatch_ = moeExpertPerRank_;
    }
    if (expertsPerBatch_ > MEGA_MOE_MAX_EXPERTS_PER_BATCH) {
        expertsPerBatch_ = MEGA_MOE_MAX_EXPERTS_PER_BATCH;
    }
    blockNumPerRank_ = tilingData->blockNumPerEP;
    maxOutputSize_ = tilingData->maxOutputSize;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(GMM1_TILE_M)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
    swigluFlagSlotsPerExpert_ = dispatchFlagSlotsPerExpert_;
    hiddenDim_ = tilingData->hiddenDim;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    params_.bGmAddr = GetTensorAddr(0, weight1);
    params_.b2GmAddr = GetTensorAddr(0, weight2);
    params_.bScaleGmAddr = GetTensorAddr(0, weightScales1);
    params_.b2ScaleGmAddr = GetTensorAddr(0, weightScales2);
    params_.sharedBGmAddr = GetTensorAddr(0, sharedWeight1);
    params_.sharedB2GmAddr = GetTensorAddr(0, sharedWeight2);
    params_.sharedBScaleGmAddr = GetTensorAddr(0, sharedWeightScales1);
    params_.sharedB2ScaleGmAddr = GetTensorAddr(0, sharedWeightScales2);

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t *)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGlobalTensor_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.expertRevTokenNumsPtr);
    // 每个 block 负责一个专家，cumsumInfo 中每个专家占 worldSize 个
    // int32_t 存 rank 维度的 cumsum 结果，blockIdx 决定了负责哪个专家。
    uint64_t cumsumStride =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), ALIGN_32);
    cumsumInfoGlobalTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr + cumsumStride * blockIdx_));
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
                      params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr,
                      params_.workspaceInfo.metaInfoPtr, tilingData->clampLimit});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    quantWinOffset_ =
        static_cast<uint64_t>(params_.peermemInfo.quantTokenScalePtr - params_.peermemInfo.rankSyncInWorldPtr);
    // maskAlignSize_ 必与 PeermemInfo 中 maskAlignSize 公式数值一致。
    compareCount_ =
        Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_256)) /
        sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_) / 8, static_cast<int64_t>(ALIGN_32));
    // 每个 win 槽位再追加 32B 存 count(源卡 SendMaskCal 同步算好), 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256)) *
        sizeof(ActivationType);
    mxQuantScaleAlignBytes_ = mxQuantScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxQuantTokenScaleAlignBytes_ =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    if constexpr (TopkWeightsPrefetch) {
        weightAlignBytes_ =
            Ops::Base::CeilAlign(static_cast<uint32_t>(topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        mxQuantTokenScaleAlignBytes_ += weightAlignBytes_;
    }
}

// =================================================================================================
// DispatchBuffInit：SendCntCal & MetaInfoCalAndDispatch & ExpertTokenNumCopyOut 中使用的 buffer 申请。
//   topkIndex/validTopkIndex 按 recvRouteItemsPerBatch_ 分配，metaInfoTensor_ 常驻 ring buffer。
// =================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::DispatchBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
{
    DispatchBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }

    revTokenElemCnt_ = k_ / A_ELEMS_PER_BYTE; // 输出 token 元素数
    revScaleElemCnt_ = Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                       MXFP_MULTI_BASE_SIZE; // 输出 token-scale 元素数，紧密排列

    // 与 route batch 无关的固定占用
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // sendCntTensor_: 每 src rank 一个 burst(32B), 共 worldsize*32B（stride 只读 count 跳过 mask 区）
    uint32_t sendCntTensorSize = worldSize_ * static_cast<uint32_t>(ALIGN_32);
    uint32_t expertTokenNumsOutTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // Dispatch 的 UB 布局与 AIV 分核无关；对应 host CalcDispatchBufferConfig 的唯一配置。
    bufferConfig = params_.tilingData->dispatchBufferConfig;
    recvRouteItemsPerBatch_ = bufferConfig.routeItemsPerBatch;
    recvRouteBatchCount_ = bufferConfig.routeBatchCount;

    // 按既定顺序落地址
    // Tensor用处：SendCntCal 函数中记录本卡各专家收到的 token 总数；
    // Tensor大小：仅记录 count 值且各专家之间复用，申请大小为 32 字节；
    uint32_t expertTokenCntTensorAddr = 0;
    expertTokenCntTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));
    // Tensor用处：SendCntCal 函数中记录本卡专家收到 token count 的 cumsum 累加值；
    // Tensor大小：worldSize_ * expertPerRank_ * sizeof(int32_t) align 至 32 字节对齐；
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    cumsumInfoTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // Tensor用处：SendCntCal 函数中 stride 只读 count 跳过 mask 区时，暂存各 src rank 的 count；
    // Tensor大小：每 src rank 一个 burst(32B)，共 worldsize*32B；
    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    sendCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr, sendCntTensorSize / sizeof(int32_t));
    // Tensor用处：MetaInfoCalAndDispatch 函数中接收当前 batch 的 mask 切片；
    // Tensor大小：recvRouteItemsPerBatch_ / 8 字节，每 bit 对应一个 route item；
    uint32_t maskBatchAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskBatchSize =
        static_cast<uint32_t>(recvRouteItemsPerBatch_ / 8) * static_cast<uint32_t>(sizeof(uint8_t));
    maskBatchTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint8_t));
    maskBatchU32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint32_t));
    // Tensor用处：MetaInfoCalAndDispatch 函数中 GatherMask 的 dst Tensor；
    // Tensor大小：recvRouteItemsPerBatch_ * sizeof(int32_t) align 至 32 字节对齐；
    uint32_t validTopkIndexTensorAddr = maskBatchAddr + maskBatchSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(recvRouteItemsPerBatch_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    // Tensor用处：MetaInfoCalAndDispatch 函数中 GatherMask 的 src Tensor（本 batch 的全局 index）；
    // Tensor大小：与 validTopkIndexTensor_ 一致，recvRouteItemsPerBatch_ * sizeof(int32_t) align 至 32 字节对齐；
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = validTopkIndexTensorSize;
    topkIndexTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIndexTensorSize / sizeof(int32_t));
    // route batch tensor 后依次放置 copyTmp ring、expert token count 输出和 32B metaInfo ring。
    // Tensor用处：MetaInfoCalAndDispatch 中的动态 dispatch ring，配合 EVENT_ID0..EVENT_ID(bufferCount-1) 做软流水；
    // 只记基址：槽视图在热路径由 DispatchCopyTmpTensor(base + bufferIdx*mxQuantTokenScaleAlignBytes_) 现场构造，
    // Tensor大小：bufferConfig.bufferCount 块(主线自适应 UB 预算给出的 2~6)，每块 mxQuantTokenScaleAlignBytes_；
    // 该值即 Init() 算好的 CeilAlign(token+scale, 32)，与 host CalcDispatchBufferConfig 的 copyBufferBytes 恒相等，
    // 且已向上对齐到 32B，故连续 ring 中每个 copyTmp 槽位的起始地址都保持 32B 对齐。
    copyTmpBaseAddr_ = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(bufferConfig.bufferCount) * mxQuantTokenScaleAlignBytes_;
    // Tensor用处：ExpertTokenNumCopyOut 函数中本卡各专家收到的 tokenCnt 数；
    // Tensor大小：expertPerRank_ * sizeof(int32_t) 对齐至 32 字节；
    uint32_t expertTokenNumsOutTensorAddr = copyTmpBaseAddr_ + copyTmpTotalSize;
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                                                     expertTokenNumsOutTensorSize / sizeof(int32_t));
    // Tensor用处：CopyGMToGMPerToken 函数中的 metaInfo ring buffer，逐 token 即时写 GM；
    // Tensor大小：bufferCount 条 * 32B，与 copyTmp 槽位和 event ID 一一对应。
    uint32_t metaInfoTensorAddr = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    uint32_t metaInfoReserveSize =
        static_cast<uint32_t>(bufferConfig.bufferCount) * static_cast<uint32_t>(INT32_PER_256B) * sizeof(int32_t);
    metaInfoTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr, metaInfoReserveSize / sizeof(int32_t));
    ubBufferUsedAddr_ = metaInfoTensorAddr + metaInfoReserveSize;
    return bufferConfig;
}

// ======================================================================================
// SendAndQuantBuffInit：SendMaskCal & ResetFlagList & QuantProcessInRank 中使用的 buffer 申请。
//   topkIds/sendMask/sendGatherOut 按 sendRouteItemsPerBatch_ 分配，reset 封顶 DISPATCH_RESET_BATCH。
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::SendMaskBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    SendMaskBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 =
        static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(swigluFlagSlotsPerExpert_) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_) +
         static_cast<uint64_t>(INT_CACHELINE) * static_cast<uint64_t>(aicNum_));
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint64_t combineSyncSlotElementCount = params_.tilingData->combineSyncSlotCountPerExpert * moeExpertPerRank_ *
                                               static_cast<uint64_t>(INT_CACHELINE);
        totalFlagInt32 = totalFlagInt32 > combineSyncSlotElementCount ? totalFlagInt32 : combineSyncSlotElementCount;
    }
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchElementCount_ = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                  static_cast<int32_t>(resetElementCountPerCore) :
                                  DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount_), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    uint32_t xOutTokenBytes =
        Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256));
    uint32_t xOutTensorSize = xOutTokenBytes + Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    if constexpr (TopkWeightsPrefetch) {
        xOutTensorSize = Ops::Base::CeilAlign(xOutTensorSize + weightAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * expertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。SendMaskCal 按
    // expertId = aivCoreIdx_ + ownedIdx * blockAivNum_ 遍历，因此前 remainder 个 core 多处理一个 expert。
    bufferConfig = aivCoreIdx_ < params_.tilingData->sendMaskCoreCountWithExtraExpert ?
                       params_.tilingData->sendMaskConfigForCoreWithExtraExpert :
                       params_.tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    sendRouteItemsPerBatch_ = bufferConfig.routeItemsPerBatch;
    sendRouteBatchCount_ = bufferConfig.routeBatchCount;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize =
        static_cast<uint32_t>(sendRouteItemsPerBatch_) * static_cast<uint32_t>(sizeof(int32_t));
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    mxTempTensor_ = LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    xOutTensor1_ =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    xOutTensor2_ =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));

    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    xInTensor1_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    xInTensor2_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    uint32_t sendGatherOutSize =
        static_cast<uint32_t>(sendRouteItemsPerBatch_) * static_cast<uint32_t>(sizeof(int32_t));

    uint32_t sendMaskTotalBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, sendMaskAddr, sendMaskTotalBytes);
    uint32_t sendGatherOutAddr = sendMaskAddr + sendMaskTotalBytes;
    sendGatherOutTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendCntAccTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
    return bufferConfig;
}

// ===============================================================================================
// ResetFlagList：对本卡workSpace上的Flag位分批清零（封顶到 DISPATCH_RESET_BATCH），
//   包括 flagSwiGluToGmm2Ptr & flagDispatchToGmm1Ptr & flagSendCntCalToUpdParamsPtr。
//   新 no-quant ready 区由 AIV1-only 的独立函数清零，避免 AIV0 进入 GM ready 路径。
// ===============================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetFlagList()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // workSpace Flag 清零
    // 总数 = SwiGluToGmm2(expertPerRank * swigluFlagSlotsPerExpert_)
    //        + DispatchToGmm1(expertPerRank * dispatchFlagSlotsPerExpert_)
    //        + SendCntCalToUpdParams(expertPerRank * aicNum_ * INT_CACHELINE)
    swigluToGmm2FlagGm_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.flagSwiGluToGmm2Ptr);
    // 只清理路由专家的 flag，共享专家的 flag 由 ProcessSharedExpert 自行管理
    int32_t flagNum = static_cast<int32_t>(moeExpertPerRank_) *
                      (swigluFlagSlotsPerExpert_ + dispatchFlagSlotsPerExpert_ +
                       static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(aicNum_));
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset, 1);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();

    for (int32_t resetElementOffset = 0; resetElementOffset < coreLen; resetElementOffset += resetBatchElementCount_) {
        int32_t currentBatchElementCount = coreLen - resetElementOffset < resetBatchElementCount_ ?
                                               coreLen - resetElementOffset :
                                               resetBatchElementCount_;
        DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(currentBatchElementCount * sizeof(int32_t)), 0U,
                                             0U, 0U};
        DataCopyPad(swigluToGmm2FlagGm_[coreOffset + resetElementOffset], resetTensor_, rankSyncCopyParams);
    }
    // combine量化模式下TokenGroupCompleteFlag清零
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        ResetGmm2CombineSyncCounters();
    }

    // prefetch 路径：清理 GMM1 tile 状态位区（含 allDone slot），避免上一轮残留导致软同步误判
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<int32_t> statusGm;
        statusGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr));
        int32_t statusSlots =
            static_cast<int32_t>(expertPerRank_) * static_cast<int32_t>(params_.tilingData->maxTilesPerExpert) + 1;
        int32_t statusCoreLen, statusCoreOffset;
        TilingByCore(statusSlots, statusCoreLen, statusCoreOffset, 1);
        for (int32_t resetElementOffset = 0; resetElementOffset < statusCoreLen;
             resetElementOffset += resetBatchElementCount_) {
            int32_t currentBatchElementCount = statusCoreLen - resetElementOffset < resetBatchElementCount_ ?
                                                   statusCoreLen - resetElementOffset :
                                                   resetBatchElementCount_;
            DataCopyExtParams statusCopyParams{1U, static_cast<uint32_t>(currentBatchElementCount * sizeof(int32_t)),
                                               0U, 0U, 0U};
            DataCopyPad(statusGm[statusCoreOffset + resetElementOffset], resetTensor_, statusCopyParams);
        }
    }
    if constexpr (ENABLE_GMM2_GM_REBUILD) {
        ResetGmm2NoQuantReadyCounters();
    }
}

// ==================================================
// ExpertTokenNumCopyOut：本卡各路由专家收到的token总数输出（不包含共享专家）
// ==================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，从 GM 恢复
    if constexpr (ENABLE_A8W4) {
        DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                    {1U, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                    {true, 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    expertTokenNumsOutTensor_.SetValue(0, cumsumInfoTensor_.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < static_cast<int32_t>(moeExpertPerRank_); expertIdx++) {
        int32_t cur = cumsumInfoTensor_.GetValue(expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx);
        int32_t prev = cumsumInfoTensor_.GetValue((expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx);
        expertTokenNumsOutTensor_.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut_, expertTokenNumsOutTensor_, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

// ======================================================================================================
// SendMaskCal：对本卡 topk 按通信域内所有专家id计算mask位，并发送至目标专家卡。
//
//   Phase 1: 本卡 topk 按 route batch 分批搬入；
//   Phase 2: 配合 topk batch 的 per-expert mask 生成+动态 ring buffer 推送。
//
// 流水（动态 2~6 buffer mask push）：
//   当前槽完成 mask 生成后即可发起 MTE3 push；Vector 继续使用后续槽生成其他 expert 的 mask，
//   直到 ring 回绕时才等待对应槽的 MTE3 完成。跨 batch 时，下一批 topk 的 MTE2 加载也可与上一批
//   尚未完成的 MTE3 push 重叠。
//   EVENT_ID0~EVENT_ID(bufferCount-1) 控制各槽 MTE3 write 完成事件，保证 ring 轮转使用不冲突。
//
// 关键细节：
//   - 非末 batch: pushBytes = sliceBytes（纯 mask 切片）
//   - 末 batch:   pushBytes = sliceBytes + 4B；末尾多写一个 int32 是该 expert 跨 batch 的累计 count
//                 （SendCntCal 通过 maskSlotSize 跳过 mask 区直接读 count，无需再翻 mask）
//   - sendCntAccTensor_[ownedIdx]: per-expert 跨 batch 累加计数，末 batch 折叠进 mask 尾部
//   - peer window 地址: maskWinOffset_ + expert*srcRank*(mask+count slot) + batchStart/8 偏移
// ======================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendMaskCal(const SendMaskBufferConfig &bufferConfig)
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // owned expert：本 AIV core 负责的 global expert 子集
    int32_t totalExperts = static_cast<int32_t>(worldSize_ * moeExpertPerRank_);
    int32_t coreIdx = static_cast<int32_t>(aivCoreIdx_);
    int32_t ownedExpertNum =
        (coreIdx < totalExperts) ? Ops::Base::CeilDiv(totalExperts - coreIdx, static_cast<int32_t>(blockAivNum_)) : 0;
    if (ownedExpertNum <= 0) {
        return;
    }

    // 准备 GM 读写句柄
    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t *)params_.expertIdxGmAddr);
    GlobalTensor<uint8_t> dstGlobalTensor;
    int32_t maskSliceBytesFull = sendRouteItemsPerBatch_ / 8;
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0};

    // per-expert 跨 batch 累加器清零
    Duplicate<int32_t>(sendCntAccTensor_, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    // mask push ring 事件：初始将实际分配的全部槽位交给 Vector。
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }

    int32_t iter = 0;
    // 外层：route batch 循环
    for (int32_t batchIdx = 0; batchIdx < sendRouteBatchCount_; ++batchIdx) {
        int32_t batchStart = batchIdx * sendRouteItemsPerBatch_;
        bool isLastBatch = (batchIdx == sendRouteBatchCount_ - 1);
        int32_t validLen = sendRouteItemsPerBatch_;
        int32_t sliceBytes = maskSliceBytesFull;
        int32_t pushBytes = sliceBytes;

        if (isLastBatch) {
            validLen = static_cast<int32_t>(sendTotalNum_ - static_cast<uint64_t>(batchStart));
            if (batchStart / 8 + sliceBytes > static_cast<int32_t>(maskAlignSize_)) {
                sliceBytes = static_cast<int32_t>(maskAlignSize_) - batchStart / 8;
            }
            pushBytes = sliceBytes + static_cast<int32_t>(sizeof(int32_t));
        }

        // 加载本 batch 的 topk
        SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
        DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(topkIdsTensor_, srcGlobalTensor[batchStart], loadParams, loadPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

        // 内层：per-expert 循环
        for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx, ++iter) {
            int32_t globalExpertId = coreIdx + ownedIdx * static_cast<int32_t>(blockAivNum_);
            int32_t dstRank = globalExpertId / static_cast<int32_t>(moeExpertPerRank_);
            int32_t localExpertId = globalExpertId % static_cast<int32_t>(moeExpertPerRank_);

            int32_t bufferIdx = iter % bufferConfig.bufferCount;
            TEventID bufEvent = static_cast<TEventID>(bufferIdx);
            LocalTensor<uint8_t> maskBuf = sendMaskTensor_[bufferIdx * bufferConfig.bufferBytes];
            LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();

            WaitFlag<AscendC::HardEvent::MTE3_V>(bufEvent);
            // DAV_3510 requires CompareScalar count * sizeof(int32_t) to be 256B-aligned, so the aligned batch
            // length is used instead of validLen. Both GatherMask consumers are bounded by validLen and ignore
            // the mask bits produced for the padded tail.
            CompareScalar(maskBuf, topkIdsTensor_, globalExpertId, AscendC::CMPMODE::EQ, sendRouteItemsPerBatch_);
            uint64_t batchMatchedRouteCount = 0;
            GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true, static_cast<uint32_t>(validLen),
                       {1, 1, 0, 0}, batchMatchedRouteCount);

            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
            int32_t expertMatchedRouteCount =
                sendCntAccTensor_.GetValue(ownedIdx) + static_cast<int32_t>(batchMatchedRouteCount);
            sendCntAccTensor_.SetValue(ownedIdx, expertMatchedRouteCount);
            if (isLastBatch) {
                maskBuf.template ReinterpretCast<int32_t>().SetValue(sliceBytes / sizeof(int32_t),
                                                                     expertMatchedRouteCount);
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

            uint64_t dstOffset = maskWinOffset_ +
                                 static_cast<uint64_t>(localExpertId * static_cast<int32_t>(worldSize_) +
                                                       static_cast<int32_t>(rankId_)) *
                                     static_cast<uint64_t>(maskSlotSize_) +
                                 static_cast<uint64_t>(batchStart / 8);
            dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)GetRankWinAddrWithOffset(dstRank, dstOffset));
            DataCopyPad(dstGlobalTensor, maskBuf, {1U, static_cast<uint32_t>(pushBytes), 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE3_V>(bufEvent);
        }
    }

    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
}

// ======================================================================
// LoadTopkWeightsToUb：权重搬运到UB（TopkWeightsPrefetch=0 时仅做 MTE2_V 同步）
// ======================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void
MegaMoe<TemplateMegaMoeTypeFunc>::LoadTopkWeightsToUb(const LocalTensor<ActivationType> &xOutTensor,
                                                      int32_t curentOffset, int32_t index, TEventID event)
{
    uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<TopkWeightsType> weightGm;
        weightGm.SetGlobalBuffer(
            (__gm__ TopkWeightsType *)(params_.probsGmAddr + (curentOffset + index) * topK_ * sizeof(TopkWeightsType)));
        if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
            LocalTensor<TopkWeightsType> weightBf16Tmp = mxTempTensor_.ReinterpretCast<TopkWeightsType>();
            DataCopyPad(weightBf16Tmp, weightGm,
                        {1U, static_cast<uint32_t>(topK_ * sizeof(TopkWeightsType)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            LocalTensor<float> weightFp32Ub = xOutTensor[weightOffsetInUb].template ReinterpretCast<float>();
            Cast(weightFp32Ub, weightBf16Tmp, AscendC::RoundMode::CAST_NONE, topK_);
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<TopkWeightsType> weightUb =
                xOutTensor[weightOffsetInUb].template ReinterpretCast<TopkWeightsType>();
            DataCopyPad(weightUb, weightGm, {1U, static_cast<uint32_t>(topK_ * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
    } else {
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    }
}

// ===================================
// QuantProcessInRank：本卡token的量化
// ===================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::QuantProcessInRank()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 分核，按照BS与aivCoreNum均分
    int32_t currentNum;
    int32_t currentOffset;
    TilingByCore(m_, currentNum, currentOffset, 1);
    uint32_t H = k_;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(H * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U, static_cast<uint32_t>(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_),
                                        0U, 0U, 0U};
    if constexpr (TopkWeightsPrefetch) {
        xCopyOutParams.blockLen =
            Ops::Base::CeilAlign(xCopyOutParams.blockLen + weightAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer(
            (__gm__ bfloat16_t *)(params_.aGmAddr + static_cast<uint64_t>(currentOffset + index) *
                                                        static_cast<uint64_t>(H) * sizeof(bfloat16_t)));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % DOUBLE_BUFFER == 0) ? xInTensor1_ : xInTensor2_;
        auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? xOutTensor1_ : xOutTensor2_;
        GlobalTensor<uint8_t> dstGlobalTensor;
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(params_.peermemInfo.quantTokenScalePtr +
                                                           static_cast<uint64_t>(currentOffset + index) *
                                                               static_cast<uint64_t>(mxQuantTokenScaleAlignBytes_)));
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        LoadTopkWeightsToUb(xOutTensor, currentOffset, index, event);
        __ubuf__ bfloat16_t *srcAddr = (__ubuf__ bfloat16_t *)xInTensor.GetPhyAddr();
        __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)mxTempTensor_.GetPhyAddr();
        __ubuf__ uint16_t *halfScaleAddr =
            (__ubuf__ uint16_t *)
                mxTempTensor_[Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))]
                    .GetPhyAddr();
        __ubuf__ int8_t *outDataAddr = (__ubuf__ int8_t *)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t *mxScaleAddr = (__ubuf__ uint16_t *)xOutTensor[mxQuantTokenAlignBytes_].GetPhyAddr();

        Quant::ComputeMaxExp(srcAddr, maxExpAddr, H); // 计算最大Exp
        Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
                                          mxQuantScaleNumAlignPerToken_); // 计算scales并填充f
        if constexpr (QuantMode == E2M1_QUANT) {
            Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        } else {
            Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// ==================================================================================================
// SendCntCal：stride 只读 count（跳过 mask 区），得到当前专家Id收到的token总数。
//
//   Phase 1: stride 读本 localExpert 的 worldsize 个 count；
//   Phase 2: 逐 rank 读 count + cumsum；
//   Phase 3: 写 expertRevNumsGlobalTensor_ + AtomicAdd 通知 AIC。
// ==================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendCntCal(int32_t localExpertId, uint64_t &sendCnt)
{
    sendCnt = 0;

    // Phase 1: stride 读本 localExpert 的 worldsize 个 count
    GlobalTensor<int32_t> cntSrcGlobal;
    cntSrcGlobal.SetGlobalBuffer((__gm__ int32_t *)(params_.peermemInfo.maskRecvPtr +
                                                    static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_ +
                                                    maskAlignSize_));
    DataCopyExtParams cntCopyParams{static_cast<uint16_t>(worldSize_), static_cast<uint32_t>(sizeof(int32_t)),
                                    static_cast<uint32_t>(maskSlotSize_ - sizeof(int32_t)), 0U, 0U};
    DataCopyPadExtParams<int32_t> cntPad{true, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, cntSrcGlobal, cntCopyParams, cntPad);

    if constexpr (ENABLE_A8W4) {
        if (localExpertId != 0) {
            // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，从 GM 加载前序 expert 的 cumsum
            DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                        {1U, static_cast<uint32_t>(worldSize_ * localExpertId * sizeof(int32_t)), 0U, 0U, 0U},
                        {true, 0U, 0U, 0U});
        }
    }

    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>(); // count 读取(标量)就绪
    if constexpr (TopkWeightsPrefetch) {
        // 权重前移路径：进入 MetaInfoCalAndDispatch 前确保 MTE2 流水线干净，
        // 避免与 MetaInfoCalAndDispatch 内 mask 搬运的 MTE2_V(ID1) 产生跨函数 flag 干扰。
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    }

    // Phase 2: 逐 rank 读 count + cumsum（4B count 按 32B burst 落位 → 下标 rank*8）
    constexpr int32_t CNT_STRIDE_I32 = ALIGN_32 / sizeof(int32_t);
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        int32_t perRankCnt = sendCntTensor_.GetValue(calRankId * CNT_STRIDE_I32);
        sendCnt += static_cast<uint64_t>(perRankCnt);
        cumsumRevCntInRank_ += static_cast<uint64_t>(perRankCnt);
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, static_cast<int32_t>(cumsumRevCntInRank_));
    }

    // Phase 3: 写 GM + 通知 AIC
    expertTokenCntTensor_.SetValue(0, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(expertRevNumsGlobalTensor_[localExpertId * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_],
                      expertTokenCntTensor_, INT32_PER_256B);
    if constexpr (ENABLE_A8W4) {
        // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，更新后写回 GM
        DataCopyPad(cumsumInfoGlobalTensor_, cumsumInfoTensor_,
                    {1U, static_cast<uint32_t>(worldSize_ * (localExpertId + 1) * sizeof(int32_t)), 0U, 0U, 0U});
    }
    PipeBarrier<PIPE_ALL>();

    __gm__ int32_t *sendCntFlag = (__gm__ int32_t *)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
                                  static_cast<uint64_t>(localExpertId) * aicNum_ * INT_CACHELINE +
                                  static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

// AIV1 owns dispatch.  Keeping this state transition in one helper makes the
// rolling scheduler advance the dispatch cursor exactly once per expert.
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchExpert(ExpertLoopState &state,
                                                                        GMMAddrInfo &gmmAddrInfo,
                                                                        uint32_t expertIdx,
                                                                        const DispatchBufferConfig &bufferConfig)
{
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ != 1) {
            return;
        }
        uint64_t sendCnt = 0;
        SendCntCal(static_cast<int32_t>(expertIdx), sendCnt);
        if (UpdateGroupParams<AddrUpdateMode::GMM1>(state, expertIdx, sendCnt)) {
            UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmmAddrInfo, state);
            MetaInfoCalAndDispatch(gmmAddrInfo, static_cast<int32_t>(expertIdx), bufferConfig);
        }
    }
}

// The rolling path can prefetch several experts before their GMM1 calls.  Read
// back this AIV1's persisted count instead of carrying a single nextSendCnt.
template <TemplateMegaMoeTypeClass>
__aicore__ inline uint64_t MegaMoe<TemplateMegaMoeTypeFunc>::ReadExpertTokenCount(uint32_t expertIdx)
{
    uint64_t offset = static_cast<uint64_t>(expertIdx) * INT32_PER_256B * aicNum_ +
                      static_cast<uint64_t>(INT32_PER_256B) * blockIdx_;
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        expertRevNumsGlobalTensor_[offset]);
    return static_cast<uint64_t>(expertRevNumsGlobalTensor_.GetValue(offset));
}

// ============================================================================
// DispatchCopyTmpTensor：由 UB 基址 + 槽偏移现场构造该槽的 buffer 视图。
//   热路径上取代 LocalTensor 数组索引，避免寄存器压力下 spill 到 GM。
// ============================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline auto
MegaMoe<TemplateMegaMoeTypeFunc>::DispatchCopyTmpTensor(int32_t bufferIdx) -> LocalTensor<ActivationType>
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC, copyTmpBaseAddr_ + static_cast<uint32_t>(bufferIdx) * mxQuantTokenScaleAlignBytes_,
        mxQuantTokenScaleAlignBytes_ / sizeof(ActivationType));
}

// ============================================================================
// FetchTokenNLoadMetaInfo：取 token 并装载元信息——MTE2 从远程 win 取该 token，S 侧组装 metaInfo(rank/token/topk)，
//   分别 set MTE2_MTE3 / S_MTE3 供 MTE3 侧消费。
//   IsBufferReuse 为编译期常量：首窗填槽实例(<false>)不生成任何复用 WaitFlag；稳态实例(<true>)才在覆盖前等该槽
//   上一轮的 MTE3 释放(buffer 用 MTE3_MTE2，metaInfo 槽用 MTE3_S)。两个实例分开成环，每个 token 便无需运行时分支。
//   TopkWeightsPrefetch=1 时，weight 数据随 token 一起搬运(copyInNum 含 weightAlignBytes_)，
//   但 weight 提取延迟到 DispatchCopyMte3 中 MTE2 完成后进行，故此处 set MTE2_S 而非 S_MTE3。
// ============================================================================
template <TemplateMegaMoeTypeClass>
template <bool IsBufferReuse>
__aicore__ inline void
MegaMoe<TemplateMegaMoeTypeFunc>::FetchTokenNLoadMetaInfo(int32_t bufferIdx, int32_t topkIndex, int32_t remoteRankIdx,
                                                          GlobalTensor<ActivationType> &remoteRankGlobalTensor,
                                                          uint32_t copyInNum)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx); // buffer 0~5 直接对应 EVENT_ID0~EVENT_ID5
    LocalTensor<ActivationType> copyTmpTensor = DispatchCopyTmpTensor(bufferIdx);
    int32_t tokenIndex = topkIndex / topK_;
    uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId); // 等该槽上一轮 token/scale 搬完，方可覆盖
    }
    DataCopy(copyTmpTensor, remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId); // 等该 metaInfo 槽被 MTE3 读走，方可覆盖
    }
    metaInfoTensor_[bufferIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
    metaInfoTensor_[bufferIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
    metaInfoTensor_[bufferIdx * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

// ============================================================================
// DispatchCopyMte3：搬出一个 dispatch 槽——token/scale/metaInfo 三段写 GM，收尾释放 buffer 与 metaInfo 槽。
//   TopkWeightsPrefetch=1 时，先 Wait<MTE2_S> 等 MTE2 搬运完成，从 copyTmp 中提取 weight 写入 WEIGHT_INDEX，
//   再 Set<S_MTE3>。
//   每 token 的元素数取自成员 revTokenElemCnt_/revScaleElemCnt_(DispatchBuffInit 算一次)，此处不再重算。
// ============================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchCopyMte3(
    int32_t bufferIdx, int32_t dstIdx, GlobalTensor<ActivationType> &tokenRevGlobalTensor,
    GlobalTensor<QuantScaleOutType> &scaleRevGlobalTensor, GlobalTensor<int32_t> &metaInfoGlobalTensor,
    int32_t copyStartIdx, int32_t copyIdx)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx); // buffer 0~5 直接对应 EVENT_ID0~EVENT_ID5
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScalebuf = DispatchCopyTmpTensor(bufferIdx);
    LocalTensor<QuantScaleOutType> bufScale =
        tokenScalebuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();

    if constexpr (TopkWeightsPrefetch) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
        LocalTensor<int32_t> bufWeightsInt32 = tokenScalebuf[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + copyIdx);
        int32_t weightBits = bufWeightsInt32.GetValue(static_cast<uint32_t>(topkIndex % topK_));
        metaInfoTensor_[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }

    DataCopyPad(tokenRevGlobalTensor[dstIdx * revTokenElemCnt_], tokenScalebuf,
                {1, static_cast<uint16_t>(revTokenElemCnt_ * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleRevGlobalTensor[dstIdx * revScaleElemCnt_], bufScale,
                {1, static_cast<uint16_t>(revScaleElemCnt_ * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId); // S 侧 metaInfo 组装完成后方可搬
    DataCopy(metaInfoGlobalTensor[dstIdx * INT32_PER_256B], metaInfoTensor_[bufferIdx * INT32_PER_256B],
             INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId); // 释放 buffer
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);    // 释放 metaInfo 槽
}

// ============================================================================
// CopyGMToGMPerToken：动态 2~6 buffer 软流水 + ring buffer metaInfo 即时写 GM
// ----------------------------------------------------------------------------
//   Phase 1  启动：先下发 token 0 的 MTE2，建立 MTE2 领先 MTE3 一个 token 的流水。
//   Phase 2a 首次填槽：buffer 1~(bufferCount-1) 尚未被用过，无需等待；每轮先下发 issueIdx，再搬出 issueIdx-1。
//   Phase 2b 稳态复用：token bufferCount 起绕环复用 buffer，覆盖前等对应 MTE3 释放，流水顺序保持不变。
//   Phase 3a 收尾搬出：上面两个循环都只搬到倒数第二个，这里补搬最后一个 token。
//   Phase 3b 收尾回收：消费本次实际用到的槽位上残留的 buffer-free event，避免影响下一次调用。
//
//   【为何 issue 必须排在 store 之前】若改成"搬完本轮再预取下一条"，预取前就得先等本轮 SetFlag<MTE3_MTE2>
//   被 MTE3 执行到，等价于把整条 MTE3 队列的完成时间压进每个 token 的关键路径，MTE3 深度被钉死为 1。
//   现在这种先 issue 后 store 的顺序下，Phase 2b 的 WaitFlag<MTE3_MTE2> 等的是 bufferCount-1 轮之前就已释放的槽，
//   实际不阻塞，MTE3 得以自由流水。
//
//   IsBufferReuse 拆成 2a/2b 两个循环从而成为编译期常量：首次填槽实例不生成 WaitFlag，稳态实例只在复用同一
//   槽位时等待，避免每个 token 做运行时分支。metaInfo 随 MTE2 下发现场组装到 ring buffer，并随 token/scale 即时
//   写 GM。event id 由槽号直接强转，buffer 由 UB base + 槽偏移构造，避免热路径数组索引 spill 到 GM。
//
//   【入参约束】copyNum >= 1 由唯一调用方 MetaInfoCalAndDispatch 的
//   `if (dispatchMatchOrdinalEnd > dispatchMatchOrdinalBegin)` 保证，故此处不做 copyNum<=0 的入口判断
//   （Phase 3a 会访问 copyNum-1，依赖该前提；若将来新增调用方，必须自行保证或恢复该判断）。
//   buffer 数取自 host 自适应配置 bufferConfig.bufferCount(2~6)，替代原先固定的 DISPATCH_BUFFER_NUM。
// ============================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyGMToGMPerToken(int32_t rowDstOffsetInCore,
                                                                            int32_t remoteRankIdx, int32_t copyStartIdx,
                                                                            int32_t copyNum,
                                                                            const DispatchBufferConfig &bufferConfig)
{
    // revTokenElemCnt_ / revScaleElemCnt_ 仅依赖 k_，已在 DispatchBuffInit 一次性算好(见成员)，此处不再逐调用重算。
    // copyInNum(输入 token-scale 拼接,非紧密排列)与 Init 里算好的 mxQuantTokenScaleAlignBytes_ 同为
    // CeilAlign(token+scale, 32)；ActivationType 恒为 1 字节(fp8 或 fp4 的 uint8 载体)，元素数即字节数，
    // 故直接复用成员，免去逐调用重算同一个 CeilAlign（本函数被每个 dispatch batch 调用，属热路径）。
    // bufferCount 为 host 自适应 UB 预算给出的 ring 深度(2~6)，与 DispatchBuffInit 分配的 copyTmp/metaInfo 槽数一致。
    int32_t bufferCount = bufferConfig.bufferCount;
    uint32_t copyInNum = mxQuantTokenScaleAlignBytes_;
    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        params_.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * revTokenElemCnt_));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * revScaleElemCnt_));
    remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(GetRankWinAddrWithOffset(remoteRankIdx, quantWinOffset_)));
    metaInfoGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params_.workspaceInfo.metaInfoPtr + rowDstOffsetInCore * INT32_PER_256B * sizeof(int32_t)));

    // 无需 PipeBarrier<PIPE_ALL>：读 validTopkIndexTensor_ 的 V→S 依赖已由 caller(MetaInfoCalAndDispatch)
    // GatherMask 后的 SyncFuncStatic<V_S> 覆盖；跨调用复用 dispatch buffer / metaInfoTensor_ 的 MTE3 已由本函数
    // 末尾 Phase 3b 排空；首调用的跨相位 UB 由 dispatch 相位入口同步覆盖。

    // Phase 1 启动：先发 token 0 的 MTE2，下一步即可在发 token 1 后搬出 token 0。
    int32_t firstTopkIndex = validTopkIndexTensor_.GetValue(copyStartIdx);
    FetchTokenNLoadMetaInfo<false>(0, firstTopkIndex, remoteRankIdx, remoteRankGlobalTensor, copyInNum);

    // Phase 2a 首次填槽：这些 buffer 没有上一轮 MTE3，用 <false> 在编译期删掉两个复用 wait。
    int32_t firstUseEnd = copyNum < bufferCount ? copyNum : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + issueIdx);
        FetchTokenNLoadMetaInfo<false>(issueIdx, topkIndex, remoteRankIdx, remoteRankGlobalTensor, copyInNum);
        DispatchCopyMte3(copyIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
                         copyStartIdx, copyIdx);
    }

    // Phase 2b 稳态复用：从 token bufferCount 起绕环覆盖旧槽，用 <true> 等该槽的 token/scale/metaInfo 均已搬出。
    for (int32_t issueIdx = bufferCount; issueIdx < copyNum; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + issueIdx);
        FetchTokenNLoadMetaInfo<true>(issueBufferIdx, topkIndex, remoteRankIdx, remoteRankGlobalTensor, copyInNum);
        DispatchCopyMte3(copyBufferIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
                         copyStartIdx, copyIdx);
    }

    // Phase 3a：补搬最后一个 token（两个循环都只搬到倒数第二个）。
    DispatchCopyMte3((copyNum - 1) % bufferCount, copyNum - 1, tokenRevGlobalTensor, scaleRevGlobalTensor,
                     metaInfoGlobalTensor, copyStartIdx, copyNum - 1);

    // Phase 3b：消费最后一轮 MTE3 产生的 buffer-free event，防止残留影响下一次调用。
    // 收支平衡：SetFlag 共 copyNum 次(每次 DispatchCopyMte3 一对)，Phase 2b 已消费 max(0, copyNum-bufferCount) 对，
    // 余下恰为 min(copyNum, bufferCount) = firstUseEnd 对，且残留槽号恰好覆盖 [0, firstUseEnd)。
    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx); // buffer i 对应 EVENT_IDi
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// ====================================================================================================
// MetaInfoCalAndDispatch：按 source rank 扫描 route mask，将本 core 负责的命中项 dispatch 到目标行。
// 坐标系：match ordinal 是当前 expert/source rank 内的命中序号；dst row 是跨 expert 累加的 workspace 行号；
// expert row 是当前 expert 内的行号，用于更新 GMM1 wave flag。
// 数据流：route mask -> compacted route index -> match ordinal -> dst row -> expert row/GMM1 wave。
// ====================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void
MegaMoe<TemplateMegaMoeTypeFunc>::MetaInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId,
                                                         const DispatchBufferConfig &bufferConfig)
{
    constexpr int32_t GMM1_WAVE_ROW_COUNT = static_cast<int32_t>(GMM1_TILE_M);
    // cumsumInfo 按 [expert][source rank] 累加；前一个 expert 的末值就是当前 expert 的全局起始行。
    int32_t expertGlobalRowBegin =
        (localExpertId == 0) ? 0 : cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);

    // A8W4 + prefetch 路径下 SwigluQuant 覆盖 V1 UB，topkIndexTensor_ 需重新初始化
    if constexpr (ENABLE_A8W4 && TopkWeightsPrefetch) {
        if (localExpertId != 0) {
            uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
                                                                static_cast<int64_t>(ALIGN_32));
            CreateVecIndex(topkIndexTensor_, 0, topkIndexTensorSize / sizeof(int32_t));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // 将 (source rank, rank 内 shard) 展平后分配给所有 block；同一 source rank 可由多个 core 并行 dispatch。
    for (uint32_t dispatchShardIdx = blockIdx_; dispatchShardIdx < worldSize_ * blockNumPerRank_;
         dispatchShardIdx += blockNum_) {
        uint32_t remoteRankIdx = dispatchShardIdx / blockNumPerRank_; // 当前扫描的 source rank
        uint32_t rankShardIdx = dispatchShardIdx % blockNumPerRank_;  // 当前 core 在该 rank 分片中的编号
        // 当前 expert 从该 source rank 接收的 token，在 dispatch workspace 中占据一个连续的 row segment。
        uint32_t rankSegmentDstRowBegin =
            ((remoteRankIdx == 0 && localExpertId == 0) ?
                 0 :
                 cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + remoteRankIdx - 1));
        // 当前 core 负责该 rank segment 中的命中序号区间 [coreMatchOrdinalBegin, coreMatchOrdinalEnd)。
        int32_t coreMatchOrdinalBegin = 0;
        int32_t coreMatchOrdinalEnd = 0;
        int32_t coreDstRowBegin = 0; // 上述区间首项在 dispatch workspace 中的全局目标行
        if (rankSegmentDstRowBegin < maxOutputSize_) {
            // rankTokenCount 是当前 source rank 发给当前 expert 的原始行数；rankDispatchRowCount 额外受
            // maxOutputSize_ 截断，是实际允许写入 workspace 的行数。
            int32_t rankTokenCount = cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + remoteRankIdx) -
                                     static_cast<int32_t>(rankSegmentDstRowBegin);
            int32_t rankDispatchRowCount = (rankSegmentDstRowBegin + rankTokenCount > maxOutputSize_) ?
                                               static_cast<int32_t>(maxOutputSize_ - rankSegmentDstRowBegin) :
                                               rankTokenCount;
            // 按行均分 rank segment；match ordinal 与该 segment 内的相对 row index 一一对应。
            int32_t rowsPerRankShard = Ops::Base::CeilDiv(rankDispatchRowCount, static_cast<int32_t>(blockNumPerRank_));
            int32_t rankShardRowBegin = rankShardIdx * rowsPerRankShard; // 当前 shard 在 rank segment 内的行偏移
            coreDstRowBegin = rankSegmentDstRowBegin + rankShardRowBegin;
            // 尾 shard 可能不足 rowsPerRankShard，需裁剪到 rank segment 的实际末尾。
            int32_t coreDispatchRowCount =
                (coreDstRowBegin + rowsPerRankShard > rankSegmentDstRowBegin + rankDispatchRowCount) ?
                    static_cast<int32_t>(rankSegmentDstRowBegin + rankDispatchRowCount - coreDstRowBegin) :
                    rowsPerRankShard;
            if (coreDispatchRowCount > 0) {
                coreMatchOrdinalBegin = rankShardRowBegin;
                coreMatchOrdinalEnd = rankShardRowBegin + coreDispatchRowCount;
            }
        }

        GlobalTensor<uint8_t> remoteRankMaskGlobal; // 当前 expert/source rank 对应的 route mask GM 视图
        int32_t matchedRouteCount = 0;  // 已扫描 batch 的累计命中数，即下一 batch 的首个 match ordinal
        int32_t dispatchedRowCount = 0; // 当前 core 已实际 dispatch 的总行数
        for (int32_t batchIdx = 0; batchIdx < recvRouteBatchCount_ && matchedRouteCount < coreMatchOrdinalEnd;
             ++batchIdx) {
            int32_t batchRouteBegin = batchIdx * recvRouteItemsPerBatch_; // 当前 batch 在原始 route 数组中的起始下标
            bool isLastBatch = (batchIdx == recvRouteBatchCount_ - 1);
            int32_t validRouteCount = recvRouteItemsPerBatch_;    // 当前 batch 的有效 route item 数
            int32_t maskSliceBytes = recvRouteItemsPerBatch_ / 8; // 当前 batch 对应的 mask 搬运字节数
            if (isLastBatch) {
                validRouteCount = static_cast<int32_t>(sendTotalNum_ - static_cast<uint64_t>(batchRouteBegin));
                if (batchRouteBegin / 8 + maskSliceBytes > static_cast<int32_t>(maskAlignSize_)) {
                    maskSliceBytes = static_cast<int32_t>(maskAlignSize_) - batchRouteBegin / 8;
                }
            }
            remoteRankMaskGlobal.SetGlobalBuffer(
                (__gm__ uint8_t *)(params_.peermemInfo.maskRecvPtr +
                                   (static_cast<uint64_t>(localExpertId) * worldSize_ + remoteRankIdx) * maskSlotSize_ +
                                   static_cast<uint64_t>(batchRouteBegin / 8)));
            DataCopy(maskBatchTensor_, remoteRankMaskGlobal, static_cast<uint32_t>(maskSliceBytes));
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
            // GatherMask 根据 mask 压缩本 batch 的全局 route index，并返回当前 batch 的命中数量。
            CreateVecIndex(topkIndexTensor_, batchRouteBegin, recvRouteItemsPerBatch_);
            uint64_t batchMatchedRouteCount = 0; // 当前 batch 中 mask=1 的 route item 数
            GatherMask(validTopkIndexTensor_, topkIndexTensor_, maskBatchU32Tensor_, true,
                       static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0}, batchMatchedRouteCount);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
            // 当前 batch 和本 core 各自在 match-ordinal 坐标系中的区间，二者交集即本次 dispatch 范围。
            int32_t batchMatchOrdinalBegin = matchedRouteCount; // 当前 batch 首个命中项的跨 batch 序号
            int32_t batchMatchOrdinalEnd = matchedRouteCount + static_cast<int32_t>(batchMatchedRouteCount);
            int32_t dispatchMatchOrdinalBegin =
                batchMatchOrdinalBegin > coreMatchOrdinalBegin ? batchMatchOrdinalBegin : coreMatchOrdinalBegin;
            int32_t dispatchMatchOrdinalEnd =
                batchMatchOrdinalEnd < coreMatchOrdinalEnd ? batchMatchOrdinalEnd : coreMatchOrdinalEnd;
            if (dispatchMatchOrdinalEnd > dispatchMatchOrdinalBegin) {
                // CopyGMToGMPerToken 的索引基于当前 batch 的压缩结果，目标行则使用跨 expert 的全局行号。
                int32_t batchLocalMatchBegin =
                    dispatchMatchOrdinalBegin - batchMatchOrdinalBegin; // 交集在 validTopkIndexTensor_ 中的起点
                int32_t batchDispatchRowCount =
                    dispatchMatchOrdinalEnd - dispatchMatchOrdinalBegin; // 本次从该 batch dispatch 的行数
                int32_t dispatchDstRowBegin = static_cast<int32_t>(rankSegmentDstRowBegin) + dispatchMatchOrdinalBegin;
                CopyGMToGMPerToken(dispatchDstRowBegin, remoteRankIdx, batchLocalMatchBegin, batchDispatchRowCount,
                                   bufferConfig);
                dispatchedRowCount += batchDispatchRowCount;
            }
            matchedRouteCount = batchMatchOrdinalEnd;
        }

        if (dispatchedRowCount > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            // GMM1 flag 按 expert 内的 wave 计数，因此先从全局 dst row 转换到 expert-local row。
            int32_t coreExpertRowBegin = coreDstRowBegin - expertGlobalRowBegin;
            int32_t coreExpertRowEnd = coreExpertRowBegin + dispatchedRowCount; // 当前 core 的 expert-local 半开区间
            int32_t firstWaveIdx = coreExpertRowBegin / GMM1_WAVE_ROW_COUNT;    // 该区间触达的首个 GMM1 wave
            int32_t lastWaveIdx = (coreExpertRowEnd - 1) / GMM1_WAVE_ROW_COUNT; // 该区间触达的末个 GMM1 wave
            __gm__ int32_t *flagBase = gmmAddrInfo.dispatchToGmm1Flag;
            for (int32_t waveIdx = firstWaveIdx; waveIdx <= lastWaveIdx; ++waveIdx) {
                int32_t waveExpertRowBegin = waveIdx * GMM1_WAVE_ROW_COUNT;
                int32_t waveExpertRowEnd = waveExpertRowBegin + GMM1_WAVE_ROW_COUNT;
                int32_t overlapRowBegin =
                    coreExpertRowBegin > waveExpertRowBegin ? coreExpertRowBegin : waveExpertRowBegin;
                int32_t overlapRowEnd = coreExpertRowEnd < waveExpertRowEnd ? coreExpertRowEnd : waveExpertRowEnd;
                // 每个 core 只累加自己与该 wave 的重叠行数；计数达到 wave 行数后 GMM1 才能消费。
                AtomicAdd(flagBase + static_cast<int64_t>(waveIdx) * INT_CACHELINE,
                          int32_t(overlapRowEnd - overlapRowBegin));
            }
        }
    }
}

// =====================================================================================================
// UpdateGroupParams：更新当前expertIdx的problemShape，偏移掉本卡前侧专家收到的cnt数
// ----------------------------------------------------------------------------------------------------
//   Phase 1: 根据problemShape中的M(前一个专家收到的count数)，偏移计算baseOffset中gmm1与gmm2的左右矩阵偏移；
//   Phase 2: 更新当前专家id收到的count数;
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx,
                                                                           uint64_t sendCnt)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
    }

    // gmm1中当前专家收到的count数是由subBlockIdx_=1的aiv计算出并写入expertRevNumsGlobalTensor_，通知后续aic/aiv0读取该值
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        if (subBlockIdx_ == 0) { // aiv1进行SendCntCal计算完成后atomicAddFlag，aic/aiv0等到该flag位后读取cnt值
            __gm__ int32_t *sendCntFlag = (__gm__ int32_t *)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
                                          static_cast<uint64_t>(expertIdx) * aicNum_ * INT_CACHELINE +
                                          static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
            while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) {
                }
            }

            uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGlobalTensor_[offsetInCnt]);
            Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor_.GetValue(offsetInCnt);
        } else {
            Get<M_VALUE>(state.problemShape) = sendCnt;
        }
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            expertRevNumsGlobalTensor_[offsetInCnt]);
        Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor_.GetValue(offsetInCnt);
    }

    if (Get<M_VALUE>(state.problemShape) == 0) {
        return false;
    }
    return true;
}

// =====================================================================================================
// UpdateSharedGroupParams：共享专家专用，M 恒为 m_，无 flag 等待与 DCache 操作。
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateSharedGroupParams(ExpertLoopState &state,
                                                                                 uint32_t expertIdx)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
    }

    Get<M_VALUE>(state.problemShape) = m_;
    return true;
}

// ==================================================================================
// UpdateGlobalBuffer：更新当前 expert 的 GMM 地址视图。
// ==================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
                                                                            const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        if constexpr (ENABLE_A8W4 || TopkWeightsPrefetch) {
            gmmAddrInfo.gmm1OutGlobal =
                params_.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.dispatchRevDataPtr + Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
                                   Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);

        gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);

        if constexpr (g_coreType == AIV) {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(state.baseOffset),
                Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                Get<IDX_FLAG_OFFSET>(state.baseOffset) * swigluFlagSlotsPerExpert_ / INT_CACHELINE,
                0L,
                0L,
                0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        if constexpr (ENABLE_A8W4 || ENABLE_A4W4 || CombineQuantMode != COMBINE_NO_QUANT ||
                      ENABLE_GMM2_GM_REBUILD) {
            gmmAddrInfo.gmm2OutGlobal =
                params_.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.swigluQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            uint64_t expertSyncSlotOffset = static_cast<uint64_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset)) *
                                            params_.tilingData->combineSyncSlotCountPerExpert;
            gmmAddrInfo.gmm2CombineSyncCounter = (__gm__ int32_t *)params_.workspaceInfo.gmm2CombineSyncCounterPtr +
                                                 expertSyncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
        }
    }
    gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t *)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                   Get<IDX_FLAG_OFFSET>(state.baseOffset) * swigluFlagSlotsPerExpert_;
    // wave-grain flags: every wave is INT_CACHELINE int32 (512B) apart, and
    // dispatchFlagSlotsPerExpert_ covers all waves owned by one expert.
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t *)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                     Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
}

// ==================================================================================
// UpdateSharedGlobalBuffer：共享专家专用，地址来自 shared* workspace，flags 为 nullptr。
// ==================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateSharedGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
                                                                                  const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertInputDataPtr;
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertInputScalePtr;
        if constexpr (ENABLE_A8W4) {
            gmmAddrInfo.gmm1OutGlobal = params_.workspaceInfo.sharedExpertGmm1OutPtr +
                                        Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.bGlobal = params_.sharedBGmAddr + Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.sharedBScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.sharedExpertResultPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertSwigluDataPtr +
                              Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertSwigluScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.sharedB2GmAddr + Get<IDX_B2_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.sharedB2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.swigluToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
}

// =============================================
// ResetGmm2CombineSyncCounters：重置 GMM2→Combine 同步计数器
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetGmm2CombineSyncCounters()
{
    // 无需 if constexpr(g_coreType==AIV) 守卫：唯一调用方 ResetFlagList 已对 AIC 提前 return，此函数只会在 AIV 进入。
    int32_t totalCounters = static_cast<int32_t>(params_.tilingData->combineSyncSlotCountPerExpert * moeExpertPerRank_ *
                                                 static_cast<uint64_t>(INT_CACHELINE));
    int32_t coreLen, coreOffset;
    TilingByCore(totalCounters, coreLen, coreOffset);
    GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
    gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.gmm2CombineSyncCounterPtr);
    if (coreLen > 0) {
        // resetTensor_ 已在 SendAndQuantBuffInit 一次性清零且全程只当零源用(从不写非零),
        // 无需在此再清零(与 ResetFlagList 同模式, 复用前面的清零); 保留 V->MTE3 同步保证零源对下面 DataCopy 可见。
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
        for (int32_t resetElementOffset = 0; resetElementOffset < coreLen;
             resetElementOffset += resetBatchElementCount_) {
            int32_t currentBatchElementCount = coreLen - resetElementOffset < resetBatchElementCount_ ?
                                                   coreLen - resetElementOffset :
                                                   resetBatchElementCount_;
            DataCopyExtParams resetCopyParams{1U, static_cast<uint32_t>(currentBatchElementCount * sizeof(int32_t)), 0U,
                                              0U, 0U};
            DataCopyPad(gmm2CombineSyncCounterGm[coreOffset + resetElementOffset], resetTensor_, resetCopyParams);
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline uint64_t MegaMoe<TemplateMegaMoeTypeFunc>::Gmm2NoQuantReadySlotStride() const
{
    if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_PRIVATE_STRIDED) {
        return static_cast<uint64_t>(blockNum_) * static_cast<uint64_t>(INT_CACHELINE);
    } else if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE) {
        constexpr uint64_t INT32_PER_64B = 64U / sizeof(int32_t);
        uint64_t readyElements =
            static_cast<uint64_t>(blockNum_) * static_cast<uint64_t>(MEGA_MOE_GMM2_READY_CORE_STRIDE);
        return (readyElements + INT32_PER_64B - 1U) / INT32_PER_64B * INT32_PER_64B;
    }
    return static_cast<uint64_t>(INT_CACHELINE);
}

// =================================================================================================
// Reset the selected AIC arrival layout and the private AIV1 release lines.
// Only physical AIV1 participates.  Every MTE3 copy is 512B so a reset never
// performs a partial-line read/modify/write over a neighbouring generation.
// =================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetGmm2NoQuantReadyCounters()
{
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ != 1) {
            return;
        }
        uint32_t logicalAiv1Idx = aivCoreIdx_ / 2U;
        uint32_t logicalAiv1Num = blockAivNum_ / 2U;
        if (logicalAiv1Num == 0U) {
            return;
        }
        uint32_t resetElements = INT_CACHELINE;
        if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE) {
            resetElements = static_cast<uint32_t>(Gmm2NoQuantReadySlotStride());
        }
        // The general resetTensor_ may be smaller than the selected ready
        // slot.  Materialize the whole zero slot in dedicated scratch.
        LocalTensor<int32_t> zeroLine(TPosition::VECCALC, GMM2_READY_SCAN_UB_ADDR, resetElements);
        Duplicate<int32_t>(zeroLine, 0, resetElements);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();

        GlobalTensor<int32_t> readyGm;
        readyGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm2NoQuantReadyPtr));
        if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_PRIVATE_STRIDED) {
            // Each logical AIV1 resets the same-indexed AIC private line.
            for (uint32_t slot = 0U; slot < expertPerRank_; ++slot) {
                uint64_t slotOffset = static_cast<uint64_t>(slot) * Gmm2NoQuantReadySlotStride() +
                                      static_cast<uint64_t>(logicalAiv1Idx) * INT_CACHELINE;
                DataCopy(readyGm[slotOffset], zeroLine, INT_CACHELINE);
            }
        } else if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE) {
            // One worker resets one complete generation at a time.  This
            // preserves all zero padding consumed by the AIV1 reduction.
            for (uint32_t slot = logicalAiv1Idx; slot < expertPerRank_; slot += logicalAiv1Num) {
                uint64_t slotOffset = static_cast<uint64_t>(slot) * Gmm2NoQuantReadySlotStride();
                DataCopy(readyGm[slotOffset], zeroLine, resetElements);
            }
        } else {
            // Shared-atomic and compact-private layouts both occupy one line
            // per slot, distributed across AIV1 reset workers.
            for (uint32_t slot = logicalAiv1Idx; slot < expertPerRank_; slot += logicalAiv1Num) {
                uint64_t slotOffset = static_cast<uint64_t>(slot) * Gmm2NoQuantReadySlotStride();
                DataCopy(readyGm[slotOffset], zeroLine, INT_CACHELINE);
            }
        }

        if constexpr (MEGA_MOE_GMM2_FANIN_MODE != MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE) {
            GlobalTensor<int32_t> releaseGm;
            releaseGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm2NoQuantReleasePtr));
            for (uint32_t slot = 0U; slot < expertPerRank_; ++slot) {
                uint64_t releaseOffset =
                    (static_cast<uint64_t>(slot) * logicalAiv1Num + logicalAiv1Idx) * INT_CACHELINE;
                DataCopy(releaseGm[releaseOffset], zeroLine, INT_CACHELINE);
            }
        }
    }
}

// ================================================================================================
// RunGmm2NoQuantReadySmokeTest：默认关闭的最小 NPU 地址 smoke。
// Every AIC publishes slot 0 and every physical AIV1 traverses the selected
// coordinator/release path.  It does not run GMM/Combine or measure latency.
// ================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RunGmm2NoQuantReadySmokeTest()
{
    PublishGmm2NoQuantReady(0U);
    WaitGmm2NoQuantReady(0U);
    SyncAll<true>();
}

// The lower 32 bits are sufficient for a single-op trace.  Consumers compare
// markers with uint32 modular subtraction so a counter wrap is harmless.
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RecordGmm2WaveTimeline(uint32_t slotIdx)
{
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
    constexpr uint32_t TRACE_WORDS = 16U;
    if (expertPerRank_ < TRACE_WORDS || slotIdx >= TRACE_WORDS) {
        return;
    }
    __gm__ uint32_t *trace = reinterpret_cast<__gm__ uint32_t *>(params_.expertTokenNumsOutGmAddr);
    WriteGmByPassDCache(trace + slotIdx, static_cast<uint32_t>(AscendC::GetSystemCycle()));
#else
    (void)slotIdx;
#endif
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::WriteGmm2WaveTimelineMetadata(uint32_t expertsPerWave)
{
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
    constexpr uint32_t TRACE_WORDS = 16U;
    if (expertPerRank_ < TRACE_WORDS) {
        return;
    }
    __gm__ uint32_t *trace = reinterpret_cast<__gm__ uint32_t *>(params_.expertTokenNumsOutGmAddr);
    WriteGmByPassDCache(trace + 10U, uint32_t(0x57415645U)); // "WAVE"
    WriteGmByPassDCache(trace + 11U, uint32_t(1U));          // trace format version
    WriteGmByPassDCache(trace + 12U, blockNum_);
    WriteGmByPassDCache(trace + 13U, expertsPerWave);
    WriteGmByPassDCache(trace + 14U, worldSize_);
    WriteGmByPassDCache(trace + 15U, uint32_t(MEGA_MOE_GMM2_READY_MODE));
#else
    (void)expertsPerWave;
#endif
}

// AIC fan-in publication.  Every AIC contributes once for the selected slot,
// including cores that owned no tile.  FIX_S orders all prior direct-GM GMM2
// writes before this core's arrival publication.
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::PublishGmm2NoQuantReady(uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        __gm__ int32_t *readyBase =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm2NoQuantReadyPtr);
        uint64_t slotOffset = static_cast<uint64_t>(slotIdx) * Gmm2NoQuantReadySlotStride();
        if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_SHARED_ATOMIC) {
            AscendC::AtomicAdd(readyBase + slotOffset, int32_t(1));
        } else if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_PRIVATE_COMPACT) {
            AscendC::WriteGmByPassDCache(readyBase + slotOffset + blockIdx_, int32_t(1));
        } else if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE) {
            AscendC::WriteGmByPassDCache(
                readyBase + slotOffset +
                    static_cast<uint64_t>(blockIdx_) * MEGA_MOE_GMM2_READY_CORE_STRIDE,
                int32_t(1));
        } else {
            AscendC::WriteGmByPassDCache(
                readyBase + slotOffset + static_cast<uint64_t>(blockIdx_) * INT_CACHELINE, int32_t(1));
        }
    }
}

// One physical AIV1 coordinates the AIC fan-in, then publishes one private
// 512B release line per AIV1 with a single MTE3 copy.  Non-coordinators poll
// only their own line, eliminating the 36-reader hot address in the baseline.
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::WaitGmm2NoQuantReady(uint32_t slotIdx)
{
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ != 1) {
            return;
        }
        uint32_t logicalAiv1Idx = aivCoreIdx_ / 2U;
        uint32_t logicalAiv1Num = blockAivNum_ / 2U;
        if (logicalAiv1Num == 0U) {
            return;
        }

        if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE) {
            constexpr uint32_t INT32_PER_DATABLOCK = ALIGN_32 / sizeof(int32_t);
            constexpr uint32_t INT32_PER_REPEAT = 256U / sizeof(int32_t);
            uint32_t readyElements = blockNum_ * static_cast<uint32_t>(MEGA_MOE_GMM2_READY_CORE_STRIDE);
            uint32_t copyElements =
                (readyElements + INT32_PER_DATABLOCK - 1U) / INT32_PER_DATABLOCK * INT32_PER_DATABLOCK;
            uint32_t fullRepeats = readyElements / INT32_PER_REPEAT;
            uint32_t tailElements = readyElements % INT32_PER_REPEAT;
            uint32_t partialCount = fullRepeats + (tailElements == 0U ? 0U : 1U);

            __gm__ int32_t *readyBase =
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm2NoQuantReadyPtr);
            uint64_t readySlotOffset = static_cast<uint64_t>(slotIdx) * Gmm2NoQuantReadySlotStride();
            GlobalTensor<int32_t> readyGm;
            readyGm.SetGlobalBuffer(readyBase);
            LocalTensor<int32_t> scanUb(TPosition::VECCALC, GMM2_READY_SCAN_UB_ADDR, copyElements);
            LocalTensor<int32_t> partialUb(TPosition::VECCALC, GMM2_READY_PARTIAL_UB_ADDR, INT_CACHELINE);
            LocalTensor<int32_t> sumUb(TPosition::VECCALC, GMM2_READY_SUM_UB_ADDR, INT_CACHELINE);

            while (true) {
                DataCopy(scanUb, readyGm[readySlotOffset], copyElements);
                SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
                WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);

                if (readyElements <= INT32_PER_REPEAT) {
                    WholeReduceSum<int32_t>(sumUb, scanUb, static_cast<int32_t>(readyElements), 1, 1, 1, 8);
                } else {
                    if (fullRepeats != 0U) {
                        WholeReduceSum<int32_t>(partialUb, scanUb, static_cast<int32_t>(INT32_PER_REPEAT),
                                                static_cast<int32_t>(fullRepeats), 1, 1, 8);
                    }
                    if (tailElements != 0U) {
                        WholeReduceSum<int32_t>(partialUb[fullRepeats], scanUb[fullRepeats * INT32_PER_REPEAT],
                                                static_cast<int32_t>(tailElements), 1, 1, 1, 8);
                    }
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum<int32_t>(sumUb, partialUb, static_cast<int32_t>(partialCount), 1, 1, 1, 8);
                }

                SetFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
                WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
                if (sumUb.GetValue(0) >= static_cast<int32_t>(blockNum_)) {
                    return;
                }
                int64_t waitStart = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - waitStart < 100) {
                }
            }
        }

        __gm__ int32_t *releaseBase =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm2NoQuantReleasePtr);
        uint64_t releaseSlotOffset = static_cast<uint64_t>(slotIdx) * logicalAiv1Num * INT_CACHELINE;
        if (logicalAiv1Idx != 0U) {
            __gm__ int32_t *ownRelease =
                releaseBase + releaseSlotOffset + static_cast<uint64_t>(logicalAiv1Idx) * INT_CACHELINE;
            while (AscendC::ReadGmByPassDCache(ownRelease) < 1) {
                int64_t waitStart = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - waitStart < 100) {
                }
            }
            return;
        }

        // Prepare all private release lines while the coordinator is waiting
        // for C cores.  This vector work is independent from the scalar poll.
        uint32_t releaseElements = logicalAiv1Num * static_cast<uint32_t>(INT_CACHELINE);
        LocalTensor<int32_t> releaseUb(TPosition::VECCALC, GMM2_READY_RELEASE_UB_ADDR, releaseElements);
        Duplicate<int32_t>(releaseUb, 1, releaseElements);
        SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);

        __gm__ int32_t *readyBase =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm2NoQuantReadyPtr);
        uint64_t readySlotOffset = static_cast<uint64_t>(slotIdx) * Gmm2NoQuantReadySlotStride();
        if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_SHARED_ATOMIC) {
            while (AscendC::ReadGmByPassDCache(readyBase + readySlotOffset) < static_cast<int32_t>(blockNum_)) {
                int64_t waitStart = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - waitStart < 100) {
                }
            }
        } else if constexpr (MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_PRIVATE_COMPACT) {
            GlobalTensor<int32_t> readyGm;
            readyGm.SetGlobalBuffer(readyBase);
            LocalTensor<int32_t> scanUb(TPosition::VECCALC, GMM2_READY_SCAN_UB_ADDR, INT_CACHELINE);
            LocalTensor<int32_t> sumUb(TPosition::VECCALC, GMM2_READY_SCAN_UB_ADDR + ALIGN_512, INT_CACHELINE);
            while (true) {
                DataCopy(scanUb, readyGm[readySlotOffset], INT_CACHELINE);
                SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
                WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
                WholeReduceSum<int32_t>(sumUb, scanUb, static_cast<int32_t>(blockNum_), 1, 1, 1, 8);
                SetFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
                WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
                if (sumUb.GetValue(0) >= static_cast<int32_t>(blockNum_)) {
                    break;
                }
                int64_t waitStart = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - waitStart < 100) {
                }
            }
        } else {
            uint64_t pending = 0U;
            for (uint32_t coreIdx = 0U; coreIdx < blockNum_; ++coreIdx) {
                pending |= uint64_t(1) << coreIdx;
            }
            while (pending != 0U) {
                for (uint32_t coreIdx = 0U; coreIdx < blockNum_; ++coreIdx) {
                    uint64_t bit = uint64_t(1) << coreIdx;
                    if ((pending & bit) != 0U &&
                        AscendC::ReadGmByPassDCache(readyBase + readySlotOffset +
                                                   static_cast<uint64_t>(coreIdx) * INT_CACHELINE) >= 1) {
                        pending &= ~bit;
                    }
                }
                if (pending == 0U) {
                    break;
                }
                int64_t waitStart = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - waitStart < 100) {
                }
            }
        }

        GlobalTensor<int32_t> releaseGm;
        releaseGm.SetGlobalBuffer(releaseBase);
        WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        DataCopy(releaseGm[releaseSlotOffset], releaseUb, releaseElements);
        SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
        WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline MegaMoeImpl::TokenRange
MegaMoe<TemplateMegaMoeTypeFunc>::GetCombineNoQuantOwnedRange(uint32_t tokenCount)
{
    uint32_t logicalAiv1Num = blockAivNum_ / 2U;
    if (logicalAiv1Num == 0U) {
        return {};
    }
    return MegaMoeImpl::GetBalancedTokenRange(tokenCount, aivCoreIdx_ / 2U, logicalAiv1Num);
}

// Dispatch and combine time-share AIV1's dynamic UB/event resources.  Dispatch
// is fully drained before this ring is initialized, so EVENT_ID0..EVENT_ID5
// can remain live across every expert in the current wave.
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitCombineNoQuantRowRing()
{
    // Keep event IDs as immediates.  A device-side event table may spill to GM
    // and add a load to every row in this hot path.
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DrainCombineNoQuantRowRing()
{
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
}

// Bulk-load one contiguous slice owned by this AIV1.  PER_BATCH owns a range
// in the whole expert-major wave; PER_EXPERT owns a range in one expert.
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::PreloadCombineNoQuantMetaInfo(
    uint64_t metaInfoGmTokenOffset, uint32_t tokenCount, uint32_t metaInfoUbTokenOffset)
{
    if (tokenCount == 0U) {
        return;
    }
    uint32_t metaInfoUbByteOffset =
        metaInfoUbTokenOffset * static_cast<uint32_t>(META_INFO_SIZE * sizeof(int32_t));
    LocalTensor<int32_t> metaInfoUb(TPosition::VECCALC, META_INFO_TENSOR_ADDR + metaInfoUbByteOffset,
                                    tokenCount * META_INFO_SIZE);
    GlobalTensor<int32_t> metaInfoGm;
    metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.metaInfoPtr));
    uint64_t metaInfoGmOffset = metaInfoGmTokenOffset * META_INFO_SIZE;
    DataCopy(metaInfoUb, metaInfoGm[metaInfoGmOffset], tokenCount * META_INFO_SIZE);
}

// Token-grain AIV1 combine.  The caller has already bulk-loaded and made the
// matching meta-info slice scalar-visible.  rowSequence is wave-scoped, hence
// six-slot row can stay in flight across the entire contiguous wave range.
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCombineNoQuantGm(
    GM_ADDR gmm2OutGlobal, uint32_t tokenStart, uint32_t tokenCount, uint32_t metaInfoUbTokenOffset,
    uint32_t &rowSequence)
{
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ != 1 || tokenCount == 0U) {
            return;
        }

        // Only the cumsum state below 64 KiB remains live here.  Six maximum
        // H=8K BF16 rows end at 160 KiB, immediately before ready scratch.
        constexpr uint32_t ROW_UB_BASE = 64U * 1024U;
        constexpr uint32_t MAX_SUPPORTED_ROW_BYTES = 8U * 1024U * sizeof(bfloat16_t);
        static_assert(ROW_UB_BASE + COMBINE_ROW_BUFFER_NUM * MAX_SUPPORTED_ROW_BYTES <= GMM2_READY_RELEASE_UB_ADDR,
                      "GMM2 row buffers overlap ready release scratch");

        uint32_t rowBytes = k_ * sizeof(bfloat16_t);
        uint32_t rowStrideBytes = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<uint64_t>(rowBytes), static_cast<uint64_t>(ALIGN_32)));
        uint32_t metaInfoUbByteOffset =
            metaInfoUbTokenOffset * static_cast<uint32_t>(META_INFO_SIZE * sizeof(int32_t));
        LocalTensor<int32_t> metaInfoUb(TPosition::VECCALC, META_INFO_TENSOR_ADDR + metaInfoUbByteOffset,
                                        tokenCount * META_INFO_SIZE);

        GlobalTensor<bfloat16_t> gmm2OutGm;
        gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gmm2OutGlobal));
        DataCopyExtParams gm2UbParams{1U, rowBytes, 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> gm2UbPad{false, 0U, 0U, 0U};
        uint64_t gmRemoteBaseOffset =
            params_.peermemInfo.combineSendPtr - params_.peermemInfo.rankSyncInWorldPtr;

        for (uint32_t tokenIdxInSlice = 0; tokenIdxInSlice < tokenCount; ++tokenIdxInSlice, ++rowSequence) {
            uint32_t tokenLocal = tokenStart + tokenIdxInSlice;
            uint32_t slot = rowSequence % COMBINE_ROW_BUFFER_NUM;
            // EVENT_ID0..5 are contiguous: calculate the scalar event ID
            // directly instead of indexing a device-side table that may spill.
            TEventID eventId = CombineNoQuantRowEventId(slot);
            LocalTensor<bfloat16_t> rowUb(TPosition::VECIN, ROW_UB_BASE + slot * rowStrideBytes,
                                          rowStrideBytes / sizeof(bfloat16_t));
            LocalTensor<int32_t> tokenMetaInfo = metaInfoUb[tokenIdxInSlice * META_INFO_SIZE];
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);

            DataCopyPad(rowUb, gmm2OutGm[static_cast<uint64_t>(tokenLocal) * k_], gm2UbParams, gm2UbPad);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            MegaMoeCombineImpl::CombineTokenRowTiled<bfloat16_t>(
                k_, gmRemoteBaseOffset, tokenMetaInfo, rowUb, params_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
    }
}

// =============================================
// InitCombineBuffers：初始化 Combine 所需的 buffer 大小
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitCombineBuffers()
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        // 下面两个只依赖 k_ 的量提成员, 供 ProcessCombine 每 expert 复用(原先每次调用重算)
        combineQuantTokenSizeBytes_ = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
        combineNTilesPerGroup_ = Ops::Base::CeilDiv(k_, L1_TILE_N);
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + combineQuantTokenSizeBytes_;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

// =============================================
// ProcessCombine：generic combine-quant 路径的 AIV 后处理。
//                 等待本 expert 的 row-group 计数满足后，读取 metaInfo 和 GMM2 输出，
//                 再执行 row-group 级 CombineRowGroup。
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCombine(const GMMAddrInfo &gmmAddrInfo,
                                                                        const ExpertLoopState &gmm2State,
                                                                        uint32_t expertIdx)
{
    uint32_t expertTokenCount = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(expertTokenCount, COMBINE_TOKEN_GROUP_SIZE);

    // generic 路径的每个 AIV 都是 logical core；A8W4/A4W4 仅 subBlockIdx=1 参与并按物理核对映射。
    uint32_t logicalCoreId = aivCoreIdx_;
    uint32_t logicalCoreCount = blockAivNum_;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        if (subBlockIdx_ != 1) {
            return; // 配对路径下仅 sub==1 的核参与 combine 后处理，sub==0 直接退出
        }
        logicalCoreId = aivCoreIdx_ / 2;
        logicalCoreCount = blockAivNum_ / 2;
    }

    uint32_t firstAssignedGroup = 0;
    uint32_t assignedGroupStride = 0;
    uint32_t coreIndexWithinGroup = 0;
    uint32_t coresAssignedToGroup = 0;
    MegaMoeImpl::ComputeCombineGroupsForCore(logicalCoreId, tokenGroupsThisExpert, logicalCoreCount, firstAssignedGroup,
                                             assignedGroupStride, coreIndexWithinGroup, coresAssignedToGroup);

    for (uint32_t groupIndex = firstAssignedGroup; groupIndex < tokenGroupsThisExpert;
         groupIndex += assignedGroupStride) {
        // 多核协作时每个 logical core 有独立 slot；一核处理多 group 时每个 group 有独立 slot。
        uint32_t syncSlotIndex = tokenGroupsThisExpert <= logicalCoreCount ? logicalCoreId : groupIndex;
        __gm__ int32_t *syncCounterAddress =
            MegaMoeImpl::GetCombineSyncCounterAddress(gmmAddrInfo.gmm2CombineSyncCounter, syncSlotIndex);
        while (AscendC::ReadGmByPassDCache(syncCounterAddress) != combineNTilesPerGroup_) {
            int64_t waitStartCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - waitStartCycle < 100) {
            }
        }

        uint32_t groupTokenStart = groupIndex * COMBINE_TOKEN_GROUP_SIZE;
        uint32_t groupTokenCount = COMBINE_TOKEN_GROUP_SIZE < expertTokenCount - groupTokenStart ?
                                       COMBINE_TOKEN_GROUP_SIZE :
                                       expertTokenCount - groupTokenStart;
        uint32_t tokensPerCore = Ops::Base::CeilDiv(groupTokenCount, coresAssignedToGroup);
        uint32_t tokenOffsetWithinGroup = coreIndexWithinGroup * tokensPerCore;
        // tail group 的 token 可能少于协作核数，部分核不分配 token。
        if (tokenOffsetWithinGroup >= groupTokenCount) {
            continue;
        }
        uint32_t tokenCountForCore = groupTokenCount - tokenOffsetWithinGroup;
        tokenCountForCore = tokenCountForCore < tokensPerCore ? tokenCountForCore : tokensPerCore;

        AscendC::SetCtrlSpr<60, 60>(0);
        int64_t offset = 0;
        LocalTensor<int32_t> metaInfoTensor =
            LocalTensor<int32_t>(TPosition::VECIN, offset, tokenCountForCore * META_INFO_SIZE);
        offset += tokenCountForCore * META_INFO_SIZE * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> metaInfoGm;
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.metaInfoPtr +
            (gmm2State.expertBeforeCnt + groupTokenStart + tokenOffsetWithinGroup) * META_INFO_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, tokenCountForCore * META_INFO_SIZE);
        PipeBarrier<PIPE_MTE2>();
        MegaMoeCombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t>(
            groupTokenStart + tokenOffsetWithinGroup, tokenCountForCore, k_, expertIdx, rankId_,
            gmmAddrInfo.gmm2OutGlobal, params_, metaInfoTensor, combineUbTensorSize_, offset,
            combineQuantTokenSizeBytes_);
    }
}

// ===============================================================
// UnpermuteLoadWeights：加载一个 token batch 的权重到 UB
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void
MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteLoadWeights(int32_t coreOffset, int32_t batchTokenOffset,
                                                       int32_t batchTokenCount, LocalTensor<bfloat16_t> &tempLocal)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float *)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(batchTokenCount * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_[(coreOffset + batchTokenOffset) * topK_], copyParams,
                    copyPadParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t *)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(batchTokenCount * topK_ * sizeof(bfloat16_t)), 0U, 0U,
                                        0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor[(coreOffset + batchTokenOffset) * topK_], copyParams,
                    copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, batchTokenCount * topK_);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
    }
}

// ===============================================================
// UnpermuteProcessToken：单个 token 的 per-expert 累加
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void
MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteProcessToken(int32_t tokenIdx, int32_t localIdx,
                                                        const GlobalTensor<bfloat16_t> &expandedX,
                                                        const UnpermuteBufferConfig &bufferConfig)
{
    for (int32_t expId = 0; expId < topK_; ++expId) {
        // Routed and shared expert results form one continuous accumulation-input sequence in the dynamic ring.
        int32_t accumulationItemIdxInBatch = localIdx * (topK_ + static_cast<int32_t>(sharedExpertNum_)) + expId;
        int32_t inputBufferIdx = accumulationItemIdxInBatch % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 = dataResTensor_[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 = dataResFp32Tensor_[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(static_cast<uint64_t>(tokenIdx) * topK_ + expId) * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
        } else {
            uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
            uint32_t quantTokenSize = k_ + nScale;
            uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(static_cast<uint64_t>(tokenIdx) * topK_ + expId) * quantEleNum],
                     quantEleNum);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            using Fp8Type =
                typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
            MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_,
                                                                  fp32ScaleTensor_, nScale, k_);
        }
        // GetValue 在 Scalar 流水读取 expScale；两条反量化路径汇合后统一等待，再由 Vector 流水消费。
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        if constexpr (TopkWeightsPrefetch) {
            if (expId == 0) {
                DataCopy(dataResFp32Tensor_, dataInFp32, k_);
            } else {
                Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
            }
        } else {
            float expScale = topKWeightsTensor_.GetValue(localIdx * topK_ + expId);
            if (expId == 0) {
                Muls(dataResFp32Tensor_, dataInFp32, expScale, k_);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, k_);
                PipeBarrier<PIPE_V>();
                Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

// ===============================================================
// UnpermuteBuffInit：分配 Unpermute 所需固定 buffer，返回本阶段的 buffer 配置
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBuffInit()
{
    // 必须与 host SetAdaptiveBufferConfigs 对 TilingByCore(m_, ..., align=1) 的完整 chunk/tail chunk
    // 推导保持一致。coreLen 为 0 的非活跃 core 已在 Unpermute 中提前返回，不会读取 tail 配置。
    UnpermuteBufferConfig bufferConfig = aivCoreIdx_ < params_.tilingData->unpermuteFullTokenChunkCoreCount ?
                                             params_.tilingData->unpermuteConfigForFullTokenChunk :
                                             params_.tilingData->unpermuteConfigForTailTokenChunk;

    uint32_t bf16ScaleBufAlign = 0;
    uint32_t fp32ScaleBufAlign = 0;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DEQUANT_BF16_SCALE_EXPANSION),
                                 static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION),
                                 static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    int32_t tokensPerBatch = bufferConfig.tokensPerBatch;
    uint32_t topKWeightsBufAlign = bufferConfig.topKWeightsBufferBytes;
    uint32_t topKWeightsConversionBufferBytes = bufferConfig.topKWeightsConversionBufferBytes;

    uint32_t dataResBufAlign = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufAlign = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    // Tensor用处：Unpermute 函数用于存储 mte2 搬入 token；
    // Tensor大小：(1 + bufferConfig.inputBufferCount) × 独立对齐后的 BF16 单槽大小；
    // 1 块用于累加/搬出，其余用于 MTE2 搬入。
    uint32_t dataResAddr = 0;
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResBufAlign / sizeof(bfloat16_t));
    // Tensor用处：Unpermute 函数用于存储 token Cast 目的 Tensor；
    // Tensor大小：(1 + bufferConfig.inputBufferCount) × 独立对齐后的 FP32 单槽大小；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32BufAlign / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufAlign;

    // weight buffer（在 scale 之前，与 master 顺序一致）
    // Tensor用处：用于存储 topKWeight；
    // Tensor大小：tokensPerBatch × topK_ × sizeof(float) align 到 32 字节对齐；
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, topKWeightsBufAlign / sizeof(float));
    tempAddr += topKWeightsBufAlign;

    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        // Tensor用处：Unpermute 中 bf16 weight 搬运中转 buffer；
        // Tensor大小：tokensPerBatch × topK_ × sizeof(bfloat16_t) align 到 32 字节；
        topKWeightsBf16Tensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr,
                                                         topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += topKWeightsConversionBufferBytes;
    }

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）
        // Tensor大小：scaleNum × sizeof(bfloat16_t) × DEQUANT_BF16_SCALE_EXPANSION
        bf16ScaleTensor_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）
        // Tensor大小：scaleNum × sizeof(float) × DEQUANT_FP32_SCALE_EXPANSION
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }

    return bufferConfig;
}

// ===============================================================
// UnpermuteSharedExpert：共享专家结果累加到当前 token 的 fp32 累加器
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void
MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteSharedExpert(int32_t tokenIdx, int32_t localIdx,
                                                        const UnpermuteBufferConfig &bufferConfig)
{
    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer((__gm__ bfloat16_t *)params_.workspaceInfo.sharedExpertResultPtr);
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        int32_t accumulationItemIdxInBatch =
            localIdx * (topK_ + static_cast<int32_t>(sharedExpertNum_)) + topK_ + static_cast<int32_t>(sharedIdx);
        int32_t inputBufferIdx = accumulationItemIdxInBatch % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 = dataResTensor_[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 = dataResFp32Tensor_[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        DataCopy(dataInBf16, sharedResult[(sharedIdx * m_ + tokenIdx) * k_], k_);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
        Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
        PipeBarrier<PIPE_V>();
        Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
        PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

// ===============================================================
// Unpermute：主入口 — 初始化 buffer → 分批循环处理
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    if (coreLen == 0) {
        return;
    }
    UnpermuteBufferConfig bufferConfig = UnpermuteBuffInit();
    int32_t tokensPerBatch = bufferConfig.tokensPerBatch;

    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t *)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t *)params_.y2GmAddr);

    // 输出槽由 Vector 写入、MTE3 读出。先将槽位交给 Vector，最终回收最后一次 MTE3 完成信号。
    constexpr TEventID kOutputBufferEvent = EVENT_ID0;
    SetFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);

    // 外层：token batch
    for (int32_t batchTokenOffset = 0; batchTokenOffset < coreLen; batchTokenOffset += tokensPerBatch) {
        int32_t batchTokenCount =
            (batchTokenOffset + tokensPerBatch > coreLen) ? (coreLen - batchTokenOffset) : tokensPerBatch;

        if constexpr (!TopkWeightsPrefetch) {
            UnpermuteLoadWeights(coreOffset, batchTokenOffset, batchTokenCount, topKWeightsBf16Tensor_);
        }

        // 内层：token 循环
        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
            SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
        for (int32_t localIdx = 0; localIdx < batchTokenCount; localIdx++) {
            int32_t tokenIdx = coreOffset + batchTokenOffset + localIdx;
            UnpermuteProcessToken(tokenIdx, localIdx, expandedX, bufferConfig);
            // 共享专家结果累加（直接加，不乘 topk_weight）
            if (sharedExpertNum_ > 0) {
                UnpermuteSharedExpert(tokenIdx, localIdx, bufferConfig);
            }
            // MTE2 使用独立输入槽，可与上一 token 的 MTE3 输出重叠；仅在覆盖输出槽前等待 MTE3 读完。
            WaitFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);
            Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[static_cast<uint64_t>(tokenIdx) * k_], dataResTensor_, k_);
            SetFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);
        }
        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
    }
    WaitFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);
}

// ==============================================================================================
// CrossRankSyncInWorldSize：全卡同步，rankSyncInWorldPtr前48K用于同步，后面区域用于记录当前syncCnt值
// ==============================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t *syncRank = (__gm__ int32_t *)(params_.peermemInfo.rankSyncInWorldPtr);
    __gm__ int32_t *syncCount =
        (__gm__ int32_t *)(params_.peermemInfo.rankSyncInWorldPtr + 48 * 1024 + aivCoreIdx_ * 64);
    int count = ReadGmByPassDCache(syncCount) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t *syncRemoteAddr = (__gm__ int32_t *)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

// ===============================================================
// SharedExpertCopyInput：将本卡量化后的交错 data+scale 拆分为连续布局
//   源: quantTokenScalePtr [token: data(256B aligned) | scale] 交错排列
//   目标: sharedExpertInputDataPtr [bs × h] 连续, sharedExpertInputScalePtr [bs × scaleN] 连续
//   AIV 执行，在量化完成后、AIC GMM1 开始前调用
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SharedExpertCopyInput()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t curentNum;
    int32_t curentOffset;
    TilingByCore(m_, curentNum, curentOffset, 1);

    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    // peermem 中每个 token 的 stride（prefetch 模式下含 weight，需用它计算偏移）
    uint32_t peermemTokenStride = mxQuantTokenScaleAlignBytes_;
    // 实际搬运量只需 token+scale，不含 weight（shared expert 不走 weight 前移路径）
    uint32_t copyInNum =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));

    GlobalTensor<ActivationType> srcGlobalTensor;
    GlobalTensor<ActivationType> dataDstGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleDstGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(params_.peermemInfo.quantTokenScalePtr));
    dataDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params_.workspaceInfo.sharedExpertInputDataPtr));
    scaleDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleOutType *>(params_.workspaceInfo.sharedExpertInputScalePtr));
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < curentNum; index++) {
        int32_t tokenIdx = curentOffset + index;
        uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIdx) * static_cast<uint64_t>(peermemTokenStride);
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto copyTmpTensor = (index % DOUBLE_BUFFER == 0) ? xOutTensor1_ : xOutTensor2_;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopy(copyTmpTensor, srcGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);

        LocalTensor<QuantScaleOutType> bufScale =
            copyTmpTensor[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(dataDstGlobalTensor[tokenIdx * widthA], copyTmpTensor,
                    {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleDstGlobalTensor[tokenIdx * widthAScale], bufScale,
                    {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
}

// ===============================================================
// GroupMatmulWithSwigluQuant：按实现路径分发到 A8W4 或 generic GMM1。
// IsShared=true 时跳过 dispatch flag 等待，供共享专家使用。
// epilogueOp 按 IsShared 选择成员实例（epilogueOp_ / sharedEpilogueOp_）。
// ===============================================================
template <TemplateMegaMoeTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::GroupMatmulWithSwigluQuant(const GMMAddrInfo &gmmAddrInfo,
                                                                                    const ExpertLoopState &state,
                                                                                    uint32_t expertIdx,
                                                                                    int32_t &vecSetSyncCom)
{
    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(state.baseOffset),
            Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
            Get<IDX_FLAG_OFFSET>(state.baseOffset),
            0L,
            0L,
            0L};
        if constexpr (IsShared) {
            sharedEpilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        } else {
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (IsShared) {
        // 共享专家不参与权重前移，走原 UB ping-pong + 硬同步路径
        if constexpr (ENABLE_A8W4) {
            MegaMoeImpl::GroupMatmulSwigluQuantA8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                                                    QuantScaleOutType, GMM1_TILE_M, MegaMoeImpl::L1_TILE_M_256, false,
                                                    IsShared>(sharedEpilogueOp_, params_, state.problemShape,
                                                              gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                                                              state.expertBeforeCnt, expertIdx);
        } else {
            if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
                params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
                MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
                                                    QuantScaleOutType, QuantScaleOutType, true, GMM1_TILE_M,
                                                    MegaMoeImpl::L1_TILE_M_256, false, IsShared, GMM1_INTERLEAVED>(
                    sharedEpilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            } else {
                MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
                                                    QuantScaleOutType, QuantScaleOutType, false, GMM1_TILE_M,
                                                    MegaMoeImpl::L1_TILE_M_256, false, IsShared, GMM1_INTERLEAVED>(
                    sharedEpilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            }
        }
    } else {
        // MoE 专家走 prefetch 路径
        if constexpr (ENABLE_A8W4) {
            MegaMoeImpl::GroupMatmulSwigluQuantA8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                                                    QuantScaleOutType, GMM1_TILE_M, EPILOGUE_TILE_M,
                                                    TopkWeightsPrefetch, IsShared>(
                epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                state.expertBeforeCnt, expertIdx);
        } else {
            if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
                params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
                // NZ format (A8W8_NZ / A4W4_NZ): isWeightNZ=true, EpilogueElementA 由 SwigluQuantOutType
                // 自动处理类型提升
                MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
                                                    QuantScaleOutType, QuantScaleOutType, true, GMM1_TILE_M,
                                                    EPILOGUE_TILE_M, TopkWeightsPrefetch, IsShared,
                                                    GMM1_INTERLEAVED>(
                    epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            } else {
                // Generic: fp8/fp4 activation × fp8/fp4 weight in ND format (includes A4W4 ND)
                MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
                                                    QuantScaleOutType, QuantScaleOutType, false, GMM1_TILE_M,
                                                    EPILOGUE_TILE_M, TopkWeightsPrefetch, IsShared,
                                                    GMM1_INTERLEAVED>(
                    epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            }
        }
    }
}

// ===============================================================
// GroupMatmulWithCombine：先按实现路径分发，再按 combine 模式分发。
// IsShared=true 时跳过 swiglu flag 等待和 Combine 后处理，供共享专家使用。
// ===============================================================
template <TemplateMegaMoeTypeClass>
template <bool IsShared>
__aicore__ inline void
MegaMoe<TemplateMegaMoeTypeFunc>::GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                         uint32_t expertIdx, int32_t &vecSetSyncCom)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode, SwigluQuantOutType, Weight1Type, bfloat16_t,
                                             QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch,
                                             IsShared, false, GMM1_INTERLEAVED>(
            params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom, state.expertBeforeCnt,
            gmm2PingPongIdx_);
    } else {
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                      QuantScaleOutType, true, false, GMM1_TILE_M, TopkWeightsPrefetch, IsShared,
                                      GMM1_INTERLEAVED>(
                params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom, state.expertBeforeCnt,
                gmm2PingPongIdx_);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                      QuantScaleOutType, false, false, GMM1_TILE_M, TopkWeightsPrefetch, IsShared,
                                      GMM1_INTERLEAVED>(
                params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom, state.expertBeforeCnt,
                gmm2PingPongIdx_);
        }
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV && !IsShared) {
        ProcessCombine(gmmAddrInfo, state, expertIdx);
    }
}

// Rolling MTE no-quant schedule:
//   AIC  : GMM1(B) -> GMM2(B) direct GM -> GMM1(B+1)
//   AIV0 : SwiGLU(B) -> SwiGLU(B+1), never enters combine/ready
//   AIV1 : fully prefetch Dispatch(B+1), then Combine(B)
// No batch-wide barrier is used; per-wave progress is carried only by the
// existing dispatch/SwiGLU dependencies and the new GMM2 ready fan-in.
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessGmm2GmWaves(
    const DispatchBufferConfig &bufferConfig, int32_t &vecSetSyncCom)
{
    GMMAddrInfo dispatchAddrInfo;
    GMMAddrInfo gmm1AddrInfo;
    GMMAddrInfo gmm2AddrInfo;
    GMMAddrInfo combineAddrInfo;
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ExpertLoopState dispatchState{initShape, initOffset, 0};
    ExpertLoopState gmm1State{initShape, initOffset, 0};
    ExpertLoopState gmm2State{initShape, initOffset, 0};
    ExpertLoopState combinePreloadState{initShape, initOffset, 0};
    ExpertLoopState combineState{initShape, initOffset, 0};

    const uint32_t expertsPerWave = expertsPerBatch_;

    uint32_t nextDispatchExpert = 0;
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1 && moeExpertPerRank_ > 0U) {
            DispatchExpert(dispatchState, dispatchAddrInfo, 0U, bufferConfig);
            nextDispatchExpert = 1U;
        }
    }

    uint32_t batchIdx = 0;
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
    bool traceGmm1B1ReadyRecorded = false;
#endif
    for (uint32_t batchBegin = 0; batchBegin < moeExpertPerRank_; batchBegin += expertsPerWave, ++batchIdx) {
        uint32_t batchEnd = batchBegin + expertsPerWave < moeExpertPerRank_ ?
                                batchBegin + expertsPerWave :
                                moeExpertPerRank_;

#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
        if (batchIdx == 1U) {
            if constexpr (g_coreType == AIV) {
                if (subBlockIdx_ == 0U && aivCoreIdx_ / 2U == 0U) {
                    RecordGmm2WaveTimeline(6U); // AIV0 SwiGLU(B+1) control entry
                }
            }
        }
#endif

        // First wave is one-expert-ahead; later waves were fully prefetched by
        // AIV1 during the preceding wave.
        for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
            if constexpr (g_coreType == AIV) {
                if (subBlockIdx_ == 1 && nextDispatchExpert < batchEnd) {
                    DispatchExpert(dispatchState, dispatchAddrInfo, nextDispatchExpert, bufferConfig);
                    ++nextDispatchExpert;
                }
            }

            uint64_t sendCnt = 0;
            if constexpr (g_coreType == AIV) {
                if (subBlockIdx_ == 1) {
                    sendCnt = ReadExpertTokenCount(expertIdx);
                }
            }
            if (!UpdateGroupParams<AddrUpdateMode::GMM1>(gmm1State, expertIdx, sendCnt)) {
                continue;
            }
            UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmm1AddrInfo, gmm1State);
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
            if constexpr (g_coreType == AIC) {
                if (batchIdx == 1U && blockIdx_ == 0U && !traceGmm1B1ReadyRecorded) {
                    // Mirror the first GMM1 wave wait in diagnostic mode so
                    // slot 2 means that this expert's dispatch data is ready,
                    // rather than merely that AIC0 entered the outer loop.
                    uint32_t mActual = static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape));
                    uint32_t targetRows = mActual < L1_TILE_M_256 ? mActual : L1_TILE_M_256;
                    while (targetRows != AscendC::ReadGmByPassDCache(gmm1AddrInfo.dispatchToGmm1Flag)) {
                        int64_t waitStart = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - waitStart < 100) {
                        }
                    }
                    RecordGmm2WaveTimeline(2U); // AIC0 GMM1(B+1) first dispatch wave ready
                    traceGmm1B1ReadyRecorded = true;
                }
            }
#endif
            GroupMatmulWithSwigluQuant(gmm1AddrInfo, gmm1State, expertIdx, vecSetSyncCom);
        }

#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
        if (batchIdx == 1U) {
            if constexpr (g_coreType == AIC) {
                if (blockIdx_ == 0U) {
                    RecordGmm2WaveTimeline(3U); // AIC0 GMM1(B+1) wave end
                }
            } else {
                if (subBlockIdx_ == 0U && aivCoreIdx_ / 2U == 0U) {
                    RecordGmm2WaveTimeline(7U); // AIV0 SwiGLU(B+1) end
                }
            }
        }
#endif

        // Full next-wave prefetch is complete before this AIV1 can enter
        // Combine(B).  Existing dispatch helpers drain their own six buffers.
        uint32_t nextBatchEnd = batchEnd + expertsPerWave < moeExpertPerRank_ ?
                                    batchEnd + expertsPerWave :
                                    moeExpertPerRank_;
        if constexpr (g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
                if (batchIdx == 0U && nextBatchEnd > batchEnd && aivCoreIdx_ / 2U == 0U) {
                    RecordGmm2WaveTimeline(0U); // AIV1 Dispatch(B+1) begin
                }
#endif
                while (nextDispatchExpert < nextBatchEnd) {
                    DispatchExpert(dispatchState, dispatchAddrInfo, nextDispatchExpert, bufferConfig);
                    ++nextDispatchExpert;
                }
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
                if (batchIdx == 0U && nextBatchEnd > batchEnd && aivCoreIdx_ / 2U == 0U) {
                    RecordGmm2WaveTimeline(1U); // AIV1 Dispatch(B+1) end
                }
#endif
            }
        }

        // AIC computes every expert in the wave.  AIV0/AIV1 only consume the
        // same virtual scheduler rotation in the direct-GM GroupMatmul2 path.
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
        if constexpr (g_coreType == AIC) {
            if (batchIdx == 0U && blockIdx_ == 0U) {
                RecordGmm2WaveTimeline(4U); // AIC0 GMM2(B) control entry
            }
        }
#endif
        for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
            if (!UpdateGroupParams<AddrUpdateMode::GMM2>(gmm2State, expertIdx)) {
                continue;
            }
            UpdateGlobalBuffer<AddrUpdateMode::GMM2>(gmm2AddrInfo, gmm2State);
            GroupMatmulWithCombine(gmm2AddrInfo, gmm2State, expertIdx, vecSetSyncCom);
            if constexpr (MEGA_MOE_GMM2_READY_MODE == MEGA_MOE_GMM2_READY_PER_EXPERT) {
                PublishGmm2NoQuantReady(expertIdx);
            }
        }
        if constexpr (MEGA_MOE_GMM2_READY_MODE == MEGA_MOE_GMM2_READY_PER_BATCH) {
            PublishGmm2NoQuantReady(batchIdx);
        }
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
        if constexpr (g_coreType == AIC) {
            if (batchIdx == 0U && blockIdx_ == 0U) {
                RecordGmm2WaveTimeline(5U); // GMM2(B) end, after FIX/Atomic publish
            }
        }
#endif

        // This block is AIV1-only.  In particular AIV0 never calls the wait
        // helper and never creates combine UB/event state.
        if constexpr (g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                if constexpr (MEGA_MOE_GMM2_FINAL_BATCH_COMBINE_ONLY != 0 &&
                              MEGA_MOE_GMM2_READY_MODE == MEGA_MOE_GMM2_READY_PER_BATCH) {
                    if (batchEnd != expertPerRank_) {
                        // Preserve the original control dependency so AIV1
                        // cannot run ahead of this wave's completed GMM2.
                        WaitGmm2NoQuantReady(batchIdx);
                        continue;
                    }
                }

                InitCombineNoQuantRowRing();
                uint32_t rowSequence = 0U;
                MegaMoeImpl::TokenRange waveOwned;
                uint32_t waveTokenBegin = 0U;
                GM_ADDR waveGmm2OutGlobal = nullptr;
                uint32_t wavePreloadedChunkCount = 0U;

                if constexpr (MEGA_MOE_GMM2_READY_MODE == MEGA_MOE_GMM2_READY_PER_BATCH) {
                    // Meta-info routing data was finalized by Dispatch(B), so it
                    // can be prefetched while AIC is still finishing GMM2(B).
                    // GMM2 rows and meta-info use the same expert-major prefix
                    // sum, so the whole wave is one contiguous row stream.
                    uint32_t waveTokenCount = 0U;
                    for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
                        bool hasTokens = UpdateGroupParams<AddrUpdateMode::GMM2>(combinePreloadState, expertIdx);
                        if (expertIdx == batchBegin) {
                            waveTokenBegin = combinePreloadState.expertBeforeCnt;
                        }
                        if (hasTokens) {
                            waveTokenCount +=
                                static_cast<uint32_t>(Get<M_VALUE>(combinePreloadState.problemShape));
                        }
                    }

                    waveOwned = GetCombineNoQuantOwnedRange(waveTokenCount);
                    if (waveOwned.count != 0U) {
                        waveGmm2OutGlobal = params_.workspaceInfo.gmm2MmadResPtr +
                                           static_cast<uint64_t>(waveTokenBegin) * k_ * sizeof(bfloat16_t);
                        wavePreloadedChunkCount = waveOwned.count < COMBINE_META_INFO_TOKEN_CAPACITY ?
                                                      waveOwned.count :
                                                      COMBINE_META_INFO_TOKEN_CAPACITY;
                        PreloadCombineNoQuantMetaInfo(
                            static_cast<uint64_t>(waveTokenBegin) + waveOwned.start, wavePreloadedChunkCount, 0U);
                        SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
                    }
                    WaitGmm2NoQuantReady(batchIdx);
                    if (wavePreloadedChunkCount != 0U) {
                        WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
                    }
                }
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
                if (batchIdx == 0U && aivCoreIdx_ / 2U == 0U) {
                    RecordGmm2WaveTimeline(8U); // AIV1 Combine(B) control entry
                }
#endif
                if constexpr (MEGA_MOE_GMM2_READY_MODE == MEGA_MOE_GMM2_READY_PER_BATCH) {
                    // The common target fits in one meta-info preload. Retain a
                    // bounded fallback for larger waves while preserving one
                    // balanced, contiguous ownership interval per AIV1.
                    uint32_t processed = 0U;
                    if (wavePreloadedChunkCount != 0U) {
                        ProcessCombineNoQuantGm(waveGmm2OutGlobal, waveOwned.start, wavePreloadedChunkCount, 0U,
                                               rowSequence);
                        processed = wavePreloadedChunkCount;
                    }
                    while (processed < waveOwned.count) {
                        uint32_t chunkCount = waveOwned.count - processed < COMBINE_META_INFO_TOKEN_CAPACITY ?
                                                  waveOwned.count - processed :
                                                  COMBINE_META_INFO_TOKEN_CAPACITY;
                        PreloadCombineNoQuantMetaInfo(
                            static_cast<uint64_t>(waveTokenBegin) + waveOwned.start + processed, chunkCount, 0U);
                        SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
                        WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
                        ProcessCombineNoQuantGm(waveGmm2OutGlobal, waveOwned.start + processed, chunkCount, 0U,
                                               rowSequence);
                        processed += chunkCount;
                    }
                } else {
                    for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
                        if (!UpdateGroupParams<AddrUpdateMode::GMM2>(combineState, expertIdx)) {
                            continue;
                        }
                        UpdateGlobalBuffer<AddrUpdateMode::GMM2>(combineAddrInfo, combineState);
                        uint32_t mExpert = static_cast<uint32_t>(Get<M_VALUE>(combineState.problemShape));
                        MegaMoeImpl::TokenRange owned = GetCombineNoQuantOwnedRange(mExpert);
                        uint32_t processed = 0U;
                        while (processed < owned.count) {
                            uint32_t chunkCount = owned.count - processed < COMBINE_META_INFO_TOKEN_CAPACITY ?
                                                      owned.count - processed :
                                                      COMBINE_META_INFO_TOKEN_CAPACITY;
                            PreloadCombineNoQuantMetaInfo(
                                static_cast<uint64_t>(combineState.expertBeforeCnt) + owned.start + processed,
                                chunkCount, 0U);
                            SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
                            if (processed == 0U) {
                                WaitGmm2NoQuantReady(expertIdx);
                            }
                            WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
                            ProcessCombineNoQuantGm(combineAddrInfo.gmm2OutGlobal, owned.start + processed, chunkCount,
                                                   0U, rowSequence);
                            processed += chunkCount;
                        }
                    }
                }
                // A slot can remain in flight across expert boundaries, but
                // dispatch for the following wave may reuse the same UB/events.
                DrainCombineNoQuantRowRing();
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
                if (batchIdx == 0U && aivCoreIdx_ / 2U == 0U) {
                    RecordGmm2WaveTimeline(9U); // AIV1 Combine(B) end
                }
#endif
            }
        }
    }

    // The non-prefetch path still has one outstanding AIC->AIV0 UB consumer
    // acknowledgement.  The prefetch path uses per-tile GM status instead.
    if constexpr (!TopkWeightsPrefetch) {
        EndSync<GMM1_UB_PINGPONG>(vecSetSyncCom, gmm1PingPongIdx_);
    }
    vecSetSyncCom = 0;
    gmm1PingPongIdx_ = 0;
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
#if MEGA_MOE_GMM2_WAVE_TIMELINE_DIAG != 0
            if (expertPerRank_ >= 16U) {
                if (aivCoreIdx_ / 2U == 0U) {
                    WriteGmm2WaveTimelineMetadata(expertsPerWave);
                }
            } else {
                ExpertTokenNumCopyOut();
            }
#else
            ExpertTokenNumCopyOut();
#endif
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm1(const TupleShape &initShape,
                                                                                 const BlockOffset &initOffset)
{
    sharedEpilogueOp_.Init({params_.workspaceInfo.sharedExpertSwigluDataPtr,
                            params_.workspaceInfo.sharedExpertSwigluScalePtr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, params_.tilingData->clampLimit});

    GMMAddrInfo sharedGmm1AddrInfo;
    ExpertLoopState sharedGmm1State{initShape, initOffset, 0};
    int32_t vecSetSyncCom = 0;
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm1State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM1>(sharedGmm1AddrInfo, sharedGmm1State);
        GroupMatmulWithSwigluQuant<true>(sharedGmm1AddrInfo, sharedGmm1State, sharedIdx, vecSetSyncCom);
    }
    EndSync<GMM1_UB_PINGPONG>(vecSetSyncCom, gmm1PingPongIdx_);
    vecSetSyncCom = 0;
    gmm1PingPongIdx_ = 0;
    startBlockIdx_ = 0; // 共享专家GMM1修改了startBlockIdx_，重置给GMM1使用
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm2(const TupleShape &initShape,
                                                                                 const BlockOffset &initOffset)
{
    GMMAddrInfo sharedGmm2AddrInfo;
    ExpertLoopState sharedGmm2State{initShape, initOffset, 0};
    int32_t vecSetSyncCom = 0;
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm2State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM2>(sharedGmm2AddrInfo, sharedGmm2State);
        GroupMatmulWithCombine<true>(sharedGmm2AddrInfo, sharedGmm2State, sharedIdx, vecSetSyncCom);
    }
    SyncAll<false>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendMaskBufferConfig sendMaskBufferConfig = SendAndQuantBuffInit();

    // Phase 1: 量化 + 共享专家输入拆分 + mask/reset (AIV)
    QuantProcessInRank();              // 对本卡token的量化
    SendMaskCal(sendMaskBufferConfig); // 源卡按所有全局专家算 mask 并推送到目标专家卡
    ResetFlagList();                   // 清理workSpace空间上的flag位
    if (sharedExpertNum_ > 0) {
        SharedExpertCopyInput();
    }
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>(); // aic需要等待flag位reset清理完成
    if constexpr (ENABLE_GMM2_GM_REBUILD && MEGA_MOE_GMM2_GM_READY_SMOKE_TEST != 0) {
        RunGmm2NoQuantReadySmokeTest();
        SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
        return;
    }

    // Phase 1.5: 共享专家 GMM1+SwiGLU (前移, 在 MoE 之前执行, 复用 MoE 函数)
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm1(initShape, initOffset);
        SyncAll<false>();
    }

    CrossRankSyncInWorldSize();

    // 2.本卡专家接收数据dispatch & GroupMatmul1 & SwigluQuant
    DispatchBufferConfig dispatchBufferConfig = DispatchBuffInit();
    int32_t vecSetSyncCom = 0;

    if constexpr (ENABLE_GMM2_GM_REBUILD) {
        ProcessGmm2GmWaves(dispatchBufferConfig, vecSetSyncCom);
    }

    if constexpr (!ENABLE_GMM2_GM_REBUILD) {
        GMMAddrInfo dispatchAddrInfo;
        GMMAddrInfo gmm1AddrInfo;
        ExpertLoopState dispatchState{initShape, initOffset, 0};
        ExpertLoopState gmm1State{initShape, initOffset, 0};

        // Dispatch-prefetch count forwarding（无成员变量耦合）：
        //   SendCntCal 将 expert token 数写入 nextSendCnt；
        //   循环顶部 nextSendCnt → curSendCnt 显式转发；
        //   GMM1 consumer 始终读 curSendCnt。
        uint64_t curSendCnt = 0;  // 当前 expert 的 sendCnt（GMM1 consumer 使用）
        uint64_t nextSendCnt = 0; // 下一 expert 的 sendCnt（dispatch prefetch 算出）

        // 预调度 expert 0。
        if constexpr (g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                SendCntCal(0, nextSendCnt);
                if (UpdateGroupParams<AddrUpdateMode::GMM1>(dispatchState, 0, nextSendCnt)) {
                    UpdateGlobalBuffer<AddrUpdateMode::GMM1>(dispatchAddrInfo, dispatchState);
                    MetaInfoCalAndDispatch(dispatchAddrInfo, 0, dispatchBufferConfig);
                }
            }
        }

        for (uint32_t localExpertId = 0; localExpertId < moeExpertPerRank_; localExpertId++) {
            curSendCnt = nextSendCnt; // forward: dispatch(e) → GMM1(e)

            // Prefetch dispatch expert e+1，与当前 GMM1 consumer expert e 并发。
            if constexpr (g_coreType == AIV) {
                if (subBlockIdx_ == 1 && localExpertId + 1 < moeExpertPerRank_) {
                    SendCntCal(localExpertId + 1, nextSendCnt);
                    if (UpdateGroupParams<AddrUpdateMode::GMM1>(dispatchState, localExpertId + 1, nextSendCnt)) {
                        UpdateGlobalBuffer<AddrUpdateMode::GMM1>(dispatchAddrInfo, dispatchState);
                        MetaInfoCalAndDispatch(dispatchAddrInfo, localExpertId + 1, dispatchBufferConfig);
                    }
                }
            }

            // GMM1 consumer 消费 expert e。
            if (!UpdateGroupParams<AddrUpdateMode::GMM1>(gmm1State, localExpertId, curSendCnt)) {
                continue;
            }
            UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmm1AddrInfo, gmm1State);
            GroupMatmulWithSwigluQuant(gmm1AddrInfo, gmm1State, localExpertId, vecSetSyncCom);
        }
        // prefetch 路径：epilogue 完成所有 expert 后写 allDone，AIC 轮询后继续；
        // 非 prefetch 路径沿用 AIC/AIV 的硬同步。
        if constexpr (TopkWeightsPrefetch) {
            if constexpr (g_coreType == AIV) {
                constexpr uint32_t epilogueSubIdx = ENABLE_A8W4 ? 1 : 0;
                if (subBlockIdx_ == epilogueSubIdx) {
                    int32_t allDoneTag = static_cast<int32_t>(expertPerRank_ + 1);
                    __gm__ int32_t *allDoneAddr =
                        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr) +
                        static_cast<int64_t>(expertPerRank_) * params_.tilingData->maxTilesPerExpert;
                    AscendC::WriteGmByPassDCache(allDoneAddr, allDoneTag);
                }
            } else {
                int32_t allDoneTag = static_cast<int32_t>(expertPerRank_ + 1);
                __gm__ int32_t *allDoneAddr =
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr) +
                    static_cast<int64_t>(expertPerRank_) * params_.tilingData->maxTilesPerExpert;
                while (AscendC::ReadGmByPassDCache(allDoneAddr) != allDoneTag) {
                    int64_t st = AscendC::GetSystemCycle();
                    while (AscendC::GetSystemCycle() - st < 100) {
                    }
                }
            }
        } else {
            EndSync<GMM1_UB_PINGPONG>(vecSetSyncCom, gmm1PingPongIdx_);
        }
        vecSetSyncCom = 0;
        gmm1PingPongIdx_ = 0;
        if constexpr (g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                ExpertTokenNumCopyOut();
            }
        }
        // prefetch 路径 AIV1 dispatch metaInfo 与后续 AIV0 消费之间需要同步。
        if constexpr (TopkWeightsPrefetch) {
            SyncAll<false>();
        }

        // 3. 本卡专家接收数据GroupMatmul2 & Combine
        vecSetSyncCom = 0;
        GMMAddrInfo gmm2AddrInfo;
        ExpertLoopState gmm2State{initShape, initOffset, 0};
        InitCombineBuffers();

        for (uint32_t expertIdx = 0; expertIdx < moeExpertPerRank_; expertIdx++) {
            if (!UpdateGroupParams<AddrUpdateMode::GMM2>(gmm2State, expertIdx)) {
                continue;
            }
            UpdateGlobalBuffer<AddrUpdateMode::GMM2>(gmm2AddrInfo, gmm2State);
            GroupMatmulWithCombine(gmm2AddrInfo, gmm2State, expertIdx, vecSetSyncCom);
        }
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            EndGMM2Sync(vecSetSyncCom, gmm2PingPongIdx_);
        }
    }

    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    // 3.5: 共享专家 GMM2 (MoE GMM2 之后, 复用 MoE 函数)
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm2(initShape, initOffset);
    }

    // 4. 本卡数据Unpermute
    if constexpr (g_coreType == AIV) {
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace MegaMoeImpl
#endif
