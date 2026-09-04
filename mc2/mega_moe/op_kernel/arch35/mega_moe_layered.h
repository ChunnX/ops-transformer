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
 * \file mega_moe_layered.h
 * \brief
 */

#ifndef MEGA_MOE_LAYERED_H
#define MEGA_MOE_LAYERED_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MEGA_MOE_LAYERED_KERNEL
#endif

#endif

#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "common/mega_moe_types.h"
#include "common/mega_moe_workspace.h"
#include "common/mega_moe_utils.h"
#include "blaze/epilogue/block_epilogue_activation_mx_quant.h"
#include "stage/mega_moe_gmm1_activation.h"
#include "stage/mega_moe_gmm2_combine.h"
#include "common/mega_moe_mxfp8_utils.h"
#if __has_include("../../common/quantize_functions.h")
#include "../../common/quantize_functions.h"
#else
#include "../../../common/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr int64_t HALF_TO_FP32 = 2U;
constexpr int64_t DEQUANT_SCALE_EXPAND = 2U;

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t>;

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeLayeredTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegaMoeLayeredTypeFunc XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, \
                                       TopkWeightsPrefetch, IsGmm1Interleaved

template <TemplateMegaMoeLayeredTypeClass>
class MegaMoeLayered {
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
    __aicore__ inline MegaMoeLayered(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t GetChannelOwnerBlock(uint32_t rank) const;
    __aicore__ inline bool IsChannelOwner(uint32_t rank) const;
    __aicore__ inline uint32_t CalcTargetWaveCount() const;
    __aicore__ inline uint32_t CalcFirstWaveExpertCount(uint32_t targetWaveCount) const;
    __aicore__ inline uint32_t CalcSteadyWaveExpertCount(uint32_t firstWaveExpertCount,
                                                        uint32_t targetWaveCount) const;
    __aicore__ inline void ProcessMoeExpertWave(const TupleShape &initShape, const BlockOffset &initOffset,
                                               int32_t &gmTileSequence);
    __aicore__ inline void PrepareDispatch(uint32_t firstWaveExpertCount);
    __aicore__ inline void RunGmm1ActivationForExpert(
        ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, int32_t &vecSetSyncCom, int32_t &gmTileSequence,
        uint32_t expertIdx);
    __aicore__ inline void RunGmm2ForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx);
    __aicore__ inline void PublishOrWaitGmm1AllDone();
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void DispatchPrepareBuffInit();
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void UnpermuteBuffInit();
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void ResetGmm2CombineSyncCounters();
    __aicore__ inline void SendMaskCal();
    __aicore__ inline void PublishExpertTokenCount(uint32_t expertIdx);
    __aicore__ inline void ReceiveDispatchExpertRange(uint32_t expertBegin, uint32_t expertEnd);
    __aicore__ inline void ReceiveRemoteDispatchBatch(uint32_t relayRank, uint32_t batchTokenCount);
    __aicore__ inline void PublishDispatchRows(uint32_t expertIdx, int32_t globalRowStart, int32_t rowCount);
    template <AddrUpdateMode Mode>
    __aicore__ inline bool UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx);
    __aicore__ inline bool UpdateSharedGroupParams(ExpertLoopState &state, uint32_t expertIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateSharedGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void Unpermute();
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombineExpertRange(ExpertLoopState waveBeginState,
                                                     uint32_t expertBegin, uint32_t expertEnd);
    __aicore__ inline void CrossRankSyncInWorldSize(bool reuseSendMaskChannelLock);
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyLocalDispatchTokens(int32_t dstRow, int32_t srcRank,
                                                  int32_t tokenOffset, int32_t tokenCount);
    __aicore__ inline void ResetContiguousGm(GM_ADDR dstAddr, uint64_t sizeBytes);
    __aicore__ inline void ResetDispatchState();
    __aicore__ inline void SendDispatchRelayQueues();
    __aicore__ inline void BuildDispatchRelayQueues();
    __aicore__ inline void AppendTokenToDispatchRelayQueues(GlobalTensor<int32_t> &topkIdsGlobal,
                                                            uint32_t tokenIdx);
    __aicore__ inline void AppendDispatchRelayData(uint32_t targetServer, uint32_t relayRank,
                                                  HcommBatchHandle &batchHandle, int32_t slot);
    __aicore__ inline void AppendDispatchRelayFlag(uint32_t targetServer, uint32_t relayRank,
                                                  HcommBatchHandle &batchHandle, int32_t slot, bool firstFlag);
    __aicore__ inline void CommitDispatchRelayBatch(uint32_t targetServer, uint32_t relayRank,
                                                   HcommBatchHandle &batchHandle, int32_t batchStart,
                                                   uint32_t batchTokenCount);
    __aicore__ inline void LoadTokenFromLocalRelay(uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx,
                                                   uint32_t copyInNum);
    __aicore__ inline void CopyTokensFromLocalRelay(int32_t rowDstOffsetInCore, uint32_t srcServer, int32_t copyNum,
                                                    int64_t widthA, int64_t widthAScale, uint32_t copyInNum);
    __aicore__ inline void QuantizeLocalTokensToRelay();
    __aicore__ inline void SetLocalRelayReadyFlags();
    __aicore__ inline uint64_t DispatchRelayQueueServerOffset(uint32_t targetServer) const;
    __aicore__ inline uint64_t RelayTokenOffset(uint32_t sourceServer, uint32_t tokenId) const;
    __aicore__ inline uint64_t RelayFlagOffset(uint32_t sourceServer, uint32_t tokenId) const;
    __aicore__ inline void SharedExpertCopyInput();
    __aicore__ inline void ProcessSharedExpertGmm1(const TupleShape &initShape, const BlockOffset &initOffset,
                                                   int32_t &gmTileSequence);
    __aicore__ inline void ProcessSharedExpertGmm2(const TupleShape &initShape, const BlockOffset &initOffset);
    __aicore__ inline void UnpermuteSharedExpert(int32_t tokenIdx);
    __aicore__ inline void LoadTopkWeightsToUb(const LocalTensor<ActivationType> &xOutTensor, int32_t currentOffset,
                                               int32_t index, TEventID event);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithActivationQuant(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                          uint32_t expertIdx, int32_t &vecSetSyncCom,
                                                          int32_t &gmTileSequence);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline bool GetCombineRankRange(uint32_t expertIdx, uint32_t dstRank,
                                               uint32_t &rowStart, uint32_t &tokenCount);
    __aicore__ inline void ProcessCombineRank(const GMMAddrInfo &gmmAddrInfo,
                                              const ExpertLoopState &gmm2State, uint32_t expertIdx,
                                              CombineImpl::LayeredCombineBatchState &batchState);
    __aicore__ inline void LockOwnedRemoteChannels();
    __aicore__ inline void DrainAndUnlockOwnedRemoteChannels();

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    __gm__ int32_t *gmmToEpilogueFlag_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    LocalTensor<uint8_t> hcommBatchWqeTensor_;
    Params params_{};

    GlobalTensor<int32_t> activationToGmm2FlagGm_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> metaInfoGlobalTensor_;
    GlobalTensor<int32_t> expertRevNumsGlobalTensor_;
    // A8W4 路径下 RunGmm1A8W4 会覆盖 V1 UB，导致 UB 上跨 expert 的状态
    // 无法保持。cumsumInfoGlobalTensor_ 作为 cumsum 数据的 GM 持久备份：
    // PublishExpertTokenCount 中 Load → 计算 → Store；Dispatch 接收和 CopyOut 从 GM 恢复。
    GlobalTensor<int32_t> cumsumInfoGlobalTensor_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t rankNumPerServer_ = 0;
    uint32_t serverNum_ = 0;
    uint32_t serverId_ = 0;
    uint32_t rankIdInServer_ = 0;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint16_t gmm1PingPongIdx_ = 0;
    uint32_t startBlockIdx_ = 0;
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t maxWavesPerExpert_ = 0;
    // A5 混合核启动约定：GetBlockNum() 等于 AIC 数，每个逻辑核对应两个 AIV 子核。
    // 因此 Host 侧仍按 blockAivNum / 2 计算逻辑核数。
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
    uint64_t sendTotalNum_ = 0;      // 本 Rank 真实 route 数：bs * topK
    uint64_t maskRouteCapacity_ = 0; // 对称 mask 容量：numMaxTokensPerRank * topK
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;      // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint32_t roundSendTotalNum_ = 0; // 分轮次：每轮处理的 token*topK 数量（256 对齐保证 GM 写 32B 对齐）
    uint32_t roundCompareCount_ = 0;          // 分轮次：每轮 CompareScalar 的元素数
    uint32_t roundMaskAlignSize_ = 0;         // 分轮次：每轮部分 mask 的字节大小（32B 对齐）
    uint32_t roundMaskSlotSize_ = 0;          // 分轮次：每轮部分 [mask|count] 槽位字节大小
    uint32_t totalRounds_ = 0;                // 分轮次：总轮数 = CeilDiv(sendTotalNum, roundSendTotalNum)
    uint32_t dispatchRoundSendTotalNum_ = 0;  // dispatch分轮次：每轮处理的 token*topK 数量（256 对齐）
    uint32_t dispatchTotalRounds_ = 0;        // dispatch分轮次：容量上界总轮数
    uint32_t dispatchRoundMaskAlignSize_ = 0; // dispatch分轮次：每轮mask的字节大小（32B对齐）
    uint64_t maskWinOffset_ = 0;              // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t dispatchWinOffset_ = 0;          // peermemInfo dispatchReceivePtr 相对 URMA win 基址的偏移
    uint64_t dispatchFlagWinOffset_ = 0;      // peermemInfo dispatchFlagPtr 相对 URMA win 基址的偏移
    uint32_t relayRecordBytes_ = 0;
    uint64_t dispatchRelayQueueBytesPerServer_ = 0;
    uint64_t dispatchRelayFlagSnapshotBytesPerBlock_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）
    uint32_t topKWeightsChunkLen_ = 0;
    uint32_t topKWeightsTempAddr_ = 0;
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;

    static constexpr uint32_t A_ELEMS_PER_BYTE = PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = PackedElementTraits<Weight1Type>::ELEMENTS_PER_BYTE;
    // ENABLE_A8W4：A8W4 路径（FP8 激活 + FP4 权重），GMM1 使用 A8W4 前处理（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4：A4W4 路径（FP4 激活 + FP4 权重），GMM2 复用 A8W4 前处理。
    // A4W4 场景下 GMM1 走通用 A4W4、GMM2 走 A8W4，避免两段都使用 A4W4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    // 仅通用 A8W8 路径支持 gate/up 交织布局，FP4 配对路径保持专用布局和同步协议。
    static constexpr bool GMM1_INTERLEAVED = IsGmm1Interleaved && !ENABLE_A8W4 && !ENABLE_A4W4;
    static constexpr int32_t DISPATCH_COPY_BUFFER_COUNT = 5;
    static constexpr uint32_t MX_QUANT_TEMP_UB_BYTES = 2U * 1024U;
    static constexpr uint32_t DISPATCH_READY_FLAG_BATCH_TOKENS = 256U;
    static constexpr uint32_t DISPATCH_MAX_TOPK = 32U;
    static constexpr uint32_t SEND_MASK_UB_LIMIT = LAYERED_USABLE_UB_BYTES;
    // 二级读取在 TopkWeightsPrefetch 路径下每 token 需要 data/scale/weight 三条 WQE。
    // 64 KiB 可容纳 1024 条 WQE，足够一次提交完整的 256-token 批次（最多 768 条 WQE）。
    static_assert(LAYERED_HCOMM_BATCH_WQE_CAPACITY >= L1_TILE_M_256 * 3U,
                  "Layered Hcomm UB must hold 256 three-WQE token records");
    // Wave 通信需要同时保留批量描述符和其他搬运张量，因此动态 UB 预算止于批量区起点。
    static constexpr uint32_t COMM_WORK_UB_LIMIT = LAYERED_HCOMM_BATCH_UB_OFFSET;
    // 计算宏 Wave、一级发送批次和二级就绪段使用独立粒度。
    // 二级读取/ready 仍以 256 行为正确性边界；一级发送尽量用满 64 KiB WQE 区。
    static constexpr uint32_t LAYERED_FIRST_WAVE_ROWS = 1024U;
    static constexpr uint32_t LAYERED_LATENCY_WAVE_COUNT = 2U;
    static constexpr uint32_t LAYERED_BALANCED_WAVE_COUNT = 6U;
    static constexpr uint32_t LAYERED_THROUGHPUT_WAVE_COUNT = 4U;
    static constexpr uint32_t LAYERED_LATENCY_ROWS_PER_EXPERT = L1_TILE_M_256;
    static constexpr uint32_t LAYERED_THROUGHPUT_ROWS_PER_EXPERT = 2048U;
    static constexpr uint32_t LAYERED_FEW_EXPERT_THRESHOLD = 8U;
    // Dispatch 一级通信严格按“数据批次先提交、就绪标志批次后提交”发布。
    // 仅这一段关闭 CQE；Dispatch 二级读和 Combine 写均使用 Hcomm 默认 WQE 配置。
    static constexpr struct UrmaWqeEntry DISPATCH_DATA_WQE_CONFIG = {
        .odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
    static constexpr struct UrmaWqeEntry DISPATCH_FIRST_FLAG_WQE_CONFIG = {
        .odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
    static constexpr struct UrmaWqeEntry DISPATCH_FLAG_WQE_CONFIG = {
        .odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
    // 一级发送尽量吃满 64 KiB WQE 区；二级接收仍以 256 token 作为完整就绪边界。
    static constexpr uint32_t DISPATCH_RELAY_QUEUE_ENTRY_BYTES = ALIGN_32;
    static constexpr uint32_t DISPATCH_SEND_BATCH_TOKEN_CAPACITY = LAYERED_HCOMM_BATCH_WQE_CAPACITY;
    static constexpr uint32_t DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY = L1_TILE_M_256;
    static constexpr uint32_t DISPATCH_BATCH_DST_ROW = 4U;
    static constexpr uint32_t DISPATCH_BATCH_EXPERT_ID = 5U;
    static constexpr uint32_t DISPATCH_BATCH_SRC_SERVER = 6U;
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<ActivationType> copyTmpTensors_[DISPATCH_COPY_BUFFER_COUNT]; // 5 路软流水：占用 EVENT_ID1..EVENT_ID5。
    // 当前 256-token 中继就绪标志窗口，复用本地搬运缓冲起始地址。
    LocalTensor<uint64_t> relayFlagTensor_;
    LocalTensor<int32_t> metaInfoTensor_;
    LocalTensor<bfloat16_t> xInTensor1_;
    LocalTensor<bfloat16_t> xInTensor2_;
    LocalTensor<ActivationType> xOutTensor1_;
    LocalTensor<ActivationType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<uint64_t> relayReadyFillTensor_;
    LocalTensor<int32_t> resetTensor_;
    int32_t resetBatchElementCount_ = 0;
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER]; // SendMaskCal 源卡算 [mask|count] 的 ping-pong 缓冲
    LocalTensor<int32_t> sendGatherOutTensor_;           // SendMaskCal GatherMask 计 count 的废弃输出 scratch
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，ActivationQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using ActivationQuantOutType =
        typename std::conditional<(QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    // ActivationQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = PackedElementTraits<ActivationQuantOutType>::ELEMENTS_PER_BYTE;

    // GMM1 固定按 256 行切块；权重前移时后处理可按 128 行细分。
    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M = TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256;

    using BlockEpilogue =
        BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, EPILOGUE_TILE_M, L1_TILE_N,
                                       TopkWeightsPrefetch, GMM1_INTERLEAVED>;
    using SharedBlockEpilogue =
        BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, L1_TILE_M_256, L1_TILE_N, false,
                                       GMM1_INTERLEAVED>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
};

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GetChannelOwnerBlock(uint32_t rank) const
{
    // 通道归属在逻辑 AIC 域内计算，不使用物理 AIV 编号。
    // Dispatch 两级通信、Combine 写入和最终 Drain 共用此映射，且只有归属 block 的 AIV1 可访问通道。
    return rank % blockNum_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::IsChannelOwner(uint32_t rank) const
{
    if constexpr (g_coreType == AIC) {
        return false;
    }
    return subBlockIdx_ == 1U && blockIdx_ == GetChannelOwnerBlock(rank);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CalcTargetWaveCount() const
{//todo:这里需要改
    if (moeExpertPerRank_ <= 1U) {
        return moeExpertPerRank_;
    }
    uint64_t totalRows = static_cast<uint64_t>(m_) * static_cast<uint64_t>(topK_);
    if (totalRows <= static_cast<uint64_t>(LAYERED_FIRST_WAVE_ROWS)) {
        return 1U;
    }
    uint64_t estimatedRowsPerExpert =
        (totalRows + static_cast<uint64_t>(moeExpertPerRank_) - 1U) /
        static_cast<uint64_t>(moeExpertPerRank_);
    uint32_t targetWaveCount = LAYERED_BALANCED_WAVE_COUNT;
    if (estimatedRowsPerExpert <= LAYERED_LATENCY_ROWS_PER_EXPERT) {
        targetWaveCount = LAYERED_LATENCY_WAVE_COUNT;
    } else if (estimatedRowsPerExpert >= LAYERED_THROUGHPUT_ROWS_PER_EXPERT) {
        targetWaveCount = LAYERED_THROUGHPUT_WAVE_COUNT;
    } else if (moeExpertPerRank_ <= LAYERED_FEW_EXPERT_THRESHOLD) {
        targetWaveCount = LAYERED_LATENCY_WAVE_COUNT;
    }
    return targetWaveCount < moeExpertPerRank_ ? targetWaveCount : moeExpertPerRank_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CalcFirstWaveExpertCount(
    uint32_t targetWaveCount) const
{
    if (moeExpertPerRank_ == 0U || targetWaveCount == 0U) {
        return 0U;
    }
    if (targetWaveCount == 1U) {
        return moeExpertPerRank_;
    }
    uint64_t totalRows = static_cast<uint64_t>(m_) * static_cast<uint64_t>(topK_);
    uint64_t estimatedRowsPerExpert =
        (totalRows + static_cast<uint64_t>(moeExpertPerRank_) - 1U) /
        static_cast<uint64_t>(moeExpertPerRank_);
    if (estimatedRowsPerExpert == 0U) {
        estimatedRowsPerExpert = 1U;
    }
    uint32_t expertsByWaveBudget =
        (moeExpertPerRank_ + targetWaveCount - 1U) / targetWaveCount;
    uint32_t expertsByWarmupRows = static_cast<uint32_t>(
        (static_cast<uint64_t>(LAYERED_FIRST_WAVE_ROWS) + estimatedRowsPerExpert - 1U) /
        estimatedRowsPerExpert);
    uint32_t firstWaveExperts = expertsByWarmupRows < expertsByWaveBudget ?
        expertsByWarmupRows : expertsByWaveBudget;
    if (firstWaveExperts == 0U) {
        firstWaveExperts = 1U;
    }
    return firstWaveExperts < moeExpertPerRank_ ? firstWaveExperts : moeExpertPerRank_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CalcSteadyWaveExpertCount(
    uint32_t firstWaveExpertCount, uint32_t targetWaveCount) const
{
    if (firstWaveExpertCount >= moeExpertPerRank_ || targetWaveCount <= 1U) {
        return moeExpertPerRank_;
    }
    uint32_t remainingExperts = moeExpertPerRank_ - firstWaveExpertCount;
    uint32_t remainingWaves = targetWaveCount - 1U;
    uint32_t steadyWaveExperts =
        (remainingExperts + remainingWaves - 1U) / remainingWaves;
    return steadyWaveExperts == 0U ? 1U : steadyWaveExperts;
}

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    aicNum_ = tilingData->aicNum;
    topK_ = tilingData->topK;
    sendTotalNum_ = static_cast<uint64_t>(m_) * static_cast<uint64_t>(topK_);
    maskRouteCapacity_ =
        static_cast<uint64_t>(tilingData->numMaxTokensPerRank) * static_cast<uint64_t>(topK_);
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    maxOutputSize_ = tilingData->maxOutputSize;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(L1_TILE_M_256)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
    hiddenDim_ = tilingData->hiddenDim;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    rankNumPerServer_ = tilingData->rankNumPerServer;
    if (rankNumPerServer_ == 0U || rankNumPerServer_ > worldSize_) {
        rankNumPerServer_ = worldSize_;
    }
    serverNum_ = Ops::Base::CeilDiv(worldSize_, rankNumPerServer_);
    serverId_ = rankId_ / rankNumPerServer_;
    rankIdInServer_ = rankId_ % rankNumPerServer_;
    for (int i = 0; i < worldSize_; i++) {
        g_winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    // 保留 TensorList 入口；每个 expert 更新地址时再按 2D-list/3D-stacked 布局解析。
    params_.bGmAddr = weight1;
    params_.b2GmAddr = weight2;
    params_.bScaleGmAddr = weightScales1;
    params_.b2ScaleGmAddr = weightScales2;
    params_.sharedBGmAddr = sharedWeight1;
    params_.sharedB2GmAddr = sharedWeight2;
    params_.sharedBScaleGmAddr = sharedWeightScales1;
    params_.sharedB2ScaleGmAddr = sharedWeightScales2;
    params_.combineCommParams.hcomm = &hcomm_;

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(g_winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE, serverNum_);
    params_.tilingData = tilingData;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmToEpilogueFlag_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                             static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    }
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t *)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGlobalTensor_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.expertRevTokenNumsPtr);
    metaInfoGlobalTensor_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.metaInfoPtr);
    // 每个 block 负责一个专家，cumsumInfo 中每个专家占 worldSize 个
    // int32_t 存 rank 维度的 cumsum 结果，blockIdx 决定了负责哪个专家。
    uint64_t cumsumStride =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), ALIGN_32);
    cumsumInfoGlobalTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr + cumsumStride * blockIdx_));
    epilogueOp_.Init({.yGmAddr = params_.workspaceInfo.activationQuantDataPtr,
                      .yScaleGmAddr = params_.workspaceInfo.activationQuantScalePtr,
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    dispatchWinOffset_ =
        static_cast<uint64_t>(params_.peermemInfo.dispatchReceivePtr - params_.peermemInfo.rankSyncInWorldPtr);
    dispatchFlagWinOffset_ =
        static_cast<uint64_t>(params_.peermemInfo.dispatchFlagPtr - params_.peermemInfo.rankSyncInWorldPtr);
    // mask 是跨 Rank 共享的槽布局，几何必须与 PeermemInfo 一样按全卡容量计算。
    maskAlignSize_ = static_cast<uint32_t>(CalcDispatchMaskAlignSize(tilingData));
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
    relayRecordBytes_ =
        Ops::Base::CeilAlign(static_cast<uint64_t>(mxQuantTokenScaleAlignBytes_), static_cast<uint64_t>(ALIGN_512));
    dispatchRelayQueueBytesPerServer_ =
        static_cast<uint64_t>(ALIGN_32) + static_cast<uint64_t>(m_) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    // 远端 flag 按 256-token 连续窗口读取，UB 占用不随 BS 增长。
    uint64_t flagSnapshotBytes = static_cast<uint64_t>(DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) * sizeof(uint64_t);
    dispatchRelayFlagSnapshotBytesPerBlock_ =
        Ops::Base::CeilAlign(flagSnapshotBytes, static_cast<uint64_t>(ALIGN_512));
}

// =================================================================================================
// DispatchBuffInit：Dispatch 接收、计数发布和 ExpertTokenNumCopyOut 使用的 buffer。
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // Dispatch 阶段 UB 布局：Hcomm 私有区、专家计数/前缀和、分轮索引/掩码、
    // 中继标志快照/本地搬运缓冲、专家计数输出和元数据区依次排列。
    // 按最大 256 行元数据预留后，通信工作区不得越过 184 KiB 处的 Hcomm WQE 专用区。
    LocalTensor<uint8_t> hcommTensor = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512);
    hcomm_.Init(hcommTensor, ALIGN_512 / sizeof(uint8_t));
    hcommBatchWqeTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, LAYERED_HCOMM_BATCH_UB_OFFSET,
                                                LAYERED_HCOMM_BATCH_UB_BYTES / sizeof(uint8_t));
    uint32_t expertTokenCntTensorAddr = ALIGN_512;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    expertTokenCntTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // 分轮次参数计算：validTopkIndexTensor_ + topkIndexTensor_ + gatherMaskTensor_ 均为 round-sized
    // 专家计数直接从 GM 读取，不再加载完整 mask slot 到 UB。
    // round 张量: validTopkIndex(R*4) + topkIndex(R*4) + gatherMask(R/8) = 8R + R/8 = 65R/8
    // copyTmpTensors_ 与 relayFlagTensor_ 复用地址，只计两者中的较大值；此外预留专家计数输出和
    // 256 行 metadata。DispatchPrepare 使用的量化缓冲生命周期不重叠，不计入本阶段。
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(DISPATCH_COPY_BUFFER_COUNT) * mxQuantTokenScaleAlignBytes_;
    uint32_t relayFlagTensorSize = Ops::Base::CeilAlign(
        static_cast<uint32_t>(DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY * sizeof(uint64_t)),
        static_cast<uint32_t>(ALIGN_32));
    uint32_t expertTokenNumsOutTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(moeExpertPerRank_ * sizeof(int32_t)),
                            static_cast<int64_t>(ALIGN_32));
    uint32_t dispatchWorkTensorSize = relayFlagTensorSize > copyTmpTotalSize ? relayFlagTensorSize : copyTmpTotalSize;
    uint32_t dispatchFixedCost = dispatchWorkTensorSize + expertTokenNumsOutTensorSize + L1_TILE_M_256 * ALIGN_32;
    uint32_t fixedBefore = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t dispatchAvailUb = 0U;
    if (COMM_WORK_UB_LIMIT > fixedBefore && COMM_WORK_UB_LIMIT - fixedBefore > dispatchFixedCost) {
        dispatchAvailUb = COMM_WORK_UB_LIMIT - fixedBefore - dispatchFixedCost;
    }
    uint32_t maxDispatchRoundSize = dispatchAvailUb * 8U / 65U;
    maxDispatchRoundSize = (maxDispatchRoundSize / 256U) * 256U;
    if (maxDispatchRoundSize == 0U) {
        maxDispatchRoundSize = 256U;
    }
    if (maskRouteCapacity_ <= static_cast<uint64_t>(maxDispatchRoundSize)) {
        dispatchRoundSendTotalNum_ = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(ALIGN_256)));
    } else {
        uint32_t minDispatchRounds = static_cast<uint32_t>(
            Ops::Base::CeilDiv(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(maxDispatchRoundSize)));
        uint32_t evenDispatchRoundSize = static_cast<uint32_t>(Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(minDispatchRounds)),
            static_cast<int64_t>(ALIGN_256)));
        dispatchRoundSendTotalNum_ =
            (evenDispatchRoundSize <= maxDispatchRoundSize) ? evenDispatchRoundSize : maxDispatchRoundSize;
    }
    if (dispatchRoundSendTotalNum_ == 0U) {
        dispatchRoundSendTotalNum_ = 256U;
    }
    dispatchTotalRounds_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maskRouteCapacity_),
                           static_cast<int64_t>(dispatchRoundSendTotalNum_)));
    dispatchRoundMaskAlignSize_ = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(dispatchRoundSendTotalNum_) / 8, static_cast<int64_t>(ALIGN_32)));

    uint32_t validTopkIndexTensorAddr = fixedBefore;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(dispatchRoundSendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(dispatchRoundSendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIndexTensorSize / sizeof(int32_t));
    // gatherMaskTensor_ 仅容纳单轮 mask 位。
    uint32_t gatherMaskTensorAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t gatherMaskTensorSize = dispatchRoundMaskAlignSize_;
    gatherMaskTensor_ =
        LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr, gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ =
        LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr, gatherMaskTensorSize / sizeof(uint32_t));
    uint32_t tokenScaleSize = mxQuantTokenScaleAlignBytes_;
    uint32_t copyTmpBufferSize = tokenScaleSize;
    uint32_t copyTmpBaseAddr = gatherMaskTensorAddr + gatherMaskTensorSize;
    for (int32_t index = 0; index < DISPATCH_COPY_BUFFER_COUNT; ++index) {
        copyTmpTensors_[index] = LocalTensor<ActivationType>(
            TPosition::VECCALC, copyTmpBaseAddr + static_cast<uint32_t>(index) * copyTmpBufferSize,
            copyTmpBufferSize / sizeof(ActivationType));
    }
    relayFlagTensor_ =
        LocalTensor<uint64_t>(TPosition::VECCALC, copyTmpBaseAddr, relayFlagTensorSize / sizeof(uint64_t));
    // 中继标志快照和本地搬运缓冲的生命期不重叠，因此复用同一段 UB。
    // 一级量化已经在 Wave 前完成，Dispatch 阶段不再保留旧的单 token 量化和稀疏槽位 UB。
    uint32_t dispatchWorkTensorEnd = copyTmpBaseAddr + dispatchWorkTensorSize;
    // Tensor用处：ExpertTokenNumCopyOut 函数中本卡各专家收到的tokenCnt数；
    // Tensor大小：moeExpertPerRank_ * sizeof(int32_t) 对齐至32字节；
    uint32_t expertTokenNumsOutTensorAddr = dispatchWorkTensorEnd;
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                                                     expertTokenNumsOutTensorSize / sizeof(int32_t));
    // 后续 Dispatch 接收从该地址申请 metadata 临时区。
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    PipeBarrier<PIPE_ALL>();
}

// =================================================================================================
// DispatchPrepareBuffInit：一级 Dispatch 前置量化使用的双缓冲。
// SendMask/reset 和本阶段之间已有全核同步，Wave 内 DispatchBuffInit 会在前置阶段结束后复用同一 UB。
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchPrepareBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    uint32_t xInTensorSize =
        Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t xInTensorAddr1 = ALIGN_512;
    uint32_t xInTensorAddr2 = xInTensorAddr1 + xInTensorSize;
    xInTensor1_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInTensorAddr1,
                                         xInTensorSize / sizeof(bfloat16_t));
    xInTensor2_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInTensorAddr2,
                                         xInTensorSize / sizeof(bfloat16_t));

    uint32_t mxTempTensorAddr = xInTensorAddr2 + xInTensorSize;
    mxTempTensor_ =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, MX_QUANT_TEMP_UB_BYTES / sizeof(uint16_t));

    uint32_t xOutTensorSize = mxQuantTokenScaleAlignBytes_;
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    xOutTensor1_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1,
                                               xOutTensorSize / sizeof(ActivationType));
    xOutTensor2_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2,
                                               xOutTensorSize / sizeof(ActivationType));

    uint32_t readyFlagTensorAddr = xOutTensorAddr2 + xOutTensorSize;
    relayReadyFillTensor_ = LocalTensor<uint64_t>(TPosition::VECCALC, readyFlagTensorAddr,
                                                  DISPATCH_READY_FLAG_BATCH_TOKENS);
}

// ======================================================================================
// SendAndQuantBuffInit：SendMaskCal & ResetFlagList localTensor申请
// --------------------------------------------------------------------------------------
//   大 bs 场景下 sendTotalNum 可能超出 UB 256KB 限制，因此按 roundSendTotalNum_ 分轮次
//   分配 UB 张量（topkIds / sendMask / sendGatherOut），每轮仅加载一部分 topkIds，
//   在 UB 中计算部分 mask 后拼写到 workspace 对应偏移，最终合并为完整 mask slot。
// ======================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // 输入准备与 DispatchBuffInit 生命期不重叠，可复用同一 UB 地址。
    // 减去 Hcomm 和分批清零区后，按 8R + R/4 + 64 字节计算单轮容量，避免清零区随 GM 总量增长。
    LocalTensor<uint8_t> hcommTensor = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
    hcomm_.Init(hcommTensor, ALIGN_512);
    hcommBatchWqeTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, LAYERED_HCOMM_BATCH_UB_OFFSET,
                                                LAYERED_HCOMM_BATCH_UB_BYTES / sizeof(uint8_t));

    // 计算 resetTensor 大小（与原逻辑一致）
    uint64_t totalFlagInt32 =
        static_cast<uint64_t>(moeExpertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_) +
         static_cast<uint64_t>(INT_CACHELINE) * static_cast<uint64_t>(aicNum_)); // 64 * (16 + 256 + 16 * 28) = 46080
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        totalFlagInt32 += static_cast<uint64_t>(aicNum_) * INT_CACHELINE;
    }
    int64_t tokenGroupResetSize = static_cast<int64_t>(moeExpertPerRank_) * blockAivNum_ * INT_CACHELINE;
    totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > tokenGroupResetSize) ?
                         static_cast<int64_t>(totalFlagInt32) :
                         tokenGroupResetSize;
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchElementCount_ = resetNumPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                  static_cast<int32_t>(resetNumPerCore) :
                                  DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount_), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);
    resetBatchElementCount_ = resetTensorSize / sizeof(int32_t);

    // 分轮次参数计算：UB 预算扣除固定开销后，求解最大 roundSendTotalNum_
    // round 张量: topkIds(R*4) + 2*maskSlot(CeilAlign(R/8,32)+32) + gatherOut(R*4)
    //            = 8R + R/4 + 64  (R 为 256 对齐时 CeilAlign(R/8,32)=R/8)
    // 解: R <= (available - 64) * 4 / 33，再 floor-align 到 256（保证 GM 写 32B 对齐）
    uint32_t fixedUbCosts = ALIGN_512 + resetTensorSize;
    uint32_t availableUb = COMM_WORK_UB_LIMIT - fixedUbCosts;
    uint32_t maxRoundSize = (availableUb - 64U) * 4U / 33U;
    if (maxRoundSize == 0) {
        maxRoundSize = 256U;
    }
    maxRoundSize = (maxRoundSize / 256U) * 256U;
    if (static_cast<uint64_t>(sendTotalNum_) <= static_cast<uint64_t>(maxRoundSize)) {
        roundSendTotalNum_ = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(ALIGN_256)));
    } else {
        uint32_t minRounds = static_cast<uint32_t>(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(maxRoundSize)));
        uint32_t evenRoundSize = static_cast<uint32_t>(Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(minRounds)),
            static_cast<int64_t>(ALIGN_256)));
        roundSendTotalNum_ = (evenRoundSize <= maxRoundSize) ? evenRoundSize : maxRoundSize;
    }
    if (roundSendTotalNum_ == 0) {
        roundSendTotalNum_ = 256U;
    }
    roundCompareCount_ = roundSendTotalNum_; // 16K
    roundMaskAlignSize_ = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(roundCompareCount_) / 8, static_cast<int64_t>(ALIGN_32)));
    roundMaskSlotSize_ = roundMaskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    totalRounds_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(roundSendTotalNum_)));

    // Tensor用处：SendMaskCal 每轮搬运本卡 topkIds 的一个子段；
    // Tensor大小：roundCompareCount_ 个 int32（256B 对齐）；
    uint32_t topkIdsTensorAddr = ALIGN_512;
    uint32_t topkIdsTensorSize = roundCompareCount_ * sizeof(int32_t);
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    // Tensor用处：SendMaskCal 每轮存储部分 mask 位；DOUBLE_BUFFER 双缓冲；
    // Tensor大小：roundMaskSlotSize_（每轮部分 [mask|count] 槽位大小）；
    uint32_t sendMaskAddr = resetTensorAddr + resetTensorSize;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] = LocalTensor<uint8_t>(
            TPosition::VECCALC, sendMaskAddr + static_cast<uint32_t>(index) * roundMaskSlotSize_, roundMaskSlotSize_);
    }

    // Tensor用处：SendMaskCal 每轮 GatherMask 计 count 的废弃输出 scratch；
    // Tensor大小：roundCompareCount_ 个 int32（256B 对齐）；
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * roundMaskSlotSize_;
    uint32_t sendGatherOutSize = roundCompareCount_ * sizeof(int32_t);
    sendGatherOutTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
}

// ===============================================================================================
// ResetFlagList：清理本卡 workspace 中连续排布的 GMM/dispatch flag 和 AIC-AIV1 ready sequence。
// ===============================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetFlagList()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // workSpace Flag 清零
    // 总数 = ActivationToGmm2(moeExpertPerRank * INT_CACHELINE)
    //        + DispatchToGmm1(moeExpertPerRank * dispatchFlagSlotsPerExpert_)
    //        + SendCntCalToUpdParams(moeExpertPerRank * aicNum_ * INT_CACHELINE)
    //        + GmmToEpilogue(aicNum_ * INT_CACHELINE, specialized A8W4/A4W4 only)
    activationToGmm2FlagGm_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.flagActivationToGmm2Ptr);
    int32_t flagNum =
        static_cast<int32_t>(moeExpertPerRank_) * (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_ +
                                                   static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(aicNum_));
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        flagNum += static_cast<int32_t>(aicNum_) * static_cast<int32_t>(INT_CACHELINE);
    }
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset, 1);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    for (int32_t resetElementOffset = 0; resetElementOffset < coreLen;
         resetElementOffset += resetBatchElementCount_) {
        int32_t currentBatchElementCount = coreLen - resetElementOffset < resetBatchElementCount_ ?
                                               coreLen - resetElementOffset :
                                               resetBatchElementCount_;
        DataCopyExtParams flagCopyParams{
            1U, static_cast<uint32_t>(currentBatchElementCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(activationToGmm2FlagGm_[coreOffset + resetElementOffset], resetTensor_, flagCopyParams);
    }
    // URMA Layered 的量化与非量化 Combine 都使用 token-group counter。
    ResetGmm2CombineSyncCounters();
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<int32_t> statusGm;
        statusGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr));
        int32_t statusElementCount =
            (static_cast<int32_t>(moeExpertPerRank_) * static_cast<int32_t>(params_.tilingData->maxTilesPerExpert) +
             1) *
            INT_CACHELINE;
        int32_t statusCoreLen, statusCoreOffset;
        TilingByCore(statusElementCount, statusCoreLen, statusCoreOffset, 1);
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
}

// ==================================================
// ExpertTokenNumCopyOut：输出本卡各 MoE 专家收到的 token 总数（不包含共享专家）。
// ==================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ExpertTokenNumCopyOut()
{
    // 归属核 Wave 在 Dispatch、GMM 和 Combine 之间复用 AIV1 UB，输出前必须恢复完整前缀和。
    DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                {1U, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    expertTokenNumsOutTensor_.SetValue(0, cumsumInfoTensor_.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < moeExpertPerRank_; expertIdx++) {
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
// SendMaskCal：对本卡 topk 按通信域内所有专家id计算mask位，并发送至目标专家卡
// ------------------------------------------------------------------------------------------------------
//   大 bs 场景下 sendTotalNum 可能超出 UB 限制，因此按 roundSendTotalNum_ 分轮次计算：
//   Phase 1: 对每个全局专家，逐轮加载 topkIds 子段 → CompareScalar → 部分 mask；
//   Phase 2: 每轮部分 mask 拼写到 workspace/win 对应字节偏移，GatherMask 累加本轮 count；
//   Phase 3: 所有轮次完成后，写入 total count，URMA 发送完整 mask slot。
// ======================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendMaskCal()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t totalExperts = static_cast<int32_t>(worldSize_);
    // 按卡处理，只使用前 worldSize 个核。
    for (uint32_t curRankId = aivCoreIdx_; curRankId < static_cast<uint32_t>(totalExperts);
         curRankId += blockAivNum_) {
        ChannelHandle maskChannel{};
        if (curRankId != rankId_) {
            maskChannel = GetUrmaCommHandle(mc2Context_, curRankId, rankId_);
            // SendMask 与紧随其后的第一次全卡同步由同一物理 AIV 负责，锁跨两个阶段复用，
            // 避免在中间没有换核时提前 Unlock 刷新通道 counter。
            hcomm_.Lock(maskChannel);
        }
        for (uint32_t expertIdIndex = 0; expertIdIndex < moeExpertPerRank_; ++expertIdIndex) { // 按专家处理
            int32_t curExpertId = static_cast<int32_t>(curRankId * moeExpertPerRank_ + expertIdIndex);
            uint64_t srcOffset = static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) +
                                                       static_cast<int32_t>(curRankId)) *
                                 static_cast<uint64_t>(maskSlotSize_);
            uint64_t dstOffset =
                maskWinOffset_ + static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) +
                                                       static_cast<int32_t>(rankId_)) *
                                     static_cast<uint64_t>(maskSlotSize_);
            uint64_t totalSendCnt = 0;

            for (uint32_t roundIdx = 0; roundIdx < totalRounds_; ++roundIdx) {
                uint64_t roundStart = static_cast<uint64_t>(roundIdx) * static_cast<uint64_t>(roundSendTotalNum_);
                uint64_t roundLen64 = (roundIdx + 1 < totalRounds_) ? static_cast<uint64_t>(roundSendTotalNum_) :
                                                                      sendTotalNum_ - roundStart;
                uint32_t roundLen = static_cast<uint32_t>(roundLen64);
                uint32_t curRoundCompareCount =
                    (roundLen == roundSendTotalNum_) ?
                        roundCompareCount_ :
                        static_cast<uint32_t>(
                            Ops::Base::CeilAlign(static_cast<int64_t>(roundLen) * static_cast<int64_t>(sizeof(int32_t)),
                                                 static_cast<int64_t>(ALIGN_256)) /
                            static_cast<int64_t>(sizeof(int32_t)));
                uint32_t curRoundMaskAlignSize =
                    (roundLen == roundSendTotalNum_) ?
                        roundMaskAlignSize_ :
                        static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<int64_t>(curRoundCompareCount) / 8,
                                                                   static_cast<int64_t>(ALIGN_32)));

                // Phase 1: 加载本轮 topkIds 子段；用非法 expert id 填充对齐尾部。
                Duplicate<int32_t>(topkIdsTensor_, -1, roundCompareCount_);
                SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
                GlobalTensor<int32_t> roundSrcGlobal;
                roundSrcGlobal.SetGlobalBuffer(
                    (__gm__ int32_t *)(params_.expertIdxGmAddr + static_cast<uint64_t>(roundStart) * sizeof(int32_t)));
                DataCopyExtParams roundLoadParams{1U, static_cast<uint32_t>(roundLen * sizeof(int32_t)), 0U, 0U, 0U};
                DataCopyPadExtParams<int32_t> roundLoadPad{false, 0U, 0U, 0U};
                DataCopyPad(topkIdsTensor_, roundSrcGlobal, roundLoadParams, roundLoadPad);
                SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

                // Phase 2: CompareScalar → 本轮部分 mask + GatherMask 累加 count
                LocalTensor<uint8_t> maskBuf = sendMaskTensor_[roundIdx % DOUBLE_BUFFER];
                LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
                CompareScalar(maskBuf, topkIdsTensor_, curExpertId, AscendC::CMPMODE::EQ, curRoundCompareCount);
                PipeBarrier<PIPE_V>();
                uint64_t roundSendCnt = 0;
                GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true, static_cast<uint32_t>(roundLen),
                           {1, 1, 0, 0}, roundSendCnt);
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
                totalSendCnt += roundSendCnt;
                SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

                // Phase 2b: 将本轮部分 mask 写入 workspace 或 local win 对应字节偏移
                // roundByteOffset = roundStart/8，因 roundSendTotalNum_ 为 256 对齐，该偏移 32B 对齐
                uint64_t roundByteOffset = static_cast<uint64_t>(roundStart) / 8U;
                uint32_t writeBytes =
                    (static_cast<uint64_t>(curRoundMaskAlignSize) <=
                     static_cast<uint64_t>(maskAlignSize_) - roundByteOffset) ?
                        curRoundMaskAlignSize :
                        static_cast<uint32_t>(static_cast<uint64_t>(maskAlignSize_) - roundByteOffset);
                DataCopyExtParams partialMaskCopyParams{1U, writeBytes, 0U, 0U, 0U};

                if (curRankId == rankId_) {
                    GlobalTensor<uint8_t> winDstGlobal;
                    winDstGlobal.SetGlobalBuffer(
                        (__gm__ uint8_t *)(GetRankWinAddrWithOffset(rankId_, dstOffset + roundByteOffset)));
                    DataCopyPad(winDstGlobal, maskBuf, partialMaskCopyParams);
                } else {
                    GlobalTensor<uint8_t> wsDstGlobal;
                    wsDstGlobal.SetGlobalBuffer(
                        (__gm__ uint8_t *)(params_.workspaceInfo.maskSlotPtr + srcOffset + roundByteOffset));
                    DataCopyPad(wsDstGlobal, maskBuf, partialMaskCopyParams);
                }
                PipeBarrier<PIPE_ALL>();
            }

            // Phase 3: 写入 total count 并发送完整 mask slot
            if (curRankId == rankId_) {
                __gm__ int32_t *winCountPtr =
                    reinterpret_cast<__gm__ int32_t *>(GetRankWinAddrWithOffset(rankId_, dstOffset + maskAlignSize_));
                WriteGmByPassDCache(winCountPtr, static_cast<int32_t>(totalSendCnt));
                PipeBarrier<PIPE_ALL>();
            } else {
                __gm__ int32_t *wsCountPtr =
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.maskSlotPtr + srcOffset + maskAlignSize_);
                WriteGmByPassDCache(wsCountPtr, static_cast<int32_t>(totalSendCnt));
                PipeBarrier<PIPE_ALL>();
                GM_ADDR remoteDataAddr = GetRankWinAddrWithOffset(curRankId, dstOffset);
                GM_ADDR localGmAddr = params_.workspaceInfo.maskSlotPtr + srcOffset;
                hcomm_.WriteNbi(maskChannel, remoteDataAddr, localGmAddr, maskSlotSize_);
            }
        }
    }
    // SendMask 使用 direct WriteNbi，不创建批量 handle。先让本 AIV 负责的所有远端通道
    // 都拥有在途请求，再统一 Drain；锁继续保留给紧随其后的 CrossRankSync 复用。
    for (uint32_t curRankId = aivCoreIdx_; curRankId < static_cast<uint32_t>(totalExperts);
         curRankId += blockAivNum_) {
        if (curRankId == rankId_) {
            continue;
        }
        ChannelHandle maskChannel = GetUrmaCommHandle(mc2Context_, curRankId, rankId_);
        hcomm_.Drain(maskChannel);
    }
}

// ======================================================================
// LoadTopkWeightsToUb：权重搬运到UB（TopkWeightsPrefetch=0 时仅做 MTE2_V 同步）
// ======================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTopkWeightsToUb(
    const LocalTensor<ActivationType> &xOutTensor, int32_t currentOffset, int32_t index, TEventID event)
{
    uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<TopkWeightsType> weightGm;
        weightGm.SetGlobalBuffer(
            (__gm__ TopkWeightsType *)(params_.probsGmAddr +
                                       static_cast<int64_t>(currentOffset + index) * topK_ * sizeof(TopkWeightsType)));
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

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchRelayQueueServerOffset(
    uint32_t targetServer) const
{
    return static_cast<uint64_t>(targetServer) * dispatchRelayQueueBytesPerServer_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RelayTokenOffset(uint32_t sourceServer,
                                                                                            uint32_t tokenId) const
{
    return (static_cast<uint64_t>(sourceServer) *
                static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank) + tokenId) *
           static_cast<uint64_t>(relayRecordBytes_);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RelayFlagOffset(uint32_t sourceServer,
                                                                                           uint32_t tokenId) const
{
    return (static_cast<uint64_t>(sourceServer) *
                static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank) + tokenId) *
           sizeof(uint64_t);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetContiguousGm(GM_ADDR dstAddr,
                                                                                         uint64_t sizeBytes)
{
    if (sizeBytes == 0 || resetBatchElementCount_ == 0) {
        return;
    }
    GlobalTensor<int32_t> dstGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dstAddr));

    uint64_t totalElements = sizeBytes / sizeof(int32_t);
    uint64_t elementsPerCore = Ops::Base::CeilDiv(totalElements, static_cast<uint64_t>(blockAivNum_));
    elementsPerCore = Ops::Base::CeilAlign(elementsPerCore, static_cast<uint64_t>(INT32_PER_256B));
    uint64_t coreOffset = static_cast<uint64_t>(aivCoreIdx_) * elementsPerCore;
    if (coreOffset >= totalElements) {
        return;
    }

    uint64_t coreElements = totalElements - coreOffset;
    coreElements = coreElements < elementsPerCore ? coreElements : elementsPerCore;
    for (uint64_t resetOffset = 0; resetOffset < coreElements; resetOffset += resetBatchElementCount_) {
        uint64_t remainingElements = coreElements - resetOffset;
        uint32_t currentElements =
            static_cast<uint32_t>(remainingElements < static_cast<uint64_t>(resetBatchElementCount_) ?
                                      remainingElements :
                                      static_cast<uint64_t>(resetBatchElementCount_));
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(currentElements * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(dstGm[coreOffset + resetOffset], resetTensor_, copyParams);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetDispatchState()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 队列在全 AIV 构建完成后一次性消费，只需清零每个目标 Server 的最终 count。
    for (uint32_t targetServer = aivCoreIdx_; targetServer < serverNum_; targetServer += blockAivNum_) {
        __gm__ int32_t *countPtr = reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.dispatchRelaySendQueuePtr + DispatchRelayQueueServerOffset(targetServer));
        WriteGmByPassDCache(countPtr, int32_t(0));
    }
    // 远端 ready flag 仍是一级数据到达 relay 后供二级 Dispatch 使用的完成协议。
    ResetContiguousGm(params_.peermemInfo.dispatchFlagPtr,
                      static_cast<uint64_t>(serverNum_) *
                          static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank) * sizeof(uint64_t));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::QuantizeLocalTokensToRelay()
{
    uint32_t baseTokenNum = m_ / blockAivNum_;
    uint32_t tokenRemainder = m_ % blockAivNum_;
    uint32_t tokenNumInCore = baseTokenNum + static_cast<uint32_t>(aivCoreIdx_ < tokenRemainder);
    uint32_t tokenStart = aivCoreIdx_ * baseTokenNum +
                          ((aivCoreIdx_ < tokenRemainder) ? aivCoreIdx_ : tokenRemainder);
    if (tokenNumInCore == 0U) {
        return;
    }

    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> workspaceDstGlobal;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params_.aGmAddr) +
                                    static_cast<uint64_t>(tokenStart) * k_);

    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(k_ * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxTempTensor_.GetPhyAddr());
    __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        mxTempTensor_[Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))]
            .GetPhyAddr());

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0U; index < tokenNumInCore; ++index) {
        bool useFirstBuffer = (index & 1U) == 0U;
        TEventID event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<bfloat16_t> xInTensor = useFirstBuffer ? xInTensor1_ : xInTensor2_;
        LocalTensor<ActivationType> xOutTensor = useFirstBuffer ? xOutTensor1_ : xOutTensor2_;
        uint32_t tokenIdx = tokenStart + index;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor[static_cast<uint64_t>(index) * k_],
                    xCopyInParams, xCopyInPadParams);
        LoadTopkWeightsToUb(xOutTensor, tokenStart, index, event);

        __ubuf__ bfloat16_t *srcAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(xInTensor.GetPhyAddr());
        __ubuf__ int8_t *outDataAddr = reinterpret_cast<__ubuf__ int8_t *>(xOutTensor.GetPhyAddr());
        __ubuf__ uint16_t *mxScaleAddr =
            reinterpret_cast<__ubuf__ uint16_t *>(xOutTensor[mxQuantTokenAlignBytes_].GetPhyAddr());
        Quant::ComputeMaxExp(srcAddr, maxExpAddr, k_);
        Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
                                          mxQuantScaleNumAlignPerToken_);
        if constexpr (QuantMode == E2M1_QUANT) {
            Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, k_);
        } else {
            Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, k_);
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        uint64_t relayOffset = RelayTokenOffset(serverId_, tokenIdx);
        GM_ADDR recordAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) + relayOffset;
        workspaceDstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(recordAddr));
        LocalTensor<uint8_t> xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(workspaceDstGlobal, xOutBytesTensor,
                    {1U, mxQuantTokenScaleAlignBytes_, 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SetLocalRelayReadyFlags()
{
    uint32_t baseTokenNum = m_ / blockAivNum_;
    uint32_t tokenRemainder = m_ % blockAivNum_;
    uint32_t tokenNumInCore = baseTokenNum + static_cast<uint32_t>(aivCoreIdx_ < tokenRemainder);
    uint32_t tokenStart = aivCoreIdx_ * baseTokenNum +
                          ((aivCoreIdx_ < tokenRemainder) ? aivCoreIdx_ : tokenRemainder);
    if (tokenNumInCore == 0U) {
        return;
    }

    Duplicate<uint64_t>(relayReadyFillTensor_, uint64_t(1), DISPATCH_READY_FLAG_BATCH_TOKENS);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    GlobalTensor<uint64_t> readyFlagGlobal;
    readyFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(params_.peermemInfo.dispatchFlagPtr) +
                                    static_cast<uint64_t>(serverId_) * params_.tilingData->numMaxTokensPerRank +
                                    tokenStart);
    for (uint32_t offset = 0U; offset < tokenNumInCore; offset += DISPATCH_READY_FLAG_BATCH_TOKENS) {
        uint32_t remaining = tokenNumInCore - offset;
        uint32_t currentTokenCount = remaining < DISPATCH_READY_FLAG_BATCH_TOKENS ?
                                         remaining : DISPATCH_READY_FLAG_BATCH_TOKENS;
        DataCopyPad(readyFlagGlobal[offset], relayReadyFillTensor_,
                    {1U, currentTokenCount * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U});
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendTokenToDispatchRelayQueues(
    GlobalTensor<int32_t> &topkIdsGlobal, uint32_t tokenIdx)
{
    uint32_t targetServers[DISPATCH_MAX_TOPK];
    uint32_t targetServerCount = 0U;
    for (uint32_t topkIdx = 0; topkIdx < topK_; ++topkIdx) {
        int32_t globalExpertId = topkIdsGlobal.GetValue(tokenIdx * topK_ + topkIdx);
        if (globalExpertId < 0 ||
            globalExpertId >= static_cast<int32_t>(worldSize_ * moeExpertPerRank_)) {
            continue;
        }
        uint32_t targetRank = static_cast<uint32_t>(globalExpertId) / moeExpertPerRank_;
        uint32_t targetServer = targetRank / rankNumPerServer_;
        if (targetServer == serverId_) {
            continue;
        }
        bool duplicatedServer = false;
        for (uint32_t index = 0U; index < targetServerCount; ++index) {
            if (targetServers[index] == targetServer) {
                duplicatedServer = true;
                break;
            }
        }
        if (duplicatedServer) {
            continue;
        }
        targetServers[targetServerCount++] = targetServer;
    }

    for (uint32_t index = 0U; index < targetServerCount; ++index) {
        uint32_t targetServer = targetServers[index];
        __gm__ int32_t *countPtr = reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.dispatchRelaySendQueuePtr + DispatchRelayQueueServerOffset(targetServer));
        int32_t slotIdx = AtomicAdd(countPtr, int32_t(1));
        uint64_t metaOffset = DispatchRelayQueueServerOffset(targetServer) + ALIGN_32 +
                              static_cast<uint64_t>(slotIdx) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
        __gm__ int32_t *metaPtr =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchRelaySendQueuePtr + metaOffset);
        WriteGmByPassDCache(metaPtr, static_cast<int32_t>(tokenIdx));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildDispatchRelayQueues()
{
    // 全部物理 AIV 扫描不重叠的 token 区间；每个 token 在核内先折叠为唯一目标 Server 集合。
    uint32_t computeIdx = aivCoreIdx_;
    uint32_t baseTokenNum = m_ / blockAivNum_;
    uint32_t tokenRemainder = m_ % blockAivNum_;
    uint32_t tokenNumInCore = baseTokenNum + static_cast<uint32_t>(computeIdx < tokenRemainder);
    uint32_t tokenStart = computeIdx * baseTokenNum + ((computeIdx < tokenRemainder) ? computeIdx : tokenRemainder);
    uint32_t tokenEnd = tokenStart + tokenNumInCore;
    GlobalTensor<int32_t> topkIdsGlobal;
    topkIdsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.expertIdxGmAddr));
    for (uint32_t tokenIdx = tokenStart; tokenIdx < tokenEnd; ++tokenIdx) {
        AppendTokenToDispatchRelayQueues(topkIdsGlobal, tokenIdx);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendDispatchRelayData(
    uint32_t targetServer, uint32_t relayRank, HcommBatchHandle &batchHandle, int32_t slot)
{
    uint64_t metaOffset = DispatchRelayQueueServerOffset(targetServer) + ALIGN_32 +
                          static_cast<uint64_t>(slot) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    __gm__ int32_t *srcMetaPtr =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchRelaySendQueuePtr + metaOffset);
    int32_t tokenIdx = ReadGmByPassDCache(srcMetaPtr);
    uint64_t relayOffset = RelayTokenOffset(serverId_, static_cast<uint32_t>(tokenIdx));
    GM_ADDR srcAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) + relayOffset;
    GM_ADDR dstAddr = GetRankWinAddrWithOffset(relayRank, dispatchWinOffset_) + relayOffset;
    hcomm_.WriteNbi<DISPATCH_DATA_WQE_CONFIG>(batchHandle, dstAddr, srcAddr, mxQuantTokenScaleAlignBytes_);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendDispatchRelayFlag(
    uint32_t targetServer, uint32_t relayRank, HcommBatchHandle &batchHandle, int32_t slot, bool firstFlag)
{
    uint64_t metaOffset = DispatchRelayQueueServerOffset(targetServer) + ALIGN_32 +
                          static_cast<uint64_t>(slot) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    __gm__ int32_t *srcMetaPtr =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchRelaySendQueuePtr + metaOffset);
    int32_t tokenIdx = ReadGmByPassDCache(srcMetaPtr);
    uint64_t flagOffset = RelayFlagOffset(serverId_, static_cast<uint32_t>(tokenIdx));
    GM_ADDR localFlagAddr = GetRankWinAddrWithOffset(rankId_, dispatchFlagWinOffset_) + flagOffset;
    GM_ADDR remoteFlagAddr = GetRankWinAddrWithOffset(relayRank, dispatchFlagWinOffset_) + flagOffset;
    if (firstFlag) {
        hcomm_.WriteNbi<DISPATCH_FIRST_FLAG_WQE_CONFIG>(batchHandle, remoteFlagAddr, localFlagAddr,
                                                        sizeof(uint64_t));
    } else {
        hcomm_.WriteNbi<DISPATCH_FLAG_WQE_CONFIG>(batchHandle, remoteFlagAddr, localFlagAddr, sizeof(uint64_t));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CommitDispatchRelayBatch(
    uint32_t targetServer, uint32_t relayRank, HcommBatchHandle &batchHandle, int32_t batchStart,
    uint32_t batchTokenCount)
{
    // 先提交本批全部 token 数据，再单独组装并提交 flag。首条 flag 使用 odr=6 建立批次边界；
    // 后续 flag 使用 odr=5。三种 WQE 均关闭 CQE，这一配置不外溢到二级读和 Combine。
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        AppendDispatchRelayData(targetServer, relayRank, batchHandle,
                                batchStart + static_cast<int32_t>(index));
    }
    hcomm_.BatchCommit(batchHandle);
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        AppendDispatchRelayFlag(targetServer, relayRank, batchHandle,
                                batchStart + static_cast<int32_t>(index), index == 0U);
    }
    hcomm_.BatchCommit(batchHandle);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendDispatchRelayQueues()
{
    for (uint32_t targetServer = 0; targetServer < serverNum_; ++targetServer) {
        if (targetServer == serverId_) {
            continue;
        }
        uint32_t relayRank = targetServer * rankNumPerServer_ + rankIdInServer_;
        // 每个目标 Server 只对应一个一级中继 rank，统一归属映射保证仅该 AIV1 访问通道。
        // token 队列仍由所有计算 block 共享。
        if (relayRank >= worldSize_ || !IsChannelOwner(relayRank)) {
            continue;
        }

        ChannelHandle channel = GetUrmaCommHandle(mc2Context_, relayRank, rankId_);
        HcommBatchHandle batchHandle = hcomm_.MakeBatchHandle(
            channel, hcommBatchWqeTensor_, LAYERED_HCOMM_BATCH_UB_BYTES,
            GetRankWinAddrWithOffset(relayRank, 0));
        // 队列已经由全部 AIV 构建并同步完成，[0, count) 可直接按连续 slot 分批发送。
        __gm__ int32_t *countPtr = reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.dispatchRelaySendQueuePtr + DispatchRelayQueueServerOffset(targetServer));
        int32_t tokenCount = ReadGmByPassDCache(countPtr);
        for (int32_t batchStart = 0; batchStart < tokenCount;
             batchStart += static_cast<int32_t>(DISPATCH_SEND_BATCH_TOKEN_CAPACITY)) {
            int32_t remaining = tokenCount - batchStart;
            uint32_t batchTokenCount = static_cast<uint32_t>(
                remaining < static_cast<int32_t>(DISPATCH_SEND_BATCH_TOKEN_CAPACITY) ?
                    remaining : static_cast<int32_t>(DISPATCH_SEND_BATCH_TOKEN_CAPACITY));
            CommitDispatchRelayBatch(targetServer, relayRank, batchHandle, batchStart, batchTokenCount);
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTokenFromLocalRelay(uint32_t srcServer,
                                                                                               uint32_t tokenIndex,
                                                                                               int32_t bufferIdx,
                                                                                               uint32_t copyInNum)
{
    __gm__ uint64_t *readyFlag = reinterpret_cast<__gm__ uint64_t *>(params_.peermemInfo.dispatchFlagPtr +
                                                                     RelayFlagOffset(srcServer, tokenIndex));
    while (ReadGmByPassDCache(readyFlag) != uint64_t(1)) {
    }

    uint64_t remoteCopyOffset = RelayTokenOffset(srcServer, tokenIndex);
    GM_ADDR localRecordAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) + remoteCopyOffset;
    GlobalTensor<ActivationType> relayGlobalTensor;
    relayGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(localRecordAddr));
    DataCopy(copyTmpTensors_[bufferIdx], relayGlobalTensor, copyInNum);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
}

// ==================================================================================================
// 计算并发布当前专家接收的 token 数。
// --------------------------------------------------------------------------------------------------
//   Phase 1: 逐 rank 从本卡 win 加载单个 [mask|count] 槽位，提取 count 并累加 cumsum；
//   Phase 2: 写 expertRevNumsGlobalTensor_ + AtomicAdd 通知 AIC;
// ==================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishExpertTokenCount(uint32_t expertIdx)
{
    uint64_t tokenCount = 0U;
    uint32_t expertRankOffset = expertIdx * worldSize_;
    // 逐 rank 直接从 GM 读取 count 并累计全局 cumsum。
    for (uint32_t srcRank = 0U; srcRank < worldSize_; ++srcRank) {
        __gm__ int32_t *rankCountPtr = reinterpret_cast<__gm__ int32_t *>(
            params_.peermemInfo.maskRecvPtr + static_cast<uint64_t>(expertIdx) * worldSize_ * maskSlotSize_ +
            static_cast<uint64_t>(srcRank) * maskSlotSize_ + maskAlignSize_);
        int32_t rankTokenCount = ReadGmByPassDCache(rankCountPtr);
        tokenCount += static_cast<uint64_t>(rankTokenCount);
        cumsumRevCntInRank_ += static_cast<uint64_t>(rankTokenCount);
        cumsumInfoTensor_.SetValue(expertRankOffset + srcRank,
                                   static_cast<int32_t>(cumsumRevCntInRank_));
    }

    // 仅写当前专家的 cumsum slice；随后发布 token 数和 AIC ready flag。
    expertTokenCntTensor_.SetValue(0, static_cast<int32_t>(tokenCount));
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(expertRevNumsGlobalTensor_[expertIdx * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_],
                      expertTokenCntTensor_, INT32_PER_256B);
    uint32_t cumsumCopyBytes = static_cast<uint32_t>(worldSize_ * sizeof(int32_t));
    if (cumsumCopyBytes % static_cast<uint32_t>(ALIGN_32) == 0U) {
        DataCopyPad(cumsumInfoGlobalTensor_[expertRankOffset], cumsumInfoTensor_[expertRankOffset],
                    {1U, cumsumCopyBytes, 0U, 0U, 0U});
    } else {
        // Packed expert slices are not UB-aligned when worldSize_ is not a multiple of 8.
        uint64_t cumsumStride =
            Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), ALIGN_32);
        __gm__ int32_t *cumsumDst = reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.cumsumInfoPtr + cumsumStride * blockIdx_) + expertRankOffset;
        for (uint32_t srcRank = 0U; srcRank < worldSize_; ++srcRank) {
            WriteGmByPassDCache(cumsumDst + srcRank, cumsumInfoTensor_.GetValue(expertRankOffset + srcRank));
        }
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID2>();
    PipeBarrier<PIPE_ALL>();
    if constexpr (TopkWeightsPrefetch) {
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    }

    __gm__ int32_t *tokenCountReadyFlag =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
        static_cast<uint64_t>(expertIdx) * aicNum_ * INT_CACHELINE +
        static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    AscendC::AtomicAdd(tokenCountReadyFlag, static_cast<int32_t>(1));
}

// ============================================================================
// CopyTokensFromLocalRelay：本卡中转使用 UB 多 buffer 搬运 token 与 scale
// ----------------------------------------------------------------------------
//   prime: 发出前 DISPATCH_COPY_BUFFER_COUNT 个 token 的 MTE2。
//   steady: 每轮执行 MTE3_out[i] + MTE2_in[i + DISPATCH_COPY_BUFFER_COUNT]，循环复用槽位。
//   drain: 收尾不再发新 MTE2，只等待 MTE3 完成。
// ============================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyTokensFromLocalRelay(
    int32_t rowDstOffsetInCore, uint32_t srcServer, int32_t copyNum, int64_t widthA, int64_t widthAScale,
    uint32_t copyInNum)
{
    constexpr TEventID kBufEvents[DISPATCH_COPY_BUFFER_COUNT] = {
        EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        params_.workspaceInfo.dispatchRevDataPtr +
        static_cast<uint64_t>(rowDstOffsetInCore) * widthA * sizeof(ActivationType)));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr +
        static_cast<uint64_t>(rowDstOffsetInCore) * widthAScale * sizeof(QuantScaleOutType)));

    for (int32_t bufferIdx = 0; bufferIdx < DISPATCH_COPY_BUFFER_COUNT; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }

    int32_t primeCount = copyNum < DISPATCH_COPY_BUFFER_COUNT ? copyNum : DISPATCH_COPY_BUFFER_COUNT;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t tokenIndex = metaInfoTensor_[primeIdx * INT32_PER_256B].GetValue(TOKEN_ID);
        TEventID eventId = kBufEvents[primeIdx];
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        LoadTokenFromLocalRelay(srcServer, static_cast<uint32_t>(tokenIndex), primeIdx, copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        if constexpr (TopkWeightsPrefetch) {
            SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
        }
    }

    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t outIdx = copyIdx % DISPATCH_COPY_BUFFER_COUNT;
        TEventID eventId = kBufEvents[outIdx];
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

        if constexpr (TopkWeightsPrefetch) {
            WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
            LocalTensor<ActivationType> weightBuf = copyTmpTensors_[outIdx];
            uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
            LocalTensor<int32_t> bufWeightsInt32 = weightBuf[weightOffsetInUb].template ReinterpretCast<int32_t>();
            int32_t topkIndex = metaInfoTensor_[copyIdx * INT32_PER_256B].GetValue(TOPK_INDEX);
            int32_t weightBits = bufWeightsInt32.GetValue(static_cast<uint32_t>(topkIndex));
            metaInfoTensor_[copyIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        }

        LocalTensor<ActivationType> tokenScaleBuf = copyTmpTensors_[outIdx];
        LocalTensor<QuantScaleOutType> scaleBuf =
            tokenScaleBuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScaleBuf,
                    {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], scaleBuf,
                    {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

        int32_t nextIdx = copyIdx + DISPATCH_COPY_BUFFER_COUNT;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = metaInfoTensor_[nextIdx * INT32_PER_256B].GetValue(TOKEN_ID);
            // 等待本轮 MTE3 完成后再复用 outIdx 槽。
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            LoadTokenFromLocalRelay(srcServer, static_cast<uint32_t>(tokenIndex), outIdx, copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            if constexpr (TopkWeightsPrefetch) {
                SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
            }
        }
    }

    for (int32_t bufferIdx = 0; bufferIdx < DISPATCH_COPY_BUFFER_COUNT; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishDispatchRows(
    uint32_t expertIdx, int32_t globalRowStart, int32_t rowCount)
{
    if (rowCount <= 0) {
        return;
    }
    int32_t priorExpertRows =
        expertIdx == 0U ? 0 : cumsumInfoTensor_.GetValue(expertIdx * worldSize_ - 1U);
    int32_t localRow = globalRowStart - priorExpertRows;
    int32_t remainingRows = rowCount;
    __gm__ int32_t *expertFlagBase =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_;
    while (remainingRows > 0) {
        int32_t waveIdx = localRow / static_cast<int32_t>(L1_TILE_M_256);
        int32_t rowsInWave = static_cast<int32_t>(L1_TILE_M_256) -
                             localRow % static_cast<int32_t>(L1_TILE_M_256);
        if (rowsInWave > remainingRows) {
            rowsInWave = remainingRows;
        }
        AtomicAdd(expertFlagBase + static_cast<int64_t>(waveIdx) * INT_CACHELINE, rowsInWave);
        localRow += rowsInWave;
        remainingRows -= rowsInWave;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveRemoteDispatchBatch(
    uint32_t relayRank, uint32_t batchTokenCount)
{
    if (batchTokenCount == 0U) {
        return;
    }

    GM_ADDR scratchAddr = params_.workspaceInfo.dispatchRemoteReadyFlagSnapshotPtr +
                          static_cast<uint64_t>(blockIdx_) * dispatchRelayFlagSnapshotBytesPerBlock_;
    GlobalTensor<uint64_t> scratchFlagGlobalTensor;
    scratchFlagGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(scratchAddr));
    DataCopyPadExtParams<uint64_t> flagCopyPadParams{true, 0U, 0U, 0U};
    ChannelHandle channel = GetUrmaCommHandle(mc2Context_, relayRank, rankId_);

    // 批次可以跨专家和源 Server，但不能跨 relay 通道。ready flag 按 token-id 的 256 槽窗口读取，
    // 只处理当前批次实际命中的窗口。token id 属于源 Rank，上界必须用全卡容量而非本卡 m_。
    for (uint32_t srcServer = 0U; srcServer < serverNum_; ++srcServer) {
        for (uint64_t windowStart64 = 0U;
             windowStart64 < static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank);
             windowStart64 += DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) {
            uint32_t windowStart = static_cast<uint32_t>(windowStart64);
            uint64_t windowEnd64 = windowStart64 + DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY;
            if (windowEnd64 > static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank)) {
                windowEnd64 = static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank);
            }
            uint32_t windowEnd = static_cast<uint32_t>(windowEnd64);
            bool windowHasToken = false;
            for (uint32_t index = 0U; index < batchTokenCount; ++index) {
                LocalTensor<int32_t> record = metaInfoTensor_[index * INT32_PER_256B];
                if (static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_SRC_SERVER)) != srcServer) {
                    continue;
                }
                uint32_t tokenIndex = static_cast<uint32_t>(record.GetValue(TOKEN_ID));
                if (tokenIndex >= windowStart && tokenIndex < windowEnd) {
                    windowHasToken = true;
                    break;
                }
            }
            if (!windowHasToken) {
                continue;
            }

            uint64_t flagSnapshotBytes = static_cast<uint64_t>(windowEnd - windowStart) * sizeof(uint64_t);
            DataCopyExtParams flagCopyParams{1U, static_cast<uint32_t>(flagSnapshotBytes), 0U, 0U, 0U};
            GM_ADDR remoteFlagAddr = GetRankWinAddrWithOffset(relayRank, dispatchFlagWinOffset_) +
                                     RelayFlagOffset(srcServer, windowStart);
            bool windowTokensReady = false;
            while (!windowTokensReady) {
                hcomm_.ReadNbi(channel, scratchAddr, remoteFlagAddr, flagSnapshotBytes);
                hcomm_.Drain(channel);
                SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID1>();
                DataCopyPad(relayFlagTensor_, scratchFlagGlobalTensor, flagCopyParams, flagCopyPadParams);
                SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID1>();

                windowTokensReady = true;
                for (uint32_t index = 0U; index < batchTokenCount; ++index) {
                    LocalTensor<int32_t> record = metaInfoTensor_[index * INT32_PER_256B];
                    if (static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_SRC_SERVER)) != srcServer) {
                        continue;
                    }
                    uint32_t tokenIndex = static_cast<uint32_t>(record.GetValue(TOKEN_ID));
                    if (tokenIndex >= windowStart && tokenIndex < windowEnd &&
                        relayFlagTensor_.GetValue(tokenIndex - windowStart) != uint64_t(1)) {
                        windowTokensReady = false;
                        break;
                    }
                }
            }
        }
    }

    HcommBatchHandle batchHandle = hcomm_.MakeBatchHandle(
        channel, hcommBatchWqeTensor_, LAYERED_HCOMM_BATCH_UB_BYTES,
        GetRankWinAddrWithOffset(relayRank, 0));
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k_),
                                              static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                          MXFP_MULTI_BASE_SIZE;
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        LocalTensor<int32_t> record = metaInfoTensor_[index * INT32_PER_256B];
        uint32_t srcServer = static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_SRC_SERVER));
        uint32_t tokenIndex = static_cast<uint32_t>(record.GetValue(TOKEN_ID));
        uint32_t dstRow = static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_DST_ROW));
        GM_ADDR remoteRecordAddr = GetRankWinAddrWithOffset(relayRank, dispatchWinOffset_) +
                                   RelayTokenOffset(srcServer, tokenIndex);
        GM_ADDR tokenDstAddr = params_.workspaceInfo.dispatchRevDataPtr +
                               static_cast<uint64_t>(dstRow) * widthA * sizeof(ActivationType);
        GM_ADDR scaleDstAddr = params_.workspaceInfo.dispatchRevScalePtr +
                               static_cast<uint64_t>(dstRow) * widthAScale * sizeof(QuantScaleOutType);
        hcomm_.ReadNbi(batchHandle, tokenDstAddr, remoteRecordAddr,
                       widthA * sizeof(ActivationType));
        hcomm_.ReadNbi(batchHandle, scaleDstAddr, remoteRecordAddr + mxQuantTokenAlignBytes_,
                       widthAScale * sizeof(QuantScaleOutType));
        if constexpr (TopkWeightsPrefetch) {
            GM_ADDR weightDstAddr = params_.workspaceInfo.dispatchRevWeightsPtr +
                                    static_cast<uint64_t>(dstRow) * weightAlignBytes_;
            hcomm_.ReadNbi(batchHandle, weightDstAddr,
                           remoteRecordAddr + mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_,
                           weightAlignBytes_);
        }
    }
    hcomm_.BatchCommit(batchHandle);
    hcomm_.Drain(batchHandle);

    if constexpr (TopkWeightsPrefetch) {
        for (uint32_t index = 0U; index < batchTokenCount; ++index) {
            LocalTensor<int32_t> record = metaInfoTensor_[index * INT32_PER_256B];
            uint32_t dstRow = static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_DST_ROW));
            uint32_t topkIndex = static_cast<uint32_t>(record.GetValue(TOPK_INDEX));
            __gm__ int32_t *weightGmI32 = reinterpret_cast<__gm__ int32_t *>(
                params_.workspaceInfo.dispatchRevWeightsPtr + static_cast<uint64_t>(dstRow) * weightAlignBytes_);
            record.SetValue(WEIGHT_INDEX, ReadGmByPassDCache(weightGmI32 + topkIndex));
        }
    }

    // 每个 expert/rank 段内的 metadata 连续。先写完全部 metadata，再发布专家就绪计数。
    uint32_t segmentStart = 0U;
    while (segmentStart < batchTokenCount) {
        LocalTensor<int32_t> firstRecord = metaInfoTensor_[segmentStart * INT32_PER_256B];
        uint32_t expertIdx = static_cast<uint32_t>(firstRecord.GetValue(DISPATCH_BATCH_EXPERT_ID));
        int32_t dstRowStart = firstRecord.GetValue(DISPATCH_BATCH_DST_ROW);
        uint32_t segmentEnd = segmentStart + 1U;
        while (segmentEnd < batchTokenCount) {
            LocalTensor<int32_t> nextRecord = metaInfoTensor_[segmentEnd * INT32_PER_256B];
            if (static_cast<uint32_t>(nextRecord.GetValue(DISPATCH_BATCH_EXPERT_ID)) != expertIdx ||
                nextRecord.GetValue(DISPATCH_BATCH_DST_ROW) !=
                    dstRowStart + static_cast<int32_t>(segmentEnd - segmentStart)) {
                break;
            }
            ++segmentEnd;
        }
        DataCopy(metaInfoGlobalTensor_[static_cast<uint64_t>(dstRowStart) * INT32_PER_256B],
                 metaInfoTensor_[segmentStart * INT32_PER_256B],
                 (segmentEnd - segmentStart) * INT32_PER_256B);
        segmentStart = segmentEnd;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();

    segmentStart = 0U;
    while (segmentStart < batchTokenCount) {
        LocalTensor<int32_t> firstRecord = metaInfoTensor_[segmentStart * INT32_PER_256B];
        uint32_t expertIdx = static_cast<uint32_t>(firstRecord.GetValue(DISPATCH_BATCH_EXPERT_ID));
        int32_t dstRowStart = firstRecord.GetValue(DISPATCH_BATCH_DST_ROW);
        uint32_t segmentEnd = segmentStart + 1U;
        while (segmentEnd < batchTokenCount) {
            LocalTensor<int32_t> nextRecord = metaInfoTensor_[segmentEnd * INT32_PER_256B];
            if (static_cast<uint32_t>(nextRecord.GetValue(DISPATCH_BATCH_EXPERT_ID)) != expertIdx ||
                nextRecord.GetValue(DISPATCH_BATCH_DST_ROW) !=
                    dstRowStart + static_cast<int32_t>(segmentEnd - segmentStart)) {
                break;
            }
            ++segmentEnd;
        }
        PublishDispatchRows(expertIdx, dstRowStart, static_cast<int32_t>(segmentEnd - segmentStart));
        segmentStart = segmentEnd;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyLocalDispatchTokens(
    int32_t dstRow, int32_t srcRank, int32_t tokenOffset, int32_t tokenCount)
{
    if (tokenCount <= 0) {
        return;
    }
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                          MXFP_MULTI_BASE_SIZE; // 输出 token-scale 长度,紧密排列
    uint32_t copyInNum = mxQuantTokenScaleAlignBytes_;
    uint32_t srcServer = static_cast<uint32_t>(srcRank) / rankNumPerServer_;

    for (int32_t i = 0; i < tokenCount; ++i) {
        int32_t topkIndex = validTopkIndexTensor_.GetValue(tokenOffset + i);
        int32_t tokenIndex = topkIndex / topK_;
        metaInfoTensor_[i * INT32_PER_256B].SetValue(RANK_ID, srcRank);
        metaInfoTensor_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        metaInfoTensor_[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
    }

    CopyTokensFromLocalRelay(dstRow, srcServer, tokenCount, widthA, widthAScale, copyInNum);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    DataCopy(metaInfoGlobalTensor_[dstRow * INT32_PER_256B], metaInfoTensor_, tokenCount * INT32_PER_256B);
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
}

// 每个 relay 通道建立一条跨专家接收流；完整批次固定为 256 token，仅宏 Wave 尾批不足 256。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveDispatchExpertRange(
    uint32_t expertBegin, uint32_t expertEnd)
{
    if (expertBegin >= expertEnd || expertBegin >= moeExpertPerRank_) {
        return;
    }
    expertEnd = expertEnd < moeExpertPerRank_ ? expertEnd : moeExpertPerRank_;
    DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                {1U, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();

    metaInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, ubBufferUsedAddr_,
                                           L1_TILE_M_256 * INT32_PER_256B);
    for (uint32_t srcRankInServer = 0U; srcRankInServer < rankNumPerServer_; ++srcRankInServer) {
        uint32_t relayRank = serverId_ * rankNumPerServer_ + srcRankInServer;
        if (relayRank >= worldSize_ || !IsChannelOwner(relayRank)) {
            continue;
        }
        bool isRemoteRelay = relayRank != rankId_;
        uint32_t remoteBatchTokenCount = 0U;

        for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
            for (uint32_t srcServer = 0U; srcServer < serverNum_; ++srcServer) {
                uint32_t srcRank = srcServer * rankNumPerServer_ + srcRankInServer;
                if (srcRank >= worldSize_) {
                    continue;
                }

                int32_t rowStart = (srcRank == 0U && expertIdx == 0U) ?
                    0 : cumsumInfoTensor_.GetValue(expertIdx * worldSize_ + srcRank - 1U);
                if (rowStart >= static_cast<int32_t>(maxOutputSize_)) {
                    continue;
                }
                __gm__ uint8_t *rankMaskBasePtr = reinterpret_cast<__gm__ uint8_t *>(
                    params_.peermemInfo.maskRecvPtr + static_cast<uint64_t>(expertIdx) * worldSize_ * maskSlotSize_ +
                    static_cast<uint64_t>(srcRank) * maskSlotSize_);
                __gm__ int32_t *rankCountPtr =
                    reinterpret_cast<__gm__ int32_t *>(rankMaskBasePtr + maskAlignSize_);
                const int32_t rankTokenCount = ReadGmByPassDCache(rankCountPtr);
                if (rankTokenCount <= 0) {
                    continue;
                }

                int32_t accumulatedRows = 0;
                for (uint32_t roundIdx = 0U;
                     roundIdx < dispatchTotalRounds_ && accumulatedRows < rankTokenCount; ++roundIdx) {
                    uint64_t roundStart =
                        static_cast<uint64_t>(roundIdx) * static_cast<uint64_t>(dispatchRoundSendTotalNum_);
                    uint32_t roundLen = (roundIdx + 1U < dispatchTotalRounds_) ?
                        dispatchRoundSendTotalNum_ : static_cast<uint32_t>(maskRouteCapacity_ - roundStart);
                    uint32_t topkIndexTensorElemCount = static_cast<uint32_t>(Ops::Base::CeilAlign(
                        static_cast<int64_t>(roundLen * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32))) /
                        sizeof(int32_t);
                    uint32_t curRoundMaskBytes = (roundLen == dispatchRoundSendTotalNum_) ?
                        dispatchRoundMaskAlignSize_ : static_cast<uint32_t>(Ops::Base::CeilAlign(
                            static_cast<int64_t>(Ops::Base::CeilDiv(roundLen, 8U)),
                            static_cast<int64_t>(ALIGN_32)));
                    uint64_t roundMaskByteOffset = roundStart / 8U;
                    GlobalTensor<uint8_t> roundMaskSrc;
                    roundMaskSrc.SetGlobalBuffer(rankMaskBasePtr + roundMaskByteOffset);
                    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
                    DataCopy(gatherMaskTensor_, roundMaskSrc, curRoundMaskBytes);
                    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
                    CreateVecIndex(topkIndexTensor_, static_cast<int32_t>(roundStart),
                                   topkIndexTensorElemCount);
                    PipeBarrier<PIPE_V>();

                    uint64_t roundTokenCount = 0U;
                    LocalTensor<uint32_t> rankMaskSlice = gatherMaskInt32Tensor_[0];
                    GatherMask(validTopkIndexTensor_, topkIndexTensor_, rankMaskSlice, true, roundLen,
                               {1, 1, 0, 0}, roundTokenCount);
                    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
                    if (roundTokenCount == 0U) {
                        continue;
                    }

                    // 发送方只覆写真实 bs 的 mask 前缀，容量尾部可能保留上次调用数据。
                    // mask 命中下标单调递增，因此每轮只消费 count 剩余量的前缀即可忽略尾部残留。
                    const int32_t remainingRankRows = rankTokenCount - accumulatedRows;
                    int32_t validRoundCount = static_cast<int32_t>(roundTokenCount);
                    if (validRoundCount > remainingRankRows) {
                        validRoundCount = remainingRankRows;
                    }
                    int32_t roundCopyCount = validRoundCount;
                    if (rowStart + accumulatedRows + roundCopyCount > static_cast<int32_t>(maxOutputSize_)) {
                        roundCopyCount = static_cast<int32_t>(maxOutputSize_) - rowStart - accumulatedRows;
                    }
                    if (roundCopyCount <= 0) {
                        accumulatedRows += validRoundCount;
                        continue;
                    }

                    if (!isRemoteRelay) {
                        for (int32_t chunkStart = 0; chunkStart < roundCopyCount;
                             chunkStart += static_cast<int32_t>(L1_TILE_M_256)) {
                            int32_t chunkRows = roundCopyCount - chunkStart;
                            if (chunkRows > static_cast<int32_t>(L1_TILE_M_256)) {
                                chunkRows = static_cast<int32_t>(L1_TILE_M_256);
                            }
                            int32_t dstRow = rowStart + accumulatedRows + chunkStart;
                            CopyLocalDispatchTokens(dstRow, static_cast<int32_t>(srcRank),
                                                    chunkStart, chunkRows);
                            PublishDispatchRows(expertIdx, dstRow, chunkRows);
                        }
                    } else {
                        for (int32_t tokenOffset = 0; tokenOffset < roundCopyCount; ++tokenOffset) {
                            int32_t topkIndex = validTopkIndexTensor_.GetValue(tokenOffset);
                            int32_t dstRow = rowStart + accumulatedRows + tokenOffset;
                            LocalTensor<int32_t> record =
                                metaInfoTensor_[remoteBatchTokenCount * INT32_PER_256B];
                            record.SetValue(RANK_ID, static_cast<int32_t>(srcRank));
                            record.SetValue(TOKEN_ID, topkIndex / static_cast<int32_t>(topK_));
                            record.SetValue(TOPK_INDEX, topkIndex % static_cast<int32_t>(topK_));
                            record.SetValue(WEIGHT_INDEX, 0);
                            record.SetValue(DISPATCH_BATCH_DST_ROW, dstRow);
                            record.SetValue(DISPATCH_BATCH_EXPERT_ID, static_cast<int32_t>(expertIdx));
                            record.SetValue(DISPATCH_BATCH_SRC_SERVER, static_cast<int32_t>(srcServer));
                            ++remoteBatchTokenCount;
                            if (remoteBatchTokenCount == DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) {
                                ReceiveRemoteDispatchBatch(relayRank, remoteBatchTokenCount);
                                remoteBatchTokenCount = 0U;
                            }
                        }
                    }
                    accumulatedRows += validRoundCount;
                }
            }
        }

        // 扫描完当前宏 Wave 后，仅通道尾批允许不足 256 token。
        if (isRemoteRelay && remoteBatchTokenCount != 0U) {
            ReceiveRemoteDispatchBatch(relayRank, remoteBatchTokenCount);
        }
    }
}

// Advance the per-expert GMM cursor and restore the current expert token count.
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGroupParams(ExpertLoopState &state,
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
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / ACTIVATION_N_HALF / C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / ACTIVATION_N_HALF / B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
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
                while (AscendC::GetSystemCycle() - st < GM_FLAG_POLL_BACKOFF_CYCLES) {
                }
            }
        }

        uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            expertRevNumsGlobalTensor_[offsetInCnt]);
        Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor_.GetValue(offsetInCnt);
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
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateSharedGroupParams(ExpertLoopState &state,
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
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / ACTIVATION_N_HALF / C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / ACTIVATION_N_HALF / B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
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
//                     GMM1 始终写 gmm1MmadResPtr；
//                     GMM2 始终写 gmm2MmadResPtr。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
                                                                                          const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        if constexpr (ENABLE_A8W4 || TopkWeightsPrefetch) {
            gmmAddrInfo.gmm1OutGlobal =
                params_.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.metaInfoGlobal = params_.workspaceInfo.metaInfoPtr;
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.dispatchRevDataPtr + Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
                                   Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);

        uint32_t expertIdx = static_cast<uint32_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset));
        if constexpr (TopkWeightsPrefetch) {
            gmmAddrInfo.gmm1TileStatus =
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr) +
                static_cast<uint64_t>(expertIdx) * params_.tilingData->maxTilesPerExpert * INT_CACHELINE;
        }
        gmmAddrInfo.bGlobal = GetExpertWeightAddr<Weight1Type>(
            params_.bGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.bScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B_SCALE_OFFSET>(state.baseOffset));

        if constexpr (g_coreType == AIV) {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(state.baseOffset),
                Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                Get<IDX_FLAG_OFFSET>(state.baseOffset),
                0L,
                0L,
                0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        gmmAddrInfo.metaInfoGlobal = params_.workspaceInfo.metaInfoPtr +
                                     static_cast<uint64_t>(state.expertBeforeCnt) * META_INFO_SIZE * sizeof(int32_t);
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.activationQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.activationQuantScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        uint32_t expertIdx = static_cast<uint32_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset));
        gmmAddrInfo.bGlobal = GetExpertWeightAddr<Weight1Type>(
            params_.b2GmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B2_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.b2ScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
        uint64_t expertSyncSlotOffset = static_cast<uint64_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset)) *
                                        params_.tilingData->combineSyncSlotCountPerExpert;
        gmmAddrInfo.gmm2CombineSyncCounter = (__gm__ int32_t *)params_.workspaceInfo.gmm2CombineSyncCounterPtr +
                                             expertSyncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
        gmmAddrInfo.gmm2CombineLogicalCoreCount = blockNum_;
    }
    gmmAddrInfo.activationToGmm2Flag = (__gm__ int32_t *)params_.workspaceInfo.flagActivationToGmm2Ptr +
                                       Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t *)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                     Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmAddrInfo.gmmToEpilogueFlag = gmmToEpilogueFlag_;
    }
}

// ==================================================================================
// UpdateSharedGlobalBuffer：共享专家专用，地址来自 shared* workspace，flags 为 nullptr。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateSharedGlobalBuffer(
    GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertInputDataPtr;
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertInputScalePtr;
        if constexpr (ENABLE_A8W4) {
            gmmAddrInfo.gmm1OutGlobal = params_.workspaceInfo.sharedExpertGmm1OutPtr +
                                        Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        uint32_t expertIdx = static_cast<uint32_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset));
        gmmAddrInfo.bGlobal = GetExpertWeightAddr<Weight1Type>(
            params_.sharedBGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.sharedBScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B_SCALE_OFFSET>(state.baseOffset));
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.sharedExpertResultPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertActivationDataPtr +
                              Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertActivationScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        uint32_t expertIdx = static_cast<uint32_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset));
        gmmAddrInfo.bGlobal = GetExpertWeightAddr<Weight1Type>(
            params_.sharedB2GmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B2_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.sharedB2ScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
    }
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmAddrInfo.gmmToEpilogueFlag = gmmToEpilogueFlag_;
    }
}

// =============================================
// ResetGmm2CombineSyncCounters：重置 GMM2→Combine 同步计数器
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetGmm2CombineSyncCounters()
{
    if constexpr (g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(params_.tilingData->combineSyncSlotCountPerExpert *
                                                     moeExpertPerRank_ * static_cast<uint64_t>(INT_CACHELINE));
        int32_t coreLen, coreOffset;
        TilingByCore(totalCounters, coreLen, coreOffset);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.gmm2CombineSyncCounterPtr);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
        for (int32_t resetElementOffset = 0; resetElementOffset < coreLen;
             resetElementOffset += resetBatchElementCount_) {
            int32_t currentBatchElementCount = coreLen - resetElementOffset < resetBatchElementCount_ ?
                                                   coreLen - resetElementOffset :
                                                   resetBatchElementCount_;
            DataCopyExtParams counterCopyParams{
                1U, static_cast<uint32_t>(currentBatchElementCount * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(gmm2CombineSyncCounterGm[coreOffset + resetElementOffset], resetTensor_, counterCopyParams);
        }
    }
}

// =============================================
// InitCombineBuffers：初始化 Combine 所需的 buffer 大小
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitCombineBuffers()
{
    if constexpr (g_coreType == AIV) {
        // Combine 仅在当前 AIV1 完成 Dispatch 张量使用后启动，并从 UB 首地址重建视图。
        // Hcomm 使用前 512 字节；Combine metadata 和计算临时区从其后开始复用 UB。
        LocalTensor<uint8_t> hcommTensor = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
        hcomm_.Init(hcommTensor, ALIGN_512);
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GetCombineRankRange(
    uint32_t expertIdx, uint32_t dstRank, uint32_t &rowStart, uint32_t &tokenCount)
{
    uint64_t expertBegin = static_cast<uint64_t>(expertIdx) * worldSize_;
    uint64_t rankEndIndex = expertBegin + dstRank;
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        cumsumInfoGlobalTensor_[rankEndIndex]);
    int32_t rankEnd = cumsumInfoGlobalTensor_.GetValue(rankEndIndex);

    int32_t expertStart = 0;
    if (expertBegin != 0U) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            cumsumInfoGlobalTensor_[expertBegin - 1U]);
        expertStart = cumsumInfoGlobalTensor_.GetValue(expertBegin - 1U);
    }
    int32_t rankStart = expertStart;
    if (dstRank != 0U) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            cumsumInfoGlobalTensor_[rankEndIndex - 1U]);
        rankStart = cumsumInfoGlobalTensor_.GetValue(rankEndIndex - 1U);
    }
    rowStart = static_cast<uint32_t>(rankStart - expertStart);
    tokenCount = static_cast<uint32_t>(rankEnd - rankStart);
    return tokenCount != 0U;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineRank(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State, uint32_t expertIdx,
    CombineImpl::LayeredCombineBatchState &batchState)
{
    constexpr bool IsQuantized = CombineQuantMode != COMBINE_NO_QUANT;
    uint32_t rowStart = 0U;
    uint32_t tokenCount = 0U;
    if (!GetCombineRankRange(expertIdx, batchState.dstRankId, rowStart, tokenCount)) {
        return;
    }
    AscendC::SetCtrlSpr<60, 60>(0);

    uint32_t mExpert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
    GroupSyncSlotLayout slotLayout = CalcGroupSyncSlotLayout(mExpert, blockNum_);
    __gm__ int32_t *expertCounterBase =
        reinterpret_cast<__gm__ int32_t *>(gmmAddrInfo.gmm2CombineSyncCounter);

    uint32_t processedCount = 0U;
    while (processedCount < tokenCount) {
        uint32_t currentRow = rowStart + processedCount;
        uint32_t targetGroup = currentRow / L1_TILE_M_256;
        uint32_t firstSyncSlot = 0;
        uint32_t syncSlotCount = 0;
        GetGroupSyncSlotRange(targetGroup, slotLayout, firstSyncSlot, syncSlotCount);
        __gm__ int32_t *counterAddr = GetCombineSyncCounterAddress(expertCounterBase, firstSyncSlot);
        if (AscendC::ReadGmByPassDCache(counterAddr) < static_cast<int32_t>(nTilesPerGroup)) {
            continue;
        }

        uint32_t groupEndRow = (targetGroup + 1U) * L1_TILE_M_256;
        groupEndRow = groupEndRow < mExpert ? groupEndRow : mExpert;
        uint32_t batchCount = groupEndRow - currentRow;
        uint32_t remainingTokens = tokenCount - processedCount;
        batchCount = batchCount < remainingTokens ? batchCount : remainingTokens;

        int64_t offset = ALIGN_512;
        LocalTensor<int32_t> metaInfoTensor(TPosition::VECIN, offset, batchCount * META_INFO_SIZE);
        offset += batchCount * META_INFO_SIZE * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> metaInfoGm;
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.metaInfoPtr +
            static_cast<uint64_t>(gmm2State.expertBeforeCnt + currentRow) * META_INFO_SIZE * sizeof(int32_t)));
        SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID1>();
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, batchCount * META_INFO_SIZE);

        CombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t, true, IsQuantized>(
            currentRow, batchCount, k_, expertIdx, rankId_, gmmAddrInfo.gmm2OutGlobal, params_, metaInfoTensor,
            combineUbTensorSize_, offset, quantTokenSizeBytes, batchState);
        processedCount += batchCount;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LockOwnedRemoteChannels()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1U) {
        return;
    }
    // Wave 内 Dispatch 一级写、Dispatch 二级读和 Combine 写始终使用同一套固定归属核。
    // 在 Wave 开始前一次性取得全部归属通道，整个 Wave 内不重复 Lock/Unlock。
    for (uint32_t rank = 0; rank < worldSize_; ++rank) {
        if (rank == rankId_ || !IsChannelOwner(rank)) {
            continue;
        }
        hcomm_.Lock(GetUrmaCommHandle(mc2Context_, rank, rankId_));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DrainAndUnlockOwnedRemoteChannels()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1U) {
        return;
    }
    // Drain 全部归属远端 rank，不能只处理最后一个专家的活跃 rank。
    // 否则当最后一个专家无 token 时，早期专家的 WriteNbi 可能遗留到流水结束之后。
    // Drain 后 Unlock 发布通道 counter 并释放所有权，允许后续同步阶段换核使用同一通道。
    for (uint32_t rank = 0; rank < worldSize_; ++rank) {
        if (rank == rankId_ || !IsChannelOwner(rank)) {
            continue;
        }
        ChannelHandle channel = GetUrmaCommHandle(mc2Context_, rank, rankId_);
        HcommBatchHandle batchHandle = hcomm_.MakeBatchHandle(
            channel, hcommBatchWqeTensor_, LAYERED_HCOMM_BATCH_UB_BYTES,
            GetRankWinAddrWithOffset(rank, 0));
        // 同一通道可能同时残留 Dispatch 一级和 Combine 的已提交批次，统一按批量句柄等待完成。
        hcomm_.Drain(batchHandle);
        hcomm_.Unlock(channel);
    }
}

// 按目标 rank 跨专家积累 Combine 批次，并在宏 Wave 尾提交不足 256-token 的尾批。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineExpertRange(
    ExpertLoopState waveBeginState, uint32_t expertBegin, uint32_t expertEnd)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1U || expertBegin >= expertEnd) {
        return;
    }

    // 每个目标 rank 在一个计算宏 Wave 内只创建一个 batch handle，WQE 可跨专家积累。
    // 满 256 token 提交，宏 Wave 尾不足 256 的部分立即 flush，保证下游不会为凑包等待。
    for (uint32_t dstRank = 0U; dstRank < worldSize_; ++dstRank) {
        if (!IsChannelOwner(dstRank)) {
            continue;
        }
        CombineImpl::LayeredCombineBatchState batchState{};
        batchState.dstRankId = dstRank;
        if (dstRank != rankId_) {
            ChannelHandle channel = GetUrmaCommHandle(mc2Context_, dstRank, rankId_);
            batchState.batchHandle = hcomm_.MakeBatchHandle(
                channel, hcommBatchWqeTensor_, LAYERED_HCOMM_BATCH_UB_BYTES,
                GetRankWinAddrWithOffset(dstRank, 0));
        }

        ExpertLoopState combineState = waveBeginState;
        GMMAddrInfo combineAddrInfo{};
        for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
            if (!UpdateGroupParams<AddrUpdateMode::GMM2>(combineState, expertIdx)) {
                continue;
            }
            UpdateGlobalBuffer<AddrUpdateMode::GMM2>(combineAddrInfo, combineState);
            ProcessCombineRank(combineAddrInfo, combineState, expertIdx, batchState);
        }
        if (dstRank != rankId_ && batchState.pendingTokenCount != 0U) {
            hcomm_.BatchCommit(batchState.batchHandle);
            batchState.pendingTokenCount = 0U;
        }
    }
}

// =============================================
// UnpermuteBuffInit：Unpermute中使用的buffer申请
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteBuffInit()
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)),
                                                    static_cast<uint32_t>(ALIGN_32));
    int32_t num =
        worldSize_ *
        Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * moeExpertPerRank_), static_cast<uint32_t>(ALIGN_128)) *
        sizeof(int32_t);
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t fixedUbBeforeTopK = dataResBufAlign + dataResFp32BufAlign;
    uint32_t scaleUbCost = 0;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        // 预算公式必须与下方实际张量完全一致：每 32 个元素一个 scale，再扩展为两份。
        uint32_t scaleNum =
            Ops::Base::CeilDiv(static_cast<uint32_t>(k_), static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        scaleUbCost = Ops::Base::CeilAlign(
                          static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DEQUANT_SCALE_EXPAND),
                          static_cast<uint32_t>(ALIGN_32)) +
                      Ops::Base::CeilAlign(
                          static_cast<uint32_t>(scaleNum * sizeof(float) * DEQUANT_SCALE_EXPAND),
                          static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t fixedUbCost = fixedUbBeforeTopK + scaleUbCost;
    uint32_t availUb = fixedUbCost < SEND_MASK_UB_LIMIT ? SEND_MASK_UB_LIMIT - fixedUbCost : 0U;
    uint32_t perTokenBytes = topK_ * sizeof(float);
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        perTokenBytes += topK_ * sizeof(bfloat16_t);
    }
    constexpr uint32_t bufferCount = Std::IsSame<TopkWeightsType, bfloat16_t>::value ? 2U : 1U;
    constexpr uint32_t alignmentSlackBytes = bufferCount * (static_cast<uint32_t>(ALIGN_32) - 1U);
    uint32_t chunkUb = availUb > alignmentSlackBytes ? availUb - alignmentSlackBytes : 0U;
    topKWeightsChunkLen_ = (perTokenBytes > 0) ? chunkUb / perTokenBytes : m_;
    if (topKWeightsChunkLen_ == 0) {
        topKWeightsChunkLen_ = 1;
    }
    if (topKWeightsChunkLen_ > m_) {
        topKWeightsChunkLen_ = m_;
    }
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(topKWeightsChunkLen_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    // Tensor用处：Unpermute 函数用于存储mte2搬入token；
    // Tensor大小：大小为3 *
    // 单个token长度，2块是用于mte2搬运的doubleBuffer，1块是用于存储累加计算Cast完的输出结果，用于搬出；
    uint32_t dataResAddr = 0;
    uint32_t dataResSize = dataResBufAlign / sizeof(bfloat16_t);
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResSize);
    // Tensor用处：Unpermute 函数用于存储token Cast 目的Tensor；
    // Tensor大小：dataResTensor_开设大小乘以BF16_TO_FP32；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    uint32_t dataResFp32Size = dataResFp32BufAlign / sizeof(float);
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32Size);
    // Tensor用处：用于存储topKWeight；
    // Tensor大小：m_ * topK_ * sizeof(float) align到32字节对齐；
    uint32_t topKWeightsAddr = dataResFp32Addr + dataResFp32BufAlign;
    uint32_t topKWeightsSize = topKWeightsBufAlign / sizeof(float);
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr, topKWeightsSize);
    uint32_t tempAddr = topKWeightsAddr + topKWeightsBufAlign;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(k_), static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）
        // Tensor大小：scaleNum * sizeof(bfloat16_t) * DEQUANT_SCALE_EXPAND，用于 scale 扩展。
        uint32_t bf16ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DEQUANT_SCALE_EXPAND),
                                 static_cast<uint32_t>(ALIGN_32));
        bf16ScaleTensor_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）
        // Tensor大小：scaleNum * sizeof(float) * DEQUANT_SCALE_EXPAND，用于 scale 扩展。
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(
            static_cast<uint32_t>(scaleNum * sizeof(float) * DEQUANT_SCALE_EXPAND), static_cast<uint32_t>(ALIGN_32));
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }
    topKWeightsTempAddr_ = tempAddr;
}

// ===============================================================
// UnpermuteSharedExpert：共享专家结果累加到当前 token 的 fp32 累加器
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteSharedExpert(int32_t tokenIdx)
{
    LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
    LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
    LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
    LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer((__gm__ bfloat16_t *)params_.workspaceInfo.sharedExpertResultPtr);
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        auto event = (sharedIdx % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (sharedIdx % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (sharedIdx % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
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
// Unpermute：对于各个专家还回来token的后处理，进行对应scale相乘与累加
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t *)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t *)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t chunkStart = coreOffset; chunkStart < coreLen + coreOffset;) {
        int32_t chunkEnd = chunkStart + static_cast<int32_t>(topKWeightsChunkLen_);
        if (chunkEnd > coreLen + coreOffset) {
            chunkEnd = coreLen + coreOffset;
        }
        int32_t chunkTokenCnt = chunkEnd - chunkStart;
        if constexpr (!TopkWeightsPrefetch) {
            if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
                GlobalTensor<float> topKWeightsGlobalTensor_;
                topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float *)params_.probsGmAddr);
                DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(chunkTokenCnt * topK_ * sizeof(float)), 0U,
                                                0U, 0U};
                DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
                DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_[chunkStart * topK_], copyParams,
                            copyPadParams);
                SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
            }
            if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
                uint32_t tempBufAlign = Ops::Base::CeilAlign(
                    static_cast<uint32_t>(topKWeightsChunkLen_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
                LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, topKWeightsTempAddr_,
                                                  tempBufAlign / sizeof(bfloat16_t));
                GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
                topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t *)params_.probsGmAddr);
                DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(chunkTokenCnt * topK_ * sizeof(bfloat16_t)),
                                                0U, 0U, 0U};
                DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
                DataCopyPad(tempLocal, topkWeightsGlobalTensor[chunkStart * topK_], copyParams, copyPadParams);
                SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
                Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, chunkTokenCnt * topK_);
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
            }
        }
        for (int32_t tokenIdx = chunkStart; tokenIdx < chunkEnd; tokenIdx++) {
            int32_t localIdx = tokenIdx - chunkStart;
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
            LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
            LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
            LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
            LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
            for (int32_t expId = 0; expId < topK_; ++expId) {
                auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
                auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
                auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
                if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                    DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * k_], k_);
                    SetFlag<AscendC::HardEvent::MTE2_V>(event);
                    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                    SetFlag<AscendC::HardEvent::S_V>(event);
                    WaitFlag<AscendC::HardEvent::S_V>(event);
                    Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
                } else {
                    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
                    uint32_t quantTokenSize = k_ + nScale;
                    uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
                    WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                    DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * quantEleNum], quantEleNum);
                    SetFlag<AscendC::HardEvent::MTE2_V>(event);
                    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                    using Fp8Type = typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t,
                                                              fp8_e5m2_t>::type;
                    SetFlag<AscendC::HardEvent::S_V>(event);
                    WaitFlag<AscendC::HardEvent::S_V>(event);
                    Mxfp8::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_, fp32ScaleTensor_,
                                                             nScale, k_);
                }
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
            // 共享专家结果累加（直接加，不乘 topk_weight）
            if (sharedExpertNum_ > 0) {
                UnpermuteSharedExpert(tokenIdx);
            }
            // fp32 -> bf16
            Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[tokenIdx * k_], dataResTensor_, k_);
        }
        chunkStart = chunkEnd;
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

// ==============================================================================================
// CrossRankSyncInWorldSize：全卡同步，rank 槽位之后的独立区域记录各 AIV 的 syncCnt。
// ==============================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CrossRankSyncInWorldSize(
    bool reuseSendMaskChannelLock)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t *syncRank = (__gm__ int32_t *)params_.peermemInfo.rankSyncInWorldPtr;
    int64_t syncCountOffset = CalcUrmaSyncCountOffset(static_cast<int64_t>(worldSize_));
    __gm__ int32_t *syncCount = (__gm__ int32_t *)(params_.peermemInfo.rankSyncInWorldPtr + syncCountOffset +
                                                  aivCoreIdx_ * PEERMEM_SYNC_SLOT_SIZE);
    int count = ReadGmByPassDCache(syncCount) + 1;
    WriteGmByPassDCache(syncCount, count);
    // 先向本 AIV 负责的所有 peer 发出通知，再进入等待。256P 场景下可同时维持多个
    // channel 在途，避免原实现逐 peer“写-等-Drain”造成的串行握手。
    for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
        if (rankIndex == rankId_) {
            continue;
        }
        ChannelHandle channel = GetUrmaCommHandle(mc2Context_, rankIndex, rankId_);
        if (!reuseSendMaskChannelLock) {
            // Wave 结束后同步任务重新按物理 AIV 分核，可能不同于 Wave 的固定归属核，因此先取得通道锁。
            hcomm_.Lock(channel);
        }
        __gm__ int32_t *syncRemoteAddr = (__gm__ int32_t *)(g_winRankAddr_[rankIndex]) + rankId_ * 16;
        hcomm_.WriteNbi(channel, (GM_ADDR)syncRemoteAddr, (GM_ADDR)syncCount,
                        static_cast<int64_t>(sizeof(int32_t)));
    }
    for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
        if (rankIndex == rankId_) {
            continue;
        }
        ChannelHandle channel = GetUrmaCommHandle(mc2Context_, rankIndex, rankId_);
        auto syncCheck = syncRank + rankIndex * 16;
        GmSignalWaitBarrier(syncCheck, count);
        // Drain 后 Unlock 发布通道 counter。第一次调用释放 SendMask 延续的锁，第二次调用释放本函数取得的锁。
        hcomm_.Drain(channel);
        hcomm_.Unlock(channel);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

// ===============================================================
// SharedExpertCopyInput：从原始 bf16 输入量化后写入共享专家专用缓冲区
//   源: aGmAddr [bs × h] bf16（layered URMA 模式下 quantTokenScalePtr 未填充，需直接从原始输入量化）
//   目标: sharedExpertInputDataPtr [bs × h] fp8 连续, sharedExpertInputScalePtr [bs × scaleN] 连续
//   AIV 执行，在 AIC GMM1 开始前调用
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SharedExpertCopyInput()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t currentNum;
    int32_t currentOffset;
    TilingByCore(m_, currentNum, currentOffset, 1);
    uint32_t H = k_;

    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    uint32_t xInAddr = ALIGN_512;
    uint32_t xInSize = Ops::Base::CeilAlign(H, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    LocalTensor<bfloat16_t> xInBuf0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAddr, xInSize / sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> xInBuf1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAddr + xInSize, xInSize / sizeof(bfloat16_t));
    uint32_t mxTempAddr = xInAddr + xInSize * 2;
    LocalTensor<uint16_t> mxTempBuf =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempAddr, MX_QUANT_TEMP_UB_BYTES / sizeof(uint16_t));
    uint32_t xOutAddr = mxTempAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t xOutSize =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    LocalTensor<ActivationType> xOutBuf0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutAddr, xOutSize / sizeof(ActivationType));
    LocalTensor<ActivationType> xOutBuf1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutAddr + xOutSize, xOutSize / sizeof(ActivationType));

    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<ActivationType> dataDstGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleDstGlobalTensor;
    dataDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params_.workspaceInfo.sharedExpertInputDataPtr));
    scaleDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleOutType *>(params_.workspaceInfo.sharedExpertInputScalePtr));

    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(H * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        int32_t tokenIdx = currentOffset + index;
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInBuf = (index % DOUBLE_BUFFER == 0) ? xInBuf0 : xInBuf1;
        auto xOutBuf = (index % DOUBLE_BUFFER == 0) ? xOutBuf0 : xOutBuf1;

        srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            params_.aGmAddr + static_cast<uint64_t>(tokenIdx) * H * sizeof(bfloat16_t)));
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInBuf, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);

        __ubuf__ bfloat16_t *srcAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(xInBuf.GetPhyAddr());
        __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxTempBuf.GetPhyAddr());
        __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
            mxTempBuf[Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))]
                .GetPhyAddr());
        __ubuf__ int8_t *outDataAddr = reinterpret_cast<__ubuf__ int8_t *>(xOutBuf.GetPhyAddr());
        __ubuf__ uint16_t *mxScaleAddr =
            reinterpret_cast<__ubuf__ uint16_t *>(xOutBuf[mxQuantTokenAlignBytes_].GetPhyAddr());

        Quant::ComputeMaxExp(srcAddr, maxExpAddr, H);
        Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, mxQuantScaleNumAlignPerToken_);
        if constexpr (QuantMode == E2M1_QUANT) {
            Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        } else {
            Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        LocalTensor<QuantScaleOutType> bufScale =
            xOutBuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(dataDstGlobalTensor[tokenIdx * widthA], xOutBuf,
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
// GroupMatmulWithActivationQuant：按实现路径分发到 A8W4 或 generic GMM1。
//                            A8W4 由 ENABLE_A8W4 控制；generic 路径的 subBlockIdx 判断已下沉到函数内部。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithActivationQuant(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx, int32_t &vecSetSyncCom,
    int32_t &gmTileSequence)
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
            RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                        L1_TILE_M_256, false, IsShared>(sharedEpilogueOp_, params_, state.problemShape, gmmAddrInfo,
                                                        startBlockIdx_, gmTileSequence, state.expertBeforeCnt,
                                                        expertIdx);
        } else {
            if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
                params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
                RunGmm1Generic<QuantOutType, ActivationQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                               QuantScaleOutType, true, GMM1_TILE_M, L1_TILE_M_256, false, IsShared,
                               GMM1_INTERLEAVED>(
                    sharedEpilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            } else {
                RunGmm1Generic<QuantOutType, ActivationQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                               QuantScaleOutType, false, GMM1_TILE_M, L1_TILE_M_256, false, IsShared,
                               GMM1_INTERLEAVED>(
                    sharedEpilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            }
        }
    } else {
        // MoE 专家走 prefetch 路径
        if constexpr (ENABLE_A8W4) {
            RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                        EPILOGUE_TILE_M, TopkWeightsPrefetch, IsShared>(
                epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, gmTileSequence,
                state.expertBeforeCnt, expertIdx);
        } else {
            if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
                params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
                // NZ format (A8W8_NZ / A4W4_NZ): isWeightNZ=true, EpilogueElementA 由 ActivationQuantOutType
                // 自动处理类型提升
                RunGmm1Generic<QuantOutType, ActivationQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                               QuantScaleOutType, true, GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch, IsShared,
                               GMM1_INTERLEAVED>(
                    epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            } else {
                // Generic: fp8/fp4 activation × fp8/fp4 weight in ND format (includes A4W4 ND)
                RunGmm1Generic<QuantOutType, ActivationQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                               QuantScaleOutType, false, GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch, IsShared,
                               GMM1_INTERLEAVED>(
                    epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            }
        }
    }
}

// ===============================================================
// GroupMatmulWithCombine：先按实现路径分发，再按 combine 模式分发。
// IsShared=true 时跳过 activation flag 等待和 Combine 后处理，供共享专家使用。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithCombine(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                    L1_TILE_M_256, TopkWeightsPrefetch, IsShared, true>(state.problemShape, gmmAddrInfo,
                                                                       startBlockIdx_);
    } else {
        // A8W8_NZ / Generic 共用 RunGmm2Generic，仅 LayoutB 不同（ZN/ND）。
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            RunGmm2Generic<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                           QuantScaleOutType, true, true, L1_TILE_M_256, TopkWeightsPrefetch, IsShared,
                           GMM1_INTERLEAVED, false>(
                state.problemShape, gmmAddrInfo, startBlockIdx_);
        } else {
            RunGmm2Generic<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                           QuantScaleOutType, false, true, L1_TILE_M_256, TopkWeightsPrefetch, IsShared,
                           GMM1_INTERLEAVED, false>(
                state.problemShape, gmmAddrInfo, startBlockIdx_);
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm1(
    const TupleShape &initShape, const BlockOffset &initOffset, int32_t &gmTileSequence)
{
    sharedEpilogueOp_.Init({.yGmAddr = params_.workspaceInfo.sharedExpertActivationDataPtr,
                            .yScaleGmAddr = params_.workspaceInfo.sharedExpertActivationScalePtr,
                            .clampLimit = params_.tilingData->clampLimit,
                            .actMode = params_.tilingData->actMode,
                            .actSubMode = params_.tilingData->actSubMode,
                            .activationAlpha = params_.tilingData->activationAlpha,
                            .activationBeta = params_.tilingData->activationBeta});

    GMMAddrInfo sharedGmm1AddrInfo{};
    ExpertLoopState sharedGmm1State{initShape, initOffset, 0};
    int32_t vecSetSyncCom = 0;
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm1State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM1>(sharedGmm1AddrInfo, sharedGmm1State);
        GroupMatmulWithActivationQuant<true>(sharedGmm1AddrInfo, sharedGmm1State, sharedIdx, vecSetSyncCom,
                                             gmTileSequence);
    }
    EndSync<GMM1_INTERLEAVED>(vecSetSyncCom, gmm1PingPongIdx_);
    gmm1PingPongIdx_ = 0U;
    startBlockIdx_ = 0; // 共享专家GMM1修改了startBlockIdx_，重置给GMM1使用
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm2(
    const TupleShape &initShape, const BlockOffset &initOffset)
{
    GMMAddrInfo sharedGmm2AddrInfo{};
    ExpertLoopState sharedGmm2State{initShape, initOffset, 0};
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm2State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM2>(sharedGmm2AddrInfo, sharedGmm2State);
        GroupMatmulWithCombine<true>(sharedGmm2AddrInfo, sharedGmm2State);
    }
    SyncAll<false>();
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareDispatch(
    uint32_t firstWaveExpertCount)
{
    // 全量量化和跨全部专家的目标 Server 队列已在 Wave 前由全部 AIV 完成。
    // Wave 入口仅发布专家计数并一次性提交一级通信，一级批次不由计算 Wave 截断。
    for (uint32_t expertIdx = 0U; expertIdx < firstWaveExpertCount; ++expertIdx) {
        PublishExpertTokenCount(expertIdx);
    }
    // 先提交一级大批次，再计算后续专家计数；BatchCommit 后网络传输可与本地计数重叠。
    SendDispatchRelayQueues();
    for (uint32_t expertIdx = firstWaveExpertCount; expertIdx < moeExpertPerRank_; ++expertIdx) {
        PublishExpertTokenCount(expertIdx);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, int32_t &vecSetSyncCom, int32_t &gmTileSequence,
    uint32_t expertIdx)
{
    if (!UpdateGroupParams<AddrUpdateMode::GMM1>(state, expertIdx)) {
        return;
    }
    UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmmAddrInfo, state);
    GroupMatmulWithActivationQuant(gmmAddrInfo, state, expertIdx, vecSetSyncCom, gmTileSequence);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RunGmm2ForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx)
{
    if (!UpdateGroupParams<AddrUpdateMode::GMM2>(state, expertIdx)) {
        return;
    }
    UpdateGlobalBuffer<AddrUpdateMode::GMM2>(gmmAddrInfo, state);
    GroupMatmulWithCombine(gmmAddrInfo, state);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishOrWaitGmm1AllDone()
{
    if constexpr (!TopkWeightsPrefetch) {
        return;
    }
    int32_t allDoneTag = static_cast<int32_t>(moeExpertPerRank_ + 1U);
    __gm__ int32_t *allDoneAddr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr) +
                                  static_cast<uint64_t>(moeExpertPerRank_) *
                                      params_.tilingData->maxTilesPerExpert * INT_CACHELINE;
    if constexpr (g_coreType == AIV) {
        constexpr uint32_t epilogueSubIdx = ENABLE_A8W4 ? 1U : 0U;
        if (subBlockIdx_ == epilogueSubIdx) {
            AscendC::WriteGmByPassDCache(allDoneAddr, allDoneTag);
        }
    } else {
        WaitUntilGmFlagEquals(allDoneAddr, allDoneTag);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessMoeExpertWave(
    const TupleShape &initShape, const BlockOffset &initOffset, int32_t &gmTileSequence)
{
    DispatchBuffInit();
    InitCombineBuffers();
    LockOwnedRemoteChannels();
    ExpertLoopState gmm1State{initShape, initOffset, 0};
    ExpertLoopState gmm2State{initShape, initOffset, 0};
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};
    int32_t vecSetSyncCom = 0;
    // 计算宏 Wave 只负责组织 GMM 阶段，不再定义一级通信提交边界或二级 256-token ready 边界。
    // 小负载控制在 1～2 个有效 Wave；中等负载约 6 个；单专家很大的吞吐场景约 4 个。
    const uint32_t targetWaveCount = CalcTargetWaveCount();
    const uint32_t firstWaveExpertCount = CalcFirstWaveExpertCount(targetWaveCount);
    const uint32_t steadyWaveExpertCount =
        CalcSteadyWaveExpertCount(firstWaveExpertCount, targetWaveCount);
    uint32_t currentWaveBegin = 0U;
    uint32_t currentWaveEnd = firstWaveExpertCount;

    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1U) {
            PrepareDispatch(firstWaveExpertCount);
            // 首个宏 Wave 就绪后即可启动计算，不再等待全部专家的二级 Dispatch。
            ReceiveDispatchExpertRange(currentWaveBegin, currentWaveEnd);
        }
    }

    while (currentWaveBegin < moeExpertPerRank_) {
        const uint32_t waveStartBlockIdx = startBlockIdx_;
        const uint32_t nextWaveBegin = currentWaveEnd;
        const uint32_t proposedWaveEnd = nextWaveBegin + steadyWaveExpertCount;
        const uint32_t nextWaveEnd = proposedWaveEnd < moeExpertPerRank_ ?
            proposedWaveEnd : moeExpertPerRank_;

        if constexpr (g_coreType == AIV) {
            if (subBlockIdx_ == 1U && nextWaveBegin < moeExpertPerRank_) {
                // 通信核预取下一宏 Wave；AIC/AIV0 同时计算当前宏 Wave。
                ReceiveDispatchExpertRange(nextWaveBegin, nextWaveEnd);
            }
        }

        for (uint32_t expertIdx = currentWaveBegin; expertIdx < currentWaveEnd; ++expertIdx) {
            RunGmm1ActivationForExpert(
                gmm1State, gmm1AddrInfo, vecSetSyncCom, gmTileSequence, expertIdx);
        }
        const uint32_t gmm1EndBlockIdx = startBlockIdx_;
        ExpertLoopState combineWaveBeginState = gmm2State;
        for (uint32_t expertIdx = currentWaveBegin; expertIdx < currentWaveEnd; ++expertIdx) {
            RunGmm2ForExpert(gmm2State, gmm2AddrInfo, expertIdx);
        }
        ProcessCombineExpertRange(combineWaveBeginState, currentWaveBegin, currentWaveEnd);
        if constexpr (!ENABLE_A8W4) {
            const bool hasNextWave = currentWaveEnd < moeExpertPerRank_;
            const bool fixedRoleResonance =
                startBlockIdx_ == waveStartBlockIdx && gmm1EndBlockIdx != waveStartBlockIdx;
            if (hasNextWave && fixedRoleResonance) {
                startBlockIdx_ = gmm1EndBlockIdx;
            }
        }
        currentWaveBegin = nextWaveBegin;
        currentWaveEnd = nextWaveEnd;
    }

    if constexpr (TopkWeightsPrefetch) {
        PublishOrWaitGmm1AllDone();
    } else {
        EndSync<GMM1_INTERLEAVED>(vecSetSyncCom, gmm1PingPongIdx_);
    }
    gmm1PingPongIdx_ = 0U;
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1U) {
            PipeBarrier<PIPE_ALL>();
            ExpertTokenNumCopyOut();
            DrainAndUnlockOwnedRemoteChannels();
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();
    SendMaskCal();        // 源卡按全局专家计算掩码，并推送到目标专家卡
    ResetFlagList();      // 清零工作空间中的同步标志
    ResetDispatchState(); // 清零跨 Server URMA Dispatch 队列和中继就绪标志
    if (sharedExpertNum_ > 0) {
        SharedExpertCopyInput();
    }
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>(); // AIC 等待同步标志清零完成

    // 共享专家 GMM1+SwiGLU 前移到 MoE 之前执行，复用 MoE 函数。
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int32_t gmTileSequence = 0; // Specialized A8W4/A4W4 GMM1 AIC-AIV1 tile ready sequence.
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm1(initShape, initOffset, gmTileSequence);
        SyncAll<false>();
    }

    if constexpr (g_coreType == AIV) {
        // 全部物理 AIV 先量化各自 token 分片，再批量发布本 Server ready flag 并构造一级队列。
        // 量化/flag 的 MTE3 与 Scalar 建队列允许重叠；CrossRankSync 尾部统一排空并同步全部 AIV。
        DispatchPrepareBuffInit();
        QuantizeLocalTokensToRelay();
        SetLocalRelayReadyFlags();
        BuildDispatchRelayQueues();
    }

    CrossRankSyncInWorldSize(true); // 同一物理 AIV 继续使用 SendMask 已取得的锁，并在同步结束后释放。

    // URMA Layered 的 A8W8、A4W4 和 A8W4 共用同一专家 Wave 状态机。
    ProcessMoeExpertWave(initShape, initOffset, gmTileSequence);

    // 3.5: 共享专家 GMM2 (MoE GMM2 之后, 复用 MoE 函数)
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm2(initShape, initOffset);
    }

    // 4. 本卡数据Unpermute
    if constexpr (g_coreType == AIV) {
        UnpermuteBuffInit();
        CrossRankSyncInWorldSize(false); // Wave 已释放固定归属锁，同步核重新取得通道后确认 Combine 完成。
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

#endif

} // namespace MegaMoeImpl
#undef TemplateMegaMoeLayeredTypeClass
#undef TemplateMegaMoeLayeredTypeFunc
#endif
