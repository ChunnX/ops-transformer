/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_ep_combine.h
 * \brief MoE Expert-Parallel Combine kernel implementation
 */
#ifndef MOE_EP_COMBINE_H
#define MOE_EP_COMBINE_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_COMBINE_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/index/arithprogression.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif

#include "moe_ep_combine_tiling_key.h"
#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "moe_ep_combine_base.h"
#include "moe_ep_combine_tiling.h"

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#endif

namespace MoeEpCombineImpl {

#if defined(ENABLE_MOE_EP_COMBINE_KERNEL)

using namespace AscendC;

#define TemplateMoeEpCombineTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineTypeFunc XType, HasTopkWeight
#define HCOMM_INIT_SIZE 512UL

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t RECV_META_FIELDS = 4;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U; // 本卡状态空间偏移地址，前面的地址给dispatch用
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t DCCI_OFFSET = 64U;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2U;
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr uint32_t PATTERN_EVERY_FOURTH = 0x11111111u; // bitmask: bit 0,4,8,... 每 4 个元素取第 1 个
constexpr uint32_t TASK_STAGE_CAP = 2048U;             // 任务暂存上限（条）
constexpr uint32_t TASK_ENTRY_FIELDS = 4U;             // 任务条目 int32 字段数
constexpr uint32_t TASK_ENTRY_BYTES = 16U;             // 任务条目字节数: [row, srcTokenIdx, srcTopKIdx, weight]
static constexpr struct UrmaWqeEntry DEFAULT_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};

template <TemplateMoeEpCombineTypeClass>
class MoeEpCombine {
public:
    __aicore__ inline MoeEpCombine(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR combinedX,
                                GM_ADDR combinedTopkWeights, GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpCombineInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &tokenPerAivNum);
    __aicore__ inline void SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr);
    __aicore__ inline void SendSlot(uint32_t tokenIndex, int32_t srcRank, int32_t srcTokenIdx, int32_t srcTopKIdx,
                                    uint32_t channelIndex, uint32_t weight);
    __aicore__ inline bool WaitDispatch(uint32_t tokenIndex, uint32_t copyCount);
    __aicore__ inline void ProcessTopKToken(uint32_t tokenIndex);
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void ExtractRankColumn(uint32_t chunkTokens, uint32_t compareCntMax);
    __aicore__ inline uint32_t GatherMatches(uint32_t ownerRank, uint32_t chunkTokens, uint32_t compareCntMax);
    __aicore__ inline void BuildTaskList();
    __aicore__ inline void GroupRemoteTasks();
    __aicore__ inline void RunLocalTasks();
    __aicore__ inline void RunRemoteTasks();
    __aicore__ inline void SendRemoteRange(uint32_t rank, uint64_t rangeStart, uint64_t rangeEnd,
                                           uint32_t channelIndex);
    __aicore__ inline void FlushTaskStage(LocalTensor<int32_t> stage, uint32_t &stageCnt, GM_ADDR dstBase,
                                          uint32_t &dstOff);
    __aicore__ inline void GetCoreAssignment(uint32_t totalBlocks, uint32_t &targetRank, uint32_t &coreIndexInGroup,
                                             uint32_t &groupSize);
    __aicore__ inline void BuffInit();
    __aicore__ inline void MaskAlign(LocalTensor<half> maskCalcSelectedTensor);
    __aicore__ inline void MaskCheck();
    __aicore__ inline void RecvPhaseReduce();

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId, uint32_t channelIndex)
    {
        return hcommHandle_[rankId * channelsPerRank_ + channelIndex];
    }
    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }
    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }
    __aicore__ inline uint32_t ReduceSumWorkNeedSize(int32_t count, int32_t typeSize)
    {
        int32_t elementsPerBlock = UB_ALIGN / typeSize;
        int32_t elementsPerRepeat = ALIGNED_LEN_256 / typeSize;
        int32_t iter1OutputCount = (count + elementsPerRepeat - 1) / elementsPerRepeat;
        uint32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
    }

    TPipe *tpipe_{nullptr};
    const MoeEpCombineInfo *tilingData_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t channelsPerRank_{1};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t numTokens_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0}; // UB对齐后的hidden size
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    uint32_t hWeightAlignSize_{0}; // token+weight对齐后的hidden size
    uint32_t XTypeAlign32Size_{0};
    uint32_t perSlotBytes_{0};
    uint64_t actualA_{0};
    uint32_t aivNum_{0};
    uint32_t localmoeNum_{0};
    uint64_t taskListOffset_{0};
    uint32_t taskCapacity_{0};
    uint32_t localSegSize_{0};
    uint32_t metaChunkTokens_{0};
    uint32_t compareCntMax_{0};

    uint32_t tStart_{0};
    uint32_t tEnd_{0};
    uint32_t tPerCore_{0};
    uint32_t ubXBytes_{0};
    uint32_t stateOffset_{0};

    uint32_t mask_tokenNum_{0};
    uint32_t bsKCastCnt_{0};
    uint32_t activeMaskAlignSize_{0};

    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> topkIdxGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;
    GlobalTensor<float> topkWeightsGm_;

    GlobalTensor<XType> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<uint32_t> statusTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint32_t> stateResetTensor_;

    LocalTensor<half> maskGenerateTensor_;
    LocalTensor<bool> maskStrideTensor_;
    LocalTensor<half> tokenTargetTensor_;

    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECOUT, 1> xOutQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> weightQue_;
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<> readStateBuf_;
    TBuf<> tokenStatusBuf_;
    TBuf<> stateBuf_;
    TBuf<> stateSumBuf_;
    TBuf<> stateResetBuf_;
    TBuf<> hcommBuf_;

    TBuf<> compareBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> tokenTargetTBuf_;
    TBuf<> metadataBuf_; // 发送阶段recvSrcMetadata的UB缓冲，批量搬运避免GetValue
    TBuf<> weightsBuf_;
    TBuf<> rankColBuf_;   // 发送阶段向量化过滤：srcRank 列
    TBuf<> positionsBuf_; // 发送阶段向量化过滤：位置序列
    TBuf<> tokenPosBuf_;  // 发送阶段向量化过滤：命中位置压缩结果
    TBuf<> patternBuf_;   // 发送阶段向量化过滤：列提取 bitmask pattern
    TBuf<> maskBuf_;      // 发送阶段向量化过滤：CompareScalar 输出 bitmask
    TBuf<> taskStageBuf_; // 任务暂存（TASK_STAGE_CAP 条）
    TBuf<> rankCntBuf_;   // Phase1A 每核按 rank 命中计数
    TBuf<> rankBaseBuf_;  // Phase1B 本核每 rank 起始偏移（displs + rankBase 行）
    TBuf<> runOffBuf_;    // Phase1B 本核每 rank 已写计数

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;                 // 通信上下文

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    GM_ADDR workspaceBase_{nullptr};
    GM_ADDR remoteListBase_{nullptr}; // 远程任务列表（按 rank 分组）
    GM_ADDR localListBase_{nullptr};  // 本地任务列表（每核固定分段）
    GM_ADDR countsBase_{nullptr};     // 计数区（rankCnt/rankBase/localCnt/displs）
    uint32_t aivId_{0};
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx,
                                                                        GM_ADDR recvSrcMetadata,
                                                                        GM_ADDR numRecvPerExpert, GM_ADDR topkWeights,
                                                                        GM_ADDR combinedX, GM_ADDR combinedTopkWeights,
                                                                        GM_ADDR workspace, GM_ADDR tilingGM,
                                                                        TPipe *pipe, const MoeEpCombineInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    workspaceBase_ = workspace;
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    numTokens_ = tilingData_->cfg.numTokens;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    aivNum_ = tilingData_->aivNum;
    localmoeNum_ = tilingData_->cfg.numLocalExperts;
    taskListOffset_ = tilingData->taskListOffset;
    taskCapacity_ = tilingData->taskCapacity;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; // UB 32字节对齐
    hWeightAlignSize_ = hAlignSize_ + UB_ALIGN;                      // UB 32字节对齐
    stateOffset_ = STATE_OFFSET;

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    rankId_ = mc2Context_->epRankId;
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (channelsPerRank_ == 0 || (epWorldSize_ > 0 && channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / epWorldSize_)) {
        channelsPerRank_ = 1;
    }
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    uint32_t handleCount = epWorldSize_ * channelsPerRank_;
    for (uint32_t i = 0; i < handleCount; ++i) {
        hcommHandle_[i] = mc2Context_->hcommHandle[i];
    }

    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    combineDataWinOffset_ = tilingData->combineDataWinOffset;

    xGm_.SetGlobalBuffer((__gm__ XType *)x);
    topkIdxGm_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    combinedXGm_.SetGlobalBuffer((__gm__ XType *)combinedX);

    // 计算actualA_的大小
    uint32_t numRecvBytes = Ceil(localmoeNum_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t reduceTmpBytes = ReduceSumWorkNeedSize(localmoeNum_, sizeof(int64_t)) * sizeof(int64_t);
    TBuf<TPosition::VECCALC> numRecvBuf;
    tpipe_->InitBuffer(numRecvBuf, numRecvBytes + reduceTmpBytes + UB_ALIGN); // 数据 + ReduceSum tmp + 输出
    LocalTensor<int64_t> numRecvLocal = numRecvBuf.Get<int64_t>();

    DataCopyExtParams numRecvCopyParams{1U, static_cast<uint32_t>(localmoeNum_ * sizeof(int64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int64_t> numRecvPadParams{false, 0U, 0U, 0U};
    DataCopyPad(numRecvLocal, numRecvPerExpertGm_, numRecvCopyParams, numRecvPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<int64_t> reduceTmp = numRecvLocal[numRecvBytes / sizeof(int64_t)];
    LocalTensor<int64_t> sumOut = reduceTmp[reduceTmpBytes / sizeof(int64_t)];
    ReduceSum<int64_t>(sumOut, numRecvLocal, reduceTmp, static_cast<int32_t>(localmoeNum_));
    SyncFunc<AscendC::HardEvent::V_S>();
    actualA_ = static_cast<uint64_t>(sumOut.GetValue(0));
    ubXBytes_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    XTypeAlign32Size_ = hAlignSize_;
    if constexpr (HasTopkWeight == 1) {
        XTypeAlign32Size_ = hWeightAlignSize_;
        topkWeightsGm_.SetGlobalBuffer((__gm__ float *)topkWeights);
        combinedTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)combinedTopkWeights);
    }
    tpipe_->InitBuffer(xQueue_, DOUBLE_BUFFER_NUM, XTypeAlign32Size_);
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN); // 32
    statusTensor_ = readStateBuf_.Get<uint32_t>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId = aivId_;

    startTokenId = sendTokenNum * newAivId; // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {     // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr)
{
    GlobalTensor<XType> outToken;
    outToken.SetGlobalBuffer((__gm__ XType *)dstAddr);

    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams copyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    LocalTensor<XType> tokenTensor = xQueue_.AllocTensor<XType>();
    DataCopyPad(tokenTensor, xGm_[tokenIndex * axisH_], copyParams, padParams);
    xQueue_.EnQue(tokenTensor);
    tokenTensor = xQueue_.DeQue<XType>();
    DataCopyPad(outToken, tokenTensor, copyParams);
    xQueue_.FreeTensor<XType>(tokenTensor);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendSlot(uint32_t tokenIndex, int32_t srcRank,
                                                                            int32_t srcTokenIdx, int32_t srcTopKIdx,
                                                                            uint32_t channelIndex, uint32_t weight)
{
    uint64_t sendTokenOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    uint64_t recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;
    GM_ADDR remoteRankWinAddr = GetUrmaWinAddrByRankId(srcRank, combineDataWinOffset_);
    GM_ADDR remoteRankStateAddr = GetUrmaStateAddrByRankId(srcRank, combineStateWinOffset_);

    if (srcRank != static_cast<int32_t>(rankId_)) {
        GM_ADDR tokenAddr = (GM_ADDR)xGm_.GetPhyAddr(tokenIndex * axisH_);
        uint64_t commHandle = GetCommHandle(srcRank, channelIndex);
        uint64_t notifyValue = 1ULL << 32;
        if constexpr (HasTopkWeight == 1) {
            notifyValue |= weight;
        }
        hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DEFAULT_WQE_CONFIG>(
            commHandle, remoteRankWinAddr + sendTokenOffset, tokenAddr, axisH_ * sizeof(XType),
            remoteRankStateAddr + recvStateOffset, notifyValue);
    } else {
        SendLocalToken(tokenIndex, remoteRankWinAddr + sendTokenOffset);
        if constexpr (HasTopkWeight == 1) {
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            statusTensor_(0) = weight;
            SyncFunc<AscendC::HardEvent::S_MTE3>();
        }
        GlobalTensor<uint32_t> state;
        state.SetGlobalBuffer((__gm__ uint32_t *)(remoteRankStateAddr + recvStateOffset));
        DataCopy(state, statusTensor_, FLOAT_PER_UB_ALIGN);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetCoreAssignment(uint32_t totalBlocks,
                                                                                     uint32_t &targetRank,
                                                                                     uint32_t &coreIndexInGroup,
                                                                                     uint32_t &groupSize)
{
    uint32_t baseGroupSize = totalBlocks / epWorldSize_;
    uint32_t remainder = totalBlocks % epWorldSize_;
    uint32_t accumulated = 0;
    for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
        uint32_t currentGroupSize = baseGroupSize + ((rank < remainder) ? 1U : 0U);
        if (aivId_ < accumulated + currentGroupSize) {
            targetRank = rank;
            groupSize = currentGroupSize;
            coreIndexInGroup = aivId_ - accumulated;
            return;
        }
        accumulated += currentGroupSize;
    }
    targetRank = epWorldSize_;
    groupSize = 0;
    coreIndexInGroup = 0;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseExpertToToken()
{
    if (actualA_ == 0 || epWorldSize_ == 0 || aivNum_ == 0) {
        return;
    }
    if (aivId_ >= aivNum_) {
        return;
    }

    // 任务列表区域（workspace: 16MB 系统区之后，冗余的每 rank 发送数据区已删除）
    remoteListBase_ = workspaceBase_ + taskListOffset_;
    localListBase_ = remoteListBase_ + static_cast<uint64_t>(taskCapacity_) * TASK_ENTRY_BYTES;
    countsBase_ = localListBase_ + (static_cast<uint64_t>(taskCapacity_) + aivNum_) * TASK_ENTRY_BYTES;
    localSegSize_ = static_cast<uint32_t>(actualA_ / aivNum_) + 1; // 每核本地分段容量（条）

    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    constexpr uint32_t metaChunkTokenMax = 4096U; // 分块上限，平衡UB占用与搬运次数
    metaChunkTokens_ = (actualA_ < metaChunkTokenMax) ? static_cast<uint32_t>(actualA_) : metaChunkTokenMax;

    uint32_t metaChunkBytes = Ceil(metaChunkTokens_ * metaBytesPerToken, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(metadataBuf_, metaChunkBytes);
    if constexpr (HasTopkWeight == 1) {
        uint32_t weightChunkBytes = Ceil(metaChunkTokens_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(weightsBuf_, weightChunkBytes);
    }
    // 向量化 rank 过滤 buffer（Engram ScatterByRank 思路）：CompareScalar 按 256B 对齐元素数执行
    compareCntMax_ = Ceil(metaChunkTokens_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    tpipe_->InitBuffer(rankColBuf_, compareCntMax_ * sizeof(int32_t));
    tpipe_->InitBuffer(positionsBuf_, compareCntMax_ * sizeof(int32_t));
    tpipe_->InitBuffer(tokenPosBuf_, compareCntMax_ * sizeof(int32_t));
    uint32_t maskBufSize = Ceil(Ceil(compareCntMax_, BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(maskBuf_, maskBufSize);
    uint32_t patternBufSize = Ceil(compareCntMax_ / BITS_PER_BYTE * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(patternBuf_, patternBufSize);
    // 任务列表相关 buffer（+1 槽容纳 displs[epWorldSize]）
    tpipe_->InitBuffer(taskStageBuf_, TASK_STAGE_CAP * TASK_ENTRY_BYTES);
    uint32_t rankBufBytes = Ceil((epWorldSize_ + 1U) * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankCntBuf_, rankBufBytes);
    tpipe_->InitBuffer(rankBaseBuf_, rankBufBytes);
    tpipe_->InitBuffer(runOffBuf_, rankBufBytes);

    BuildTaskList();     // Phase 1A: 全核扫描计数 + 本地任务落盘
    GroupRemoteTasks();  // Phase 1B: 远程任务按 rank 分组落盘（0 核前缀和）
    RunLocalTasks();     // Phase 2:  本地任务分全部核执行
    RunRemoteTasks();    // Phase 2:  远程任务只给通信核执行

    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ExtractRankColumn(uint32_t chunkTokens,
                                                                                     uint32_t compareCntMax)
{
    // metadata 为 [srcRank, srcTokenIdx, srcTopKIdx, pad] x4 int32 交错布局，
    // 用 bitmask pattern 每 4 个元素取第 1 个，将 srcRank 列压缩到 rankColBuf_（Engram GatherMask 同款语义）
    LocalTensor<uint32_t> pattern = patternBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(pattern, PATTERN_EVERY_FOURTH, compareCntMax / BITS_PER_BYTE);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    LocalTensor<int32_t> rankCol = rankColBuf_.Get<int32_t>();
    uint64_t rsvdCnt = 0;
    GatherMask(rankCol, metadataLocal, pattern, true, chunkTokens * RECV_META_FIELDS, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_S>();
    ascendc_assert(rsvdCnt == chunkTokens, "ExtractRankColumn count mismatch, expect=%u, got=%llu", chunkTokens,
                   rsvdCnt);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline uint32_t MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GatherMatches(uint32_t ownerRank,
                                                                                     uint32_t chunkTokens,
                                                                                     uint32_t compareCntMax)
{
    // CompareScalar 输出 bitmask（每元素 1 bit），GatherMask 按 mask 压缩 positions，
    // 得到 ownerRank 在本 chunk 内的命中位置列表（Engram GatherRankTokens 思路）。
    // 尾块时 compareCntMax 超出 chunkTokens 的部分为无效 bit，GatherMask 以 chunkTokens 截断
    LocalTensor<int32_t> rankCol = rankColBuf_.Get<int32_t>();
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
    CompareScalar(mask, rankCol, static_cast<int32_t>(ownerRank), CMPMODE::EQ, compareCntMax);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
    LocalTensor<int32_t> tokenPos = tokenPosBuf_.Get<int32_t>();
    uint64_t rsvdCnt = 0;
    GatherMask(tokenPos, positions, mask.ReinterpretCast<uint32_t>(), true, chunkTokens, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_S>();
    return static_cast<uint32_t>(rsvdCnt);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::FlushTaskStage(LocalTensor<int32_t> stage,
                                                                                  uint32_t &stageCnt,
                                                                                  GM_ADDR dstBase, uint32_t &dstOff)
{
    // 标量 SetValue 填写的 stage 经 MTE3 批量落盘，dstOff 以条为单位推进
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    GlobalTensor<int32_t> dstGm;
    dstGm.SetGlobalBuffer((__gm__ int32_t *)(dstBase + static_cast<uint64_t>(dstOff) * TASK_ENTRY_BYTES));
    DataCopyParams stageParams{1U, static_cast<uint16_t>(stageCnt * TASK_ENTRY_BYTES), 0U, 0U};
    DataCopyPad(dstGm, stage, stageParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(dstGm);
    dstOff += stageCnt;
    stageCnt = 0;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuildTaskList()
{
    // Phase 1A: 每核扫 1/aivNum_ 的 metadata 区间，向量化过滤后：
    // 本地命中压成任务落盘到本核本地分段（供 Phase 2 全核执行），远程命中只按 rank 计数
    uint64_t tokensPerCore = actualA_ / aivNum_;
    uint64_t remainder = actualA_ % aivNum_;
    uint64_t scanStart = 0;
    uint64_t scanEnd = 0;
    if (aivId_ < remainder) {
        scanStart = aivId_ * (tokensPerCore + 1);
        scanEnd = scanStart + tokensPerCore + 1;
    } else {
        scanStart = aivId_ * tokensPerCore + remainder;
        scanEnd = scanStart + tokensPerCore;
    }

    LocalTensor<uint32_t> rankCnt = rankCntBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(rankCnt, (uint32_t)0, epWorldSize_);
    SyncFunc<AscendC::HardEvent::V_S>();

    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    LocalTensor<int32_t> tokenPos = tokenPosBuf_.Get<int32_t>();
    LocalTensor<int32_t> stage = taskStageBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};
    uint32_t localCnt = 0;
    uint32_t stageCnt = 0;
    uint32_t localOff = 0;
    GM_ADDR localSegBase = localListBase_ + static_cast<uint64_t>(aivId_) * localSegSize_ * TASK_ENTRY_BYTES;

    for (uint64_t chunkStart = scanStart; chunkStart < scanEnd; chunkStart += metaChunkTokens_) {
        uint64_t chunkEnd = (chunkStart + metaChunkTokens_ > scanEnd) ? scanEnd : (chunkStart + metaChunkTokens_);
        uint32_t curChunkTokens = static_cast<uint32_t>(chunkEnd - chunkStart);

        LocalTensor<uint32_t> weightsForNotify;
        if constexpr (HasTopkWeight == 1) {
            weightsForNotify = weightsBuf_.Get<uint32_t>();
            DataCopyExtParams weightCopyParams{1U, static_cast<uint32_t>(curChunkTokens * sizeof(uint32_t)), 0U, 0U,
                                               0U};
            const DataCopyPadExtParams<uint32_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(weightsForNotify, topkWeightsGm_[chunkStart].template ReinterpretCast<uint32_t>(),
                        weightCopyParams, padParams);
        }

        DataCopyExtParams metaCopyParams{1U, curChunkTokens * (RECV_META_FIELDS * static_cast<uint32_t>(sizeof(int32_t))), 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // 向量流水构建位置序列并提取 srcRank 列，与标量流水上的任务压缩并行
        ArithProgression<int32_t>(positionsBuf_.Get<int32_t>(), 0, 1, curChunkTokens);
        ExtractRankColumn(curChunkTokens, compareCntMax_);

        for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
            uint32_t matchCnt = GatherMatches(rank, curChunkTokens, compareCntMax_);
            if (rank == rankId_) {
                // 本地任务: 逐条压缩到 stage，满批落盘本核本地分段
                for (uint32_t j = 0; j < matchCnt; ++j) {
                    uint32_t pos = static_cast<uint32_t>(tokenPos.GetValue(j));
                    uint32_t row = static_cast<uint32_t>(chunkStart) + pos;
                    uint32_t weight = 0;
                    if constexpr (HasTopkWeight == 1) {
                        weight = weightsForNotify(pos);
                    }
                    stage.SetValue(stageCnt * TASK_ENTRY_FIELDS + 0, static_cast<int32_t>(row));
                    stage.SetValue(stageCnt * TASK_ENTRY_FIELDS + 1,
                                   metadataLocal.GetValue(pos * RECV_META_FIELDS + 1));
                    stage.SetValue(stageCnt * TASK_ENTRY_FIELDS + 2,
                                   metadataLocal.GetValue(pos * RECV_META_FIELDS + 2));
                    stage.SetValue(stageCnt * TASK_ENTRY_FIELDS + 3, static_cast<int32_t>(weight));
                    stageCnt++;
                    if (stageCnt >= TASK_STAGE_CAP) {
                        FlushTaskStage(stage, stageCnt, localSegBase, localOff);
                    }
                }
                localCnt += matchCnt;
            } else if (matchCnt > 0) {
                rankCnt.SetValue(rank, rankCnt.GetValue(rank) + matchCnt);
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>(); // 下一块 DataCopyPad 覆盖 metadataLocal 前，标量读必须完成
    }
    if (stageCnt > 0) {
        FlushTaskStage(stage, stageCnt, localSegBase, localOff);
    }

    // 计数落 GM 计数区: rankCnt[aivId_][]（紧凑）+ localCnt[aivId_]（32B 槽，槽号 = 核号）
    uint64_t rankCntBytes = Ceil(static_cast<uint64_t>(aivNum_) * epWorldSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    SyncFunc<AscendC::HardEvent::S_MTE3>(); // rankCnt 由标量 SetValue 写入，MTE3 读前同步
    GlobalTensor<uint32_t> rankCntGm;
    rankCntGm.SetGlobalBuffer(
        (__gm__ uint32_t *)(countsBase_ + static_cast<uint64_t>(aivId_) * epWorldSize_ * sizeof(uint32_t)));
    DataCopyParams rankCntParams{1U, static_cast<uint16_t>(epWorldSize_ * sizeof(uint32_t)), 0U, 0U};
    DataCopyPad(rankCntGm, rankCnt, rankCntParams);

    stage.SetValue(0, static_cast<int32_t>(localCnt));
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    GlobalTensor<int32_t> localCntGm;
    localCntGm.SetGlobalBuffer(
        (__gm__ int32_t *)(countsBase_ + 2 * rankCntBytes + static_cast<uint64_t>(aivId_) * UB_ALIGN));
    DataCopy(localCntGm, stage, FLOAT_PER_UB_ALIGN);
    SyncFunc<AscendC::HardEvent::MTE3_S>(); // 计数写完成后再 clean，避免失效未落地的缓存行

    GlobalTensor<int32_t> countsGm;
    countsGm.SetGlobalBuffer((__gm__ int32_t *)countsBase_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(countsGm);
    GlobalTensor<int32_t> localSegGm;
    localSegGm.SetGlobalBuffer((__gm__ int32_t *)localSegBase);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(localSegGm);
    SyncAll<true>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GroupRemoteTasks()
{
    uint64_t rankCntBytes = Ceil(static_cast<uint64_t>(aivNum_) * epWorldSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    uint64_t localCntBytes = static_cast<uint64_t>(aivNum_) * UB_ALIGN; // localCnt 32B 槽

    // 0 核: 读 rankCnt 矩阵 → 每核每 rank 前缀偏移 rankBase + 全局 displs（共 epWorldSize_+1 项，末项=远程总数）
    if (aivId_ == 0) {
        GlobalTensor<uint32_t> countsGm;
        countsGm.SetGlobalBuffer((__gm__ uint32_t *)countsBase_);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(countsGm);

        LocalTensor<uint32_t> rowRead = metadataBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> cum = rankBaseBuf_.Get<uint32_t>();
        Duplicate<uint32_t>(cum, (uint32_t)0, epWorldSize_);
        SyncFunc<AscendC::HardEvent::V_S>();
        DataCopyExtParams rowParams{1U, static_cast<uint32_t>(epWorldSize_ * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<uint32_t> rowPad{false, 0U, 0U, 0U};
        DataCopyParams rowWriteParams{1U, static_cast<uint16_t>(epWorldSize_ * sizeof(uint32_t)), 0U, 0U};
        for (uint32_t c = 0; c < aivNum_; ++c) {
            DataCopyPad(rowRead, countsGm[c * epWorldSize_], rowParams, rowPad);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            GlobalTensor<uint32_t> baseRowGm;
            baseRowGm.SetGlobalBuffer((__gm__ uint32_t *)(countsBase_ + rankCntBytes +
                                                         static_cast<uint64_t>(c) * epWorldSize_ * sizeof(uint32_t)));
            DataCopyPad(baseRowGm, cum, rowWriteParams);
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                cum.SetValue(r, cum.GetValue(r) + rowRead.GetValue(r));
            }
            SyncFunc<AscendC::HardEvent::S_MTE3>();
        }
        // 写全局 displs: displs[r+1] = 前 r+1 个 rank 的远程任务总数
        LocalTensor<uint32_t> displsUb = rankCntBuf_.Get<uint32_t>();
        displsUb.SetValue(0, 0);
        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            displsUb.SetValue(r + 1, cum.GetValue(r));
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<uint32_t> displsGm;
        displsGm.SetGlobalBuffer((__gm__ uint32_t *)(countsBase_ + 2 * rankCntBytes + localCntBytes));
        DataCopyParams displsWriteParams{1U, static_cast<uint16_t>((epWorldSize_ + 1U) * sizeof(uint32_t)), 0U, 0U};
        DataCopyPad(displsGm, displsUb, displsWriteParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>(); // displs 写完成后再 clean，避免失效未落地的缓存行
        DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(countsGm);
    }
    SyncAll<true>();

    // 每核: 读 displs 与本核 rankBase 行，合成每 rank 的写入基址 base[r]
    GlobalTensor<uint32_t> countsGmRead;
    countsGmRead.SetGlobalBuffer((__gm__ uint32_t *)countsBase_);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(countsGmRead);
    GlobalTensor<uint32_t> displsGm;
    displsGm.SetGlobalBuffer((__gm__ uint32_t *)(countsBase_ + 2 * rankCntBytes + localCntBytes));
    LocalTensor<uint32_t> displsUb = rankCntBuf_.Get<uint32_t>();
    DataCopyExtParams displsParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint32_t> rowPad{false, 0U, 0U, 0U};
    DataCopyPad(displsUb, displsGm, displsParams, rowPad);
    GlobalTensor<uint32_t> baseRowGm;
    baseRowGm.SetGlobalBuffer((__gm__ uint32_t *)(countsBase_ + rankCntBytes +
                                                 static_cast<uint64_t>(aivId_) * epWorldSize_ * sizeof(uint32_t)));
    LocalTensor<uint32_t> baseUb = rankBaseBuf_.Get<uint32_t>();
    DataCopyExtParams rowParams{1U, static_cast<uint32_t>(epWorldSize_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPad(baseUb, baseRowGm, rowParams, rowPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t r = 0; r < epWorldSize_; ++r) {
        baseUb.SetValue(r, baseUb.GetValue(r) + displsUb.GetValue(r));
    }
    LocalTensor<uint32_t> runOff = runOffBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(runOff, (uint32_t)0, epWorldSize_);
    SyncFunc<AscendC::HardEvent::V_S>();

    // 重扫自己区间，远程任务按 rank 连续段落盘（每核写自己的槽位，与 Phase 1A 同区间分片）
    uint64_t tokensPerCore = actualA_ / aivNum_;
    uint64_t remainder = actualA_ % aivNum_;
    uint64_t scanStart = 0;
    uint64_t scanEnd = 0;
    if (aivId_ < remainder) {
        scanStart = aivId_ * (tokensPerCore + 1);
        scanEnd = scanStart + tokensPerCore + 1;
    } else {
        scanStart = aivId_ * tokensPerCore + remainder;
        scanEnd = scanStart + tokensPerCore;
    }

    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    LocalTensor<int32_t> tokenPos = tokenPosBuf_.Get<int32_t>();
    LocalTensor<int32_t> stage = taskStageBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};

    for (uint64_t chunkStart = scanStart; chunkStart < scanEnd; chunkStart += metaChunkTokens_) {
        uint64_t chunkEnd = (chunkStart + metaChunkTokens_ > scanEnd) ? scanEnd : (chunkStart + metaChunkTokens_);
        uint32_t curChunkTokens = static_cast<uint32_t>(chunkEnd - chunkStart);

        LocalTensor<uint32_t> weightsForNotify;
        if constexpr (HasTopkWeight == 1) {
            weightsForNotify = weightsBuf_.Get<uint32_t>();
            DataCopyExtParams weightCopyParams{1U, static_cast<uint32_t>(curChunkTokens * sizeof(uint32_t)), 0U, 0U,
                                               0U};
            const DataCopyPadExtParams<uint32_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(weightsForNotify, topkWeightsGm_[chunkStart].template ReinterpretCast<uint32_t>(),
                        weightCopyParams, padParams);
        }

        DataCopyExtParams metaCopyParams{1U, curChunkTokens * (RECV_META_FIELDS * static_cast<uint32_t>(sizeof(int32_t))), 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        ArithProgression<int32_t>(positionsBuf_.Get<int32_t>(), 0, 1, curChunkTokens);
        ExtractRankColumn(curChunkTokens, compareCntMax_);

        for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
            if (rank == rankId_) {
                continue; // 本地任务已在 Phase 1A 落盘
            }
            uint32_t matchCnt = GatherMatches(rank, curChunkTokens, compareCntMax_);
            if (matchCnt == 0) {
                continue;
            }
            // 本 rank 的槽位区间: [base[r] + runOff[r], base[r] + runOff[r] + matchCnt)
            uint32_t slotCursor = baseUb.GetValue(rank) + runOff.GetValue(rank);
            uint32_t stageCnt = 0;
            for (uint32_t j = 0; j < matchCnt; ++j) {
                uint32_t pos = static_cast<uint32_t>(tokenPos.GetValue(j));
                uint32_t row = static_cast<uint32_t>(chunkStart) + pos;
                uint32_t weight = 0;
                if constexpr (HasTopkWeight == 1) {
                    weight = weightsForNotify(pos);
                }
                stage.SetValue(stageCnt * TASK_ENTRY_FIELDS + 0, static_cast<int32_t>(row));
                stage.SetValue(stageCnt * TASK_ENTRY_FIELDS + 1,
                               metadataLocal.GetValue(pos * RECV_META_FIELDS + 1));
                stage.SetValue(stageCnt * TASK_ENTRY_FIELDS + 2,
                               metadataLocal.GetValue(pos * RECV_META_FIELDS + 2));
                stage.SetValue(stageCnt * TASK_ENTRY_FIELDS + 3, static_cast<int32_t>(weight));
                stageCnt++;
                if (stageCnt >= TASK_STAGE_CAP) {
                    FlushTaskStage(stage, stageCnt, remoteListBase_, slotCursor);
                }
            }
            if (stageCnt > 0) {
                FlushTaskStage(stage, stageCnt, remoteListBase_, slotCursor);
            }
            runOff.SetValue(rank, slotCursor - baseUb.GetValue(rank));
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>(); // 下一块 DataCopyPad 覆盖 metadataLocal 前，标量读必须完成
    }

    GlobalTensor<int32_t> remoteGm;
    remoteGm.SetGlobalBuffer((__gm__ int32_t *)remoteListBase_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(remoteGm);
    SyncAll<true>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::RunLocalTasks()
{
    uint64_t rankCntBytes = Ceil(static_cast<uint64_t>(aivNum_) * epWorldSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    uint64_t localCntBytes = static_cast<uint64_t>(aivNum_) * UB_ALIGN; // localCnt 32B 槽

    // 读每核本地任务数（32B 槽，取每槽第 0 元素）并压缩到 localCntUb[0..aivNum_)
    GlobalTensor<int32_t> localListGm;
    localListGm.SetGlobalBuffer((__gm__ int32_t *)localListBase_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(localListGm);
    GlobalTensor<uint32_t> localCntGm;
    localCntGm.SetGlobalBuffer((__gm__ uint32_t *)(countsBase_ + 2 * rankCntBytes));
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(localCntGm);
    LocalTensor<uint32_t> localCntUb = taskStageBuf_.Get<uint32_t>();
    DataCopyExtParams localCntParams{1U, static_cast<uint32_t>(localCntBytes), 0U, 0U, 0U};
    DataCopyPadExtParams<uint32_t> cntPad{false, 0U, 0U, 0U};
    DataCopyPad(localCntUb, localCntGm, localCntParams, cntPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t localTotal = 0;
    for (uint32_t c = 0; c < aivNum_; ++c) {
        uint32_t v = localCntUb.GetValue(c * FLOAT_PER_UB_ALIGN);
        localCntUb.SetValue(c, v);
        localTotal += v;
    }
    if (localTotal == 0) {
        return;
    }
    // 本地任务逻辑列表等量切分给全部核
    uint64_t perCore = localTotal / aivNum_;
    uint64_t rem = localTotal % aivNum_;
    uint64_t start = 0;
    uint64_t end = 0;
    if (aivId_ < rem) {
        start = aivId_ * (perCore + 1);
        end = start + perCore + 1;
    } else {
        start = aivId_ * perCore + rem;
        end = start + perCore;
    }
    if (start >= end) {
        return;
    }

    // statusTensor_ 懒初始化：仅本核有本地任务时触发
    Duplicate<uint32_t>(statusTensor_, (uint32_t)1, FLOAT_PER_UB_ALIGN);
    SyncFunc<AscendC::HardEvent::V_MTE3>();

    // 定位 start 所在段（每核分段固定 localSegSize_，有效条数为 localCntUb[coreIdx]）
    uint32_t coreIdx = 0;
    uint64_t segStart = 0;
    while (coreIdx < aivNum_ && start >= segStart + localCntUb.GetValue(coreIdx)) {
        segStart += localCntUb.GetValue(coreIdx);
        coreIdx++;
    }
    uint32_t offInSeg = static_cast<uint32_t>(start - segStart);

    LocalTensor<int32_t> taskUb = metadataBuf_.Get<int32_t>();
    uint64_t logical = start;
    while (logical < end && coreIdx < aivNum_) {
        // 跳过空分段，避免 batch=0 导致死循环
        while (coreIdx < aivNum_ && localCntUb.GetValue(coreIdx) == 0) {
            coreIdx++;
        }
        if (coreIdx >= aivNum_) {
            break;
        }
        uint32_t segLeft = localCntUb.GetValue(coreIdx) - offInSeg;
        uint64_t left = end - logical;
        uint32_t batch = (left < segLeft) ? static_cast<uint32_t>(left) : segLeft;
        if (batch > metaChunkTokens_) {
            batch = metaChunkTokens_;
        }
        GlobalTensor<int32_t> taskGm;
        taskGm.SetGlobalBuffer((__gm__ int32_t *)(localListBase_ +
                                                 (static_cast<uint64_t>(coreIdx) * localSegSize_ + offInSeg) *
                                                     TASK_ENTRY_BYTES));
        DataCopyExtParams taskParams{1U, batch * TASK_ENTRY_BYTES, 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> taskPad{false, 0U, 0U, 0U};
        DataCopyPad(taskUb, taskGm, taskParams, taskPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t j = 0; j < batch; ++j) {
            uint32_t row = static_cast<uint32_t>(taskUb.GetValue(j * TASK_ENTRY_FIELDS + 0));
            int32_t srcTokenIdx = taskUb.GetValue(j * TASK_ENTRY_FIELDS + 1);
            int32_t srcTopKIdx = taskUb.GetValue(j * TASK_ENTRY_FIELDS + 2);
            uint32_t weight = static_cast<uint32_t>(taskUb.GetValue(j * TASK_ENTRY_FIELDS + 3));
            // 与 SendSlot 本地分支逐字一致的发送 + 状态槽写入语义
            uint64_t slotOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
            GM_ADDR dst = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_) + slotOffset;
            SendLocalToken(row, dst);
            if constexpr (HasTopkWeight == 1) {
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                statusTensor_(0) = weight;
                SyncFunc<AscendC::HardEvent::S_MTE3>();
            }
            GM_ADDR stateAddr = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) +
                                (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;
            GlobalTensor<uint32_t> stateGm;
            stateGm.SetGlobalBuffer((__gm__ uint32_t *)stateAddr);
            DataCopy(stateGm, statusTensor_, FLOAT_PER_UB_ALIGN);
        }
        logical += batch;
        offInSeg += batch;
        if (offInSeg >= localCntUb.GetValue(coreIdx)) {
            segStart += localCntUb.GetValue(coreIdx);
            coreIdx++;
            offInSeg = 0;
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendRemoteRange(uint32_t rank,
                                                                                   uint64_t rangeStart,
                                                                                   uint64_t rangeEnd,
                                                                                   uint32_t channelIndex)
{
    if (rangeStart >= rangeEnd) {
        return;
    }
    LocalTensor<int32_t> taskUb = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> taskPad{false, 0U, 0U, 0U};
    for (uint64_t off = rangeStart; off < rangeEnd;) {
        uint64_t left = rangeEnd - off;
        uint32_t batch = (left < metaChunkTokens_) ? static_cast<uint32_t>(left) : metaChunkTokens_;
        GlobalTensor<int32_t> taskGm;
        taskGm.SetGlobalBuffer((__gm__ int32_t *)(remoteListBase_ + off * TASK_ENTRY_BYTES));
        DataCopyExtParams taskParams{1U, batch * TASK_ENTRY_BYTES, 0U, 0U, 0U};
        DataCopyPad(taskUb, taskGm, taskParams, taskPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t j = 0; j < batch; ++j) {
            uint32_t row = static_cast<uint32_t>(taskUb.GetValue(j * TASK_ENTRY_FIELDS + 0));
            int32_t srcTokenIdx = taskUb.GetValue(j * TASK_ENTRY_FIELDS + 1);
            int32_t srcTopKIdx = taskUb.GetValue(j * TASK_ENTRY_FIELDS + 2);
            uint32_t weight = static_cast<uint32_t>(taskUb.GetValue(j * TASK_ENTRY_FIELDS + 3));
            SendSlot(row, static_cast<int32_t>(rank), srcTokenIdx, srcTopKIdx, channelIndex, weight);
        }
        off += batch;
    }
    SyncFunc<AscendC::HardEvent::S_MTE2>(); // 保护 taskUb 复用
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::RunRemoteTasks()
{
    uint32_t remoteExecNum = aivNum_;
    uint32_t maxChannelAivNum = epWorldSize_ * channelsPerRank_;
    if (remoteExecNum > maxChannelAivNum) {
        remoteExecNum = maxChannelAivNum;
    }
    if (remoteExecNum == 0 || aivId_ >= remoteExecNum) {
        return;
    }

    uint64_t rankCntBytes = Ceil(static_cast<uint64_t>(aivNum_) * epWorldSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    uint64_t localCntBytes = static_cast<uint64_t>(aivNum_) * UB_ALIGN; // localCnt 32B 槽

    // 读全局 displs（末项为远程任务总数），并 invalidate 远程列表与计数区
    GlobalTensor<int32_t> remoteListGm;
    remoteListGm.SetGlobalBuffer((__gm__ int32_t *)remoteListBase_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(remoteListGm);
    GlobalTensor<uint32_t> displsGm;
    displsGm.SetGlobalBuffer((__gm__ uint32_t *)(countsBase_ + 2 * rankCntBytes + localCntBytes));
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(displsGm);
    LocalTensor<uint32_t> displsUb = rankCntBuf_.Get<uint32_t>();
    DataCopyExtParams displsParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint32_t> pad{false, 0U, 0U, 0U};
    DataCopyPad(displsUb, displsGm, displsParams, pad);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 每 (rank, channel) handle 单写者：remoteExecNum >= epWorldSize 时按 rank 分组，
    // 组内核 k 用 channel k；否则每核跨步独占整 rank（channel 0）
    if (remoteExecNum >= epWorldSize_) {
        uint32_t targetRank = epWorldSize_;
        uint32_t coreIdxInGroup = 0;
        uint32_t groupSize = 1;
        GetCoreAssignment(remoteExecNum, targetRank, coreIdxInGroup, groupSize);
        if (targetRank >= epWorldSize_ || groupSize == 0) {
            return;
        }
        uint64_t rankStart = displsUb.GetValue(targetRank);
        uint64_t rankEnd = displsUb.GetValue(targetRank + 1);
        uint64_t cnt = rankEnd - rankStart;
        uint64_t perCore = cnt / groupSize;
        uint64_t rem = cnt % groupSize;
        uint64_t s = rankStart + coreIdxInGroup * perCore + ((coreIdxInGroup < rem) ? coreIdxInGroup : rem);
        uint64_t e = s + perCore + ((coreIdxInGroup < rem) ? 1U : 0U);
        SendRemoteRange(targetRank, s, e, coreIdxInGroup);
    } else {
        for (uint32_t rank = aivId_; rank < epWorldSize_; rank += remoteExecNum) {
            SendRemoteRange(rank, displsUb.GetValue(rank), displsUb.GetValue(rank + 1), 0);
        }
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuffInit()
{
    tpipe_->Reset();
    SplitToCore(numTokens_, aivNum_, tStart_, tEnd_, tPerCore_);
    if (tStart_ >= numTokens_) {
        return;
    }

    mask_tokenNum_ = tPerCore_ * topK_;
    uint32_t bsKInt32Align = Ceil(mask_tokenNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t bsKFloatAlign = Ceil(mask_tokenNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t bsKHalfAlign = Ceil(mask_tokenNum_ * sizeof(half), UB_ALIGN) * UB_ALIGN;
    uint32_t bsHalfAlign = Ceil(tPerCore_ * sizeof(half), UB_ALIGN) * UB_ALIGN;

    bsKCastCnt_ = Ceil(mask_tokenNum_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    activeMaskAlignSize_ = tPerCore_ * Ceil(topK_ * sizeof(half), UB_ALIGN) * UB_ALIGN;
    bsKInt32Align = (bsKInt32Align > bsKCastCnt_ ? bsKInt32Align : bsKCastCnt_);
    bsKHalfAlign = (bsKHalfAlign > activeMaskAlignSize_ ? bsKHalfAlign : activeMaskAlignSize_);
    bsKFloatAlign = (bsKFloatAlign > bsKCastCnt_ ? bsKFloatAlign : bsKCastCnt_);
    bsKFloatAlign = (bsKFloatAlign > activeMaskAlignSize_ ? bsKFloatAlign : activeMaskAlignSize_);

    tpipe_->InitBuffer(tokenTargetTBuf_, bsHalfAlign);
    tpipe_->InitBuffer(compareBuf_, bsKInt32Align);
    tpipe_->InitBuffer(rowTmpFloatBuf_, bsKFloatAlign);
    tpipe_->InitBuffer(tokenBuf_, bsKHalfAlign);

    if constexpr (HasTopkWeight == 1) {
        tpipe_->InitBuffer(weightQue_, DOUBLE_BUFFER_NUM, UB_ALIGN);
    }
    tpipe_->InitBuffer(xInQue_, DOUBLE_BUFFER_NUM, ubXBytes_);
    tpipe_->InitBuffer(xOutQue_, DOUBLE_BUFFER_NUM, ubXBytes_);
    tpipe_->InitBuffer(ubAccFp32Buf_, axisH_ * sizeof(float));
    tpipe_->InitBuffer(ubTmpFp32Buf_, axisH_ * sizeof(float));

    ubAccFp32_ = ubAccFp32Buf_.Get<float>();
    ubTmpFp32_ = ubTmpFp32Buf_.Get<float>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline bool MoeEpCombine<TemplateMoeEpCombineTypeFunc>::WaitDispatch(uint32_t tokenIndex, uint32_t copyCount)
{
    // 计算地址偏移
    GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + tokenIndex * topK_ * WIN_ADDR_ALIGN;
    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);

    LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopyExtParams params = {static_cast<uint16_t>(topK_), UB_ALIGN, WIN_ADDR_ALIGN - UB_ALIGN, 0, 0};
    DataCopyPadExtParams<uint32_t> padParams = {false, 0, 0, 0};

    DataCopyPad<uint32_t>(stateTensor, stateGMTensor, params, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<uint32_t> stateSumTensor = stateSumBuf_.Get<uint32_t>();
    uint32_t shape[] = {topK_, UB_ALIGN / sizeof(uint32_t)};
    ReduceSum<uint32_t, AscendC::Pattern::Reduce::RA, false>(stateSumTensor, stateTensor, shape, true);
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t localState = stateSumTensor(1);
    if (localState == copyCount) {
        return true;
    }
    return false;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ProcessTopKToken(uint32_t tokenIndex)
{
    Duplicate<float>(ubAccFp32_, (float)0, axisH_);
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};
    DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
    for (uint32_t topkId = 0U; topkId < topK_; topkId++) {
        // 读取expert_id
        uint64_t slotOffset = (static_cast<uint64_t>(tokenIndex) * topK_ + topkId) * perSlotBytes_;
        GM_ADDR wAddr = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_) + slotOffset;
        GlobalTensor<XType> srcTokenTensor;
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(wAddr));
        LocalTensor<XType> xLocal = xInQue_.AllocTensor<XType>();
        DataCopyPad(xLocal, srcTokenTensor, xCopyParams, padParams);
        xInQue_.EnQue(xLocal);
        LocalTensor<XType> xIn = xInQue_.DeQue<XType>();
        Cast(ubTmpFp32_, xIn, AscendC::RoundMode::CAST_NONE, axisH_);
        Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_, axisH_);
        xInQue_.FreeTensor(xIn);

        if constexpr (HasTopkWeight == 1) {
            GM_ADDR weightAddr = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) +
                                 (tokenIndex * topK_ + topkId) * WIN_ADDR_ALIGN;
            GlobalTensor<float> srcWeightTensor;
            srcWeightTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightAddr));
            LocalTensor<float> weightLocal = weightQue_.AllocTensor<float>();
            DataCopyPad(weightLocal, srcWeightTensor, weightCopyParams, padParams);
            weightQue_.EnQue(weightLocal);
            LocalTensor<float> weightOut = weightQue_.DeQue<float>();
            DataCopyPad(combinedTopkWeightsGm_[tokenIndex * topK_ + topkId], weightOut, weightCopyParams);
            weightQue_.FreeTensor(weightOut);
        }
    }
    // 累加结果双缓冲入队
    LocalTensor<XType> ubResultBf16 = xOutQue_.AllocTensor<XType>();
    Cast(ubResultBf16, ubAccFp32_, RoundMode::CAST_RINT, axisH_);
    xOutQue_.EnQue(ubResultBf16);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::RecvPhaseReduce()
{
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    tpipe_->InitBuffer(tokenStatusBuf_, Ceil(tPerCore_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(stateBuf_, topK_ * STATE_OFFSET);
    tpipe_->InitBuffer(stateSumBuf_, UB_ALIGN);
    tpipe_->InitBuffer(stateResetBuf_, topK_ * STATE_OFFSET); // 清理状态区
    stateResetTensor_ = stateResetBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(stateResetTensor_, (uint32_t)0.0, static_cast<uint32_t>(topK_ * FLOAT_PER_UB_ALIGN));
    LocalTensor<int32_t> tokenStatusTensor = tokenStatusBuf_.Get<int32_t>();
    Duplicate<int32_t>(tokenStatusTensor, static_cast<int32_t>(0), tPerCore_);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t CompletedtokenNum = static_cast<uint32_t>(0);
    uint32_t copyCount = topK_;
    while (CompletedtokenNum != tPerCore_) {
        for (uint32_t tokenIdx = tStart_; tokenIdx < tEnd_; ++tokenIdx) {
            if (tokenStatusTensor(tokenIdx - tStart_) == 1) {
                continue;
            }
            if (!WaitDispatch(tokenIdx, copyCount)) {
                continue;
            }
            CompletedtokenNum++;
            tokenStatusTensor.SetValue(tokenIdx - tStart_, 1);
            ProcessTopKToken(tokenIdx);
            LocalTensor<XType> ubResultBf16 = xOutQue_.DeQue<XType>();
            DataCopyPad(combinedXGm_[tokenIdx * axisH_], ubResultBf16, xCopyParams);
            xOutQue_.FreeTensor(ubResultBf16);

            GM_ADDR stateGM =
                GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + tokenIdx * topK_ * WIN_ADDR_ALIGN;
            GlobalTensor<uint32_t> stateGMTensor;
            stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);
            DataCopyExtParams resetParams = {static_cast<uint16_t>(topK_), UB_ALIGN, 0, WIN_ADDR_ALIGN - UB_ALIGN, 0};
            DataCopyPad<uint32_t>(stateGMTensor, stateResetTensor_, resetParams);
        }
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Process()
{
    SendPhaseExpertToToken();
    PipeBarrier<PIPE_ALL>(); // MaskCheck中包含reset操作，需确保前面操作完成
    BuffInit();
    RecvPhaseReduce();
}

#endif

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H
