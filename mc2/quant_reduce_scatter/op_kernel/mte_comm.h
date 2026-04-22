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
 * \file mte_comm.h
 * \brief quant_all_reduce 与quant_reduce_scatter MTE通信相关 kernel代码公共部分
 */

#ifndef MTE_COMMON_H
#define MTE_COMMON_H

#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "../../common/op_kernel/moe_distribute_base.h"

namespace QuantMTECommImpl {

using namespace AscendC;
// 后缀_BYTES 表示单位为字节大小B, _NUM 表示单位为个
constexpr static uint32_t UB_ALIGN_BYTES = 32U;     // UB按32B对齐
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;         // float格式下32B对齐需要 32/4 =8个
constexpr uint32_t BUFFER_NUM = 2U;                 // 用于double buffer
constexpr static uint32_t SCALE_BLCOK_BYTES = 32U;  // 一个scale数据块固定32B，搬运要求32B对齐
constexpr static uint32_t X_BLOCK_BYTES = 1024U;    // 当前一个x数据块固定1024B = 128 * 8(PT) = 32 * 32(MX)，为穿刺取值
constexpr static uint32_t MAX_PER_BLOCK_NUM = 1024U * 15U; // 单块最大数据量限制（参考MTE-skill）
constexpr static uint64_t STATE_WIN_SIZE = 1024UL * 1024UL; // Win区前1MB作为状态区相关，状态0区(462KB) + 状态1区(462KB) + 标志位区(100KB)

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
constexpr static uint64_t SINGLE_STATE_REGION_SIZE = 462UL * 1024UL; // 0/1分区单个状态区大小
constexpr static uint32_t ZERONE_STATE_POS = 0U; // 0/1分区标志位的index
constexpr static uint64_t WIN_ADDR_ALIGN = 512UL; // 每个核的标志位间512B对齐

#define TemplateTypeClass typename XType, typename ScalesType, typename OutputType
#define TemplateType XType, ScalesType, OutputType

template<TemplateTypeClass>
class MTECommunication {
public:
    __aicore__ inline MTECommunication() {};
    __aicore__ inline void InitHcclContext();
    __aicore__ inline void ComputeBlockParams();
    __aicore__ inline void InitParams();
    __aicore__ inline void InitParamsForAIVGroup(uint64_t xNums);
    __aicore__ inline void InitGMTensor(GM_ADDR x, GM_ADDR scales, GM_ADDR output, uint64_t alignedXSize, uint64_t dataSpaceGmSize);
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void SetBlockSize(uint32_t elementsPerBlock, uint64_t aivNum, uint64_t lastBlockNum);
    __aicore__ inline void SetSendCoreNumPerRank(uint32_t sendCoreNumPerRank);
    template <bool isReduceScatter = false>
    __aicore__ inline void CopyDataToWin(uint64_t xSliceSizeNums = 0, uint64_t scaleSliceNums = 0);
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void CopyResultToOutput(uint64_t outOffsetGM, LocalTensor<float>& localResultTensor, uint32_t count);
    __aicore__ inline void ComputeTailAivId(uint64_t totalAivCount);
    __aicore__ inline GM_ADDR GetWinAddrGm(uint32_t rankId, uint64_t offset = 0);
    __aicore__ inline GM_ADDR GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag);
    __aicore__ inline GM_ADDR GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag);
    
    // ========== PUT 模式新增函数 ==========
    __aicore__ inline void WriteBlockToTargetWin(uint32_t targetRank, uint64_t winXOffset, uint64_t winScaleOffset, 
                                                  uint32_t count, uint64_t xSize, LocalTensor<XType>& xInTensor, LocalTensor<ScalesType>& scaleInTensor);
    template <bool isReduceScatter = false>
    __aicore__ inline void PutDataToAllWin(uint64_t xSize, uint64_t xSliceSizeNums = 0, uint64_t scaleSliceNums = 0);
    __aicore__ inline void GatherFromSelfWin(uint64_t xSize, uint64_t perRankOutSize, uint64_t scalePerRankSize);

    __gm__ Mc2Kernel::HcclOpParam *hcclContext_;
    uint32_t aivId_{0};
    uint64_t aivNum_{0};
    uint32_t round_{0};
    uint32_t tailBlockNums_{0};
    uint32_t assignedBlockNums_{0};
    uint64_t scaleNumsPerBlcok_{0};
    uint64_t xOffset_{0};  
    uint64_t scaleOffset_{0};
    uint64_t lastAivId_{0};
    uint32_t winBufferFlags_{0};
    uint32_t xNumPerBlock_{0};  // 动态计算的数据块大小（元素个数）
    
    // AIV 分组相关参数（CV 1:1）
    uint32_t remoteRankId_{0};        // 当前核负责的 rank ID
    uint32_t coreInnerIndex_{0};      // 当前核在组内的序号
    uint32_t sendCoreNumPerRank_{1};  // 每个 rank 分配的核数
    uint64_t partitionOffset_{0};     // 分区起始偏移（AIV 分组时使用）
private:
    uint64_t tailXNums_{0};

    __aicore__ inline void CopyDataBlock(uint64_t curXOffset, uint64_t curScaleOffset, uint32_t count);

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;
    GlobalTensor<XType> localWinXGMTensor_;
    GlobalTensor<ScalesType> localWinScaleGMTensor_;
    GlobalTensor<OutputType> outputTensor_;
    GlobalTensor<uint32_t> selfWinFlagGMTensor_;

    LocalTensor<XType> xTmpTensor_;
    LocalTensor<ScalesType> scaleTmpTensor_;
    LocalTensor<float> stateResetTensor_;
    LocalTensor<OutputType> xOutTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_, scaleQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> winFlagsBuf_;
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> stateResetBuf_;
};

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitHcclContext()
{
    hcclContext_ = (__gm__ Mc2Kernel::HcclOpParam*)GetHcclContext<HCCL_GROUP_ID_0>();
}

/**
 * @brief 动态计算数据块大小（元素个数）
 * 
 * 参考MTE-skill的ComputeBlockParams，根据UB可用空间动态计算每块最大数据量。
 * 考虑状态区预留空间后，计算可用于数据搬运的UB空间，并设置上限保护。
 * 当前使用保守策略：保持原来的固定块大小作为上限。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::ComputeBlockParams()
{
    // 状态区预留：writeStateBuf + readStateBuf + stateResetBuf + winFlagsBuf
    // readStateBuf和stateResetBuf各需要 rankDim * UB_ALIGN_BYTES
    uint64_t stateBufSize = UB_ALIGN_BYTES * 3 + hcclContext_->rankDim * UB_ALIGN_BYTES * 2;
    // scale缓冲区预留：SCALE_BLCOK_BYTES (固定32B)
    uint64_t scaleBufSize = SCALE_BLCOK_BYTES;
    // 计算缓冲区预留（vec_comp需要）
    // brcbBuf + xCastBuf + xCastBf16Buf + xCastfp16Buf ≈ xNumPerBlock * 10B
    // 使用保守估计：每元素约10B预留空间（float + bf16 + fp16）
    uint64_t calcBufPerElem = sizeof(float) + sizeof(bfloat16_t) + sizeof(float16_t);
    // sumBuf预留：每元素4B
    uint64_t sumBufPerElem = sizeof(float);
    
    // 可用于x数据搬运的UB空间（考虑double buffer）
    // 需要预留：xQueue + xOutQueue + vec_comp buffers + sumBuf
    // 保守估计：每元素需要 sizeof(XType) * 2 (double buffer) + sizeof(OutputType) * 2 + calcBufPerElem + sumBufPerElem
    uint64_t elemCostBytes = sizeof(XType) * BUFFER_NUM + sizeof(OutputType) * BUFFER_NUM + calcBufPerElem + sumBufPerElem + scaleBufSize;
    uint64_t availableUbForData = TOTAL_UB_SIZE - stateBufSize;
    
    // 向下对齐到32B
    uint64_t maxPerBlockBytes = availableUbForData / elemCostBytes;
    maxPerBlockBytes = maxPerBlockBytes / UB_ALIGN_BYTES * UB_ALIGN_BYTES;  // FloorAlign
    uint64_t maxPerBlockNum = maxPerBlockBytes / sizeof(XType);
    
    // 设置上限保护（保持原来的固定值作为上限）
    uint64_t originalBlockNum = X_BLOCK_BYTES / sizeof(XType);  // 原来每块的元素个数
    xNumPerBlock_ = static_cast<uint32_t>(MIN(originalBlockNum, MIN(MAX_PER_BLOCK_NUM, maxPerBlockNum)));
    scaleNumsPerBlcok_ = SCALE_BLCOK_BYTES / sizeof(ScalesType);
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitParams()
{
    aivId_ = GetBlockIdx(); // 获取当前核Id
    assignedBlockNums_ = aivId_ < tailBlockNums_ ? round_ + 1 : round_; // 当前核分配到的数据块数量，顺序分核，序号小的核多搬一轮
    uint64_t blockIdx = aivId_ * round_ + (aivId_ < tailBlockNums_ ? aivId_ : tailBlockNums_); // 计算当前核分派到的首个数据块序列号
    xOffset_ = blockIdx * xNumPerBlock_; // 块数 * 一块有多少数据，得到要搬第几个x数据
    scaleOffset_ = blockIdx * scaleNumsPerBlcok_; // 计算要搬第几个 scale
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::SetSendCoreNumPerRank(uint32_t sendCoreNumPerRank)
{
    sendCoreNumPerRank_ = sendCoreNumPerRank;
    if (sendCoreNumPerRank_ == 0) {
        sendCoreNumPerRank_ = 1;
    }
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitParamsForAIVGroup(uint64_t xNums)
{
    aivId_ = GetBlockIdx();
    
    // AIV 分组：每核只负责一个 rank 的数据（CV 1:1）
    remoteRankId_ = aivId_ / sendCoreNumPerRank_;
    if (remoteRankId_ >= hcclContext_->rankDim) {
        remoteRankId_ = hcclContext_->rankDim - 1;
    }
    coreInnerIndex_ = aivId_ % sendCoreNumPerRank_;
    
    // 分区起始偏移：remoteRankId_ * 每分区大小
    partitionOffset_ = remoteRankId_ * xNums;
    
    // 按 sendCoreNumPerRank 切分数据块
    assignedBlockNums_ = coreInnerIndex_ < tailBlockNums_ ? round_ + 1 : round_;
    if (assignedBlockNums_ == 0) {
        assignedBlockNums_ = 1;
    }
    
    uint64_t coreBlockIdx = coreInnerIndex_ * round_ + (coreInnerIndex_ < tailBlockNums_ ? coreInnerIndex_ : tailBlockNums_);
    xOffset_ = MIN(coreBlockIdx * xNumPerBlock_, xNums);
    scaleOffset_ = coreBlockIdx * scaleNumsPerBlcok_;
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitGMTensor(GM_ADDR x, GM_ADDR scales, GM_ADDR output, uint64_t xSize, uint64_t winSpaceGmSize)
{
    // 入参相关数据的GMTensor
    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)scales);
    outputTensor_.SetGlobalBuffer((__gm__ OutputType*)output);

    // =========== Win区相关 =========== 
    // Win区OOM检测适配，告知OOM框架Win区地址和大小
    #if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        for(uint64_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
            OOMCheckAddrRange(GetWinAddrGm(curRank), winSpaceGmSize);
        }
    #endif

    //  处理 0/1 分区 标志位
    uint64_t currCoreFlagOffset = 2UL * SINGLE_STATE_REGION_SIZE + aivId_ * WIN_ADDR_ALIGN; // 计算当前核的标志位在Win区的偏移
    selfWinFlagGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)GetWinAddrGm(hcclContext_->rankId, currCoreFlagOffset));
    LocalTensor<uint32_t> winFlagLocalTensor = winFlagsBuf_.Get<uint32_t>();
    DataCopyExtParams flagCopyInParams{1, static_cast<uint32_t>(UB_ALIGN_BYTES), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> flagPadParams{false, 0, 0, 0};
    DataCopyPad(winFlagLocalTensor, selfWinFlagGMTensor_, flagCopyInParams, flagPadParams); // GM -> UB
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    winBufferFlags_ = winFlagLocalTensor.GetValue(ZERONE_STATE_POS); // 获取标志位
    winFlagLocalTensor.SetValue(ZERONE_STATE_POS, 1 - winBufferFlags_); // 翻转标志位, 0→1, 1→0
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams flagCopyOutParams{1, static_cast<uint32_t>(UB_ALIGN_BYTES), 0, 0, 0};
    DataCopyPad(selfWinFlagGMTensor_, winFlagLocalTensor, flagCopyOutParams); // 覆写标志位回GM

    // 获取本卡地址写数据
    // 通过rankId和0/1分区标志位获取本地winIn区地址对应卡的数据区域
    GM_ADDR localDataSpaceGm = GetWinDataAddrGm(hcclContext_->rankId, winBufferFlags_);
    localWinXGMTensor_.SetGlobalBuffer((__gm__ XType*)localDataSpaceGm);
    localWinScaleGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)(localDataSpaceGm + xSize)); // sclae数据跟在x后
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitBuffer(TPipe *tPipe)
{
    tPipe->InitBuffer(xQueue_, BUFFER_NUM, xNumPerBlock_ * sizeof(XType));    // 使用动态计算的x块大小
    tPipe->InitBuffer(scaleQueue_, BUFFER_NUM, UB_ALIGN_BYTES);     
    tPipe->InitBuffer(xOutQueue_, BUFFER_NUM, xNumPerBlock_ * sizeof(OutputType)); // 用于输出的OutPutTensor
    tPipe->InitBuffer(winFlagsBuf_, UB_ALIGN_BYTES); // 用于读取0/1分区的标志位
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN_BYTES); // 状态位每一个按32B对齐
    tPipe->InitBuffer(readStateBuf_, hcclContext_->rankDim * UB_ALIGN_BYTES); // 每次读 rankDim 个状态位
    tPipe->InitBuffer(stateResetBuf_, hcclContext_->rankDim * UB_ALIGN_BYTES); // 用于清理状态区

    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM)); // 用于状态区清零
}

/**
 * @brief 配置数据块划分参数，用于多核并行计算
 * 
 * @param elementsPerBlock 每个标准数据块包含的X元素数量
 * @param aivNum AIV核的总数，用于数据分发和负载均衡
 * @param lastBlockNum 最后一个核处理的尾部数据块元素数量
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::SetBlockSize(uint32_t elementsPerBlock, uint64_t aivNum, uint64_t lastBlockNum)
{
    xNumPerBlock_ = elementsPerBlock;
    aivNum_ = aivNum;
    tailXNums_ = lastBlockNum;
}

/**
 * @brief 计算负责处理尾部数据的AI核ID
 * 
 * @param totalAivCount AIV核的总数量
 * 
 * @note 当数据量较小时，此时计算仅仅只有一轮（round_ == 0），此时处理尾块的aiv并非最后一个，
 *       需根据当前数据块数量计算。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::ComputeTailAivId(uint64_t totalAivCount)
{
    if (round_ == 0) {
        // 小数据量时，如果只有一轮搬运，负责尾块的aiv由此时计算的总块数决定
        lastAivId_ = tailBlockNums_ - 1;
    } else {
        // 轮次大于一轮时，负责尾块的aiv必定是最后一个
        lastAivId_ = totalAivCount - 1;
    }
}

/**
 * @brief 复制数据块从GM到Win区
 * 
 * 复制流程：
 * 1. 量化数据（x）复制：GM → UB（队列分配缓冲区） → Win区
 * 2. 缩放因子（scale）复制：GM → UB（队列分配缓冲区） → Win区
 * 
 * @param curXOffset 量化数据x在全局内存(GM)中的偏移量
 * @param curScaleOffset 缩放因子scale在全局内存(GM)中的偏移量
 * @param count 当前x数据块dataCopy的元素数量
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::CopyDataBlock(uint64_t curXOffset, uint64_t curScaleOffset, uint32_t count)
{
    // 先拷贝data数据， 再拷贝scales
    /* x 从 GM -> UB -> Win */
    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    if constexpr (AscendC::IsSameType<XType, int8_t>::value) {
        DataCopyExtParams xCopyInParams{1, static_cast<uint32_t>(count * sizeof(XType)), 0, 0, 0};
        DataCopyPadExtParams<XType> xPadParams{false, 0, 0, 0};
        DataCopyPad(xTmpTensor_, xGMTensor_[curXOffset], xCopyInParams, xPadParams);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XType>();
        DataCopyExtParams xCopyOutParams{1, static_cast<uint32_t>(count * sizeof(XType)), 0, 0, 0};
        DataCopyPad(localWinXGMTensor_[curXOffset], xTmpTensor_, xCopyOutParams);
    } else {
        DataCopy(xTmpTensor_, xGMTensor_[curXOffset], count);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XType>();
        DataCopy(localWinXGMTensor_[curXOffset], xTmpTensor_, count);
    }
    xQueue_.FreeTensor<XType>(xTmpTensor_);

    /* scale 从 GM -> UB -> Win */
    scaleTmpTensor_ = scaleQueue_.AllocTensor<ScalesType>();
    if constexpr (AscendC::IsSameType<ScalesType, float>::value) {
        DataCopyExtParams scaleCopyInParams{1, static_cast<uint32_t>(scaleNumsPerBlcok_ * sizeof(ScalesType)), 0, 0, 0};
        DataCopyPadExtParams<ScalesType> scalePadParams{false, 0, 0, 0};
        DataCopyPad(scaleTmpTensor_, scalesGMTensor_[curScaleOffset], scaleCopyInParams, scalePadParams);
        scaleQueue_.EnQue(scaleTmpTensor_);
        scaleTmpTensor_ = scaleQueue_.DeQue<ScalesType>();
        DataCopyExtParams scaleCopyOutParams{1, static_cast<uint32_t>(scaleNumsPerBlcok_ * sizeof(ScalesType)), 0, 0, 0};
        DataCopyPad(localWinScaleGMTensor_[curScaleOffset], scaleTmpTensor_, scaleCopyOutParams);
    } else {
        DataCopy(scaleTmpTensor_, scalesGMTensor_[curScaleOffset], scaleNumsPerBlcok_);
        scaleQueue_.EnQue(scaleTmpTensor_);
        scaleTmpTensor_ = scaleQueue_.DeQue<ScalesType>();
        DataCopy(localWinScaleGMTensor_[curScaleOffset], scaleTmpTensor_, scaleNumsPerBlcok_);
    }
    scaleQueue_.FreeTensor<ScalesType>(scaleTmpTensor_);
}

/**
 * @brief 将数据复制到Win区（支持ReduceScatter和AllReduce两种通信模式）
 * 
 * 该函数根据通信模式将数据从全局内存GM全部复制到本卡Win区，支持两种分布式通信模式：
 * 1. ReduceScatter模式：由于数据按Rank（设备）分片，需按rankDim遍历以搬运所有数据
 * 2. AllReduce模式：数据全局复制，每个核直接将数据复制到Win区
 * 
 * @tparam isReduceScatter 通信模式标识，true表示ReduceScatter模式，false表示AllReduce模式
 * 
 * @param[in] xSliceSizeNums 每个Rank的数据分片大小（元素数量），ReduceScatter模式使用
 * @param[in] scaleSliceNums 每个Rank的缩放因子分片大小（元素数量），ReduceScatter模式使用
 */
template <TemplateTypeClass>
template <bool isReduceScatter>
__aicore__ inline void MTECommunication<TemplateType>::CopyDataToWin(uint64_t xSliceSizeNums, uint64_t scaleSliceNums)
{
    // 遍历每个核需要搬运的数据块
    for (uint64_t curBlock = 0; curBlock < assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset_ + curBlock * xNumPerBlock_; // 分区内偏移
        uint64_t curScaleOffset = scaleOffset_ + curBlock * scaleNumsPerBlcok_;
        uint32_t copyBlockNum = xNumPerBlock_;
        
        if constexpr (isReduceScatter) {
            // ReduceScatter: 使用原始的尾块判断（lastAivId_）
            if ((aivId_ == lastAivId_) && (curBlock == assignedBlockNums_ - 1)) {
                copyBlockNum = tailXNums_;
            }
            // ReduceScatter过程，数据按卡均分，需要对卡进行遍历
            for(uint64_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
                // all2all过程，加上卡偏移
                uint64_t curRankXOffset = curXOffset + curRank * xSliceSizeNums;
                uint64_t curRankScaleOffset = curScaleOffset + curRank * scaleSliceNums;

                // 搬运当前数据块到Win区
                CopyDataBlock(curRankXOffset, curRankScaleOffset, copyBlockNum);
            }
        } else {
            // AllReduce过程，AIV分组：加上分区偏移
            uint32_t lastCoreIndexForRank = sendCoreNumPerRank_ - 1;
            if ((coreInnerIndex_ == lastCoreIndexForRank) && (curBlock == assignedBlockNums_ - 1)) {
                copyBlockNum = tailXNums_;
            }
            // GM 偏移 = 分区偏移 + 分区内偏移
            uint64_t gmXOffset = partitionOffset_ + curXOffset;
            uint64_t gmScaleOffset = partitionOffset_ / 128 + curScaleOffset;  // scale 偏移按 128 元素一组
            CopyDataBlock(gmXOffset, gmScaleOffset, copyBlockNum);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

/**
 * @brief 向Win区状态区写入本核完成数据搬运状态
 * 
 * 该函数负责将当前AI Core（核）的数据搬运完成状态写入到所有Rank的状态Win区中，
 * 通知其他设备当前核的数据准备已完成。这是集群通信软同步机制的一部分，
 * 每个核需要向所有Rank（包括本机和其他设备）的状态窗口写入标识，
 * 确保所有设备都能感知到当前核的完成状态。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::WriteStatusToWin()
{
    uint32_t coreOffset = aivId_ * hcclContext_->rankDim; // Win区大小为 aivNum * rankDim, 此处计算核偏移
    // 遍历每一张卡，给每一张卡都要写入状态
    for (uint32_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
        // 写入状态到对端，每个核写一个状态，表示自己的数据块已经写完
        LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
        DataCopy<float>(statusTensor, stateResetTensor_, FLOAT_UB_ALIGN_NUM); // 先重置statusTensor数据，后面累加需要Tensor内全部数据，防止脏数据
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        statusTensor(0) = (float)1.0;  // 用1标识
        GM_ADDR remoteWinStateGM = GetWinStatusAddrGm(curRank, winBufferFlags_); // 获取当前要写对端卡的状态区地址
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
        // 不同卡上的核的状态写到相邻位置，读时可以一次读rankDim个状态, 状态区大小设计为 aivNum * ranDim
        uint64_t curOffset = (coreOffset + hcclContext_->rankId) * FLOAT_UB_ALIGN_NUM; // 当前核偏移 + 卡偏移， 按32B对齐
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams statusCopyOutParams{1, static_cast<uint32_t>(FLOAT_UB_ALIGN_NUM * sizeof(float)), 0, 0, 0};
        DataCopyPad(stateGMTensor[curOffset], statusTensor, statusCopyOutParams); // 按32B对齐拷贝
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

/**
 * @brief 读取Win区状态区并等待同步状态
 * 
 * 该函数从本Rank的状态Win区读取同步标识，并在本地进行忙等待，直到所有参与通信的Rank
 * 都已完成状态设置。这种软同步机制确保所有Rank上当前核（AI Core）所需的数据都已
 * 准备就绪，从而避免跨设备数据不一致性问题。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::ReadStatus()
{
    GM_ADDR stateGM = GetWinStatusAddrGm(hcclContext_->rankId, winBufferFlags_); // 获取本卡的状态区用于读取
    GlobalTensor<float> selfStatusWinTensor;
    uint32_t offset = aivId_ * hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM; // 获取当前核所需读取状态位的头地址，状态按32B对齐
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0; // 用于计算状态和
    uint32_t statusCnt = hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM; // 一次读rankDim个，按32B对齐
    SumParams sumParams{1, statusCnt, statusCnt};
    float minTarget = hcclContext_->rankDim - (float)0.5;
    float maxTarget = hcclContext_->rankDim + (float)0.5;
    // 读取statusCnt个数据求和
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopyExtParams statusCopyInParams{1, static_cast<uint32_t>(statusCnt * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> statusPadParams{false, 0, 0, 0};
        DataCopyPad(statusTensor, selfStatusWinTensor[offset], statusCopyInParams, statusPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusTensor, statusTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        flag = statusTensor(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams statusCopyOutParams{1, static_cast<uint32_t>(statusCnt * sizeof(float)), 0, 0, 0};
    DataCopyPad(selfStatusWinTensor[offset], stateResetTensor_, statusCopyOutParams); // 相关状态区重新置零
}

/**
 * @brief 将计算得到的UB上的结果Tensor的数据复制到GM上的OutPutTensor，并进行类型转换（如果需要）。
 * 
 * @param outputOffset 输出OutputTensor的GM偏移量，用于指定目标位置。
 * @param sourceTensor 本地UB上计算结果Tensor，包含计算完成的数据。
 * @param count 当前每次处理数据块的元素数量
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::CopyResultToOutput(uint64_t outOffsetGM, LocalTensor<float>& localResultTensor, uint32_t count)
{
    // 将计算好的数据拷贝到输出tensor，如果是非float数据类型需要先转换成目标数据类型
    xOutTensor_ = xOutQueue_.AllocTensor<OutputType>();
    if constexpr (AscendC::IsSameType<OutputType, float>::value) {
        DataCopy(xOutTensor_, localResultTensor, count);
    } else {
        Cast(xOutTensor_, localResultTensor, RoundMode::CAST_RINT, count);
    }
    xOutQueue_.EnQue(xOutTensor_);
    xOutTensor_ = xOutQueue_.DeQue<OutputType>();
    DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(count * sizeof(OutputType)), 0, 0, 0};
    DataCopyPad(outputTensor_[outOffsetGM], xOutTensor_, outCopyParams);
    xOutQueue_.FreeTensor(xOutTensor_);
}

// 获取指定rank, 指定偏移下Win区的GM地址
template <TemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<TemplateType>::GetWinAddrGm(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(hcclContext_->windowsIn[rankId] + offset);
}

// 根据0/1分区标志位，获取对应rank的Win区数据区的地址
template <TemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<TemplateType>::GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag)
{   
    if (winFlag == 0U) {
        // 若使用 0 分区, 加上状态区的偏移
        return GetWinAddrGm(rankId, STATE_WIN_SIZE);
    }
    else {
        // 若使用 1 分区，即WinOut
        return (GM_ADDR)(hcclContext_->windowsOut[rankId]);
    }
}

// 根据0/1分区标志位，获取对应rank的Win区状态区的地址
template <TemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<TemplateType>::GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag)
{   
    if (winFlag == 0U) {
        // 若使用 0 分区, 即无偏移
        return GetWinAddrGm(rankId);
    }
    else {
        // 若使用 1 分区，则加上状态0区的大小偏移
        return GetWinAddrGm(rankId, SINGLE_STATE_REGION_SIZE);
    }
}

/**
 * @brief [PUT 模式] 将数据块写入指定 rank 的 Win Area
 * 
 * PUT 模式核心差异：写入其他 rank 的 Win Area（而非只写本地）
 * 
 * @param targetRank 目标 rank ID
 * @param winXOffset Win Area 中 x 数据区的偏移（元素数）
 * @param winScaleOffset Win Area 中 scale 数据区的偏移（元素数）
 * @param count 写入元素数
 * @param xSize x 数据的总大小（字节）
 * @param xInTensor 输入 x UB tensor
 * @param scaleInTensor 输入 scale UB tensor
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::WriteBlockToTargetWin(
    uint32_t targetRank, uint64_t winXOffset, uint64_t winScaleOffset,
    uint32_t count, uint64_t xSize, LocalTensor<XType>& xInTensor, LocalTensor<ScalesType>& scaleInTensor)
{
    // 获取目标 rank 的 Win Area 数据区地址
    GM_ADDR targetWinDataGm = GetWinDataAddrGm(targetRank, winBufferFlags_);
    
    // 写入 x 数据到目标 Win Area
    GlobalTensor<XType> targetWinXTensor;
    targetWinXTensor.SetGlobalBuffer((__gm__ XType*)targetWinDataGm);
    DataCopyExtParams xCopyOutParams{1, static_cast<uint32_t>(count * sizeof(XType)), 0, 0, 0};
    DataCopyPad(targetWinXTensor[winXOffset], xInTensor, xCopyOutParams);
    
    // 写入 scale 数据到目标 Win Area（scale 数据紧跟在 x 数据之后）
    GlobalTensor<ScalesType> targetWinScaleTensor;
    // scale 数据区在 Win Area 中紧随 x 数据区
    targetWinScaleTensor.SetGlobalBuffer((__gm__ ScalesType*)(targetWinDataGm + xSize));
    DataCopyExtParams scaleCopyOutParams{1, static_cast<uint32_t>(scaleNumsPerBlcok_ * sizeof(ScalesType)), 0, 0, 0};
    DataCopyPad(targetWinScaleTensor[winScaleOffset], scaleInTensor, scaleCopyOutParams);
}

/**
 * @brief [PUT 模式] 本卡数据写入所有 rank 的 Win Area（AIV 分组）
 * 
 * PUT 模式核心流程：
 * - 每组核负责一个 rank 分区的数据写入
 * - 写入位置：winDataOffset = rankId * perRankDataSize + partitionOffset_ + dataCoreOffset
 * 
 * AllReduce: 每核写入所有 rank 的 Win Area 中本卡对应分区
 * ReduceScatter: 每核只写入目标 rank 分区（AllToAll 语义）
 * 
 * @tparam isReduceScatter true 表示 ReduceScatter 模式
 * @param xSize x 数据的总大小（字节）
 * @param xSliceSizeNums ReduceScatter 时每 rank 分区大小
 * @param scaleSliceNums ReduceScatter 时每 rank 分区 scale 大小
 */
template <TemplateTypeClass>
template <bool isReduceScatter>
__aicore__ inline void MTECommunication<TemplateType>::PutDataToAllWin(uint64_t xSize, uint64_t xSliceSizeNums, uint64_t scaleSliceNums)
{
    // ReduceScatter: Win Area[targetRank] 存储所有rank对该分区的数据，大小 = xSliceSizeNums * rankDim = xSize
    // AllReduce: Win Area[targetRank] 存储所有rank的全量数据，大小 = perRankXNums * rankDim = xSize * rankDim
    uint64_t scaleRegionOffset;
    if constexpr (isReduceScatter) {
        scaleRegionOffset = xSize;  // ReduceScatter: Win Area大小 = xSize
    } else {
        scaleRegionOffset = xSize * hcclContext_->rankDim;  // AllReduce: Win Area大小 = xSize * rankDim
    }
    
    uint64_t perRankXNums = xSize / sizeof(XType);
    uint64_t perRankScaleNums = perRankXNums / 128;  // hif8量化
    
    for (uint64_t curBlock = 0; curBlock < assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset_ + curBlock * xNumPerBlock_;
        uint64_t curScaleOffset = scaleOffset_ + curBlock * scaleNumsPerBlcok_;
        uint32_t copyBlockNum = xNumPerBlock_;
        
        if ((aivId_ == lastAivId_) && (curBlock == assignedBlockNums_ - 1)) {
            copyBlockNum = tailXNums_;
        }
        
        xTmpTensor_ = xQueue_.AllocTensor<XType>();
        if constexpr (AscendC::IsSameType<XType, int8_t>::value) {
            DataCopyExtParams xCopyInParams{1, static_cast<uint32_t>(copyBlockNum * sizeof(XType)), 0, 0, 0};
            DataCopyPadExtParams<XType> xPadParams{false, 0, 0, 0};
            DataCopyPad(xTmpTensor_, xGMTensor_[curXOffset], xCopyInParams, xPadParams);
        } else {
            DataCopy(xTmpTensor_, xGMTensor_[curXOffset], copyBlockNum);
        }
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XType>();
        
        scaleTmpTensor_ = scaleQueue_.AllocTensor<ScalesType>();
        if constexpr (AscendC::IsSameType<ScalesType, float>::value) {
            DataCopyExtParams scaleCopyInParams{1, static_cast<uint32_t>(scaleNumsPerBlcok_ * sizeof(ScalesType)), 0, 0, 0};
            DataCopyPadExtParams<ScalesType> scalePadParams{false, 0, 0, 0};
            DataCopyPad(scaleTmpTensor_, scalesGMTensor_[curScaleOffset], scaleCopyInParams, scalePadParams);
        } else {
            DataCopy(scaleTmpTensor_, scalesGMTensor_[curScaleOffset], scaleNumsPerBlcok_);
        }
        scaleQueue_.EnQue(scaleTmpTensor_);
        scaleTmpTensor_ = scaleQueue_.DeQue<ScalesType>();
        
        if constexpr (isReduceScatter) {
            // ReduceScatter (AllToAll): 写入目标 rank Win Area 中本卡分区
            // Win Area[remoteRankId_] 布局：rank j 分区起始 = j * xSliceSizeNums
            uint64_t winXOffset = hcclContext_->rankId * xSliceSizeNums + curXOffset;
            uint64_t winScaleOffset = hcclContext_->rankId * scaleSliceNums + curScaleOffset;
            WriteBlockToTargetWin(remoteRankId_, winXOffset, winScaleOffset, copyBlockNum, scaleRegionOffset, xTmpTensor_, scaleTmpTensor_);
        } else {
            for (uint32_t i = 0; i < hcclContext_->rankDim; ++i) {
                uint32_t targetRank = (hcclContext_->rankId + i) % hcclContext_->rankDim;
                uint64_t winXOffset = hcclContext_->rankId * perRankXNums + curXOffset;
                uint64_t winScaleOffset = hcclContext_->rankId * perRankScaleNums + curScaleOffset;
                WriteBlockToTargetWin(targetRank, winXOffset, winScaleOffset, copyBlockNum, scaleRegionOffset, xTmpTensor_, scaleTmpTensor_);
            }
        }
        
        xQueue_.FreeTensor(xTmpTensor_);
        scaleQueue_.FreeTensor(scaleTmpTensor_);
    }
    PipeBarrier<PIPE_ALL>();
}

/**
 * @brief [PUT 模式] 从本卡 Win Area 读取所有 rank 数据并输出
 * 
 * PUT 模式：数据已由所有 rank 写入本卡 Win Area，直接从本卡读取
 * AIV 分组：每核负责一个 rank 分区的数据读取
 * 
 * @param xSize x 数据的总大小（字节）
 * @param perRankOutSize 每 rank 输出分区大小（元素数）
 * @param scalePerRankSize 每 rank scale 分区大小
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::GatherFromSelfWin(uint64_t xSize, uint64_t perRankOutSize, uint64_t scalePerRankSize)
{
    // 获取本卡 Win Area 数据区地址
    GM_ADDR selfWinDataGm = GetWinDataAddrGm(hcclContext_->rankId, winBufferFlags_);
    GlobalTensor<XType> selfWinXTensor;
    selfWinXTensor.SetGlobalBuffer((__gm__ XType*)selfWinDataGm);
    GlobalTensor<ScalesType> selfWinScaleTensor;
    selfWinScaleTensor.SetGlobalBuffer((__gm__ ScalesType*)(selfWinDataGm + xSize));
    
    // AIV 分组：本核负责 remoteRankId_ 分区的读取
    uint64_t winBaseXOffset = remoteRankId_ * perRankOutSize;
    uint64_t winBaseScaleOffset = remoteRankId_ * scalePerRankSize;
    
    // 遍历本核负责的数据块
    for (uint64_t curBlock = 0; curBlock < assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset_ + curBlock * xNumPerBlock_;
        uint64_t curScaleOffset = scaleOffset_ + curBlock * scaleNumsPerBlcok_;
        uint32_t copyBlockNum = xNumPerBlock_;
        
        uint32_t lastCoreIndex = sendCoreNumPerRank_ - 1;
        if ((coreInnerIndex_ == lastCoreIndex) && (curBlock == assignedBlockNums_ - 1)) {
            copyBlockNum = tailXNums_;
        }
        
        // 从本卡 Win Area 读取数据（直接读取，无需 DataCopyPad 处理脏数据问题）
        xTmpTensor_ = xQueue_.AllocTensor<XType>();
        if constexpr (AscendC::IsSameType<XType, int8_t>::value) {
            DataCopyExtParams xCopyInParams{1, static_cast<uint32_t>(copyBlockNum * sizeof(XType)), 0, 0, 0};
            DataCopyPadExtParams<XType> xPadParams{false, 0, 0, 0};
            DataCopyPad(xTmpTensor_, selfWinXTensor[winBaseXOffset + curXOffset], xCopyInParams, xPadParams);
        } else {
            DataCopy(xTmpTensor_, selfWinXTensor[winBaseXOffset + curXOffset], copyBlockNum);
        }
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<XType>();
        
        scaleTmpTensor_ = scaleQueue_.AllocTensor<ScalesType>();
        if constexpr (AscendC::IsSameType<ScalesType, float>::value) {
            DataCopyExtParams scaleCopyInParams{1, static_cast<uint32_t>(scaleNumsPerBlcok_ * sizeof(ScalesType)), 0, 0, 0};
            DataCopyPadExtParams<ScalesType> scalePadParams{false, 0, 0, 0};
            DataCopyPad(scaleTmpTensor_, selfWinScaleTensor[winBaseScaleOffset + curScaleOffset], scaleCopyInParams, scalePadParams);
        } else {
            DataCopy(scaleTmpTensor_, selfWinScaleTensor[winBaseScaleOffset + curScaleOffset], scaleNumsPerBlcok_);
        }
        scaleQueue_.EnQue(scaleTmpTensor_);
        scaleTmpTensor_ = scaleQueue_.DeQue<ScalesType>();
        
        // 将数据输出到 GM（AIV 分组：输出位置 = 分区偏移 + 分区内偏移）
        uint64_t outXOffset = winBaseXOffset + curXOffset;
        DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(copyBlockNum * sizeof(XType)), 0, 0, 0};
        DataCopyPad(localWinXGMTensor_[outXOffset], xTmpTensor_, outCopyParams);
        
        xQueue_.FreeTensor(xTmpTensor_);
        scaleQueue_.FreeTensor(scaleTmpTensor_);
    }
    PipeBarrier<PIPE_ALL>();
}

} // QuantMTECommImpl
#endif  // MTE_COMMON_H