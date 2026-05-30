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
 * \file quant_reduce_scatter_mte.h
 * \brief quant_reduce_scatter mte通信kernel代码逻辑
 */

#ifndef QUANT_REDUCE_SCATTER_MTE_H
#define QUANT_REDUCE_SCATTER_MTE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_reduce_scatter_tiling_data.h"
#include "utils.h"
#include "mte_comm.h"
#include "vec_comp.h"

namespace QuantReduceScatterImpl {

using namespace QuantMTECommImpl;
using namespace VectorComputeImpl;
using namespace AscendC;

constexpr static uint64_t MX_SCALES_LAST_DIM = 2U; // MX量化scales最后一维的大小

template<TemplateTypeClass>
class QuantReduceScatterMte {
public:
    __aicore__ inline QuantReduceScatterMte() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                TPipe *pipe, const QuantReduceScatterTilingData *tilDat);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ClearSumTensor();
    __aicore__ inline void ReadDataBlockReduceSum(uint64_t curXOffset, uint64_t curScaleOffset,
                                                    uint32_t xNum, uint32_t scaleNum);
    __aicore__ inline void ExecuteReduceScatter();
    __aicore__ inline void ParseTilingInfo(const QuantReduceScatterTilingData *tilDat);
    __aicore__ inline void ComputeXPerBlock(const QuantReduceScatterTilingData *tilDat, TPipe *tPipe);
    __aicore__ inline void ComputeBlockDistribution(uint64_t aivNum);
    __aicore__ inline void InitSubModules(GM_ADDR x, GM_ADDR scales, GM_ADDR output, TPipe *tPipe, uint64_t aivNum);

    MTECommunication<TemplateType> mteCom2_; // MTE 通信相关实现
    VectorCompute<TemplateType> vecCmp2_; // vector 计算相关实现

    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    LocalTensor<float> sumTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue_; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_; // 用于Reduce_sum 求和

    uint64_t xSize_{0};
    uint64_t totalWinSize_{0};
    // reduceScater进行all2all过程，数据要按卡数进行均分，以下为计算切分相关的参数
    uint64_t scaleSize_{0};
    uint64_t xSliceSizeNums_{0};      // 按卡均分后每片x数据的个数
    uint64_t scaleSliceNums_{0};      // 按卡均分后每片scale数据的个数

    // 动态分块参数（基于xPerBlk_最大化UB利用）
    uint32_t xPerBlk_{0};           // 动态计算的每块数据元素数
    uint32_t sclNumsPerBlk_{0};   // 每块对应的scale数量
    uint32_t totalBlockNums_{0};    // 总块数（基于xPerBlk_）
    uint32_t assignedBlockNums_{0}; // 当前核分配的块数
    uint32_t rnd_{0};             // 基础轮次数
    uint32_t tailBlockNums_{0};     // 尾块数（分配给前几个核）
    uint32_t lastAivId_{0};         // 最后一个核的ID
    uint64_t xOffset_{0};           // 当前核的起始x偏移
    uint64_t scaleOffset_{0};       // 当前核的起始scale偏移
    uint32_t tailXNums_{0};         // 尾块的x数量
};

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const QuantReduceScatterTilingData *tilDat)
{
    mteCom2_.InitHcclContext();
    ParseTilingInfo(tilDat);
    tPipe->Reset();
    ComputeXPerBlock(tilDat, tPipe);
    uint64_t aivNum = tilDat->quantReduceScatterTilingInfo.aivNum;
    ComputeBlockDistribution(aivNum);
    InitSubModules(x, scales, output, tPipe, aivNum);
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ParseTilingInfo(
    const QuantReduceScatterTilingData *tilDat)
{
    auto&& info = tilDat->quantReduceScatterTilingInfo;
    totalWinSize_ = info.totalWinSize;
    xSize_ = info.bs * info.hiddenSize * sizeof(XType);
    scaleSize_ = info.bs * info.scaleHiddenSize * sizeof(ScalesType);
    if constexpr(AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleSize_ *= MX_SCALES_LAST_DIM;
    }
    uint64_t xSliceSize = xSize_ / (mteCom2_.hcclContext_->rankDim);
    xSliceSizeNums_ = xSliceSize / sizeof(XType);
    scaleSliceNums_ = scaleSize_ / (mteCom2_.hcclContext_->rankDim * sizeof(ScalesType));
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ComputeXPerBlock(
    const QuantReduceScatterTilingData *tilDat, TPipe *tPipe)
{
    auto&& info = tilDat->quantReduceScatterTilingInfo;

    // 固定开销：不随 xPerBlk_ 变化的 buffer
    uint64_t mteCommFixedSpace = BUFFER_NUM * X_BLOCK_BYTES +      // scaleQueue_
                                  UB_ALIGN_BYTES +                  // winFlagsBuf_
                                  UB_ALIGN_BYTES +                  // writeStateBuf_
                                  mteCom2_.hcclContext_->rankDim * UB_ALIGN_BYTES +  // readStateBuf_
                                  mteCom2_.hcclContext_->rankDim * UB_ALIGN_BYTES;   // stateResetBuf_

    // 动态开销：每增加 1 个 x 需要的 UB 字节（整数部分，分数部分见下方比例校正）
    uint64_t baseDynamic = BUFFER_NUM * sizeof(OutputType) +  // xOutQueue_
                           BUFFER_NUM * sizeof(XType) +       // xQueue_
                           sizeof(float) +                    // brcbBuf_
                           sizeof(float) +                    // xCastBuf_
                           sizeof(bfloat16_t) +               // xCastBf16Buf_
                           sizeof(float16_t) +                // xCastfp16Buf_
                           sizeof(float) +                    // sumBuf_
                           BUFFER_NUM * sizeof(XType);        // xInQueue_

    uint64_t remainingSpace = TOTAL_UB_SIZE - mteCommFixedSpace;
    uint64_t maxXPerBlock;
    // scale buffer 的开销以 FracDiv 精确计入 UB 预算（避免 < 1 B per X 被整数截断）
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        maxXPerBlock = FracDiv(remainingSpace, baseDynamic, 4U, 1U);   // + 1/4 B per X
    } else {
        maxXPerBlock = FracDiv(remainingSpace, baseDynamic, 32U, 3U);  // + 3/32 B per X
    }

    // host 推荐值作为上限，kernel UB 约束作为 safety net
    uint32_t hostRecommended = info.xPerBlock;
    if (hostRecommended != 0U && maxXPerBlock > hostRecommended) {
        maxXPerBlock = hostRecommended;
    }

    // 每卡分片数据量上限约束
    xPerBlk_ = static_cast<uint32_t>(maxXPerBlock < xSliceSizeNums_ ? maxXPerBlock : xSliceSizeNums_);

    // 对齐到 alignBlock
    uint32_t alignBlock = info.alignBlock;
    xPerBlk_ = FloorAlign(xPerBlk_, alignBlock);
    if (xPerBlk_ == 0) {
        xPerBlk_ = alignBlock;
    }

    // 每块 scale 数量（real 值，GM 偏移用；UB 分配按 32B 对齐）
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        sclNumsPerBlk_ = xPerBlk_ / MX_SIZE;
    } else {
        sclNumsPerBlk_ = xPerBlk_ / PER_GROUP_SIZE;
    }

    uint32_t scaleAlignNum = UB_ALIGN_BYTES / sizeof(ScalesType);
    uint32_t scaleAllocCount = CeilAlignU32(sclNumsPerBlk_, scaleAlignNum);
    uint64_t scaleInQueSize = scaleAllocCount * sizeof(ScalesType);

    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xPerBlk_ * sizeof(XType));
    tPipe->InitBuffer(scaleInQue_, BUFFER_NUM, scaleInQueSize);
    tPipe->InitBuffer(sumBuf_, xPerBlk_ * sizeof(float));
    sumTensor_ = sumBuf_.Get<float>();
}

// 计算多核分块参数：每核分配块数、尾块归属、起始偏移
template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ComputeBlockDistribution(uint64_t aivNum)
{
    uint64_t aivId = GetBlockIdx();
    totalBlockNums_ = CeilDiv(xSliceSizeNums_, xPerBlk_);
    rnd_ = totalBlockNums_ / aivNum;
    tailBlockNums_ = totalBlockNums_ % aivNum;
    assignedBlockNums_ = aivId < tailBlockNums_ ? rnd_ + 1 : rnd_;
    lastAivId_ = (rnd_ == 0) ? tailBlockNums_ - 1 : aivNum - 1;

    uint64_t localIdx = aivId * rnd_ + (aivId < tailBlockNums_ ? aivId : tailBlockNums_);
    xOffset_ = localIdx * xPerBlk_;
    scaleOffset_ = localIdx * sclNumsPerBlk_;
    tailXNums_ = BlockAlignMod(xSliceSizeNums_, xPerBlk_);
}

// 初始化 vecComp、mteComm 子模块参数并分配其 UB buffer，绑定 GM Tensor
template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::InitSubModules(
    GM_ADDR x, GM_ADDR scales, GM_ADDR output, TPipe *tPipe, uint64_t aivNum)
{
    vecCmp2_.SetBlockSize(xPerBlk_);
    vecCmp2_.SetScaleNums(sclNumsPerBlk_);
    mteCom2_.rnd_ = rnd_;
    mteCom2_.tailBlockNums_ = tailBlockNums_;
    mteCom2_.ComputeTailAivId(aivNum);
    mteCom2_.SetBlockSize(xPerBlk_, aivNum, tailXNums_, sclNumsPerBlk_);
    mteCom2_.InitParams();
    mteCom2_.InitBuffer(tPipe);
    vecCmp2_.InitBuffer(tPipe);
    mteCom2_.InitGMTensor(x, scales, output, xSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ReadDataBlockReduceSum(uint64_t curXOffset,
    uint64_t curScaleOffset, uint32_t xNum, uint32_t scaleNum)
{
    /* 读取 x 从 Win -> UB */
    LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
    DataCopy(xTmpTensor, remoteWinXTensor_[curXOffset], xNum);
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<XType>();

    /* 读取 scale 从 Win -> UB（DataCopyPad 处理 sub-32B 对齐） */
    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue_.AllocTensor<ScalesType>();
    DataCopyParams scaleCopyParams;
    scaleCopyParams.blockLen = scaleNum * sizeof(ScalesType);
    scaleCopyParams.blockCount = 1;
    DataCopyPadParams scalePadParams;
    DataCopyPad(scaleTmpTensor, remoteWinScaleTensor_[curScaleOffset], scaleCopyParams, scalePadParams);
    scaleInQue_.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue_.DeQue<ScalesType>();

    /* 反量化计算与ReduceSum求和 */
    vecCmp2_.DequantReduceSum(xTmpTensor, scaleTmpTensor, sumTensor_);
    xInQueue_.FreeTensor(xTmpTensor);
    scaleInQue_.FreeTensor(scaleTmpTensor);
}


template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ClearSumTensor()
{
    Duplicate<float>(sumTensor_, (float)0.0, xPerBlk_); // sumTensor 清零
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ExecuteReduceScatter()
{
    // 读状态位，软同步
    mteCom2_.ReadStatus();
    uint64_t aivBlockId = GetBlockIdx();

    for (uint64_t curBlock = 0; curBlock < assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset_ + curBlock * xPerBlk_;
        uint64_t curScaleOffset = scaleOffset_ + curBlock * sclNumsPerBlk_;

        uint32_t curXNum = xPerBlk_;
        uint32_t curScaleNum = sclNumsPerBlk_;
        if ((aivBlockId == lastAivId_) && (curBlock == assignedBlockNums_ - 1)) {
            curXNum = tailXNums_;
            if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
                curScaleNum = CeilDivU32(curXNum, MX_SIZE);
            } else {
                curScaleNum = CeilDivU32(curXNum, PER_GROUP_SIZE);
            }
        }

        ClearSumTensor(); // sumTensor 清零

        // 遍历每张卡，读取其Win区的数据，采取错卡序读取，从自己卡上读起
        /* rank0: [0,1,2]; rank1: [1,2,0]; rank2: [2,0,1] */
        uint32_t baseRankId = mteCom2_.hcclContext_->rankId;
        for (uint32_t rankIdx = 0; rankIdx < mteCom2_.hcclContext_->rankDim; ++rankIdx) {
            uint32_t peerRank = (baseRankId + rankIdx) % mteCom2_.hcclContext_->rankDim;

            GM_ADDR remoteDataGm = mteCom2_.GetWinDataAddrGm(peerRank, mteCom2_.winBufferFlags_);

            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteDataGm);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)(remoteDataGm + xSize_));

            // 读取对端对应地址的 x 和 scale数据，进行反量化和求和
            // ReduceScatter过程，all2all仅需与rankId相关的数据，加上本卡偏移
            uint64_t curRankXOffset = curXOffset + mteCom2_.hcclContext_->rankId * xSliceSizeNums_;
            uint64_t curRankScaleOffset = curScaleOffset + mteCom2_.hcclContext_->rankId * scaleSliceNums_;
            ReadDataBlockReduceSum(curRankXOffset, curRankScaleOffset, curXNum, curScaleNum);
        }

        // 将计算好的数据拷贝到输出tensor
        mteCom2_.CopyResultToOutput(curXOffset, sumTensor_, curXNum);
    }
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Process()
{
    // 纯AIV过程
    if ASCEND_IS_AIC {
        return;
    }

    // 一次性拷贝完所有数据到本地卡win区
    mteCom2_.template CopyDataToWin<true>(xSliceSizeNums_, scaleSliceNums_);
    // 写入状态到状态区
    mteCom2_.WriteStatusToWin();
    // 执行ReduceScatter过程：等待状态区同步，读取数据并进行反量化ReduceSum
    ExecuteReduceScatter();
}
} // QuantReduceScatterImpl
#endif  // QUANT_REDUCE_SCATTER_MTE_H