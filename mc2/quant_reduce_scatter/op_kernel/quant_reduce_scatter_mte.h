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

// 之后可修改成从tiling侧获取数据切块大小
constexpr static uint32_t X_BLOCK_ALIGN_NUM = 1024U;  // x 分块大小的最小对齐粒度（元素数，与 Win 区数据布局对齐）
constexpr static uint64_t MX_SCALES_LAST_DIM = 2U; // MX量化scales最后一维的大小

template<TemplateTypeClass>
class QuantReduceScatterMte {
public:
    __aicore__ inline QuantReduceScatterMte() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                TPipe *pipe, const QuantReduceScatterTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ClearSumTensor();
    __aicore__ inline void ReadDataBlockReduceSum(uint64_t curXOffset, uint64_t curScaleOffset, uint32_t xNum, uint32_t scaleNum);
    __aicore__ inline void ExecuteReduceScatter();

    MTECommunication<TemplateType> mteComm_; // MTE 通信相关实现
    VectorCompute<TemplateType> vecComp_; // vector 计算相关实现

    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    LocalTensor<float> sumTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_; // 用于Reduce_sum 求和

    uint64_t xSize_{0};
    uint64_t totalWinSize_{0};
    uint64_t alignedXSize_{0};
    // reduceScater进行all2all过程，数据要按卡数进行均分，以下为计算切分相关的参数
    uint64_t scaleSize_{0};
    uint64_t xSliceSize_{0};
    uint64_t xSliceSizeNums_{0};      // 按卡均分后每片x数据的个数
    uint64_t scaleSliceNums_{0};      // 按卡均分后每片scale数据的个数

    // 动态分块参数（基于xPerBlock_最大化UB利用）
    uint32_t xPerBlock_{0};           // 动态计算的每块数据元素数
    uint32_t scaleNumsPerBlock_{0};   // 每块对应的scale数量
    uint32_t myTotalBlockNums_{0};    // 总块数（基于xPerBlock_）
    uint32_t myAssignedBlockNums_{0}; // 当前核分配的块数
    uint32_t myRound_{0};             // 基础轮次数
    uint32_t myTailBlockNums_{0};     // 尾块数（分配给前几个核）
    uint32_t myLastAivId_{0};         // 最后一个核的ID
    uint64_t myXOffset_{0};           // 当前核的起始x偏移
    uint64_t myScaleOffset_{0};       // 当前核的起始scale偏移
    uint32_t myTailXNums_{0};         // 尾块的x数量
};

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const QuantReduceScatterTilingData *tilingData)
{
    // 初始化HcclContext
    mteComm_.InitHcclContext();

    /* quant_reduce_scatter自己的数据 */
    auto&& tiliingDatainfo = tilingData->quantReduceScatterTilingInfo;
    uint64_t aivNum = tiliingDatainfo.aivNum;
    uint64_t aivId = GetBlockIdx();

    totalWinSize_ = tiliingDatainfo.totalWinSize;
    xSize_ = tiliingDatainfo.bs * tiliingDatainfo.hiddenSize * sizeof(XType);
    scaleSize_ = tiliingDatainfo.bs * tiliingDatainfo.scaleHiddenSize * sizeof(ScalesType);
    // 对于mx的scale是三维，最后一维为2，总scales的数据量需要再乘以2
    if constexpr(AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleSize_ *= MX_SCALES_LAST_DIM;
    }
    xSliceSize_ = xSize_ / (mteComm_.hcclContext_->rankDim); // all2all过程，数据需要按卡数均分
    xSliceSizeNums_ = xSliceSize_ / sizeof(XType); // 按卡均分后每片x数据的个数
    scaleSliceNums_ = scaleSize_ / (mteComm_.hcclContext_->rankDim * sizeof(ScalesType)); // all2all过程，每张卡需要的scale数据个数

    tPipe->Reset();

    // ============ 动态计算 UB 空间分配，最大化 xPerBlock_ ============
    // 1. 计算固定开销（不随 xPerBlock_ 变化的 buffer）
    // 注：mteComm_ 的 xQueue_/scaleQueue_ 改为动态分块（与 ExecuteReduceScatter 一致），不再是固定开销
    //   xQueue_ 移到 dynamic（BUFFER_NUM * sizeof(XType)）
    //   scaleQueue_ 实际开销极小（≤ 1/16 byte/x），用 BUFFER_NUM * X_BLOCK_BYTES 上限估算放固定
    uint64_t mteCommFixedSpace = BUFFER_NUM * X_BLOCK_BYTES +                    // scaleQueue_ 上限估算
                                  UB_ALIGN_BYTES +                                // winFlagsBuf_
                                  UB_ALIGN_BYTES +                                // writeStateBuf_
                                  mteComm_.hcclContext_->rankDim * UB_ALIGN_BYTES +  // readStateBuf_
                                  mteComm_.hcclContext_->rankDim * UB_ALIGN_BYTES;   // stateResetBuf_

    uint64_t vecCompFixedSpace;
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        vecCompFixedSpace = MX_MAX_SCALE_NUM * sizeof(float) +         // castScaleBuf_
                            MX_MAX_SCALE_NUM * sizeof(bfloat16_t);     // tempBf16ScaleBuf_
    } else {
        vecCompFixedSpace = KG_MAX_SCALE_NUM * sizeof(float);          // castScaleBuf_
    }

    uint64_t selfFixedSpace = 0;  // scaleInQue改为动态，不算在固定开销中

    // 2. 计算动态开销系数（每单位 xPerBlock_ 需要的字节）
    uint64_t mteCommDynamicBytes = BUFFER_NUM * sizeof(OutputType) +  // xOutQueue_ (双 buffer)
                                    BUFFER_NUM * sizeof(XType);        // xQueue_ (双 buffer，新增动态)
    uint64_t vecCompDynamicBytes = sizeof(float) + sizeof(float) + sizeof(bfloat16_t) + sizeof(float16_t);  // brcbBuf_ + xCastBuf_ + xCastBf16Buf_ + xCastfp16Buf_

    // scaleInQue动态大小：每xPerBlock_需要的scale数量
    // MXFP: scaleNum = xPerBlock_/32, size = scaleNum*1B
    // KG: scaleNum = xPerBlock_/128, size = scaleNum*4B = xPerBlock_/32B
    uint64_t scaleInQueBytesPerX = sizeof(ScalesType) / (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value ? MX_SIZE : PER_GROUP_SIZE);
    uint64_t selfDynamicBytes = sizeof(float) + BUFFER_NUM * sizeof(XType) + BUFFER_NUM * scaleInQueBytesPerX;

    uint64_t totalDynamicBytes = mteCommDynamicBytes + vecCompDynamicBytes + selfDynamicBytes;
    uint64_t totalFixedSpace = mteCommFixedSpace + vecCompFixedSpace + selfFixedSpace;

    // 3. 根据剩余 UB 空间计算最大 xPerBlock_
    uint64_t remainingSpace = TOTAL_UB_SIZE - totalFixedSpace;
    uint64_t maxXPerBlock = remainingSpace / totalDynamicBytes;

    // 3.5 stage3：以 host 推荐值（基于 TARGET_ITER 公式）作为上限，kernel 端 ub_max 仍作 safety net
    uint32_t hostRecommended = tiliingDatainfo.xPerBlock;
    if (hostRecommended != 0U && maxXPerBlock > hostRecommended) {
        maxXPerBlock = hostRecommended;
    }

    // 4. 考虑数据量限制：不超过每卡分片数据量 xSliceSizeNums_
    xPerBlock_ = static_cast<uint32_t>(maxXPerBlock < xSliceSizeNums_ ? maxXPerBlock : xSliceSizeNums_);

    // 5. 对齐到 X_BLOCK_ALIGN_NUM（保持与Win区数据布局兼容）
    xPerBlock_ = FloorAlign(xPerBlock_, X_BLOCK_ALIGN_NUM);
    if (xPerBlock_ == 0) {
        xPerBlock_ = X_BLOCK_ALIGN_NUM;  // 最小值 X_BLOCK_ALIGN_NUM (512)
    }

    // 6. 计算scale数量（real 值，用于 GM 偏移与 DataCopy 字节数）
    if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleNumsPerBlock_ = xPerBlock_ / MX_SIZE;
    } else {
        scaleNumsPerBlock_ = xPerBlock_ / PER_GROUP_SIZE;
    }
    // UB buffer 大小按 32B 对齐分配（硬件要求），但 GM 偏移与 DataCopy 用 real 值（DataCopyPad 处理 sub-32B 拷贝）
    uint32_t scaleAlignNum = UB_ALIGN_BYTES / sizeof(ScalesType);
    uint32_t scaleAllocCount = CeilAlignU32(scaleNumsPerBlock_, scaleAlignNum);
    uint64_t scaleInQueSize = scaleAllocCount * sizeof(ScalesType);

    // 7. 分配 UB buffer（动态大小）
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, xPerBlock_ * sizeof(XType));
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, scaleInQueSize);
    tPipe->InitBuffer(sumBuf_, xPerBlock_ * sizeof(float));
    sumTensor_ = sumBuf_.Get<float>();

    // ============ 计算动态分块参数（基于xPerBlock_） ============
    myTotalBlockNums_ = CeilDiv(xSliceSizeNums_, xPerBlock_);  // 总块数（基于每卡分片数据量）
    myRound_ = myTotalBlockNums_ / aivNum;                    // 每核基础轮次
    myTailBlockNums_ = myTotalBlockNums_ % aivNum;            // 尾块数
    myAssignedBlockNums_ = aivId < myTailBlockNums_ ? myRound_ + 1 : myRound_;
    myLastAivId_ = (myRound_ == 0) ? myTailBlockNums_ - 1 : aivNum - 1;

    // 计算当前核的起始偏移（基于分片数据）
    uint64_t blockIdx = aivId * myRound_ + (aivId < myTailBlockNums_ ? aivId : myTailBlockNums_);
    myXOffset_ = blockIdx * xPerBlock_;
    myScaleOffset_ = blockIdx * scaleNumsPerBlock_;
    myTailXNums_ = BlockAlignMod(xSliceSizeNums_, xPerBlock_);

    // 设置vecComp的块大小
    vecComp_.SetBlockSize(xPerBlock_);

    // MTE搬运参数初始化：与 ExecuteReduceScatter 用相同的动态分块策略（基于 xPerBlock_）
    // 这样每核 CopyDataToWin 搬运的范围 = ExecuteReduceScatter 读取的范围，
    // 跨 rank 同步后每核读自己搬的数据，不会跨核读取
    mteComm_.round_ = myRound_;
    mteComm_.tailBlockNums_ = myTailBlockNums_;
    mteComm_.ComputeTailAivId(aivNum);
    mteComm_.SetBlockSize(xPerBlock_, aivNum, myTailXNums_, scaleNumsPerBlock_);
    mteComm_.InitParams();
    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);

    // 初始化GM上的Tensor，包括Win区
    mteComm_.InitGMTensor(x, scales, output, xSize_, totalWinSize_);
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
    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    DataCopyParams scaleCopyParams;
    scaleCopyParams.blockLen = scaleNum * sizeof(ScalesType);
    scaleCopyParams.blockCount = 1;
    DataCopyPadParams scalePadParams;
    DataCopyPad(scaleTmpTensor, remoteWinScaleTensor_[curScaleOffset], scaleCopyParams, scalePadParams);
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();

    /* 反量化计算与ReduceSum求和 */
    vecComp_.DequantReduceSum(xTmpTensor, scaleTmpTensor, sumTensor_);
    xInQueue_.FreeTensor(xTmpTensor);
    scaleInQue.FreeTensor(scaleTmpTensor);
}


template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ClearSumTensor()
{
    Duplicate<float>(sumTensor_, (float)0.0, xPerBlock_); // sumTensor 清零
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ExecuteReduceScatter()
{
    // 读状态位，软同步
    mteComm_.ReadStatus();
    uint64_t aivId = GetBlockIdx();

    // 遍历需要处理的数据块（基于动态xPerBlock_分块）
    for (uint64_t curBlock = 0; curBlock < myAssignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = myXOffset_ + curBlock * xPerBlock_;
        uint64_t curScaleOffset = myScaleOffset_ + curBlock * scaleNumsPerBlock_;

        // 计算当前块的x数量（尾块处理）
        uint32_t curXNum = xPerBlock_;
        uint32_t curScaleNum = scaleNumsPerBlock_;
        if ((aivId == myLastAivId_) && (curBlock == myAssignedBlockNums_ - 1)) {
            curXNum = myTailXNums_;
            // 尾块的scale数量：real 值（DataCopyPad 处理 sub-32B 拷贝，不再需要硬对齐）
            if constexpr (AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
                curScaleNum = CeilDivU32(curXNum, MX_SIZE);
            } else {
                curScaleNum = CeilDivU32(curXNum, PER_GROUP_SIZE);
            }
        }

        ClearSumTensor(); // sumTensor 清零

        // 遍历每张卡，读取其Win区的数据，采取错卡序读取，从自己卡上读起
        /* rank0: [0,1,2]; rank1: [1,2,0]; rank2: [2,0,1] */
        uint32_t startRankId = mteComm_.hcclContext_->rankId;
        for (uint32_t i = 0; i < mteComm_.hcclContext_->rankDim; ++i) {
            uint32_t remoteRankId = (startRankId + i) % mteComm_.hcclContext_->rankDim;

            // 获取对端Win区中数据区相关的地址
            GM_ADDR remoteDataSpaceGm = mteComm_.GetWinDataAddrGm(remoteRankId, mteComm_.winBufferFlags_);

            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteDataSpaceGm);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)(remoteDataSpaceGm + xSize_));

            // 读取对端对应地址的 x 和 scale数据，进行反量化和求和
            // ReduceScatter过程，all2all仅需与rankId相关的数据，加上本卡偏移
            uint64_t curRankXOffset = curXOffset + mteComm_.hcclContext_->rankId * xSliceSizeNums_;
            uint64_t curRankScaleOffset = curScaleOffset + mteComm_.hcclContext_->rankId * scaleSliceNums_;
            ReadDataBlockReduceSum(curRankXOffset, curRankScaleOffset, curXNum, curScaleNum);
        }

        // 将计算好的数据拷贝到输出tensor
        mteComm_.CopyResultToOutput(curXOffset, sumTensor_, curXNum);
    }
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Process()
{
    mteComm_.template CopyDataToWin<true>(xSliceSizeNums_, scaleSliceNums_);
    mteComm_.WriteStatusToWin();
    SyncAll();                        // AIV全核同步，确保CopyDataToWin完成
    // 执行ReduceScatter过程：ReadStatus内leader自旋+SyncAll广播
    ExecuteReduceScatter();
}
} // QuantReduceScatterImpl
#endif  // QUANT_REDUCE_SCATTER_MTE_H