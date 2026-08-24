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
 * \file li_v2_cube_service_common.h
 * \brief LightningIndexerV2 与 QuantLightningIndexerV2 共用的 cube 服务函数
 *        Shared cube service functions used by LightningIndexerV2 and QuantLightningIndexerV2
 */
#ifndef LIGHTNING_INDEXER_V2_CUBE_SERVICE_COMMON_H
#define LIGHTNING_INDEXER_V2_CUBE_SERVICE_COMMON_H

#include "kernel_operator.h"
#include "li_v2_common_utils.h"

using namespace AscendC;

namespace liCubeCommon {

using namespace LICommon;

static constexpr uint64_t BLOCK_CUBE = 16;
static constexpr uint64_t FP16_BLOCK_CUBE = 16;
static constexpr uint64_t FP8_BLOCK_CUBE = 32;
static constexpr FixpipeConfig LI_V2_FIXPIPE_CFG = {CO2Layout::ROW_MAJOR, true};

__aicore__ inline void AllocCubeEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + 0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + 1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + 2);

    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5 + 0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5 + 1);

    SetFlag<HardEvent::M_MTE1>(EVENT_ID3 + 0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID3 + 1);

    SetFlag<HardEvent::FIX_M>(EVENT_ID2 + 0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID2 + 1);
}

__aicore__ inline void FreeCubeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5 + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5 + 1);

    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3 + 0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3 + 1);

    WaitFlag<HardEvent::FIX_M>(EVENT_ID2 + 0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID2 + 1);
}

// GM->L1 的 key 数据搬运，ND 格式转 NZ 格式
// key GM->L1 copy with ND2NZ transform
template <typename K_T>
__aicore__ inline void KeyNd2Nz(LocalTensor<K_T> &keyL1, const GlobalTensor<K_T> &keyGm, uint64_t keyBufIdx,
                                uint64_t keyBufNum, uint64_t keyBufOffset, uint64_t dim, uint64_t s2L1RealSize,
                                uint64_t s2GmOffset, uint64_t tensorKeyOffset)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize; // 行数
    nd2nzPara.dValue = dim;
    nd2nzPara.srcDValue = dim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1[(keyBufIdx % keyBufNum) * keyBufOffset], keyGm[tensorKeyOffset + s2GmOffset * dim], nd2nzPara);
}

// PA 场景的 key GM->L1 搬运，按 block table 索引每块数据
// PA path key GM->L1 copy driven by the block table
template <typename K_T>
__aicore__ inline void KeyNd2NzForPA(LocalTensor<K_T> &keyL1, const GlobalTensor<K_T> &keyGm,
                                     const GlobalTensor<int32_t> &blkTableGm, uint64_t keyBufIdx, uint64_t keyBufNum,
                                     uint64_t keyBufOffset, uint64_t dim, uint64_t blockCube, uint64_t s2L1RealSize,
                                     uint64_t s2GmOffset, uint64_t kCacheBlockSize, uint64_t bIdx,
                                     uint64_t maxBlockNumPerBatch, uint64_t keyStride0)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % kCacheBlockSize;
        uint64_t keyGmOffset =
            blkTableGm.GetValue(bIdx * maxBlockNumPerBatch + s2BlkId) * keyStride0 + s2BlkOffset * dim;

        uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= kCacheBlockSize ? kCacheBlockSize - s2BlkOffset : s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        nd2nzPara.dValue = dim;
        nd2nzPara.srcDValue = dim;
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1[(keyBufIdx % keyBufNum) * keyBufOffset + s2L1Offset * blockCube], keyGm[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// batch, s1, n2, g, d
template <typename Q_T>
__aicore__ inline void QueryNd2Nz(LocalTensor<Q_T> &queryL1, const GlobalTensor<Q_T> &queryGm, uint64_t queryBufIdx,
                                  uint64_t queryBufNum, uint64_t queryBufOffset, uint64_t dim, uint64_t gSize,
                                  uint64_t s1gL1RealSize, uint64_t s1gGmOffset, uint64_t tensorQueryOffset)
{
    uint64_t dstNzC0Stride = CeilAlign(s1gL1RealSize, gSize * 2);
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    nd2nzPara.dValue = dim;
    nd2nzPara.srcDValue = dim;
    nd2nzPara.dstNzC0Stride = CeilAlign(dstNzC0Stride, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1[(queryBufIdx % queryBufNum) * queryBufOffset], queryGm[tensorQueryOffset + s1gGmOffset * dim],
             nd2nzPara);
}

template <typename C_T, typename L0_Q_T, typename L0_K_T>
__aicore__ inline void ComputeL0c(LocalTensor<C_T> &cL0, LocalTensor<L0_Q_T> &queryL0, LocalTensor<L0_K_T> &keyL0,
                                  uint64_t l0BufIdx, uint64_t l0BufNum, uint64_t l0cBufferOffset,
                                  uint64_t l0qBufferOffset, uint64_t l0kBufferOffset, uint64_t s1gL0RealSize,
                                  uint64_t s2L0RealSize, uint64_t headDim)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0[(l0BufIdx % l0BufNum) * l0cBufferOffset], queryL0[(l0BufIdx % l0BufNum) * l0qBufferOffset],
         keyL0[(l0BufIdx % l0BufNum) * l0kBufferOffset], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

// L1 按 2D 分块搬入 L0（非 MX 路径），参数 kStep 根据算子类型由调用侧确定
// L1->L0 2D partial copy for the non-MX path, kStep computed by the call site
template <typename SRC_T, typename DST_T>
__aicore__ inline void LoadData2DPartToL0(LocalTensor<DST_T> &dstL0, LocalTensor<SRC_T> &srcL1, uint64_t l0BufIdx,
                                          uint64_t l0BufNum, uint64_t l0BufOffset, uint64_t l1BufIdx, uint64_t l1BufNum,
                                          uint64_t l1BufOffset, uint64_t l1Offset, uint64_t l1RealSize,
                                          uint64_t l0RealSize, uint64_t kStep)
{
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = LICommon::CeilDiv(l1Offset, BLOCK_CUBE);
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = LICommon::CeilDiv(l0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.kStep = kStep;
    loadData2DParamsV2.srcStride = LICommon::CeilDiv(l1RealSize, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = LICommon::CeilDiv(l0RealSize, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;
    LoadData(dstL0[(l0BufIdx % l0BufNum) * l0BufOffset], srcL1[(l1BufIdx % l1BufNum) * l1BufOffset],
             loadData2DParamsV2);
}

// L0C 结果搬回 UB：float/int32 走普通导出路径，fp8 走固定点量化随路路径
// L0C->UB Fixpipe: plain export for float/int32, on-the-fly fixed-point quantization for fp8
template <typename QK_T, typename CL0_T>
__aicore__ inline void FixpImpl(LocalTensor<QK_T> &mm1ResUB, LocalTensor<CL0_T> &cL0, uint64_t l0BufIdx,
                                uint64_t l0BufNum, uint64_t l0cBufferOffset, uint64_t mFixEvent, uint64_t s1gL0RealSize,
                                uint64_t s1gSizeAlign2G, uint64_t s2L0RealSize, uint64_t mBaseSizeMax,
                                uint64_t s2BaseSize, uint64_t s2BlockL0, uint64_t loop)
{
    SetFlag<HardEvent::M_FIX>(mFixEvent + l0BufIdx % l0BufNum);
    WaitFlag<HardEvent::M_FIX>(mFixEvent + l0BufIdx % l0BufNum);

    if constexpr (std::is_same_v<QK_T, float> || std::is_same_v<QK_T, int32_t>) {
        // s1gL0RealSize：2*gSize(128)对齐, 最大256
        // s2L0RealSize <= S2_BASIC_BLOCK_L0, 未约束
        uint32_t nSize = (s2L0RealSize + 7) >> 3 << 3; // 32B对齐
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上bmm1结果矩阵N方向的size大小 同mmadParams.n 8个元素（32B)对齐
        fixpipeParams.nSize = nSize;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = s1gSizeAlign2G;
        // L0C上matmul结果相邻连续数据片断间隔，单位16*sizeof(T) 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = (s1gL0RealSize + 1) >> 1 << 1;
        // mmResUb上两行之间的间隔，单位 element；ND方案(S1 * S2)脏数据用mask剔除
        fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(QK_T);
        // 双目标模式，按M维度拆分 M/2*N写入每个UB M必须为2的倍数
        fixpipeParams.dualDstCtl = 1;
        if (nSize <= (256 / sizeof(float))) {
            // N方向小于一个bank(256B) 只需搬一个ND块 且不用补齐
            fixpipeParams.nSize = nSize;
            fixpipeParams.params.ndNum = 1;
            fixpipeParams.params.srcNdStride = 0;
            fixpipeParams.params.dstNdStride = 0;
        } else {
            // N方向在(256B, 512B]范围 分2个ND搬
            fixpipeParams.nSize = s2BlockL0 / 2;
            fixpipeParams.params.ndNum = 2;
            fixpipeParams.params.srcNdStride = ((fixpipeParams.mSize + 15) / 16) * fixpipeParams.nSize;
            fixpipeParams.params.dstNdStride = s2BaseSize * mBaseSizeMax / 2;
        }
        Fixpipe<QK_T, CL0_T, LI_V2_FIXPIPE_CFG>(mm1ResUB[(loop % 2) * s2BaseSize / 2],
                                                cL0[(l0BufIdx % l0BufNum) * l0cBufferOffset], fixpipeParams);
    } else {
        uint32_t nSize = CeilAlign(s2L0RealSize, static_cast<uint64_t>(UB_BLOCK / sizeof(QK_T)));
        // 有效数据不足16行，只需输出部分行即可; L0C上bmm1结果矩阵M方向size必须是偶数
        uint32_t mSize = s1gSizeAlign2G;
        // L0C上matmul结果相邻连续数据片断间隔, 单位为16 * sizeof(T)
        uint32_t srcStride = ((mSize + 15) / 16) * 16;
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
        fixpipeParams.nSize = nSize;
        fixpipeParams.mSize = mSize / 2; // M方向每个AIV一半
        fixpipeParams.srcStride = srcStride;
        fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(QK_T); // 落到同一个bank
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        // F322BF16和ReLU属于随路功能，不能与dualDstCtl同时使用，分别写入两个SUB BLOCK。
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.quantPre = F322BF16;
        fixpipeParams.reluEn = true;
        fixpipeParams.subBlockId = 0;
        Fixpipe<QK_T, CL0_T, LI_V2_FIXPIPE_CFG>(mm1ResUB[(loop % 2) * (UB_BANK_STRIDE / sizeof(QK_T))],
                                                cL0[(l0BufIdx % l0BufNum) * l0cBufferOffset], fixpipeParams);

        fixpipeParams.subBlockId = 1;
        Fixpipe<QK_T, CL0_T, LI_V2_FIXPIPE_CFG>(mm1ResUB[(loop % 2) * (UB_BANK_STRIDE / sizeof(QK_T))],
                                                cL0[(l0BufIdx % l0BufNum) * l0cBufferOffset + mSize / 2 * 16],
                                                fixpipeParams);
    }
}

} // namespace liCubeCommon

#endif // LIGHTNING_INDEXER_V2_CUBE_SERVICE_COMMON_H
