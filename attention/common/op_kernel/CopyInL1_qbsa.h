/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file CopyInL1_qbsa.h
 * \brief QBSA-specific MXFP8 helpers.
 */
#ifndef COPYINL1_QBSA_H
#define COPYINL1_QBSA_H

#include "CopyInL1.h"

template <typename INPUT_T>
__aicore__ inline void MxCopyScaleToL1Nd2Nz(const LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmTensor,
                                            uint32_t nValue, uint32_t dValue, uint32_t srcDValue,
                                            uint32_t dstNzC0Stride)
{
    // MXFP8 scale 为 fp8_e8m0，但 scale 搬运路径按 bf16 粒度执行。
    // value scale 的 GM shape 是 [physicalBlock, N, blockSize/64, DV, 2]，最后一维两个 e8m0
    // reinterpret 成一个 bf16 lane，因此压缩的是 D 方向，block 内 token group 行数不能除以 2。
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = 1;
    gm2L1Nd2NzParams.nValue = nValue;
    gm2L1Nd2NzParams.dValue = dValue >> 1;
    gm2L1Nd2NzParams.srcNdMatrixStride = 0;
    gm2L1Nd2NzParams.srcDValue = srcDValue >> 1;
    gm2L1Nd2NzParams.dstNzC0Stride = dstNzC0Stride;
    gm2L1Nd2NzParams.dstNzNStride = 1;
    gm2L1Nd2NzParams.dstNzMatrixStride = gm2L1Nd2NzParams.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, gm2L1Nd2NzParams);
}

template <typename INPUT_T>
__aicore__ inline void MxCopyScaleToL1Dn2Nz(const LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmTensor,
                                            uint32_t nValue, uint32_t dValue, uint32_t srcDValue)
{
    // DN2NZ 版本用于跟随 BMM1 转置操作数的 scale。
    // QScale GM shape 为 [T,N,D/64,2]；KScale GM shape 为 [blockNum,N,blockSize,D/64,2]。
    // reinterpret 为 bf16 后，
    // 最后一维两个 e8m0 元素变成一个拷贝元素。这样可以保证 Q/K 按转置语义
    // 被 MX LoadData 消费时，scale layout 与数据 layout 对齐。
    Dn2NzParams gm2L1Dn2NzParams;
    gm2L1Dn2NzParams.dnNum = 1;
    gm2L1Dn2NzParams.nValue = nValue >> 1;
    gm2L1Dn2NzParams.dValue = dValue;
    gm2L1Dn2NzParams.srcDnMatrixStride = 0;
    gm2L1Dn2NzParams.srcDValue = srcDValue >> 1;
    gm2L1Dn2NzParams.dstNzC0Stride = nValue >> 1;
    gm2L1Dn2NzParams.dstNzNStride = 1;
    gm2L1Dn2NzParams.dstNzMatrixStride = gm2L1Dn2NzParams.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t *)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, gm2L1Dn2NzParams);
}

template <typename L1Type>
__aicore__ inline void MxGmFlatScaleCopyInToL1PAForDN(LocalTensor<L1Type> &l1Tensor, GlobalTensor<L1Type> &gmTensor,
                                                      GlobalTensor<int32_t> &blockTableGm, const PAShape &shape,
                                                      const Position &startPos)
{
    // PA scale helper，用于 K per-token-group scale：
    //   GM 逻辑 shape：[physicalBlock, N, blockSize, D/64, 2]
    //   PA 查表流程：逻辑 token 偏移 -> block_table -> 物理 block id + block 内 token offset
    //   L1 拷贝方式：DN2NZ，因为对应的数据操作数按转置方式加载。
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = static_cast<uint64_t>(startPos.bIdx) * shape.maxblockNumPerBatch;
    uint64_t curS2Idx = startPos.s2Offset;
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint32_t remainRowCnt = static_cast<uint32_t>(curS2Idx % shape.blockSize);
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }

        // Flat BNBD scale rows use headDim as their source row stride.
        uint64_t offset = idInBlockTable * shape.pageStride +
                          static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.headDim +
                          static_cast<uint64_t>(remainRowCnt) * shape.headDim + startPos.dIdx;
        MxCopyScaleToL1Dn2Nz(l1Tensor[copyFinishRowCnt * shape.actHeadDim], gmTensor[offset], shape.actHeadDim,
                             copyRowCnt, shape.headDim);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename L1Type>
__aicore__ inline void MxGmFlatScaleCopyInToL1PAForND(LocalTensor<L1Type> &l1Tensor, GlobalTensor<L1Type> &gmTensor,
                                                      GlobalTensor<int32_t> &blockTableGm, const PAShape &shape,
                                                      const Position &startPos)
{
    // PA scale helper，用于 value per-channel-group scale：
    // GM 逻辑 shape：[physicalBlock, N, blockSize/64, DV, 2]
    // 每个拷贝行覆盖 block 内 64 个 value token。
    // 这里使用 ND2NZ，因为 V 在 BMM2 中作为非转置右操作数被消费。
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = static_cast<uint64_t>(startPos.bIdx) * shape.maxblockNumPerBatch;
    uint64_t curS2Idx = startPos.s2Offset;
    constexpr uint32_t blockElementCnt = 32U / sizeof(L1Type);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint32_t remainRowCnt = static_cast<uint32_t>(curS2Idx % shape.blockSize);
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }

        // Flat BNBD scale rows use headDim as their source row stride.
        uint64_t offset = idInBlockTable * shape.pageStride +
                          static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.headDim +
                          static_cast<uint64_t>(remainRowCnt) * shape.headDim + startPos.dIdx;
        MxCopyScaleToL1Nd2Nz(l1Tensor[copyFinishRowCnt * blockElementCnt], gmTensor[offset], copyRowCnt,
                             shape.actHeadDim, shape.headDim, shape.copyRowNumAlign);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

#endif // COPYINL1_QBSA_H
