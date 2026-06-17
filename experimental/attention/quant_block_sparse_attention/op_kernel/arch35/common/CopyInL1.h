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
 * \file CopyInL1.h
 * \brief
 */
#ifndef COPYINL1_H
#define COPYINL1_H

enum class KVLAYOUT
{
    BNBD, // [blockNums, headNum, blockSize, headDim]
    BBH, // [blockNums, blockSize, headNum * headDim]
    NZ // [blockNums, headNum, d1, blockSize, d0], d1 = headDim / d0, d0 = 32 (block byte) / sizeof(KV_T)
};

struct CopyParam{
    uint32_t width;
    uint32_t height;
    uint32_t orgWidth;
};

struct PAShape{
    uint32_t blockNum;
    uint32_t blockSize;
    uint32_t headNum; // 一般为kv的head num
    uint32_t headDim; // mla下rope为64， 非rope为512
    uint32_t maxblockNumPerBatch; // block table 每一行的最大个数
    uint32_t actHeadDim; // 实际拷贝col大小，考虑到N切块 s*d， 对应d
    uint32_t copyRowNum;
    uint32_t copyRowNumAlign;
    uint32_t pageStride;
    uint32_t rowStride;
};

struct Position{
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Offset;
    uint32_t dIdx; // N轴被切，对应D轴被切
};

// 场景：key、value GM to L1
// GM按ND格式存储
// L1按NZ格式存储
// GM的行、列、列的stride（D or ND）BNSD 和 BSH的区别
template<typename L1Type>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                        uint32_t rowAct, 
                                        uint32_t rowAlign,
                                        uint32_t col, // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1; 
    nd2nzPara.nValue = rowAct; // 行数

    nd2nzPara.dValue = col; 
    nd2nzPara.srcDValue = colStride; 
    nd2nzPara.dstNzC0Stride = rowAlign; 
    nd2nzPara.dstNzNStride = 1; 
    nd2nzPara.srcNdMatrixStride = 0; 
    nd2nzPara.dstNzMatrixStride = 0; 
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

template<typename L1Type>
__aicore__ inline void DataCopyGmScaleNDToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                        uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col, // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct;
 
    nd2nzPara.dValue = col;
    nd2nzPara.srcDValue = colStride;
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = nd2nzPara.nValue;
 
    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t*)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, nd2nzPara);
}

template<typename L1Type>
__aicore__ inline void DataCopyGmScaleDNToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                        uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,
                                        uint32_t colStride)
{
    Dn2NzParams dn2nzPara;
    dn2nzPara.dnNum = 1;
    dn2nzPara.nValue = col / 2;
    dn2nzPara.dValue = rowAct;
    dn2nzPara.srcDValue = colStride / 2;
    dn2nzPara.dstNzC0Stride = dn2nzPara.nValue;
    dn2nzPara.dstNzNStride = 1;
    dn2nzPara.srcDnMatrixStride = 0;
    dn2nzPara.dstNzMatrixStride = dn2nzPara.nValue;
 
    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t*)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, dn2nzPara);
}

template<typename L1Type>
__aicore__ inline void DataCopyGmNZToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                        uint32_t rowAct,
                                        uint32_t dstRowStride,
                                        uint32_t srcRowStride,
                                        uint32_t col)
{
    uint32_t blockElementCnt = 32U / sizeof(L1Type);
    if constexpr (IsSameType<L1Type, int4b_t>::value) {
        blockElementCnt = 64U;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = col / blockElementCnt;
    intriParams.blockLen = rowAct;
    intriParams.dstStride = dstRowStride;
    intriParams.srcStride = srcRowStride;
    DataCopy(l1Tensor, gmTensor, intriParams);
}

template<typename L1Type>
__aicore__ inline void GmCopyInToL1PA(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor, KVLAYOUT kvLayout,
                                      const regbaseutil::RunInfo &runInfo, const regbaseutil::ConstInfo &constInfo)
{   
    // PA_BNSD格式
    uint64_t blockTableBaseOffset = runInfo.boIdx * constInfo.maxBlockNumPerBatch;
    uint64_t offset = runInfo.phyBlkNumIdx1 * constInfo.paBlockStride + runInfo.n2oIdx * constInfo.kvSparseBlockSize * constInfo.combineDim;
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = runInfo.phyBlkNumIdx2 == -1 ? 1 : 2; 
    nd2nzPara.nValue = constInfo.kvSparseBlockSize; // 行数

    nd2nzPara.dValue = constInfo.dSize; 
    nd2nzPara.srcNdMatrixStride = runInfo.phyBlkNumIdx2 == -1 ? 0 :
                                  (runInfo.phyBlkNumIdx2 - runInfo.phyBlkNumIdx1) * constInfo.paBlockStride; 
    nd2nzPara.srcDValue = constInfo.combineDim; 
    nd2nzPara.dstNzC0Stride = runInfo.phyBlkNumIdx2 == -1 ?
                              (runInfo.s2SparseBlk1RealSize + 31) >> 5 << 5 :
                              (runInfo.s2SparseBlk1RealSize + runInfo.s2SparseBlk2RealSize + 31) >> 5 << 5; // Fp8 32 对齐, 非Fp8 16对齐
    nd2nzPara.dstNzNStride = 1; 
    nd2nzPara.dstNzMatrixStride = runInfo.phyBlkNumIdx2 == -1 ? 0 : runInfo.s2SparseBlk1RealSize * 32;
    DataCopy(l1Tensor, gmTensor[offset], nd2nzPara);
}

template<typename INPUT_T>
__aicore__ inline void CopyToL1Nd2Nz(const LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmTensor,
    uint32_t nValue, uint32_t dValue, uint32_t srcDValue)
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
    gm2L1Nd2NzParams.nValue = nValue; // 单个ND矩阵的实际行数，单位为元素个数
    gm2L1Nd2NzParams.dValue = dValue; // 单个ND矩阵的实际列数，单位为元素个数
    gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
    gm2L1Nd2NzParams.srcDValue = srcDValue; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102)
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
        IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 31) >> 5 << 5;
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4; // NZ矩阵相邻Block起始地址之间的偏移， 单位为Block个数
#endif
    gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
    gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}

template<typename INPUT_T>
__aicore__ inline void CopyToL1Nd2NzGS1Merge(const LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmTensor,
    uint32_t ndNum, uint32_t nValue, uint32_t dValue, uint32_t srcNdMatrixStride, uint32_t srcDValue, uint32_t dstNzC0Stride) // BSNGD 合轴拷贝
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = ndNum; // ND矩阵的个数
    gm2L1Nd2NzParams.nValue = nValue; // 单个ND矩阵的实际行数，单位为元素个数
    gm2L1Nd2NzParams.dValue = dValue; // 单个ND矩阵的实际列数，单位为元素个数
    gm2L1Nd2NzParams.srcNdMatrixStride = srcNdMatrixStride; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
    gm2L1Nd2NzParams.srcDValue = srcDValue; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102)
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
        IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride = (dstNzC0Stride + 31) >> 5 << 5; // NZ矩阵相邻Block起始地址之间的偏移，单位为Block个数，32对齐
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride = (dstNzC0Stride + 15) >> 4 << 4; // NZ矩阵相邻Block起始地址之间的偏移，单位为Block个数，16对齐
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride = (dstNzC0Stride + 15) >> 4 << 4; // NZ矩阵相邻Block起始地址之间的偏移，单位为Block个数，16对齐
#endif
    gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
    gm2L1Nd2NzParams.dstNzMatrixStride = nValue * 32 / sizeof(INPUT_T); // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}
#endif
