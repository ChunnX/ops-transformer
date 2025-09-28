/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file memory_copy.h
   GM->L1
   PA
   PARope
 * \brief
 */
#ifndef MEMMORY_COPY_H
#define MEMMORY_COPY_H
#include "fia_public_define.h"

struct PAShape {
    //uint32_t blockNum;
    uint32_t blockSize;
    uint32_t headNum; //一般为kv的head num，对应n2
    uint32_t headDim; //mla下rope为64，nope为512, 对应d
    uint32_t maxblockNumPerBatch; //block table 每一行的最大个数
    uint32_t actHeadDim; //实际拷贝col大小,考虑到N切块   s*d, 对应d
    uint32_t copyRowNum; //总共要拷贝的行数
    uint32_t copyRowNumAlign;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx; //bnsd的D轴被切，对应N轴被切
};

enum DstLayout {
    NZ = 0,
    GNZ
};

// 场景：query、queryRope、key、value GM to L1
// GM按ND格式存储
// L1按NZ格式存储
// GM的行、列、列的stride (D or ND)  BNSD 和 BSH的区别
template <typename T>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<T> &l1Tensor, GlobalTensor<T> &gmTensor,
                                        uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,       // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct; //nd军阵的行数
    // T为int4场景下，dValue = col / 2，srcDValue = colStride / 2
    nd2nzPara.dValue = col;  //nd矩阵的列数
    nd2nzPara.srcDValue = colStride; //同一nd矩阵相邻行起始地址间的偏移
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

// 场景：GM to L1 PA_NZ（N=1）
// GM按NZ格式存储 -> L1按NZ格式存储 
template <typename T>
__aicore__ inline void DataCopyGmNZToL1(LocalTensor<T>& l1Tensor, GlobalTensor<T> &gmTensor,
                                        uint32_t rowAct,        // 实际需要拷贝的行数
                                        uint32_t dstRowStride,      // Align(Sinner or SinnerTail, 16U)
                                        uint32_t srcRowStride,  // blockSize - rowAct
                                        uint32_t col)   // D
{
    //T为4bit场景下，blockElementCnt*2
    uint32_t blockElementCnt = 32 / sizeof(T);
    DataCopyParams intriParams;
    intriParams.blockCount = col / blockElementCnt;
    intriParams.blockLen = rowAct;
    intriParams.dstStride = dstRowStride; 
    intriParams.srcStride = srcRowStride;
    DataCopy(l1Tensor, gmTensor, intriParams);
}

/*
    适用PA数据从GM拷贝到L1，支持ND、NZ数据；
    PA的layout分 BNBD（blockNum,N,blockSize,D） BBH（blockNum,blockSize,N*D） PA_NZ（Block,D/16,blockSize,16）
    BSH\BSND\TND 为BBH，BNSD为BNBD，PA_NZ 单独一种格式
    DstLayout 后续考虑GNZ
    shape.copyRowNumAlign 需要16字节对齐，如拷贝k矩阵，一次拷贝128*512，遇到尾块 10*512 需对齐到16*512
*/
template <typename T, FIA_LAYOUT SRC_LAYOUT>
__aicore__ inline void DataCopyPA(LocalTensor<T> &dstTensor, //l1
                                  GlobalTensor<T> &srcTensor, //gm
                                  GlobalTensor<int32_t> &blockTableGm,
                                  const PAShape &shape,  // blockSize, headNum, headDim                           
                                  const Position &startPos)  // bacthIdx nIdx curSeqIdx
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Idx;
    //T为4bit场景下，blockElementCnt*2
    uint32_t blockElementCnt = 32 / sizeof(T);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize; // 获取block table上的索引
        uint64_t reaminRowCnt = curS2Idx % shape.blockSize;  // 获取在单个块上超出的行数
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
        // 计算可以拷贝行数
        uint32_t copyRowCnt = shape.blockSize - reaminRowCnt; //一次只能处理一个Block
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;  //一个block未拷满
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim ; //PA的偏移

        if constexpr (SRC_LAYOUT == FIA_LAYOUT::NZ) {
            // layout为（BlockNum, N，D/16, BlockSize, 16）
            offset += (uint64_t)(startPos.n2Idx * shape.blockSize * shape.headDim) + reaminRowCnt * blockElementCnt + startPos.dIdx*shape.blockSize; //mla下n2等于1， n2不等于1的时候怎么搬运？？      
            LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<T> tmpSrcTensor = srcTensor[offset];

            DataCopyGmNZToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, (shape.copyRowNumAlign - copyRowCnt), (shape.blockSize - copyRowCnt), shape.actHeadDim);
        } else {
            uint64_t dStride = shape.headDim;
            if constexpr (SRC_LAYOUT == FIA_LAYOUT::BSH || SRC_LAYOUT == FIA_LAYOUT::BSND || SRC_LAYOUT == FIA_LAYOUT::TND) {
                offset += (uint64_t)(startPos.n2Idx * shape.headDim) + reaminRowCnt * shape.headDim * shape.headNum + startPos.dIdx;
                dStride = shape.headDim * shape.headNum;
            } else {
                offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) + reaminRowCnt * shape.headDim + startPos.dIdx;
            }

            uint32_t dValue = shape.actHeadDim;
            uint32_t srcDValue = dStride;
            LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<T> tmpSrcTensor = srcTensor[offset];

            DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);                     
        }            

        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

#endif
