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
 * \file matmul_all_reduce_dequant_perchannel.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_DEQUANT_PERCHANNEL_H
#define MATMUL_ALL_REDUCE_DEQUANT_PERCHANNEL_H

namespace MatmulAllReduceDequantPerchannelImpl {
using namespace AscendC;
using namespace std;

constexpr uint32_t TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 = 191 * 1024;
constexpr uint16_t DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8 = 2;
constexpr uint32_t BF16_BUF_CNT_MATMUL_ALLREDUCE_INT8 = 8;
constexpr uint32_t FP16_BUF_CNT_MATMUL_ALLREDUCE_INT8 = 4;
constexpr int32_t FP16_BUF_SPLIT_MN_CNT_MATMUL_ALLREDUCE_INT8 = 6;
constexpr int32_t BF16_BUF_SPLIT_MN_CNT_MATMUL_ALLREDUCE_INT8 = 10;
constexpr uint32_t BYTE512_MATMUL_ALLREDUCE_INT8 = 512;
constexpr uint32_t BYTE32_MATMUL_ALLREDUCE_INT8 = 32;
constexpr uint32_t SPLIT_M_MATMUL_ALLREDUCE_INT8 = 1;
constexpr uint32_t SPLIT_MN_MATMUL_ALLREDUCE_INT8 = 2;

template <class T>
class MatmulAllReduceDequantPerchannel {
public:
    __aicore__ inline MatmulAllReduceDequantPerchannel()
    {
    }
    __aicore__ inline void Init(TPipe *tPipe, uint32_t quantBufSz)
    {
        pipe = tPipe;
        uint32_t scBufSz = quantBufSz;
        if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            scBufSz = dequantAlginN_;
        }
        if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            pipe->InitBuffer(inQueueXS, 1, scBufSz * sizeof(T));
        } else {
            pipe->InitBuffer(inQueueXS, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, scBufSz * sizeof(T));
        }
        pipe->InitBuffer(inQueueX2, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, quantBufSz * sizeof(int8_t));
        pipe->InitBuffer(outQueueZ2, DOUBLE_BUFFER_MATMUL_ALLREDUCE_INT8, quantBufSz * sizeof(T));
        pipe->InitBuffer(aGatherTypeTmp, quantBufSz * sizeof(T)); // 先将int8转化为T类型再进行dequant
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {            // bf16场景下需要转换成fp32类型再做乘法
            pipe->InitBuffer(dqScaleFltTmp, quantBufSz * sizeof(float));
            pipe->InitBuffer(aGatherFltTmp, quantBufSz * sizeof(float));
        }
    }

    __aicore__ inline void MatmulAllReduceDequantPerchannelSplitM(const uint32_t quantBufSz, int64_t &addrBlkOff,
                                                                  uint32_t &tileCalCntM, uint32_t &tailCalCntM,
                                                                  uint32_t &aivLoopNum)
    {
        uint32_t vectorIndex = GetBlockIdx();                 // [0, 23]
        uint32_t singleVecM = dequantM_ / dequantAivCoreNum_; // 单核要计算的总行数（多次循环累计）
        uint32_t aivPlusOneIdx = dequantAivCoreNum_ + 1; // 要多算一轮的核的下标，如果不均分，使用后面 [aivPlusOneIdx,
                                                          // dequantAivCoreNum_ - 1] 核来完成多余一轮的计算
        if ((dequantM_ % dequantAivCoreNum_) != 0) {
            aivPlusOneIdx = dequantAivCoreNum_ - (dequantM_ % dequantAivCoreNum_);
        }

        if (singleVecM == 0) { // M小于核数，singleVecM为0，核计算行数更新及偏移计算
            uint32_t usedVecIdx = dequantAivCoreNum_ - aivPlusOneIdx;
            if (vectorIndex < usedVecIdx) {
                singleVecM += 1;
                addrBlkOff = static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleVecM) *
                                  static_cast<int64_t>(dequantN_);
            } else {
                // 不用计算的核直接返回
                return;
            }
        } else { // M大于核数，singleVecM>0，核计算行数更新及偏移计算
            if ((aivPlusOneIdx < dequantAivCoreNum_ + 1) && (vectorIndex >= aivPlusOneIdx)) {
                // 多算一行
                singleVecM += 1;
                addrBlkOff = (static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleVecM) -
                                   static_cast<int64_t>(aivPlusOneIdx)) *
                                  static_cast<int64_t>(dequantN_);
            } else {
                addrBlkOff = static_cast<int64_t>(vectorIndex) * static_cast<int64_t>(singleVecM) *
                                  static_cast<int64_t>(dequantN_);
            }
        }

        tileCalCntM = quantBufSz / dequantAlginN_; // 单次循环计算行数
        aivLoopNum = singleVecM / tileCalCntM;      // 循环次数
        if (singleVecM % (quantBufSz / dequantAlginN_) != 0) {
            aivLoopNum += 1;
            tailCalCntM = singleVecM % tileCalCntM;
        }
    }

    __aicore__ inline void MatmulAllReduceDequantPerchannelSplitMAndN(int32_t &quantBufSz, uint32_t &needAivCoreNum,
                                                                      uint32_t &tileBlockCnt, uint32_t &tailBlockCnt,
                                                                      uint32_t &aivLoopNum, uint32_t &blockNumPerRow)
    {
        quantBufSz = TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 /
                      (FP16_BUF_SPLIT_MN_CNT_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T))); // 1+1+0.5+1
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            quantBufSz = TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 / (BF16_BUF_SPLIT_MN_CNT_MATMUL_ALLREDUCE_INT8 *
                                                                static_cast<int32_t>(sizeof(T))); // 1+1+0.5+1+2+2+2
        }
        quantBufSz = quantBufSz / static_cast<int32_t>(BYTE512_MATMUL_ALLREDUCE_INT8) *
                      static_cast<int32_t>(BYTE512_MATMUL_ALLREDUCE_INT8);
        if (quantBufSz >= static_cast<int32_t>(dequantN_)) {
            tileBlockCnt = dequantN_;
            quantBufSz = static_cast<int32_t>(dequantN_);
        } else {
            blockNumPerRow = Ceil(dequantN_, static_cast<uint32_t>(quantBufSz));
            tileBlockCnt = static_cast<uint32_t>(quantBufSz);
            if ((dequantN_ % static_cast<uint32_t>(quantBufSz)) != 0) {
                tailBlockCnt = dequantN_ % static_cast<uint32_t>(quantBufSz);
            }
        }
        needAivCoreNum = blockNumPerRow * dequantM_;
        aivLoopNum = needAivCoreNum / dequantAivCoreNum_;
        if ((needAivCoreNum % dequantAivCoreNum_) != 0) {
            aivLoopNum += 1;
        }
    }

    __aicore__ inline void InitInner(uint32_t loopIdx, int64_t addrBlkOffSplitM, uint32_t blockCntM,
                                     int64_t addrBlkOffSplitMN, int64_t scaleAddrOffsetSplitMN,
                                     uint32_t blockCntSpiltMN)
    {
        uint64_t curBlockCnt = 0;
        int64_t curBlockAddrOffset = 0;
        uint64_t curScaleCnt = 0;
        int64_t curScaleAddrOffset = 0;

        if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            curBlockCnt = blockCntM * dequantN_;
            curBlockAddrOffset = addrBlkOffSplitM;
            curScaleCnt = dequantN_;
        } else {
            curBlockCnt = blockCntSpiltMN;
            curBlockAddrOffset = addrBlkOffSplitMN;
            curScaleCnt = blockCntSpiltMN;
            curScaleAddrOffset = scaleAddrOffsetSplitMN;
        }

        allgatherOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(this->allgatherOut) + curBlockAddrOffset,
                                       curBlockCnt); // 用于读入数据
        dequantedOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(this->dequantedOut) + curBlockAddrOffset,
                                       curBlockCnt); // 用于输出数据
        if (((loopIdx == 0) && (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8)) ||
            (this->splitMode == SPLIT_MN_MATMUL_ALLREDUCE_INT8)) {
            dequantScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(this->dequantScale) + curScaleAddrOffset,
                                           curScaleCnt);
        }
    }

    __aicore__ inline void Process(uint32_t loopIdx, uint32_t aivLoopNum, uint32_t curBlockCntM,
                                   uint32_t blockCntSpiltMN)
    {
        uint32_t calcCnt = blockCntSpiltMN;
        uint16_t copyInUbStride = 0;
        uint16_t copyOutUbStride = 0;
        uint16_t copyBlockCnt = 1;
        uint32_t copyBlockLen = calcCnt;
        if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // 搬多行
            calcCnt = dequantAlginN_ * curBlockCntM;
            copyOutUbStride = (dequantAlginN_ * sizeof(T) - Ceil(dequantN_ * sizeof(T), BYTE32_MATMUL_ALLREDUCE_INT8) *
                                                                BYTE32_MATMUL_ALLREDUCE_INT8) /
                              BYTE32_MATMUL_ALLREDUCE_INT8;
            copyInUbStride =
                (dequantAlginN_ * sizeof(int8_t) -
                 Ceil(dequantN_ * sizeof(int8_t), BYTE32_MATMUL_ALLREDUCE_INT8) * BYTE32_MATMUL_ALLREDUCE_INT8) /
                BYTE32_MATMUL_ALLREDUCE_INT8;
            copyBlockCnt = curBlockCntM;
            copyBlockLen = dequantN_;
        }
        DataCopyParams allgatherOutCopyParams = {copyBlockCnt, static_cast<uint16_t>(copyBlockLen * sizeof(int8_t)), 0,
                                                 copyInUbStride};
        DataCopyParams dequantedOutCopyParams = {copyBlockCnt, static_cast<uint16_t>(copyBlockLen * sizeof(T)),
                                                 copyOutUbStride, 0};
        DataCopyParams quantScaleCopyParams = {1, static_cast<uint16_t>(copyBlockLen * sizeof(T)), 0, 0};
        DataCopyPadParams padParams = {false, 0, 0, copyInUbStride};

        LocalTensor<int8_t> allgatherOutLocal = inQueueX2.AllocTensor<int8_t>();
        DataCopyPad(allgatherOutLocal, allgatherOutGm, allgatherOutCopyParams, padParams);
        inQueueX2.EnQue(allgatherOutLocal);
        if (((loopIdx == 0) && (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8)) ||
            (this->splitMode == SPLIT_MN_MATMUL_ALLREDUCE_INT8)) {
            dequantScaleLocal = inQueueXS.AllocTensor<T>();
            DataCopyPad(dequantScaleLocal, dequantScaleGm, quantScaleCopyParams, padParams);
            inQueueXS.EnQue(dequantScaleLocal);
            sLocal = inQueueXS.DeQue<T>();
            if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
                // mul不支持bf16类型直接相乘，将scale和allgatherOut都转换为fp32类型就行Mul计算
                dequantScaleFloatLocalTemp = dqScaleFltTmp.Get<float>();
                Cast(dequantScaleFloatLocalTemp, sLocal, RoundMode::CAST_NONE, copyBlockLen); // bf16->fp32
            }
        }
        LocalTensor<int8_t> xLocal = inQueueX2.DeQue<int8_t>();
        LocalTensor<T> zLocal = outQueueZ2.AllocTensor<T>();

        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            // 将allgatherOut从int8类型转换为fp16类型再转换为fp32类型
            LocalTensor<half> allgatherOutHalfLocalTemp = aGatherTypeTmp.Get<half>();    // int8->fp16
            LocalTensor<float> allgatherOutFloatLocalTemp = aGatherFltTmp.Get<float>(); // fp16->fp32
            Cast(allgatherOutHalfLocalTemp, xLocal, RoundMode::CAST_NONE, calcCnt);                 // int8->fp16
            PipeBarrier<PIPE_V>();
            Cast(allgatherOutFloatLocalTemp, allgatherOutHalfLocalTemp, RoundMode::CAST_NONE, calcCnt); // fp16->fp32
            PipeBarrier<PIPE_V>();
            if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // BF16 切M
                for (uint32_t idx = 0; idx < curBlockCntM; i++) {
                    Mul(allgatherOutFloatLocalTemp[i * dequantAlginN_], allgatherOutFloatLocalTemp[i * dequantAlginN_],
                        dequantScaleFloatLocalTemp, dequantN_); // fp32 类型下进行 Mul 运算
                }
                PipeBarrier<PIPE_V>();
            } else { // BF16 切MN
                Mul(allgatherOutFloatLocalTemp, allgatherOutFloatLocalTemp, dequantScaleFloatLocalTemp,
                    calcCnt); // fp32 类型下进行 Mul 运算
                PipeBarrier<PIPE_V>();
            }
            Cast(zLocal, allgatherOutFloatLocalTemp, RoundMode::CAST_RINT, calcCnt); // fp32->bf16
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<T> allgatherOutTtypeLocalTemp = aGatherTypeTmp.Get<T>();
            Cast(allgatherOutTtypeLocalTemp, xLocal, RoundMode::CAST_NONE, calcCnt); // int8->fp16
            PipeBarrier<PIPE_V>();
            if (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) { // FP16 切 M
                for (uint32_t idx = 0; idx < curBlockCntM; i++) {
                    Mul(zLocal[i * dequantAlginN_], allgatherOutTtypeLocalTemp[i * dequantAlginN_], sLocal,
                        dequantN_); // fp16 类型下进行 Mul 运算
                }
                PipeBarrier<PIPE_V>();
            } else {                                                      // FP16 切 MN
                Mul(zLocal, allgatherOutTtypeLocalTemp, sLocal, calcCnt); // fp16 类型下进行 Mul 运算
                PipeBarrier<PIPE_V>();
            }
        }

        outQueueZ2.EnQue<T>(zLocal);
        inQueueX2.FreeTensor(allgatherOutLocal);
        if ((loopIdx == (aivLoopNum - 1)) && (this->splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8)) {
            inQueueXS.FreeTensor(dequantScaleLocal);
        } else if (this->splitMode == SPLIT_MN_MATMUL_ALLREDUCE_INT8) {
            inQueueXS.FreeTensor(dequantScaleLocal);
        }

        LocalTensor<T> outLocal = outQueueZ2.DeQue<T>();
        PipeBarrier<PIPE_V>();
        DataCopyPad(dequantedOutGm, outLocal, dequantedOutCopyParams);
        outQueueZ2.FreeTensor(zLocal);
    }

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    TQue<QuePosition::VECIN, 1> inQueueXS;
    TQue<QuePosition::VECOUT, 1> outQueueZ2;
    TBuf<TPosition::VECCALC> aGatherTypeTmp;
    TBuf<TPosition::VECCALC> dqScaleFltTmp;
    TBuf<TPosition::VECCALC> allgatherOutHalfLocalTmp;  // int8先转换为fp16
    TBuf<TPosition::VECCALC> aGatherFltTmp; // fp16再转换为fp32
    TBuf<TPosition::VECCALC> dequantedOutFloatLocalTmp; // 临时存放fp32类型的量化结果
    GlobalTensor<int8_t> allgatherOutGm;
    GlobalTensor<T> dequantScaleGm;
    GlobalTensor<T> dequantedOutGm;
    LocalTensor<T> dequantScaleLocal;
    LocalTensor<T> sLocal;
    LocalTensor<float> dequantScaleFloatLocalTemp;
    uint32_t dequantAlginN_;     // 512B 对齐后的输入 N，512B 对齐可以命中 L2 cache 缓存
    uint32_t dequantN_;          // 输入 N
    uint32_t dequantM_;          // 输入 M
    uint32_t dequantAivCoreNum_; // vector 核数
    uint32_t splitMode;
    GM_ADDR allgatherOut;
    GM_ADDR dequantScale;
    GM_ADDR dequantedOut;
};

/*
 * 接口说明：&outerLoop: 总的外循环个数，结合outerLoopId判断是否为最后一个循环
 */
template <class T>
__aicore__ inline void MatmulAllReduceDequantPerchannelCommInt8(GM_ADDR allgatherOut, GM_ADDR dequantScale,
                                                                GM_ADDR dequantedOut, TPipe *tPipe, uint32_t N,
                                                                uint32_t M)
{
    uint32_t nowDequantAivCoreNum = GetBlockNum() * GetTaskRation();
    if ((g_coreType == AIC) || (GetBlockIdx() >= nowDequantAivCoreNum)) {
        return;
    }
    uint32_t tileBlockCnt = 0;
    uint32_t tailBlockCnt = 0;
    uint32_t dequantAivLoopNum = 0;
    int64_t addrBlkOff = 0;
    uint32_t tailCalCntM = 0;
    uint32_t tileCalCntM = 0;
    uint32_t needAivCoreNum = 0;
    uint32_t blockNumPerRow = 1;

    tPipe->Reset();
    int32_t nowDequantAlginN = Ceil(N * sizeof(int8_t), BYTE512_MATMUL_ALLREDUCE_INT8) * BYTE512_MATMUL_ALLREDUCE_INT8;
    int32_t nowDequantBufSz = (TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 -
                                static_cast<int32_t>(nowDequantAlginN) * static_cast<int32_t>(sizeof(T))) /
                               (FP16_BUF_CNT_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T))); // fp16
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        nowDequantBufSz = (TOTAL_UBSIZE_MATMUL_ALLREDUCE_INT8 -
                            static_cast<int32_t>(nowDequantAlginN) * static_cast<int32_t>(sizeof(T))) /
                           (BF16_BUF_CNT_MATMUL_ALLREDUCE_INT8 * static_cast<int32_t>(sizeof(T))); // bf16
    }
    nowDequantBufSz = (nowDequantBufSz / nowDequantAlginN) * nowDequantAlginN;
    MatmulAllReduceDequantPerchannel<T> op;
    op.dequantM_ = M;
    op.dequantN_ = N;
    op.dequantAlginN_ = static_cast<uint32_t>(nowDequantAlginN);
    op.dequantAivCoreNum_ = nowDequantAivCoreNum;
    op.allgatherOut = allgatherOut;
    op.dequantScale = dequantScale;
    op.dequantedOut = dequantedOut;
    if (nowDequantBufSz > M * nowDequantAlginN) {
        nowDequantBufSz = M * nowDequantAlginN;
    }
    if (nowDequantBufSz >= nowDequantAlginN) { // 分核切M， 不切N
        op.splitMode = SPLIT_M_MATMUL_ALLREDUCE_INT8;
        op.MatmulAllReduceDequantPerchannelSplitM(nowDequantBufSz, addrBlkOff, tileCalCntM, tailCalCntM,
                                                  dequantAivLoopNum);
    } else { // N超大，分核策略采取切MN
        op.splitMode = SPLIT_MN_MATMUL_ALLREDUCE_INT8;
        op.MatmulAllReduceDequantPerchannelSplitMAndN(nowDequantBufSz, needAivCoreNum, tileBlockCnt, tailBlockCnt,
                                                      dequantAivLoopNum, blockNumPerRow);
    }

    if (dequantAivLoopNum == 0) {
        return;
    }
    op.Init(tPipe, nowDequantBufSz);                                    // 按照最大去开
    for (uint32_t loopIdx = 0; loopIdx < dequantAivLoopNum; loopIdx++) { // 一轮外层循环对应着一次核间并行
        int64_t addrBlkOffSplitM = 0;
        uint32_t blockCntM = tileCalCntM;
        uint32_t blockCntSpiltMN = tileBlockCnt;
        int64_t addrBlkOffSplitMN = 0;
        int64_t scaleAddrOffsetSplitMN = 0;
        if (op.splitMode == SPLIT_M_MATMUL_ALLREDUCE_INT8) {
            if ((tailCalCntM != 0) && (loopIdx == dequantAivLoopNum - 1)) {
                blockCntM = tailCalCntM;
            }
            addrBlkOffSplitM = static_cast<int64_t>(addrBlkOff) +
                                    static_cast<int64_t>(loopIdx) * static_cast<int64_t>(tileCalCntM) *
                                        static_cast<int64_t>(N); // 取到当前核的偏移，再+当前块的地址偏移
        } else {
            uint32_t dequantGlobalBlockIdx = loopIdx * nowDequantAivCoreNum + GetBlockIdx();
            if (dequantGlobalBlockIdx > (needAivCoreNum - 1)) {
                return;
            }
            if ((tailBlockCnt != 0) && (dequantGlobalBlockIdx % blockNumPerRow == blockNumPerRow - 1)) {
                blockCntSpiltMN = tailBlockCnt;
            }
            addrBlkOffSplitMN =
                static_cast<int64_t>(dequantGlobalBlockIdx / blockNumPerRow) * static_cast<int64_t>(N) +
                static_cast<int64_t>(dequantGlobalBlockIdx % blockNumPerRow) *
                    static_cast<int64_t>(tileBlockCnt); // 计算当前块的地址偏移
            scaleAddrOffsetSplitMN =
                static_cast<int64_t>(dequantGlobalBlockIdx % blockNumPerRow) * static_cast<int64_t>(tileBlockCnt);
        }
        op.InitInner(loopIdx, addrBlkOffSplitM, blockCntM, addrBlkOffSplitMN, scaleAddrOffsetSplitMN,
                     blockCntSpiltMN);
        op.Process(loopIdx, dequantAivLoopNum, blockCntM, blockCntSpiltMN);
    }
}
} // namespace MatmulAllReduceDequantPerchannelImpl

#endif // MATMUL_ALL_REDUCE_DEQUANT_PERCHANNEL_H