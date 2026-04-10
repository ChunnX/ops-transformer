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
 * \file mhc_sinkhorn.h
 * \brief mhc_sinkhorn
 */

#ifndef ASCENDC_MHC_SINKHORN_H
#define ASCENDC_MHC_SINKHORN_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "mhc_sinkhorn_struct.h"
#include "mhc_sinkhorn_tiling_key.h"

namespace MhcSinkhorn {
using namespace AscendC;

constexpr int64_t MASK_BUFFER_SIZE = 64;
constexpr int64_t MAX_BUFFER_SIZE = 256;
constexpr uint32_t MASK_4 = 0b00000000000000001111111111111111;
constexpr uint32_t MASK_6 = 0b00000000111111111111111111111111;
constexpr uint32_t MASK_8 = 0b11111111111111111111111111111111;
constexpr int64_t MASK_NUM = 8;
constexpr int64_t INDEX_BLOCK_LEN = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;

template <typename T, bool OUT_FLAG>
class MhcSinkhornSimd {
public:
    __aicore__ inline MhcSinkhornSimd(TPipe &pipe, const MhcSinkhornTilingData &tilingData)
        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR h_res, GM_ADDR y, GM_ADDR norm_out, GM_ADDR sum_out, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcSoftmax(int32_t handleNum);
    template <bool ADDS_EPS>
    __aicore__ inline void CalcColNorm(int32_t handleNum);
    __aicore__ inline void CalcRowNorm(int32_t handleNum);
    __aicore__ inline void CopyIn(int32_t loopIdx, int32_t handleNum);
    __aicore__ inline void CopyOut(int32_t loopIdx, int32_t handleNum);
    __aicore__ inline void ScatterOutFromNorm(int32_t handleNum);

    GlobalTensor<float> hRes_;
    GlobalTensor<float> y_;
    GlobalTensor<float> normOut_;
    GlobalTensor<float> sumOut_;

    TPipe &pipe_;
    TQue<QuePosition::VECIN, 1> inputQue_;
    TQue<QuePosition::VECOUT, 1> outputQue_;
    TQue<QuePosition::VECOUT, 1> normOutQue_;
    TQue<QuePosition::VECOUT, 1> sumColQue_;
    TQue<QuePosition::VECOUT, 1> sumRowQue_;

    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> midBuffer_;
   
    const MhcSinkhornTilingData &tilingData_;
    int64_t blockIdx_;
    int64_t loop_;
    int64_t tailHandelSize_;
    int64_t nnSize_;
    int64_t nAlign_;
    int64_t dataBlockNum_;
};

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::Init(GM_ADDR h_res, GM_ADDR y, GM_ADDR norm_out, GM_ADDR sum_out,
                                             GM_ADDR tiling)
{
    blockIdx_ = GetBlockIdx();
    dataBlockNum_ = Ops::Base::GetUbBlockSize() / sizeof(float);
    nAlign_ = Ops::Base::CeilAlign(tilingData_.n, dataBlockNum_);
    nnSize_ = tilingData_.n * tilingData_.n;

    hRes_.SetGlobalBuffer((__gm__ float *)(h_res));
    y_.SetGlobalBuffer((__gm__ float *)(y));
    normOut_.SetGlobalBuffer((__gm__ float *)(norm_out), GetBlockIdx() * tilingData_.tNormCore * tilingData_.n * nAlign_);
    sumOut_.SetGlobalBuffer((__gm__ float *)(sum_out));

    pipe_.InitBuffer(maskBuffer_, MASK_BUFFER_SIZE);
    pipe_.InitBuffer(midBuffer_, MAX_BUFFER_SIZE);
    pipe_.InitBuffer(inputQue_, DOUBLE_BUFFER, tilingData_.tUbFactor * tilingData_.n * nAlign_ * sizeof(float));
    pipe_.InitBuffer(outputQue_, DOUBLE_BUFFER, tilingData_.tUbFactor * nnSize_ * sizeof(float));  // 32字节对齐
    pipe_.InitBuffer(normOutQue_, DOUBLE_BUFFER, tilingData_.tUbFactor * nAlign_ * sizeof(float));
    pipe_.InitBuffer(sumColQue_, DOUBLE_BUFFER, tilingData_.tUbFactor *  nAlign_ * sizeof(float));

    loop_ = tilingData_.tNormCoreLoop;
    tailHandelSize_ = tilingData_.tUbFactorTail;      // 尾循环处理的数量
    if (blockIdx_ == tilingData_.usedCoreNum - 1) { //  尾核
        loop_ = tilingData_.tTailCoreLoop;
        tailHandelSize_ = tilingData_.tUbTailTail;
    }
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CopyIn(int32_t loopIdx, int32_t handleNum)
{
    LocalTensor<float> inputLocal = inputQue_.AllocTensor<float>();
    int64_t inputOffset = ((blockIdx_ * tilingData_.tNormCore + loopIdx * tilingData_.tUbFactor) * nnSize_);
    uint32_t handleSize = handleNum * nnSize_ * sizeof(float);
    DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyExtParams{1, handleSize, 0, 0, 0};
    DataCopyPad(inputLocal, hRes_[inputOffset], dataCopyExtParams, dataCopyPadExtParams);

    inputQue_.EnQue(inputLocal);
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CopyOut(int32_t loopIdx, int32_t handleNum)
{
    LocalTensor<float> outputLocal = outputQue_.DeQue<float>();
    int64_t outOffset = ((blockIdx_ * tilingData_.tNormCore + loopIdx * tilingData_.tUbFactor) * nnSize_);
    uint32_t handleSize = handleNum * nnSize_ * sizeof(float);
    
    DataCopyExtParams dataCopyExtParams{1, handleSize, 0, 0, 0};
    DataCopyPad(y_[outOffset], outputLocal, dataCopyExtParams);
    outputQue_.FreeTensor<float>(outputLocal);
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CalcSoftmax(int32_t handleNum)
{
    LocalTensor<float> inputLocal = inputQue_.DeQue<float>();
    LocalTensor<float> midResLocal = midBuffer_.Get<float>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();

    __ubuf__ float *inputAddr = (__ubuf__ float *)inputLocal.GetPhyAddr();
    __ubuf__ float *midResAddr = (__ubuf__ float *)midResLocal.GetPhyAddr();
    __ubuf__ uint32_t *maskAddr = (__ubuf__ uint32_t *)maskLocal.GetPhyAddr();

    LocalTensor<float> normOutLocal = normOutQue_.AllocTensor<float>();
    LocalTensor<float> sumOutLocal = sumColQue_.AllocTensor<float>();
    __ubuf__ float *normOutAddr = (__ubuf__ float *)normOutLocal.GetPhyAddr();
    __ubuf__ float *sumOutAddr = (__ubuf__ float *)sumOutLocal.GetPhyAddr();

    uint16_t vflen = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t blockCnt = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
    uint32_t handleSize = handleNum * vflen;
    uint32_t handleLen = handleNum;
    uint16_t loopSize = Ops::Base::CeilDiv(static_cast<uint16_t>(handleNum), blockCnt);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> midReg;
        MicroAPI::RegTensor<float> sumoutReg;

        MicroAPI::RegTensor<int32_t> orderReg;
        MicroAPI::RegTensor<int32_t> tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpReg2;
        MicroAPI::RegTensor<int32_t> tmpReg3;
        MicroAPI::RegTensor<int32_t> duplicateReg1;
        MicroAPI::RegTensor<int32_t> duplicateReg2;
        MicroAPI::RegTensor<int32_t> rowIndexReg;

        MicroAPI::AddrReg normAddrReg;
        MicroAPI::AddrReg sumAddrReg;

        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg maskRegCopy;
        MicroAPI::MaskReg maskRegMid;

        MicroAPI::Duplicate(duplicateReg1, static_cast<int32_t>(INDEX_BLOCK_LEN));
        MicroAPI::Duplicate(duplicateReg2, static_cast<int32_t>(nnSize_));
        MicroAPI::Arange(orderReg, static_cast<int32_t>(0));
        MicroAPI::Div(tmpReg1, orderReg, duplicateReg1, maskReg);
        MicroAPI::Mul(tmpReg2, tmpReg1, duplicateReg1, maskReg);
        MicroAPI::Sub(tmpReg2, orderReg, tmpReg2, maskReg);
        MicroAPI::Mul(tmpReg3, tmpReg1, duplicateReg2, maskReg);
        MicroAPI::Add(rowIndexReg, tmpReg2, tmpReg3, maskReg);

        for (uint16_t i = 0; i < loopSize; i++) {
            /* 32B按mask处理 */
            maskReg = MicroAPI::UpdateMask<uint32_t>(handleSize);
            MicroAPI::LoadAlign(maskReg, maskAddr);

            /* 32B全部处理 */
            maskRegCopy = MicroAPI::UpdateMask<uint32_t>(handleSize);
            maskRegMid = MicroAPI::UpdateMask<uint32_t>(handleLen);

            MicroAPI::Duplicate(sumoutReg, static_cast<float>(0));
            auto inputOfset = inputAddr + (i * blockCnt * nnSize_);
            for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(tilingData_.n); nIdx++) {
                inputOfset += nIdx * tilingData_.n;
                MicroAPI::DataCopyGather(inputReg, inputOfset, (MicroAPI::RegTensor<uint32_t> &)(rowIndexReg), maskReg);
                MicroAPI::ReduceMaxWithDataBlock(midReg, inputReg, maskReg);
                MicroAPI::DataCopy(midResAddr, midReg, maskRegMid);
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(midReg, midResAddr);
                MicroAPI::Sub(inputReg, inputReg, midReg, maskReg);
                MicroAPI::Exp(inputReg, inputReg, maskReg);

                MicroAPI::ReduceSumWithDataBlock(midReg, inputReg, maskReg);
                MicroAPI::DataCopy(midResAddr, midReg, maskRegMid);
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(midReg, midResAddr);
                MicroAPI::Div(inputReg, inputReg, midReg, maskReg);
                /* 搬出normout[0] */
                normAddrReg = MicroAPI::CreateAddrReg<uint32_t>(i, vflen, nIdx, handleSize);
                MicroAPI::DataCopy(normOutAddr, inputReg, normAddrReg, maskRegCopy);
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                MicroAPI::Adds(inputReg, inputReg, tilingData_.eps, maskReg);
                MicroAPI::Add(sumoutReg, sumoutReg, inputReg, maskReg);
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            }
            /* 搬出sumout[1] */
            sumAddrReg = MicroAPI::CreateAddrReg<uint32_t>(i, vflen);
            // MicroAPI::Adds(sumoutReg, sumoutReg, eps, maskReg);
            MicroAPI::DataCopy(sumOutAddr, sumoutReg, sumAddrReg, maskRegCopy);
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }
    }

    inputQue_.FreeTensor<float>(inputLocal);
    normOutQue_.EnQue(normOutLocal);
    sumColQue_.EnQue(sumOutLocal);
}

template <typename T, bool OUT_FLAG>
template <bool ADDS_EPS>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CalcColNorm(int32_t handleNum)
{
    LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
    LocalTensor<float> sumOutLocal = sumColQue_.DeQue<float>();
    __ubuf__ float *normOutAddr = (__ubuf__ float *)normOutLocal.GetPhyAddr();
    __ubuf__ float *sumOutAddr = (__ubuf__ float *)sumOutLocal.GetPhyAddr();

    LocalTensor<float> midResLocal = midBuffer_.Get<float>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    __ubuf__ float *midResAddr = (__ubuf__ float *)midResLocal.GetPhyAddr();
    __ubuf__ uint32_t *maskAddr = (__ubuf__ uint32_t *)maskLocal.GetPhyAddr();

    uint16_t vflen = Ops::Base::GetVRegSize() / sizeof(float);
    uint32_t handleSize = handleNum * vflen;
    uint32_t handleLen = handleNum;
    uint16_t blockCnt = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
    uint16_t loopSize = Ops::Base::CeilDiv(static_cast<uint16_t>(handleNum), blockCnt);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> normReg;
        MicroAPI::RegTensor<float> sumInReg;
        MicroAPI::RegTensor<float> sumOutReg;
        MicroAPI::RegTensor<float> midReg;

        MicroAPI::AddrReg normAddrReg;
        MicroAPI::AddrReg sumAddrReg;
        MicroAPI::MaskReg maskRegCalc;
        MicroAPI::MaskReg maskRegCopy;
        MicroAPI::MaskReg maskRegMid;

        for (uint16_t i = 0; i < loopSize; i++) {
            maskRegCalc = MicroAPI::UpdateMask<uint32_t>(handleSize);
            maskRegCopy = MicroAPI::UpdateMask<uint32_t>(handleSize);
            maskRegMid = MicroAPI::UpdateMask<uint32_t>(handleLen);

            sumAddrReg = MicroAPI::CreateAddrReg<uint32_t>(i, vflen);
            MicroAPI::DataCopy(sumInReg, sumOutAddr, sumAddrReg);
            MicroAPI::Adds(sumOutReg, sumInReg, tilingData_.eps, maskRegCalc);
            if constexpr (OUT_FLAG) {
                sumAddrReg = MicroAPI::CreateAddrReg<uint32_t>(i, vflen);
                MicroAPI::DataCopy(sumOutAddr, sumOutReg, sumAddrReg, maskRegCopy);
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            }

            for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(tilingData_.n); nIdx++) {
                MicroAPI::LoadAlign(maskRegCalc, maskAddr);
                normAddrReg = MicroAPI::CreateAddrReg<uint32_t>(i, vflen, nIdx, handleSize);
                MicroAPI::DataCopy(normReg, normOutAddr, normAddrReg);
                if constexpr (ADDS_EPS) {
                    MicroAPI::Adds(normReg, normReg, tilingData_.eps, maskRegCalc); // 这里有区别
                }
                MicroAPI::Div(normReg, normReg, sumOutReg, maskRegCalc);
                MicroAPI::DataCopy(normOutAddr, normReg, normAddrReg, maskRegCopy);
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            }
        }
    }
    normOutQue_.EnQue(normOutLocal);
    sumColQue_.EnQue(sumOutLocal);
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::CalcRowNorm(int32_t handleNum)
{
    LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
    LocalTensor<float> sumRowLocal = sumColQue_.DeQue<float>();
    LocalTensor<float> sumColLocal = outputQue_.DeQue<float>(); // 复用outputQue

    __ubuf__ float *normOutAddr = (__ubuf__ float *)normOutLocal.GetPhyAddr();
    __ubuf__ float *sumRowOutAddr = (__ubuf__ float *)sumRowLocal.GetPhyAddr();
    __ubuf__ float *sumColOutAddr = (__ubuf__ float *)sumColLocal.GetPhyAddr();

    LocalTensor<float> midResLocal = midBuffer_.Get<float>();
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    __ubuf__ float *midResAddr = (__ubuf__ float *)midResLocal.GetPhyAddr();
    __ubuf__ uint32_t *maskAddr = (__ubuf__ uint32_t *)maskLocal.GetPhyAddr();
   
    uint16_t vflen = Ops::Base::GetVRegSize() / sizeof(float);
    uint32_t handleSize = handleNum * vflen;
    uint16_t blockCnt = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
    uint16_t loopSize = Ops::Base::CeilDiv(static_cast<uint16_t>(handleNum), blockCnt);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> normReg;
        MicroAPI::RegTensor<float> midReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<float> sumoutReg;

        MicroAPI::AddrReg sumAddrReg;
        MicroAPI::AddrReg normAddrReg;
        MicroAPI::MaskReg maskRegCalc;
        MicroAPI::MaskReg maskRegCopy;
        MicroAPI::MaskReg maskRegMid;

        for (uint16_t i = 0; i < loopSize; i++) {
            maskRegCalc = MicroAPI::UpdateMask<uint32_t>(handleSize);
            maskRegCopy = MicroAPI::UpdateMask<uint32_t>(handleSize);

            MicroAPI::Duplicate(sumoutReg, static_cast<float>(0));
            for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(tilingData_.n); nIdx++) {
                MicroAPI::LoadAlign(maskRegCalc, maskAddr);

                normAddrReg = MicroAPI::CreateAddrReg<uint32_t>(i, vflen, nIdx, handleSize);
                MicroAPI::DataCopy(normReg, normOutAddr, normAddrReg);
                MicroAPI::ReduceSumWithDataBlock(sumReg, normReg, maskRegCalc);
                MicroAPI::DataCopy(midResAddr, sumReg, maskRegCopy);
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                if constexpr (OUT_FLAG) {
                    /* 搬出sumout[2*i] */
                    auto sumOfset = sumRowOutAddr + i * 64;
                    MicroAPI::Adds(sumReg, sumReg, tilingData_.eps, maskRegCalc);
                    /* 非对齐，需要按t_对齐 */
                    MicroAPI::DataCopy(sumOfset, sumReg, maskRegCopy);
                    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                }
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumReg, midResAddr);
                MicroAPI::Adds(sumReg, sumReg, tilingData_.eps, maskRegCalc);
                MicroAPI::Div(normReg, normReg, sumReg, maskRegCalc);
                MicroAPI::DataCopy(normOutAddr, normReg, normAddrReg, maskRegCopy);
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
                MicroAPI::Add(sumoutReg, sumoutReg, normReg, maskRegCalc);
            }
            /* 搬出sumout[2*i+1] */
            sumAddrReg = MicroAPI::CreateAddrReg<uint32_t>(i, vflen);
            MicroAPI::Adds(sumoutReg, sumoutReg, tilingData_.eps, maskRegCalc);
            MicroAPI::DataCopy(sumColOutAddr, sumoutReg, sumAddrReg, maskRegCopy);
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }
    }
    normOutQue_.EnQue(normOutLocal);
    sumColQue_.EnQue(sumRowLocal);
    outputQue_.EnQue(sumColLocal);
}


template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::ScatterOutFromNorm(int32_t handleNum)
{
    LocalTensor<float> outputLocal = outputQue_.AllocTensor<float>();
    LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
    __ubuf__ float *normOutAddr = (__ubuf__ float *)normOutLocal.GetPhyAddr();
    __ubuf__ float *outputAddr = (__ubuf__ float *)outputLocal.GetPhyAddr();

    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    __ubuf__ uint32_t *maskAddr = (__ubuf__ uint32_t *)maskLocal.GetPhyAddr();

    uint16_t vflen = Ops::Base::GetVRegSize() / sizeof(float);
    uint32_t handleSize = handleNum * vflen;
    uint16_t blockCnt = Ops::Base::GetVRegSize() / Ops::Base::GetUbBlockSize();
    uint16_t loopSize = Ops::Base::CeilDiv(static_cast<uint16_t>(handleNum), blockCnt);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> normReg;
        MicroAPI::RegTensor<int32_t> orderReg;
        MicroAPI::RegTensor<int32_t> tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpReg2;
        MicroAPI::RegTensor<int32_t> tmpReg3;
        MicroAPI::RegTensor<int32_t> dupReg1;
        MicroAPI::RegTensor<int32_t> dupReg2;
        MicroAPI::RegTensor<int32_t> indexReg;

        MicroAPI::AddrReg normAddrReg;
        MicroAPI::MaskReg maskReg;

        MicroAPI::Duplicate(dupReg1, INDEX_BLOCK_LEN);
        MicroAPI::Duplicate(dupReg2, static_cast<int32_t>(nnSize_));
        MicroAPI::Arange(orderReg, static_cast<int32_t>(0));
        MicroAPI::Div(tmpReg1, orderReg, dupReg1, maskReg);
        MicroAPI::Mul(tmpReg2, tmpReg1, dupReg1, maskReg);
        MicroAPI::Sub(tmpReg2, orderReg, tmpReg2, maskReg);
        MicroAPI::Mul(tmpReg3, tmpReg1, dupReg2, maskReg);
        MicroAPI::Add(indexReg, tmpReg2, tmpReg3, maskReg);
        for (uint16_t i = 0; i < loopSize; i++) {
            maskReg = MicroAPI::UpdateMask<uint32_t>(handleSize);
            for (uint16_t nIdx = 0; nIdx < static_cast<uint16_t>(tilingData_.n); nIdx++) {
                MicroAPI::LoadAlign(maskReg, maskAddr);
                normAddrReg = MicroAPI::CreateAddrReg<uint32_t>(i, vflen, nIdx, handleSize);
                MicroAPI::DataCopy(normReg, normOutAddr, normAddrReg);
                MicroAPI::DataCopyScatter(outputAddr, normReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            }
        }
    }
    outputQue_.EnQue(outputLocal);
    normOutQue_.EnQue(normOutLocal);
}

template <typename T, bool OUT_FLAG>
__aicore__ inline void MhcSinkhornSimd<T, OUT_FLAG>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    uint32_t mask = 0;
    if (tilingData_.n == N_VALID_4) {
        mask = MASK_4;
    } else if (tilingData_.n == N_VALID_6) {
        mask = MASK_6;
    } else if (tilingData_.n == N_VALID_8) {
        mask = MASK_8;
    }
    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    Duplicate(maskLocal, mask, MASK_NUM);

    for (int64_t i = 0; i < loop_; i++) {
        auto handleNum = (i == loop_ - 1) ? tailHandelSize_ : tilingData_.tUbFactor;
        CopyIn(i, handleNum);
        CalcSoftmax(handleNum);
        
        if constexpr (OUT_FLAG) {
            int64_t normOutOfset = i * tilingData_.n * tilingData_.tUbFactor * nAlign_;
            uint32_t normOutLen = (tilingData_.n * handleNum * nAlign_) * sizeof(float);
            DataCopyExtParams normCopyExtParams{1, normOutLen, 0, 0, 0};
            LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
            DataCopyPad(normOut_[normOutOfset], normOutLocal, normCopyExtParams);
            normOutQue_.EnQue(normOutLocal);
        }

        CalcColNorm<true>(handleNum);
        if constexpr (OUT_FLAG) {
            int64_t normOutOfset = i * tilingData_.tUbFactor * nnSize_;
            uint32_t normOutLen = (tilingData_.n * handleNum * nAlign_) * sizeof(float);
            DataCopyExtParams normCopyExtParams{1, normOutLen, 0, 0, 0};
            LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
            DataCopyPad(normOut_[normOutOfset], normOutLocal, normCopyExtParams);
            normOutQue_.EnQue(normOutLocal);

            int64_t sumOutOfset = i * tilingData_.tUbFactor * nAlign_;
            uint32_t sumOutLen = (handleNum * nAlign_) * sizeof(float);
            DataCopyExtParams sumCopyExtParams{1, sumOutLen, 0, 0, 0};
            LocalTensor<float> sumOutLocal = sumColQue_.DeQue<float>();
            DataCopyPad(sumOut_[sumOutOfset], sumOutLocal, sumCopyExtParams);
            sumColQue_.EnQue(sumOutLocal);
        }

        for (int64_t iter = 1; iter < tilingData_.num_iters; iter++) {
            CalcRowNorm(handleNum);
            if constexpr (OUT_FLAG) {
                int64_t normOutOfset = i * tilingData_.tUbFactor * nnSize_;
                uint32_t normOutLen = (tilingData_.n * handleNum * nAlign_) * sizeof(float);
                DataCopyExtParams normCopyExtParams{1, normOutLen, 0, 0, 0};
                LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
                DataCopyPad(normOut_[normOutOfset], normOutLocal, normCopyExtParams);
                normOutQue_.EnQue(normOutLocal);

                int64_t sumOutOfset = i * tilingData_.tUbFactor * nAlign_;
                uint32_t sumOutLen = (handleNum * nAlign_) * sizeof(float);
                DataCopyExtParams sumCopyExtParams{1, sumOutLen, 0, 0, 0};
                LocalTensor<float> sumOutLocal = sumColQue_.DeQue<float>();
                DataCopyPad(sumOut_[normOutOfset], normOutLocal, sumCopyExtParams);
                sumColQue_.EnQue(sumOutLocal);

            }
            
            CalcColNorm<false>(handleNum);
            if constexpr (OUT_FLAG) {
                int64_t normOutOfset = i * tilingData_.tUbFactor * nnSize_;
                uint32_t normOutLen = (tilingData_.n * handleNum * nAlign_) * sizeof(float);
                DataCopyExtParams normCopyExtParams{1, normOutLen, 0, 0, 0};
                LocalTensor<float> normOutLocal = normOutQue_.DeQue<float>();
                DataCopyPad(normOut_[normOutOfset], normOutLocal, normCopyExtParams);
                normOutQue_.EnQue(normOutLocal);

                int64_t sumOutOfset = i * tilingData_.tUbFactor * nAlign_;
                uint32_t sumOutLen = (handleNum * nAlign_) * sizeof(float);
                DataCopyExtParams sumCopyExtParams{1, sumOutLen, 0, 0, 0};
                LocalTensor<float> sumOutLocal = sumColQue_.DeQue<float>();
                DataCopyPad(sumOut_[normOutOfset], normOutLocal, sumCopyExtParams);
                sumColQue_.EnQue(sumOutLocal);
            }
        }
        ScatterOutFromNorm(handleNum);
        CopyOut(i, handleNum);
    }
}
} // namespace MhcSinkhorn

#endif