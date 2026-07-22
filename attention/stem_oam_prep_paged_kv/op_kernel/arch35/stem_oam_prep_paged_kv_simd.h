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
 * \file stem_oam_prep_paged_kv_simd.h
 * \brief
 */

#ifndef STEM_OAM_PREP_PAGED_KV_SIMD_H
#define STEM_OAM_PREP_PAGED_KV_SIMD_H

#include <algorithm>
#include <cstdint>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "stem_oam_prep_paged_kv_tiling_data.h"
#include "stem_oam_prep_paged_kv_base.h"

using namespace AscendC;

constexpr static int64_t KVMAXLEN = 256 * 1024;
constexpr static int64_t STEMBLOCKMINSIZE = 32;
constexpr static int64_t KFLATDIMS = 0;
constexpr static int64_t KFLATBATCHSIZE = 1;
constexpr static int64_t KFLATHKV = 2;
constexpr static int64_t KFLATMAXKB = 3;
constexpr static int64_t KFLATDIM = 4;
constexpr static int64_t VBIASDIMS = 9;
constexpr static int64_t VBIASBATCHSIZE = 10;
constexpr static int64_t VBIASHKV = 11;
constexpr static int64_t VBIASDIM = 12;
constexpr static int64_t OFFSET1 = 8;
constexpr static int64_t OFFSET2 = 16;

class StemOamPrepPagedKvSimd {
public:
    __aicore__ inline StemOamPrepPagedKvSimd() = default;
    __aicore__ inline void Process(void);
    __aicore__ inline void Init(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices, GM_ADDR kvSeqLens,
                                GM_ADDR kScaleCache, GM_ADDR vScale, GM_ADDR kFlat, GM_ADDR vBias, GM_ADDR shapeOut,
                                GM_ADDR workspace, const StemOamPrepPagedKvTilingData *tilingData, TPipe *pipe);

private:
    __aicore__ inline void CopyInKVCache(int32_t batchIdx, int32_t headIDx, int32_t stemBlockIdx, int64_t actualRows);
    __aicore__ inline void CopyInVFlatOut(int32_t batchIdx, int32_t headIdx, int64_t numStemBlocks);
    __aicore__ inline void CopyInKvSeqLens(void);

    __aicore__ inline void ComputeKFlat(void);
    __aicore__ inline void ComputeVFlat(int64_t headIdx, int64_t stemBlockIdx, int32_t kvLen);
    __aicore__ inline void ComputeVMean(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                        __local_mem__ float *vFLatOutAddr);
    template <bool KDownLenFlag>
    __aicore__ inline void ComputeVStd(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                       __local_mem__ float *logValsAddr, float vMean);
    __aicore__ inline void ComputeVBias(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                        __local_mem__ float *logValsAddr, float vMean, float vSTd);
    __aicore__ inline void ComputeVBiasOut(int32_t batchIdx, int32_t headIdx, int64_t kDownLen, int64_t numStemBlocks);
    __aicore__ inline void ProcessVBias(void);
    __aicore__ inline void ProcessKVFlat(void);
    __aicore__ inline void ComputTotalStemBlockIndex(void);

    __aicore__ inline void CopyOutKFlat(int32_t batchIdx, int32_t headIDx, int32_t stemBlocksIdx);
    __aicore__ inline void CopyOutVFlat(int32_t batchIdx, int32_t headIDx, int32_t stemBlocksIdx);
    __aicore__ inline void CopyOutVBias(int32_t batchIdx, int32_t headIdx, int64_t numStemBlocks);

    __aicore__ inline void InitKFlatAndVBias(void);
    template <typename OUTPUT_TYPE>
    __aicore__ inline void InitOutPutGm(const GlobalTensor<OUTPUT_TYPE> &gmTensor,
                                                                const LocalTensor<OUTPUT_TYPE> &ublTensor,
                                                                int64_t totalLen, int64_t lineLen);

private:
    TPipe *pipe_;
    StemOamPrepPagedKvTilingData tilingData_;

    GlobalTensor<fp8_e4m3fn_t> kCacheGm_;
    GlobalTensor<fp8_e4m3fn_t> vCacheGm_;
    GlobalTensor<int32_t> kvIndicesGm_;
    GlobalTensor<int32_t> kvSeqLensGm_;
    GlobalTensor<float> kScaleCacheGm_;
    GlobalTensor<float> vScaleGm_;
    GlobalTensor<uint64_t> outShapeGm_;

    GlobalTensor<bfloat16_t> kFlatGm_;
    GlobalTensor<float> vBiasGm_;

    GlobalTensor<float> vFlatOutWs_;

    TQue<QuePosition::VECIN, 1> kCacheQue_;
    TQue<QuePosition::VECIN, 1> vCacheQue_;
    TQue<QuePosition::VECIN, 1> kvSeqLensQue_;
    TQue<QuePosition::VECIN, 1> vFLatOutQue_;

    TQue<QuePosition::VECOUT, 1> groupSumQue_;
    TQue<QuePosition::VECOUT, 1> vNormDownQue_;
    TQue<QuePosition::VECOUT, 1> vBiasOutQue_;

    TBuf<TPosition::VECCALC> kScalarBuf_;
    TBuf<TPosition::VECCALC> totalStemBlockIndexBuf_;

    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 0;
    int64_t kvLayout_ = 0;
    int64_t kvBlockSize_ = 0;
    int64_t hKv_ = 0;
    int64_t maxKvBlocks_ = 0;
    int64_t dimQk_ = 0;
    int64_t dimV_ = 0;
    int64_t maxKb_ = 0;
    int64_t kflatDim_ = 0;
    int64_t batchSize_ = 0;
    int64_t stemBlockSize_ = 0;
    int64_t stemStride_ = 0;
    int64_t totalBh_ = 0;
    float lambdaMag_ = 0;
    int64_t r_ = 0;
    int64_t kCacheStride_[4] = {0};
    int64_t vCacheStride_[4] = {0};
    int64_t kScaleCacheStride_[4] = {0};
    int64_t vCacheOffset_ = 0;
    int64_t totalStemBlockNum_ = 0;
    int64_t curCoreProcessStemBlock_ = 0;
    int64_t tailCoreProcessStemBlock_ = 0;
    int64_t kvUsedCoreNum_ = 0;

    constexpr static int64_t V_REG_SIZE = Ops::Base::GetVRegSize();
};

__aicore__ inline void StemOamPrepPagedKvSimd::Init(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices,
                                                    GM_ADDR kvSeqLens, GM_ADDR kScaleCache, GM_ADDR vScale,
                                                    GM_ADDR kFlat, GM_ADDR vBias, GM_ADDR shapeOut, GM_ADDR workspace,
                                                    const StemOamPrepPagedKvTilingData *tilingData, TPipe *pipe)
{
    kCacheGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)kCache);
    vCacheGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)vCache);
    kvIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)kvIndices);
    kvSeqLensGm_.SetGlobalBuffer((__gm__ int32_t *)kvSeqLens);
    kScaleCacheGm_.SetGlobalBuffer((__gm__ float *)kScaleCache);
    vScaleGm_.SetGlobalBuffer((__gm__ float *)vScale);
    kFlatGm_.SetGlobalBuffer((__gm__ bfloat16_t *)kFlat);
    vBiasGm_.SetGlobalBuffer((__gm__ float *)vBias);
    outShapeGm_.SetGlobalBuffer((__gm__ uint64_t *)shapeOut);

    vFlatOutWs_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    kvLayout_ = tilingData->kvLayout;
    kvBlockSize_ = tilingData->kvBlockSize;
    hKv_ = tilingData->numKvHeads;
    maxKvBlocks_ = tilingData->maxKvBlocks;
    dimQk_ = tilingData->dimQk;
    dimV_ = tilingData->dimV;
    kflatDim_ = tilingData->kflatDim;
    batchSize_ = tilingData->batchSize;
    stemBlockSize_ = tilingData->stemBlockSize;
    stemStride_ = tilingData->stemStride;
    lambdaMag_ = tilingData->lambdaMag;
    maxKb_ = tilingData->maxKb;
    r_ = tilingData->rVal;
    for (uint16_t i = 0; i < 4; i++) {
        kCacheStride_[i] = tilingData->kCacheStride[i];
        vCacheStride_[i] = tilingData->vCacheStride[i];
        kScaleCacheStride_[i] = tilingData->kScaleCacheStride[i];
    }
    totalBh_ = batchSize_ * hKv_;
    vCacheOffset_ = stemBlockSize_ * dimQk_;
    int64_t maxStemlockNum = KVMAXLEN / STEMBLOCKMINSIZE * 8 * hKv_ / blockNum_;
    // inti UB
    pipe_->InitBuffer(kCacheQue_, 1, stemBlockSize_ * dimQk_ * sizeof(fp8_e4m3fn_t)); // 32K
    pipe_->InitBuffer(vCacheQue_, 1, stemBlockSize_ * dimQk_ * sizeof(fp8_e4m3fn_t)); // 32K
    pipe_->InitBuffer(kvSeqLensQue_, 1, batchSize_ * sizeof(int32_t));                //

    pipe_->InitBuffer(groupSumQue_, 1, stemStride_ * dimQk_ * sizeof(bfloat16_t)); // 16K
    pipe_->InitBuffer(vNormDownQue_, 1, stemBlockSize_ * sizeof(float));           // 1K

    pipe_->InitBuffer(kScalarBuf_, stemBlockSize_ * sizeof(float));             // 1K
    pipe_->InitBuffer(totalStemBlockIndexBuf_, maxStemlockNum * sizeof(float)); // 24K

    pipe_->InitBuffer(vFLatOutQue_, 1, maxKb_ * r_ * sizeof(float)); // 64k
    pipe_->InitBuffer(vBiasOutQue_, 1, maxKb_ * sizeof(float));      // 32K
    if (blockIdx_ == 0) {
        InitOutput<bfloat16_t>(kFlatGm_, batchSize_ * hKv_ * maxKb_ * kflatDim_, bfloat16_t(0));
        InitOutput<float>(vBiasGm_, batchSize_ * hKv_ * maxKb_, float(0));
    }
    InitKFlatAndVBias();
    SyncAll();
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputTotalStemBlockIndex(void)
{
    CopyInKvSeqLens();
    LocalTensor<int32_t> kvSeqLensLocal = kvSeqLensQue_.DeQue<int32_t>();
    LocalTensor<int32_t> totalStemBlockIndexLocal = totalStemBlockIndexBuf_.Get<int32_t>();

    __local_mem__ int32_t *kvSeqLenAddr = (__ubuf__ int32_t *)kvSeqLensLocal.GetPhyAddr();
    __local_mem__ int32_t *totalStemBlockIndexAddr = (__ubuf__ int32_t *)totalStemBlockIndexLocal.GetPhyAddr();
    __local_mem__ int32_t *kvStemBlockAddr = kvSeqLenAddr;
    uint32_t vfLen = V_REG_SIZE / sizeof(int32_t);
    uint16_t loopCnt = (batchSize_ + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> kvSeqLensReg;
        AscendC::MicroAPI::RegTensor<int32_t> divNumReg;
        AscendC::MicroAPI::RegTensor<int32_t> kvstemBlockReg;
        AscendC::MicroAPI::RegTensor<int32_t> kvstemBlockAllReg;
        AscendC::MicroAPI::RegTensor<int32_t> stemBlockNumReg;
        AscendC::MicroAPI::RegTensor<int32_t> stemBlockSumNumReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        uint32_t maskLen = static_cast<uint32_t>(batchSize_);
        auto stemBlockSumNumAddr = totalStemBlockIndexAddr;
        AscendC::MicroAPI::MaskReg allMaskReg =
            AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg oneMaskReg =
            AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate(divNumReg, int32_t(stemBlockSize_), allMaskReg);
        MicroAPI::Duplicate(stemBlockSumNumReg, int32_t(0), oneMaskReg);
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<int32_t>(maskLen);
            AscendC::MicroAPI::AddrReg kvSeqLensAddrOfst = AscendC::MicroAPI::CreateAddrReg<int32_t>(i, vfLen);
            AscendC::MicroAPI::LoadAlign(kvSeqLensReg, kvSeqLenAddr, kvSeqLensAddrOfst);
            AscendC::MicroAPI::Adds(kvstemBlockReg, kvSeqLensReg, int32_t(stemBlockSize_ - 1), valueMaskReg);
            AscendC::MicroAPI::Div(kvstemBlockReg, kvstemBlockReg, divNumReg, valueMaskReg);
            AscendC::MicroAPI::Muls(kvstemBlockAllReg, kvstemBlockReg, int32_t(hKv_), valueMaskReg);
            AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, int32_t, int32_t>(stemBlockNumReg, kvstemBlockAllReg,
                                                                                   valueMaskReg);
            AscendC::MicroAPI::Add(stemBlockSumNumReg, stemBlockSumNumReg, stemBlockNumReg, oneMaskReg);
            AscendC::MicroAPI::StoreAlign(kvStemBlockAddr + i * vfLen, kvstemBlockReg, valueMaskReg);
        }
        AscendC::MicroAPI::Store(stemBlockSumNumAddr, stemBlockSumNumReg, 1);
    }
    EventMsg<HardEvent::V_S>();
    int32_t batchIdx = 0;
    int32_t hKvIdx = 0;
    int32_t stemBlockIdx = 0;
    int32_t baseStemBlockNum = 0;
    totalStemBlockNum_ = totalStemBlockIndexLocal.GetValue(0);
    int32_t blockFlag = 0;
    int32_t setN = 0;
    curCoreProcessStemBlock_ = Ops::Base::CeilDiv(totalStemBlockNum_, blockNum_);
    kvUsedCoreNum_ = Ops::Base::CeilDiv(totalStemBlockNum_, curCoreProcessStemBlock_);
    tailCoreProcessStemBlock_ = totalStemBlockNum_ - (kvUsedCoreNum_ - 1) * curCoreProcessStemBlock_;
    if (blockIdx_ < kvUsedCoreNum_) {
        auto startStemBlock = blockIdx_ * curCoreProcessStemBlock_;
        auto endStemBlock = (blockIdx_ * curCoreProcessStemBlock_ >= totalStemBlockNum_) ?
                                totalStemBlockNum_ - 1 :
                                (startStemBlock + curCoreProcessStemBlock_);
        for (int32_t i = 0; i < totalStemBlockNum_; i++) {
            while (batchIdx < batchSize_ && kvSeqLensLocal.GetValue(batchIdx) == 0) {
                batchIdx++;
                hKvIdx = 0;
            }
            stemBlockIdx = i - baseStemBlockNum;
            if (kvSeqLensLocal.GetValue(batchIdx) == 0) {
                batchIdx++;
            } else {
                if (stemBlockIdx >= kvSeqLensLocal.GetValue(batchIdx)) {
                    baseStemBlockNum += kvSeqLensLocal.GetValue(batchIdx);
                    hKvIdx++;
                    stemBlockIdx = 0;
                }
                if (hKvIdx >= hKv_) {
                    batchIdx++;
                    hKvIdx = 0;
                    while (batchIdx < batchSize_ && kvSeqLensLocal.GetValue(batchIdx) == 0) {
                        batchIdx++;
                        hKvIdx = 0;
                    }
                }
            }
            if (i >= startStemBlock && i < endStemBlock) {
                totalStemBlockIndexLocal.SetValue(setN * 3, batchIdx);
                totalStemBlockIndexLocal.SetValue(setN * 3 + 1, hKvIdx);
                totalStemBlockIndexLocal.SetValue(setN * 3 + 2, stemBlockIdx);
                setN++;
            }
            if (i == endStemBlock) {
                break;
            }
        }
    }
    kvSeqLensQue_.FreeTensor(kvSeqLensLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::InitKFlatAndVBias(void)
{
    LocalTensor<bfloat16_t> kFlatInitLocal = groupSumQue_.AllocTensor<bfloat16_t>();
    LocalTensor<float> vBiasInitLocal = kFlatInitLocal.ReinterpretCast<float>();
    int64_t bufSize = stemStride_ * dimQk_ * sizeof(bfloat16_t);
    int64_t kFlatSize = batchSize_ * hKv_ * maxKb_ * kflatDim_;
    int64_t vBiasSize = batchSize_ * hKv_ * maxKb_;
    int64_t curCoreKFlatInitSize = Ops::Base::CeilDiv(kFlatSize, blockNum_);
    curCoreKFlatInitSize =
        ((curCoreKFlatInitSize * sizeof(bfloat16_t)) >= bufSize) ? curCoreKFlatInitSize : bufSize / sizeof(bfloat16_t);
    int64_t curCoreVBiasInitSize = Ops::Base::CeilDiv(vBiasSize, blockNum_);
    curCoreVBiasInitSize =
        ((curCoreVBiasInitSize * sizeof(float)) >= bufSize) ? curCoreVBiasInitSize : bufSize / sizeof(float);
    int64_t kFlatUsedCoreNum = Ops::Base::CeilDiv(kFlatSize, curCoreKFlatInitSize);
    int64_t vBiasUsedCoreNum = Ops::Base::CeilDiv(vBiasSize, curCoreVBiasInitSize);
    int64_t tailCoreKFlatInitSize = kFlatSize - (kFlatUsedCoreNum - 1) * curCoreKFlatInitSize;
    int64_t tailCoreVBiasInitSize = vBiasSize - (vBiasUsedCoreNum - 1) * curCoreVBiasInitSize;

    Duplicate(kFlatInitLocal, bfloat16_t(0), stemStride_ * dimQk_);

    if (blockIdx_ < kFlatUsedCoreNum) {
        auto kFlatGmOffset = blockIdx_ * curCoreKFlatInitSize;
        auto kFlatInitLen = (blockIdx_ == kFlatUsedCoreNum - 1) ? tailCoreKFlatInitSize : curCoreKFlatInitSize;
        InitOutPutGm<bfloat16_t>(kFlatGm_[kFlatGmOffset], kFlatInitLocal, kFlatInitLen, bufSize / sizeof(bfloat16_t));
    }

    if (blockIdx_ < vBiasUsedCoreNum) {
        auto vBiasGmOffset = blockIdx_ * curCoreVBiasInitSize;
        auto vBiasInitLen = (blockIdx_ == vBiasUsedCoreNum - 1) ? tailCoreVBiasInitSize : curCoreVBiasInitSize;
        InitOutPutGm<float>(vBiasGm_[vBiasGmOffset], vBiasInitLocal, vBiasInitLen, bufSize / sizeof(float));
    }

    groupSumQue_.FreeTensor(kFlatInitLocal);
}

template <typename OUTPUT_TYPE>
__aicore__ inline void StemOamPrepPagedKvSimd::InitOutPutGm(const GlobalTensor<OUTPUT_TYPE> &gmTensor,
                                                            const LocalTensor<OUTPUT_TYPE> &ublTensor, int64_t totalLen,
                                                            int64_t lineLen)
{
    int64_t loopNum = Ops::Base::CeilDiv(totalLen, lineLen);
    int64_t tailDataLen = totalLen - (loopNum - 1) * lineLen;
    for (int64_t i = 0; i < loopNum; i++) {
        auto gmOffset = i * lineLen;
        auto copyDataLen = (i == (loopNum - 1)) ? tailDataLen : lineLen;
        CopyOut(gmTensor[gmOffset], ublTensor, 1, copyDataLen, 0, 0);
    }
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyInKvSeqLens(void)
{
    LocalTensor<int32_t> kvSeqLensLocal = kvSeqLensQue_.AllocTensor<int32_t>();

    CopyIn(kvSeqLensLocal, kvSeqLensGm_, 1, batchSize_, 0, 0);

    kvSeqLensQue_.EnQue(kvSeqLensLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyInVFlatOut(int32_t batchIdx, int32_t headIdx, int64_t numStemBlocks)
{
    LocalTensor<float> vFLatOutLocal = vFLatOutQue_.AllocTensor<float>();
    auto vFlatOutWsOffset = batchIdx * hKv_ * maxKb_ * r_ + headIdx * maxKb_ * r_;

    CopyIn(vFLatOutLocal, vFlatOutWs_[vFlatOutWsOffset], 1, numStemBlocks * r_, 0, 0);

    vFLatOutQue_.EnQue(vFLatOutLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyInKVCache(int32_t batchIdx, int32_t headIdx, int32_t stemBlockIdx,
                                                             int64_t actualRows)
{
    LocalTensor<int8_t> kCacheLocal = kCacheQue_.AllocTensor<int8_t>();
    LocalTensor<int8_t> vCacheLocal = vCacheQue_.AllocTensor<int8_t>();
    LocalTensor<float> kScalarLocal = kScalarBuf_.Get<float>();
    GlobalTensor<int8_t> kCacheGm = kCacheGm_.ReinterpretCast<int8_t>();
    GlobalTensor<int8_t> vCacheGm = vCacheGm_.ReinterpretCast<int8_t>();
    EventMsg<HardEvent::V_S>();
    for (int64_t i = 0; i < stemBlockSize_; i++) {
        int64_t kvIndicesIdx = (stemBlockIdx * stemBlockSize_ + i) / kvBlockSize_;
        int64_t kvBlockSizeIdx = (stemBlockIdx * stemBlockSize_ + i) % kvBlockSize_;
        int64_t logicIdx = kvIndicesGm_.GetValue(batchIdx * maxKvBlocks_ + kvIndicesIdx);
        int64_t kCacheOffset =
            logicIdx * kCacheStride_[0] + headIdx * kCacheStride_[1] + kvBlockSizeIdx * kCacheStride_[2];
        int64_t vCacheOffset =
            logicIdx * vCacheStride_[0] + headIdx * vCacheStride_[1] + kvBlockSizeIdx * vCacheStride_[2];
        int64_t kScaleCacheOffset =
            logicIdx * kScaleCacheStride_[0] + headIdx * kScaleCacheStride_[1] + kvBlockSizeIdx * kScaleCacheStride_[2];
        if ((stemBlockIdx * stemBlockSize_ + i) < actualRows) {
            CopyIn(kCacheLocal[i * dimQk_], kCacheGm[kCacheOffset], 1, dimQk_, 0, 0); // R * StemStride * dim_qk
            CopyIn(vCacheLocal[i * dimV_], vCacheGm[vCacheOffset], 1, dimV_, 0, 0);
            kScalarLocal.SetValue(i, kScaleCacheGm_.GetValue(kScaleCacheOffset));
        } else {
            Duplicate(kCacheLocal[i * dimQk_], int8_t(0), dimQk_);
            Duplicate(vCacheLocal[i * dimV_], int8_t(0), dimV_);
            kScalarLocal.SetValue(i, 0);
        }
    }
    EventMsg<HardEvent::S_V>();
    kCacheQue_.EnQue(kCacheLocal);
    vCacheQue_.EnQue(vCacheLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyOutKFlat(int32_t batchIdx, int32_t headIdx, int32_t stemBlocksIdx)
{
    LocalTensor<bfloat16_t> groupSumLocal = groupSumQue_.DeQue<bfloat16_t>();
    int64_t kFlatGmOffset =
        batchIdx * hKv_ * maxKb_ * kflatDim_ + headIdx * maxKb_ * kflatDim_ + stemBlocksIdx * kflatDim_;
    CopyOut(kFlatGm_[kFlatGmOffset], groupSumLocal, 1, stemStride_ * dimQk_, 0, 0);
    groupSumQue_.FreeTensor(groupSumLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyOutVFlat(int32_t batchIdx, int32_t headIdx, int32_t stemBlocksIdx)
{
    LocalTensor<float> vNormDownLocal = vNormDownQue_.DeQue<float>();
    int64_t vFlatOutOffset = batchIdx * hKv_ * maxKb_ * r_ + headIdx * maxKb_ * r_ + stemBlocksIdx * r_;

    CopyOut(vFlatOutWs_[vFlatOutOffset], vNormDownLocal, 1, r_, 0, 0);

    vNormDownQue_.FreeTensor(vNormDownLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::CopyOutVBias(int32_t batchIdx, int32_t headIdx, int64_t numStemBlocks)
{
    LocalTensor<float> vBiasOutLocal = vBiasOutQue_.DeQue<float>();

    auto vBiasOffset = batchIdx * hKv_ * maxKb_ + headIdx * maxKb_;
    CopyOut(vBiasGm_[vBiasOffset], vBiasOutLocal, 1, numStemBlocks, 0, 0);

    vBiasOutQue_.FreeTensor(vBiasOutLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeKFlat(void)
{
    LocalTensor<bfloat16_t> kGropSumLocal = groupSumQue_.AllocTensor<bfloat16_t>();
    LocalTensor<fp8_e4m3fn_t> kCacheLocal = kCacheQue_.DeQue<fp8_e4m3fn_t>();
    LocalTensor<float> kScalarLocal = kScalarBuf_.Get<float>();
    __local_mem__ fp8_e4m3fn_t *kCacheAddr = (__ubuf__ fp8_e4m3fn_t *)kCacheLocal.GetPhyAddr();
    __local_mem__ bfloat16_t *kGropSumAddr = (__ubuf__ bfloat16_t *)kGropSumLocal.GetPhyAddr();
    __local_mem__ float *kScalarAddr = (__ubuf__ float *)kScalarLocal.GetPhyAddr();

    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (dimQk_ + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<fp8_e4m3fn_t> kCacheReg;
        AscendC::MicroAPI::RegTensor<float> kCacheFp32Reg;
        AscendC::MicroAPI::RegTensor<float> kHReg;
        AscendC::MicroAPI::RegTensor<float> kGropSumReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> kFlatOutReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        for (uint16_t i = 0; i < stemStride_; i++) {
            uint32_t maskLen = static_cast<uint32_t>(dimQk_);
            for (uint16_t j = 0; j < loopCnt; j++) {
                auto kGropSumAddrStart = kGropSumAddr + (stemStride_ - i - 1) * dimQk_ + j * vfLen;
                valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                MicroAPI::Duplicate(kGropSumReg, float(0), valueMaskReg);
                for (uint16_t k = 0; k < r_; k++) {
                    auto kCacheAddrStart = kCacheAddr + k * stemStride_ * dimQk_ + i * dimQk_ + j * vfLen;
                    float kScalarValue = *(kScalarAddr + k * stemStride_ + i);
                    AscendC::MicroAPI::LoadAlign<fp8_e4m3fn_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                        kCacheReg, kCacheAddrStart);
                    AscendC::MicroAPI::Cast<float, fp8_e4m3fn_t, castTraitFp8ToFloat>(kCacheFp32Reg, kCacheReg,
                                                                                      valueMaskReg);
                    AscendC::MicroAPI::Muls(kHReg, kCacheFp32Reg, kScalarValue, valueMaskReg);
                    AscendC::MicroAPI::Add(kGropSumReg, kGropSumReg, kHReg, valueMaskReg);
                }
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFloatTo16>(kFlatOutReg, kGropSumReg, valueMaskReg);
                AscendC::MicroAPI::StoreAlign<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(
                    kGropSumAddrStart, kFlatOutReg, valueMaskReg);
            }
        }
    }
    groupSumQue_.EnQue(kGropSumLocal);
    kCacheQue_.FreeTensor(kCacheLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeVFlat(int64_t headIdx, int64_t stemBlockIdx, int32_t kvLen)
{
    LocalTensor<fp8_e4m3fn_t> vCacheLocal = vCacheQue_.DeQue<fp8_e4m3fn_t>();
    LocalTensor<float> vNormDownLocal = vNormDownQue_.AllocTensor<float>();
    float vScaleH = vScaleGm_.GetValue(headIdx);
    EventMsg<HardEvent::S_V>();
    int64_t rowIdsStart = stemBlockIdx * stemBlockSize_;

    __local_mem__ fp8_e4m3fn_t *vCacheAddr = (__ubuf__ fp8_e4m3fn_t *)vCacheLocal.GetPhyAddr();
    __local_mem__ float *normsAddr = (__ubuf__ float *)vNormDownLocal.GetPhyAddr();

    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (dimV_ + vfLen - 1) / vfLen;
    uint16_t loopCntWhere = (r_ * stemStride_ + vfLen - 1) / vfLen;
    uint16_t loopCntMax = (stemStride_ + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<fp8_e4m3fn_t> vCacheReg;
        AscendC::MicroAPI::RegTensor<float> vCacheFp32Reg;
        AscendC::MicroAPI::RegTensor<float> vHReg;
        AscendC::MicroAPI::RegTensor<float> vH2Reg;
        AscendC::MicroAPI::RegTensor<float> vRowsSqrtReg;
        AscendC::MicroAPI::RegTensor<float> vHSumReg;
        AscendC::MicroAPI::RegTensor<float> normsReg;
        AscendC::MicroAPI::RegTensor<float> normsWhereReg;
        AscendC::MicroAPI::RegTensor<float> normsMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> rowIdxsReg;
        AscendC::MicroAPI::RegTensor<float> zerosReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::MaskReg whereMaskReg;
        AscendC::MicroAPI::MaskReg maxMaskReg;
        AscendC::MicroAPI::MaskReg oneMaskReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::MaskReg allMaskReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zerosReg, float(0), allMaskReg);
        auto normsAddrStart = normsAddr;
        for (uint16_t i = 0; i < r_ * stemStride_; i++) {
            uint32_t maskLen = static_cast<uint32_t>(dimV_);
            MicroAPI::Duplicate(normsReg, float(0), oneMaskReg);
            for (uint16_t j = 0; j < loopCnt; j++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                auto vCacheAddrStart = vCacheAddr + i * dimV_ + j * vfLen;
                AscendC::MicroAPI::LoadAlign<fp8_e4m3fn_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    vCacheReg, vCacheAddrStart);
                AscendC::MicroAPI::Cast<float, fp8_e4m3fn_t, castTraitFp8ToFloat>(vCacheFp32Reg, vCacheReg,
                                                                                  valueMaskReg);
                AscendC::MicroAPI::Muls(vHReg, vCacheFp32Reg, vScaleH, valueMaskReg);
                AscendC::MicroAPI::Mul(vH2Reg, vHReg, vHReg, valueMaskReg);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float>(vHSumReg, vH2Reg, valueMaskReg);
                AscendC::MicroAPI::Add(normsReg, normsReg, vHSumReg, oneMaskReg);
            }
            AscendC::MicroAPI::Sqrt(vRowsSqrtReg, normsReg, valueMaskReg);
            AscendC::MicroAPI::Store(normsAddrStart + i, vRowsSqrtReg, 1); //  norms sum(dim=-1)   [R, StemStride]
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        normsAddrStart = normsAddr;
        uint32_t maskLen = static_cast<uint32_t>(r_ * stemStride_);
        for (uint16_t i = 0; i < loopCntWhere; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
            AscendC::MicroAPI::LoadAlign(normsReg, normsAddrStart + i * vfLen);
            AscendC::MicroAPI::Arange(rowIdxsReg, rowIdsStart + i * vfLen);
            AscendC::MicroAPI::Compares<int32_t, AscendC::CMPMODE::LT>(whereMaskReg, rowIdxsReg, kvLen, valueMaskReg);
            AscendC::MicroAPI::Select(normsWhereReg, normsReg, zerosReg, whereMaskReg);
            AscendC::MicroAPI::StoreAlign(normsAddrStart + i * vfLen, normsWhereReg, valueMaskReg); //  Where
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>(); // where  [R, StemStride]

        auto vNormDownAddr = normsAddr;
        for (uint16_t i = 0; i < r_; i++) {
            uint32_t maskLen = static_cast<uint32_t>(stemStride_);
            MicroAPI::Duplicate(normsMaxReg, float(FLT_MIN), allMaskReg);
            for (uint16_t j = 0; j < loopCntMax; j++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                auto normsWhereAddrStart = normsAddr + i * stemStride_ + j * vfLen;
                AscendC::MicroAPI::LoadAlign(normsReg, normsWhereAddrStart);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::MAX, float, float>(normsReg, normsReg, valueMaskReg);
                AscendC::MicroAPI::Compare<float, AscendC::CMPMODE::GT>(maxMaskReg, normsReg, normsMaxReg, oneMaskReg);
                AscendC::MicroAPI::Select(normsMaxReg, normsReg, normsMaxReg, maxMaskReg);
            }
            AscendC::MicroAPI::Store(vNormDownAddr + i, normsMaxReg, 1); // v_norm_down    [R]
        }
    }
    vNormDownQue_.EnQue(vNormDownLocal);
    vCacheQue_.FreeTensor(vCacheLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeVMean(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                                            __local_mem__ float *vFLatOutAddr)
{
    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (numStemBlocks * r_ + vfLen - 1) / vfLen;
    float divNum = 1 / float(numStemBlocks * r_);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vFLatOutReg;
        AscendC::MicroAPI::RegTensor<float> logValsReg;
        AscendC::MicroAPI::RegTensor<float> logValsSumReg;
        AscendC::MicroAPI::RegTensor<float> vMeanReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::MaskReg oneMaskReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate(vMeanReg, float(0), oneMaskReg);
        uint32_t maskLen = static_cast<uint32_t>(numStemBlocks * r_);
        auto logValsAddrStart = vFLatOutAddr;
        auto vMeanAddrStart = vBiasOutAddr;
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
            AscendC::MicroAPI::AddrReg vFLatOutAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(i, vfLen);
            AscendC::MicroAPI::LoadAlign(vFLatOutReg, vFLatOutAddr, vFLatOutAddrOfst);
            AscendC::MicroAPI::Adds(vFLatOutReg, vFLatOutReg, float(1e-6), valueMaskReg);
            AscendC::MicroAPI::Log(logValsReg, vFLatOutReg, valueMaskReg);
            AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float>(logValsSumReg, logValsReg, valueMaskReg);
            AscendC::MicroAPI::Add(vMeanReg, vMeanReg, logValsSumReg, oneMaskReg);
            AscendC::MicroAPI::StoreAlign(logValsAddrStart + i * vfLen, logValsReg, valueMaskReg);
        }
        AscendC::MicroAPI::Muls(vMeanReg, vMeanReg, divNum, oneMaskReg);
        AscendC::MicroAPI::Store(vMeanAddrStart, vMeanReg, 1);
    }
}

template <bool KDownLenFlag>
__aicore__ inline void StemOamPrepPagedKvSimd::ComputeVStd(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                                           __local_mem__ float *logValsAddr, float vMean)
{
    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (numStemBlocks * r_ + vfLen - 1) / vfLen;
    float divNum = 1 / float(numStemBlocks * r_ - 1);
    float vMeanNum = vMean * (-1);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> sqrtReg;
        auto vStdAddrStart = vBiasOutAddr + 1;
        AscendC::MicroAPI::MaskReg oneMaskReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        if constexpr (KDownLenFlag) {
            AscendC::MicroAPI::RegTensor<float> logValsReg;
            AscendC::MicroAPI::RegTensor<float> subReg;
            AscendC::MicroAPI::RegTensor<float> squareReg;
            AscendC::MicroAPI::RegTensor<float> sumReg;
            AscendC::MicroAPI::MaskReg valueMaskReg;
            MicroAPI::Duplicate(sumReg, float(0), oneMaskReg);
            uint32_t maskLen = static_cast<uint32_t>(numStemBlocks * r_);
            auto logValsAddrStart = logValsAddr;
            for (uint16_t i = 0; i < loopCnt; i++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                AscendC::MicroAPI::AddrReg logValsAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(i, vfLen);
                AscendC::MicroAPI::LoadAlign(logValsReg, logValsAddrStart, logValsAddrOfst);
                AscendC::MicroAPI::Adds(subReg, logValsReg, vMeanNum, valueMaskReg);
                AscendC::MicroAPI::Mul(squareReg, subReg, subReg, valueMaskReg);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float>(squareReg, squareReg, valueMaskReg);
                AscendC::MicroAPI::Add(sumReg, sumReg, squareReg, oneMaskReg);
            }
            AscendC::MicroAPI::Muls(sumReg, sumReg, divNum, oneMaskReg);
            AscendC::MicroAPI::Sqrt(sqrtReg, sumReg, oneMaskReg);
        } else {
            MicroAPI::Duplicate(sqrtReg, float(0), oneMaskReg);
        }
        AscendC::MicroAPI::Store(vStdAddrStart, sqrtReg, 1);
    }
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeVBias(int64_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                                            __local_mem__ float *logValsAddr, float vMean, float vSTd)
{
    float vMeanNum = vMean * (-1);
    float invStd = float(1.0) / (vSTd + float(1e-6));
    float divNum = 1 / float(r_);

    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (numStemBlocks * r_ + vfLen - 1) / vfLen;
    uint16_t loopRCnt = (r_ + vfLen - 1) / vfLen;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> logValsReg;
        AscendC::MicroAPI::RegTensor<float> normalizedReg;
        AscendC::MicroAPI::RegTensor<float> reluReg;
        AscendC::MicroAPI::RegTensor<float> vFinalReg;
        AscendC::MicroAPI::RegTensor<float> vFinalBlockReg;
        AscendC::MicroAPI::RegTensor<float> vFinalBlockSumReg;
        AscendC::MicroAPI::RegTensor<float> vFinalBlockMeanReg;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::MaskReg oneMaskReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t maskLen = static_cast<uint32_t>(numStemBlocks * r_);

        auto logValsAddrStart = logValsAddr;
        auto vFinalAddrStart = logValsAddr;
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
            AscendC::MicroAPI::AddrReg vFLatOutAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(i, vfLen);
            AscendC::MicroAPI::LoadAlign(logValsReg, logValsAddrStart, vFLatOutAddrOfst);
            AscendC::MicroAPI::Adds(normalizedReg, logValsReg, vMeanNum, valueMaskReg);
            AscendC::MicroAPI::Muls(normalizedReg, normalizedReg, invStd, valueMaskReg);
            AscendC::MicroAPI::Relu(reluReg, normalizedReg, valueMaskReg);
            AscendC::MicroAPI::Muls(vFinalReg, reluReg, lambdaMag_, valueMaskReg);
            AscendC::MicroAPI::StoreAlign(vFinalAddrStart + i * vfLen, vFinalReg, valueMaskReg);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        auto vBiasOutAddrStart = vBiasOutAddr;
        for (uint16_t i = 0; i < numStemBlocks; i++) {
            uint32_t maskLen = static_cast<uint32_t>(r_);
            MicroAPI::Duplicate(vFinalBlockSumReg, float(0), oneMaskReg);
            for (uint16_t j = 0; j < loopRCnt; j++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                auto vFinalBlockAddrStart = logValsAddr + i * r_ + j * vfLen;
                AscendC::MicroAPI::LoadUnAlignPre(u0, vFinalBlockAddrStart);
                AscendC::MicroAPI::LoadUnAlign(vFinalBlockReg, u0, vFinalBlockAddrStart);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float>(vFinalBlockReg, vFinalBlockReg,
                                                                                   valueMaskReg);
                AscendC::MicroAPI::Add(vFinalBlockSumReg, vFinalBlockSumReg, vFinalBlockReg, oneMaskReg);
            }
            AscendC::MicroAPI::Muls(vFinalBlockMeanReg, vFinalBlockSumReg, divNum, oneMaskReg);
            AscendC::MicroAPI::Store(vBiasOutAddrStart + i, vFinalBlockMeanReg, 1);
        }
    }
}

__aicore__ inline void StemOamPrepPagedKvSimd::ComputeVBiasOut(int32_t batchIdx, int32_t headIdx, int64_t kDownLen,
                                                               int64_t numStemBlocks)
{
    LocalTensor<float> vBiasOutLocal = vBiasOutQue_.AllocTensor<float>();
    LocalTensor<float> vFLatOutLocal = vFLatOutQue_.DeQue<float>();

    __local_mem__ float *vBiasOutAddr = (__ubuf__ float *)vBiasOutLocal.GetPhyAddr();
    __local_mem__ float *vFLatOutAddr = (__ubuf__ float *)vFLatOutLocal.GetPhyAddr();

    ComputeVMean(numStemBlocks, vBiasOutAddr, vFLatOutAddr); //  保存在vBiasOutAddr第一个位置,vFLatOutAddr保存LogVals
    EventMsg<HardEvent::V_S>();
    float vMean = vBiasOutLocal.GetValue(0);
    EventMsg<HardEvent::S_V>();
    if (kDownLen > 1) {
        ComputeVStd<true>(numStemBlocks, vBiasOutAddr, vFLatOutAddr, vMean); //  保存在vBiasOutAddr第二个位置
    } else {
        ComputeVStd<false>(numStemBlocks, vBiasOutAddr, vFLatOutAddr, vMean);
    }
    EventMsg<HardEvent::V_S>();
    float vSTd = vBiasOutLocal.GetValue(1);
    EventMsg<HardEvent::S_V>();
    ComputeVBias(numStemBlocks, vBiasOutAddr, vFLatOutAddr, vMean, vSTd);
    vBiasOutQue_.EnQue(vBiasOutLocal);
    vFLatOutQue_.FreeTensor(vFLatOutLocal);
}

__aicore__ inline void StemOamPrepPagedKvSimd::ProcessVBias(void)
{
    for (int32_t bhIdx_ = blockIdx_; bhIdx_ < totalBh_; bhIdx_ += blockNum_) {
        int32_t batchIdx = bhIdx_ / hKv_;
        int32_t headIdx = bhIdx_ % hKv_;
        int32_t kvLen = kvSeqLensGm_.GetValue(batchIdx);
        int64_t kPadded = (static_cast<int64_t>(kvLen) + stemBlockSize_ - 1) / stemBlockSize_ * stemBlockSize_;
        int64_t numStemBlocks = kPadded / stemBlockSize_;
        int64_t kDownLen = kPadded / stemStride_;
        if (kDownLen <= 0) {
            continue;
        }
        CopyInVFlatOut(batchIdx, headIdx, numStemBlocks);
        ComputeVBiasOut(batchIdx, headIdx, kDownLen, numStemBlocks);
        CopyOutVBias(batchIdx, headIdx, numStemBlocks);
    }
}

__aicore__ inline void StemOamPrepPagedKvSimd::ProcessKVFlat(void)
{
    if (blockIdx_ >= kvUsedCoreNum_) {
        return;
    }
    LocalTensor<int32_t> totalStemBlockIndexLocal = totalStemBlockIndexBuf_.Get<int32_t>();
    auto stemBlockNum = (blockIdx_ == kvUsedCoreNum_ - 1) ? tailCoreProcessStemBlock_ : curCoreProcessStemBlock_;
    for (int32_t idx = 0; idx < stemBlockNum; idx++) {
        int32_t batchIdx = totalStemBlockIndexLocal.GetValue(idx * 3);
        int32_t headIdx = totalStemBlockIndexLocal.GetValue(idx * 3 + 1);
        int32_t stemBlockIdx = totalStemBlockIndexLocal.GetValue(idx * 3 + 2);
        int32_t kvLen = kvSeqLensGm_.GetValue(batchIdx);
        int64_t kPadded = (static_cast<int64_t>(kvLen) + stemBlockSize_ - 1) / stemBlockSize_ * stemBlockSize_;
        int64_t numKVBlocks = (static_cast<int64_t>(kvLen) + kvBlockSize_ - 1) / kvBlockSize_;
        int64_t actualRows = min((int64_t)kvLen, (int64_t)numKVBlocks * kvBlockSize_);
        CopyInKVCache(batchIdx, headIdx, stemBlockIdx, actualRows);
        ComputeKFlat();
        ComputeVFlat(headIdx, stemBlockIdx, kvLen);
        CopyOutKFlat(batchIdx, headIdx, stemBlockIdx);
        CopyOutVFlat(batchIdx, headIdx, stemBlockIdx);
    }
}

__aicore__ inline void StemOamPrepPagedKvSimd::Process(void)
{
    ComputTotalStemBlockIndex();
    ProcessKVFlat();
    SyncAll();
    ProcessVBias();
}


#endif //  STEM_OAM_PREP_PAGED_KV_SIMD_H