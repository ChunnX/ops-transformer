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
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "stem_oam_prep_paged_kv_tiling_data.h"
#include "stem_oam_prep_paged_kv_base.h"

using namespace AscendC;

class StemOamPrepPagedKv {
public:
    __aicore__ inline StemOamPrepPagedKv() = default;
    __aicore__ inline void Process(void);
    __aicore__ inline void Init(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices, GM_ADDR kvSeqLens,
                                GM_ADDR kScaleCache, GM_ADDR vScale, GM_ADDR kFlat, GM_ADDR vBias, GM_ADDR workspace,
                                const StemPrepKVTilingData *tilingData, TPipe *pipe);

private:
    __aicore__ inline void CopyInKVCache(int64_t batchIdx, int64_t headIDx, int64_t stemBlockIdx, uint32_t actualRows);
    __aicore__ inline void CopyInVFlatOut(int64_t batchIdx, int64_t headIdx, uint32_t numStemBlocks);

    __aicore__ inline void ComputeKFlat(void);
    __aicore__ inline void ComputeVFlat(uint32_t headIdx, uint32_t stemBlockIdx, uint32_t kvLen, uint32_t kDownLen);
    __aicore__ inline void ComputeVMean(uint32_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                        __local_mem__ float *vFLatOutAddr);
    template <bool KDownLenFlag>
    __aicore__ inline void ComputeVStd(uint32_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                       __local_mem__ float *logValsAddr);
    __aicore__ inline void ComputeVBias(uint32_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                        __local_mem__ float *logValsAddr);
    __aicore__ inline void ComputeVBiasOut(int64_t batchIdx, int64_t headIdx, uint32_t kDownLen,
                                           uint32_t numStemBlocks);
    __aicore__ inline void ProcessVBias(void);
    __aicore__ inline void ProcessKVFlat(void);

    __aicore__ inline void CopyOutKFlat(int64_t batchIdx, int64_t headIDx, int64_t stemBlocksIdx);
    __aicore__ inline void CopyOutVFlat(int64_t batchIdx, int64_t headIDx, int64_t stemBlocksIdx);
    __aicore__ inline void CopyOutVBias(int64_t batchIdx, int64_t headIdx, uint32_t numStemBlocks);

private:
    TPipe *pipe_;
    StemPrepKVTilingData tilingData_;

    GlobalTensor<fp8_e4m3fn_t> kCacheGm_;
    GlobalTensor<fp8_e4m3fn_t> vCacheGm_;
    GlobalTensor<int32_t> kvIndicesGm_;
    GlobalTensor<int32_t> kvSeqLensGm_;
    GlobalTensor<float> kScaleCacheGm_;
    GlobalTensor<float> vScaleGm_;

    GlobalTensor<bfloat16_t> kFlatGm_;
    GlobalTensor<float> vBiasGm_;

    GlobalTensor<float> vFlatOutWs_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> kvCacheQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> vCacheQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> kvIndicesQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> vFLatOutQue_;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> groupSumQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> vNormDownQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> vBiasOutQue_;

    TBuf<TPosition::VECCALC> groupSumRowBuf_;
    TBuf<TPosition::VECCALC> kScalarBuf_;
    TBuf<TPosition::VECCALC> vNormDownBuf_;
    TBuf<TPosition::VECCALC> totalStemBlockIndexBuf_;

    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t totalTasks_ = 0;
    uint32_t kvLayout_ = 0;
    uint32_t kvBlockSize_ = 0;
    uint32_t hKv_ = 0;
    uint32_t splitMode_ = 0;
    uint32_t maxKvBlocks_ = 0;
    uint32_t dimQk_ = 0;
    uint32_t dimV_ = 0;
    uint32_t maxKb_ = 0;
    uint32_t kflatDim_ = 0;
    uint32_t batchSize_ = 0;
    int32_t stemBlockSize_ = 0;
    int32_t stemStride_ = 0;
    int32_t kvBlockSize_ = 0;
    uint32_t totalBh_ = 0;
    float lambdaMag_ = 0;
    int32_t r_ = 0;
    int64_t kCacheCopySize_ = 0;
    int64_t vCacheCopySize_ = 0;
    int64_t kFlatDim_ = 0;
    uint32_t kCacheStride_[4] = {0};
    uint32_t vCacheStride_[4] = {0};
    uint32_t kScakarCacheStride_[4] = {0};
    int64_t vCacheOffset_ = 0;

    constexpr static uint32_t V_REG_SIZE = Ops::Base::GetVRegSize();
};

__aicore__ inline void StemOamPrepPagedKv::Init(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices, GM_ADDR kvSeqLens,
                                                GM_ADDR kScaleCache, GM_ADDR vScale, GM_ADDR kFlat, GM_ADDR vBias,
                                                GM_ADDR workspace, const StemPrepKVTilingData *tilingData, TPipe *pipe)
{
    kCacheGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)kCache);
    vCacheGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)vCache);
    kvIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)kvIndices);
    kvSeqLensGm_.SetGlobalBuffer((__gm__ int32_t *)kvSeqLens);
    kScaleCacheGm_.SetGlobalBuffer((__gm__ float *)kScaleCache);
    vScaleGm_.SetGlobalBuffer((__gm__ float *)vScale);
    kFlatGm_.SetGlobalBuffer((__gm__ bfloat16_t *)kFlat);
    vBiasGm_.SetGlobalBuffer((__gm__ float *)vBias);

    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    vCacheOffset_ = kvBlockSize_ * dimQk_;

    // inti UB
    pipe_->InitBuffer(kCacheQue_, DOUBLE_BUFFER, );
    pipe_->InitBuffer(vCacheQue_, DOUBLE_BUFFER, );

    // init OutPut 0
    if (blockIdx_ == 0) {
        InitOutput<bfloat16>(kFlat, batchSize_ * hKv_ * maxKb_ * kflatDim_, bfloat16_t(0));
        InitOutput<U>(vBias, batchSize_ * hKv_ * maxKb_, float(0));
    }
    SyncAll();
}

__aicore__ inline void StemOamPrepPagedKv::CopyInVFlatOut(int64_t batchIdx, int64_t headIdx, uint32_t numStemBlocks)
{
    LocalTensor<float> vFLatOutLocal = vFLatOutQue_.AllocTensor<float>();
    auto vFlatOutWsOffset = batchIdx * hKv_ * maxKb_ * r_ + headIdx * maxKb_ * r_;

    CopyIn(vFLatOutLocal, vFlatOutWs_[vFlatOutWsOffset], 1, numStemBlocks * r_, 0, 0);

    vFLatOutQue_.EnQue(vFLatOutLocal);
}

__aicore__ inline void StemOamPrepPagedKv::CopyInKVCache(int64_t batchIdx, int64_t headIdx, int64_t stemBlockIdx,
                                                         uint32_t actualRows)
{
    LocalTensor<fp8_e4m3fn_t> kCacheLocal = kvCacheQue_.AllocTensor<fp8_e4m3fn_t>();
    LocalTensor<fp8_e4m3fn_t> vCacheLocal = kCacheLocal[vCacheOffset_];
    LocalTensor<float> kScalarLocal = kScalarBuf_.Get<float>();
    for (uint32_t i = 0; i < stemBlockSize_; i++) {
        int64_t kvIndicesIdx = (stemBlockIdx * stemBlockSize_ + i) / kvBlockSize_;
        int64_t kvBlockSizeIdx = (stemBlockIdx * stemBlockSize_ + i) % kvBlockSize_;
        int64_t logicIdx = kvIndicesGm_.GetValue(batchIdx * maxKvBlocks_ + kvIndicesIdx);
        int64_t kCacheOffset =
            logicIdx * kCacheStride_[0] + headIdx * kCacheStride_[1] + kvBlockSizeIdx * kCacheStride_[2];
        int64_t vCacheOffset =
            logicIdx * vCacheStride_[0] + headIdx * vCacheStride_[1] + kvBlockSizeIdx * vCacheStride_[2];
        int64_t kScaleCacheOffset = logicIdx * kScakarCacheStride_[0] + headIdx * kScakarCacheStride_[1] +
                                    kvBlockSizeIdx * kScakarCacheStride_[2];
        if ((stemBlockIdx * stemBlockSize_ + i) < actualRows) {
            CopyIn(kCacheLocal[i * dimQk_], kCacheGm_[kCacheOffset], 1, dimQk_, 0, 0); // R * StemStride * dim_qk
            CopyIn(vCacheLocal[i * dimV_], vCacheGm_[vCacheOffset], 1, dimV_, 0, 0);
            kScalarLocal.SetValue(i, kScaleCacheGm_.GetValue(kScaleCacheOffset));
        } else {
            Duplicate(kCacheLocal[i * dimQk_], fp8_e4m3fn_t(0), dimQk_);
            Duplicate(vCacheLocal[i * dimV_], fp8_e4m3fn_t(0), dimV_);
            kScalarLocal.SetValue(i, 0);
        }
    }
    kCacheQue_.EnQue(kCacheLocal);
}

__aicore__ inline void StemOamPrepPagedKv::CopyOutKFlat(int64_t batchIdx, int64_t headIdx, int64_t stemBlocksIdx)
{
    LocalTensor<bfloat16_t> groupSumLocal = groupSumQue_.DeQue<bfloat16_t>();
    auto kFlatGmOffset =
        batchIdx * hKv_ * maxKb_ * kflatDim_ + headIdx * maxKb_ * kflatDim_ + numStemBlocksIdx * kflatDim_;
    CopyOut(kFlatGm_[kFlatGmOffset], groupSumLocal, 1, kflatDim_, 0, 0);
    groupSumQue_.FreeTensor(groupSumLocal);
}

__aicore__ inline void StemOamPrepPagedKv::CopyOutVFlat(int64_t batchIdx, int64_t headIdx, int64_t stemBlocksIdx)
{
    LocalTensor<float> vNormDownLocal = vNormDownQue_.DeQue<float>();
    int64_t vFlatOutOffset = batchIdx * hKv_ * maxKb_ * r_ + headIdx * maxKb_ * r_ + stemBlocksIdx * r_;

    CopyOut(vFlatOutWs_[vFlatOutOffset], vNormDownLocal, 1, r_, 0, 0);

    vNormDownQue_.FreeTensor(vNormDownLocal);
}

__aicore__ inline void StemOamPrepPagedKv::CopyOutVBias(int64_t batchIdx, int64_t headIdx, uint32_t numStemBlocks)
{
    LocalTensor<float> vBiasOutLocal = vBiasOutQue_.DeQue<float>();

    auto vBiasOffset = batchId * hKv_ * maxKb_ + headIdx * maxKb_;
    CopyOut(vBiasGm_[vBiasOffset], vBiasOutLocal, 1, numStemBlocks, 0, 0);

    vBiasOutQue_.FreeTensor(vBiasOutLocal);
}

__aicore__ inline void StemOamPrepPagedKv::ComputeKFlat(void)
{
    LocalTensor<bfloat16_t> groupSumLocal = groupSumQue_.AllocTensor<bfloat16_t>();
    LocalTensor<fp8_e4m3fn_t> kCacheLocal = kvCacheQue_.DeQue();
    LocalTensor<fp8_e4m3fn_t> vCacheLocal = kCacheLocal[vCacheOffset_];
    LocalTensor<float> kScalarLocal = kScalarBuf_.Get<float>();
    __local_mem__ fp8_e4m3fn_t *kCacheAddr = (__ubuf__ fp8_e4m3fn_t *)kCacheLocal.GetPhyAddr();
    __local_mem__ bfloat16_t *kGropSumAddr = (__ubuf__ bfloat16_t *)kGropSumLocal.GetPhyAddr();
    __local_mem__ float *kScalarAddr = (__ubuf__ CAST_TYPE *)kScalarLocal.GetPhyAddr();

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
                AscendC::MicroAPI::AddrReg kCacheAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLen);
                for (uint16_t k = 0; k < r; k++) {
                    float kScalarValue = *(kScalarAddr + k * stemStride_ + j);
                    AscendC::MicroAPI::LoadAlign<fp8_e4m3fn_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                        kCacheReg, kCacheAddr, kCacheAddrOfst);
                    AscendC::MicroAPI::Cast<float, fp8_e4m3fn_t, castTraitFp8ToFloat>(kCacheFp32Reg, kCacheReg,
                                                                                      valueMaskReg);
                    AscendC::MicroAPI::Muls(kHReg, kCacheFp32Reg, kScalarValue, valueMaskReg);
                    AscendC::MicroAPI::Add(kGropSumReg, kGropSumReg, kHReg, valueMaskReg);
                }
                AscendC::MicroAPI::Cast<bfloat16_t, float, castTraitFloatTo16>(kFlatOut, kGropSumReg, valueMaskReg);
                AscendC::MicroAPI::StoreAlign<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(kGropSumAddrStart,
                                                                                              kFlatOut, valueMaskReg)
            }
        }
    }

    groupSumQue_.EnQue(groupSumLocal);
    kvCacheQue_.FreeTensor(kCacheLocal);
}

__aicore__ inline void StemOamPrepPagedKv::ComputeVFlat(uint32_t headIdx, uint32_t stemBlockIdx, uint32_t kvLen,
                                                        uint32_t kDownLen)
{
    LocalTensor<fp8_e4m3fn_t> kCacheLocal = kvCacheQue_.DeQue();
    LocalTensor<fp8_e4m3fn_t> vCacheLocal = kCacheLocal[vCacheOffset_];
    LocalTensor<float> vNormDownLocal = vNormDownQue_.AllocTensor<float>();
    float vScaleH = vScaleGm_.GetValue(headIDx);
    uint32_t rowIdsStart = stemBlockIdx * stemBlockSize_;

    __local_mem__ fp8_e4m3fn_t *vCacheeAddr = (__ubuf__ fp8_e4m3fn_t *)vCacheLocal.GetPhyAddr();
    __local_mem__ float *normsAddr = (__ubuf__ float *)vNormDownLocal.GetPhyAddr();
    __local_mem__ float *vNormDownAddr = (__ubuf__ float *)vNormDownLocal.GetPhyAddr();

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
        AscendC::MicroAPI::RegTensor<uint32_t> rowIdxsReg;
        AscendC::MicroAPI::RegTensor<float> zerosReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::MaskReg whereMaskReg;
        AscendC::MicroAPI::MaskReg maxMaskReg;
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::MaskReg allMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zerosReg, float(0), allMaskReg);
        MicroAPI::Duplicate(normsMaxReg, float(FLT_MIN), allMaskReg);
        auto normsAddrStart = normsAddr;
        for (uint16_t i = 0; i < r_ * stemStride_; i++) {
            auto vCacheAddrStart = vCacheAddr + i * dimV_;
            uint32_t maskLen = static_cast<uint32_t>(dimV_);
            MicroAPI::Duplicate(normsReg, float(0), OneMaskReg);
            for (int j = 0; j < loopCnt; j++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                AscendC::MicroAPI::AddrReg vCacheAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLen);
                AscendC::MicroAPI::LoadAlign<fp8_e4m3fn_t, AscendC::MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    vCacheReg, vCacheAddrStart, vCacheAddrOfst);
                AscendC::MicroAPI::Cast<float, fp8_e4m3fn_t, castTraitFp8ToFloat>(vCacheFp32Reg, vCacheReg,
                                                                                  valueMaskReg);
                AscendC::MicroAPI::Muls(vHReg, vCacheFp32Reg, vScaleH, valueMaskReg);
                AscendC::MicroAPI::Mul(vH2Reg, vHReg, vHReg, valueMaskReg);
                AscendC::MicroAPI::Sqrt(vRowsSqrtReg, vH2Reg, valueMaskReg);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float>(vHSumReg, vHReg, valueMaskReg);
                AscendC::MicroAPI::Add(normsReg, vHSumReg, OneMaskReg);
            }
            AscendC::MicroAPI::Store(normsAddr, normsReg, 1); //  norms sum(dim=-1)   [R, StemStride]
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        normsAddrStart = normsAddr;
        uint32_t maskLen = static_cast<uint32_t>(r_ * stemStride_);
        for (uint16_t i = 0; i < loopCntWhere; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
            AscendC::MicroAPI::AddrReg normsAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(i, vfLen);
            AscendC::MicroAPI::LoadAlign(normsReg, normsAddrStart, normsAddrOfst);
            AscendC::MicroAPI::Arange(rowIdxsReg, rowIdsStart + i * vfLen);
            AscendC::MicroAPI::Compares<uint32_t, AscendC::MicroAPI::CMPMODE::LT>(whereMaskReg, rowIdxsReg, kvLen,
                                                                                  valueMaskReg);
            AscendC::MicroAPI::Select(normsWhereReg, normsReg, zerosReg, whereMaskReg);
            AscendC::MicroAPI::StoreAlign(normsAddrStart + i * vfLen, normsWhereReg, valueMaskReg); //  Where
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        auto vNormDownAddr = normsAddr;
        for (int i = 0; i < r_; i++) {
            uint32_t maskLen = static_cast<uint32_t>(stemStride_);
            normsAddrStart = normsAddr + i * stemStride_;
            for (uint16_t j = 0; j < loopCntMax; j++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                AscendC::MicroAPI::AddrReg normsAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLen);
                AscendC::MicroAPI::LoadAlign(normsReg, normsAddrStart, normsAddrOfst);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::MAX, float, float>(normsReg, normsReg, valueMaskReg);
                AscendC::MicroAPI::Compare<uint32_t, AscendC::MicroAPI::CMPMODE::LT>(maxMaskReg, normsReg, normsMaxReg,
                                                                                     OneMaskReg);
                AscendC::MicroAPI::Select(normsMaxReg, normsReg, normsMaxReg, maxMaskReg);
            }
            AscendC::MicroAPI::Store(vNormDownAddr, normsMaxReg, 1); // v_norm_down    [R]
        }
    }
    vNormDownQue_.EnQue(vNormDownLocal);
}

__aicore__ inline void StemOamPrepPagedKv::ComputeVMean(uint32_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
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
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate(vMeanRe, float(0), OneMaskReg);
        uint32_t maskLen = static_cast<uint32_t>(numStemBlocks * r_);
        auto vFLatOutAddrStart = vFLatOutAddr;
        auto vMeanAddrStart = vBiasOutAddr;
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
            AscendC::MicroAPI::AddrReg vFLatOutAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(i, vfLen);
            AscendC::MicroAPI::LoadAlign(vFLatOutReg, vFLatOutAddrStart, vFLatOutAddrOfst);
            AscendC::MicroAPI::Adds(vFLatOutReg, vFLatOutReg, float(1e-6), valueMaskReg);
            AscendC::MicroAPI::Log(logValsReg, vFLatOutReg, valueMaskRe);
            AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float>(logValsSumReg, logValsReg, valueMaskReg);
            AscendC::MicroAPI::Add(vMeanReg, vMeanReg, logValsSumReg, OneMaskReg);
            AscendC::MicroAPI::StoreAlign(vFLatOutAddrStart + i * vfLen, logValsReg, valueMaskReg);
        }
        AscendC::MicroAPI::Muls(vMeanReg, vMeanReg, divNum, OneMaskReg);
        AscendC::MicroAPI::Store(vMeanAddrStart, vMeanReg, 1);
    }
}

template <bool KDownLenFlag>
__aicore__ inline void StemOamPrepPagedKv::ComputeVStd(uint32_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                                       __local_mem__ float *logValsAddr)
{
    uint32_t vfLen = V_REG_SIZE / sizeof(float);
    uint16_t loopCnt = (numStemBlocks * r_ + vfLen - 1) / vfLen;
    float divNum = 1 / float(numStemBlocks * r_ - 1);
    float vMean = *(vBiasOutAddr) * (-1);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> sqrtReg;
        if constexpr (KDownLenFlag) {
            AscendC::MicroAPI::RegTensor<float> logValsReg;
            AscendC::MicroAPI::RegTensor<float> subReg;
            AscendC::MicroAPI::RegTensor<float> squareReg;
            AscendC::MicroAPI::RegTensor<float> sumReg;
            AscendC::MicroAPI::MaskReg valueMaskReg;
            AscendC::MicroAPI::MaskReg OneMaskReg =
                AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();
            MicroAPI::Duplicate(sumReg, float(0), OneMaskReg);
            uint32_t maskLen = static_cast<uint32_t>(numStemBlocks * r_);
            auto logValsAddrStart = logValsAddr;
            auto vStdAddrStart = vBiasOutAddr + 1;
            for (uint16_t i = 0; i < loopCnt; i++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                AscendC::MicroAPI::AddrReg logValsAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(i, vfLen);
                AscendC::MicroAPI::LoadAlign(logValsReg, logValsAddrStart, logValsAddrOfst);
                AscendC::MicroAPI::Adds(subReg, logValsReg, vMean, valueMaskReg);
                AscendC::MicroAPI::Mul(squareReg, subReg, subReg, valueMaskReg);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float>(squareReg, squareReg, valueMaskReg);
                AscendC::MicroAPI::Add(sumReg, sumReg, squareReg, OneMaskReg);
            }
            AscendC::MicroAPI::Muls(sumReg, sumReg, divNum, OneMaskReg);
            AscendC::MicroAPI::Sqrt(sqrtReg, sumReg, OneMaskReg);
        } else {
            MicroAPI::Duplicate(sqrtReg, float(0), OneMaskReg);
        }
        AscendC::MicroAPI::Store(vStdAddrStart, sqrtReg, 1);
    }
}

__aicore__ inline void StemOamPrepPagedKv::ComputeVBias(uint32_t numStemBlocks, __local_mem__ float *vBiasOutAddr,
                                                        __local_mem__ float *logValsAddr)
{
    float vSTd = *(vBiasOutAddr);
    float vMean = *(vBiasOutAddr + 1) * (-1);
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
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate(sumReg, float(0), OneMaskReg);
        uint32_t maskLen = static_cast<uint32_t>(numStemBlocks * r_);

        auto logValsAddrStart = logValsAddr;
        auto vFinalAddrStart = logValsAddr;
        for (uint16_t i = 0; i < loopCnt; i++) {
            valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
            AscendC::MicroAPI::AddrReg vFLatOutAddrOfst = AscendC::MicroAPI::CreateAddrReg<float>(i, vfLen);
            AscendC::MicroAPI::LoadAlign(logValsReg, logValsAddrStart, vFLatOutAddrOfst);
            AscendC::MicroAPI::Adds(normalizedReg, logValsReg, vMean, valueMaskReg);
            AscendC::MicroAPI::Muls(normalizedReg, normalizedReg, invStd, valueMaskReg);
            AscendC::MicroAPI::Relu(reluReg, normalizedReg, valueMaskReg);
            AscendC::MicroAPI::Muls(vFinalReg, reluReg, lambdaMag_, valueMaskReg);
            AscendC::MicroAPI::StoreAlign(vFinalAddrStar + i * vfLen, vFinalReg, valueMaskReg);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        auto vBiasOutAddrStart = vBiasOutAddr;
        for (uint16_t i = 0; i < numStemBlocks; i++) {
            maskLen = static_cast<uint32_t>(r_);
            MicroAPI::Duplicate(vFinalBlockSumReg, float(0), OneMaskReg);
            for (uint16_t j = 0; j < loopRCnt; j++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                auto vFinalBlockAddrStart = logValsAddr + i * r_ + j * vfLen;
                AscendC::MicroAPI::LoadUnAlignPre(u0, vFinalBlockAddrStart);
                AscendC::MicroAPI::LoadUnAlign(vFinalBlockReg, u0, vFinalBlockAddrStart);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float>(vFinalBlockReg, vFinalBlockReg,
                                                                                   valueMaskReg);
                AscendC::MicroAPI::Add(vFinalBlockSumReg, vFinalBlockSumReg, vFinalBlockReg, OneMaskReg);
            }
            AscendC::MicroAPI::Muls(vFinalBlockMeanReg, vFinalBlockSumReg, divNum, OneMaskReg);
            AscendC::MicroAPI::Store(vBiasOutAddrStart, vFinalBlockMeanReg, 1);
        }
    }
}

__aicore__ inline void StemOamPrepPagedKv::ComputeVBiasOut(int64_t batchIdx, int64_t headIdx, uint32_t kDownLen,
                                                           uint32_t numStemBlocks)
{
    LocalTensor<float> vBiasOutLocal = vBiasOutQue_.AllocTensor<float>();
    LocalTensor<float> vFLatOutLocal = vFLatOutQue_.DeQue<float>();

    __local_mem__ float *vBiasOutAddr = (__ubuf__ float *)vBiasOutLocal.GetPhyAddr();
    __local_mem__ float *vFLatOutAddr = (__ubuf__ float *)vFLatOutLocal.GetPhyAddr();

    ComputeVMean(numStemBlocks, vBiasOutAddr, vFLatOutAddr); //  保存在vBiasOutAddr第一个位置
    if (kDownLen > 1) {
        ComputeVStd<true>(numStemBlocks, vBiasOutAddr, vFLatOutAddr);
    } else {
        ComputeVStd<false>(numStemBlocks, vBiasOutAddr, vFLatOutAddr);
    }
    ComputeVBias(numStemBlocks, vBiasOutAddr, vFLatOutAdd);
    CopyOutVBias(batchIdx, headIdx, numStemBlocks);
    vBiasOutQue_.EnQue(vBiasOutLocal);
    vFLatOutQue_.FreeTensor(vFLatOutLocal);
}

__aicore__ inline void StemOamPrepPagedKv::ProcessVBias(void)
{
    for (uint32_t bhIdx_ = 0; bhIdx_ < totalBh_; bhIdx_ += blockNum_) {
        uint32_t batchIdx = bhIdx_ / hKv_;
        uint32_t headIdx = bhIdx_ % hKv_;
        uint32_t kPadded = (static_cast<int64_t>(kvLen) + stemBlockSize_ - 1) / stemBlockSize_ * stemBlockSize_;
        uint32_t numStemBlocks = kPadded / stemBlockSize_;
        uint32_t kDownLen = kPadded / stemStride;
        CopyInVFlatOut(batchIdx, headIdx, numStemBlocks);
        ComputeVBiasOut(batchIdx, headIdx, kDownLen, numStemBlocks);
        CopyOutVBias(batchIdx, headIdx, numStemBlocks);
    }
}

__aicore__ inline void StemOamPrepPagedKv::ProcessKVFlat(void)
{
    for (uint32_t bhIdx_ = 0; bhIdx_ < totalBh_; bhIdx_++) {
        uint32_t batchIdx = bhIdx_ / hKv_;
        uint32_t headIdx = bhIdx_ % hKv_;
        uint32_t kvLen = kvSeqLensGm_.GetValue(batchIdx);
        uint32_t kPadded = (static_cast<int64_t>(kvLen) + stemBlockSize_ - 1) / stemBlockSize_ * stemBlockSize_;
        uint32_t numKVBlocks = (static_cast<int64_t>(kvLen) + kvBlockSize_ - 1) / kvBlockSize_;
        uint32_t numStemBlocks = kPadded / stemBlockSize_;
        uint32_t actualRows = min(kvLen, numKVBlocks * kvBlockSize_);
        uint32_t kDownLen = kPadded / stemStride;
        for (uint32_t stemBlockIdx = blockIdx_; stemBlockIdx < numStemBlocks; numStemBlocks += blockNum_) {
            CopyInKVCache(batchIdx, headIdx, stemBlockIdx, actualRows);
            ComputeKFlat();
            ComputeVFlat();
            CopyOutKFlat(batchIdx, headIdx, stemBlockIdx);
            CopyOutVFlat(batchIdx, headIdx, stemBlockIdx);
        }
    }
}

__aicore__ inline void StemOamPrepPagedKv::Process(void)
{
    ProcessKVFlat();
    SyncAll();
    ProcessVBias();
}


#endif //  STEM_OAM_PREP_PAGED_KV_SIMD_H