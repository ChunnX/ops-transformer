/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_flat_quant.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/epilogue/fusion/default_fusion_op.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

namespace Constant {
constexpr uint8_t GATHER_PATTERN = 7;
constexpr int32_t CEIL_SIZE = 16;
constexpr int32_t GROUP_SIZE = 32;
constexpr int32_t VEC_N_LEN = 64;
constexpr int32_t MN_SIZE = 64 * 1024;
constexpr int32_t OUT_SIZE = 32 * 1024;
constexpr int32_t EMAX_SIZE = 2 * 1024;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t BLOCK_SCALE = 2;
constexpr float ZERO_FLOAT = 0.0f;
constexpr float SIX_FLOAT = 6.0f;
constexpr float SEVEN_FLOAT = 7.0f;
constexpr float TWELVE_FLOAT = 12.0f;
constexpr uint32_t STORE_UNALIGN_STRIDE_BYTES = 8;
constexpr uint32_t SCALE_STORE_STRIDE = 32;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f;
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t vfLen16 = AscendC::GetVecLen() / sizeof(uint16_t);
constexpr uint32_t vfLen32 = AscendC::GetVecLen() / sizeof(uint32_t);
} // namespace Constant

struct FlatQuantShapeInfo {
    int64_t k{0};
    int64_t m{0};
    int64_t n{0};
    int64_t mCeil{0};
    int64_t nCeil{0};
};

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_,
          typename FusionOp_ = Fusion::DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueFlatQuant {
public:
    using DataTypeIn = DataTypeIn_;
    using DataTypeOut = DataTypeOut_;
    using DataTypeScale = DataTypeScale_;
    using FusionOp = FusionOp_;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        ProblemShape problemShape{};
        float dstTypeMax{0.0f};
        float invDstTypeMax{0.0f};
    };

    __aicore__ inline BlockEpilogueFlatQuant() {}

    __aicore__ inline void Init(Params const& params);
    __aicore__ inline void operator()(uint64_t startBatchIdx, uint64_t iterBatch);

private:
    __aicore__ inline void Quant(uint64_t batchIdx, uint64_t iterIdx);
    __aicore__ inline void ClearDirtyData();
    __aicore__ inline void ClearScaleTensor();
    __aicore__ inline void CopyOutputFromUbToGm(uint64_t offset, AscendC::LocalTensor<int8_t>& src);
    __aicore__ inline void CopyScaleFromUbToGm(uint64_t offset, AscendC::LocalTensor<int8_t>& src);
    __aicore__ inline void ComputeMxQuant(LocalTensor<bfloat16_t>& xTensor, LocalTensor<int8_t>& yTensor,
                                          LocalTensor<uint16_t>& eMaxTensor, LocalTensor<int8_t>& scaleTensor,
                                          LocalTensor<uint16_t>& deQuantScaleTensor, uint32_t totalDataInUB,
                                          uint64_t inputOffset);
    __aicore__ inline void ComputeTransLayout(LocalTensor<int8_t>& scaleTensor, LocalTensor<int8_t>& scaleBlockTensor,
                                              uint16_t m, uint16_t n);

    static __simd_vf__ inline void SaveTailVf(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* srcPtr, uint32_t count);
    static __simd_vf__ inline void ClearTailVf(__ubuf__ uint16_t* dstPtr, uint32_t count);
    static __simd_vf__ inline void RestoreTailVf(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* srcPtr, uint32_t count);

    static __simd_vf__ inline void ExpMaxVf(__ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* srcPtr, uint32_t count,
                                            uint16_t repeatTimes, uint32_t oneRepeatSize);
    static __simd_vf__ inline void ExpMaxVfcuBLAS(__ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* srcPtr,
                                                  uint32_t count, uint16_t repeatTimes, uint32_t oneRepeatSize);
    static __simd_vf__ inline void ScaleVf(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                           __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes);
    static __simd_vf__ inline void ScaleVfDynamic(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                                  __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes,
                                                  uint16_t addValueBit);
    static __simd_vf__ inline void ScaleVfcuBLAS(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                                 __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes,
                                                 float invDstTypeMax);
    static __simd_vf__ inline void QuantVf(__ubuf__ int8_t* dstPtr, __ubuf__ bfloat16_t* srcPtr,
                                           __ubuf__ uint16_t* src2Ptr, uint32_t oneRepeatSize, uint16_t repeatTimes);
    static __simd_vf__ inline void TransLayoutVf(__ubuf__ int8_t* scaleAddr, __ubuf__ int8_t* scaleBlkAddr,
                                                 uint16_t mSize, uint16_t scaleBlockN);

    // ---- Shape ----
    FlatQuantShapeInfo shape_;

    // ---- Pipe / UB ----
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> bufQueue_;
    AscendC::LocalTensor<bfloat16_t> xTensor_;
    AscendC::LocalTensor<int8_t> yTensor_;
    AscendC::LocalTensor<uint16_t> eMaxTensor_;
    AscendC::LocalTensor<int8_t> scaleTensor_;
    AscendC::LocalTensor<uint16_t> deQuantScaleTensor_;
    AscendC::LocalTensor<int8_t> scaleBlockTensor_;

    // ---- GM ----
    AscendC::GlobalTensor<int8_t> cGlobal_;
    AscendC::GlobalTensor<int8_t> scaleGlobal_;

    // ---- Problem / Params ----
    ProblemShape problemShape_;
    int64_t alignM_ = 0;
    float dstTypeMax_ = 0.0f;
    float invDstTypeMax_ = 0.0f;
    uint16_t addValueBit_ = 0;

    // ---- Events ----
    event_t eventIdVToMte3_;
    event_t eventIdMte3ToV_;
};

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::Init(
    Params const& params)
{
    cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.outGmAddr));
    scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.scaleGmAddr));
    problemShape_ = params.problemShape;

    shape_.m = AscendC::Te::Get<Gemm::MNK_M>(problemShape_);
    shape_.n = AscendC::Te::Get<Gemm::MNK_N>(problemShape_);
    shape_.k = AscendC::Te::Get<Gemm::MNK_B>(problemShape_);
    dstTypeMax_ = params.dstTypeMax;
    invDstTypeMax_ = params.invDstTypeMax;
    if (dstTypeMax_ == Constant::SIX_FLOAT) {
        addValueBit_ = Constant::ADD_VALUE_FOR_BF16_MAN1;
    } else if (dstTypeMax_ == Constant::SEVEN_FLOAT) {
        addValueBit_ = Constant::ADD_VALUE_FOR_BF16_MAN2;
    }

    shape_.mCeil = Gemm::CeilAlign(static_cast<int64_t>(shape_.m), static_cast<int64_t>(Constant::CEIL_SIZE));
    shape_.nCeil = Constant::VEC_N_LEN;
    alignM_ = Gemm::CeilDiv(static_cast<int64_t>(shape_.m * shape_.n), static_cast<int64_t>(Constant::VEC_N_LEN));

    pipe_.InitBuffer(bufQueue_, AscendC::TOTAL_UB_SIZE);
    xTensor_ = bufQueue_.Get<bfloat16_t>();
    yTensor_ = xTensor_[Constant::MN_SIZE].template ReinterpretCast<int8_t>();
    eMaxTensor_ = yTensor_[Constant::OUT_SIZE].template ReinterpretCast<uint16_t>();
    deQuantScaleTensor_ = eMaxTensor_[Constant::EMAX_SIZE];
    scaleTensor_ = deQuantScaleTensor_[Constant::EMAX_SIZE].template ReinterpretCast<int8_t>();
    scaleBlockTensor_ = scaleTensor_[Constant::EMAX_SIZE];

    eventIdVToMte3_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
    eventIdMte3ToV_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ClearDirtyData()
{
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.src0RepeatStride = Gemm::CeilAlign(static_cast<int64_t>(shape_.n),
                                              static_cast<int64_t>(Constant::CEIL_SIZE)) *
                              sizeof(DataTypeIn) / Constant::GROUP_SIZE;
    params.src1RepeatStride = 0;
    params.repeatTimes = shape_.m;
    uint64_t rvdCnt = 0Ul;
    AscendC::GatherMask(xTensor_, xTensor_, Constant::GATHER_PATTERN, true, shape_.n, params, rvdCnt);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ClearScaleTensor()
{
    AscendC::Duplicate(scaleTensor_, static_cast<int8_t>(0), Constant::EMAX_SIZE);
    AscendC::Duplicate(scaleBlockTensor_, static_cast<int8_t>(0), Constant::EMAX_SIZE);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::SaveTailVf(
    __ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* srcPtr, uint32_t count)
{
    AscendC::Reg::RegTensor<uint16_t> vReg;
    AscendC::Reg::UnalignRegForLoad u0;
    AscendC::Reg::LoadUnAlignPre(u0, srcPtr);
    AscendC::Reg::LoadUnAlign(vReg, u0, srcPtr);
    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<uint16_t>(count);
    AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_NORM_B16>(dstPtr, vReg, mask);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ClearTailVf(
    __ubuf__ uint16_t* dstPtr, uint32_t count)
{
    AscendC::Reg::RegTensor<uint16_t> zeroReg;
    AscendC::Reg::Duplicate(zeroReg, static_cast<uint16_t>(0));
    AscendC::Reg::UnalignRegForStore u1;

    constexpr uint32_t strideElems = Constant::STORE_UNALIGN_STRIDE_BYTES / sizeof(uint16_t);
    uint32_t loopCount = (count + strideElems - 1) / strideElems;
    for (uint32_t i = 0; i < loopCount; i++) {
        AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            dstPtr, zeroReg, u1, Constant::STORE_UNALIGN_STRIDE_BYTES);
    }
    AscendC::Reg::StoreUnAlignPost(dstPtr, u1, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::RestoreTailVf(
    __ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* srcPtr, uint32_t count)
{
    constexpr uint32_t strideElems = Constant::STORE_UNALIGN_STRIDE_BYTES / sizeof(uint16_t);
    uint32_t loopCount = (count + strideElems - 1) / strideElems;
    AscendC::Reg::UnalignRegForStore u1;
    for (uint32_t i = 0; i < loopCount; i++) {
        AscendC::Reg::RegTensor<uint16_t> vReg;
        AscendC::Reg::UnalignRegForLoad u0;
        __ubuf__ uint16_t* curSrcPtr = srcPtr + i * strideElems;
        AscendC::Reg::LoadUnAlignPre(u0, curSrcPtr);
        AscendC::Reg::LoadUnAlign(vReg, u0, curSrcPtr);
        AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            dstPtr, vReg, u1, Constant::STORE_UNALIGN_STRIDE_BYTES);
    }
    AscendC::Reg::StoreUnAlignPost(dstPtr, u1, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_,
                                              FusionOp_>::CopyOutputFromUbToGm(uint64_t offset,
                                                                               AscendC::LocalTensor<int8_t>& src)
{
    uint64_t alignedOffset = offset >> 1;
    copy_ubuf_to_gm_align_v2(cGlobal_[alignedOffset].GetPhyAddr(), (__ubuf__ void*)src.GetPhyAddr(), 0, 1,
                             static_cast<uint32_t>((shape_.m * shape_.n * sizeof(int8_t)) >> 1), 0, 0, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_,
                                              FusionOp_>::CopyScaleFromUbToGm(uint64_t offset,
                                                                              AscendC::LocalTensor<int8_t>& src)
{
    uint32_t blockCount = static_cast<uint32_t>(
        Gemm::CeilDiv(static_cast<uint64_t>(alignM_ * shape_.nCeil), Gemm::MXFP_DIVISOR_SIZE));
    copy_ubuf_to_gm_align_v2(scaleGlobal_[offset].GetPhyAddr(), (__ubuf__ void*)src.GetPhyAddr(), 0, blockCount,
                             Constant::BLOCK_SCALE, 0, Constant::BLOCK_SCALE, 32);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ComputeMxQuant(
    LocalTensor<bfloat16_t>& xTensor, LocalTensor<int8_t>& yTensor, LocalTensor<uint16_t>& eMaxTensor,
    LocalTensor<int8_t>& scaleTensor, LocalTensor<uint16_t>& deQuantScaleTensor, uint32_t totalDataInUB,
    uint64_t inputOffset)
{
    uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(DataTypeIn);
    uint16_t repeatCount = static_cast<uint16_t>(
        Gemm::CeilDiv(static_cast<uint64_t>(totalDataInUB), static_cast<uint64_t>(oneRepeatSize * 2)));
    uint16_t scaleNum = static_cast<uint16_t>(
        Gemm::CeilDiv(static_cast<uint64_t>(totalDataInUB), static_cast<uint64_t>(Constant::GROUP_SIZE)));
    uint16_t repeatScaleCount = static_cast<uint16_t>(
        Gemm::CeilDiv(static_cast<uint64_t>(scaleNum), static_cast<uint64_t>(oneRepeatSize)));
    uint16_t repeatScaleHalfCount = static_cast<uint16_t>(
        Gemm::CeilDiv(static_cast<uint64_t>(scaleNum), static_cast<uint64_t>(oneRepeatSize / 2)));

    __ubuf__ bfloat16_t* xAddr = (__ubuf__ bfloat16_t*)xTensor.GetPhyAddr() + inputOffset;
    __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)eMaxTensor.GetPhyAddr();
    if (dstTypeMax_ >= Constant::SIX_FLOAT && dstTypeMax_ <= Constant::TWELVE_FLOAT) {
        AscendC::VF_CALL<ExpMaxVfcuBLAS>(maxExpAddr, xAddr, totalDataInUB, repeatCount, oneRepeatSize);
    } else {
        AscendC::VF_CALL<ExpMaxVf>(maxExpAddr, xAddr, totalDataInUB, repeatCount, oneRepeatSize);
    }

    __ubuf__ uint16_t* deScaleAddr = (__ubuf__ uint16_t*)deQuantScaleTensor.GetPhyAddr();
    __ubuf__ uint16_t* scaleAddr = (__ubuf__ uint16_t*)scaleTensor.GetPhyAddr();
    if (dstTypeMax_ == Constant::ZERO_FLOAT) {
        AscendC::VF_CALL<ScaleVf>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleCount);
    } else if (dstTypeMax_ == Constant::SIX_FLOAT || dstTypeMax_ == Constant::SEVEN_FLOAT) {
        AscendC::VF_CALL<ScaleVfDynamic>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleCount, addValueBit_);
    } else {
        AscendC::VF_CALL<ScaleVfcuBLAS>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleHalfCount,
                                        invDstTypeMax_);
    }

    __ubuf__ int8_t* yAddr = (__ubuf__ int8_t*)yTensor.GetPhyAddr();
    AscendC::VF_CALL<QuantVf>(yAddr, xAddr, deScaleAddr, totalDataInUB, repeatCount);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ComputeTransLayout(
    LocalTensor<int8_t>& scaleTensor, LocalTensor<int8_t>& scaleBlockTensor, uint16_t m, uint16_t n)
{
    uint16_t scaleBlockN = Gemm::CeilDiv(static_cast<uint64_t>(n), static_cast<uint64_t>(Gemm::MXFP_DIVISOR_SIZE)) * 2;

    __ubuf__ int8_t* qscaleAddr = (__ubuf__ int8_t*)scaleTensor.GetPhyAddr();
    __ubuf__ int8_t* qscaleBlkAddr = (__ubuf__ int8_t*)scaleBlockTensor.GetPhyAddr();
    AscendC::VF_CALL<TransLayoutVf>(qscaleAddr, qscaleBlkAddr, m, scaleBlockN);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::Quant(
    uint64_t batchIdx, uint64_t iterIdx)
{
    int64_t mnSize = shape_.m * shape_.n;
    uint64_t yOffset = batchIdx * static_cast<uint64_t>(mnSize);
    uint64_t scaleOffset = batchIdx *
                           Gemm::CeilDiv(static_cast<uint64_t>(mnSize),
                                         static_cast<uint64_t>(Gemm::MXFP_DIVISOR_SIZE)) *
                           2;
    uint32_t totalDataInUB = static_cast<uint32_t>(mnSize);
    uint64_t inputOffset = iterIdx * totalDataInUB;
    ClearScaleTensor();

    if (shape_.n % 16 != 0) {
        ClearDirtyData();
    }

    uint32_t tailRemainder = totalDataInUB % Constant::GROUP_SIZE;
    if (tailRemainder != 0) {
        uint32_t tailSize = Constant::GROUP_SIZE - tailRemainder;
        __ubuf__ uint16_t* tailAddr = (__ubuf__ uint16_t*)xTensor_.GetPhyAddr() + inputOffset + totalDataInUB;
        __ubuf__ uint16_t* saveAddr = (__ubuf__ uint16_t*)scaleBlockTensor_.GetPhyAddr();
        AscendC::VF_CALL<SaveTailVf>(saveAddr, tailAddr, tailSize);
        AscendC::VF_CALL<ClearTailVf>(tailAddr, tailSize);
    }

    ComputeMxQuant(xTensor_, yTensor_, eMaxTensor_, scaleTensor_, deQuantScaleTensor_, totalDataInUB, inputOffset);

    if (tailRemainder != 0) {
        uint32_t tailSize = Constant::GROUP_SIZE - tailRemainder;
        __ubuf__ uint16_t* tailAddr = (__ubuf__ uint16_t*)xTensor_.GetPhyAddr() + inputOffset + totalDataInUB;
        __ubuf__ uint16_t* saveAddr = (__ubuf__ uint16_t*)scaleBlockTensor_.GetPhyAddr();
        AscendC::VF_CALL<RestoreTailVf>(tailAddr, saveAddr, tailSize);
    }

    ComputeTransLayout(scaleTensor_, scaleBlockTensor_, static_cast<uint16_t>(alignM_),
                       static_cast<uint16_t>(shape_.nCeil));
    AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
    AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);

    CopyOutputFromUbToGm(yOffset, yTensor_);
    CopyScaleFromUbToGm(scaleOffset, scaleBlockTensor_);
    AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
    AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__aicore__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::operator()(
    uint64_t startBatchIdx, uint64_t iterBatch)
{
    for (uint64_t iter = 0; iter < iterBatch; ++iter) {
        Quant(startBatchIdx + iter, iter);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ExpMaxVf(
    __ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* srcPtr, uint32_t count, uint16_t repeatTimes,
    uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<bfloat16_t> vSrcReg0;
    AscendC::Reg::RegTensor<bfloat16_t> vSrcReg1;
    AscendC::Reg::RegTensor<uint16_t> vExpExtract0;
    AscendC::Reg::RegTensor<uint16_t> vExpExtract1;
    AscendC::Reg::RegTensor<uint16_t> vdMaxExp;

    AscendC::Reg::RegTensor<uint16_t> expMaskBF16;
    AscendC::Reg::Duplicate(expMaskBF16, Constant::MAX_EXP_FOR_BF16);

    AscendC::Reg::MaskReg maskReg;
    AscendC::Reg::UnalignRegForStore u1;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; i++) {
        aReg = AscendC::Reg::CreateAddrReg<uint32_t>(i, oneRepeatSize);
        maskReg = AscendC::Reg::UpdateMask<bfloat16_t>(count);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vSrcReg0, vSrcReg1, srcPtr, aReg);
        AscendC::Reg::And(vExpExtract0, (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg0, expMaskBF16, maskReg);
        AscendC::Reg::And(vExpExtract1, (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg1, expMaskBF16, maskReg);
        AscendC::Reg::Max(vdMaxExp, vExpExtract0, vExpExtract1, maskReg);
        AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(vdMaxExp, vdMaxExp, maskReg);
        AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            dstPtr, vdMaxExp, u1, Constant::STORE_UNALIGN_STRIDE_BYTES);
    }
    AscendC::Reg::StoreUnAlignPost(dstPtr, u1, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ExpMaxVfcuBLAS(
    __ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* srcPtr, uint32_t count, uint16_t repeatTimes,
    uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<bfloat16_t> vSrcReg0;
    AscendC::Reg::RegTensor<bfloat16_t> vSrcReg1;
    AscendC::Reg::RegTensor<uint16_t> vdMaxExp;

    AscendC::Reg::RegTensor<uint16_t> absMask16Bit;
    AscendC::Reg::Duplicate(absMask16Bit, Constant::ABS_MASK_FOR_16BIT);

    AscendC::Reg::MaskReg maskReg;
    AscendC::Reg::UnalignRegForStore u1;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; i++) {
        aReg = AscendC::Reg::CreateAddrReg<uint32_t>(i, oneRepeatSize);
        maskReg = AscendC::Reg::UpdateMask<bfloat16_t>(count);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vSrcReg0, vSrcReg1, srcPtr, aReg);
        AscendC::Reg::And((AscendC::Reg::RegTensor<uint16_t>&)vSrcReg0, (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg0,
                          absMask16Bit, maskReg);
        AscendC::Reg::And((AscendC::Reg::RegTensor<uint16_t>&)vSrcReg1, (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg1,
                          absMask16Bit, maskReg);
        AscendC::Reg::Max(vdMaxExp, (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg0,
                          (AscendC::Reg::RegTensor<uint16_t>&)vSrcReg1, maskReg);

        AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(vdMaxExp, vdMaxExp, maskReg);
        AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
            dstPtr, vdMaxExp, u1, Constant::STORE_UNALIGN_STRIDE_BYTES);
    }
    AscendC::Reg::StoreUnAlignPost(dstPtr, u1, 0);
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ScaleVf(
    __ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr, __ubuf__ uint16_t* srcPtr, uint32_t scaleNum,
    uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
    AscendC::Reg::Duplicate(expMask, Constant::MAX_EXP_FOR_BF16);
    AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
    AscendC::Reg::RegTensor<bfloat16_t> vdExp0, vdExp1;
    AscendC::Reg::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
    AscendC::Reg::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
    AscendC::Reg::Duplicate(maxExpValue, Constant::FP4_E2M1_MAX_EXP);
    AscendC::Reg::Duplicate(scaleBias, Constant::BF16_EXP_BIAS);
    AscendC::Reg::Duplicate(fp8NanRegTensor, Constant::MAX_EXP_FOR_FP8);
    AscendC::Reg::Duplicate(zeroRegTensor, 0);
    AscendC::Reg::Duplicate(nanRegTensor, Constant::NAN_CUSTOMIZATION);

    AscendC::Reg::MaskReg invalidDataMask, specialDataMask;
    AscendC::Reg::Duplicate(specialExpRegTensor, Constant::SPECIAL_EXP_THRESHOLD);
    for (uint16_t i = 0; i < repeatTimes; i++) {
        preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(scaleNum);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, srcPtr, 128);
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
        AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::Reg::ShiftRights(scaleValue, sharedExp, Constant::SHR_NUM_FOR_BF16, preMaskScale);
        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK_B16>(dstPtr, scaleValue, 64, preMaskScale);
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);

        AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst2Ptr, halfScale, 128,
                                                                                        preMaskScale);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ScaleVfDynamic(
    __ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr, __ubuf__ uint16_t* srcPtr, uint32_t scaleNum,
    uint16_t repeatTimes, uint16_t addValueBit)
{
    AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
    AscendC::Reg::RegTensor<uint16_t> sharedExp;
    AscendC::Reg::RegTensor<uint16_t> scaleValue;
    AscendC::Reg::RegTensor<uint16_t> halfScale;
    AscendC::Reg::RegTensor<uint16_t> vdMaxExpAdd;
    AscendC::Reg::RegTensor<uint16_t> vdMaxExpOnly;

    AscendC::Reg::RegTensor<uint16_t> expMask;
    AscendC::Reg::Duplicate(expMask, Constant::MAX_EXP_FOR_BF16);
    AscendC::Reg::RegTensor<uint16_t> addValue;
    AscendC::Reg::Duplicate(addValue, addValueBit);
    AscendC::Reg::RegTensor<uint16_t> maxExpValue;
    AscendC::Reg::Duplicate(maxExpValue, Constant::FP4_E2M1_MAX_EXP);
    AscendC::Reg::RegTensor<uint16_t> scaleBias;
    AscendC::Reg::Duplicate(scaleBias, Constant::BF16_EXP_BIAS);
    AscendC::Reg::RegTensor<uint16_t> fp8NanRegTensor;
    AscendC::Reg::Duplicate(fp8NanRegTensor, Constant::MAX_EXP_FOR_FP8);
    AscendC::Reg::RegTensor<uint16_t> zeroRegTensor;
    AscendC::Reg::Duplicate(zeroRegTensor, 0);
    AscendC::Reg::RegTensor<uint16_t> nanRegTensor;
    AscendC::Reg::Duplicate(nanRegTensor, Constant::NAN_CUSTOMIZATION);
    AscendC::Reg::RegTensor<uint16_t> specialExpRegTensor;
    AscendC::Reg::Duplicate(specialExpRegTensor, Constant::SPECIAL_EXP_THRESHOLD);

    AscendC::Reg::MaskReg cmpResult;
    AscendC::Reg::MaskReg zeroMask;
    AscendC::Reg::MaskReg invalidDataMask;
    AscendC::Reg::MaskReg specialDataMask;
    AscendC::Reg::MaskReg preMaskScale;

    for (uint16_t i = 0; i < repeatTimes; i++) {
        preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(scaleNum);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, srcPtr,
                                                                                       Constant::vfLen16);
        AscendC::Reg::And(vdMaxExpOnly, vdMaxExp, expMask, preMaskScale);
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExpOnly, expMask, preMaskScale);
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExpOnly, zeroRegTensor, preMaskScale);
        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::LT>(invalidDataMask, vdMaxExpOnly, maxExpValue, preMaskScale);

        AscendC::Reg::Add(vdMaxExpAdd, vdMaxExp, addValue, preMaskScale);
        AscendC::Reg::And(vdMaxExpAdd, vdMaxExpAdd, expMask, preMaskScale);
        AscendC::Reg::Select<uint16_t>(vdMaxExpAdd, maxExpValue, vdMaxExpAdd, invalidDataMask);
        AscendC::Reg::Sub(sharedExp, vdMaxExpAdd, maxExpValue, preMaskScale);

        AscendC::Reg::ShiftRights(scaleValue, sharedExp, Constant::SHR_NUM_FOR_BF16, preMaskScale);
        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK_B16>(dstPtr, scaleValue, Constant::vfLen32,
                                                                         preMaskScale);

        AscendC::Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
        AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_NORM>(dst2Ptr, halfScale, Constant::vfLen16,
                                                                     preMaskScale);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::ScaleVfcuBLAS(
    __ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr, __ubuf__ uint16_t* srcPtr, uint32_t scaleNum,
    uint16_t repeatTimes, float invDstTypeMax)
{
    AscendC::Reg::RegTensor<uint16_t> max16;
    AscendC::Reg::RegTensor<uint32_t> max32;
    AscendC::Reg::RegTensor<uint32_t> exp32;
    AscendC::Reg::RegTensor<uint32_t> man32;
    AscendC::Reg::RegTensor<uint32_t> normalExp32;
    AscendC::Reg::RegTensor<uint32_t> expAddOne32;
    AscendC::Reg::RegTensor<uint32_t> extractExp;
    AscendC::Reg::RegTensor<uint16_t> expOut;
    AscendC::Reg::RegTensor<uint32_t> halfScale;
    AscendC::Reg::RegTensor<uint16_t> recExpOut;

    AscendC::Reg::RegTensor<uint32_t> manMaskFP32;
    AscendC::Reg::Duplicate(manMaskFP32, Constant::MAN_MASK_FLOAT);
    AscendC::Reg::RegTensor<uint32_t> expMask;
    AscendC::Reg::Duplicate(expMask, Constant::MAX_EXP_FOR_FP32);
    AscendC::Reg::RegTensor<uint32_t> zeroRegTensor32;
    AscendC::Reg::Duplicate(zeroRegTensor32, 0);
    AscendC::Reg::RegTensor<uint32_t> scaleBias;
    AscendC::Reg::Duplicate(scaleBias, Constant::FP32_EXP_BIAS_CUBLAS);
    AscendC::Reg::RegTensor<uint32_t> nanRegTensor;
    AscendC::Reg::Duplicate(nanRegTensor, Constant::NAN_CUSTOMIZATION_PACK);
    AscendC::Reg::RegTensor<uint32_t> fp4NanRegTensor;
    AscendC::Reg::Duplicate(fp4NanRegTensor, Constant::MAX_EXP_FOR_FP8_IN_FP32);
    AscendC::Reg::RegTensor<float> invMax;
    AscendC::Reg::Duplicate(invMax, invDstTypeMax);

    AscendC::Reg::MaskReg cmpResult;
    AscendC::Reg::MaskReg zeroMask;
    AscendC::Reg::MaskReg p0;
    AscendC::Reg::MaskReg p1;
    AscendC::Reg::MaskReg p2;
    AscendC::Reg::MaskReg preMaskScale;
    uint32_t maskCount = 64;
    AscendC::Reg::MaskReg dataMaskB16Half = AscendC::Reg::UpdateMask<uint16_t>(maskCount);

    static constexpr AscendC::Reg::CastTrait castTraitHalf2Float = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    for (uint16_t i = 0; i < repeatTimes; i++) {
        preMaskScale = AscendC::Reg::UpdateMask<uint32_t>(scaleNum);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_UNPACK_B16>(max16, srcPtr, Constant::vfLen32);

        AscendC::Reg::Cast<float, bfloat16_t, castTraitHalf2Float>(
            (AscendC::Reg::RegTensor<float>&)max32, (AscendC::Reg::RegTensor<bfloat16_t>&)max16, preMaskScale);
        AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
        AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);

        AscendC::Reg::Mul((AscendC::Reg::RegTensor<float>&)max32, (AscendC::Reg::RegTensor<float>&)max32, invMax,
                          preMaskScale);
        AscendC::Reg::ShiftRights(exp32, max32, Constant::SHR_NUM_FOR_FP32, preMaskScale);
        AscendC::Reg::And(man32, max32, manMaskFP32, preMaskScale);

        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::GT>(p0, exp32, Constant::NUMBER_ZERO, preMaskScale);
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::LT>(p1, exp32, Constant::NUMBER_TWO_FIVE_FOUR, preMaskScale);
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::GT>(p2, man32, Constant::NUMBER_ZERO, preMaskScale);
        AscendC::Reg::And(p0, p0, p1, preMaskScale);
        AscendC::Reg::And(p0, p0, p2, preMaskScale);

        AscendC::Reg::Adds(expAddOne32, exp32, 1, preMaskScale);
        AscendC::Reg::Select(extractExp, expAddOne32, exp32, p0);
        AscendC::Reg::Select<uint32_t>(extractExp, extractExp, fp4NanRegTensor, cmpResult);
        AscendC::Reg::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
        AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(expOut, extractExp);

        AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(
            dstPtr + i * Constant::SCALE_STORE_STRIDE, expOut, dataMaskB16Half);

        AscendC::Reg::ShiftLefts(extractExp, extractExp, Constant::SHR_NUM_FOR_BF16, preMaskScale);
        AscendC::Reg::Sub(halfScale, scaleBias, extractExp, preMaskScale);
        AscendC::Reg::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        AscendC::Reg::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
        AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(recExpOut, halfScale);

        AscendC::Reg::StoreAlign<uint16_t>(dst2Ptr + i * Constant::vfLen32, recExpOut, dataMaskB16Half);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::QuantVf(
    __ubuf__ int8_t* dstPtr, __ubuf__ bfloat16_t* srcPtr, __ubuf__ uint16_t* src2Ptr, uint32_t oneRepeatSize,
    uint16_t repeatTimes)
{
    AscendC::Reg::MaskReg dataMask1;
    AscendC::Reg::MaskReg dataMask2;
    AscendC::Reg::RegTensor<uint16_t> halfScaleForMul;
    AscendC::Reg::RegTensor<bfloat16_t> vdExp0;
    AscendC::Reg::RegTensor<bfloat16_t> vdExp1;
    AscendC::Reg::RegTensor<bfloat16_t> vdExp0Convert;
    AscendC::Reg::RegTensor<bfloat16_t> vdExp1Convert;

    AscendC::Reg::RegTensor<bfloat16_t> vdExp0BF16;
    AscendC::Reg::RegTensor<bfloat16_t> vdExp1Bf16;

    AscendC::Reg::RegTensor<fp4x2_e2m1_t> vdExp0FP4;
    AscendC::Reg::RegTensor<fp4x2_e2m1_t> vdExp1FP4;

    AscendC::Reg::RegTensor<bfloat16_t> vdBf16Exp0FP4;
    AscendC::Reg::RegTensor<bfloat16_t> vdBf16Exp1FP4;

    AscendC::Reg::AddrReg aReg;
    static constexpr AscendC::Reg::CastTrait castTrait = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};
    for (uint16_t i = 0; i < repeatTimes; i++) {
        aReg = AscendC::Reg::CreateAddrReg<uint16_t>(i, oneRepeatSize);
        dataMask1 = AscendC::Reg::UpdateMask<bfloat16_t>(oneRepeatSize);
        dataMask2 = AscendC::Reg::UpdateMask<bfloat16_t>(oneRepeatSize);

        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcPtr, 256);
        AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, src2Ptr, 8);
        AscendC::Reg::Mul(vdExp0, vdExp0, (AscendC::Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
        AscendC::Reg::Mul(vdExp1, vdExp1, (AscendC::Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
        AscendC::Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
        AscendC::Reg::Cast<fp4x2_e2m1_t, bfloat16_t, castTrait>(vdExp0FP4, vdExp0, dataMask1);
        AscendC::Reg::Cast<fp4x2_e2m1_t, bfloat16_t, castTrait>(vdExp1FP4, vdExp1, dataMask2);

        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            dstPtr, (AscendC::Reg::RegTensor<int8_t>&)vdExp0FP4, 64, dataMask1);
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_PACK4_B32>(
            dstPtr, (AscendC::Reg::RegTensor<int8_t>&)vdExp1FP4, 64, dataMask2);
    }
}

template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_, typename FusionOp_>
__simd_vf__ inline void BlockEpilogueFlatQuant<DataTypeIn_, DataTypeOut_, DataTypeScale_, FusionOp_>::TransLayoutVf(
    __ubuf__ int8_t* scaleAddr, __ubuf__ int8_t* scaleBlkAddr, uint16_t mSize, uint16_t scaleBlockN)
{
    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        uint32_t eleNum = scaleBlockN;
        AscendC::Reg::MaskReg maskScaleN = AscendC::Reg::UpdateMask<int8_t>(eleNum);
        AscendC::Reg::RegTensor<int8_t> vReg0;
        AscendC::Reg::UnalignRegForLoad u0, u1;
        auto srcUb = scaleAddr + mIdx * scaleBlockN;
        AscendC::Reg::LoadUnAlignPre(u0, srcUb);
        AscendC::Reg::LoadUnAlign(vReg0, u0, srcUb);
        auto dstUb = scaleBlkAddr + mIdx * 32;
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_NORM_B8>(dstUb, vReg0, maskScaleN);
    }
}

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
