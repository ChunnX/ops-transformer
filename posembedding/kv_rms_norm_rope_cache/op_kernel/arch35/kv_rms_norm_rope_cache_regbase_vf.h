/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_rms_norm_rope_cache_regbase_vf.h
 * \brief Register-level vector functions of KvRmsNormRopeCache on Ascend 950 (arch35).
 *
 * All __simd_vf__ entry functions operate on UB addresses and scalar arguments;
 * GM transfers, synchronization, and dispatch are handled by the host classes.
 * Shared Reg helpers and constants used across the three host files are also
 * defined here as __simd_callee__ free functions.
 *
 * Naming convention:
 *   - RmsNormXxxSingleVF / DoubleVF: rSize fits one / two vector lanes.
 *   - RmsNormXxxSmallColVF: small-column direct path (actualCol <= 2 * VL).
 *   - FullLoad* / Recompute* prefix: contributed by the host file of the same name.
 */

#ifndef KV_RMS_NORM_ROPE_CACHE_REGBASE_VF_H_
#define KV_RMS_NORM_ROPE_CACHE_REGBASE_VF_H_

#include "kernel_operator.h"
#include "kv_rms_norm_rope_cache_regbase_platform.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

// ===== 常量（自 kv_rms_norm_rope_cache_regbase_base.h 移入，避免同 namespace 重复定义）=====

constexpr static uint32_t VL_FP32 = static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);

constexpr static AscendC::Reg::CastTrait CAST_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

static constexpr AscendC::Reg::CastTrait CAST_FP16_TO_INT8 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                              AscendC::Reg::MaskMergeMode::ZEROING,
                                                              AscendC::RoundMode::CAST_TRUNC};

constexpr static AscendC::Reg::CastTrait CAST_FP32_TO_FP16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::Reg::CastTrait CAST_INT32_TO_FP32 = {
    AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::Reg::CastTrait CAST_FP32_TO_INT16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

static constexpr AscendC::Reg::CastTrait CAST_INT16_TO_FP16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};

constexpr AscendC::Reg::CastTrait CAST_FP32_TO_FLOAT8 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                         AscendC::Reg::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait CAST_FP32_TO_HIFLOAT8 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_ROUND};

constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static float CONST_MINUS_ONE = -1.0;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_THREE = 3;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_FIVE = 5;
constexpr static int64_t CONST_SIX = 6;
constexpr static int64_t CONST_SEVEN = 7;
constexpr static int64_t CONST_EIGHT = 8;
constexpr static int64_t CONST_SIXTY_THREE = 63;

// ===== __simd_callee__ 嵌套 helper（仅被 VF 体内调用）=====

template <typename T>
__simd_callee__ inline void LoadTensorForDtypeT(__ubuf__ T* input, AscendC::Reg::RegTensor<float>& dst,
                                                AscendC::Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        AscendC::Reg::RegTensor<half> xFp16;
        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xFp16,
                                                                               ((__ubuf__ half*)(input) + (offset)));
        Cast<float, half, CAST_B16_TO_B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        AscendC::Reg::RegTensor<bfloat16_t> xBf16;
        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
            xBf16, ((__ubuf__ bfloat16_t*)(input) + (offset)));
        Cast<float, bfloat16_t, CAST_B16_TO_B32>(dst, xBf16, preg);
    } else {
        LoadAlign(dst, ((__ubuf__ float*)(input) + (offset)));
    }
}

template <typename T>
__simd_callee__ inline void StoreTensorForDtypeTOut(__ubuf__ T* dst, AscendC::Reg::RegTensor<float>& src,
                                                    AscendC::Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::Reg::RegTensor<T> xFp16;
        Cast<T, float, CAST_FP32_TO_FP16>(xFp16, src, preg);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

// store 非对齐的float32类型的src(寄存器)数据到output(ub)中
// output数据类型支持int8, hifloat8, float8_e5m2, float8_e4m3fn, bfloat16, float16, float32
template <typename T>
__simd_callee__ inline void StoreUnAlignOneTensor(__ubuf__ T*& output, Reg::RegTensor<float>& src,
                                                  Reg::UnalignRegForStore& uValue, Reg::MaskReg& preg,
                                                  uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::RegTensor<half> xFp16Pack;
        Cast<half, float, CAST_FP32_TO_FP16>(xFp16, src, preg);
        Pack((Reg::RegTensor<uint16_t>&)xFp16Pack, (Reg::RegTensor<uint32_t>&)xFp16);
        StoreUnAlign(output, xFp16Pack, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::RegTensor<bfloat16_t> xBf16Pack;
        Cast<bfloat16_t, float, CAST_FP32_TO_FP16>(xBf16, src, preg);
        Pack((Reg::RegTensor<uint16_t>&)xBf16Pack, (Reg::RegTensor<uint32_t>&)xBf16);
        StoreUnAlign(output, xBf16Pack, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, int8_t>::value) {
        // 量化为int8
        AscendC::Reg::RegTensor<half> tmpHalf;
        AscendC::Reg::RegTensor<int16_t> tmpInt16;
        AscendC::Reg::RegTensor<int8_t> quantInt8;
        AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmpInt16, src, preg);
        AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmpHalf, tmpInt16, preg);
        AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantInt8, tmpHalf, preg);
        Pack((Reg::RegTensor<uint16_t>&)tmpInt16, (Reg::RegTensor<uint32_t>&)quantInt8);
        Pack((Reg::RegTensor<uint8_t>&)quantInt8, (Reg::RegTensor<uint16_t>&)tmpInt16);
        StoreUnAlign(output, quantInt8, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, hifloat8_t>::value) {
        // 量化为hifloat8
        AscendC::Reg::RegTensor<hifloat8_t> quantHifloat8Pack;
        AscendC::Reg::RegTensor<hifloat8_t> quantHifloat8;
        AscendC::Reg::Cast<hifloat8_t, float, CAST_FP32_TO_HIFLOAT8>(quantHifloat8, src, preg);
        Pack((Reg::RegTensor<uint16_t>&)quantHifloat8Pack, (Reg::RegTensor<uint32_t>&)quantHifloat8);
        Pack((Reg::RegTensor<uint8_t>&)quantHifloat8, (Reg::RegTensor<uint16_t>&)quantHifloat8Pack);
        StoreUnAlign(output, quantHifloat8, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, fp8_e5m2_t>::value) {
        // 量化fp8_e5m2
        AscendC::Reg::RegTensor<fp8_e5m2_t> quantFloat8E5m2Pack;
        AscendC::Reg::RegTensor<fp8_e5m2_t> quantFloat8E5m2;
        AscendC::Reg::Cast<fp8_e5m2_t, float, CAST_FP32_TO_FLOAT8>(quantFloat8E5m2, src, preg);
        Pack((Reg::RegTensor<uint16_t>&)quantFloat8E5m2Pack, (Reg::RegTensor<uint32_t>&)quantFloat8E5m2);
        Pack((Reg::RegTensor<uint8_t>&)quantFloat8E5m2, (Reg::RegTensor<uint16_t>&)quantFloat8E5m2Pack);
        StoreUnAlign(output, quantFloat8E5m2, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
        // 量化为fp8_e4m3fn
        AscendC::Reg::RegTensor<fp8_e4m3fn_t> quantFloat8E4m3fnPack;
        AscendC::Reg::RegTensor<fp8_e4m3fn_t> quantFloat8E4m3fn;
        AscendC::Reg::Cast<fp8_e4m3fn_t, float, CAST_FP32_TO_FLOAT8>(quantFloat8E4m3fn, src, preg);
        Pack((Reg::RegTensor<uint16_t>&)quantFloat8E4m3fnPack, (Reg::RegTensor<uint32_t>&)quantFloat8E4m3fn);
        Pack((Reg::RegTensor<uint8_t>&)quantFloat8E4m3fn, (Reg::RegTensor<uint16_t>&)quantFloat8E4m3fnPack);
        StoreUnAlign(output, quantFloat8E4m3fn, uValue, postUpdateStride);
    } else {
        StoreUnAlign(output, src, uValue, postUpdateStride);
    }
}

template <typename T>
__simd_callee__ inline void LoadDataAndCast2Fp32(AscendC::Reg::RegTensor<float>& dst, __ubuf__ T* xAddr,
                                                 AscendC::Reg::MaskReg& mask)
{
    if constexpr (!IsSameType<T, float>::value) {
        AscendC::Reg::RegTensor<T> vregB16;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregB16, xAddr);
        AscendC::Reg::Cast<float, T, CAST_B16_TO_B32>(dst, vregB16, mask);
    } else {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(dst, xAddr);
    }
}

template <typename T>
__simd_callee__ inline void StoreDataAndCastFromFp32(__ubuf__ T*& dst, AscendC::Reg::RegTensor<float>& src,
                                                     AscendC::Reg::UnalignRegForStore& uReg,
                                                     AscendC::Reg::MaskReg& mask, uint32_t postUpdateStride)
{
    if constexpr (!IsSameType<T, float>::value) {
        AscendC::Reg::RegTensor<T> vregB16, vregB16Pack;
        AscendC::Reg::Cast<T, float, CAST_FP32_TO_FP16>(vregB16, src, mask);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst, vregB16, mask);
    } else {
        AscendC::Reg::StoreAlign(dst, src, mask);
    }
}

template <typename T>
__simd_callee__ inline void StoreQuantAndCastFromFp32(__ubuf__ T*& dst, AscendC::Reg::RegTensor<float>& src,
                                                      AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<T> vregQuant;
    if constexpr (IsSameType<T, int8_t>::value) {
        AscendC::Reg::RegTensor<int16_t> vregInt16;
        AscendC::Reg::RegTensor<half> vregHalf;
        AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(vregInt16, src, mask);
        AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(vregHalf, vregInt16, mask);
        AscendC::Reg::Cast<T, half, CAST_FP16_TO_INT8>(vregQuant, vregHalf, mask);
    } else if constexpr (IsSameType<T, hifloat8_t>::value) {
        AscendC::Reg::Cast<T, float, CAST_FP32_TO_HIFLOAT8>(vregQuant, src, mask);
    } else if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value) {
        AscendC::Reg::Cast<T, float, CAST_FP32_TO_FLOAT8>(vregQuant, src, mask);
    }
    AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK4_B32>(dst, vregQuant, mask);
}

template <typename T_KV>
__simd_callee__ inline void RopeBasicComputeVF(
    __ubuf__ T_KV* x, AscendC::Reg::RegTensor<float>& x_0, AscendC::Reg::RegTensor<float>& x_1,
    AscendC::Reg::RegTensor<float>& cos_0, AscendC::Reg::RegTensor<float>& cos_1, AscendC::Reg::RegTensor<float>& sin_0,
    AscendC::Reg::RegTensor<float>& sin_1, __ubuf__ T_KV* realCos, __ubuf__ T_KV* imgCos, __ubuf__ T_KV* realSin,
    __ubuf__ T_KV* imgSin, __ubuf__ float* ws, uint32_t xOffset, uint32_t cosSinOffset, uint16_t rowId, uint16_t colId,
    AscendC::Reg::MaskReg mask, AscendC::Reg::MaskReg maskAll)
{
    // cast to float32
    LoadTensorForDtypeT<T_KV>(x, x_0, maskAll, colId * CONST_TWO * VL_FP32 + rowId * xOffset);
    LoadTensorForDtypeT<T_KV>(x, x_1, maskAll, (colId * CONST_TWO + CONST_ONE) * VL_FP32 + rowId * xOffset);
    StoreTensorForDtypeTOut<float>(ws, x_0, maskAll, 0);
    StoreTensorForDtypeTOut<float>(ws, x_1, maskAll, VL_FP32);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    // get x
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(x_0, x_1, ((__ubuf__ float*)(ws)));
    LoadTensorForDtypeT<T_KV>(realCos, cos_0, mask, colId * VL_FP32 + rowId * cosSinOffset);
    LoadTensorForDtypeT<T_KV>(imgCos, cos_1, mask, colId * VL_FP32 + rowId * cosSinOffset);
    LoadTensorForDtypeT<T_KV>(realSin, sin_0, mask, colId * VL_FP32 + rowId * cosSinOffset);
    LoadTensorForDtypeT<T_KV>(imgSin, sin_1, mask, colId * VL_FP32 + rowId * cosSinOffset);
    AscendC::Reg::Mul(cos_0, x_0, cos_0, mask);
    AscendC::Reg::Mul(cos_1, x_1, cos_1, mask);
    AscendC::Reg::Muls(x_1, x_1, -1, mask);
    AscendC::Reg::Mul(sin_0, x_1, sin_0, mask);
    AscendC::Reg::Mul(sin_1, x_0, sin_1, mask);
    AscendC::Reg::Add(sin_0, cos_0, sin_0, mask);
    AscendC::Reg::Add(sin_1, cos_1, sin_1, mask);
}

// ===== base.h 的 VF 入口（宿主：KvRmsNormRopeCacheRegbase 各成员函数）=====

template <typename T_KV>
__simd_vf__ inline void RmsNormPostWithMulSingleVF(__ubuf__ float* dst, __ubuf__ T_KV* x, __ubuf__ T_KV* gamma,
                                                   uint16_t loopTimes, int64_t rSize, int64_t stride, float reciprocal,
                                                   float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 行
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadTensorForDtypeT<T_KV>(x, reg0, pMask, i * stride);
        LoadTensorForDtypeT<T_KV>(gamma, reg1, pMask, 0);
        AscendC::Reg::Mul(reg2, reg0, reg0, pMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg2, reg2, pMask);
        AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
        AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
        AscendC::Reg::Sqrt(reg5, reg4, pFull);
        Duplicate(reg5, reg5, pFull);
        AscendC::Reg::Div(reg6, reg0, reg5, pMask);
        AscendC::Reg::Mul(reg7, reg1, reg6, pMask);
        StoreTensorForDtypeTOut<float>(dst, reg7, pMask, i * stride);
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormPostWithMulDoubleVF(__ubuf__ float* dst, __ubuf__ T_KV* x, __ubuf__ T_KV* x_1,
                                                   __ubuf__ T_KV* gamma, __ubuf__ T_KV* gamma_1, uint16_t loopTimes,
                                                   int64_t rSize, int64_t stride, float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg0_1, reg1_1, reg2, reg2_1;
    AscendC::Reg::RegTensor<float> reg3, reg4, reg5, reg6, reg7, reg8, reg9;
    AscendC::Reg::RegTensor<T_V_CACHE> regqunat12, regqunat12_1;
    AscendC::Reg::RegTensor<T_KV> reg12, reg12_1;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 行循环
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadTensorForDtypeT<T_KV>(x, reg0, pFull, i * stride);
        LoadTensorForDtypeT<T_KV>(x_1, reg0_1, pMask, i * stride);
        LoadTensorForDtypeT<T_KV>(gamma, reg1, pFull, 0);
        LoadTensorForDtypeT<T_KV>(gamma_1, reg1_1, pMask, 0);
        AscendC::Reg::Mul(reg2, reg0, reg0, pFull);
        AscendC::Reg::Mul(reg2_1, reg0_1, reg0_1, pMask);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg2_1, reg2, reg2_1, pMask);
        AscendC::Reg::Move<float, AscendC::Reg::MaskMergeMode::MERGING>(reg2, reg2_1, pMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg2, reg2, pFull);

        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
        AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
        AscendC::Reg::Sqrt(reg5, reg4, pFull);
        Duplicate(reg5, reg5, pFull);
        // 因为输入分为两段，所以这里也要分为两段
        AscendC::Reg::Div(reg6, reg0, reg5, pFull);
        AscendC::Reg::Mul(reg7, reg1, reg6, pFull);
        StoreTensorForDtypeTOut<float>(dst, reg7, pFull, i * stride);
        // 剩余的一部分
        AscendC::Reg::Div(reg8, reg0_1, reg5, pMask);
        AscendC::Reg::Mul(reg9, reg1_1, reg8, pMask);
        StoreTensorForDtypeTOut<float>(dst, reg9, pMask, i * stride + VL_FP32);
    }
}

template <typename T_KV>
__simd_vf__ inline void RmsNormBasicComputeImplVF(__ubuf__ float* wsBase, __ubuf__ T_KV* foldSrcX0A,
                                                  __ubuf__ T_KV* foldSrcX0B, __ubuf__ T_KV* tailSrcX0A,
                                                  __ubuf__ T_KV* tailSrcX0B, __ubuf__ T_KV* unFoldX0,
                                                  uint16_t outerLoopTimes, uint16_t mainFoldLoopTimes,
                                                  uint16_t tailFoldLoopTimes, uint32_t tailFoldElemCount,
                                                  uint16_t unFoldLoopTimes, uint32_t outerLoopStride,
                                                  uint32_t innerLoopStride, int64_t outerLoopDstStride)
{
    __ubuf__ float* dst = wsBase;
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::UnalignRegForStore UReg;
    // 每一行循环一次
    for (uint16_t i = 0; i < outerLoopTimes; ++i) {
        dst = wsBase + i * outerLoopDstStride;
        // 完全能折叠
        for (uint16_t j = 0; j < mainFoldLoopTimes; ++j) {
            AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg2_1;
            LoadTensorForDtypeT<T_KV>(foldSrcX0A, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
            LoadTensorForDtypeT<T_KV>(foldSrcX0B, reg1, pFull, i * outerLoopStride + j * innerLoopStride);

            AscendC::Reg::Mul(reg2, reg0, reg0, pFull);
            AscendC::Reg::Mul(reg2_1, reg1, reg1, pFull);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg2, reg2, reg2_1, pFull);
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg2, reg2, pFull);
            AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, reg2, UReg, 1);
        }
        // 一个能折叠，另外一个短了
        for (uint16_t j = 0; j < tailFoldLoopTimes; ++j) {
            uint32_t count = tailFoldElemCount;
            AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg2_1;
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            LoadTensorForDtypeT<T_KV>(tailSrcX0A, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
            LoadTensorForDtypeT<T_KV>(tailSrcX0B, reg1, pMask, i * outerLoopStride + j * innerLoopStride);
            AscendC::Reg::Mul(reg2, reg0, reg0, pFull);
            AscendC::Reg::Mul(reg2_1, reg1, reg1, pMask);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg2_1, reg2, reg2_1, pMask);
            AscendC::Reg::Move<float, AscendC::Reg::MaskMergeMode::MERGING>(reg2, reg2_1, pMask);
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg2, reg2, pFull);
            AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, reg2, UReg, 1);
        }
        // 无法折叠
        for (uint16_t j = 0; j < unFoldLoopTimes; ++j) {
            AscendC::Reg::RegTensor<float> reg0, reg1;
            LoadTensorForDtypeT<T_KV>(unFoldX0, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
            AscendC::Reg::Mul(reg1, reg0, reg0, pFull);
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg1, reg1, pFull);
            AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, reg1, UReg, 1);
        }
        AscendC::Reg::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
    }
}

template <typename T_KV>
__simd_vf__ inline void RmsNormPostSingleVF(__ubuf__ T_KV* dst, __ubuf__ T_KV* x, __ubuf__ T_KV* gamma,
                                            __ubuf__ float* sumTmp, uint16_t loopTimes, uint16_t rLoopCount,
                                            uint32_t oriR, uint32_t oriRAligned, int64_t stride, int64_t rSize,
                                            float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 每行循环一次
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmp + i * stride);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg1, reg0, pMask);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        AscendC::Reg::Muls(reg2, reg1, reciprocal, pFull);
        AscendC::Reg::Adds(reg3, reg2, epsilon, pFull);
        AscendC::Reg::Sqrt(reg4, reg3, pFull);
        Duplicate(reg5, reg4, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            // gramma只有一行，每次都是取一样的数值
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
        }
    }
}

template <typename T_KV>
__simd_vf__ inline void RmsNormPostDoubleVF(__ubuf__ T_KV* dst, __ubuf__ float* sumTmpA, __ubuf__ float* sumTmpB,
                                            __ubuf__ T_KV* x, __ubuf__ T_KV* gamma, uint16_t loopTimes,
                                            uint16_t rLoopCount, uint32_t oriR, uint32_t oriRAligned, int64_t stride,
                                            int64_t rSize, float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 行
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmpA + i * stride);
        LoadAlign(reg1, (__ubuf__ float*)sumTmpB + i * stride);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
        AscendC::Reg::Move<float, AscendC::Reg::MaskMergeMode::MERGING>(reg0, reg1, pMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg2, reg0, pFull);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        // reducesum后的结果就一个数，所以这几个都是一个数在做计算
        AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
        AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
        AscendC::Reg::Sqrt(reg5, reg4, pFull);
        Duplicate(reg5, reg5, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormAsymQuantWithKvPostSingleVF(
    __ubuf__ T_KV* dst, __ubuf__ float* vScale, __ubuf__ float* vOffset, __ubuf__ T_V_CACHE* vQuant, __ubuf__ T_KV* x,
    __ubuf__ T_KV* gamma, __ubuf__ float* sumTmp, uint16_t loopTimes, uint16_t rLoopCount, uint32_t oriR,
    uint32_t oriRAligned, uint32_t quantOutOffset, int64_t stride, int64_t rSize, float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::RegTensor<float> scale, offset;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;

    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 每行循环一次
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmp + i * stride);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg1, reg0, pMask);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        AscendC::Reg::Muls(reg2, reg1, reciprocal, pFull);
        AscendC::Reg::Adds(reg3, reg2, epsilon, pFull);
        AscendC::Reg::Sqrt(reg4, reg3, pFull);
        Duplicate(reg5, reg4, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            // gramma只有一行，每次都是取一样的数值
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
            LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
            LoadTensorForDtypeT<float>(vOffset, offset, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(scale, reg8, scale, maskOri);
            AscendC::Reg::Add(offset, offset, scale, maskOri);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, maskOri);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, maskOri);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormAsymQuantWithKvPostDoubleVF(__ubuf__ T_KV* dst, __ubuf__ float* vScale,
                                                           __ubuf__ float* vOffset, __ubuf__ T_V_CACHE* vQuant,
                                                           __ubuf__ float* sumTmpA, __ubuf__ float* sumTmpB,
                                                           __ubuf__ T_KV* x, __ubuf__ T_KV* gamma, uint16_t loopTimes,
                                                           uint16_t rLoopCount, uint32_t oriR, uint32_t oriRAligned,
                                                           uint32_t quantOutOffset, int64_t stride, int64_t rSize,
                                                           float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::RegTensor<float> scale, offset;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 行
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmpA + i * stride);
        LoadAlign(reg1, (__ubuf__ float*)sumTmpB + i * stride);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
        AscendC::Reg::Move<float, AscendC::Reg::MaskMergeMode::MERGING>(reg0, reg1, pMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg2, reg0, pFull);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        // reducesum后的结果就一个数，所以这几个都是一个数在做计算
        AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
        AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
        AscendC::Reg::Sqrt(reg5, reg4, pFull);
        Duplicate(reg5, reg5, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
            LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
            LoadTensorForDtypeT<float>(vOffset, offset, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(scale, reg8, scale, maskOri);
            AscendC::Reg::Add(offset, offset, scale, maskOri);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, maskOri);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, maskOri);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormSymQuantWithKvPostSingleVF(__ubuf__ T_KV* dst, __ubuf__ T_KV* x, __ubuf__ T_KV* gamma,
                                                          __ubuf__ float* sumTmp, __ubuf__ float* vScale,
                                                          __ubuf__ T_V_CACHE* vQuant, uint16_t loopTimes,
                                                          uint16_t rLoopCount, uint32_t oriR, uint32_t oriRAligned,
                                                          uint32_t quantOutOffset, int64_t stride, int64_t rSize,
                                                          float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::RegTensor<float> scale;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;

    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 每行循环一次
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmp + i * stride);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg1, reg0, pMask);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        AscendC::Reg::Muls(reg2, reg1, reciprocal, pFull);
        AscendC::Reg::Adds(reg3, reg2, epsilon, pFull);
        AscendC::Reg::Sqrt(reg4, reg3, pFull);
        Duplicate(reg5, reg4, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            // gramma只有一行，每次都是取一样的数值
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);

            LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(scale, reg8, scale, maskOri);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, maskOri);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, maskOri);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormSymQuantWithKvPostDoubleVF(
    __ubuf__ T_KV* dst, __ubuf__ float* sumTmpA, __ubuf__ float* sumTmpB, __ubuf__ T_KV* x, __ubuf__ T_KV* gamma,
    __ubuf__ float* vScale, __ubuf__ T_V_CACHE* vQuant, uint16_t loopTimes, uint16_t rLoopCount, uint32_t oriR,
    uint32_t oriRAligned, uint32_t quantOutOffset, int64_t stride, int64_t rSize, float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::RegTensor<float> scale;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 行
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmpA + i * stride);
        LoadAlign(reg1, (__ubuf__ float*)sumTmpB + i * stride);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
        AscendC::Reg::Move<float, AscendC::Reg::MaskMergeMode::MERGING>(reg0, reg1, pMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg2, reg0, pFull);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        // reducesum后的结果就一个数，所以这几个都是一个数在做计算
        AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
        AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
        AscendC::Reg::Sqrt(reg5, reg4, pFull);
        Duplicate(reg5, reg5, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);

            LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(scale, reg8, scale, maskOri);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, maskOri);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, maskOri);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormAsymQuantPostSingleVF(__ubuf__ T_KV* x, __ubuf__ T_KV* gamma, __ubuf__ float* sumTmp,
                                                     __ubuf__ float* vScale, __ubuf__ float* vOffset,
                                                     __ubuf__ T_V_CACHE* vQuant, uint16_t loopTimes,
                                                     uint16_t rLoopCount, uint32_t oriR, uint32_t oriRAligned,
                                                     uint32_t quantOutOffset, int64_t stride, int64_t rSize,
                                                     float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::RegTensor<float> scale, offset;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 每行循环一次
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmp + i * stride);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg1, reg0, pMask);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        AscendC::Reg::Muls(reg2, reg1, reciprocal, pFull);
        AscendC::Reg::Adds(reg3, reg2, epsilon, pFull);
        AscendC::Reg::Sqrt(reg4, reg3, pFull);
        Duplicate(reg5, reg4, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            // gramma只有一行，每次都是取一样的数值
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
            LoadTensorForDtypeT<float>(vOffset, offset, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(scale, reg8, scale, maskOri);
            AscendC::Reg::Add(offset, offset, scale, maskOri);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, maskOri);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, maskOri);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormAsymQuantPostDoubleVF(__ubuf__ float* sumTmpA, __ubuf__ float* sumTmpB, __ubuf__ T_KV* x,
                                                     __ubuf__ T_KV* gamma, __ubuf__ float* vScale,
                                                     __ubuf__ float* vOffset, __ubuf__ T_V_CACHE* vQuant,
                                                     uint16_t loopTimes, uint16_t rLoopCount, uint32_t oriR,
                                                     uint32_t oriRAligned, uint32_t quantOutOffset, int64_t stride,
                                                     int64_t rSize, float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::RegTensor<float> scale, offset;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 行
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmpA + i * stride);
        LoadAlign(reg1, (__ubuf__ float*)sumTmpB + i * stride);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
        AscendC::Reg::Move<float, AscendC::Reg::MaskMergeMode::MERGING>(reg0, reg1, pMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg2, reg0, pFull);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        // reducesum后的结果就一个数，所以这几个都是一个数在做计算
        AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
        AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
        AscendC::Reg::Sqrt(reg5, reg4, pFull);
        Duplicate(reg5, reg5, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
            LoadTensorForDtypeT<float>(vOffset, offset, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(scale, reg8, scale, maskOri);
            AscendC::Reg::Add(offset, offset, scale, maskOri);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, maskOri);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, maskOri);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormSymQuantPostSingleVF(__ubuf__ T_KV* x, __ubuf__ T_KV* gamma, __ubuf__ float* sumTmp,
                                                    __ubuf__ float* vScale, __ubuf__ T_V_CACHE* vQuant,
                                                    uint16_t loopTimes, uint16_t rLoopCount, uint32_t oriR,
                                                    uint32_t oriRAligned, uint32_t quantOutOffset, int64_t stride,
                                                    int64_t rSize, float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::RegTensor<float> scale;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 每行循环一次
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmp + i * stride);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg1, reg0, pMask);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        AscendC::Reg::Muls(reg2, reg1, reciprocal, pFull);
        AscendC::Reg::Adds(reg3, reg2, epsilon, pFull);
        AscendC::Reg::Sqrt(reg4, reg3, pFull);
        Duplicate(reg5, reg4, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            // gramma只有一行，每次都是取一样的数值
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(scale, reg8, scale, maskOri);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, maskOri);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, maskOri);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormSymQuantPostDoubleVF(__ubuf__ float* sumTmpA, __ubuf__ float* sumTmpB, __ubuf__ T_KV* x,
                                                    __ubuf__ T_KV* gamma, __ubuf__ float* vScale,
                                                    __ubuf__ T_V_CACHE* vQuant, uint16_t loopTimes, uint16_t rLoopCount,
                                                    uint32_t oriR, uint32_t oriRAligned, uint32_t quantOutOffset,
                                                    int64_t stride, int64_t rSize, float reciprocal, float epsilon)
{
    uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
    AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
    AscendC::Reg::RegTensor<float> scale;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
    AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskOri;
    // 行
    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadAlign(reg0, (__ubuf__ float*)sumTmpA + i * stride);
        LoadAlign(reg1, (__ubuf__ float*)sumTmpB + i * stride);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
        AscendC::Reg::Move<float, AscendC::Reg::MaskMergeMode::MERGING>(reg0, reg1, pMask);
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reg2, reg0, pFull);
        // Calc: xSum = xSum * reciprocal
        // Calc: xSum = xSum + epsilon
        // Calc: xSum = sqrt(xSum)
        // reducesum后的结果就一个数，所以这几个都是一个数在做计算
        AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
        AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
        AscendC::Reg::Sqrt(reg5, reg4, pFull);
        Duplicate(reg5, reg5, pFull);
        uint32_t sreg0 = oriR;
        for (uint16_t j = 0; j < rLoopCount; ++j) {
            maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
            uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
            LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
            AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
            LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
            LoadTensorForDtypeT<float>(vScale, scale, maskOri, j * VL_FP32);
            AscendC::Reg::Mul(scale, reg8, scale, maskOri);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, maskOri);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, maskOri);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, maskOri);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, maskOri);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + j * VL_FP32 + i * quantOutOffset, quant, maskOri);
        }
    }
}

template <typename T_KV>
__simd_vf__ inline void RmsNormSmallColCastVF(__ubuf__ T_KV* dst, __ubuf__ float* ws, uint16_t row,
                                              uint16_t colLoopCount, uint32_t actualCol, uint32_t alignedCol)
{
    AscendC::Reg::RegTensor<float> x;
    AscendC::Reg::MaskReg mask;
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = actualCol;
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
            StoreTensorForDtypeTOut<T_KV>(dst, x, mask, i * VL_FP32 + rowId * alignedCol);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormAsymQuantWithKvSmallColVF(__ubuf__ T_KV* dst, __ubuf__ float* ws, __ubuf__ float* vScale,
                                                         __ubuf__ float* vOffset, __ubuf__ T_V_CACHE* vQuant,
                                                         uint16_t row, uint16_t colLoopCount, uint32_t actualCol,
                                                         uint32_t alignedCol, uint32_t quantOutOffset)
{
    AscendC::Reg::RegTensor<float> x, out, scale, offset;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg mask;
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = actualCol;
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
            StoreTensorForDtypeTOut<T_KV>(dst, x, mask, i * VL_FP32 + rowId * alignedCol);
            LoadTensorForDtypeT<float>(vScale, scale, mask, i * VL_FP32);
            LoadTensorForDtypeT<float>(vOffset, offset, mask, i * VL_FP32);
            AscendC::Reg::Mul(scale, x, scale, mask);
            AscendC::Reg::Add(offset, offset, scale, mask);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, mask);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, mask);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, mask);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, mask);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, mask);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + i * VL_FP32 + rowId * quantOutOffset, quant, mask);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormSymQuantWithKvSmallColVF(__ubuf__ T_KV* dst, __ubuf__ float* ws, __ubuf__ float* vScale,
                                                        __ubuf__ T_V_CACHE* vQuant, uint16_t row, uint16_t colLoopCount,
                                                        uint32_t actualCol, uint32_t alignedCol,
                                                        uint32_t quantOutOffset)
{
    AscendC::Reg::RegTensor<float> x, scale, offset;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg mask;
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = actualCol;
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
            StoreTensorForDtypeTOut<T_KV>(dst, x, mask, i * VL_FP32 + rowId * alignedCol);
            LoadTensorForDtypeT<float>(vScale, scale, mask, i * VL_FP32);
            AscendC::Reg::Mul(scale, x, scale, mask);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, mask);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, mask);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, mask);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, mask);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, mask);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + i * VL_FP32 + rowId * quantOutOffset, quant, mask);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormAsymQuantSmallColVF(__ubuf__ float* ws, __ubuf__ float* vScale, __ubuf__ float* vOffset,
                                                   __ubuf__ T_V_CACHE* vQuant, uint16_t row, uint16_t colLoopCount,
                                                   uint32_t actualCol, uint32_t alignedCol, uint32_t quantOutOffset)
{
    AscendC::Reg::RegTensor<float> x, out, scale, offset;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg mask;
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = actualCol;
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
            LoadTensorForDtypeT<float>(vScale, scale, mask, i * VL_FP32);
            LoadTensorForDtypeT<float>(vOffset, offset, mask, i * VL_FP32);
            AscendC::Reg::Mul(scale, x, scale, mask);
            AscendC::Reg::Add(offset, offset, scale, mask);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, offset, mask);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, mask);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, mask);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, offset, mask);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, offset, mask);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + i * VL_FP32 + rowId * quantOutOffset, quant, mask);
        }
    }
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void RmsNormSymQuantSmallColVF(__ubuf__ float* ws, __ubuf__ float* vScale,
                                                  __ubuf__ T_V_CACHE* vQuant, uint16_t row, uint16_t colLoopCount,
                                                  uint32_t actualCol, uint32_t alignedCol, uint32_t quantOutOffset)
{
    AscendC::Reg::RegTensor<float> x, out, scale, offset;
    AscendC::Reg::RegTensor<half> tmp;
    AscendC::Reg::RegTensor<int16_t> tmp_int16;
    AscendC::Reg::RegTensor<T_V_CACHE> quant;
    AscendC::Reg::MaskReg mask;
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = actualCol;
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
            LoadTensorForDtypeT<float>(vScale, scale, mask, i * VL_FP32);
            AscendC::Reg::Mul(scale, x, scale, mask);

            if constexpr (IsSameType<T_V_CACHE, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmp_int16, scale, mask);
                AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmp, tmp_int16, mask);
                AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(quant, tmp, mask);
            } else if constexpr (IsSameType<T_V_CACHE, hifloat8_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_HIFLOAT8>(quant, scale, mask);
            } else if constexpr (IsSameType<T_V_CACHE, fp8_e5m2_t>::value ||
                                 IsSameType<T_V_CACHE, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<T_V_CACHE, float, CAST_FP32_TO_FLOAT8>(quant, scale, mask);
            }
            AscendC::Reg::StoreAlign<T_V_CACHE, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                vQuant + i * VL_FP32 + rowId * quantOutOffset, quant, mask);
        }
    }
}

// ===== full_load.h 的 VF 入口（宿主：KvRmsNormRopeCacheRegbaseFullLoad 各成员函数）=====

template <typename T_KV, typename T_K_CACHE>
__simd_vf__ inline void FullLoadRopeAsymQuantWithKvComputeVF(
    __ubuf__ T_KV* outFront, __ubuf__ T_KV* outBack, __ubuf__ T_K_CACHE* quantFront, __ubuf__ T_K_CACHE* quantBack,
    __ubuf__ T_KV* x, __ubuf__ T_KV* realCos, __ubuf__ T_KV* imgCos, __ubuf__ T_KV* realSin, __ubuf__ T_KV* imgSin,
    __ubuf__ float* realKScale, __ubuf__ float* imgKScale, __ubuf__ float* realKOffset, __ubuf__ float* imgKOffset,
    __ubuf__ float* ws, uint16_t row, uint16_t colLoopCount, uint32_t colCosSin, uint32_t xOffset,
    uint32_t cosSinOffset, uint32_t quantOutOffset, uint16_t kVFTail)
{
    AscendC::Reg::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1, offset_0, offset_1;
    AscendC::Reg::UnalignRegForStore UReg0, UReg1, UReg2, UReg3;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = colCosSin;
        __ubuf__ T_KV* out0 = outFront + rowId * xOffset;
        __ubuf__ T_KV* out1 = outBack + rowId * xOffset;
        __ubuf__ T_K_CACHE* out2 = quantFront + rowId * quantOutOffset;
        __ubuf__ T_K_CACHE* out3 = quantBack + rowId * quantOutOffset;
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                     xOffset, cosSinOffset, rowId, i, mask, maskAll);
            StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
            StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
            LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
            LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
            AscendC::Reg::Mul(scale_0, sin_0, scale_0, mask);
            AscendC::Reg::Mul(scale_1, sin_1, scale_1, mask);
            LoadTensorForDtypeT(realKOffset, offset_0, mask, i * VL_FP32);
            LoadTensorForDtypeT(imgKOffset, offset_1, mask, i * VL_FP32);
            AscendC::Reg::Add(offset_0, offset_0, scale_0, mask);
            AscendC::Reg::Add(offset_1, offset_1, scale_1, mask);
            StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, VL_FP32);
            StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, VL_FP32);
        }
        mask = AscendC::Reg::UpdateMask<float>(sreg);
        RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                                 cosSinOffset, rowId, colLoopCount, mask, maskAll);
        StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, kVFTail);
        StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, kVFTail);
        LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
        LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
        AscendC::Reg::Mul(scale_0, sin_0, scale_0, mask);
        AscendC::Reg::Mul(scale_1, sin_1, scale_1, mask);
        LoadTensorForDtypeT(realKOffset, offset_0, mask, colLoopCount * VL_FP32);
        LoadTensorForDtypeT(imgKOffset, offset_1, mask, colLoopCount * VL_FP32);
        AscendC::Reg::Add(offset_0, offset_0, scale_0, mask);
        AscendC::Reg::Add(offset_1, offset_1, scale_1, mask);
        StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, kVFTail);
        StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, kVFTail);
        AscendC::Reg::StoreUnAlignPost(out0, UReg0, 0);
        AscendC::Reg::StoreUnAlignPost(out1, UReg1, 0);
        AscendC::Reg::StoreUnAlignPost(out2, UReg2, 0);
        AscendC::Reg::StoreUnAlignPost(out3, UReg3, 0);
    }
}

template <typename T_KV, typename T_K_CACHE>
__simd_vf__ inline void FullLoadRopeSymQuantWithKvComputeVF(
    __ubuf__ T_KV* outFront, __ubuf__ T_KV* outBack, __ubuf__ T_K_CACHE* quantFront, __ubuf__ T_K_CACHE* quantBack,
    __ubuf__ T_KV* x, __ubuf__ T_KV* realCos, __ubuf__ T_KV* imgCos, __ubuf__ T_KV* realSin, __ubuf__ T_KV* imgSin,
    __ubuf__ float* realKScale, __ubuf__ float* imgKScale, __ubuf__ float* ws, uint16_t row, uint16_t colLoopCount,
    uint32_t colCosSin, uint32_t xOffset, uint32_t cosSinOffset, uint32_t quantOutOffset, uint16_t kVFTail)
{
    AscendC::Reg::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1;
    AscendC::Reg::UnalignRegForStore UReg0, UReg1, UReg2, UReg3;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = colCosSin;
        __ubuf__ T_KV* out0 = outFront + rowId * xOffset;
        __ubuf__ T_KV* out1 = outBack + rowId * xOffset;
        __ubuf__ T_K_CACHE* out2 = quantFront + rowId * quantOutOffset;
        __ubuf__ T_K_CACHE* out3 = quantBack + rowId * quantOutOffset;
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                     xOffset, cosSinOffset, rowId, i, mask, maskAll);
            StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
            StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
            LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
            LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
            AscendC::Reg::Mul(scale_0, sin_0, scale_0, mask);
            AscendC::Reg::Mul(scale_1, sin_1, scale_1, mask);
            StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, VL_FP32);
            StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, VL_FP32);
        }
        mask = AscendC::Reg::UpdateMask<float>(sreg);
        RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                                 cosSinOffset, rowId, colLoopCount, mask, maskAll);
        StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, kVFTail);
        StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, kVFTail);
        LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
        LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
        AscendC::Reg::Mul(scale_0, sin_0, scale_0, mask);
        AscendC::Reg::Mul(scale_1, sin_1, scale_1, mask);
        StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, kVFTail);
        StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, kVFTail);
        AscendC::Reg::StoreUnAlignPost(out0, UReg0, 0);
        AscendC::Reg::StoreUnAlignPost(out1, UReg1, 0);
        AscendC::Reg::StoreUnAlignPost(out2, UReg2, 0);
        AscendC::Reg::StoreUnAlignPost(out3, UReg3, 0);
    }
}

template <typename T_KV, typename T_K_CACHE>
__simd_vf__ inline void FullLoadRopeAsymQuantComputeVF(__ubuf__ T_K_CACHE* quantFront, __ubuf__ T_K_CACHE* quantBack,
                                                       __ubuf__ T_KV* x, __ubuf__ T_KV* realCos, __ubuf__ T_KV* imgCos,
                                                       __ubuf__ T_KV* realSin, __ubuf__ T_KV* imgSin,
                                                       __ubuf__ float* realKScale, __ubuf__ float* imgKScale,
                                                       __ubuf__ float* realKOffset, __ubuf__ float* imgKOffset,
                                                       __ubuf__ float* ws, uint16_t row, uint16_t colLoopCount,
                                                       uint32_t colCosSin, uint32_t xOffset, uint32_t cosSinOffset,
                                                       uint32_t quantOutOffset, uint16_t kVFTail)
{
    AscendC::Reg::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1, offset_0, offset_1;
    AscendC::Reg::UnalignRegForStore UReg2, UReg3;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = colCosSin;
        __ubuf__ T_K_CACHE* out2 = quantFront + rowId * quantOutOffset;
        __ubuf__ T_K_CACHE* out3 = quantBack + rowId * quantOutOffset;
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                     xOffset, cosSinOffset, rowId, i, mask, maskAll);
            LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
            LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
            AscendC::Reg::Mul(scale_0, sin_0, scale_0, mask);
            AscendC::Reg::Mul(scale_1, sin_1, scale_1, mask);
            LoadTensorForDtypeT(realKOffset, offset_0, mask, i * VL_FP32);
            LoadTensorForDtypeT(imgKOffset, offset_1, mask, i * VL_FP32);
            AscendC::Reg::Add(offset_0, offset_0, scale_0, mask);
            AscendC::Reg::Add(offset_1, offset_1, scale_1, mask);
            StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, VL_FP32);
            StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, VL_FP32);
        }
        mask = AscendC::Reg::UpdateMask<float>(sreg);
        RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                                 cosSinOffset, rowId, colLoopCount, mask, maskAll);
        LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
        LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
        AscendC::Reg::Mul(scale_0, sin_0, scale_0, mask);
        AscendC::Reg::Mul(scale_1, sin_1, scale_1, mask);
        LoadTensorForDtypeT(realKOffset, offset_0, mask, colLoopCount * VL_FP32);
        LoadTensorForDtypeT(imgKOffset, offset_1, mask, colLoopCount * VL_FP32);
        AscendC::Reg::Add(offset_0, offset_0, scale_0, mask);
        AscendC::Reg::Add(offset_1, offset_1, scale_1, mask);
        StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, kVFTail);
        StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, kVFTail);
        AscendC::Reg::StoreUnAlignPost(out2, UReg2, 0);
        AscendC::Reg::StoreUnAlignPost(out3, UReg3, 0);
    }
}

template <typename T_KV, typename T_K_CACHE>
__simd_vf__ inline void FullLoadRopeSymQuantComputeVF(__ubuf__ T_K_CACHE* quantFront, __ubuf__ T_K_CACHE* quantBack,
                                                      __ubuf__ T_KV* x, __ubuf__ T_KV* realCos, __ubuf__ T_KV* imgCos,
                                                      __ubuf__ T_KV* realSin, __ubuf__ T_KV* imgSin,
                                                      __ubuf__ float* realKScale, __ubuf__ float* imgKScale,
                                                      __ubuf__ float* ws, uint16_t row, uint16_t colLoopCount,
                                                      uint32_t colCosSin, uint32_t xOffset, uint32_t cosSinOffset,
                                                      uint32_t quantOutOffset, uint16_t kVFTail)
{
    AscendC::Reg::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1;
    AscendC::Reg::UnalignRegForStore UReg2, UReg3;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = colCosSin;
        __ubuf__ T_K_CACHE* out2 = quantFront + rowId * quantOutOffset;
        __ubuf__ T_K_CACHE* out3 = quantBack + rowId * quantOutOffset;
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                     xOffset, cosSinOffset, rowId, i, mask, maskAll);
            LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
            LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
            AscendC::Reg::Mul(scale_0, sin_0, scale_0, mask);
            AscendC::Reg::Mul(scale_1, sin_1, scale_1, mask);
            StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, VL_FP32);
            StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, VL_FP32);
        }
        mask = AscendC::Reg::UpdateMask<float>(sreg);
        RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                                 cosSinOffset, rowId, colLoopCount, mask, maskAll);
        LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
        LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
        AscendC::Reg::Mul(scale_0, sin_0, scale_0, mask);
        AscendC::Reg::Mul(scale_1, sin_1, scale_1, mask);
        StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, kVFTail);
        StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, kVFTail);
        AscendC::Reg::StoreUnAlignPost(out2, UReg2, 0);
        AscendC::Reg::StoreUnAlignPost(out3, UReg3, 0);
    }
}

template <typename T_KV>
__simd_vf__ inline void FullLoadRopeComputeVF(__ubuf__ T_KV* outFront, __ubuf__ T_KV* outBack, __ubuf__ T_KV* x,
                                              __ubuf__ T_KV* realCos, __ubuf__ T_KV* imgCos, __ubuf__ T_KV* realSin,
                                              __ubuf__ T_KV* imgSin, __ubuf__ float* ws, uint16_t row,
                                              uint16_t colLoopCount, uint32_t colCosSin, uint32_t xOffset,
                                              uint32_t cosSinOffset, uint16_t kVFTail)
{
    AscendC::Reg::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1;
    AscendC::Reg::UnalignRegForStore UReg0;
    AscendC::Reg::UnalignRegForStore UReg1;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    for (uint16_t rowId = 0; rowId < row; rowId++) {
        uint32_t sreg = colCosSin;
        // 跨行地址重刷新
        __ubuf__ T_KV* out0 = outFront + rowId * xOffset;
        __ubuf__ T_KV* out1 = outBack + rowId * xOffset;
        // 整块
        for (uint16_t i = 0; i < colLoopCount; i++) {
            mask = AscendC::Reg::UpdateMask<float>(sreg);
            RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws,
                                     xOffset, cosSinOffset, rowId, i, mask, maskAll);
            StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
            StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
        }
        // 尾块
        mask = AscendC::Reg::UpdateMask<float>(sreg);
        RopeBasicComputeVF<T_KV>(x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                                 cosSinOffset, rowId, colLoopCount, mask, maskAll);
        StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, kVFTail);
        StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, kVFTail);
        AscendC::Reg::StoreUnAlignPost(out0, UReg0, 0);
        AscendC::Reg::StoreUnAlignPost(out1, UReg1, 0);
    }
}

// ===== recompute.h 的 VF 入口（宿主：KvRmsNormRopeCacheRegbaseRecompute 各成员函数）=====

template <typename T_KV>
__simd_vf__ inline void CastPowComputeVF(__ubuf__ float* xFp32Ptr, __ubuf__ T_KV* xPtr, uint32_t ubFactor,
                                         uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> vreg1, vreg2;
    AscendC::Reg::MaskReg mask;

    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto xAddr = xPtr + j * VL_FP32;
        auto xFp32Addr = xFp32Ptr + j * VL_FP32;
        LoadDataAndCast2Fp32<T_KV>(vreg1, xAddr, mask);
        AscendC::Reg::Mul(vreg2, vreg1, vreg1, mask);
        AscendC::Reg::StoreAlign(xFp32Addr, vreg2, mask);
    }

    // 尾VL计算
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
    auto xFp32Addr = xFp32Ptr + repeatTimesFloor * VL_FP32;
    LoadDataAndCast2Fp32<T_KV>(vreg1, xAddr, mask);
    AscendC::Reg::Mul(vreg2, vreg1, vreg1, mask);
    AscendC::Reg::StoreAlign(xFp32Addr, vreg2, mask);
}

__simd_vf__ inline void FoldBlockComputeVF(__ubuf__ float* x1Ptr, __ubuf__ float* x2Ptr, uint32_t ubFactor,
                                           uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> vreg1, vreg2, vreg3;
    AscendC::Reg::MaskReg mask;

    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto x1Addr = x1Ptr + j * VL_FP32;
        auto x2Addr = x2Ptr + j * VL_FP32;
        LoadDataAndCast2Fp32<float>(vreg1, x1Addr, mask);
        LoadDataAndCast2Fp32<float>(vreg2, x2Addr, mask);
        AscendC::Reg::Add(vreg3, vreg1, vreg2, mask);
        AscendC::Reg::StoreAlign(x1Addr, vreg3, mask);
    }

    // 尾VL计算
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    auto x1Addr = x1Ptr + repeatTimesFloor * VL_FP32;
    auto x2Addr = x2Ptr + repeatTimesFloor * VL_FP32;
    LoadDataAndCast2Fp32<float>(vreg1, x1Addr, mask);
    LoadDataAndCast2Fp32<float>(vreg2, x2Addr, mask);
    AscendC::Reg::Add(vreg3, vreg1, vreg2, mask);
    AscendC::Reg::StoreAlign(x1Addr, vreg3, mask);
}

__simd_vf__ inline void UpdateCacheComputeVF(__ubuf__ float* dst, __ubuf__ float* cache, __ubuf__ float* src,
                                             uint16_t outerLoopTimes, uint16_t innerLoopTimes, uint32_t outerLoopStride,
                                             uint32_t innerLoopStride, int64_t count)
{
    uint32_t sreg = static_cast<uint32_t>(count);
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg pMask;
    for (uint16_t i = 0; i < outerLoopTimes; ++i) {
        pMask = AscendC::Reg::UpdateMask<float>(sreg);
        AscendC::Reg::LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride);
        for (uint16_t j = 0; j < innerLoopTimes; ++j) {
            AscendC::Reg::LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride);
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
        }
        AscendC::Reg::StoreAlign((__ubuf__ float*)cache + i * outerLoopStride, aReg, pMask);
    }
}

template <typename T_KV>
__simd_vf__ inline void CalculateVOutComputeVF(__ubuf__ T_KV* yPtr, __ubuf__ T_KV* xPtr, __ubuf__ T_KV* gammaPtr,
                                               __ubuf__ float* xSumPtr, uint32_t ubFactor, float reciprocal,
                                               float epsilon, uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
    AscendC::Reg::RegTensor<T_KV> vregY;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;

    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
    AscendC::Reg::Muls(sumReg, sumReg, reciprocal, mask);
    AscendC::Reg::Adds(sumReg, sumReg, epsilon, mask);
    AscendC::Reg::Sqrt(sumReg, sumReg, mask);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto xAddr = xPtr + j * VL_FP32;
        auto yAddr = yPtr + j * VL_FP32;
        auto gammaAddr = gammaPtr + j * VL_FP32;
        LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
        LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
        AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
        AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);
        StoreDataAndCastFromFp32<T_KV>(yAddr, vregTmp, uReg, mask, width);
    }

    // 尾VL
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
    auto yAddr = yPtr + repeatTimesFloor * VL_FP32;
    auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
    LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
    LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
    AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
    AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);
    StoreDataAndCastFromFp32<T_KV>(yAddr, vregTmp, uReg, mask, width);
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void CalculateVOutAsymQuantWithKvComputeVF(__ubuf__ T_KV* outPtr, __ubuf__ T_KV* xPtr,
                                                              __ubuf__ T_KV* gammaPtr, __ubuf__ float* xSumPtr,
                                                              __ubuf__ float* vScalePtr, __ubuf__ float* vOffsetPtr,
                                                              __ubuf__ T_V_CACHE* vQuantPtr, uint32_t ubFactor,
                                                              float reciprocal, float epsilon,
                                                              uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
    AscendC::Reg::RegTensor<float> vregScale, vregOffset;
    AscendC::Reg::RegTensor<half> vregHalf;
    AscendC::Reg::RegTensor<int16_t> vregInt16;
    AscendC::Reg::RegTensor<T_V_CACHE> vregQuant;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;

    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
    AscendC::Reg::Muls(sumReg, sumReg, reciprocal, mask);
    AscendC::Reg::Adds(sumReg, sumReg, epsilon, mask);
    AscendC::Reg::Sqrt(sumReg, sumReg, mask);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto xAddr = xPtr + j * VL_FP32;
        auto gammaAddr = gammaPtr + j * VL_FP32;
        auto vScaleAddr = vScalePtr + j * VL_FP32;
        auto vOffsetAddr = vOffsetPtr + j * VL_FP32;
        auto outAddr = outPtr + j * VL_FP32;
        auto vQuantAddr = vQuantPtr + j * VL_FP32;
        LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
        LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
        AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
        AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);
        StoreDataAndCastFromFp32<T_KV>(outAddr, vregTmp, uReg, mask, width);

        LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
        LoadDataAndCast2Fp32<float>(vregOffset, vOffsetAddr, mask);
        AscendC::Reg::Mul(vregTmp, vregTmp, vregScale, mask);
        AscendC::Reg::Add(vregTmp, vregTmp, vregOffset, mask);
        StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
    }

    // 尾VL
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
    auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
    auto vScaleAddr = vScalePtr + repeatTimesFloor * VL_FP32;
    auto vOffsetAddr = vOffsetPtr + repeatTimesFloor * VL_FP32;
    auto outAddr = outPtr + repeatTimesFloor * VL_FP32;
    auto vQuantAddr = vQuantPtr + repeatTimesFloor * VL_FP32;
    LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
    LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
    AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
    AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);
    StoreDataAndCastFromFp32<T_KV>(outAddr, vregTmp, uReg, mask, width);

    LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
    LoadDataAndCast2Fp32<float>(vregOffset, vOffsetAddr, mask);
    AscendC::Reg::Mul(vregTmp, vregTmp, vregScale, mask);
    AscendC::Reg::Add(vregTmp, vregTmp, vregOffset, mask);
    StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void CalculateVOutSymQuantWithKvComputeVF(__ubuf__ T_KV* outPtr, __ubuf__ T_KV* xPtr,
                                                             __ubuf__ T_KV* gammaPtr, __ubuf__ float* xSumPtr,
                                                             __ubuf__ float* vScalePtr, __ubuf__ T_V_CACHE* vQuantPtr,
                                                             uint32_t ubFactor, float reciprocal, float epsilon,
                                                             uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
    AscendC::Reg::RegTensor<float> vregScale;
    AscendC::Reg::RegTensor<half> vregHalf;
    AscendC::Reg::RegTensor<int16_t> vregInt16;
    AscendC::Reg::RegTensor<T_V_CACHE> vregQuant;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;

    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
    AscendC::Reg::Muls(sumReg, sumReg, reciprocal, mask);
    AscendC::Reg::Adds(sumReg, sumReg, epsilon, mask);
    AscendC::Reg::Sqrt(sumReg, sumReg, mask);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto xAddr = xPtr + j * VL_FP32;
        auto gammaAddr = gammaPtr + j * VL_FP32;
        auto vScaleAddr = vScalePtr + j * VL_FP32;
        auto outAddr = outPtr + j * VL_FP32;
        auto vQuantAddr = vQuantPtr + j * VL_FP32;
        LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
        LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
        AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
        AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);
        StoreDataAndCastFromFp32<T_KV>(outAddr, vregTmp, uReg, mask, width);

        LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
        AscendC::Reg::Mul(vregTmp, vregTmp, vregScale, mask);
        StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
    }

    // 尾VL
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
    auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
    auto vScaleAddr = vScalePtr + repeatTimesFloor * VL_FP32;
    auto outAddr = outPtr + repeatTimesFloor * VL_FP32;
    auto vQuantAddr = vQuantPtr + repeatTimesFloor * VL_FP32;
    LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
    LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
    AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
    AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);
    StoreDataAndCastFromFp32<T_KV>(outAddr, vregTmp, uReg, mask, width);

    LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
    AscendC::Reg::Mul(vregTmp, vregTmp, vregScale, mask);
    StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void CalculateVOutAsymQuantComputeVF(__ubuf__ T_KV* xPtr, __ubuf__ T_KV* gammaPtr,
                                                        __ubuf__ float* xSumPtr, __ubuf__ float* vScalePtr,
                                                        __ubuf__ float* vOffsetPtr, __ubuf__ T_V_CACHE* vQuantPtr,
                                                        uint32_t ubFactor, float reciprocal, float epsilon,
                                                        uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
    AscendC::Reg::RegTensor<float> vregScale, vregOffset;
    AscendC::Reg::RegTensor<half> vregHalf;
    AscendC::Reg::RegTensor<int16_t> vregInt16;
    AscendC::Reg::RegTensor<T_V_CACHE> vregQuant;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;

    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
    AscendC::Reg::Muls(sumReg, sumReg, reciprocal, mask);
    AscendC::Reg::Adds(sumReg, sumReg, epsilon, mask);
    AscendC::Reg::Sqrt(sumReg, sumReg, mask);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto xAddr = xPtr + j * VL_FP32;
        auto gammaAddr = gammaPtr + j * VL_FP32;
        auto vScaleAddr = vScalePtr + j * VL_FP32;
        auto vOffsetAddr = vOffsetPtr + j * VL_FP32;
        auto vQuantAddr = vQuantPtr + j * VL_FP32;
        LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
        LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
        AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
        AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);

        LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
        LoadDataAndCast2Fp32<float>(vregOffset, vOffsetAddr, mask);
        AscendC::Reg::Mul(vregTmp, vregTmp, vregScale, mask);
        AscendC::Reg::Add(vregTmp, vregTmp, vregOffset, mask);
        StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
    }

    // 尾VL
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
    auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
    auto vScaleAddr = vScalePtr + repeatTimesFloor * VL_FP32;
    auto vOffsetAddr = vOffsetPtr + repeatTimesFloor * VL_FP32;
    auto vQuantAddr = vQuantPtr + repeatTimesFloor * VL_FP32;
    LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
    LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
    AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
    AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);

    LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
    LoadDataAndCast2Fp32<float>(vregOffset, vOffsetAddr, mask);
    AscendC::Reg::Mul(vregTmp, vregTmp, vregScale, mask);
    AscendC::Reg::Add(vregTmp, vregTmp, vregOffset, mask);
    StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
}

template <typename T_KV, typename T_V_CACHE>
__simd_vf__ inline void CalculateVOutSymQuantComputeVF(__ubuf__ T_V_CACHE* vQuantPtr, __ubuf__ T_KV* xPtr,
                                                       __ubuf__ T_KV* gammaPtr, __ubuf__ float* xSumPtr,
                                                       __ubuf__ float* vScalePtr, uint32_t ubFactor, float reciprocal,
                                                       float epsilon, uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> sumReg, vregX, vregGamma, vregTmp;
    AscendC::Reg::RegTensor<float> vregScale;
    AscendC::Reg::RegTensor<half> vregHalf;
    AscendC::Reg::RegTensor<int16_t> vregInt16;
    AscendC::Reg::RegTensor<T_V_CACHE> vregQuant;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;

    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
    AscendC::Reg::Muls(sumReg, sumReg, reciprocal, mask);
    AscendC::Reg::Adds(sumReg, sumReg, epsilon, mask);
    AscendC::Reg::Sqrt(sumReg, sumReg, mask);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto xAddr = xPtr + j * VL_FP32;
        auto gammaAddr = gammaPtr + j * VL_FP32;
        auto vScaleAddr = vScalePtr + j * VL_FP32;
        auto vQuantAddr = vQuantPtr + j * VL_FP32;
        LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
        LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
        AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
        AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);

        LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
        AscendC::Reg::Mul(vregTmp, vregTmp, vregScale, mask);
        StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
    }

    // 尾VL
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    auto xAddr = xPtr + repeatTimesFloor * VL_FP32;
    auto gammaAddr = gammaPtr + repeatTimesFloor * VL_FP32;
    auto vScaleAddr = vScalePtr + repeatTimesFloor * VL_FP32;
    auto vQuantAddr = vQuantPtr + repeatTimesFloor * VL_FP32;
    LoadDataAndCast2Fp32<T_KV>(vregX, xAddr, mask);
    LoadDataAndCast2Fp32<T_KV>(vregGamma, gammaAddr, mask);
    AscendC::Reg::Div(vregTmp, vregX, sumReg, mask);
    AscendC::Reg::Mul(vregTmp, vregTmp, vregGamma, mask);

    LoadDataAndCast2Fp32<float>(vregScale, vScaleAddr, mask);
    AscendC::Reg::Mul(vregTmp, vregTmp, vregScale, mask);
    StoreQuantAndCastFromFp32<T_V_CACHE>(vQuantAddr, vregTmp, mask);
}

template <typename T_KV>
__simd_vf__ inline void RecomputeRopeComputeVF(__ubuf__ T_KV* outPtr1, __ubuf__ T_KV* outPtr2, __ubuf__ T_KV* ropePtr,
                                               __ubuf__ T_KV* cosPtr1, __ubuf__ T_KV* cosPtr2, __ubuf__ T_KV* sinPtr1,
                                               __ubuf__ T_KV* sinPtr2, __ubuf__ float* tmpBufferPtr, uint32_t ubFactor,
                                               uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32, vregSin2Fp32;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;
    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
        auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
        auto outAddr1 = outPtr1 + j * VL_FP32;
        auto outAddr2 = outPtr2 + j * VL_FP32;
        auto cosAddr1 = cosPtr1 + j * VL_FP32;
        auto cosAddr2 = cosPtr2 + j * VL_FP32;
        auto sinAddr1 = sinPtr1 + j * VL_FP32;
        auto sinAddr2 = sinPtr2 + j * VL_FP32;

        LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                                ((__ubuf__ float*)(tmpBufferPtr)));

        LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

        AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
        AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
        AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

        StoreDataAndCastFromFp32<T_KV>(outAddr1, vregCos1Fp32, uReg, mask, width);
        StoreDataAndCastFromFp32<T_KV>(outAddr2, vregCos2Fp32, uReg, mask, width);
    }

    // 尾VL计算范式
    auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
    auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
    auto outAddr1 = outPtr1 + repeatTimesFloor * VL_FP32;
    auto outAddr2 = outPtr2 + repeatTimesFloor * VL_FP32;
    auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
    auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
    auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
    auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;

    LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                            ((__ubuf__ float*)(tmpBufferPtr)));

    LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

    AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
    AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
    AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    StoreDataAndCastFromFp32<T_KV>(outAddr1, vregCos1Fp32, uReg, mask, width);
    StoreDataAndCastFromFp32<T_KV>(outAddr2, vregCos2Fp32, uReg, mask, width);
}

template <typename T_KV, typename T_K_CACHE>
__simd_vf__ inline void RecomputeRopeSymQuantComputeVF(__ubuf__ T_K_CACHE* quantPtr1, __ubuf__ T_K_CACHE* quantPtr2,
                                                       __ubuf__ T_KV* ropePtr, __ubuf__ T_KV* cosPtr1,
                                                       __ubuf__ T_KV* cosPtr2, __ubuf__ T_KV* sinPtr1,
                                                       __ubuf__ T_KV* sinPtr2, __ubuf__ float* scalePtr1,
                                                       __ubuf__ float* scalePtr2, __ubuf__ float* tmpBufferPtr,
                                                       uint32_t ubFactor, uint16_t repeatTimesFloor,
                                                       uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32, vregSin2Fp32;
    AscendC::Reg::RegTensor<float> vregScale1Fp32, vregScale2Fp32;
    AscendC::Reg::RegTensor<half> vregHalf;
    AscendC::Reg::RegTensor<int16_t> vregInt16;
    AscendC::Reg::RegTensor<T_K_CACHE> vregQuant;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;
    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
        auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
        auto cosAddr1 = cosPtr1 + j * VL_FP32;
        auto cosAddr2 = cosPtr2 + j * VL_FP32;
        auto sinAddr1 = sinPtr1 + j * VL_FP32;
        auto sinAddr2 = sinPtr2 + j * VL_FP32;
        auto scaleAddr1 = scalePtr1 + j * VL_FP32;
        auto scaleAddr2 = scalePtr2 + j * VL_FP32;
        auto kQuantAddr1 = quantPtr1 + j * VL_FP32;
        auto kQuantAddr2 = quantPtr2 + j * VL_FP32;

        LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                                ((__ubuf__ float*)(tmpBufferPtr)));

        LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

        AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
        AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
        AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

        // 量化部分
        LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
        LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
        AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
        AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);

        // 将结果cast成对应的量化数据类型
        StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
        StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
    }

    // 尾VL计算范式
    auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
    auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
    auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
    auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
    auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
    auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;
    auto scaleAddr1 = scalePtr1 + repeatTimesFloor * VL_FP32;
    auto scaleAddr2 = scalePtr2 + repeatTimesFloor * VL_FP32;
    auto kQuantAddr1 = quantPtr1 + repeatTimesFloor * VL_FP32;
    auto kQuantAddr2 = quantPtr2 + repeatTimesFloor * VL_FP32;

    LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                            ((__ubuf__ float*)(tmpBufferPtr)));

    LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

    AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
    AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
    AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

    // 量化部分
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
    LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
    AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
    AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);

    // 将结果cast成对应的量化类型
    StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
    StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
}

template <typename T_KV, typename T_K_CACHE>
__simd_vf__ inline void RecomputeRopeSymQuantWithComputeVF(
    __ubuf__ T_KV* outPtr1, __ubuf__ T_KV* outPtr2, __ubuf__ T_K_CACHE* quantPtr1, __ubuf__ T_K_CACHE* quantPtr2,
    __ubuf__ T_KV* ropePtr, __ubuf__ T_KV* cosPtr1, __ubuf__ T_KV* cosPtr2, __ubuf__ T_KV* sinPtr1,
    __ubuf__ T_KV* sinPtr2, __ubuf__ float* scalePtr1, __ubuf__ float* scalePtr2, __ubuf__ float* tmpBufferPtr,
    uint32_t ubFactor, uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32, vregSin2Fp32;
    AscendC::Reg::RegTensor<float> vregScale1Fp32, vregScale2Fp32;
    AscendC::Reg::RegTensor<half> vregHalf;
    AscendC::Reg::RegTensor<int16_t> vregInt16;
    AscendC::Reg::RegTensor<T_K_CACHE> vregQuant;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;
    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
        auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
        auto cosAddr1 = cosPtr1 + j * VL_FP32;
        auto cosAddr2 = cosPtr2 + j * VL_FP32;
        auto sinAddr1 = sinPtr1 + j * VL_FP32;
        auto sinAddr2 = sinPtr2 + j * VL_FP32;
        auto scaleAddr1 = scalePtr1 + j * VL_FP32;
        auto scaleAddr2 = scalePtr2 + j * VL_FP32;
        auto kQuantAddr1 = quantPtr1 + j * VL_FP32;
        auto kQuantAddr2 = quantPtr2 + j * VL_FP32;
        auto kOutAddr1 = outPtr1 + j * VL_FP32;
        auto kOutAddr2 = outPtr2 + j * VL_FP32;

        LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                                ((__ubuf__ float*)(tmpBufferPtr)));

        LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

        AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
        AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
        AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

        // 中间结果
        StoreDataAndCastFromFp32<T_KV>(kOutAddr1, vregCos1Fp32, uReg, mask, width);
        StoreDataAndCastFromFp32<T_KV>(kOutAddr2, vregCos2Fp32, uReg, mask, width);

        // 量化部分
        LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
        LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
        AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
        AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);

        // 将结果cast成对应的量化类型
        StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
        StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
    }

    // 尾VL计算范式
    auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
    auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
    auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
    auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
    auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
    auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;
    auto scaleAddr1 = scalePtr1 + repeatTimesFloor * VL_FP32;
    auto scaleAddr2 = scalePtr2 + repeatTimesFloor * VL_FP32;
    auto kQuantAddr1 = quantPtr1 + repeatTimesFloor * VL_FP32;
    auto kQuantAddr2 = quantPtr2 + repeatTimesFloor * VL_FP32;
    auto kOutAddr1 = outPtr1 + repeatTimesFloor * VL_FP32;
    auto kOutAddr2 = outPtr2 + repeatTimesFloor * VL_FP32;

    LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                            ((__ubuf__ float*)(tmpBufferPtr)));

    LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

    AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
    AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
    AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

    // 中间结果
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    StoreDataAndCastFromFp32<T_KV>(kOutAddr1, vregCos1Fp32, uReg, mask, width);
    StoreDataAndCastFromFp32<T_KV>(kOutAddr2, vregCos2Fp32, uReg, mask, width);

    // 量化部分
    LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
    LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
    AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
    AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);

    // 将结果cast成对应的量化类型
    StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
    StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
}

template <typename T_KV, typename T_K_CACHE>
__simd_vf__ inline void RecomputeRopeAsymQuantComputeVF(
    __ubuf__ T_K_CACHE* quantPtr1, __ubuf__ T_K_CACHE* quantPtr2, __ubuf__ T_KV* ropePtr, __ubuf__ T_KV* cosPtr1,
    __ubuf__ T_KV* cosPtr2, __ubuf__ T_KV* sinPtr1, __ubuf__ T_KV* sinPtr2, __ubuf__ float* scalePtr1,
    __ubuf__ float* scalePtr2, __ubuf__ float* offsetPtr1, __ubuf__ float* offsetPtr2, __ubuf__ float* tmpBufferPtr,
    uint32_t ubFactor, uint16_t repeatTimesFloor, uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32, vregSin2Fp32;
    AscendC::Reg::RegTensor<float> vregScale1Fp32, vregScale2Fp32, vregOffset1Fp32, vregOffset2Fp32;
    AscendC::Reg::RegTensor<half> vregHalf;
    AscendC::Reg::RegTensor<int16_t> vregInt16;
    AscendC::Reg::RegTensor<T_K_CACHE> vregQuant;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;
    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
        auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
        auto cosAddr1 = cosPtr1 + j * VL_FP32;
        auto cosAddr2 = cosPtr2 + j * VL_FP32;
        auto sinAddr1 = sinPtr1 + j * VL_FP32;
        auto sinAddr2 = sinPtr2 + j * VL_FP32;
        auto scaleAddr1 = scalePtr1 + j * VL_FP32;
        auto scaleAddr2 = scalePtr2 + j * VL_FP32;
        auto offsetAdd1 = offsetPtr1 + j * VL_FP32;
        auto offsetAdd2 = offsetPtr2 + j * VL_FP32;
        auto kQuantAddr1 = quantPtr1 + j * VL_FP32;
        auto kQuantAddr2 = quantPtr2 + j * VL_FP32;

        LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                                ((__ubuf__ float*)(tmpBufferPtr)));

        LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

        AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
        AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
        AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

        // 量化部分
        LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
        LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
        AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
        AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);
        LoadDataAndCast2Fp32<float>(vregOffset1Fp32, offsetAdd1, mask);
        LoadDataAndCast2Fp32<float>(vregOffset2Fp32, offsetAdd2, mask);
        AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregOffset1Fp32, mask);
        AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregOffset2Fp32, mask);

        // 将结果cast成对应的量化类型
        StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
        StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
    }

    // 尾VL计算范式
    auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
    auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
    auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
    auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
    auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
    auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;
    auto scaleAddr1 = scalePtr1 + repeatTimesFloor * VL_FP32;
    auto scaleAddr2 = scalePtr2 + repeatTimesFloor * VL_FP32;
    auto offsetAdd1 = offsetPtr1 + repeatTimesFloor * VL_FP32;
    auto offsetAdd2 = offsetPtr2 + repeatTimesFloor * VL_FP32;
    auto kQuantAddr1 = quantPtr1 + repeatTimesFloor * VL_FP32;
    auto kQuantAddr2 = quantPtr2 + repeatTimesFloor * VL_FP32;

    LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                            ((__ubuf__ float*)(tmpBufferPtr)));

    LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

    AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
    AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
    AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);

    // 量化部分
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
    LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
    AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
    AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);
    LoadDataAndCast2Fp32<float>(vregOffset1Fp32, offsetAdd1, mask);
    LoadDataAndCast2Fp32<float>(vregOffset2Fp32, offsetAdd2, mask);
    AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregOffset1Fp32, mask);
    AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregOffset2Fp32, mask);

    // 将结果cast成对应的量化类型
    StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
    StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
}

template <typename T>
__simd_vf__ inline void ScatterKNzOddDk1RearrangeVF(__ubuf__ T* srcPtr, __ubuf__ T* dstPtr, uint16_t repeatFloor,
                                                    uint32_t tailW)
{
    AscendC::Reg::RegTensor<T> reg;
    AscendC::Reg::UnalignRegForStore ureg;
    for (uint16_t j = 0; j < repeatFloor; j++) {
        AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(reg, srcPtr,
                                                                                static_cast<int32_t>(VL_FP32));
        AscendC::Reg::StoreUnAlign(dstPtr, reg, ureg, static_cast<uint32_t>(VL_FP32));
    }
    AscendC::Reg::LoadAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(reg, srcPtr, static_cast<int32_t>(tailW));
    AscendC::Reg::StoreUnAlign(dstPtr, reg, ureg, tailW);
    AscendC::Reg::StoreUnAlignPost(dstPtr, ureg, static_cast<int32_t>(0));
}

template <typename T_KV, typename T_K_CACHE>
__simd_vf__ inline void RecomputeRopeAsymQuantWithKvComputeVF(
    __ubuf__ T_KV* outPtr1, __ubuf__ T_KV* outPtr2, __ubuf__ T_K_CACHE* quantPtr1, __ubuf__ T_K_CACHE* quantPtr2,
    __ubuf__ T_KV* ropePtr, __ubuf__ T_KV* cosPtr1, __ubuf__ T_KV* cosPtr2, __ubuf__ T_KV* sinPtr1,
    __ubuf__ T_KV* sinPtr2, __ubuf__ float* scalePtr1, __ubuf__ float* scalePtr2, __ubuf__ float* offsetPtr1,
    __ubuf__ float* offsetPtr2, __ubuf__ float* tmpBufferPtr, uint32_t ubFactor, uint16_t repeatTimesFloor,
    uint16_t regTailWidth)
{
    AscendC::Reg::RegTensor<float> vregRope1Fp32, vregRope2Fp32, vregCos1Fp32, vregCos2Fp32, vregSin1Fp32, vregSin2Fp32;
    AscendC::Reg::RegTensor<float> vregScale1Fp32, vregScale2Fp32, vregOffset1Fp32, vregOffset2Fp32;
    AscendC::Reg::RegTensor<half> vregHalf;
    AscendC::Reg::RegTensor<int16_t> vregInt16;
    AscendC::Reg::RegTensor<T_K_CACHE> vregQuant;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::UnalignRegForStore uReg;
    uint32_t width = VL_FP32;
    mask = AscendC::Reg::UpdateMask<float>(width);

    // 标准VL计算范式
    for (uint16_t j = 0; j < repeatTimesFloor; j++) {
        auto ropeAddr1 = ropePtr + j * VL_FP32 * CONST_TWO;
        auto ropeAddr2 = ropePtr + (j * CONST_TWO + CONST_ONE) * VL_FP32;
        auto cosAddr1 = cosPtr1 + j * VL_FP32;
        auto cosAddr2 = cosPtr2 + j * VL_FP32;
        auto sinAddr1 = sinPtr1 + j * VL_FP32;
        auto sinAddr2 = sinPtr2 + j * VL_FP32;
        auto scaleAddr1 = scalePtr1 + j * VL_FP32;
        auto scaleAddr2 = scalePtr2 + j * VL_FP32;
        auto offsetAdd1 = offsetPtr1 + j * VL_FP32;
        auto offsetAdd2 = offsetPtr2 + j * VL_FP32;
        auto kQuantAddr1 = quantPtr1 + j * VL_FP32;
        auto kQuantAddr2 = quantPtr2 + j * VL_FP32;
        auto kOutAddr1 = outPtr1 + j * VL_FP32;
        auto kOutAddr2 = outPtr2 + j * VL_FP32;

        LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
        AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                                ((__ubuf__ float*)(tmpBufferPtr)));

        LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
        LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

        AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
        AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
        AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
        AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
        AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);
        // 中间结果
        StoreDataAndCastFromFp32<T_KV>(kOutAddr1, vregCos1Fp32, uReg, mask, width);
        StoreDataAndCastFromFp32<T_KV>(kOutAddr2, vregCos2Fp32, uReg, mask, width);

        // 量化部分
        LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
        LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
        AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
        AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);
        LoadDataAndCast2Fp32<float>(vregOffset1Fp32, offsetAdd1, mask);
        LoadDataAndCast2Fp32<float>(vregOffset2Fp32, offsetAdd2, mask);
        AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregOffset1Fp32, mask);
        AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregOffset2Fp32, mask);

        // 将结果cast成对应的量化类型
        StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
        StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
    }

    // 尾VL计算范式
    auto ropeAddr1 = ropePtr + repeatTimesFloor * VL_FP32 * CONST_TWO;
    auto ropeAddr2 = ropePtr + (repeatTimesFloor * CONST_TWO + CONST_ONE) * VL_FP32;
    auto cosAddr1 = cosPtr1 + repeatTimesFloor * VL_FP32;
    auto cosAddr2 = cosPtr2 + repeatTimesFloor * VL_FP32;
    auto sinAddr1 = sinPtr1 + repeatTimesFloor * VL_FP32;
    auto sinAddr2 = sinPtr2 + repeatTimesFloor * VL_FP32;
    auto scaleAddr1 = scalePtr1 + repeatTimesFloor * VL_FP32;
    auto scaleAddr2 = scalePtr2 + repeatTimesFloor * VL_FP32;
    auto offsetAdd1 = offsetPtr1 + repeatTimesFloor * VL_FP32;
    auto offsetAdd2 = offsetPtr2 + repeatTimesFloor * VL_FP32;
    auto kQuantAddr1 = quantPtr1 + repeatTimesFloor * VL_FP32;
    auto kQuantAddr2 = quantPtr2 + repeatTimesFloor * VL_FP32;
    auto kOutAddr1 = outPtr1 + repeatTimesFloor * VL_FP32;
    auto kOutAddr2 = outPtr2 + repeatTimesFloor * VL_FP32;

    LoadDataAndCast2Fp32<T_KV>(vregRope1Fp32, ropeAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregRope2Fp32, ropeAddr2, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr, vregRope1Fp32, mask);
    AscendC::Reg::StoreAlign(tmpBufferPtr + VL_FP32, vregRope2Fp32, mask);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(vregRope1Fp32, vregRope2Fp32,
                                                                            ((__ubuf__ float*)(tmpBufferPtr)));

    LoadDataAndCast2Fp32<T_KV>(vregCos1Fp32, cosAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregCos2Fp32, cosAddr2, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin1Fp32, sinAddr1, mask);
    LoadDataAndCast2Fp32<T_KV>(vregSin2Fp32, sinAddr2, mask);

    AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Muls(vregRope2Fp32, vregRope2Fp32, CONST_MINUS_ONE, mask);
    AscendC::Reg::Mul(vregSin1Fp32, vregSin1Fp32, vregRope2Fp32, mask);
    AscendC::Reg::Mul(vregSin2Fp32, vregSin2Fp32, vregRope1Fp32, mask);
    AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregSin1Fp32, mask);
    AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregSin2Fp32, mask);
    // 中间结果
    width = regTailWidth;
    mask = AscendC::Reg::UpdateMask<float>(width);
    StoreDataAndCastFromFp32<T_KV>(kOutAddr1, vregCos1Fp32, uReg, mask, width);
    StoreDataAndCastFromFp32<T_KV>(kOutAddr2, vregCos2Fp32, uReg, mask, width);

    // 量化部分
    LoadDataAndCast2Fp32<float>(vregScale1Fp32, scaleAddr1, mask);
    LoadDataAndCast2Fp32<float>(vregScale2Fp32, scaleAddr2, mask);
    AscendC::Reg::Mul(vregCos1Fp32, vregCos1Fp32, vregScale1Fp32, mask);
    AscendC::Reg::Mul(vregCos2Fp32, vregCos2Fp32, vregScale2Fp32, mask);
    LoadDataAndCast2Fp32<float>(vregOffset1Fp32, offsetAdd1, mask);
    LoadDataAndCast2Fp32<float>(vregOffset2Fp32, offsetAdd2, mask);
    AscendC::Reg::Add(vregCos1Fp32, vregCos1Fp32, vregOffset1Fp32, mask);
    AscendC::Reg::Add(vregCos2Fp32, vregCos2Fp32, vregOffset2Fp32, mask);

    // 将结果cast成对应的量化类型
    StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr1, vregCos1Fp32, mask);
    StoreQuantAndCastFromFp32<T_K_CACHE>(kQuantAddr2, vregCos2Fp32, mask);
}

} // namespace KvRmsNormRopeCache

#endif // KV_RMS_NORM_ROPE_CACHE_REGBASE_VF_H_
