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
 * \file vf_dynamic_quant.h
 * \brief 
 */

#ifndef VF_DYNAMIC_QUANT_H
#define VF_DYNAMIC_QUANT_H

#include "kernel_tensor.h"

namespace MlaProlog {
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float FP8_E4M3FN_MIN_VALUE = -448.0f;
constexpr uint32_t FP8_E4M3FN_BLOCK_SIZE = 32;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t DIGIT_TWO = 2;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float NEG_INFINITY = -INFINITY;
constexpr uint16_t REDUCE_SIZE = 8;

template<typename T, typename C, typename O>
__simd_vf__ void ComputeVFImpl(__ubuf__ T* xAddr, __ubuf__ O* yAddr, __ubuf__ float* scaleAddr,
    uint32_t rowIndex, uint32_t rowCount, uint32_t dtypeSize, uint16_t VL, uint16_t vfLoop, const float maxValue)
{
    constexpr static AscendC::MicroAPI::CastTrait castTraitB16ToB32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32Tofp8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true};
    AscendC::MicroAPI::RegTensor<T> vreg1; // 搬入的x
    AscendC::MicroAPI::RegTensor<float> vreg2; // cast成float之后的x
    AscendC::MicroAPI::RegTensor<float> vreg3; // x的绝对值absvalue
    AscendC::MicroAPI::RegTensor<float> vreg4; // x的abs与-inf比较的结果，可以认为是x的absvalue的一些最大值
    AscendC::MicroAPI::RegTensor<float> vreg5; // reduceMax
    AscendC::MicroAPI::RegTensor<float> vreg6; // scale
    AscendC::MicroAPI::RegTensor<float> vreg7; // Duplicate之后的scale，为了和input一起得到y
    AscendC::MicroAPI::RegTensor<float> vreg8; // input/scale
    AscendC::MicroAPI::RegTensor<O> vreg9; // 最终y

    AscendC::MicroAPI::MaskReg preg0;
    AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg preg2;

    AscendC::MicroAPI::UnalignRegForStore ureg0;
    AscendC::MicroAPI::Duplicate(vreg4, NEG_INFINITY, preg1);
    uint32_t sreg0 = rowCount;

    for (uint16_t j = 0; j < vfLoop; j++) {
        preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, xAddr + rowIndex * rowCount + j * VL);
        AscendC::MicroAPI::Cast<float, T, castTraitB16ToB32>(vreg2, vreg1, preg0);
        AscendC::MicroAPI::Abs(vreg3, vreg2, preg0);
        AscendC::MicroAPI::Max(vreg4, vreg3, vreg4, preg1);
    }
    AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
                            vreg5, vreg4, preg1);
    AscendC::MicroAPI::Muls(vreg6, vreg5, maxValue, preg1);
    AscendC::MicroAPI::Duplicate(vreg7, vreg6, preg1);
    AscendC::MicroAPI::StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(scaleAddr, vreg6, ureg0, 1);

    uint32_t sreg1 = rowCount;
    for (uint16_t j = 0; j < vfLoop; j++) {
        auto addr = yAddr + rowIndex * rowCount + j * VL;
        preg2 = AscendC::MicroAPI::UpdateMask<float>(sreg1);
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, xAddr + rowIndex * rowCount + j * VL);
        AscendC::MicroAPI::Cast<float, T, castTraitB16ToB32>(vreg2, vreg1, preg2);
        AscendC::MicroAPI::Div(vreg8, vreg2, vreg7, preg2);
        AscendC::MicroAPI::Cast<O, float, castTraitF32Tofp8>(vreg9, vreg8, preg2);
        AscendC::MicroAPI::StoreAlign<O, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vreg9, preg2);
    }
    AscendC::MicroAPI::StoreUnAlignPost(scaleAddr, ureg0, 0);
}

template<typename T, typename C, typename O>
__aicore__ inline void ComputeVF(__ubuf__ T* xAddr, __ubuf__ O* yAddr, __ubuf__ float* scaleAddr,
    uint32_t rowIndex, uint32_t col)
{
    uint32_t dtypeSize = sizeof(float);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint32_t rowCount = col;
    uint16_t vfLoop = (rowCount + VL - 1) / VL;
    const float maxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;

    ComputeVFImpl<T, C, O>(xAddr, yAddr, scaleAddr, rowIndex, rowCount, dtypeSize, VL, vfLoop, maxValue);
}

/**
 * @brief DynamicQuantQnVf 对row行进行dynamicquant, BF16 ---> FP8E4M3, 每一行出一个系数。
 * @param outputLocal 输出tensor [row , col]
 * @param scale 输出每行的反量化系数 [row]
 * @param inputLocal 输入tensor [row , col]
 * @param row 待处理的行数
 * @param col 待处理的列数
 */
template<typename T, typename C, typename O>
__aicore__ inline void DynamicQuantQnVf(const LocalTensor<O> &outputLocal, const LocalTensor<C> &scale,
    const LocalTensor<T> &inputLocal, uint64_t row, uint64_t col)
{
    auto xAddr = (__local_mem__ T*)inputLocal.GetPhyAddr();
    auto yAddr = (__local_mem__ O*)outputLocal.GetPhyAddr();
    auto scaleAddr = (__local_mem__ C*)scale.GetPhyAddr();
    for (int32_t i = 0; i < row; i++) {
        ComputeVF<T, C, O>(xAddr, yAddr, scaleAddr + i, i, col);
    }
}

template<typename T, typename U>
__simd_vf__ void ComputeMaxExpVF(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, uint32_t totalCountInUB,
    uint16_t loopNum, uint16_t vecLen)
{
    AscendC::MicroAPI::RegTensor<T> vdExp0;
    AscendC::MicroAPI::RegTensor<T> vdExp1;
    AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
    AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;

    AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
    AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

    AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
    AscendC::MicroAPI::MaskReg scaleMask1;
    AscendC::MicroAPI::MaskReg scaleMask2;
    AscendC::MicroAPI::UnalignRegForStore u1;

    for (uint16_t i = 0; i < loopNum; i++) {
        scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
        scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vecLen * DIGIT_TWO);
        // 通过位与运算得到bf16的指数位保留，尾数位置0所对应的值, 0x7f80是bf16 8个指数位为1，7个尾数位为0对应的值
        AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16,
            scaleMask1);

        AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16,
            scaleMask1);
        // 得到指数位最大的值
        AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
        // 得到每个block(32个元素)中的指数位最大的值
        AscendC::MicroAPI::ReduceDataBlock<MicroAPI::ReduceType::MAX, uint16_t, MicroAPI::MaskMergeMode::ZEROING>(
            vdMaxExp, vdMaxExp, scaleMask1);
        AscendC::MicroAPI::StoreUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxExpAddr,
            vdMaxExp, u1, REDUCE_SIZE);
    }
    AscendC::MicroAPI::StoreUnAlignPost(maxExpAddr, u1, 0);
}


template<typename T, typename U>
__simd_vf__ void ComputeScaleVF(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
    __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB, uint16_t loopNumScale, uint16_t vecLen)
{
    AscendC::MicroAPI::RegTensor<uint16_t> expMask;
    AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
    AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
    AscendC::MicroAPI::RegTensor<T> vdExp0;
    AscendC::MicroAPI::RegTensor<T> vdExp1;

    AscendC::MicroAPI::MaskReg cmpResult;
    AscendC::MicroAPI::MaskReg zeroMask;
    AscendC::MicroAPI::MaskReg cmpResultSub;
    AscendC::MicroAPI::MaskReg preMaskScale;
    AscendC::MicroAPI::MaskReg invalidDataMask;
    AscendC::MicroAPI::MaskReg specialDataMask;

    AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
    AscendC::MicroAPI::Duplicate(maxExpValue, FP8_E4M3_MAX_EXP);
    AscendC::MicroAPI::RegTensor<uint16_t> sharedExp;
    AscendC::MicroAPI::RegTensor<uint16_t> scaleValue;
    AscendC::MicroAPI::RegTensor<uint16_t> scaleBias;
    AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
    AscendC::MicroAPI::RegTensor<uint16_t> halfScale;
    AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
    AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
    AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
    AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
    AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
    AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
    AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
    AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

    for (uint16_t i = 0; i < loopNumScale; i++) {
        preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
        AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vdMaxExp,
            maxExpAddr, vecLen);
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);    //INF/NAN
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
        // 如果vdMaxExp小于等于maxExpValue, 则置为maxExpValue, maxExpValue为FP8E4M3最大正整数的指数位8左移7位是0x400
        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
        // vdMaxExp - maxExpValue后右移7位得到FP8E8M0的值，右移7位是因为bf16的指数位从第7位开始
        AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
        AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

        AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue, vecLen / DIGIT_TWO,
            preMaskScale);

        AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
        // 0x7f00 - sharedExp得到1/sharedExp
        AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

        AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr,
            halfScale, vecLen, preMaskScale);
    }
}

template<typename T, typename U>
__simd_vf__ void ComputeDataVF(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB, uint32_t totalCountInUB2, uint16_t loopNum, uint16_t vecLen)
{
    AscendC::MicroAPI::MaskReg dataMask1;
    AscendC::MicroAPI::MaskReg dataMask2;
    AscendC::MicroAPI::MaskReg dataMask3;
    AscendC::MicroAPI::MaskReg dataMask4;
    AscendC::MicroAPI::MaskReg nanResult;

    AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
    AscendC::MicroAPI::RegTensor<float> floatScaleForMul;
    AscendC::MicroAPI::RegTensor<T> vdExp0;
    AscendC::MicroAPI::RegTensor<T> vdExp1;
    AscendC::MicroAPI::RegTensor<T> vdExp0Convert;
    AscendC::MicroAPI::RegTensor<T> vdExp1Convert;
    AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero;
    AscendC::MicroAPI::RegTensor<float> vdExp0FP32One;
    AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero;
    AscendC::MicroAPI::RegTensor<float> vdExp1FP32One;
    AscendC::MicroAPI::RegTensor<float> maxFp8Value;
    AscendC::MicroAPI::Duplicate(maxFp8Value, FP8_E4M3FN_MAX_VALUE);
    AscendC::MicroAPI::RegTensor<float> minFp8Value;
    AscendC::MicroAPI::Duplicate(minFp8Value, FP8_E4M3FN_MIN_VALUE);
    AscendC::MicroAPI::RegTensor<U> vdExp0FP8Zero;
    AscendC::MicroAPI::RegTensor<U> vdExp0FP8One;
    AscendC::MicroAPI::RegTensor<U> vdExp1FP8Zero;
    AscendC::MicroAPI::RegTensor<U> vdExp1FP8One;

    static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t i = 0; i < loopNum; i++) {
        dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
        dataMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
        dataMask3 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
        dataMask4 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB2);
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vecLen * DIGIT_TWO);
        AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr, REDUCE_SIZE);

        // X / mxscale
        AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
        AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T> &)halfScaleForMul, dataMask1);
        AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
        AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
        AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
        AscendC::MicroAPI::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
        // 大于448.0的值设为448.0
        AscendC::MicroAPI::Compare<float, CMPMODE::GT>(nanResult, vdExp0FP32Zero, maxFp8Value, dataMask1);
        AscendC::MicroAPI::Select<float>(vdExp0FP32Zero, maxFp8Value, vdExp0FP32Zero, nanResult);
        AscendC::MicroAPI::Compare<float, CMPMODE::GT>(nanResult, vdExp0FP32One, maxFp8Value, dataMask1);
        AscendC::MicroAPI::Select<float>(vdExp0FP32One, maxFp8Value, vdExp0FP32One, nanResult);
        // 小于-448.0的值设为-448。0
        AscendC::MicroAPI::Compare<float, CMPMODE::LT>(nanResult, vdExp0FP32Zero, minFp8Value, dataMask1);
        AscendC::MicroAPI::Select<float>(vdExp0FP32Zero, minFp8Value, vdExp0FP32Zero, nanResult);
        AscendC::MicroAPI::Compare<float, CMPMODE::LT>(nanResult, vdExp0FP32One, minFp8Value, dataMask1);
        AscendC::MicroAPI::Select<float>(vdExp0FP32One, minFp8Value, vdExp0FP32One, nanResult);
        // 将结果转为FP8E4M3
        AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
        AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);

        AscendC::MicroAPI::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
        AscendC::MicroAPI::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
        AscendC::MicroAPI::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
        // 大于448.0的值设为448.0
        AscendC::MicroAPI::Compare<float, CMPMODE::GT>(nanResult, vdExp1FP32Zero, maxFp8Value, dataMask2);
        AscendC::MicroAPI::Select<float>(vdExp1FP32Zero, maxFp8Value, vdExp1FP32Zero, nanResult);
        AscendC::MicroAPI::Compare<float, CMPMODE::GT>(nanResult, vdExp1FP32One, maxFp8Value, dataMask2);
        AscendC::MicroAPI::Select<float>(vdExp1FP32One, maxFp8Value, vdExp1FP32One, nanResult);
        // 小于-448.0的值设为-448。0
        AscendC::MicroAPI::Compare<float, CMPMODE::LT>(nanResult, vdExp1FP32Zero, minFp8Value, dataMask2);
        AscendC::MicroAPI::Select<float>(vdExp1FP32Zero, minFp8Value, vdExp1FP32Zero, nanResult);
        AscendC::MicroAPI::Compare<float, CMPMODE::LT>(nanResult, vdExp1FP32One, minFp8Value, dataMask2);
        AscendC::MicroAPI::Select<float>(vdExp1FP32One, minFp8Value, vdExp1FP32One, nanResult);
        // 将结果转为FP8E4M3
        AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
        AscendC::MicroAPI::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
        AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
            (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
        AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
            (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
        AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
            (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
        AscendC::MicroAPI::StoreAlign<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
            AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(outLocalAddr,
            (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
    }
}

/**
 * @brief DynamicQuantCqVf 对row进行dynamicquant, BF16 ---> FP8E4M3, 每个BLOCK出一个系数。
 * @param outputLocal 输出tensor [row, col]
 * @param outputScale 输出每行的反量化系数 [row, col/32]
 * @param inputLocal 输入tensor [row, col]
 * @param tmpLocal 临时buffer,所需空间大小row * col * 2
 * @param row 待处理的行数
 * @param col 待处理的列数
 */
/**
 shared_exp = floor(log2(max(|Vi|))) - emax
 mxscale = 2^shared_exp
 Pi = cast_to_dst_type(Vi/mxscale, round_mode)
**/
template<typename T, typename U>
__aicore__ inline void DynamicQuantCqVf(const LocalTensor<int8_t>& outputLocal, const LocalTensor<uint16_t>& outputScale,
    const LocalTensor<T>& inputLocal, const LocalTensor<uint8_t>& tmpLocal, uint32_t row, uint32_t col)
{
    LocalTensor<uint16_t> maxExpLocal = tmpLocal.ReinterpretCast<uint16_t>();
    uint32_t totalScaleInUB = row * col / FP8_E4M3FN_BLOCK_SIZE;
    uint32_t totalCountInUB = row * col;
    uint16_t vecLen = AscendC::VECTOR_REG_WIDTH / sizeof(T);
    uint16_t loopNum = (totalCountInUB + vecLen * DIGIT_TWO - 1) / (vecLen * DIGIT_TWO);
    uint16_t loopNumScale = (totalScaleInUB + vecLen - 1) / vecLen;

    auto srcAddr = reinterpret_cast<__ubuf__ T *>(inputLocal.GetPhyAddr());
    auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpLocal.GetPhyAddr());
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(outputScale.GetPhyAddr());
    LocalTensor<uint16_t> halfScaleLocal = maxExpLocal[totalCountInUB].template ReinterpretCast<uint16_t>();
    auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScaleLocal.GetPhyAddr());
    auto outLocalAddr = reinterpret_cast<__ubuf__ int8_t *>(outputLocal.GetPhyAddr());
    ComputeMaxExpVF<T, U>(srcAddr, maxExpAddr, totalCountInUB, loopNum, vecLen);
    ComputeScaleVF<T, U>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, totalScaleInUB, loopNumScale, vecLen);

    srcAddr = reinterpret_cast<__ubuf__ T *>(inputLocal.GetPhyAddr());
    halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScaleLocal.GetPhyAddr());

    uint32_t totalCountInUB2 = totalCountInUB * DIGIT_TWO;
    ComputeDataVF<T, U>(srcAddr, halfScaleLocalAddr, outLocalAddr, totalCountInUB, totalCountInUB2, loopNum, vecLen);
}

}
#endif