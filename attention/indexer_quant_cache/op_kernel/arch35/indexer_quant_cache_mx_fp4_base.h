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
 * \file indexer_quant_cache_mx_fp4_base.h
 * \brief MX-FP4 (quant_mode=3) per-block dynamic quantization primitives.
 *
 * The MX-FP4 quantization algorithm (shared-exponent / e8m0 scale + fp4 packing) is ported
 * from cann-recipes-infer moe_init_routing_group_quant/op_kernel/moe_v3_gather_mxfp4_quant_gather.h.
 * One MX block = 32 elements -> one e8m0 scale; two fp4 values are packed into one byte.
 */

#ifndef INDEXER_QUANT_CACHE_MX_FP4_BASE_H
#define INDEXER_QUANT_CACHE_MX_FP4_BASE_H

#include "kernel_operator.h"
#include "indexer_quant_cache_base.h"

namespace IndexerQuantCache {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// MX量化一个块的元素数量为32个，即1个scale对应32个x的元素
constexpr int64_t MX_BLOCK_SIZE_MXFP4 = 32LL;
// 一个VL放多少fp32->fp8的元素个数（按fp32算的一个VL能放几个元素）
constexpr int64_t OUT_ELE_NUM_ONE_BLK_MAXFP4 = 64LL;
// fp16中指数部分Mask，同时也表示fp16的INF值
constexpr uint16_t FP16_EMASK_AND_INF_VAL_MAXFP4 = 0x7c00;
// bf16中指数部分Mask，同时也表示bf16的INF值
constexpr uint16_t BF16_EMASK_AND_INF_VAL_MAXFP4 = 0x7f80;
// bfloat16的nan值（与inf值不同）
constexpr uint16_t BF16_NAN_VAL_MAXFP4 = 0x7f81;
// bf16指数位的偏移位数
constexpr int16_t BF16_EXP_SHR_BITS_MAXFP4 = 7;
// 用于计算halfScale=BF16_EXP_INVSUB-sharedExp，得到的halfScale就是1/realScale，可用于量化xQuant=x*halfScale
constexpr uint16_t BF16_EXP_INVSUB_MAXFP4 = 0x7f00;
constexpr int64_t DIGIT_TWO_MAXFP4 = 2;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t NEW_MANTISSA_MAXFP4 = 0x0008;
// e8m0 的特殊值（与 moe 参考保持一致），CANN 头未提供，故在此定义
constexpr uint16_t FP8_E8M0_NAN_VAL_MAXFP4 = 0x00ff;
constexpr uint16_t FP8_E8M0_SPECIAL_MIN_MAXFP4 = 0x0040;

// e8m0 scale 每行在 UB 中的元素跨度(e8m0=1B/元素，故元素数==字节数)。DIST_PACK_B16 存储要求目的地址按
// VL/2=64B 对齐，多行场景下每行起始偏移必须 64B 对齐，否则 r>=1 行漏写。vf 写入(scaleRowBytes)、
// InitBuffer、CopyOut 三处必须使用本对齐，保持一致。
__aicore__ inline uint32_t ScaleRowAlignMxFp4(uint32_t scaleCol)
{
    return ((scaleCol + 63U) / 64U) * 64U;
}

template <typename T, typename U>
__aicore__ inline void FP16ConvertMXFP4(
    AscendC::MicroAPI::RegTensor<half>& output, AscendC::MicroAPI::RegTensor<half>& input,
    AscendC::MicroAPI::MaskReg& mask)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> specialValueTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> newMantissa;
        AscendC::MicroAPI::RegTensor<uint16_t> andResult;
        AscendC::MicroAPI::RegTensor<uint16_t> newValue;
        AscendC::MicroAPI::MaskReg specialMask;
        AscendC::MicroAPI::MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (IsSameType<U, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        AscendC::MicroAPI::Duplicate(specialValueTensor, specialValue);
        AscendC::MicroAPI::Duplicate(newMantissa, NEW_MANTISSA_MAXFP4);
        AscendC::MicroAPI::And(andResult, (AscendC::MicroAPI::RegTensor<uint16_t>&)input, specialValueTensor, mask);
        AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA_MAXFP4, mask);
        AscendC::MicroAPI::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        AscendC::MicroAPI::Or(newValue, (AscendC::MicroAPI::RegTensor<uint16_t>&)input, newMantissa, mask);
        AscendC::MicroAPI::Select<uint16_t>(
            (AscendC::MicroAPI::RegTensor<uint16_t>&)output, newValue, (AscendC::MicroAPI::RegTensor<uint16_t>&)input,
            specialMask);
    }
}

// 计算每个 MX 块(32元素)的最大指数, 输出到 maxExpAddr
template <typename T, typename U>
__aicore__ inline void vfComputeMaxExpMXFP4(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr,
    uint32_t totalCountInUB, uint16_t loopNum, uint32_t vlForB16, uint32_t numUbBlocksPerVReg)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect0;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpSelect1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, BF16_EMASK_AND_INF_VAL_MAXFP4);
        AscendC::MicroAPI::RegTensor<uint16_t> invalidmaskfp16;
        AscendC::MicroAPI::Duplicate(invalidmaskfp16, FP16_EMASK_AND_INF_VAL_MAXFP4);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::MaskReg scaleMask2;
        AscendC::MicroAPI::MaskReg invalidDataMask0;
        AscendC::MicroAPI::MaskReg invalidDataMask1;
        AscendC::MicroAPI::UnalignReg u1;
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            scaleMask2 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForB16 * DIGIT_TWO_MAXFP4);

            if constexpr (IsSameType<T, half>::value) {
                AscendC::MicroAPI::And(
                    vdExpSelect0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, invalidmaskfp16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpSelect1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, invalidmaskfp16, scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask0, vdExpSelect0, invalidmaskfp16, scaleMask1);
                AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask1, vdExpSelect1, invalidmaskfp16, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                AscendC::MicroAPI::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                AscendC::MicroAPI::And(
                    vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16, scaleMask1);
                AscendC::MicroAPI::And(
                    vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16, scaleMask1);
            }

            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, numUbBlocksPerVReg);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
    return;
}

// 由每块最大指数计算 e8m0 scale(写到 mxScaleLocalAddr) 与量化用的 halfScale(写到 halfScaleLocalAddr)
// validScaleInUB: 实际有效的 scale 列数(scaleCol)，用于所有计算/读取/halfScale 存储；
// packScaleInUB:  向上对齐(偶数)后的列数，仅用于 DIST_PACK_B16 的存储 mask —— PACK_B16 按“成对”
//   方式把 b16 低字节打包成连续字节，存储 mask 必须覆盖偶数个 lane，否则在有效 lane 数为奇数/过少
//   (如 scaleCol<=2)时会漏写正确值(写成 0)。参考 swiglu_group_quant 的 maskLoop/maskValid 双 mask 用法。
//   多写出的补位字节落在每行 32B scratch 内，CopyOut 只拷贝有效 scaleCol 字节，故无副作用。
template <typename T, typename U>
__aicore__ inline void vfComputeScaleMXFP4(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
    __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t validScaleInUB, uint32_t packScaleInUB, uint16_t loopNumScale,
    uint16_t f4Emax, uint32_t vlForB16)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, BF16_EMASK_AND_INF_VAL_MAXFP4);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;

        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg preMaskScale;
        AscendC::MicroAPI::MaskReg packMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
        AscendC::MicroAPI::Duplicate(maxExpValue, f4Emax);
        AscendC::MicroAPI::RegTensor<uint16_t> sharedExp;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleValue;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_INVSUB_MAXFP4);
        AscendC::MicroAPI::RegTensor<uint16_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, FP8_E8M0_NAN_VAL_MAXFP4);
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, BF16_NAN_VAL_MAXFP4);
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, FP8_E8M0_SPECIAL_MIN_MAXFP4);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(validScaleInUB);
            packMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(packScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vlForB16);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, BF16_EXP_SHR_BITS_MAXFP4, preMaskScale);

            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForB16 / DIGIT_TWO_MAXFP4, packMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, vlForB16, preMaskScale);
        }
    }
}

// 用 halfScale 量化 x 并将 fp4 打包(2个/字节)写到 outLocalAddr
template <typename T, typename U>
__aicore__ inline void vfComputeDataMXFP4(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
    __ubuf__ int8_t* outLocalAddr, uint32_t totalCountInUB, uint16_t loopNum, uint32_t vlForB16,
    uint32_t numUbBlocksPerVReg)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0;
        AscendC::MicroAPI::RegTensor<T> vdExp1;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1BF16;

        AscendC::MicroAPI::RegTensor<U> vdExp0FP4;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP4;

        static constexpr AscendC::MicroAPI::CastTrait castTrait = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<
                T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForB16 * DIGIT_TWO_MAXFP4);
            AscendC::MicroAPI::DataCopy<
                uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, numUbBlocksPerVReg);

            if constexpr (IsSameType<T, half>::value) {
                FP16ConvertMXFP4<T, U>(vdExp0, vdExp0, dataMask1);
                FP16ConvertMXFP4<T, U>(vdExp1, vdExp1, dataMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                AscendC::MicroAPI::Mul(
                    vdExp0BF16, vdExp0BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(
                    vdExp1BF16, vdExp1BF16, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                AscendC::MicroAPI::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);
            } else {
                AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, dataMask1);
                AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                AscendC::MicroAPI::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            }

            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK_MAXFP4, dataMask1);
            AscendC::MicroAPI::DataCopy<
                int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK_MAXFP4, dataMask1);
        }
    }
}

// 对 curRowNum 行、每行 curColNum(=d) 个元素做 MX-FP4 量化。
// yLocal: 打包后的 fp4 cache(int8 字节寻址), scaleLocal: e8m0 scale, xLocal: 输入(half/bf16)。
// TCache 仅用于按 cache 输出 dtype(float4_e2m1x2_t / float4_e1m2x2_t)在编译期选择 fp4 子格式
// (E2M1 / E1M2)；cache 存储统一按字节(int8)寻址，避开 float4_*x2_t 的 sizeof()==2 导致的偏移翻倍。
template <typename TCache, typename TScale, typename TX>
__aicore__ inline void VFProcessDynamicMxFp4Quant(
    const LocalTensor<int8_t>& yLocal, const LocalTensor<TScale>& scaleLocal, const LocalTensor<TX>& xLocal,
    const LocalTensor<uint16_t>& maxExpLocal, const LocalTensor<uint16_t>& halfScaleLocal,
    const uint16_t curRowNum, const uint32_t curColNum)
{
    // fp4 子格式：E1M2 时 cache dtype 为 float4_e1m2x2_t，否则 E2M1。f4Emax 随之不同。
    constexpr bool isE1M2 = AscendC::IsSameType<TCache, float4_e1m2x2_t>::value;

    constexpr uint32_t vRegSize = VL_FP32 * static_cast<uint32_t>(sizeof(float));  // 256B
    constexpr uint32_t ubBlockSize = 32U;
    const uint32_t vlForB16 = vRegSize / static_cast<uint32_t>(sizeof(TX));          // b16:128
    const uint32_t numUbBlocksPerVReg = vRegSize / ubBlockSize;                      // 8

    const uint32_t scaleCol = (curColNum + MX_BLOCK_SIZE_MXFP4 - 1) / MX_BLOCK_SIZE_MXFP4;
    // 各 per-row 输出 buffer 的对齐跨度（maxExp / halfScale 为行内瞬态, 跨行复用同一块scratch）
    const uint32_t xRowAlign = RoundUp<TX>(curColNum);
    const uint32_t cacheRowAlign = RoundUp<int8_t>((curColNum + 1) / 2);   // fp4 打包后字节数(按字节对齐)

    // DIST_PACK_B16 按成对(2个b16低字节->2连续字节)打包，存储 mask 覆盖的 lane 数须为偶数(与 swiglu
    // 的 maskLoop 取值一致)；scaleCol 为奇数时若 mask 只覆盖单 lane 会漏写最后半对，故向上对齐到偶数。
    // 补位 lane 的值经压缩后写入每行 64B scratch 的尾部，CopyOut 只拷贝有效 scaleCol 字节，无副作用。
    // 配合 64B 对齐的每行跨度(scaleRowBytes)与 64B 对齐的 buffer 基址(scale Que 最先 InitBuffer)，
    // 共同满足 PACK_B16 的 64B 对齐写入要求。
    const uint32_t scaleColPack = (scaleCol + 1U) & ~1U;

    uint16_t loopNumX = static_cast<uint16_t>((curColNum + vlForB16 * DIGIT_TWO_MAXFP4 - 1) /
                                              (vlForB16 * DIGIT_TWO_MAXFP4));
    uint16_t loopNumScale = static_cast<uint16_t>((scaleColPack + vlForB16 - 1) / vlForB16);

    // scale 输出为 e8m0(1B/元素)。DIST_PACK_B16 存储要求目的地址按 VL/2=64B 对齐，否则在 r>=1 的非
    // 64B 对齐偏移上会漏写(写成 0)。因此每行 scale 在 UB 中按 64B 对齐步进；CopyOut/InitBuffer 必须用
    // 相同的 64B 行跨度(见 {single,multi}_row_mx_fp4.h 的 SCALE_ROW_ALIGN_MXFP4)。
    const uint32_t scaleRowBytes = ((scaleCol * static_cast<uint32_t>(sizeof(TScale)) + 63U) / 64U) * 64U;

    auto xAddr = reinterpret_cast<__ubuf__ TX*>(xLocal.GetPhyAddr());
    auto cacheAddr = reinterpret_cast<__ubuf__ int8_t*>(yLocal.GetPhyAddr());
    auto scaleByteAddr = reinterpret_cast<__ubuf__ uint8_t*>(scaleLocal.GetPhyAddr());
    auto maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
    auto halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(halfScaleLocal.GetPhyAddr());

    for (uint16_t r = 0; r < curRowNum; r++) {
        __ubuf__ TX* rowXAddr = xAddr + r * xRowAlign;
        __ubuf__ int8_t* rowCacheAddr = cacheAddr + r * cacheRowAlign;
        __ubuf__ uint16_t* rowScaleAddr =
            reinterpret_cast<__ubuf__ uint16_t*>(scaleByteAddr + r * scaleRowBytes);

        if constexpr (isE1M2) {
            vfComputeMaxExpMXFP4<TX, fp4x2_e1m2_t>(
                rowXAddr, maxExpAddr, curColNum, loopNumX, vlForB16, numUbBlocksPerVReg);
            vfComputeScaleMXFP4<TX, fp4x2_e1m2_t>(
                maxExpAddr, rowScaleAddr, halfScaleAddr, scaleCol, scaleColPack, loopNumScale,
                FP4_E1M2_MAX_EXP, vlForB16);
            vfComputeDataMXFP4<TX, fp4x2_e1m2_t>(
                rowXAddr, halfScaleAddr, rowCacheAddr, curColNum, loopNumX, vlForB16, numUbBlocksPerVReg);
        } else {
            vfComputeMaxExpMXFP4<TX, fp4x2_e2m1_t>(
                rowXAddr, maxExpAddr, curColNum, loopNumX, vlForB16, numUbBlocksPerVReg);
            vfComputeScaleMXFP4<TX, fp4x2_e2m1_t>(
                maxExpAddr, rowScaleAddr, halfScaleAddr, scaleCol, scaleColPack, loopNumScale,
                FP4_E2M1_BF16_MAX_EXP, vlForB16);
            vfComputeDataMXFP4<TX, fp4x2_e2m1_t>(
                rowXAddr, halfScaleAddr, rowCacheAddr, curColNum, loopNumX, vlForB16, numUbBlocksPerVReg);
        }
    }
}

} // namespace IndexerQuantCache

#endif // INDEXER_QUANT_CACHE_MX_FP4_BASE_H