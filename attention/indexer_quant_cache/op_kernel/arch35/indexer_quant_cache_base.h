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
 * \file indexer_quant_cache_base.h
 * \brief
 */

#ifndef INDEXER_QUANT_CACHE_BASE_H
#define INDEXER_QUANT_CACHE_BASE_H

#include "kernel_operator.h"

namespace IndexerQuantCache {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t PER_BLOCK_FP16 = 128;
constexpr uint32_t REPEAT_SIZE = 256;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float FP8_E5M2_MIN_VALUE = -57344.0f;
constexpr float FP8_E4M3FN_MIN_VALUE = -448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr uint32_t INV_HIFP8_MAX_VALUE = 0x38000000;
constexpr int64_t HIFLOAT_QUANT_MODE = 2;

#define FLOAT_OVERFLOW_MODE_CTRL 60
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float POS_INFINITY = INFINITY;
constexpr float NEG_INFINITY = -INFINITY;

__aicore__ inline int32_t CeilDiv(int32_t a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

// Decode a flat slot into its physical offset in a 4D paged buffer
// [blockNum, blockSize, 1, headDim] whose blockNum dimension is non-contiguous:
//   offset = (slot / blockSize) * blockStride + (slot % blockSize) * rowStride
// blockSize==1 (contiguous default) degrades to slot * rowStride.
__aicore__ inline int64_t PagedSlotOffset(int64_t slot, int64_t blockSize, int64_t blockStride, int64_t rowStride)
{
    int64_t block = slot / blockSize;
    int64_t pos = slot % blockSize;
    return block * blockStride + pos * rowStride;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num, int32_t elemNum)
{
    return CeilAlign(num, elemNum);
}

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitF32toFp8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitU32toU8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

// B16->B32 拆偶/拆奇布局 (与 swiglu_group_quant 一致): Layout0 取偶数位元素, Layout1 取奇数位元素。
// castTraitB162B32Even 即 Layout0 (RegLayout::ZERO), 此处补充 Layout1 (RegLayout::ONE)。
constexpr AscendC::MicroAPI::CastTrait traitB16ToB32Layout1 = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitF32toh8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND
};


template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float>& dst, __local_mem__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOutputData(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toFp8Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, hifloat8_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toh8>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreMxFp8Scale(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    RegTensor<uint32_t> tmp;
    RegTensor<uint8_t> tmp1;
    ShiftRights(tmp, (RegTensor<uint32_t> &)src, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregLoop);
    Cast<uint8_t, uint32_t, castTraitU32toU8Even>(tmp1, tmp, pregLoop);
    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B8>(dst + dstOffset, (RegTensor<T> &)tmp1, pregLoop);
}


// MX-FP8 量化, 每 32 个元素一个 e8m0 scale。两函数式实现 (与 moe_v3_common.h 的
// VFProcessSwigluMxFp8InvScale + VFProcessSwigluMxFp8Quant 一致, 在其基础上做最小修改并适配 indexer):
//   - VFProcessMxFp8InvScale: 每次处理 128 个 B16 (=4 个 MX 块)。把 x 拆偶/拆奇 cast 成两路 fp32,
//     Interleave 还原自然顺序写入 fp32 中间 buffer yLocal; 用 Abs->Max->ReduceMaxWithDataBlock->
//     DeInterleave->Max 求每 32 块 amax; 算 e8m0 scale 与 invScale, 把 invScale 经 2 次 Interleave
//     广播 (每 scale x4) 存入 invScaleLocal, e8m0 字节存入 scaleLocal。
//   - VFProcessMxFp8Quant: 从 yLocal 读 fp32, 从 invScaleLocal 经 DIST_E2B_B32 广播 (每元素 x8) 读
//     invScale, 相乘后 cast 成 fp8 写出。
// 适配点: (1) indexer 的 scale 输出每行按 RoundUp(scaleCol,32) 跨步, 故每行把 scale 字节流基址复位到
//   本行 32B 对齐基址 (moe 是跨行连续);  (2) moe 假设 128 对齐, indexer 仅 32 对齐, 最后一个 128/64
//   chunk 用 curBlocks/curChunk 掩码适配尾块;  (3) MX 含 round/非 round 两条 scale 计算路径 (moe 只有
//   round): 非 round 时 e8m0 存 amax/fp8max 的指数, invScale 取 amax/fp8max 的倒数。
//   越界读规避: 128 宽读最多到 RoundUp(d,128), 故 x 输入 buffer 按 128 对齐 (见 mx_fp8 调用方/tiling)。
template <typename TX, typename TS, bool roundScale = true>
__aicore__ inline void VFProcessMxFp8InvScale(
    const LocalTensor<float>& yLocal, const LocalTensor<TS>& scaleLocal, const LocalTensor<float>& invScaleLocal,
    const LocalTensor<TX>& xLocal, float coeff, const uint16_t curRowNum, const uint32_t curColNum)
{
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ uint8_t* scaleOriginAddr = (__local_mem__ uint8_t*)scaleLocal.GetPhyAddr();
    __local_mem__ float* invScaleLocalAddr = (__local_mem__ float*)invScaleLocal.GetPhyAddr();
    __local_mem__ TX* xLocalAddr = (__local_mem__ TX*)xLocal.GetPhyAddr();
    constexpr uint32_t MXB = 32;
    uint16_t vlLen = REPEAT_SIZE / sizeof(TX);                 // 128 elements (B16) per chunk
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint32_t xRowAlign = RoundUp<TX>(curColNum, vlLen);        // x 输入 buffer 按 128 对齐 (enlarged)
    uint32_t curColNumAlignFloat = RoundUp<float>(curColNum);  // fp32 中间 buffer 每行跨步
    uint32_t scaleCol = CeilDiv(curColNum, MXB);
    uint32_t scaleColNumAlign = RoundUp<TS>(scaleCol);         // e8m0 scale 每行 32B 对齐跨步
    uint32_t invScaleColNumAlign = RoundUp<float>(CeilDiv(curColNum, 8));
    __VEC_SCOPE__
    {
        RegTensor<TX> raw;
        RegTensor<float> x0Layout0;
        RegTensor<float> x0Layout1;
        RegTensor<float> y0;
        RegTensor<float> y1;
        RegTensor<float> yMax0;
        RegTensor<float> yMax1;
        RegTensor<float> yMax1Layout0;
        RegTensor<float> yMax1Layout1;
        RegTensor<float> scale;
        RegTensor<float> clampScale;
        RegTensor<float> invScale;
        RegTensor<float> invScale0;
        RegTensor<float> invScale1;
        RegTensor<float> one;
        RegTensor<uint32_t> scale2;
        RegTensor<uint32_t> scale3;
        RegTensor<uint32_t> scaleSel;
        RegTensor<int32_t> scale4;
        RegTensor<int32_t> scale5;
        RegTensor<uint8_t> scale6;
        RegTensor<uint16_t> scale7;
        RegTensor<uint32_t> zeroUint32;
        RegTensor<uint32_t> oneUint32;
        RegTensor<uint32_t> cAndExp;
        RegTensor<uint32_t> cAndMan;
        RegTensor<uint32_t> c127;
        RegTensor<int32_t> c127Int;
        RegTensor<uint8_t> tmp5;
        UnalignReg uReg;
        MaskReg pregMain0 = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMain1 = CreateMask<TX, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg compareMask0;
        Duplicate(one, 1.0f, pregMain0);
        Duplicate(zeroUint32, static_cast<uint32_t>(0), pregMain0);
        Duplicate(oneUint32, static_cast<uint32_t>(1), pregMain0);
        Duplicate(cAndExp, FAST_LOG_AND_VALUE1, pregMain0);
        Duplicate(cAndMan, FAST_LOG_AND_VALUE2, pregMain0);
        Duplicate(c127, static_cast<uint32_t>(127), pregMain0);
        Duplicate(c127Int, static_cast<int32_t>(127), pregMain0);
        for (uint16_t i = 0; i < curRowNum; i++) {
            __local_mem__ uint8_t* scaleRowAddr = scaleOriginAddr + i * scaleColNumAlign;  // 本行 32B 对齐基址
            for (uint16_t j = 0; j < loopCount; j++) {
                uint32_t curChunk = (curColNum - j * vlLen) < vlLen ? (curColNum - j * vlLen) : vlLen;
                uint32_t curBlocks = curChunk / MXB;        // 本 chunk 有效 MX 块数 (尾块 <4)
                // UpdateMask 会消耗(置零)其入参, 故用独立副本; curBlocks 后面还要用作 scale 字节数
                uint32_t mergeCnt = curBlocks;
                uint32_t merge1Cnt = 4 * curBlocks;
                MaskReg pregMerge = UpdateMask<float>(mergeCnt);      // scale lane 掩码
                MaskReg pregMerge1 = UpdateMask<float>(merge1Cnt);    // invScale 广播 lane 掩码
                // 载入 128 个 B16, 拆偶/拆奇 -> 两路 fp32, Interleave 还原自然顺序写 fp32 中间 buffer
                DataCopy(raw, xLocalAddr + i * xRowAlign + j * vlLen);
                Cast<float, TX, castTraitB162B32Even>(x0Layout0, raw, pregMain1);  // even
                Cast<float, TX, traitB16ToB32Layout1>(x0Layout1, raw, pregMain1);  // odd
                Interleave(y0, y1, x0Layout0, x0Layout1);
                uint32_t n0 = curChunk < static_cast<uint32_t>(VL_FP32) ? curChunk : static_cast<uint32_t>(VL_FP32);
                MaskReg my0 = UpdateMask<float>(n0);
                StoreOutputData<float>(yLocalAddr, y0, my0, 2 * j * VL_FP32 + i * curColNumAlignFloat);
                if (curChunk > static_cast<uint32_t>(VL_FP32)) {
                    uint32_t n1 = curChunk - static_cast<uint32_t>(VL_FP32);
                    MaskReg my1 = UpdateMask<float>(n1);
                    StoreOutputData<float>(yLocalAddr, y1, my1, (2 * j + 1) * VL_FP32 + i * curColNumAlignFloat);
                }
                // per-32-block amax
                Abs(x0Layout0, x0Layout0, pregMain0);
                Abs(x0Layout1, x0Layout1, pregMain0);
                Max(yMax0, x0Layout0, x0Layout1, pregMain0);
                ReduceMaxWithDataBlock(yMax1, yMax0, pregMain0);
                DeInterleave(yMax1Layout0, yMax1Layout1, yMax1, yMax1);
                Max(scale, yMax1Layout0, yMax1Layout1, pregMerge);
                Maxs(clampScale, scale, static_cast<float>(1e-4), pregMerge);   // amax
                Muls(scale, clampScale, coeff, pregMerge);                      // sf = amax / fp8max
                if constexpr (roundScale) {
                    // e8m0 = exp + (mantissa != 0); 存储 scale = 2^(exp_scale); invScale = 2^(-exp_scale)
                    ShiftRights(scale2, (RegTensor<uint32_t>&)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                    And(scale2, scale2, cAndExp, pregMerge);                    // exp
                    And(scale3, (RegTensor<uint32_t>&)scale, cAndMan, pregMerge);  // man_bits
                    Compare<uint32_t, AscendC::CMPMODE::NE>(compareMask0, scale3, zeroUint32, pregMerge);
                    Select(scaleSel, oneUint32, zeroUint32, compareMask0);     // man_bits != 0
                    Sub(scale3, scale2, c127, pregMerge);                       // exp - 127
                    Add((RegTensor<uint32_t>&)scale4, scale3, scaleSel, pregMerge);  // exp_scale
                    Adds(scale5, scale4, 127, pregMerge);                       // exp_scale + 127
                    ShiftLefts((RegTensor<int32_t>&)scale, scale5, static_cast<int16_t>(23), pregMerge);  // 存储 scale
                    Sub(scale5, c127Int, scale4, pregMerge);                    // 127 - exp_scale
                    ShiftLefts((RegTensor<int32_t>&)invScale, scale5, static_cast<int16_t>(23), pregMerge);  // 2^(-exp_scale)
                } else {
                    // 非 round: e8m0 存 amax/fp8max 的指数, invScale = 1 / (amax/fp8max) (倒数)
                    Div(invScale, one, scale, pregMerge);
                }
                // e8m0 字节存储 (尾块只写 curBlocks 字节, 落入本行 32B scratch)
                ShiftRights(scale2, (RegTensor<uint32_t>&)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                Cast<uint8_t, int32_t, castTraitU32toU8Even>(tmp5, (RegTensor<int32_t>&)scale2, pregMerge);
                Pack(scale7, (RegTensor<uint32_t>&)tmp5);
                Pack(scale6, scale7);
                DataCopyUnAlign<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    scaleRowAddr, scale6, uReg, curBlocks);
                // invScale 广播 (每 scale x4) 存入 invScaleLocal
                Interleave(invScale0, invScale1, invScale, invScale);
                Interleave(invScale1, invScale, invScale0, invScale0);
                StoreOutputData<float>(invScaleLocalAddr, invScale1, pregMerge1, j * 16 + i * invScaleColNumAlign);
            }
            DataCopyUnAlignPost(scaleRowAddr, uReg, 0);
        }
    }
}

template <typename TY>
__aicore__ inline void VFProcessMxFp8Quant(
    const LocalTensor<TY>& yQuantLocal, const LocalTensor<float>& yLocal, const LocalTensor<float>& invScaleLocal,
    const uint16_t curRowNum, const uint32_t curColNum)
{
    __local_mem__ TY* yQuantLocalAddr = (__local_mem__ TY*)yQuantLocal.GetPhyAddr();
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)invScaleLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<TY>(curColNum);
    uint32_t invScaleColNumAlign = RoundUp<float>(CeilDiv(curColNum, 8));
    __VEC_SCOPE__
    {
        RegTensor<float> y;
        RegTensor<float> invScale;
        MaskReg pregMain;
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregMain = UpdateMask<float>(sreg);   // 尾块按剩余元素数掩码
                LoadInputData<float>(y, yLocalAddr, pregMain, i * curColNumAlign + j * VL_FP32);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_E2B_B32>(
                    invScale, scaleLocalAddr + j * 8 + i * invScaleColNumAlign);
                Mul(y, y, invScale, pregMain);
                StoreOutputData<TY>(yQuantLocalAddr, y, pregMain, j * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }
}

// Normal 动态量化, 整行一个 float32 scale (scale = rowmax / fp8max)。两遍处理:
// 第一遍跨整行求 amax, 第二遍按该 scale 量化。NaN 安全 (排除 NaN 求 max, 结果 NaN 时保留原值)。
// roundScale=true 时把 scale 向上取到最近的 2 的幂 (与 kv_compress_epilog 的 roundScale 一致)。
template <typename T0, typename T1, bool roundScale = true>
__aicore__ inline void VFProcessDynamicBlockQuant(
    const LocalTensor<T0>& yLocal, const LocalTensor<float>& scaleLocal, const LocalTensor<T1>& xLocal,
    float coeff, const uint16_t curRowNum, const uint32_t curColNum)
{
    (void)coeff;
    __local_mem__ T0* yLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ T1* xLocalAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint32_t scaleRowAlign = RoundUp<float>(1);  // 整行一个 scale
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<T0, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<T0, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    } else if constexpr (IsSameType<T0, hifloat8_t>::value) {
        maxValueInt = INV_HIFP8_MAX_VALUE;
    }

    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> xAbs;
        RegTensor<float> partMax;
        RegTensor<float> rowMax;
        RegTensor<float> x0;
        RegTensor<float> x1;
        RegTensor<float> scale;
        RegTensor<float> one;
        RegTensor<float> invScale;
        RegTensor<float> dupInvScale;
        RegTensor<float> inf;
        RegTensor<float> zero;
        RegTensor<uint32_t> coeffReg;
        RegTensor<uint32_t> rsExp;
        RegTensor<uint32_t> rsInvExp;
        RegTensor<uint32_t> rsMant;
        RegTensor<uint32_t> rsSel;
        RegTensor<uint32_t> rsZeroU;
        RegTensor<uint32_t> rsOneU;
        RegTensor<uint32_t> rs254;
        RegTensor<uint32_t> rsMantMask;
        MaskReg pregMain = CreateMask<float>();
        MaskReg preg1 = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg pm;
        MaskReg cmpAbs;
        MaskReg cmpScalar;
        MaskReg cmpQ;
        MaskReg rsCmp;
        Duplicate(coeffReg, maxValueInt, pregMain);
        Duplicate(zero, 0.0f, pregMain);
        Duplicate(one, 1.0f, pregMain);
        Duplicate(inf, 1.0f, pregMain);
        Div<float, &mode>(inf, inf, zero, pregMain);
        if constexpr (roundScale) {
            Duplicate(rsZeroU, static_cast<uint32_t>(0), preg1);
            Duplicate(rsOneU, static_cast<uint32_t>(1), preg1);
            Duplicate(rs254, static_cast<uint32_t>(254), preg1);
            Duplicate(rsMantMask, FAST_LOG_AND_VALUE2, preg1);
        }
        for (uint16_t i = 0; i < curRowNum; i++) {
            // pass 1: 整行 amax (排除 NaN)
            Duplicate(rowMax, 0.0f, preg1);
            uint32_t sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pm = UpdateMask<float>(sreg);
                LoadInputData<T1>(x, xLocalAddr, pm, j * VL_FP32 + i * curColNumAlign);
                Muls(xAbs, x, 0.0f, pm);
                Compare<float, CMPMODE::NE>(cmpAbs, xAbs, xAbs, pm);
                MaskNot(cmpAbs, cmpAbs, pm);
                Abs(xAbs, x, cmpAbs);
                ReduceMax(partMax, xAbs, cmpAbs);
                Max(rowMax, rowMax, partMax, preg1);
            }
            CompareScalar<float, CMPMODE::NE>(cmpScalar, rowMax, (float)0.0, preg1);
            Mul(scale, rowMax, (RegTensor<float>&)coeffReg, cmpScalar);  // rowmax / fp8max
            Min(scale, scale, inf, preg1);
            // 与 VFProcessSwigluGroupQuant 的 if constexpr (!hasRoundScale) {...} else {...} 结构一致:
            // 两条路径各自算出 invScale, pass2 统一用 Mul(x, invScale)。
            if constexpr (!roundScale) {
                // 非 round: 存储 scale = rowmax/fp8max, invScale = 1/scale (倒数)。
                Div<float, &mode>(invScale, one, scale, preg1);
            } else {
                // round: scale 向上取最近的 2 的幂 (exp + (mantissa != 0)), invScale = 2^-(exp_scale)。
                ShiftRights(rsExp, (RegTensor<uint32_t>&)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), preg1);
                And(rsMant, (RegTensor<uint32_t>&)scale, rsMantMask, preg1);
                Compare<uint32_t, AscendC::CMPMODE::NE>(rsCmp, rsMant, rsZeroU, preg1);
                Select(rsSel, rsOneU, rsZeroU, rsCmp);
                Add(rsExp, rsExp, rsSel, preg1);   // 存储 scale 的指数字段 (scale = 2^(rsExp-127))
                ShiftLefts((RegTensor<int32_t>&)scale, (RegTensor<int32_t>&)rsExp, static_cast<int16_t>(23), preg1);
                Sub((RegTensor<int32_t>&)rsInvExp, (RegTensor<int32_t>&)rs254, (RegTensor<int32_t>&)rsExp, preg1);
                ShiftLefts((RegTensor<int32_t>&)invScale, (RegTensor<int32_t>&)rsInvExp, static_cast<int16_t>(23), preg1);
            }
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                scaleLocalAddr + i * scaleRowAlign, scale, preg1);
            // pass 2: 两条路径统一 Mul(x, invScale)。NaN 安全 (结果 NaN 时保留原值)。
            Duplicate(dupInvScale, invScale, pregMain);
            sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pm = UpdateMask<float>(sreg);
                LoadInputData<T1>(x, xLocalAddr, pm, j * VL_FP32 + i * curColNumAlign);
                Mul(x0, x, dupInvScale, pm);
                Muls(x1, x0, 0.0f, pm);
                Compare<float, CMPMODE::NE>(cmpQ, x1, x1, pm);
                Select(x, x, x0, cmpQ);
                StoreOutputData<T0>(yLocalAddr, x, pm, j * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }
}

template <typename T0, typename T1>
__aicore__ inline void VFProcessHifp8Quant(
    const LocalTensor<T0>& yLocal, const LocalTensor<float>& scaleLocal, const LocalTensor<T1>& xLocal,
    float coeff, const uint16_t curRowNum, const uint32_t curColNum, const float scales)
{
    LocalTensor<hifloat8_t> outLocal = yLocal.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* yLocalAddr = (__local_mem__ hifloat8_t*)outLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ T1* xLocalAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<hifloat8_t> tmp;
        MaskReg pregLoop = CreateMask<float>();
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T1>(xReg, xLocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
                Muls(xReg, xReg, scales, pregLoop);
                Cast<hifloat8_t, float, castTraitF32toh8>(tmp, xReg, pregLoop);
                DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    yLocalAddr + j * VL_FP32 + i * dstCurColNumAlign, tmp, pregLoop);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CopyIn(
    const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}


template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
__aicore__ inline void CopyOut(
    const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad<T, mode>(outputGm, outputTensor, dataCopyParams);
}

} // namespace IndexerQuantCache

#endif