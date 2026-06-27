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
 * \file quant_lightning_indexer_vector1.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_VECTOR1_H
#define QUANT_LIGHTNING_INDEXER_VECTOR1_H

#include "kernel_operator.h"

namespace vector1 {

template <typename T>
struct FloatSortTraits;

// fp32
template <>
struct FloatSortTraits<float> {
    using UInt = uint32_t;
    static constexpr UInt ZERO      = 0x00000000;
    static constexpr UInt SIGN_MASK = 0x80000000;
    static constexpr UInt NAN_MASK  = 0x7FC00000;
    static constexpr UInt ALL_ONE   = 0xFFFFFFFF;
};

// bf16
template <>
struct FloatSortTraits<bfloat16_t> {
    using UInt = uint16_t;
    static constexpr UInt ZERO      = 0x0000;
    static constexpr UInt SIGN_MASK = 0x8000;
    static constexpr UInt NAN_MASK  = 0x7FC0;
    static constexpr UInt ALL_ONE   = 0xFFFF;
};


template <typename FloatT>
struct FloatSortConstCtx {
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;
    AscendC::MicroAPI::RegTensor<UInt> zeros;
    AscendC::MicroAPI::RegTensor<UInt> allOne;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};


template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT>& ctx, AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros,    Traits::ZERO,      maskAll);
    AscendC::MicroAPI::Duplicate(ctx.allOne,   Traits::ALL_ONE,   maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan,      Traits::NAN_MASK,  maskAll);
}


template <typename FloatT>
__simd_callee__ inline void FloatToSortableKey(AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey,
                                               AscendC::MicroAPI::RegTensor<FloatT>& inVal,
                                               FloatSortConstCtx<FloatT>& ctx,
                                               AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp;
    AscendC::MicroAPI::RegTensor<UInt> regMask;
    AscendC::MicroAPI::MaskReg regSelectNan;
    AscendC::MicroAPI::MaskReg regSelectSign;

    auto& inBits = (AscendC::MicroAPI::RegTensor<UInt>&)inVal;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan, inBits, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey, ctx.allOne, inBits, regSelectNan);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.allOne, ctx.signMask, regSelectSign);
    AscendC::MicroAPI::Xor(outKey, outKey, regMask, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatX2ToSortableKey(AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey0,
                                                 AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey1,
                                                 AscendC::MicroAPI::RegTensor<FloatT>& inVal0,
                                                 AscendC::MicroAPI::RegTensor<FloatT>& inVal1,
                                                 FloatSortConstCtx<FloatT>& ctx,
                                                 AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp[2];
    AscendC::MicroAPI::RegTensor<UInt> regMask[2];
    AscendC::MicroAPI::MaskReg regSelectNan[2];
    AscendC::MicroAPI::MaskReg regSelectSign[2];

    auto& inBits0 = (AscendC::MicroAPI::RegTensor<UInt>&)inVal0;
    auto& inBits1 = (AscendC::MicroAPI::RegTensor<UInt>&)inVal1;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[0], inBits0, ctx.nan, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[1], inBits1, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey0, ctx.allOne, inBits0, regSelectNan[0]);
    AscendC::MicroAPI::Select(outKey1, ctx.allOne, inBits1, regSelectNan[1]);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp[0], outKey0, ctx.signMask, maskAll);
    AscendC::MicroAPI::And(regTemp[1], outKey1, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[0], regTemp[0], ctx.zeros, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[1], regTemp[1], ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask[0], ctx.allOne, ctx.signMask, regSelectSign[0]);
    AscendC::MicroAPI::Select(regMask[1], ctx.allOne, ctx.signMask, regSelectSign[1]);
    AscendC::MicroAPI::Xor(outKey0, outKey0, regMask[0], maskAll);
    AscendC::MicroAPI::Xor(outKey1, outKey1, regMask[1], maskAll);
}

// float in uint32 out
template <typename QK_T>
__simd_vf__ inline void MulRSquareAndAddVBiasVFImpl(__ubuf__ uint32_t* outBuf,
                                                    __ubuf__ QK_T* qkBuf,
                                                    __ubuf__ float* vBiasBuf,
                                                    const float rSquare,
                                                    const int gS1BasePerVecSize_,
                                                    const int s2BaseSize,
                                                    const int mrgValueLen)
{
    MicroAPI::RegTensor<float> regvBias[4];
    MicroAPI::RegTensor<float> regQK[4];
    MicroAPI::RegTensor<float> regQKMul[4];
    MicroAPI::RegTensor<float> regQKBias[4];
    MicroAPI::RegTensor<uint32_t> regOut[4];
    MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    // float -> uint32相关参数结构体
    FloatSortConstCtx<float> fp32Ctx;
    InitFloatSortConstCtx(fp32Ctx, maskAllB32);

    // s2BaseSize固定为256时，搬运vbias写在外面性能最佳
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[0], vBiasBuf);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[1], vBiasBuf + 64);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[2], vBiasBuf + 128);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[3], vBiasBuf + 192);

    // 一次处理s2BaseSize 256个元素
    for (uint16_t i = 0; i < (uint16_t)(gS1BasePerVecSize_); ++i) {
        MicroAPI::LoadAlign<float>(regQK[0], qkBuf + s2BaseSize * i); // RowStride是128, 行都落在一个bank上
        MicroAPI::LoadAlign<float>(regQK[1], qkBuf + s2BaseSize * i + 64);
        MicroAPI::LoadAlign<float>(regQK[2], qkBuf + s2BaseSize * i + 128);
        MicroAPI::LoadAlign<float>(regQK[3], qkBuf + s2BaseSize * i + 192);
        // qk * (stem_block_size / stem_stride) ^ 2
        MicroAPI::Muls(regQKMul[0], regQK[0], rSquare, maskAllB32);
        MicroAPI::Muls(regQKMul[1], regQK[1], rSquare, maskAllB32);
        MicroAPI::Muls(regQKMul[2], regQK[2], rSquare, maskAllB32);
        MicroAPI::Muls(regQKMul[3], regQK[3], rSquare, maskAllB32);
        // + vbias [1，s2BaseSize] 重复gS1BasePerVecSize_行
        MicroAPI::Add(regQKBias[0], regQKMul[0], regvBias[0], maskAllB32);
        MicroAPI::Add(regQKBias[1], regQKMul[1], regvBias[1], maskAllB32);
        MicroAPI::Add(regQKBias[2], regQKMul[2], regvBias[2], maskAllB32);
        MicroAPI::Add(regQKBias[3], regQKMul[3], regvBias[3], maskAllB32);
        // float -> uint32
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regQKBias[0], regQKBias[1], fp32Ctx, maskAllB32);
        FloatX2ToSortableKey<float>(regOut[2], regOut[3], regQKBias[2], regQKBias[3], fp32Ctx, maskAllB32);
        // 搬入outBuf的value部分
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i), regOut[0], maskAllB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i) + 64, regOut[1], maskAllB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i) + 128, regOut[2], maskAllB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i) + 192, regOut[3], maskAllB32);
    }
}

/**
 * @brief MUL RSquare后加上偏移
 * @param outLocal VEC1结果 转为uint32_t
 * @param qkLocal mm1 res的结果 注意力分数
 * @param vBiasLocal kb的偏移向量
 * @param rSquare (stemStride / stemBlockSize) ^ 2
 * @param gS1BasePerVecSize_ g1S方向每个vec核分到基本块大小
 * @param s2BaseSize s2方向每个vec核分到基本块大小
 * @param mrgValueLen 每次流式topk分到的长度
 */
template<typename QK_T>
__aicore__ inline void MulRSquareAndAddVBiasVF(const LocalTensor<uint32_t> &outLocal,   // out    [S2Base]       [256   ]
                                               const LocalTensor<QK_T> &qkLocal,        // q*k^t  [qbG, S2Base]  [48 256]
                                               const LocalTensor<float> &vBiasLocal,    // vBias  [1, S2Base]    [256   ]
                                               const float rSquare,
                                               const int gS1BasePerVecSize_,
                                               const int s2BaseSize,
                                               const int mrgValueLen)
{
    __ubuf__ QK_T * qkBuf = (__ubuf__ QK_T *)qkLocal.GetPhyAddr();
    __ubuf__ float * vBiasBuf = (__ubuf__ float *)vBiasLocal.GetPhyAddr();
    __ubuf__ uint32_t * outBuf = (__ubuf__ uint32_t *)outLocal.GetPhyAddr();

    MulRSquareAndAddVBiasVFImpl(outBuf, qkBuf, vBiasBuf, rSquare, gS1BasePerVecSize_, s2BaseSize, mrgValueLen);
}

__simd_vf__ inline void PackSparseIndicesVFImpl(__ubuf__ uint32_t* outBuf,
                                                __ubuf__ uint32_t* topkBuf,
                                                const uint32_t copyLen,
                                                const uint32_t initialBlocks,
                                                const uint32_t topkSelectNum,
                                                const uint32_t windowSize,
                                                const uint32_t windowBase)
{
    // 创建 uint32 全 lane 有效的掩码，后续 VF 指令默认处理 64 个 uint32。
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

    // 当前 64-lane 内的全局输出列号，取值为 i * 64 到 i * 64 + 63。
    // Arange 不支持 uint32_t，这里先用 int32_t 生成，再在无符号计算处按 uint32_t 解释。
    MicroAPI::RegTensor<int32_t> idx;
    // 当前 64-lane 的拼接输出寄存器，最终写回 outBuf。
    MicroAPI::RegTensor<uint32_t> out;
    // topK 段访问 topkBuf 时使用的相对索引，等于 idx - initialBlocks。
    MicroAPI::RegTensor<uint32_t> topkIdx;
    // 从 topkBuf gather 出来的 topK 索引值。
    MicroAPI::RegTensor<uint32_t> topkValue;
    // window 段生成的连续索引值，等于 windowBase + idx - topkEnd。
    MicroAPI::RegTensor<uint32_t> windowValue;
    // 保存 initialBlocks 的广播寄存器，用于计算 topK 相对索引。
    MicroAPI::RegTensor<uint32_t> initialBlocksValue;
    // 保存 topkEnd 的广播寄存器，用于计算 window 相对索引。
    MicroAPI::RegTensor<uint32_t> topkEndValue;
    // idx 的 uint32_t 视图，用于和 uint32_t 的 mask/Select/Sub/Gather 指令配合。
    auto& idxU32 = (MicroAPI::RegTensor<uint32_t>&)idx;

    // copyLen 可能不是 64 的整数倍，这里向上取整，最后一组多写的 lane 仍在 mrgValueLocal_ 行容量内。
    uint16_t vfLoop = static_cast<uint16_t>((copyLen + 63U) / 64U);
    // topK 段结束位置，也是 window 段起始位置。
    uint32_t topkEnd = initialBlocks + topkSelectNum;
    // window 段结束位置，超过该位置的 lane 保持 -1。
    uint32_t windowEnd = topkEnd + windowSize;
    // 只有存在 topK 段时才准备 initialBlocks 广播寄存器，避免无用 VF 指令。
    if (topkSelectNum > 0U) {
        // 将 initialBlocks 广播到寄存器，供 Sub(topkIdx, idx, initialBlocksValue) 使用。
        MicroAPI::Duplicate(initialBlocksValue, initialBlocks, pregB32);
    }
    // 只有存在 window 段时才准备 topkEnd 广播寄存器，避免无用 VF 指令。
    if (windowSize > 0U) {
        // 将 topkEnd 广播到寄存器，供 Sub(windowValue, idx, topkEndValue) 使用。
        MicroAPI::Duplicate(topkEndValue, topkEnd, pregB32);
    }

    // 每轮处理 64 个 uint32 输出元素，拼成 [init][topK][window][-1...] 的固定前缀行。
    for (uint16_t i = 0; i < vfLoop; ++i) {
        // 生成当前 64-lane 对应的输出列号。
        MicroAPI::Arange(idx, static_cast<int32_t>(i) * 64);
        // 默认把整组 lane 填为 -1，后续 init/topK/window 只覆盖各自有效区间。
        MicroAPI::Duplicate(out, static_cast<uint32_t>(-1), pregB32);

        // initialBlocks > 0 时，输出前缀 [0, initialBlocks) 的连续索引。
        if (initialBlocks > 0U) {
            // 声明 init 段掩码，表示 idx < initialBlocks 的 lane。
            MicroAPI::MaskReg pregInit;
            // 直接得到 idx < initialBlocks，也就是 init 段。
            MicroAPI::Compares<uint32_t, CMPMODE::LT>(pregInit, idxU32, initialBlocks, pregB32);
            // 在 init 段把输出值替换为 idx，其余 lane 继续保持原 out 值。
            MicroAPI::Select(out, idxU32, out, pregInit);
        }

        // topkSelectNum > 0 时，把 topkBuf 中的 topK 索引拼到 init 段之后。
        if (topkSelectNum > 0U) {
            // topK 起始掩码，默认全有效；当 initialBlocks > 0 时再收窄为 idx >= initialBlocks。
            MicroAPI::MaskReg pregTopkStart = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            // topK 结束掩码，表示 idx < topkEnd。
            MicroAPI::MaskReg pregTopkEnd;
            // topK 最终掩码，表示 initialBlocks <= idx < topkEnd。
            MicroAPI::MaskReg pregTopk;
            // initialBlocks > 0 时，需要排除 init 段 lane。
            if (initialBlocks > 0U) {
                // 得到 idx >= initialBlocks，也就是 topK 段的起始条件。
                MicroAPI::Compares<uint32_t, CMPMODE::GE>(pregTopkStart, idxU32, initialBlocks, pregB32);
            }
            // 得到 idx < topkEnd，也就是 topK 段的结束条件。
            MicroAPI::Compares<uint32_t, CMPMODE::LT>(pregTopkEnd, idxU32, topkEnd, pregB32);
            // 合并 topK 起始和结束条件，得到完整 topK 段掩码。
            MicroAPI::And(pregTopk, pregTopkStart, pregTopkEnd, pregB32);
            // 将输出列号换算为 topkBuf 内的相对索引。
            MicroAPI::Sub(topkIdx, idxU32, initialBlocksValue, pregTopk);
            // 在 topK 段按相对索引从 topkBuf 中取出已经计算好的 globalIndex。
            MicroAPI::Gather(topkValue, topkBuf, topkIdx, pregTopk);
            // 在 topK 段把输出值替换为 gather 结果，其余 lane 继续保持原 out 值。
            MicroAPI::Select(out, topkValue, out, pregTopk);
        }

        // windowSize > 0 时，把尾部 window 的连续索引拼到 topK 段之后。
        if (windowSize > 0U) {
            // window 起始掩码，默认全有效；当 topkEnd > 0 时再收窄为 idx >= topkEnd。
            MicroAPI::MaskReg pregWindowStart = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            // window 结束掩码，表示 idx < windowEnd。
            MicroAPI::MaskReg pregWindowEnd;
            // window 最终掩码，表示 topkEnd <= idx < windowEnd。
            MicroAPI::MaskReg pregWindow;
            // topkEnd > 0 时，需要排除 init + topK 段 lane。
            if (topkEnd > 0U) {
                // 得到 idx >= topkEnd，也就是 window 段的起始条件。
                MicroAPI::Compares<uint32_t, CMPMODE::GE>(pregWindowStart, idxU32, topkEnd, pregB32);
            }
            // 得到 idx < windowEnd，也就是 window 段的结束条件。
            MicroAPI::Compares<uint32_t, CMPMODE::LT>(pregWindowEnd, idxU32, windowEnd, pregB32);
            // 合并 window 起始和结束条件，得到完整 window 段掩码。
            MicroAPI::And(pregWindow, pregWindowStart, pregWindowEnd, pregB32);
            // 先把输出列号换算为 window 段内的相对偏移。
            MicroAPI::Sub(windowValue, idxU32, topkEndValue, pregWindow);
            // 加上真实 window 起始位置，得到最终要写出的 K 方向索引。
            MicroAPI::Adds(windowValue, windowValue, windowBase, pregWindow);
            // 在 window 段把输出值替换为连续 window 索引，其余 lane 继续保持原 out 值。
            MicroAPI::Select(out, windowValue, out, pregWindow);
        }

        // 将当前 64-lane 写回 UB；写回位置始终 64 对齐，满足 StoreAlign 要求。
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + static_cast<uint32_t>(i) * 64U,
                                                                       out, pregB32);
    }
}

__aicore__ inline void PackSparseIndicesVF(const LocalTensor<uint32_t> &outLocal,
                                           const LocalTensor<uint32_t> &topkLocal,
                                           const uint32_t copyLen,
                                           const uint32_t initialBlocks,
                                           const uint32_t topkSelectNum,
                                           const uint32_t windowSize,
                                           const uint32_t windowBase)
{
    __ubuf__ uint32_t *outBuf = (__ubuf__ uint32_t *)outLocal.GetPhyAddr();
    __ubuf__ uint32_t *topkBuf = (__ubuf__ uint32_t *)topkLocal.GetPhyAddr();

    PackSparseIndicesVFImpl(outBuf, topkBuf, copyLen, initialBlocks, topkSelectNum, windowSize, windowBase);
}

}

#endif
