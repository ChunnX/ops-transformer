/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gated_activation.h
 * \brief A5 (arch35) gated activation implementations for BlockEpilogueActivationMxQuant.
 *
 * 本文件包含 4 种门控激活的完整计算流（主循环 + tail + pad），均为
 * BlockEpilogueActivationMxQuant 类模板的 out-of-class 成员函数定义：
 *   - ActivationFlowSwiGLU      : swish(gate) * up
 *   - ActivationFlowSiTU        : beta*tanh(gate/beta)*sigmoid(gate) * up [或 linear 变体]
 *   - ActivationFlowSwiGLUStep  : min(swish(gate), clamp) * clip(up, ±clamp)   [新增]
 *   - ActivationFlowSwiGLUOai   : (up+beta)*min(gate,clamp)*sigmoid(alpha*min(gate,clamp)) [新增]
 *
 * 启用方式：在 block_epilogue_activation_mx_quant.h 中，BlockEpilogueActivationMxQuant
 * 类声明完成之后、MegaMoeImpl 命名空间闭合之前，#include 本文件。
 *
 * 外部依赖（须由 include 方在 include 本文件之前提供）：
 *   - 命名空间 MegaMoeImpl 已打开，且 using namespace ActivationQuantMsg 已生效
 *   - 宏 BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS / BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS
 *   - ActivationContext<DataTypeIn> 结构体定义
 *   - BlockEpilogueActivationMxQuant 类模板完整声明（含 ActivationFlowSwiGLU 等成员声明）
 *   - ActivationQuantMsg 命名空间常量：VF_LEN_FP32, CAST_ZERO, CAST_FP32_TO_FP16_BF16 等
 *   - mega_moe_constants.h: INT32_PER_256B, WEIGHT_INDEX（经 mega_moe_utils.h 间接引入）
 */

#ifndef MEGA_MOE_ARCH35_GATED_ACTIVATION_H
#define MEGA_MOE_ARCH35_GATED_ACTIVATION_H

#if defined(__DAV_C310__)

/*
 * COMPUTE_TANH_TWOPATH: 两段式 tanh 宏（多项式路 + sigmoid 分解路）
 *   多项式路(|x|<0.6): x*(1 + c1*x^2 + c2*x^4 + c3*x^6 + c4*x^8), Horner + FusedMulDstAdd
 *   sigmoid路(|x|>=0.6): 2/(1+e^{-2x}) - 1, 天然保号无相消
 *   系数来源: tanh_dag.h / situ_mx_quant_common.h
 *   c1=-0.333327681, c2=0.133152977, c3=-0.0523039624, c4=0.0157396831
 * 用法: COMPUTE_TANH_TWOPATH(result, zReg, msk, oneReg, absReg, sqrReg,
 *                           polyReg, tmpReg, expReg, c1Reg, c2Reg, cmpMask);
 * 注意: 宏在函数内展开，所用寄存器须在调用点函数作用域内声明；
 *       宏会覆盖 sqrReg/polyReg/tmpReg/expReg/absReg/cmpMask，结果写入 result
 */
#define COMPUTE_TANH_TWOPATH(result, zReg, msk, oneReg, absReg, sqrReg, \
                             polyReg, tmpReg, expReg, c1Reg, c2Reg, cmpMask) \
    do { \
        AscendC::MicroAPI::Mul(sqrReg, zReg, zReg, msk); \
        AscendC::MicroAPI::Muls(polyReg, sqrReg, 0.0157396831f, msk); \
        AscendC::MicroAPI::Adds(polyReg, polyReg, -0.0523039624f, msk); \
        AscendC::MicroAPI::FusedMulDstAdd(polyReg, sqrReg, c2Reg, msk); \
        AscendC::MicroAPI::FusedMulDstAdd(polyReg, sqrReg, c1Reg, msk); \
        AscendC::MicroAPI::Mul(polyReg, polyReg, sqrReg, msk); \
        AscendC::MicroAPI::FusedMulDstAdd(polyReg, zReg, zReg, msk); \
        AscendC::MicroAPI::Muls(expReg, zReg, -2.0f, msk); \
        AscendC::MicroAPI::Exp(expReg, expReg, msk); \
        AscendC::MicroAPI::Adds(expReg, expReg, 1.0f, msk); \
        AscendC::MicroAPI::Div(tmpReg, oneReg, expReg, msk); \
        AscendC::MicroAPI::Muls(tmpReg, tmpReg, 2.0f, msk); \
        AscendC::MicroAPI::Adds(tmpReg, tmpReg, -1.0f, msk); \
        AscendC::MicroAPI::Abs(absReg, zReg, msk); \
        AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::GE>(cmpMask, absReg, \
                                                                      0.60000002384185791016f, msk); \
        AscendC::MicroAPI::Select(result, tmpReg, polyReg, cmpMask); \
    } while (0)

/*
 * ActivationFlowSwiGLU: SwiGLU 激活完整计算流
 *   swish(x1) * x2, 其中 swish(x) = x / (exp(-x) + 1) = x * sigmoid(x)
 *   含主循环 + 尾循环 + pad 填零，所有寄存器在本函数内声明
 */
BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
template <bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ActivationFlowSwiGLU(
    const ActivationContext<DataTypeIn> &ctx)
{
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t numZero = 0;
    uint32_t mask1Num = ctx.mask1Num;
    uint32_t mask2Num = ctx.mask2Num;
    uint32_t mask3Num = ctx.mask3Num;
    uint16_t dim0VfTimes = ctx.dim0VfTimes;
    uint16_t dim1VfTimes = ctx.dim1VfTimes;
    uint16_t dim1TailTimes = ctx.dim1TailTimes;
    uint16_t dim1Tail2 = ctx.dim1Tail2;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask);

                // === SwiGLU: swish(x1) * x2 ===
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask); // -x
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                 // exp(-x)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);    // exp(-x)+1
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);   // swish(x)=x/(exp(-x)+1)
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);   // swish(x)*y
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t tailIdx = 0; tailIdx < dim1TailTimes; tailIdx++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask1);

                // === SwiGLU: swish(x1) * x2 (tail, mask1) ===
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.activationTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t padIdx = 0; padIdx < dim1Tail2; padIdx++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.activationTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

/*
 * ActivationFlowSiTU: SiTU 激活完整计算流
 *   situ_a = beta * tanh(gate/beta) * sigmoid(gate)
 *   ActSub==LINEAR: out = situ_a * (alpha * tanh(up/alpha))
 *   ActSub==DEFAULT: out = situ_a * up
 *   tanh 采用两段式（多项式路 + sigmoid 分解路），保号无相消
 *   参照 situ_mx_quant_common.h::ComputeVfSitu，含主循环 + 尾循环 + pad 填零
 */
BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
template <ActSubMode ActSub, bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ActivationFlowSiTU(
    const ActivationContext<DataTypeIn> &ctx)
{
    // tanh 多项式近似的奇次项系数（8 阶 Padé 近似，仅取奇次项）
    // 参见 COMPUTE_TANH_TWOPATH 宏注释，c1/c2 对应 x^3/x^5 的系数
    const float tanhC1 = -0.333327681f; // c1: tanh 多项式 x^3 项系数，约 -1/3
    const float tanhC2 = 0.133152977f;  // c2: tanh 多项式 x^5 项系数，约 2/15
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t numZero = 0;
    uint32_t mask1Num = ctx.mask1Num;
    uint32_t mask2Num = ctx.mask2Num;
    uint32_t mask3Num = ctx.mask3Num;
    uint16_t dim0VfTimes = ctx.dim0VfTimes;
    uint16_t dim1VfTimes = ctx.dim1VfTimes;
    uint16_t dim1TailTimes = ctx.dim1TailTimes;
    uint16_t dim1Tail2 = ctx.dim1Tail2;

    __VEC_SCOPE__
    {
        // 通用寄存器（搬入/clamp/store）
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;

        // SiTU 专用寄存器
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> zReg;
        AscendC::MicroAPI::RegTensor<float> betaReg;
        AscendC::MicroAPI::RegTensor<float> oneReg;

        // tanh 两段式专用寄存器
        AscendC::MicroAPI::RegTensor<float> absReg;
        AscendC::MicroAPI::RegTensor<float> sqrReg;
        AscendC::MicroAPI::RegTensor<float> polyReg;
        AscendC::MicroAPI::RegTensor<float> tanhTmpReg;
        AscendC::MicroAPI::RegTensor<float> c1Reg;
        AscendC::MicroAPI::RegTensor<float> c2Reg;
        AscendC::MicroAPI::MaskReg cmpMaskReg;

        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);

        // 循环前常量初始化（全量广播，不依赖 mask）
        AscendC::MicroAPI::Duplicate(oneReg, scalarOne);
        AscendC::MicroAPI::Duplicate(c1Reg, tanhC1);
        AscendC::MicroAPI::Duplicate(c2Reg, tanhC2);

        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask);

                // === SiTU 激活（主循环, mask）===
                // gate路: z = gate / beta
                AscendC::MicroAPI::Muls(zReg, vregX1F, ctx.invBeta, mask);
                // tanh(gate/beta) — 两段式, 结果存 sigmoidReg
                COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask, oneReg, absReg, sqrReg, polyReg, tanhTmpReg, expReg,
                                     c1Reg, c2Reg, cmpMaskReg);
                // beta * sigmoid(gate) = beta / (exp(-gate)+1)
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask); // -gate
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                 // exp(-gate)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);    // exp(-gate)+1
                AscendC::MicroAPI::Duplicate(betaReg, ctx.beta, mask);        // betaReg = beta
                AscendC::MicroAPI::Div(expReg, betaReg, addsReg, mask);       // beta/(exp(-gate)+1)
                // situ_a = tanh(gate/beta) * beta * sigmoid(gate)
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, expReg, mask); // outFReg = situ_a

                if constexpr (ActSub == ActSubMode::LINEAR) {
                    // up路: alpha * tanh(up/alpha), 用 vregX1F 暂存 gate_result
                    AscendC::MicroAPI::Muls(vregX1F, outFReg, scalarOne, mask); // vregX1F = gate_result (保存)
                    AscendC::MicroAPI::Muls(zReg, vregX2F, ctx.invAlpha, mask); // zReg = up/alpha
                    COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask, oneReg, absReg, sqrReg, polyReg, tanhTmpReg, expReg,
                                         c1Reg, c2Reg, cmpMaskReg);
                    // alpha * tanh(up/alpha)
                    AscendC::MicroAPI::Muls(sigmoidReg, sigmoidReg, ctx.alpha, mask);
                    // out = gate_result * up_transformed
                    AscendC::MicroAPI::Mul(outFReg, vregX1F, sigmoidReg, mask);
                } else {
                    // SITU: out = situ_a * up
                    AscendC::MicroAPI::Mul(outFReg, outFReg, vregX2F, mask);
                }
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t tailIdx = 0; tailIdx < dim1TailTimes; tailIdx++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask1);

                // === SiTU 激活（尾循环, mask1）===
                AscendC::MicroAPI::Muls(zReg, vregX1F, ctx.invBeta, mask1);
                COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask1, oneReg, absReg, sqrReg, polyReg, tanhTmpReg, expReg,
                                     c1Reg, c2Reg, cmpMaskReg);
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Duplicate(betaReg, ctx.beta, mask1);
                AscendC::MicroAPI::Div(expReg, betaReg, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, expReg, mask1);

                if constexpr (ActSub == ActSubMode::LINEAR) {
                    AscendC::MicroAPI::Muls(vregX1F, outFReg, scalarOne, mask1);
                    AscendC::MicroAPI::Muls(zReg, vregX2F, ctx.invAlpha, mask1);
                    COMPUTE_TANH_TWOPATH(sigmoidReg, zReg, mask1, oneReg, absReg, sqrReg, polyReg, tanhTmpReg, expReg,
                                         c1Reg, c2Reg, cmpMaskReg);
                    AscendC::MicroAPI::Muls(sigmoidReg, sigmoidReg, ctx.alpha, mask1);
                    AscendC::MicroAPI::Mul(outFReg, vregX1F, sigmoidReg, mask1);
                } else {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, vregX2F, mask1);
                }
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.activationTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t padIdx = 0; padIdx < dim1Tail2; padIdx++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.activationTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

/*
 * ActivationFlowSwiGLUStep: SwiGLU-Step 激活完整计算流 [新增]
 *   公式: min(SiLU(gate), clamp) × clip(up, ±clamp)
 *   与 SwiGLU 差异:
 *     - gate 不做 Mins 裁剪（删除 SwiGLU 的 Mins(vregX1F, ..., clampLimit)）
 *     - 对 SiLU(gate) 结果做 Mins 上限裁剪（SwiGLU 无此步）
 *   寄存器声明与 SwiGLU 完全相同，无额外寄存器
 */
BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
template <bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ActivationFlowSwiGLUStep(
    const ActivationContext<DataTypeIn> &ctx)
{
    const float scalarOne = 1.0f;
    const float negScalarOne = -1.0f;
    bfloat16_t numZero = 0;
    uint32_t mask1Num = ctx.mask1Num;
    uint32_t mask2Num = ctx.mask2Num;
    uint32_t mask3Num = ctx.mask3Num;
    uint16_t dim0VfTimes = ctx.dim0VfTimes;
    uint16_t dim1VfTimes = ctx.dim1VfTimes;
    uint16_t dim1TailTimes = ctx.dim1TailTimes;
    uint16_t dim1Tail2 = ctx.dim1Tail2;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                // ===== SwiGLUStep 差异: gate 不做 Mins 裁剪 =====
                // (SwiGLU 有: Mins(vregX1F, vregX1F, ctx.clampLimit, mask); — 此处删除)
                // up 对称裁剪（与 SwiGLU 相同）
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask);

                // === SwiGLUStep: SiLU(gate) * up ===
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask); // -gate
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                 // exp(-gate)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);    // exp(-gate)+1
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);   // SiLU(gate)=gate/(exp(-gate)+1)

                // ===== SwiGLUStep 差异: 对 SiLU 结果做上限裁剪 =====
                AscendC::MicroAPI::Mins(sigmoidReg, sigmoidReg, ctx.clampLimit, mask);

                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);   // SiLU(gate)*up
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t tailIdx = 0; tailIdx < dim1TailTimes; tailIdx++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                // ===== SwiGLUStep 差异: gate 不裁剪 =====
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask1);

                // === SwiGLUStep: SiLU(gate) * up (tail, mask1) ===
                AscendC::MicroAPI::Muls(negReg, vregX1F, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);

                // ===== SwiGLUStep 差异: 对 SiLU 结果做上限裁剪 =====
                AscendC::MicroAPI::Mins(sigmoidReg, sigmoidReg, ctx.clampLimit, mask1);

                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.activationTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t padIdx = 0; padIdx < dim1Tail2; padIdx++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.activationTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

/*
 * ActivationFlowSwiGLUOai: SwiGLU-OAI 激活完整计算流 [新增]
 *   公式: (up + beta) × min(gate, clamp) × sigmoid(alpha × min(gate, clamp))
 *        = (clip(up,±clamp)+beta) × min(gate,clamp) / (1+exp(-alpha×min(gate,clamp)))
 *   与 SwiGLU 差异:
 *     a. up 裁剪后 += beta（Adds）
 *     b. sigmoid 分母中 -gate 改为 -alpha*gate（Muls 标量用 -ctx.alpha）
 *   gate 裁剪与 SwiGLU 相同（Mins 上限）
 *   寄存器声明与 SwiGLU 完全相同，无额外寄存器
 */
BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
template <bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::ActivationFlowSwiGLUOai(
    const ActivationContext<DataTypeIn> &ctx)
{
    const float scalarOne = 1.0f;
    bfloat16_t numZero = 0;
    uint32_t mask1Num = ctx.mask1Num;
    uint32_t mask2Num = ctx.mask2Num;
    uint32_t mask3Num = ctx.mask3Num;
    uint16_t dim0VfTimes = ctx.dim0VfTimes;
    uint16_t dim1VfTimes = ctx.dim1VfTimes;
    uint16_t dim1TailTimes = ctx.dim1TailTimes;
    uint16_t dim1Tail2 = ctx.dim1Tail2;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX1;
        AscendC::MicroAPI::RegTensor<DataTypeIn> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> negReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> zeroReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            if constexpr (TopkWeightsPrefetch) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    weightReg, ctx.weightUbAddr + dim0vfLoopIdx * INT32_PER_256B + WEIGHT_INDEX);
            }
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(
                    dim0vfLoopIdx, ctx.nSrcUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstSrc, srcIdxOffset);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondSrc, srcIdxOffset);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask);

                // gate: 上限裁剪（与 SwiGLU 相同）
                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask);
                // up: 对称裁剪（与 SwiGLU 相同）
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask);

                // ===== SwiGLUOai 差异 a: up += beta =====
                AscendC::MicroAPI::Adds(vregX2F, vregX2F, ctx.beta, mask);

                // === SwiGLUOai: (up+beta) * gate * sigmoid(alpha*gate) ===
                // ===== SwiGLUOai 差异 b: -alpha*gate（SwiGLU 用 -1.0f）=====
                AscendC::MicroAPI::Muls(negReg, vregX1F, -ctx.alpha, mask);  // -alpha*gate
                AscendC::MicroAPI::Exp(expReg, negReg, mask);                 // exp(-alpha*gate)
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);    // exp(-alpha*gate)+1
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask);   // gate*sigmoid(alpha*gate)
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);   // (up+beta)*gate*sigmoid(alpha*gate)
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(
                    dim0vfLoopIdx, ctx.nDstUbAligned, dim1vfLoopIdx, VF_LEN_FP32);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.gluResAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 =
                AscendC::MicroAPI::CreateAddrReg<DataTypeIn>(dim0vfLoopIdx, ctx.nSrcUbAligned);
            AscendC::MicroAPI::AddrReg outOffset1 =
                AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(dim0vfLoopIdx, ctx.nDstUbAligned);
            for (uint16_t tailIdx = 0; tailIdx < dim1TailTimes; tailIdx++) {
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX1, ctx.firstTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<DataTypeIn, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX2, ctx.secondTailAddr, srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX1F, vregX1, mask1);
                AscendC::MicroAPI::Cast<float, DataTypeIn, CAST_ZERO>(vregX2F, vregX2, mask1);

                AscendC::MicroAPI::Mins(vregX1F, vregX1F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Mins(vregX2F, vregX2F, ctx.clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2F, vregX2F, -ctx.clampLimit, mask1);

                // ===== SwiGLUOai 差异 a: up += beta =====
                AscendC::MicroAPI::Adds(vregX2F, vregX2F, ctx.beta, mask1);

                // === SwiGLUOai (tail, mask1) ===
                AscendC::MicroAPI::Muls(negReg, vregX1F, -ctx.alpha, mask1);
                AscendC::MicroAPI::Exp(expReg, negReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, vregX1F, addsReg, mask1);
                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask1);
                // === 激活结束 ===

                if constexpr (TopkWeightsPrefetch) {
                    AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask1);
                }

                AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    ctx.activationTailAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t padIdx = 0; padIdx < dim1Tail2; padIdx++) {
                AscendC::MicroAPI::Duplicate(zeroReg, numZero);
                AscendC::MicroAPI::DataCopy<bfloat16_t>(ctx.activationTailAddr2, zeroReg, outOffset1, mask3);
            }
        }
    }
}

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_GATED_ACTIVATION_H
