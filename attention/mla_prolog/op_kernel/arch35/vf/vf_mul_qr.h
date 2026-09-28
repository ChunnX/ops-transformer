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
 * \file vf_mul_qr.h
 * \brief
 */

#ifndef VF_MUL_QR_H
#define VF_MUL_QR_H

namespace MlaProlog {
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
// ===== 9201(Ascend A6) 阶段5批次2(5D/V6)优化: 常量 Duplicate 外提到循环外 =====
// 与索引无关的常量在循环内每次重复生成——官方《VF性能优化》"循环内指令分布优化"反例的原样修复。
// 每次调用省 repeatTimes-1 条指令。源自恢复版优化(60.48μs 态), 适配主线 Reg:: 命名空间。
template <typename T>
__simd_vf__ void MulQrVFImpl(__ubuf__ T *inputBuf, __ubuf__ T *outputBuf, __ubuf__ T *dequantScaleBrcbBuf,
                             const uint16_t floatRepSize, uint16_t repeatTimes, float quantScaleCkvRope,
                             uint64_t computeBlockAlign)
{
    Reg::MaskReg pregAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();

    // 常量外提: 与循环索引无关, 每次调用只做一次
    Reg::RegTensor<T> vregQuantScale;
    Reg::Duplicate(vregQuantScale, quantScaleCkvRope);

    for (uint16_t i = 0; i < repeatTimes; i++) {
        Reg::RegTensor<T> vregSrc;
        Reg::RegTensor<T> vregDequantScaleVrcb;
        Reg::RegTensor<T> vregMulScale;
        Reg::RegTensor<T> vregRes;
        uint16_t loopOffset = i * floatRepSize;             // 计算数据偏移
        uint16_t dequantLoopOffset = i * computeBlockAlign; // 动态量化参数偏移

        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregSrc, inputBuf + loopOffset);
        // broadcast量化系数
        Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vregDequantScaleVrcb, dequantScaleBrcbBuf + dequantLoopOffset);

        Reg::Div(vregMulScale, vregQuantScale, vregDequantScaleVrcb, pregAll);

        Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregSrc, vregMulScale, pregAll);

        Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregRes, pregAll);
    }
}
#else
template <typename T>
__simd_vf__ void MulQrVFImpl(__ubuf__ T *inputBuf, __ubuf__ T *outputBuf, __ubuf__ T *dequantScaleBrcbBuf,
                             const uint16_t floatRepSize, uint16_t repeatTimes, float quantScaleCkvRope,
                             uint64_t computeBlockAlign)
{
    Reg::MaskReg pregAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();

    for (uint16_t i = 0; i < repeatTimes; i++) {
        Reg::RegTensor<T> vregSrc;
        Reg::RegTensor<T> vregQuantScale;
        Reg::RegTensor<T> vregDequantScaleVrcb;
        Reg::RegTensor<T> vregMulScale;
        Reg::RegTensor<T> vregRes;
        uint16_t loopOffset = i * floatRepSize;             // 计算数据偏移
        uint16_t dequantLoopOffset = i * computeBlockAlign; // 动态量化参数偏移

        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregSrc, inputBuf + loopOffset);
        // broadcast量化系数
        Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vregDequantScaleVrcb, dequantScaleBrcbBuf + dequantLoopOffset);
        Reg::Duplicate(vregQuantScale, quantScaleCkvRope);

        Reg::Div(vregMulScale, vregQuantScale, vregDequantScaleVrcb, pregAll);

        Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregSrc, vregMulScale, pregAll);

        Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregRes, pregAll);
    }
}
#endif

/**
 * @brief MulQrVF 对于RopeQr输出的结果进行mul系数计算
 * @param outputLocal 输出tensor [subRow, colRope]，row目前均为1
 * @param inputLocal 输入tensor [subRow, colRope]
 * @param dequantScaleBrcbLocal 动态量化参数tensor [row, computeBlockAlign]
 * @param quantScaleCkvRope 量化系数
 * @param computeSizeRope 输入参数的大小 subRow * colRope
 * @param computeBlockAlign 输入动态量化参数的单个数据块存放多少个动态量化参数
 */
template <typename T>
__aicore__ inline void MulQrVF(const LocalTensor<T> &outputLocal, const LocalTensor<T> &inputLocal,
                               const LocalTensor<T> &dequantScaleBrcbLocal, float quantScaleCkvRope,
                               uint64_t computeSizeRope, uint64_t computeBlockAlign)
{
    const uint16_t floatRepSize = 64;                                           // 一个寄存器能够存放64个FP32
    uint16_t repeatTimes = (computeSizeRope + floatRepSize - 1) / floatRepSize; // 对尾块处理的扩展，循环处理的次数

    __ubuf__ T *inputBuf = (__ubuf__ T *)inputLocal.GetPhyAddr();
    __ubuf__ T *outputBuf = (__ubuf__ T *)outputLocal.GetPhyAddr();
    __ubuf__ T *dequantScaleBrcbBuf = (__ubuf__ T *)dequantScaleBrcbLocal.GetPhyAddr();
    MulQrVFImpl<T>(inputBuf, outputBuf, dequantScaleBrcbBuf, floatRepSize, repeatTimes, quantScaleCkvRope,
                   computeBlockAlign);
}
} // namespace MlaProlog
#endif
