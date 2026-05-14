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
 * \file moe_distribute_quant_kernel.h
 * \brief QuantKernel - 无状态量化计算组件
 *        QuantArgs - 运行时参数 POD struct
 */

#ifndef MOE_DISTRIBUTE_QUANT_KERNEL_H
#define MOE_DISTRIBUTE_QUANT_KERNEL_H

#include "moe_distribute_quant_layout.h"
#include "moe_distribute_v2_constant.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#if __has_include("../../common/op_kernel/mc2_kernel_utils.h")
#include "../../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../common/op_kernel/mc2_kernel_utils.h"
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "quantize_functions.h"
#endif

namespace Mc2Kernel {
namespace MoeDistributeQuant {

using namespace AscendC;

struct QuantArgs {
    uint32_t expertIndex;
    uint32_t scalesCount;
    uint32_t currentTokenIndex;
    GlobalTensor<float> scalesInGM;
    GlobalTensor<uint8_t> dynamicScalesOutGM;
};

template <typename XIn, typename XOut, typename ExpandXOut, int32_t QuantMode, bool IsSmoothScaleExist>
struct QuantKernel {
    __aicore__ static inline void Process(LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale,
                                          LocalTensor<XIn> const &inToken, LocalTensor<uint8_t> &workUb,
                                          QuantArgs const &args, uint32_t axisH);

    __aicore__ static inline void WriteScaleToGM(GlobalTensor<uint8_t> &scaleGM, LocalTensor<XOut> const &outToken,
                                                 uint32_t tokenIdx, QuantArgs const &args, uint32_t axisH,
                                                 uint32_t scaleOutBytes, DataCopyExtParams &scaleOutParams);

private:
    __aicore__ static inline void DoStatic(LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale,
                                           LocalTensor<XIn> const &inToken, LocalTensor<uint8_t> &workUb,
                                           QuantArgs const &args, uint32_t axisH);

    __aicore__ static inline void DoPerToken(LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale,
                                             LocalTensor<XIn> const &inToken, LocalTensor<uint8_t> &workUb,
                                             QuantArgs const &args, uint32_t axisH);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ static inline void DoMx(LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale,
                                       LocalTensor<XIn> const &inToken, LocalTensor<uint8_t> &workUb,
                                       QuantArgs const &args, uint32_t axisH);

    __aicore__ static inline void DoPerGroup(LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale,
                                             LocalTensor<XIn> const &inToken, LocalTensor<uint8_t> &workUb,
                                             QuantArgs const &args, uint32_t axisH);
#endif

    __aicore__ static inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
};

template <typename XIn, typename XOut, typename ExpandXOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline void QuantKernel<XIn, XOut, ExpandXOut, QuantMode, IsSmoothScaleExist>::Process(
    LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale, LocalTensor<XIn> const &inToken,
    LocalTensor<uint8_t> &workUb, QuantArgs const &args, uint32_t axisH)
{
    if constexpr (QuantMode == STATIC_QUANT) {
        DoStatic(outToken, outScale, inToken, workUb, args, axisH);
    } else if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
        DoPerToken(outToken, outScale, inToken, workUb, args, axisH);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    else if constexpr (QuantMode == MX_QUANT) {
        DoMx(outToken, outScale, inToken, workUb, args, axisH);
    } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
        DoPerGroup(outToken, outScale, inToken, workUb, args, axisH);
    }
#endif
}

template <typename XIn, typename XOut, typename ExpandXOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline void QuantKernel<XIn, XOut, ExpandXOut, QuantMode, IsSmoothScaleExist>::DoStatic(
    LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale, LocalTensor<XIn> const &inToken,
    LocalTensor<uint8_t> &workUb, QuantArgs const &args, uint32_t axisH)
{
    LocalTensor<float> floatLocalTemp = workUb.ReinterpretCast<float>();
    LocalTensor<float> smoothScalesTensor = workUb[axisH * sizeof(float)].ReinterpretCast<float>();

    Cast(floatLocalTemp, inToken, RoundMode::CAST_NONE, axisH);
    if constexpr (Std::IsSame<ExpandXOut, int8_t>::value) {
        DataCopyParams scalesInParams = {1U, static_cast<uint16_t>(axisH * sizeof(float)), 0U, 0U};
        DataCopyPadParams scalesPadParams = {false, 0, 0, 0};

        if (args.scalesCount == 1) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(args.scalesInGM);
            float scaleVal = args.scalesInGM(0);
            SyncFunc<AscendC::HardEvent::S_V>();
            Muls(floatLocalTemp, floatLocalTemp, scaleVal, axisH);
        } else if (args.scalesCount == axisH) {
            DataCopyPad(smoothScalesTensor, args.scalesInGM, scalesInParams, scalesPadParams);
        } else {
            DataCopyPad(smoothScalesTensor, args.scalesInGM[args.expertIndex * axisH], scalesInParams, scalesPadParams);
        }
        if (args.scalesCount != 1) {
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor, axisH);
        }

        LocalTensor<int32_t> tokenI32LT = floatLocalTemp.ReinterpretCast<int32_t>();
        Cast(tokenI32LT, floatLocalTemp, RoundMode::CAST_RINT, axisH);
        LocalTensor<half> tokenF16LT = floatLocalTemp.ReinterpretCast<half>();
        SetDeqScale((half)1.000000e+00f);
        Cast(tokenF16LT, tokenI32LT, RoundMode::CAST_ROUND, axisH);
        Cast(outToken, tokenF16LT, RoundMode::CAST_TRUNC, axisH);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    else if constexpr (Std::IsSame<ExpandXOut, hifloat8_t>::value) {
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(args.scalesInGM);
        float scaleVal = args.scalesInGM(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, scaleVal, axisH);
        Cast(outToken, floatLocalTemp, RoundMode::CAST_ROUND, axisH);
    }
#endif
}

template <typename XIn, typename XOut, typename ExpandXOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline void QuantKernel<XIn, XOut, ExpandXOut, QuantMode, IsSmoothScaleExist>::DoPerToken(
    LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale, LocalTensor<XIn> const &inToken,
    LocalTensor<uint8_t> &workUb, QuantArgs const &args, uint32_t axisH)
{
    LocalTensor<float> floatLocalTemp = workUb.ReinterpretCast<float>();
    LocalTensor<float> smoothScalesTensor = workUb[axisH * sizeof(float)].ReinterpretCast<float>();
    LocalTensor<float> reduceMaxBuf = smoothScalesTensor;

    float dynamicScale = 0.0f;
    float maxVal = INT8_MAX_VALUE;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr (Std::IsSame<ExpandXOut, fp8_e5m2_t>::value) {
        maxVal = FP8_E5M2_MAX_VALUE;
    } else if constexpr (Std::IsSame<ExpandXOut, fp8_e4m3fn_t>::value) {
        maxVal = FP8_E4M3_MAX_VALUE;
    }
#endif

    Cast(floatLocalTemp, inToken, RoundMode::CAST_NONE, axisH);
    PipeBarrier<PIPE_V>();

    if constexpr (IsSmoothScaleExist) {
        DataCopyParams scalesInParams = {1U, static_cast<uint16_t>(axisH * sizeof(float)), 0U, 0U};
        DataCopyPadParams scalesPadParams = {true, 0, 0, 0};
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(smoothScalesTensor, args.scalesInGM[args.expertIndex * axisH], scalesInParams, scalesPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor, axisH);
        PipeBarrier<PIPE_V>();
    }

    Abs(reduceMaxBuf, floatLocalTemp, axisH);
    PipeBarrier<PIPE_V>();
    ReduceMaxInplace(reduceMaxBuf, axisH);
    SyncFunc<AscendC::HardEvent::V_S>();
    dynamicScale = maxVal / reduceMaxBuf.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH);
    PipeBarrier<PIPE_V>();

    if constexpr (Std::IsSame<ExpandXOut, int8_t>::value) {
        LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
        LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
        Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH);
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH);
        PipeBarrier<PIPE_V>();
        Cast(outToken, halfLocalTemp, RoundMode::CAST_TRUNC, axisH);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    else if constexpr (Std::IsSame<ExpandXOut, fp8_e4m3fn_t>::value || Std::IsSame<ExpandXOut, fp8_e5m2_t>::value) {
        Cast(outToken, floatLocalTemp, RoundMode::CAST_RINT, axisH);
    }
#endif

    LocalTensor<float> tokenF32Tmp = outToken.template ReinterpretCast<float>();
    tokenF32Tmp.SetValue((Ceil(axisH, UB_ALIGN) * UB_ALIGN) / sizeof(float), float(1.0) / dynamicScale);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <typename XIn, typename XOut, typename ExpandXOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline void QuantKernel<XIn, XOut, ExpandXOut, QuantMode, IsSmoothScaleExist>::DoPerGroup(
    LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale, LocalTensor<XIn> const &inToken,
    LocalTensor<uint8_t> &workUb, QuantArgs const &args, uint32_t axisH)
{
    if constexpr (Std::IsSame<ExpandXOut, fp8_e4m3fn_t>::value || Std::IsSame<ExpandXOut, fp8_e5m2_t>::value) {
        LocalTensor<float> smoothScalesTensor = workUb.ReinterpretCast<float>();

        if constexpr (IsSmoothScaleExist) {
            DataCopyParams scalesInParams = {1U, static_cast<uint16_t>(Align128(axisH) * sizeof(float)), 0U, 0U};
            DataCopyPadParams scalesPadParams = {true, 0, 0, 0};

            uint32_t smoothTailOffset = Ceil(axisH * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
            uint32_t inputReadCount = Align128(axisH);
            uint32_t smoothTailCount = inputReadCount > smoothTailOffset ? (inputReadCount - smoothTailOffset) : 0U;

            DataCopyPad(smoothScalesTensor, args.scalesInGM[args.expertIndex * axisH], scalesInParams, scalesPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            if (smoothTailCount > 0U) {
                Duplicate<float>(smoothScalesTensor[smoothTailOffset], 0.0f, smoothTailCount);
                PipeBarrier<PIPE_V>();
            }
        }

        __ubuf__ XIn *srcAddr = (__ubuf__ XIn *)inToken.GetPhyAddr();
        __ubuf__ float *smoothLocalAddr = (__ubuf__ float *)smoothScalesTensor.GetPhyAddr();
        __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)outToken.GetPhyAddr();
        __ubuf__ float *scaleOutLocalAddr = (__ubuf__ float *)outToken[Align128<uint32_t>(axisH)].GetPhyAddr();

        quant::ComputePerTileDynamic<XIn, ExpandXOut, AscendC::RoundMode::CAST_RINT, IsSmoothScaleExist>(
            srcAddr, smoothLocalAddr, scaleOutLocalAddr, outLocalAddr, axisH);
    }
}

template <typename XIn, typename XOut, typename ExpandXOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline void QuantKernel<XIn, XOut, ExpandXOut, QuantMode, IsSmoothScaleExist>::DoMx(
    LocalTensor<XOut> &outToken, LocalTensor<uint8_t> &outScale, LocalTensor<XIn> const &inToken,
    LocalTensor<uint8_t> &workUb, QuantArgs const &args, uint32_t axisH)
{
    uint32_t mxScaleNum = Align2(Ceil32(axisH));
    LocalTensor<uint16_t> floatLocalTemp = workUb.ReinterpretCast<uint16_t>();

    __ubuf__ XIn *srcAddr = (__ubuf__ XIn *)inToken.GetPhyAddr();
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)floatLocalTemp.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)floatLocalTemp[Align32(mxScaleNum)].GetPhyAddr();
    __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)outToken.GetPhyAddr();
    __ubuf__ uint16_t *mxScaleLocalAddr = (__ubuf__ uint16_t *)outToken[Align256<uint32_t>(axisH)].GetPhyAddr();

    if constexpr (Std::IsSame<ExpandXOut, fp4x2_e2m1_t>::value || Std::IsSame<ExpandXOut, fp4x2_e1m2_t>::value) {
        mxScaleLocalAddr =
            (__ubuf__ uint16_t *)outToken[Align256<uint32_t>(Ceil(axisH, FP4_ELEMS_PER_BYTE))].GetPhyAddr();
    }

    quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH);
    quant::ComputeScale<ExpandXOut>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);

    if constexpr (Std::IsSame<ExpandXOut, fp8_e4m3fn_t>::value || Std::IsSame<ExpandXOut, fp8_e5m2_t>::value) {
        quant::ComputeFp8Data<XIn, ExpandXOut, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, axisH);
    } else {
        quant::ComputeFp4Data<XIn, ExpandXOut, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, axisH);
    }
}
#endif

template <typename XIn, typename XOut, typename ExpandXOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline void QuantKernel<XIn, XOut, ExpandXOut, QuantMode, IsSmoothScaleExist>::WriteScaleToGM(
    GlobalTensor<uint8_t> &scaleGM, LocalTensor<XOut> const &outToken, uint32_t tokenIdx, QuantArgs const &args,
    uint32_t axisH, uint32_t scaleOutBytes, DataCopyExtParams &scaleOutParams)
{
    if constexpr (((QuantMode > UNQUANT) && (QuantMode != STATIC_QUANT)) ||
                  ((QuantMode == UNQUANT) && IsSmoothScaleExist)) {
        auto scaleLT = outToken[(Ceil(axisH, UB_ALIGN) * UB_ALIGN)].template ReinterpretCast<uint8_t>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if constexpr ((QuantMode == MX_QUANT) &&
                      (Std::IsSame<ExpandXOut, fp4x2_e2m1_t>::value || Std::IsSame<ExpandXOut, fp4x2_e1m2_t>::value)) {
            scaleLT = outToken[Align256<uint32_t>(Ceil(axisH, FP4_ELEMS_PER_BYTE))].template ReinterpretCast<uint8_t>();
        } else if constexpr ((QuantMode == MX_QUANT) && (Std::IsSame<ExpandXOut, fp8_e5m2_t>::value ||
                                                         Std::IsSame<ExpandXOut, fp8_e4m3fn_t>::value)) {
            scaleLT = outToken[Align256<uint32_t>(axisH)].template ReinterpretCast<uint8_t>();
        } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
            scaleLT = outToken[Align128<uint32_t>(axisH)].template ReinterpretCast<uint8_t>();
        } else if constexpr ((QuantMode == UNQUANT) && (Std::IsSame<ExpandXOut, fp4x2_e2m1_t>::value ||
                                                        Std::IsSame<ExpandXOut, fp4x2_e1m2_t>::value)) {
            scaleLT = outToken[(Ceil(Ceil(axisH, FP4_ELEMS_PER_BYTE), UB_ALIGN) * UB_ALIGN)]
                          .template ReinterpretCast<uint8_t>();
        }
#endif
        DataCopyPad(scaleGM[tokenIdx * scaleOutBytes], scaleLT, scaleOutParams);
    }
}

template <typename XIn, typename XOut, typename ExpandXOut, int32_t QuantMode, bool IsSmoothScaleExist>
__aicore__ inline void
QuantKernel<XIn, XOut, ExpandXOut, QuantMode, IsSmoothScaleExist>::ReduceMaxInplace(const LocalTensor<float> &srcLocal,
                                                                                    uint32_t count)
{
    uint64_t repsFp32 = count >> 6;
    uint64_t offsetsFp32 = repsFp32 << 6;
    uint64_t remsFp32 = count & 0x3f;
    const uint64_t elemPerRefFp32 = 64UL;

    if (likely(repsFp32 > 1)) {
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
}

} // namespace MoeDistributeQuant
} // namespace Mc2Kernel

#endif // MOE_DISTRIBUTE_QUANT_KERNEL_H