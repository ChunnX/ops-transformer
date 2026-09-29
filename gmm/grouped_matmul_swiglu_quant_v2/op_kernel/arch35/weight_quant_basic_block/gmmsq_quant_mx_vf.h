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
 * \file gmmsq_quant_mx_vf.h
 * \brief SIMD-VF functions for GMM SwiGLU + MX Quantization.
 */

#ifndef GMMSQ_QUANT_MX_VF_H
#define GMMSQ_QUANT_MX_VF_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "basic_block_vf_mx.h"

namespace GMMSQWeightQuant {

namespace Reg = AscendC::Reg;
#include "gmmsq_quant_mx_vf_common.inc"

template <typename xType, typename wType>
__simd_vf__ inline void GmmsqAntiQuantMxA8W4NzNkVf(MxA8W4NzParams<xType, wType> mxA8W4NzParams)
{
    Reg::RegTensor<int8_t> wShrReg, wShlReg, wAndReg, wLoad, wShl, wShr0, wShr1, wSel, wAnd;
    Reg::MaskReg preg = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    Reg::MaskReg pregVsel = Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();

    Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    for (uint16_t loopKIdx = 0; loopKIdx < mxA8W4NzParams.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < mxA8W4NzParams.innerLoopNum; ++innerLoopIdx) {
            Reg::AddrReg aregWeightB8In = Reg::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * mxA8W4NzParams.nRealSizeAlign) >> 1, innerLoopIdx, VECTOR_REG_WIDTH >> 1);
            Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_US_B8>((Reg::RegTensor<uint8_t> &)wLoad,
                                                               (__ubuf__ uint8_t *&)mxA8W4NzParams.weightLowBitPhyAddr,
                                                               aregWeightB8In);

            Reg::ShiftRight(wShr0, wLoad, wShrReg, preg);
            Reg::ShiftLeft(wShl, wLoad, wShlReg, preg);
            Reg::ShiftRight(wShr1, wShl, wShrReg, preg);
            Reg::Select(wSel, wShr1, wShr0, pregVsel);
            Reg::And(wAnd, wSel, wAndReg, preg);

            Reg::AddrReg aregWeightB8Out = Reg::CreateAddrReg<uint8_t>(loopKIdx, mxA8W4NzParams.loopKDstStride,
                                                                       innerLoopIdx, mxA8W4NzParams.innerDstStride);
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightHighBitPhyAddr, (Reg::RegTensor<uint8_t> &)wAnd,
                aregWeightB8Out, preg);
        }
    }
}

} // namespace GMMSQWeightQuant

#endif // GMMSQ_QUANT_MX_VF_H
