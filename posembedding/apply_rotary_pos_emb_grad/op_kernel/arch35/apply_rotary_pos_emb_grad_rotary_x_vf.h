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
 * \file apply_rotary_pos_emb_grad_rotary_x_vf.h
 * \brief apply_rotary_pos_emb_grad rotate_half VF 计算函数
 * 直接复用原有 rotary_position_embedding_grad_rotary_x_vf.h 的 HalfRotaryVF，命名空间改为 ApplyRotaryPosEmbGrad
 */

#ifndef __APPLY_ROTARY_POS_EMB_GRAD_ROTARY_X_VF__
#define __APPLY_ROTARY_POS_EMB_GRAD_ROTARY_X_VF__

#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

using namespace AscendC;

namespace ApplyRotaryPosEmbGrad {

constexpr uint32_t BLOCK_TYPE_BYTES = 32;

/*
 * HalfRotaryVF — half 模式 rotate_half
 * x = [-x[1], x[0]]
 * 直接复用原有实现，命名空间改为 ApplyRotaryPosEmbGrad
 */
template <typename T>
__aicore__ inline void HalfRotaryVF(
    const LocalTensor<T>& inTensor, const LocalTensor<T>& rotaryTensor, const uint32_t dLen, const uint32_t dAlign,
    const uint16_t currDNum)
{
    __local_mem__ T* inUb = (__local_mem__ T*)inTensor.GetPhyAddr();
    __local_mem__ T* outUb = (__local_mem__ T*)rotaryTensor.GetPhyAddr();
    __local_mem__ T* currInUb;
    __local_mem__ T* currOutUb;
    uint32_t vecLen = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t halfD = dLen / 2;
    uint32_t halfDAlign = Ops::Base::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_BYTES / sizeof(T)));
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, vecLen);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregIn;
        MicroAPI::RegTensor<T> vregHalfIn;
        MicroAPI::RegTensor<T> vregOut;
        MicroAPI::RegTensor<T> vregHalfOut;
        MicroAPI::RegTensor<T> vregNeg;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg;
        Duplicate(vregNeg, static_cast<T>(-1.0), pregAll);
        for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
            currInUb = inUb + idxD * dAlign;
            currOutUb = outUb + idxD * dAlign;
            uint32_t updateCnt = halfD;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<T>(updateCnt);
                int32_t offset = i * vecLen;
                int32_t halfOffset = offset + halfDAlign;
                MicroAPI::DataCopy(vregIn, currInUb + offset);
                MicroAPI::DataCopy(vregHalfIn, currInUb + halfOffset);
                MicroAPI::Mul(vregHalfIn, vregHalfIn, vregNeg, preg);
                MicroAPI::DataCopy(currOutUb + offset, vregHalfIn, preg);
                MicroAPI::DataCopy(currOutUb + halfOffset, vregIn, preg);
            }
        }
    }
}

} // namespace ApplyRotaryPosEmbGrad

#endif // __APPLY_ROTARY_POS_EMB_GRAD_ROTARY_X_VF__
