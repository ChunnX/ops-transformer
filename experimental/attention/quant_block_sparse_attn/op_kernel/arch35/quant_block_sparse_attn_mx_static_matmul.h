/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_MATMUL_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_MATMUL_H_

#include "common/matmul.h"
#include "quant_block_sparse_attn_mx_static.h"

namespace BaseApi {
namespace MxStatic {

// Static-Tensor counterpart of fa_base_matmul::MxMatmulFull for the fixed MX QBSA path.
// Keep LoadData/Mmad parameter construction in lockstep with common/matmul.h::MatmulFull.
template <typename A, typename B, typename C, typename AScaleType = fp8_e8m0_t,
          typename BScaleType = fp8_e8m0_t, typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MxMatmulFullStatic(
    const AscendC::LocalTensor<A> &aL1Tensor, const AscendC::LocalTensor<B> &bL1Tensor,
    AscendC::LocalTensor<uint8_t> *l0aBuffers, AscendC::LocalTensor<uint8_t> *l0bBuffers,
    uint32_t &l0aCursor, uint32_t &l0bCursor, const AscendC::LocalTensor<C> &cL0Tensor,
    fa_base_matmul::MMParam &param,
    const AscendC::LocalTensor<AScaleType> &aScaleL1Tensor = AscendC::LocalTensor<AScaleType>(),
    const AscendC::LocalTensor<BScaleType> &bScaleL1Tensor = AscendC::LocalTensor<BScaleType>())
{
    const uint32_t l0aSlot = l0aCursor;
    l0aCursor ^= 1U;
    const AscendC::TEventID l0aEventId =
        l0aSlot == 0U ? MxStatic::L0A_EVENT_ID0 : MxStatic::L0A_EVENT_ID1;
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0aEventId);
    AscendC::LocalTensor<L0ADType> l0aTensor =
        l0aBuffers[l0aSlot].template ReinterpretCast<L0ADType>();
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
    if constexpr (AscendC::IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        fa_base_matmul::LoadDataToL0AMx<A, L0ADType>(l0aTensor, aL1Tensor, aScaleL1Tensor, param, 0,
                                                     param.singleK, param.singleM);
    } else
#endif
    {
        fa_base_matmul::LoadDataToL0A(l0aTensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0aEventId);

    const uint32_t l0bSlot = l0bCursor;
    l0bCursor ^= 1U;
    const AscendC::TEventID l0bEventId =
        l0bSlot == 0U ? MxStatic::L0B_EVENT_ID0 : MxStatic::L0B_EVENT_ID1;
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0bEventId);
    AscendC::LocalTensor<L0BDType> l0bTensor =
        l0bBuffers[l0bSlot].template ReinterpretCast<L0BDType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (AscendC::IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
        fa_base_matmul::LoadDataToL0BMx<B, L0BDType>(l0bTensor, bL1Tensor, bScaleL1Tensor, param, 0,
                                                     param.singleK, param.singleN);
    } else
#endif
    {
        fa_base_matmul::LoadDataToL0B(l0bTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0bEventId);

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0aEventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0bEventId);

    AscendC::MmadParams mmParams;
    mmParams.m = param.realM != 0U ? param.realM : param.singleM;
    mmParams.n = param.singleN;
    mmParams.k = param.singleK;
    mmParams.cmatrixInitVal = param.isOutKFisrt;
    mmParams.cmatrixSource = false;
    mmParams.unitFlag = param.unitFlag;
    if (mmParams.m == 1U) {
        mmParams.m = 16U;
    }
    AscendC::Mmad(cL0Tensor, l0aTensor, l0bTensor, mmParams);

    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0aEventId);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0bEventId);
}

} // namespace MxStatic
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_MATMUL_H_
