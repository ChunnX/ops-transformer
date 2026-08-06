/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_FP8_STATIC_MATMUL_H_
#define QUANT_BLOCK_SPARSE_ATTN_FP8_STATIC_MATMUL_H_

#include "common/matmul.h"
#include "quant_block_sparse_attn_fp8_static.h"

namespace BaseApi {
namespace Fp8Static {

template <typename A, typename B, typename C, typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulFullStatic(
    const AscendC::LocalTensor<A> &aL1Tensor, const AscendC::LocalTensor<B> &bL1Tensor,
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L0A> *l0aBuffers,
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L0B> *l0bBuffers,
    uint32_t &l0aCursor, uint32_t &l0bCursor, const AscendC::LocalTensor<C> &cL0Tensor,
    fa_base_matmul::MMParam &param)
{
    const uint32_t l0aSlot = l0aCursor;
    l0aCursor ^= 1U;
    l0aBuffers[l0aSlot].template Wait<AscendC::HardEvent::M_MTE1>();
    AscendC::LocalTensor<L0ADType> l0aTensor = l0aBuffers[l0aSlot].template GetTensor<L0ADType>();
    fa_base_matmul::LoadDataToL0A(l0aTensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    l0aBuffers[l0aSlot].template Set<AscendC::HardEvent::MTE1_M>();

    const uint32_t l0bSlot = l0bCursor;
    l0bCursor ^= 1U;
    l0bBuffers[l0bSlot].template Wait<AscendC::HardEvent::M_MTE1>();
    AscendC::LocalTensor<L0BDType> l0bTensor = l0bBuffers[l0bSlot].template GetTensor<L0BDType>();
    fa_base_matmul::LoadDataToL0B(l0bTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    l0bBuffers[l0bSlot].template Set<AscendC::HardEvent::MTE1_M>();

    l0aBuffers[l0aSlot].template Wait<AscendC::HardEvent::MTE1_M>();
    l0bBuffers[l0bSlot].template Wait<AscendC::HardEvent::MTE1_M>();

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

    l0aBuffers[l0aSlot].template Set<AscendC::HardEvent::M_MTE1>();
    l0bBuffers[l0bSlot].template Set<AscendC::HardEvent::M_MTE1>();
}

} // namespace Fp8Static
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_FP8_STATIC_MATMUL_H_
