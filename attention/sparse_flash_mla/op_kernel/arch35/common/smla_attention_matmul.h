/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Operator-private overload. Other callers retain the original shared helper.
#ifndef SMLA_ATTENTION_MATMUL_H
#define SMLA_ATTENTION_MATMUL_H
#include "static_matmul.h"
#include "smla_attention_ablation.h"
namespace fa_base_matmul {
// 切K
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          bool USE_RESIDENT_FIRST_A = false, typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulKStatic(const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor,
                                     RingBuffer<L0AType> &l0A, RingBuffer<L0BType> &l0B,
                                     const LocalTensor<C> &cL0Tensor, const MMParam &param,
                                     const LocalTensor<AScaleType> &aScaleL1Tensor,
                                     const LocalTensor<BScaleType> &bScaleL1Tensor,
                                     const LocalTensor<L0AType> &residentFirstA,
                                     bool loadResidentFirstA,
                                     const SMLAKernel::AttentionAblation &ablation,
                                     int64_t task)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;

        StaticBuffer<L0AType> &aBuf = l0A.GetNext();
        StaticBuffer<L0BType> &bBuf = l0B.GetNext();

        WaitFlag<HardEvent::M_MTE1>(MATMUL_STATIC_L0AB_ID(aBuf.idx)); // 等 M 用完该 AB 槽
        LocalTensor<L0ADType> L0ATensor = aBuf.tensor.template ReinterpretCast<L0ADType>();
        if constexpr (USE_RESIDENT_FIRST_A) {
            // The owner protects this independent slot until the last reuse of k=0.
            if (k == 0) {
                L0ATensor = residentFirstA.template ReinterpretCast<L0ADType>();
            }
            if ((k != 0 || loadResidentFirstA) && ablation.Transfer(task, SMLAKernel::AttentionAblation::RQ)) {
                LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
            }
        } else if (ablation.Transfer(task, SMLAKernel::AttentionAblation::RQ)) {
            LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
        }

        LocalTensor<L0BDType> L0BTensor = bBuf.tensor.template ReinterpretCast<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        if (ablation.Transfer(task, SMLAKernel::AttentionAblation::RK)) {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        }

        SetFlag<HardEvent::MTE1_M>(MATMUL_STATIC_L0AB_ID(aBuf.idx));  // MTE1 搬运完，通知 M
        WaitFlag<HardEvent::MTE1_M>(MATMUL_STATIC_L0AB_ID(aBuf.idx)); // M 等数据就绪

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) {
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                      UNITFLAG_EN_OUTER_LAST :
                                      UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        SetFlag<HardEvent::M_MTE1>(MATMUL_STATIC_L0AB_ID(aBuf.idx)); // M 用完，释放该 AB 槽
    }
}

} // namespace fa_base_matmul
#endif
