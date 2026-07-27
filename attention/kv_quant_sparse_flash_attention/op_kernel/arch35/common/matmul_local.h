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
 * \file matmul_local.h
 * \brief QSFA matmul wrappers with tensor-based L0 interface (not DB-policy based)
 */
#ifndef QSFA_MATMUL_LOCAL_H
#define QSFA_MATMUL_LOCAL_H

#include "../../common/op_kernel/matmul.h"

using namespace AscendC;
using namespace fa_base_matmul;

namespace BaseApi {

// 切K, 不拆分(单次K循环)
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType>
__aicore__ inline void MatmulKFull(
    const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor, LocalTensor<L0AType> &aL0Tensor,
    LocalTensor<L0BType> &bL0Tensor, const LocalTensor<C> &cL0Tensor, const MMParam &param,
    uint32_t l0ABBufId, uint32_t &l0ABBufIdx)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t l1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t l1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || \
        IsSameType<A, fp8_e4m3fn_t>::value || IsSameType<A, hifloat8_t>::value) {
        l1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK;
        l1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK;
    }
    if constexpr (IsSameType<A, float32_t>::value) {
        l1Aoffset = param.isLeftTranspose ? baseK << 3 : ((param.singleM + 15) >> 4 << 4) * baseK;
        l1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 3;
    }
#endif
    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        LocalTensor<L0AType> curAL0Tensor = aL0Tensor[BUFFER_SIZE_8K * l0ABBufIdx];
        WaitFlag<HardEvent::M_MTE1>(l0ABBufId + l0ABBufIdx);
        LoadDataToL0A(curAL0Tensor, aL1Tensor, param, k * l1Aoffset, tileK, param.singleM);

        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        LocalTensor<L0BType> curBL0Tensor = bL0Tensor[BUFFER_SIZE_16K * l0ABBufIdx];
        LoadDataToL0B(curBL0Tensor, bL1Tensor, param, k * l1Boffset, tileK, param.singleN, loopNum);

        SetFlag<HardEvent::MTE1_M>(l0ABBufId + l0ABBufIdx);
        WaitFlag<HardEvent::MTE1_M>(l0ABBufId + l0ABBufIdx);
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
                                UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor, curAL0Tensor, curBL0Tensor, mmadParams);
        SetFlag<HardEvent::M_MTE1>(l0ABBufId + l0ABBufIdx);
        l0ABBufIdx ^= 1;
    }
}

// 切N
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType>
__aicore__ inline void MatmulN1(
    const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor, LocalTensor<L0AType> &aL0Tensor,
    LocalTensor<L0BType> &bL0Tensor, const LocalTensor<C> &cL0Tensor, const MMParam &param,
    uint32_t l0ABBufId, uint32_t &l0ABBufIdx)
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN;
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t l1BOffset = param.isRightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || \
        IsSameType<A, fp8_e4m3fn_t>::value || IsSameType<A, hifloat8_t>::value) {
        l1BOffset = ((param.singleK + 31) >> 5 << 5) * baseN;
    }
#endif
    uint64_t l0COffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        l0COffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.k = param.singleK;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;
    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;
        mmadParams.n = tileN;

        WaitFlag<HardEvent::M_MTE1>(l0ABBufId + l0ABBufIdx);
        LocalTensor<L0AType> curAL0Tensor = aL0Tensor[BUFFER_SIZE_8K * l0ABBufIdx];
        if (n == 0 || n == 1) {
            LoadDataToL0A(curAL0Tensor, aL1Tensor, param, 0, param.singleK, param.singleM);
        }

        uint64_t loopNum = param.isRightTranspose ? nLoops : 1;
        LocalTensor<L0BType> curBL0Tensor = bL0Tensor[BUFFER_SIZE_16K * l0ABBufIdx];
        LoadDataToL0B(curBL0Tensor, bL1Tensor, param, n * l1BOffset, param.singleK, tileN, loopNum);
        SetFlag<HardEvent::MTE1_M>(l0ABBufId + l0ABBufIdx);
        WaitFlag<HardEvent::MTE1_M>(l0ABBufId + l0ABBufIdx);

        Mmad(cL0Tensor[n * l0COffset], curAL0Tensor, curBL0Tensor, mmadParams);

        SetFlag<HardEvent::M_MTE1>(l0ABBufId + l0ABBufIdx);
        l0ABBufIdx ^= 1;
    }
}

}
#endif // QSFA_MATMUL_LOCAL_H
