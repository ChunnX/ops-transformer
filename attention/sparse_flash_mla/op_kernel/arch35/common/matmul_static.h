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
 * \file matmul_static.h
 * \brief 静态 tensor + bufferId + SetFlag/WaitFlag 的 MatmulK/MatmulN 实现。
 *        与 fa_base_matmul::MatmulK/MatmulN 逻辑一致，仅把 L0A/L0B 的 Buffer 管理换成 RingBuffer，
 *        并把 Buffer::Wait/Set 换成等价的 SetFlag/WaitFlag，核内同步语义保持不变。
 */
#ifndef MATMUL_STATIC_H
#define MATMUL_STATIC_H

#if __has_include("../../../common/op_kernel/matmul.h")
#include "../../../common/op_kernel/matmul.h"
#elif __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/op_kernel/matmul.h"
#endif
#include "static_buffer.h"

using namespace AscendC;
namespace fa_base_matmul {

// 切K，L0A/L0B 通过 RingBuffer 管理，MTE1<->M 同步使用 SetFlag/WaitFlag
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK,
          typename L0AElemType = A, typename L0BElemType = B>
__aicore__ inline void MatmulKStatic(
    const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor,
    RingBuffer<L0AElemType> &l0A, RingBuffer<L0BElemType> &l0B,
    const LocalTensor<C> &cL0Tensor, const MMParam &param,
    TEventID l0aMte1ToMId[2], TEventID l0bMte1ToMId[2], TEventID l0bMToMte1Id[2])
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t l1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t l1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        l1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK;
        l1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK;
    }
    if constexpr (IsSameType<A, float>::value) {
        l1Aoffset = param.isLeftTranspose ? baseK << 3 : ((param.singleM + 15) >> 4 << 4) * baseK;
        l1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 3;
    }
#endif

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;

        StaticBuffer<L0AElemType> &aBuf = l0A.GetNext();
        WaitFlag<HardEvent::M_MTE1>(l0aMte1ToMId[aBuf.idx]);
        LoadDataToL0A(aBuf.tensor, aL1Tensor, param, k * l1Aoffset, tileK, param.singleM);

        StaticBuffer<L0BElemType> &bBuf = l0B.GetNext();
        WaitFlag<HardEvent::M_MTE1>(l0bMte1ToMId[bBuf.idx]);
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        LoadDataToL0B(bBuf.tensor, bL1Tensor, param, k * l1Boffset, tileK, param.singleN, loopNum);
        SetFlag<HardEvent::MTE1_M>(l0bMToMte1Id[bBuf.idx]);
        WaitFlag<HardEvent::MTE1_M>(l0bMToMte1Id[bBuf.idx]);

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
        Mmad(cL0Tensor, aBuf.tensor, bBuf.tensor, mmadParams);

        SetFlag<HardEvent::M_MTE1>(l0aMte1ToMId[aBuf.idx]);
        SetFlag<HardEvent::M_MTE1>(l0bMte1ToMId[bBuf.idx]);
    }
}

// 切N，L0A/L0B 通过 RingBuffer 管理
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK,
          typename L0AElemType = A, typename L0BElemType = B>
__aicore__ inline void MatmulNStatic(
    const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor,
    RingBuffer<L0AElemType> &l0A, RingBuffer<L0BElemType> &l0B,
    const LocalTensor<C> &cL0Tensor, const MMParam &param,
    TEventID l0aMte1ToMId[2], TEventID l0bMte1ToMId[2], TEventID l0bMToMte1Id[2])
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN;
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t l1Boffset = param.isRightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        l1Boffset = ((param.singleK + 31) >> 5 << 5) * baseN;
    }
#endif
    uint64_t l0Coffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        l0Coffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }

    StaticBuffer<L0AElemType> &aBuf = l0A.GetNext();
    WaitFlag<HardEvent::M_MTE1>(l0aMte1ToMId[aBuf.idx]);
    LoadDataToL0A(aBuf.tensor, aL1Tensor, param, 0, param.singleK, param.singleM);

    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;

        StaticBuffer<L0BElemType> &bBuf = l0B.GetNext();
        WaitFlag<HardEvent::M_MTE1>(l0bMte1ToMId[bBuf.idx]);
        uint64_t loopNum = param.isRightTranspose ? nLoops : 1;
        LoadDataToL0B(bBuf.tensor, bL1Tensor, param, n * l1Boffset, param.singleK, tileN, loopNum);
        SetFlag<HardEvent::MTE1_M>(l0bMToMte1Id[bBuf.idx]);
        WaitFlag<HardEvent::MTE1_M>(l0bMToMte1Id[bBuf.idx]);

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = tileN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) {
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = param.unitFlag;
        Mmad(cL0Tensor[n * l0Coffset], aBuf.tensor, bBuf.tensor, mmadParams);

        SetFlag<HardEvent::M_MTE1>(l0bMte1ToMId[bBuf.idx]);
    }
    SetFlag<HardEvent::M_MTE1>(l0aMte1ToMId[aBuf.idx]);
}

} // namespace fa_base_matmul
#endif // MATMUL_STATIC_H
