/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, and you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You should have received a copy of the License along with this program.
 * If not, see <https://www.hiascend.com/legal/cannlicense>.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
 * PURPOSE.
 */

/*!
 * \file vf_calc_rescale_factor.h
 * \brief V1 跨 S2 分块 rescale 因子（原 VfExpSub 改名）：整数减法 + IEEE fp32 位构造（无 exp），[128]
 */

#ifndef VF_CALC_RESCALE_FACTOR_H_
#define VF_CALC_RESCALE_FACTOR_H_

#include "kernel_tensor.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

/*
 * @ingroup VfaVectorApi
 * @brief 计算跨 S2 分块 rescale 因子 softmaxExpUB[loop%3] = 2^(K^{blkStart} - K^{final})
 *
 * 输入均为 half log2 整数域量化指数（整数 k）。差值为整数，通过 IEEE fp32 位构造
 * 直接拼出 2^ΔK 的位模式（指数域 = ΔK + 127，尾数域全零），精确无舍入。
 * V2 侧（VfFlashUpdateAcc）按 [DV,S1] 元素级段加载使用。
 *
 * @param [out] softmaxExp   输出 [128] fp32（softmaxExpUB 槽位，IEEE 位模式）
 * @param [in]  blkStartMax  块开始量化指数 K^{blkStart}（blockStartMaxUB，half 整数）
 * @param [in]  finalMax     块最终量化指数 K^{final}（softmaxMaxUB[stateSlot]，half 整数）
 *
 * 指令链：Sub(half 整数减) → Adds 127 → Maxs 0 → Cast<int32> 双舍入 → ShiftLefts 23 → Store
 * 镜像：MxFP4 vf_computeScale_dn_mxfp4.h L60-76（位构造，量产验证）
 */
__simd_vf__ inline void VfCalcRescaleFactor(__ubuf__ float *softmaxExp, const __ubuf__ half *blkStartMax,
                                 const __ubuf__ half *finalMax)
{
    RegTensor<half> vregBlkStart;
    RegTensor<half> vregFinal;
    RegTensor<half> vregDeltaK;       // ΔK = K^{blkStart} - K^{final}（整数，≤ 0）
    RegTensor<int32_t> vregBitsA;     // fp32 位模式（舍入 A）
    RegTensor<int32_t> vregBitsB;     // fp32 位模式（舍入 B）
    MaskReg pregHalf = CreateMask<half, MaskPattern::ALL>();
    MaskReg pregI32 = CreateMask<int32_t, MaskPattern::ALL>();

    // [128] half = 1 个寄存器；[128] fp32 输出 = 2 × 64 元素寄存器
    for (uint32_t i = 0; i < 2; i++) {
        // ① 整数减法：ΔK = K^{blkStart} - K^{final}
        LoadAlign(vregBlkStart, blkStartMax + i * 64);
        LoadAlign(vregFinal, finalMax + i * 64);
        Sub(vregDeltaK, vregBlkStart, vregFinal, pregHalf);

        // ② +127（fp32 指数偏置）并防御负值
        Adds(vregDeltaK, vregDeltaK, 127.0h, pregHalf);
        Maxs(vregDeltaK, vregDeltaK, 0.0h, pregHalf);

        // ③ Cast int32 双舍入（MxFP4 同款防整数边界歧义）
        Cast<int32_t, half, castTraitZero>(vregBitsA, vregDeltaK, pregHalf);
        Cast<int32_t, half, castTraitOne>(vregBitsB, vregDeltaK, pregHalf);

        // ④ ShiftLefts 23：整数值移入 fp32 指数域位置
        //    (ΔK + 127) << 23 = fp32 位模式，尾数全零，指数域 = ΔK + 127
        ShiftLefts(vregBitsA, vregBitsA, 23, pregI32);
        ShiftLefts(vregBitsB, vregBitsB, 23, pregI32);

        // ⑤ Interleave 双舍入合并（MxFP4 同款），按 fp32 位模式存
        Interleave(vregBitsA, vregBitsB, vregBitsA, vregBitsB);
        StoreAlign(softmaxExp + i * 64, vregBitsA, pregI32);
    }
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_CALC_RESCALE_FACTOR_H_
