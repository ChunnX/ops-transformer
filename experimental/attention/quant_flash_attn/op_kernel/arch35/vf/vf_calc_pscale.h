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
 * \file vf_calc_pscale.h
 * \brief V1 末 subLoop 统一算 pscale：整数减法 + 字节加 127 → e8m0，[128]
 */

#ifndef VF_CALC_PSCALE_H_
#define VF_CALC_PSCALE_H_

#include "kernel_tensor.h"
#include "vf_common_def.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

/*
 * @ingroup VfaVectorApi
 * @brief 计算第 i 个 subLoop 的 pscale = e8m0(k_i - K_final + 127)
 *
 * 输入均为 half log2 整数域量化指数。输出 e8m0 字节值 = 指数差 + 127。
 *
 * @param [out] pscale      输出 [128] e8m0 字节（pscaleUB[taskSlot][i] 槽位，以 uint8_t 视图传入）
 * @param [in]  subLoopMax  本 subLoop 量化指数 k_i（subLoopUsedMaxUB[i]，half 整数）
 * @param [in]  finalMax    分块最终量化指数 K^{final}（softmaxMaxUB[stateSlot]，half 整数）
 *
 * 分层约定（MxFP4 同款）：vf 参数用 uint8_t*（原始字节），e8m0 语义由 __aicore__
 * 调用层负责（传入前 ReinterpretCast）。
 */
__simd_vf__ inline void VfCalcPScaleVF(__ubuf__ uint8_t *pscale, __ubuf__ half *subLoopMax,
                                         __ubuf__ half *finalMax)
{
    RegTensor<half> vregSubLoopMax;
    RegTensor<half> vregFinalMax;
    RegTensor<half> vregDeltaK;
    RegTensor<uint8_t> vregScaleA;
    RegTensor<uint8_t> vregScaleB;
    MaskReg pregHalf = CreateMask<half, MaskPattern::ALL>();
    MaskReg pregU8 = CreateMask<uint8_t, MaskPattern::ALL>();
    uint32_t pscaleValidCount = 128;  // 有效数据 128 字节（防 ALL 掩码写 256B 溢出到下一槽位）
    MaskReg pregU8Valid = UpdateMask<uint8_t>(pscaleValidCount);

    // ① half 域整数运算：Δk = k_i - K_final，+127 → e8m0 字节值
    LoadAlign(vregSubLoopMax, subLoopMax);
    LoadAlign(vregFinalMax, finalMax);
    Sub(vregDeltaK, vregSubLoopMax, vregFinalMax, pregHalf);
    Adds(vregDeltaK, vregDeltaK, NUM_127, pregHalf);
    Maxs(vregDeltaK, vregDeltaK, ZERO_VALUE, pregHalf);

    // ② 窄化 half[128] → uint8[128]：双 Cast 各取偶/奇位 + Or 合并（MxFP4 mxscale 同款）
    Cast<uint8_t, half, castTraitZero>(vregScaleA, vregDeltaK, pregHalf);
    Cast<uint8_t, half, castTraitOne>(vregScaleB, vregDeltaK, pregHalf);
    Or(vregScaleA, vregScaleA, vregScaleB, pregU8);

    // ③ 存 e8m0 buffer（uint8 位模式 = 2^Δk 的 e8m0 编码）
    //    用 128 元素部分掩码（非 ALL 256），只写有效 128 字节，防溢出到 pscaleUB 相邻槽位
    StoreAlign(pscale, vregScaleA, pregU8Valid);
}

/*
 * @ingroup VfaVectorApi
 * @brief VfCalcPScale 的 __aicore__ wrapper：接收 LocalTensor + 槽位索引，内部转指针调 vf
 *
 * @param [in] pscaleUB        pscale 缓冲（uint8_t 视图，[taskSlot][subLoopIdx][128] 布局）
 * @param [in] usedMaxUB       subLoop k_i 缓冲（[subLoopIdx][128] 布局）
 * @param [in] softmaxMaxUB    accMax 缓冲（[stateSlot][128] 布局）
 * @param [in] taskSlot        任务槽（0..2）
 * @param [in] subLoopIdx      subLoop 序号（0..2）
 * @param [in] stateSlot       accMax 状态槽（0..2）
 */
__aicore__ inline void VfCalcPScale(const LocalTensor<uint8_t> &pscaleUB,
                                     const LocalTensor<half> &usedMaxUB,
                                     const LocalTensor<half> &softmaxMaxUB, uint32_t taskSlot,
                                     uint32_t subLoopIdx, uint32_t stateSlot)
{
    constexpr uint32_t pscaleSlot = 128;        // 单 pscale 槽字节数
    constexpr uint32_t pscaleSubLoops = 3;      // 每 task 槽的 subLoop 数
    __ubuf__ uint8_t *pscalePtr = reinterpret_cast<__ubuf__ uint8_t *>(pscaleUB.GetPhyAddr())
                                    + (taskSlot * pscaleSubLoops + subLoopIdx) * pscaleSlot;
    __ubuf__ half *subLoopMaxPtr = reinterpret_cast<__ubuf__ half *>(usedMaxUB.GetPhyAddr())
                                     + subLoopIdx * pscaleSlot;
    __ubuf__ half *finalMaxPtr = reinterpret_cast<__ubuf__ half *>(softmaxMaxUB.GetPhyAddr())
                                   + stateSlot * pscaleSlot;
    VfCalcPScaleVF(pscalePtr, subLoopMaxPtr, finalMaxPtr);
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_CALC_PSCALE_H_
