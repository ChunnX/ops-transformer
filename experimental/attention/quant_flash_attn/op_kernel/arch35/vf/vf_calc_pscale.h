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
                                         __ubuf__ half *finalMax, __ubuf__ uint8_t *nzIdx)
{
    RegTensor<half> vreg_sub_loop_max;
    RegTensor<half> vreg_final_max;
    RegTensor<half> vreg_delta_k;       // Δk = k_i - K_final（整数，≤ 0）
    RegTensor<uint8_t> vreg_scale;      // Cast 产物（偶位 = 值，奇位 = 零）→ Gather 压缩
    RegTensor<uint8_t> vreg_idx;        // 解交织索引表（与 P 存储共用一张 pNzIdxUB）
    MaskReg preg_half = CreateMask<half, MaskPattern::ALL>();
    MaskReg preg_u8 = CreateMask<uint8_t, MaskPattern::ALL>();
    uint32_t pscale_valid_count = 128;  // 有效数据 128 字节（防 ALL 掩码写 256B 溢出到下一槽位）
    MaskReg preg_u8_valid = UpdateMask<uint8_t>(pscale_valid_count);

    // ① half 域整数运算：Δk = k_i - K_final，+127 → e8m0 字节值
    LoadAlign(vreg_sub_loop_max, subLoopMax);
    LoadAlign(vreg_final_max, finalMax);
    Sub(vreg_delta_k, vreg_sub_loop_max, vreg_final_max, preg_half);
    Adds(vreg_delta_k, vreg_delta_k, NUM_127, preg_half);
    Maxs(vreg_delta_k, vreg_delta_k, ZERO_VALUE, preg_half);

    // ② 窄化 half[128] → uint8[128]（2026-09-20 修正，设计 §7.3 定稿）：
    //    官方文档证实窄化 Cast 为交织放置——原同源双 Cast+Or 得重复对（后 64 值丢失，bug）。
    //    改单 Cast：128 个值落偶数位（奇位被清零）→ Gather 取偶位压缩为顺序字节。
    //    共用表（idx[k]=2k, k<128；idx[128+k]=2k+1）对两消费者都成立：
    //    P 解交织得 [行i | 行i+1]；pscale 得 [顺序 128 字节 | 零]（奇位源是零）
    Cast<uint8_t, half, castTraitZero>(vreg_scale, vreg_delta_k, preg_half);
    LoadAlign(vreg_idx, nzIdx);
    Gather(vreg_scale, vreg_scale, vreg_idx);

    // ③ 存 e8m0 buffer（uint8 位模式 = 2^Δk 的 e8m0 编码；部分掩码只写有效 128 字节）
    StoreAlign(pscale, vreg_scale, preg_u8_valid);
}

/*
 * @ingroup VfaVectorApi
 * @brief VfCalcPScale 的 __aicore__ wrapper：接收 LocalTensor + 槽位索引，内部转指针调 vf
 *
 * @param [in] pscaleUB        pscale 缓冲（uint8_t 视图，[taskSlot][subLoopIdx][128] 布局）
 * @param [in] usedMaxUB       subLoop k_i 缓冲（[subLoopIdx][128] 布局）
 * @param [in] softmaxMaxUB    accMax 缓冲（[stateSlot][128] 布局）
 * @param [in] nzIdxUB         解交织索引表（与 P 存储共用 pNzIdxUB，256B）
 * @param [in] taskSlot        任务槽（0..2）
 * @param [in] subLoopIdx      subLoop 序号（0..2）
 * @param [in] stateSlot       accMax 状态槽（0..2）
 */
__aicore__ inline void VfCalcPScale(const LocalTensor<uint8_t> &pscaleUB,
                                     const LocalTensor<half> &usedMaxUB,
                                     const LocalTensor<half> &softmaxMaxUB,
                                     const LocalTensor<uint8_t> &nzIdxUB, uint32_t taskSlot,
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
    __ubuf__ uint8_t *nzIdxPtr = reinterpret_cast<__ubuf__ uint8_t *>(nzIdxUB.GetPhyAddr());
    VfCalcPScaleVF(pscalePtr, subLoopMaxPtr, finalMaxPtr, nzIdxPtr);
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_CALC_PSCALE_H_
