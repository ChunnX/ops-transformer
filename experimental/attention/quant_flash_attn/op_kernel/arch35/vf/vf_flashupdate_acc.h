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
 * \file vf_flashupdate_acc.h
 * \brief V2 跨 S2 分块累积 rescale：dstOut = dstOut×factor + cur，[DV,S1] 段加载
 */

#ifndef VF_FLASHUPDATE_ACC_H_
#define VF_FLASHUPDATE_ACC_H_

#include "kernel_tensor.h"
#include "vf_common_def.h"

namespace QFA_KERNEL {
namespace QfaVectorApi {

using namespace AscendC;
using namespace AscendC::Reg;

/*
 * @ingroup VfaVectorApi
 * @brief V2 跨 S2 分块累积：分子/分母 rescale 后累加新分块结果
 *
 * dstSum = dstSum × factor + rowsum（分母 [128] per S1）
 * dstOut[i][:] = dstOut[i][:] × factor[:] + cur[i][:]（分子 [128,128] [DV,S1]）
 *
 * factor 沿 S1 列方向逐元素（[DV,S1] 行主序红利：factor 向量与数据行形状对齐，无需广播）。
 * factor 分 2 个寄存器（列前半/后半），载入一次全程复用。
 *
 * @param [in/out] dstOut   stage2OutUB [128,128] fp32（[DV,S1]）
 * @param [in/out] dstSum   accRowsumUB [128] fp32（per S1）
 * @param [in]     cur      mm2ResUB[0:128] fp32（当前分块 PV）
 * @param [in]     rowsum   mm2ResUB[128] fp32（当前分块 rowsum）
 * @param [in]     factor   softmaxExpUB[loop%3] fp32（VfCalcRescaleFactor 输出）
 */
__simd_vf__ inline void VfFlashUpdateAcc(__ubuf__ float *dstOut, __ubuf__ float *dstSum,
                                         __ubuf__ float *cur, __ubuf__ float *rowsum,
                                         __ubuf__ float *factor)
{
    RegTensor<float> vregFactorA;    // factor[0..63]，列前半，全程复用
    RegTensor<float> vregFactorB;    // factor[64..127]，列后半，全程复用
    RegTensor<float> vregSumAcc;     // dstSum 累积值（Mul+Add 中间结果）
    RegTensor<float> vregSumNew;     // rowsum 新值
    RegTensor<float> vregOut0, vregOut1, vregOut2, vregOut3;  // 4 路行展开
    RegTensor<float> vregCur0, vregCur1, vregCur2, vregCur3;
    MaskReg pregF32 = CreateMask<float, MaskPattern::ALL>();   // 64 lanes fp32

    // ① factor 载入（2 regs，全程复用）
    LoadAlign(vregFactorA, factor);
    LoadAlign(vregFactorB, factor + 64);

    // ② dstSum 更新：dstSum = dstSum × factor + rowsum（镜像 DN processUpdateVF L38-44）
    //    列前半 [0..63]
    LoadAlign(vregSumAcc, dstSum);
    Mul(vregSumAcc, vregSumAcc, vregFactorA, pregF32);
    LoadAlign(vregSumNew, rowsum);
    Add(vregSumAcc, vregSumAcc, vregSumNew, pregF32);
    StoreAlign(dstSum, vregSumAcc, pregF32);

    //    列后半 [64..127]
    LoadAlign(vregSumAcc, dstSum + 64);
    Mul(vregSumAcc, vregSumAcc, vregFactorB, pregF32);
    LoadAlign(vregSumNew, rowsum + 64);
    Add(vregSumAcc, vregSumAcc, vregSumNew, pregF32);
    StoreAlign(dstSum + 64, vregSumAcc, pregF32);

    // ③ dstOut 更新：128 行 × 2 列半，4 路行展开（32 迭代 × 2 列半）
    //    [DV,S1] 行主序，行 stride = 128 fp32；列前半 offset+0 用 factorA，后半 +64 用 factorB
    for (uint32_t row = 0; row < 128; row += 4) {
        uint32_t base0 = row * 128;          // 行 row 的起始偏移
        uint32_t base1 = (row + 1) * 128;
        uint32_t base2 = (row + 2) * 128;
        uint32_t base3 = (row + 3) * 128;

        // ===== 列前半 [0..63]（factorA）=====
        LoadAlign(vregOut0, dstOut + base0);
        LoadAlign(vregOut1, dstOut + base1);
        LoadAlign(vregOut2, dstOut + base2);
        LoadAlign(vregOut3, dstOut + base3);
        Mul(vregOut0, vregOut0, vregFactorA, pregF32);
        Mul(vregOut1, vregOut1, vregFactorA, pregF32);
        Mul(vregOut2, vregOut2, vregFactorA, pregF32);
        Mul(vregOut3, vregOut3, vregFactorA, pregF32);
        LoadAlign(vregCur0, cur + base0);
        LoadAlign(vregCur1, cur + base1);
        LoadAlign(vregCur2, cur + base2);
        LoadAlign(vregCur3, cur + base3);
        Add(vregOut0, vregOut0, vregCur0, pregF32);
        Add(vregOut1, vregOut1, vregCur1, pregF32);
        Add(vregOut2, vregOut2, vregCur2, pregF32);
        Add(vregOut3, vregOut3, vregCur3, pregF32);
        StoreAlign(dstOut + base0, vregOut0, pregF32);
        StoreAlign(dstOut + base1, vregOut1, pregF32);
        StoreAlign(dstOut + base2, vregOut2, pregF32);
        StoreAlign(dstOut + base3, vregOut3, pregF32);

        // ===== 列后半 [64..127]（factorB）=====
        LoadAlign(vregOut0, dstOut + base0 + 64);
        LoadAlign(vregOut1, dstOut + base1 + 64);
        LoadAlign(vregOut2, dstOut + base2 + 64);
        LoadAlign(vregOut3, dstOut + base3 + 64);
        Mul(vregOut0, vregOut0, vregFactorB, pregF32);
        Mul(vregOut1, vregOut1, vregFactorB, pregF32);
        Mul(vregOut2, vregOut2, vregFactorB, pregF32);
        Mul(vregOut3, vregOut3, vregFactorB, pregF32);
        LoadAlign(vregCur0, cur + base0 + 64);
        LoadAlign(vregCur1, cur + base1 + 64);
        LoadAlign(vregCur2, cur + base2 + 64);
        LoadAlign(vregCur3, cur + base3 + 64);
        Add(vregOut0, vregOut0, vregCur0, pregF32);
        Add(vregOut1, vregOut1, vregCur1, pregF32);
        Add(vregOut2, vregOut2, vregCur2, pregF32);
        Add(vregOut3, vregOut3, vregCur3, pregF32);
        StoreAlign(dstOut + base0 + 64, vregOut0, pregF32);
        StoreAlign(dstOut + base1 + 64, vregOut1, pregF32);
        StoreAlign(dstOut + base2 + 64, vregOut2, pregF32);
        StoreAlign(dstOut + base3 + 64, vregOut3, pregF32);
    }
}

} // namespace QfaVectorApi
} // namespace QFA_KERNEL

#endif // VF_FLASHUPDATE_ACC_H_
