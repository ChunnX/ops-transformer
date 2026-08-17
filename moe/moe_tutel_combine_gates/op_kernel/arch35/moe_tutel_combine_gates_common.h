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
 * \file moe_tutel_combine_gates_common.h
 * \brief arch35 公共定义：常量、VF 点积、分核区间求解
 */
#ifndef MOE_TUTEL_COMBINE_GATES_COMMON_H_
#define MOE_TUTEL_COMBINE_GATES_COMMON_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/load_store_utils.h"
#include "moe_tutel_combine_gates_tiling_data.h"

namespace MoeTutelCombineGates {
using namespace AscendC;
using optiling::MoeTutelCombineGatesTilingData;
using namespace AscendC::MicroAPI;

constexpr uint32_t BLOCK_BYTE_SIZE = Ops::Base::GetUbBlockSize();
constexpr uint32_t VL_FP32 = static_cast<uint32_t>(Ops::Base::GetVRegSize()) / sizeof(float);
constexpr int32_t DOUBLE_BUFFER = 2;

// 本核在某一维上负责的区间 [start, start + len)。
// 前 remain 个分片各多 1 个元素，保证 sum(len) == 总长度且各分片最多相差 1。
struct CoreRange {
    int64_t start;
    int64_t len;
};

__aicore__ inline CoreRange SolveCoreRange(int64_t coreIdx, int64_t perCoreBase, int64_t remain)
{
    CoreRange range;
    if (coreIdx < remain) {
        range.len = perCoreBase + 1;
        range.start = coreIdx * range.len;
    } else {
        range.len = perCoreBase;
        range.start = coreIdx * perCoreBase + remain;
    }
    return range;
}

/*
 * 沿 H 的点积：sum_{i<count} xUb[i] * yUb[i]，结果以 fp32 写入 dstUb 的第 dstOffset 个元素。
 *
 * 归约方式：逐 VL 块 Mul 后按**固定顺序**累加到一个 lane-wise fp32 累加寄存器，
 * 最后做一次 VL 通道内 ReduceSum。归约顺序只依赖 count，不依赖核数或调度，故逐位可复现（INV-6）。
 * 尾块由 UpdateMask 生成的部分掩码承接，Mul 在非活跃通道上按 ZEROING 语义置 0，
 * 因此 UB 行尾（hFactorAlign 对齐补位）的残留数据不会污染结果（INV-4 的补充保证）。
 *
 * 调用方须保证 count > 0，且 xUb / yUb 的可读长度不少于 ceil(count / VL_FP32) * VL_FP32 个元素。
 */
template <typename T>
__aicore__ inline void DotProductVF(__local_mem__ float *dstUb, uint32_t dstOffset, __local_mem__ T *xUb,
                                    __local_mem__ T *yUb, uint32_t count)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> yReg;
        RegTensor<float> mulReg;
        RegTensor<float> accReg;
        RegTensor<float> sumReg;
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        uint32_t remain = count;
        // 首块单独展开，省掉累加器的清零指令，也避免在 VF 域内做运行期分支。
        pregLoop = UpdateMask<float>(remain);
        ops::LoadOneTensorForDtypeT<T>(xUb, xReg, pregLoop, 0);
        ops::LoadOneTensorForDtypeT<T>(yUb, yReg, pregLoop, 0);
        Mul(accReg, xReg, yReg, pregLoop);

        for (uint16_t i = 1; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(remain);
            ops::LoadOneTensorForDtypeT<T>(xUb, xReg, pregLoop, i * VL_FP32);
            ops::LoadOneTensorForDtypeT<T>(yUb, yReg, pregLoop, i * VL_FP32);
            Mul(mulReg, xReg, yReg, pregLoop);
            Add(accReg, accReg, mulReg, pregAll);
        }

        ReduceSum(sumReg, accReg, pregAll);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstUb + dstOffset, sumReg, pregOne);
    }
}

/*
 * 把 srcUb[0, count) 这 count 个 fp32 求和，结果写入 dstUb 的第 dstOffset 个元素。
 * CUT_H 路径用它把各 H 分块的部分和合并；归约顺序只依赖 count，故逐位可复现（INV-6）。
 *
 * 前置条件（调用方保证）：count > 0，且 srcUb 的 [count, ceil(count/VL)*VL) 段已被清零。
 * 因此这里可以整块无掩码累加，不依赖任何"非活跃通道置 0"的语义。
 */
__aicore__ inline void SumVF(__local_mem__ float *dstUb, uint32_t dstOffset, __local_mem__ float *srcUb,
                             uint32_t count)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg;
        RegTensor<float> accReg;
        RegTensor<float> sumReg;
        MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

        // 上一轮 DotProductVF 对 srcUb 的写入与本轮读取之间需要一次 UB 读写屏障。
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        ops::LoadOneTensorForDtypeT<float>(srcUb, accReg, pregAll, 0);
        for (uint16_t i = 1; i < loopCount; i++) {
            ops::LoadOneTensorForDtypeT<float>(srcUb, srcReg, pregAll, i * VL_FP32);
            Add(accReg, accReg, srcReg, pregAll);
        }

        ReduceSum(sumReg, accReg, pregAll);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstUb + dstOffset, sumReg, pregOne);
    }
}

/*
 * 把 srcUb[0, count) 的 fp32 结果 cast 成输出 dtype 写入 dstUb（fp32 时为直拷）。
 * 这是输出方向唯一一次 cast（fp16/bf16 走 CAST_RINT），对应 INV-5。
 * 调用方须保证 count > 0。
 */
template <typename T>
__aicore__ inline void CastRowVF(__local_mem__ T *dstUb, __local_mem__ float *srcUb, uint32_t count)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg;
        MaskReg pregLoop;

        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        uint32_t remain = count;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(remain);
            ops::LoadOneTensorForDtypeT<float>(srcUb, srcReg, pregLoop, i * VL_FP32);
            ops::StoreOneTensorForDtypeT<T>(dstUb, srcReg, pregLoop, i * VL_FP32);
        }
    }
}
} // namespace MoeTutelCombineGates

#endif // MOE_TUTEL_COMBINE_GATES_COMMON_H_
