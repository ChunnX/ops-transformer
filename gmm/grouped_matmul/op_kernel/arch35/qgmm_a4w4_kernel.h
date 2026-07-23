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
 * \file qgmm_a4w4_kernel.h
 * \brief GroupMatmulA4W4 的桥接文件（transformer 仓 → tensor 仓 Blaze 模板）。
 */

#ifndef QGMM_A4W4_KERNEL_H
#define QGMM_A4W4_KERNEL_H

#include "blaze/gemm/kernel/kernel_qgmm_a4w4_mix.h"
#include "grouped_matmul_tiling_data_apt.h"

template <class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType,
          class xLayout, class wLayout, class yLayout, class l0cType>
__aicore__ inline void GmmA4W4Kernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                      GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                      GM_ADDR workspace,
                                      const GroupedMatmulTilingData::GMMBaseParamsA4W4* gmmBaseParams,
                                      const TCubeTiling* mmTilingData,
                                      AscendC::TPipe* que)
{
    // TODO(A5): 实现 A4W4 Blaze kernel 的完整功能
    //   使用 Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, A4W4BlockMmad, BlockEpilogue, BlockScheduler>
    //   配合 Blaze::Gemm::GroupedMatmulA4W4<> 作为调度策略。
    //   参考 gqmm_tensor_api_mx_kernel.h 中 GmmTensorApiMxKernel 的实现模式。
    (void)x; (void)weight; (void)bias; (void)scale;
    (void)groupList; (void)perTokenScale; (void)y; (void)workspace;
    (void)gmmBaseParams; (void)mmTilingData; (void)que;
}

#endif  // QGMM_A4W4_KERNEL_H
