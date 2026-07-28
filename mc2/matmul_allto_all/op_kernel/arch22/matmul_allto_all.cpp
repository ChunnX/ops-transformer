/**
鑱? Copyright (c) 2025 Huawei Technologies Co., Ltd.
鑱? This program is free software, you can redistribute it and/or modify it under the terms and conditions of
鑱? CANN Open Software License Agreement Version 2.0 (the "License").
鑱? Please refer to the License for details. You may not use this file except in compliance with the License.
鑱? THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
鑱? INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
鑱? See LICENSE in the root of the software repository for the full text of the License.
鑱?/

/*!
 * \file matmul_allto_all.cpp
 * \brief kernel閸愬懏鐗崇€圭偟骞?
 */
#include <lib/matmul_intf.h>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "matmul_allto_all_tiling_key.h"
#include "matmul_allto_all_tiling.h"
#include "matmul_allto_all.h"
#include "../matmul_allto_all_pipeline.h" // A3鐠嬪啰鏁ernel濡剝婢樻惔?#include "../matmul_allto_all_kernel_base.h"   // A3闂堢偤鍣洪崠鏍︾瑢A5閸忚京鏁ゅù浣规寜

using namespace AscendC;
using namespace matmul_allto_all_910b_tiling_key;
using namespace Mc2Kernel;
using MC2KernelTemplate::MC2AlltoAllContext;
using MC2KernelTemplate::MC2AlltoAllPrimitives;

#ifndef MATMUL_ALLTO_ALL_A3_FP_IMPL
#define MATMUL_ALLTO_ALL_A3_FP_IMPL(tilingData, pipe)                                                                  \
    do {                                                                                                               \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_A3_FP(ComputationContextType);                                \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_A3_FP(false, MM_ALLTO_ALL_TRANS_X2, ComputationType);                 \
        ComputationType matmulImplName(&pipe);                                                                         \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType);                                             \
        TransposeType transposeImplName(&pipe);                                                                        \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_AICPU, MC2AlltoAllContext,           \
                                          MatmulAlltoAllTilingDataA3, MC2AlltoAllPrimitives, 1, 0, CommunicationType); \
        CommunicationType commImplName(&tilingData);                                                                   \
        using SchedulerContextType = MatmulAlltoAllPipelineContext<ComputationContextType>;                            \
        using SchedulerType =                                                                                          \
            MatmulAlltoAllPipeLine<ComputationType, TransposeType, CommunicationType, SchedulerContextType>;           \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);                               \
        Mc2Kernel::MatmulAlltoAllKernelBase<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataA3> op(       \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, workspaceGM, &tilingData, &pipe);                                                     \
        op.Process();                                                                                                  \
    } while (0)
#endif

template <bool MM_ALLTO_ALL_TRANS_X2, bool MM_ALLTO_ALL_HAS_BIAS, int MM_ALLTO_ALL_BIAS_DTYPE,
          int MM_ALLTO_ALL_SOC_VERSION>
__global__ __aicore__ void matmul_allto_all(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                            GM_ADDR comm_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y,
                                            GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // A2閵嗕竸3閻ㄥ埍ernel task闁粙鍣伴悽?:2濡€崇础
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    if constexpr (MM_ALLTO_ALL_SOC_VERSION == SOC_ASCEND910B) {
        MatmulAlltoAll<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y, MM_ALLTO_ALL_HAS_BIAS,
                       MM_ALLTO_ALL_TRANS_X2>
            op;
        op.Init(x1, x2, bias, x1_scale, x2_scale, y, workspaceGM, tilingGM);
        op.Process();
    } else if constexpr (MM_ALLTO_ALL_SOC_VERSION == SOC_ASCEND910_93) {
        TPipe pipe;
        // A3缂佹挻鐎担鎾存暈閸愬瞼鏁眛ilingkey閻ㄥ嫮绱拠鎴濈暞ASCENDC_TPL_TILING_STRUCT_SEL鐎圭偟骞?
        GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingDataA3, tilingData, tilingGM);
        if constexpr (MM_ALLTO_ALL_BIAS_DTYPE == TILINGKEY_TPL_FP16) {
            using DtypeBias = DTYPE_X1;
            MATMUL_ALLTO_ALL_A3_FP_IMPL(tilingData, pipe);
        } else if constexpr (MM_ALLTO_ALL_BIAS_DTYPE == TILINGKEY_TPL_FP32) {
            using DtypeBias = float;
            MATMUL_ALLTO_ALL_A3_FP_IMPL(tilingData, pipe);
        }
    }
#else
    if constexpr (MM_ALLTO_ALL_SOC_VERSION == SOC_ASCEND910B) {
        if constexpr (MM_ALLTO_ALL_BIAS_DTYPE == TILINGKEY_TPL_FP16) { // bias閺勵垰褰查柅澶婂弳閸欏偊绱?
            MatmulAlltoAll<DTYPE_X1, DTYPE_X2, float16_t, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y,
                           MM_ALLTO_ALL_HAS_BIAS, MM_ALLTO_ALL_TRANS_X2>
                op;
            op.Init(x1, x2, bias, x1_scale, x2_scale, y, workspaceGM, tilingGM);
            op.Process();
        } else if constexpr (MM_ALLTO_ALL_BIAS_DTYPE == TILINGKEY_TPL_FP32) {
            MatmulAlltoAll<DTYPE_X1, DTYPE_X2, float32_t, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y,
                           MM_ALLTO_ALL_HAS_BIAS, MM_ALLTO_ALL_TRANS_X2>
                op;
            op.Init(x1, x2, bias, x1_scale, x2_scale, y, workspaceGM, tilingGM);
            op.Process();
        } else if constexpr (MM_ALLTO_ALL_BIAS_DTYPE == TILINGKEY_TPL_BF16) {
            MatmulAlltoAll<DTYPE_X1, DTYPE_X2, bfloat16_t, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y,
                           MM_ALLTO_ALL_HAS_BIAS, MM_ALLTO_ALL_TRANS_X2>
                op;
            op.Init(x1, x2, bias, x1_scale, x2_scale, y, workspaceGM, tilingGM);
            op.Process();
        } else if constexpr (MM_ALLTO_ALL_BIAS_DTYPE == TILINGKEY_TPL_BF16) {
            MatmulAlltoAll<DTYPE_X1, DTYPE_X2, bfloat16_t, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y,
                           MM_ALLTO_ALL_HAS_BIAS, MM_ALLTO_ALL_TRANS_X2>
                op;
            op.Init(x1, x2, bias, x1_scale, x2_scale, y, workspaceGM, tilingGM);
            op.Process();
        }
    }
#endif
}
