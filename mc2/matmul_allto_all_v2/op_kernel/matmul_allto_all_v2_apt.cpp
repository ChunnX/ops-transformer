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
 * \file matmul_allto_all_v2.cpp
 * \brief
 */
#include <cstring>
#include <lib/matmul_intf.h>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "common.h"
#include "./arch35/matmul_allto_all_v2_arch35.h"
#include "./arch35/matmul_allto_all_v2_pipeline.h"
#include "./arch35/matmul_allto_all_v2_tiling_key.h"
#include "./arch35/matmul_allto_all_v2_tiling_data.h"
#include "./arch35/kc_quant_matmul_allto_all_v2_arch35.h"
#include "./arch35/mx_quant_matmul_allto_all_v2_arch35.h"

using AscendC::HcclServerType;
using AscendC::TPipe;
using Mc2Kernel::MatmulAlltoAllV2PipeLine;
using Mc2Kernel::MatmulAlltoAllV2PipelineContext;
using MC2KernelTemplate::MC2AlltoAllContext;
using MC2KernelTemplate::MC2AlltoAllPrimitives;

#ifndef MATMUL_ALLTO_ALL_V2_APT_FP_IMPL
#define MATMUL_ALLTO_ALL_V2_APT_FP_IMPL(tilingData, pipe)                                                                 \
    do {                                                                                                               \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP(ComputationContextType);                                   \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(false, X2TRANSPOSE, ComputationType);                              \
        ComputationType matmulImplName(&pipe);                                                                         \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType);                                             \
        TransposeType transposeImplName(&pipe);                                                                        \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,             \
                                          MatmulAlltoAllV2TilingData, MC2AlltoAllPrimitives, 1, 0, CommunicationType);   \
        CommunicationType commImplName(&tilingData);                                                                   \
        using SchedulerContextType = MatmulAlltoAllV2PipelineContext<ComputationContextType>;                            \
        using SchedulerType =                                                                                          \
            MatmulAlltoAllV2PipeLine<ComputationType, TransposeType, CommunicationType, SchedulerContextType>;           \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);                               \
        Mc2Kernel::MatmulAlltoAllV2Arch35<SchedulerType, SchedulerContextType, MatmulAlltoAllV2TilingData> op(             \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, workspaceGM, &tilingData, &pipe);                                                     \
        op.Process();                                                                                                  \
    } while (0)
#endif

#ifndef MATMUL_ALLTO_ALL_V2_APT_KC_QUANT_IMPL
#define MATMUL_ALLTO_ALL_V2_APT_KC_QUANT_IMPL(tilingData, pipe)                                                           \
    do {                                                                                                               \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_QUANT(KcComputationContextType);                              \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(KcComputationType, DTYPE_X1, DTYPE_X2);                         \
        KcComputationType matmulImplName(&pipe);                                                                       \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, KcTransposeType);                                           \
        KcTransposeType transposeImplName(&pipe);                                                                      \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,             \
                                          QuantMatmulAlltoAllV2TilingData, MC2AlltoAllPrimitives, 1, 0,                  \
                                          CommunicationType);                                                          \
        CommunicationType commImplName(&tilingData);                                                                   \
        using SchedulerContextType = MatmulAlltoAllV2PipelineContext<KcComputationContextType>;                          \
        using SchedulerType =                                                                                          \
            MatmulAlltoAllV2PipeLine<KcComputationType, KcTransposeType, CommunicationType, SchedulerContextType, true>; \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);                               \
        Mc2Kernel::KcQuantMatmulAlltoAllV2Arch35<SchedulerType, SchedulerContextType, QuantMatmulAlltoAllV2TilingData> op( \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, x1Scale, x2Scale, x2Offset, workspaceGM, &tilingData, &pipe);                         \
        op.Process();                                                                                                  \
    } while (0)
#endif

#ifndef MATMUL_ALLTO_ALL_V2_APT_MX_QUANT_IMPL
#define MATMUL_ALLTO_ALL_V2_APT_MX_QUANT_IMPL(tilingData, pipe)                                                           \
    do {                                                                                                               \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_MX_QUANT(MxComputationContextType);                           \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_MX_QUANT(MxComputationType);                                          \
        MxComputationType matmulImplName(&pipe);                                                                       \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType);                                             \
        TransposeType transposeImplName(&pipe);                                                                        \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, HcclServerType::HCCL_SERVER_TYPE_CCU, MC2AlltoAllContext,             \
                                          QuantMatmulAlltoAllV2TilingData, MC2AlltoAllPrimitives, 1, 0,                  \
                                          CommunicationType);                                                          \
        CommunicationType commImplName(&tilingData);                                                                   \
        using SchedulerContextType = MatmulAlltoAllV2PipelineContext<MxComputationContextType>;                          \
        using SchedulerType =                                                                                          \
            MatmulAlltoAllV2PipeLine<MxComputationType, TransposeType, CommunicationType, SchedulerContextType>;         \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);                               \
        Mc2Kernel::MxQuantMatmulAlltoAllV2Arch35<SchedulerType, SchedulerContextType, QuantMatmulAlltoAllV2TilingData> op( \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, x1Scale, x2Scale, workspaceGM, &tilingData, &pipe);                                   \
        op.Process();                                                                                                  \
    } while (0)
#endif

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS>
__global__ __aicore__ void matmul_allto_all_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                            GM_ADDR commScale, GM_ADDR x1Offset, GM_ADDR x2Offset, GM_ADDR y,
                                            GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // kernel的使用类型，这里是cube和vic混用，cube是主核，cube:vec=1:2
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    // 注册默认的tilingdata，需要保证有且只有一个默认tilingdata被注册
    REGISTER_TILING_DEFAULT(MatmulAlltoAllV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllV2TilingData, tilingData, tilingGM);

    if constexpr (DTYPEBIAS == DTYPE_BIAS_SAME_WITH_X) {
        using DtypeBias = DTYPE_X1;
        MATMUL_ALLTO_ALL_V2_APT_FP_IMPL(tilingData, pipe);
    } else if constexpr (DTYPEBIAS == DTYPE_BIAS_FP32) {
        using DtypeBias = float;
        MATMUL_ALLTO_ALL_V2_APT_FP_IMPL(tilingData, pipe);
    }
#elif ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1))
    REGISTER_TILING_DEFAULT(QuantMatmulAlltoAllV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulAlltoAllV2TilingData, tilingData, tilingGM);

    MATMUL_ALLTO_ALL_V2_APT_MX_QUANT_IMPL(tilingData, pipe);
#else
    REGISTER_TILING_DEFAULT(QuantMatmulAlltoAllV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulAlltoAllV2TilingData, tilingData, tilingGM);

    if constexpr (QUANTMODE == KC_QUANT_MODE) {
        MATMUL_ALLTO_ALL_V2_APT_KC_QUANT_IMPL(tilingData, pipe);
    } else if constexpr (QUANTMODE == MX_QUANT_MODE) {
        MATMUL_ALLTO_ALL_V2_APT_MX_QUANT_IMPL(tilingData, pipe);
    }
#endif
}