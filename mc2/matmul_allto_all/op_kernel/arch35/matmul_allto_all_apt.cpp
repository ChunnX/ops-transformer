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
#include "../matmul_allto_all_kernel_base.h"
#include "../matmul_allto_all_pipeline.h"
#include "matmul_allto_all_tiling_key.h"
#include "matmul_allto_all_tiling_data.h"
#include "kc_quant_matmul_allto_all_arch35.h"
#include "mx_quant_matmul_allto_all_arch35.h"

using AscendC::HcclServerType;
using AscendC::TPipe;
using Mc2Kernel::MatmulAlltoAllPipeLine;
using Mc2Kernel::MatmulAlltoAllPipelineContext;
using MC2KernelTemplate::MC2AlltoAllContext;
using MC2KernelTemplate::MC2AlltoAllPrimitives;

#ifndef MATMUL_ALLTO_ALL_APT_FP_IMPL
#define MATMUL_ALLTO_ALL_APT_FP_IMPL(tilingData, pipe, ServerType)                                                     \
    do {                                                                                                               \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_FP(ComputationContextType);                                   \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_FP(false, X2TRANSPOSE, ComputationType);                              \
        ComputationType matmulImplName(&pipe);                                                                         \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType);                                             \
        TransposeType transposeImplName(&pipe);                                                                        \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, ServerType, MC2AlltoAllContext, MatmulAlltoAllTilingData,             \
                                          MC2AlltoAllPrimitives, 1, 0, CommunicationType);                             \
        CommunicationType commImplName(&tilingData);                                                                   \
        using SchedulerContextType = MatmulAlltoAllPipelineContext<ComputationContextType>;                            \
        using SchedulerType =                                                                                          \
            MatmulAlltoAllPipeLine<ComputationType, TransposeType, CommunicationType, SchedulerContextType>;           \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);                               \
        Mc2Kernel::MatmulAlltoAllKernelBase<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingData> op(         \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, workspaceGM, &tilingData, &pipe);                                                     \
        op.Process();                                                                                                  \
    } while (0)
#endif

#ifndef MATMUL_ALLTO_ALL_APT_KC_QUANT_IMPL
#define MATMUL_ALLTO_ALL_APT_KC_QUANT_IMPL(tilingData, pipe, ServerType)                                               \
    do {                                                                                                               \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_QUANT(KcComputationContextType);                              \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_QUANT(KcComputationType, DTYPE_X1, DTYPE_X2);                         \
        KcComputationType matmulImplName(&pipe);                                                                       \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, KcTransposeType);                                           \
        KcTransposeType transposeImplName(&pipe);                                                                      \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, ServerType, MC2AlltoAllContext, QuantMatmulAlltoAllTilingData,        \
                                          MC2AlltoAllPrimitives, 1, 0, CommunicationType);                             \
        CommunicationType commImplName(&tilingData);                                                                   \
        using SchedulerContextType = MatmulAlltoAllPipelineContext<KcComputationContextType>;                          \
        using SchedulerType =                                                                                          \
            MatmulAlltoAllPipeLine<KcComputationType, KcTransposeType, CommunicationType, SchedulerContextType, true>; \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);                               \
        Mc2Kernel::KcQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, QuantMatmulAlltoAllTilingData> op( \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, x1Scale, x2Scale, x2Offset, workspaceGM, &tilingData, &pipe);                         \
        op.Process();                                                                                                  \
    } while (0)
#endif

#ifndef MATMUL_ALLTO_ALL_APT_MX_QUANT_IMPL
#define MATMUL_ALLTO_ALL_APT_MX_QUANT_IMPL(tilingData, pipe, ServerType)                                               \
    do {                                                                                                               \
        DEFINE_MC2_MATMUL_CONTEXT_FOR_MATMUL_COMPUTATION_MX_QUANT(MxComputationContextType);                           \
        DEFINE_MC2_MATMUL_FOR_MATMUL_COMPUTATION_MX_QUANT(MxComputationType);                                          \
        MxComputationType matmulImplName(&pipe);                                                                       \
        DEFINE_MC2_TRANSPOSE_FOR_MATH_COMPUTATION(DTYPE_Y, TransposeType);                                             \
        TransposeType transposeImplName(&pipe);                                                                        \
        DEFINE_MC2_HCCL_FOR_COMMUNICATION(false, ServerType, MC2AlltoAllContext, QuantMatmulAlltoAllTilingData,        \
                                          MC2AlltoAllPrimitives, 1, 0, CommunicationType);                             \
        CommunicationType commImplName(&tilingData);                                                                   \
        using SchedulerContextType = MatmulAlltoAllPipelineContext<MxComputationContextType>;                          \
        using SchedulerType =                                                                                          \
            MatmulAlltoAllPipeLine<MxComputationType, TransposeType, CommunicationType, SchedulerContextType>;         \
        SchedulerType SchedulerImpl(&matmulImplName, &transposeImplName, &commImplName);                               \
        Mc2Kernel::MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, QuantMatmulAlltoAllTilingData> op( \
            &SchedulerImpl);                                                                                           \
        op.Init(x1, x2, bias, y, x1Scale, x2Scale, workspaceGM, &tilingData, &pipe);                                   \
        op.Process();                                                                                                  \
    } while (0)
#endif

template <uint32_t QUANTMODE, bool X2TRANSPOSE, uint32_t DTYPEBIAS, uint32_t COMMMODE>
__global__ __aicore__ void matmul_allto_all(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                            GM_ADDR commScale, GM_ADDR x1Offset, GM_ADDR x2Offset, GM_ADDR y,
                                            GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // kernel閻ㄥ嫪濞囬悽銊ц閸ㄥ绱濇潻娆撳櫡閺勭棗ube閸滃瘉ic濞ｉ鏁ら敍瀹憉be閺勵垯瀵岄弽闈╃礉cube:vec=1:2
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    constexpr HcclServerType commServerType =
        (COMMMODE == COMMMODE_CCU) ? HcclServerType::HCCL_SERVER_TYPE_CCU : HcclServerType::HCCL_SERVER_TYPE_AICPU;

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    // 濞夈劌鍞芥妯款吇閻ㄥ墖ilingdata閿涘矂娓剁憰浣风箽鐠囦焦婀佹稉鏂垮涧閺堝绔存稉顏堢帛鐠侇槡ilingdata鐞氼偅鏁為崘?    REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingData, tilingData, tilingGM);

    if constexpr (DTYPEBIAS == DTYPE_BIAS_SAME_WITH_X) {
        using DtypeBias = DTYPE_X1;
        MATMUL_ALLTO_ALL_APT_FP_IMPL(tilingData, pipe, commServerType);
    } else if constexpr (DTYPEBIAS == DTYPE_BIAS_FP32) {
        using DtypeBias = float;
        MATMUL_ALLTO_ALL_APT_FP_IMPL(tilingData, pipe, commServerType);
    }
#elif ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1))
    REGISTER_TILING_DEFAULT(QuantMatmulAlltoAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulAlltoAllTilingData, tilingData, tilingGM);

    MATMUL_ALLTO_ALL_APT_MX_QUANT_IMPL(tilingData, pipe, commServerType);
#else
    REGISTER_TILING_DEFAULT(QuantMatmulAlltoAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMatmulAlltoAllTilingData, tilingData, tilingGM);

    if constexpr (QUANTMODE == KC_QUANT_MODE) {
        MATMUL_ALLTO_ALL_APT_KC_QUANT_IMPL(tilingData, pipe, commServerType);
    } else if constexpr (QUANTMODE == MX_QUANT_MODE) {
        MATMUL_ALLTO_ALL_APT_MX_QUANT_IMPL(tilingData, pipe, commServerType);
    }
#endif
}
