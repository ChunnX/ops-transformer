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
 * \file aclnn_all_gather_quant_matmul.h
 * \brief
 */
#ifndef OP_API_INC_ALL_GATHER_QUANT_MATMUL_
#define OP_API_INC_ALL_GATHER_QUANT_MATMUL_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现 AllGather + QuantMatmul 融合计算
 * @brief aclnnAllGatherQuantMatmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x1: AllGather的fp8输入A，数据类型支持 FLOAT8_E4M3FN、FLOAT8_E5M2，数据格式支持 ND。
 * @param [in] x2: QBMM的fp8权重B，数据类型支持 FLOAT8_E4M3FN、FLOAT8_E5M2，数据格式支持 ND、FRACTAL_NZ。
 * @param [in] scaleInv1: A矩阵的e8m0 scale，数据类型支持 FLOAT8_E8M0，数据格式支持 ND。
 * @param [in] scaleInv2: B矩阵的e8m0 scale，数据类型支持 FLOAT8_E8M0，数据格式支持 ND。
 * @param [in] group: 通信域标识，数据类型支持：string。
 * @param [in] rankSize: 通信域大小，数据类型支持：int64。
 * @param [in] transposeX2: x2是否转置，数据类型支持：bool，当前只支持false。
 * @param [out] output: QBMM bf16输出，数据类型支持 BFLOAT16、FLOAT16，数据格式支持 ND。
 * @param [out] allGatherDataOut: AllGather后的fp8数据，数据类型支持 FLOAT8_E4M3FN、FLOAT8_E5M2，数据格式支持 ND。
 * @param [out] allGatherScalesOut: AllGather后的e8m0 scales，数据类型支持 FLOAT8_E8M0，数据格式支持 ND。
 * @param [out] workspaceSize: 返回需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAllGatherQuantMatmulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* scaleInv1, const aclTensor* scaleInv2,
    const char* group, int64_t rankSize, bool transposeX2,
    aclTensor* output, aclTensor* allGatherDataOut, aclTensor* allGatherScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAllGatherQuantMatmul的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存地址。
 * @param [in] workspacesize: 在Device侧申请的workspace大小，由第一段接口aclnnAllGatherQuantMatmulGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的Stream。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAllGatherQuantMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ALL_GATHER_QUANT_MATMUL_