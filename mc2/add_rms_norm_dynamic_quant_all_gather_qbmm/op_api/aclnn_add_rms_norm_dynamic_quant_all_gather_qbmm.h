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
 * \file aclnn_add_rms_norm_dynamic_quant_all_gather_qbmm.h
 * \brief
 */
#ifndef OP_API_INC_ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_
#define OP_API_INC_ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * TODO: complete the comments 
 * 算子功能：实现add + rmsNorm + dynamicQuant + allGather + qbmm融合计算
 * @brief aclnnAddRmsNormDynamicQuantAllGatherQbmm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x1: AddRmsNorm的输入x1，不支持空Tensor，数据类型支持BFLOAT16、FLOAT16，数据格式支持ND。
 * @param [in] x2: QBMM的输入x2，不支持空Tensor，数据类型支持INT8，数据格式支持ND、FRACTAL_NZ。
 * @param [in] residual: AddRmsNorm的输入residual，不支持空Tensor，数据类型支持BFLOAT16、FLOAT16，数据格式支持ND。
 * @param [in] y: AddRmsNorm的输入y，不支持空Tensor，数据类型支持BFLOAT16、FLOAT16，数据格式支持ND。
 * @param [in] gamma: AddRmsNorm的输入gamma，不支持空Tensor，数据类型支持FLOAT，数据格式支持ND。
 * @param [in] scale: QBMM的输入scale，不支持空Tensor，数据类型支持BFLOAT16、FLOAT，数据格式支持ND。
 * @param [in] smoothScale: AddRmsNorm的输入smoothScale，支持空Tensor，数据类型支持FLOAT，数据格式支持ND。
 * @param [in] bias: QBMM的输入bias，支持空Tensor，数据类型支持INT32，数据格式支持ND。
 * @param [in] group: 通信域标识，数据类型支持：string。
 * @param [in] rankSize: 通信域大小，数据类型支持：int64。
 * @param [in] transposeX2: x2是否转置，数据类型支持：bool，当前只支持false。
 * @param [in] dtype: 输出类型，数据类型支持：int64，当前只支持-1，即int8。
 * @param [in] residualNormMode: 暂未使用，数据类型支持：int64，当前只支持0。
 * @param [in] outputOptional: 是否输出中间过程ARN、DynamicQuant以及AllGather的结果，数据类型支持：bool。
 * @param [out] output: 整体输出，不支持空Tensor，数据类型支持BFLOAT16、FLOAT16，数据格式支持ND。
 * @param [out] z: residualNormMode为0时，对应x1+y+residual的输出；residualNormMode为0时，对应NormOut的输出；不支持空Tensor，数据类型支持BFLOAT16、FLOAT16，数据格式支持ND。
 * @param [out] dynamicQuantOut: ARN+dynamic quant的输出，不支持空Tensor，数据类型支持INT8，数据格式支持ND。
 * @param [out] allGatherDataOut: AllGather后的数据，不支持空Tensor，数据类型支持INT8，数据格式支持ND。
 * @param [out] allGatherScalesOut: AllGather后的scales，不支持空Tensor，数据类型支持FLOAT，数据格式支持ND。
 * @param [out] workspaceSize: 返回需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAddRmsNormDynamicQuantAllGatherQbmmGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* residual, const aclTensor* y, const aclTensor* gamma,
    const aclTensor* scaleInv1, const aclTensor* scaleInv2, const aclTensor* smoothScale, const aclTensor* bias, const char* group, int64_t rankSize,
    bool transposeX2, int64_t dtype, int64_t residualNormMode, bool outputOptional, aclTensor* output, aclTensor* z,
    aclTensor* dynamicQuantOut, aclTensor* allGatherDataOut, aclTensor* allGatherScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddRmsNormDynamicQuantAllGatherQbmm的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存地址。
 * @param [in] workspacesize: 在Device侧申请的workspace大小，由第一段接口aclnnAddRmsNormDynamicQuantAllGatherQbmmGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的Stream。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAddRmsNormDynamicQuantAllGatherQbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_