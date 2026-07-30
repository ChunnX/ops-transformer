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
 * \file aclnn_allto_all_matmul_apace.h
 * \brief aclnn API for AlltoAllMatmulApace (MX-quant FP8, apace UDMA path)
 */

#ifndef OP_API_INC_ALL_TO_ALL_MATMUL_APACE_H_
#define OP_API_INC_ALL_TO_ALL_MATMUL_APACE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAlltoAllMatmulApace 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] context: apace通信上下文张量，数据类型int32。
 * @param [in] x1: 左矩阵输入张量，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2。
 * @param [in] x2: 右矩阵输入张量，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2。
 * @param [in] biasOptional: 可选输入张量，偏置项，暂不支持，传入nullptr。
 * @param [in] x1Scale: 左矩阵的MX量化参数，数据类型FLOAT8_E8M0。
 * @param [in] x2Scale: 右矩阵的MX量化参数，数据类型FLOAT8_E8M0。
 * @param [in] group: 通信域标识。
 * @param [in] worldSize: 通信域大小。
 * @param [in] hcclBufferSize: HCCL共享buffer大小。
 * @param [in] transposeX2: 右矩阵是否转置。
 * @param [in] all2allAxesOptional: all2all方向，暂不支持，传入nullptr。
 * @param [out] output: 输出结果。
 * @param [out] all2allOutOptional: all-to-all通信的输出结果，暂不支持，传入nullptr。
 * @param [out] workspaceSize: workspace大小。
 * @param [out] executor: 执行器。
 *
 * @return aclnnStatus: 0成功。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllMatmulApaceGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional, const aclTensor *x2ScaleOptional, const aclTensor *commScaleOptional,
    const aclTensor *x1OffsetOptional, const aclTensor *x2OffsetOptional, const char *group, int64_t worldSize,
    int64_t hcclBufferSize, const aclIntArray *alltoAllAxesOptional, int64_t x1QuantMode, int64_t x2QuantMode,
    int64_t commQuantMode, int64_t commQuantDtype, int64_t x1QuantDtype, bool transposeX1, bool transposeX2,
    int64_t groupSize, const char *commMode, int64_t precisionMode, const aclTensor *output,
    const aclTensor *alltoAllOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Second phase: execute AlltoAllMatmulApace on the given stream.
 *
 * @param [in] workspace: Device workspace memory.
 * @param [in] workspaceSize: Workspace size (from GetWorkspaceSize).
 * @param [in] executor: Executor handle (from GetWorkspaceSize).
 * @param [in] stream: aclrtStream for kernel launch.
 *
 * @return aclnnStatus: 0 on success.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllMatmulApace(void *workspace, uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_TO_ALL_MATMUL_APACE_H_
