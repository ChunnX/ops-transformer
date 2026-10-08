/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_GROUPED_ROTATE_QUANT_H
#define ACLNN_GROUPED_ROTATE_QUANT_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Prepare grouped block rotation and MX quantization on Ascend 950.
 * @domain aclnn_ops_infer
 *
 * @param [in] x FLOAT16/BFLOAT16 ND tensors, with a common dtype. For groupType=-1,
 * each tensor is an independent RotateQuant input of rank 1..7. For groupType=0,
 * use one [M,N] tensor or E tensors [M_i,N], with a common N.
 * @param [in] rotation One shared tensor or E tensors, with the same dtype as x.
 * Each tensor is [R,R] or [N/R,R,R], R in {32,64,128}, N divisible by R.
 * The first dimension of a rank-3 rotation is the number of column blocks, not E.
 * @param [in] alpha Optional shared BFLOAT16 ND tensor [1], as in RotateQuant.
 * Clamping is enabled only for 0 < alpha < 1; other values or null skip it.
 * @param [in] groupListOptional Device INT64 ND tensor [E]. Required for a single
 * M-grouped x, optional for E separate x tensors, and null for groupType=-1.
 * Boundaries must describe exactly all M rows and agree with separate x/y shapes.
 * Empty groups are allowed. Values are consumed at runtime, not read by this API.
 * @param [in] splitItem As in GMM V5: 0/1 selects E outputs; 2/3 selects one
 * concatenated output. The input list length determines the normalized value.
 * groupType=-1 only supports 0/1. yOut and scaleOut use the same grouping.
 * @param [in] groupType -1: independent tensors; 0: M grouping. N/K grouping is unsupported.
 * @param [in] groupListType 0: cumulative end offsets; 1: per-group row counts.
 * Sparse groupListType=2 is unsupported. The number of groups is in [1,1024].
 * @param [in] axis -1 or the last axis of every x tensor; normalized to -1 internally.
 * @param [in] roundMode Null defaults to "rint". MXFP4 supports "rint", "round", "floor";
 * MXFP8 supports "rint".
 * @param [in] scaleAlg MXFP4: 0/2; MXFP8: 0/1.
 * @param [in] dstTypeMax [6,12] for scaleAlg=2; otherwise 0.
 * @param [in] trans Output transpose, currently only false.
 * @param [in] tuningConfigOptional Optional tuning hints, as in GMM V5. The first
 * element is the expected row count per group, in [0,max(x[i].M)]; remaining
 * elements are reserved for tiling. This hint never replaces runtime group sizes.
 * @param [out] yOut ND tensors of one common dtype: FLOAT4_E2M1, FLOAT8_E4M3FN or
 * FLOAT8_E5M2. Shapes preserve input dimensions, with M split/concatenated as above.
 * @param [out] scaleOut FLOAT8_E8M0 ND tensors. Replace each yOut's last dimension N
 * by ceil(N/64) and append dimension 2. Both outputs support non-contiguous views.
 * @param [out] workspaceSize Required device workspace size.
 * @param [out] executor Prepared execution graph; consumed by the second-stage API.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedRotateQuantGetWorkspaceSize(
    const aclTensorList *x, const aclTensorList *rotation, const aclTensor *alpha,
    const aclTensor *groupListOptional, int64_t splitItem, int64_t groupType, int64_t groupListType,
    int64_t axis, const char *roundMode, int64_t scaleAlg, double dstTypeMax, bool trans,
    const aclIntArray *tuningConfigOptional, aclTensorList *yOut, aclTensorList *scaleOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/** @brief Execute the graph prepared by aclnnGroupedRotateQuantGetWorkspaceSize. */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedRotateQuant(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // ACLNN_GROUPED_ROTATE_QUANT_H
