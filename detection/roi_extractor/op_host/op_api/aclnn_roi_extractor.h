/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_roi_extractor.h
 * \brief aclnnRoiExtractor two-segment API.
 */
#ifndef LEVEL2_ACLNN_ROI_EXTRACTOR_H_
#define LEVEL2_ACLNN_ROI_EXTRACTOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRoiExtractor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] feats: npu device侧的aclTensorList，1~5个层级特征图（NCHW），
 * 数据类型支持FLOAT16、FLOAT32，数据格式支持ND。
 * @param [in] rois: npu device侧的aclTensor，[K,5]=(batch_idx,x1,y1,x2,y2)或
 * [K,4]（搭配index），数据类型与feats一致。
 * @param [in] index: npu device侧的aclTensor，[K] INT32，rois为[K,4]时必填。
 * @param [in] finestScale: host侧int64_t，层级映射基准尺度，默认56。
 * @param [in] roiScaleFactor: host侧double，>0时层级映射前RoI坐标除以该因子，默认0。
 * @param [in] spatialScales: host侧aclFloatArray，每层坐标缩放因子（乘法语义），
 * 长度等于feats层数，典型{0.25, 0.125, 0.0625, 0.03125}。
 * @param [in] pooledHeight: host侧int64_t，输出特征高，[1, 56]，支持非方形。
 * @param [in] pooledWidth: host侧int64_t，输出特征宽，[1, 56]。
 * @param [in] samplingRatio: host侧int64_t，每bin采样点数，0=自适应ceil(bin)。
 * @param [in] poolMode: host侧char*，池化模式，"avg"或"max"。
 * @param [in] aligned: host侧bool，true时坐标-0.5对齐（mmdet默认语义）。
 * @param [in] y: npu device侧的aclTensor，[K, C, pooledHeight, pooledWidth]，
 * 数据类型与feats一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRoiExtractorGetWorkspaceSize(
    const aclTensorList *feats, const aclTensor *rois, const aclTensor *index, int64_t finestScale,
    double roiScaleFactor, const aclFloatArray *spatialScales, int64_t pooledHeight, int64_t pooledWidth,
    int64_t samplingRatio, char *poolMode, bool aligned, aclTensor *y, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnRoiExtractor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRoiExtractor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // LEVEL2_ACLNN_ROI_EXTRACTOR_H_
