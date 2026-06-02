/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_DISPATCH_LAYOUT_H_
#define ACLNN_DISPATCH_LAYOUT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

/**
 * @brief 获取 DispatchLayout 算子 workspace 大小（第一段接口）。
 *
 * round = ceil(numTokens / perRoundTokens)
 *
 * @param [in] topkIdx 专家索引，int64，shape [numTokens, numTopk]，无效专家为 -1。
 * @param [in] numTokens 本 rank token 数，须与 topkIdx 第 0 维一致。
 * @param [in] numRanks EP 组大小。
 * @param [in] numExperts 专家总数，须能被 numRanks 整除。
 * @param [in] numTopk 每 token Top-K，须与 topkIdx 第 1 维一致。
 * @param [in] localRankSize 单机 rank 数。
 * @param [in] perRoundTokens 每轮 token 上限。
 * @param [in] rankId 当前 EP rank，[0, numRanks)。
 * @param [out] numTokensPerRank int32，shape [numRanks]。
 * @param [out] numTokensPerExpert int32，shape [round * numExperts]。
 * @param [out] isTokenInRank int32（0/1），shape [round * perRoundTokens, numRanks]。
 * @param [out] extHandleData 长度为 2 的 TensorList：
 *        [0] notifySendData，shape [round * (numExperts / numRanks)]；
 *        [1] sendTokenIdxSmall，shape [numTokens * numTopk]。
 * @param [out] workspaceSize workspace 字节数。
 * @param [out] executor 执行器上下文。
 */
ACLNN_API aclnnStatus aclnnDispatchLayoutGetWorkspaceSize(
    const aclTensor *topkIdx, int64_t numTokens, int64_t numRanks, int64_t numExperts, int64_t numTopk,
    int64_t localRankSize, int32_t perRoundTokens, int32_t rankId, const aclTensor *numTokensPerRank,
    const aclTensor *numTokensPerExpert, const aclTensor *isTokenInRank, const aclTensorList *extHandleData,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief 执行 DispatchLayout 计算（第二段接口）。
 */
ACLNN_API aclnnStatus aclnnDispatchLayout(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DISPATCH_LAYOUT_
