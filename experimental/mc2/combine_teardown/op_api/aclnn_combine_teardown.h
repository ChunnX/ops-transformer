/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  aclnn_combine_teardown.h —— CombineTeardown 的对外 aclnn 两段式接口。
 *  入参对应 op 定义：combine_data(fp8)/combine_scale(e8m0)/expert_scales/expand_idx/expert_ids(输入) → x(bf16 输出)
 *                    + attr rank_num/expert_per_rank。
 */
#ifndef OP_API_INC_COMBINE_TEARDOWN_H
#define OP_API_INC_COMBINE_TEARDOWN_H

#include "aclnn/aclnn_base.h"
#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnCombineTeardownGetWorkspaceSize(
    const aclTensor *combineData, const aclTensor *combineScale, const aclTensor *expertScales,
    const aclTensor *expandIdx, const aclTensor *expertIds, int64_t rankNum, int64_t expertPerRank,
    aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnCombineTeardown(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_COMBINE_TEARDOWN_H
