/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  aclnn_combine_teardown.cpp —— 对外 aclnn 薄壳，转发框架生成的内层 aclnnInnerCombineTeardown*。
 *  内层签名约定：全部输入 → 全部 attr(OpDef 序) → 全部输出 → ws → executor。纯 AIV、不调 NnopbaseSetHcclServerType。
 */
#include "aclnn_combine_teardown.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerCombineTeardownGetWorkspaceSize(
    const aclTensor *combineData, const aclTensor *combineScale, const aclTensor *expertScales,
    const aclTensor *expandIdx, const aclTensor *expertIds, int64_t rankNum, int64_t expertPerRank,
    aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerCombineTeardown(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream);

aclnnStatus aclnnCombineTeardownGetWorkspaceSize(
    const aclTensor *combineData, const aclTensor *combineScale, const aclTensor *expertScales,
    const aclTensor *expandIdx, const aclTensor *expertIds, int64_t rankNum, int64_t expertPerRank,
    aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerCombineTeardownGetWorkspaceSize(
        combineData, combineScale, expertScales, expandIdx, expertIds, rankNum, expertPerRank,
        x, workspaceSize, executor);
}

aclnnStatus aclnnCombineTeardown(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream)
{
    return aclnnInnerCombineTeardown(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
