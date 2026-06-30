/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  aclnn_dispatch_teardown.cpp —— 对外 aclnn 薄壳，转发框架生成的内层 aclnnInnerDispatchTeardown*。
 *  内层签名约定：全部输入 → 全部 attr(OpDef 序) → 全部输出 → ws → executor。纯 AIV，不调 NnopbaseSetHcclServerType。
 */
#include "aclnn_dispatch_teardown.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerDispatchTeardownGetWorkspaceSize(
    const aclTensor *tokenData, const aclTensor *tokenScale, const aclTensor *wishCnt, const aclTensor *expertIds,
    int64_t rankNum, int64_t expertPerRank,
    aclTensor *expandX, aclTensor *expandScale, aclTensor *sendCounts, aclTensor *expandIdx,
    aclTensor *expertTokenNums,
    uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerDispatchTeardown(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);

aclnnStatus aclnnDispatchTeardownGetWorkspaceSize(
    const aclTensor *tokenData, const aclTensor *tokenScale, const aclTensor *wishCnt, const aclTensor *expertIds,
    int64_t rankNum, int64_t expertPerRank,
    aclTensor *expandX, aclTensor *expandScale, aclTensor *sendCounts, aclTensor *expandIdx,
    aclTensor *expertTokenNums,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerDispatchTeardownGetWorkspaceSize(
        tokenData, tokenScale, wishCnt, expertIds, rankNum, expertPerRank,
        expandX, expandScale, sendCounts, expandIdx, expertTokenNums, workspaceSize, executor);
}

aclnnStatus aclnnDispatchTeardown(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream)
{
    return aclnnInnerDispatchTeardown(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
