/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  aclnn_dispatch_teardown.h —— DispatchTeardown(mxfp8 多核) 对外 aclnn 两段式接口。
 *  顺序：全部输入 → 全部 attr → 全部输出 → ws → executor。
 *  输入 token_data(fp8)/token_scale(e8m0)/wish_cnt/expert_ids；输出 expand_x(fp8)/expand_scale(e8m0)/
 *  send_counts/expand_idx/expert_token_nums。
 */
#ifndef OP_API_INC_DISPATCH_TEARDOWN_H
#define OP_API_INC_DISPATCH_TEARDOWN_H

#include "aclnn/aclnn_base.h"
#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnDispatchTeardownGetWorkspaceSize(
    const aclTensor *tokenData, const aclTensor *tokenScale, const aclTensor *wishCnt, const aclTensor *expertIds,
    int64_t rankNum, int64_t expertPerRank,
    aclTensor *expandX, aclTensor *expandScale, aclTensor *sendCounts, aclTensor *expandIdx,
    aclTensor *expertTokenNums,
    uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnDispatchTeardown(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DISPATCH_TEARDOWN_H
