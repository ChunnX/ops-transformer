/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Bench：薄封装，将对外 aclnn 接口转发到 msopgen 生成的 `aclnnInnerInterleaveRope*`（与 ApplyRotaryPosEmb / RotaryPositionEmbedding 同源写法）。
 */

#include "aclnn_interleave_rope.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerInterleaveRopeGetWorkspaceSize(const aclTensor* x, const aclTensor* cos,
                                                          const aclTensor* sin, aclTensor* out, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor);
extern aclnnStatus aclnnInnerInterleaveRope(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

aclnnStatus aclnnInterleaveRopeGetWorkspaceSize(const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
                                                aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnInnerInterleaveRopeGetWorkspaceSize(x, cos, sin, out, workspaceSize, executor);
}

aclnnStatus aclnnInterleaveRope(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerInterleaveRope(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
