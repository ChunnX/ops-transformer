#ifndef ACLNN_SAGE_ATTENTION2_H
#define ACLNN_SAGE_ATTENTION2_H

#include <stdint.h>
#include <stdbool.h>
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const char *tensorLayout,
    bool isCausal,
    const char *qkQuantGran,
    double smScale,
    const char *pvAccumDtype,
    bool smoothK,
    bool smoothV,
    bool returnLse,
    const aclTensor *out,
    const aclTensor *lse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SAGE_ATTENTION2_H
