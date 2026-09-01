
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_ALL_GATHER_MATMUL_V3_H_
#define ACLNN_INNER_ALL_GATHER_MATMUL_V3_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerAllGatherMatmulV3GetWorkspaceSize
 * parameters :
 * context : required
 * x1 : required
 * x2 : required
 * biasOptional : optional
 * x1ScaleOptional : optional
 * x2ScaleOptional : optional
 * group : required
 * hcclBufferSize : required
 * isTransA : optional
 * isTransB : optional
 * rankSize : optional
 * groupSize : optional
 * yDtype : optional
 * commModeOptional : optional
 * yOut : required
 * gatherOutOut : required
 * amaxOutOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerAllGatherMatmulV3GetWorkspaceSize(
    const aclTensor *context,
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional,
    char *group,
    int64_t hcclBufferSize,
    bool isTransA,
    bool isTransB,
    int64_t rankSize,
    int64_t groupSize,
    int64_t yDtype,
    char *commModeOptional,
    const aclTensor *yOut,
    const aclTensor *gatherOutOut,
    const aclTensor *amaxOutOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerAllGatherMatmulV3
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerAllGatherMatmulV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
