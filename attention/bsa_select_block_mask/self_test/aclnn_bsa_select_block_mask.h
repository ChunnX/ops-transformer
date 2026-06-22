
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BSASELECT_BLOCK_MASK_H_
#define ACLNN_BSASELECT_BLOCK_MASK_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBSASelectBlockMaskGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * blockShapeOptionalOptional : optional
 * postBlockShapeOptionalOptional : optional
 * actualSeqLengthsOptionalOptional : optional
 * actualSeqLengthsKvOptionalOptional : optional
 * actualBlockLenQueryOptionalOptional : optional
 * actualBlockLenKeyOptionalOptional : optional
 * qInputLayoutOptional : optional
 * kvInputLayoutOptional : optional
 * numKeyValueHeads : optional
 * scaleValue : optional
 * sparsity : required
 * blockSparseMaskOutOut : required
 * qCompOutOut : required
 * kCompOutOut : required
 * scoreOutOut : required
 * softmaxOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBSASelectBlockMaskGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclIntArray *blockShapeOptionalOptional,
    const aclIntArray *postBlockShapeOptionalOptional,
    const aclIntArray *actualSeqLengthsOptionalOptional,
    const aclIntArray *actualSeqLengthsKvOptionalOptional,
    const aclIntArray *actualBlockLenQueryOptionalOptional,
    const aclIntArray *actualBlockLenKeyOptionalOptional,
    char *qInputLayoutOptional,
    char *kvInputLayoutOptional,
    int64_t numKeyValueHeads,
    double scaleValue,
    double sparsity,
    const aclTensor *blockSparseMaskOutOut,
    const aclTensor *qCompOutOut,
    const aclTensor *kCompOutOut,
    const aclTensor *scoreOutOut,
    const aclTensor *softmaxOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBSASelectBlockMask
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBSASelectBlockMask(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
