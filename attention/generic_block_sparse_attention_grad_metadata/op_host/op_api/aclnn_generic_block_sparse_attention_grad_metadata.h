#ifndef ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_H
#define ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
    const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
    const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads,
    int64_t headDim, const aclIntArray *blockShape, int64_t isPackedGQA,
    char *layoutQ, char *layoutKv, int64_t maskType, int64_t softmaxPrecision,
    int64_t winLeft, int64_t winRight, aclTensor *metadata,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGradMetadata(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
