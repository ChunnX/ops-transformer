#ifndef ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_TORCH_H
#define ACLNN_GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_TORCH_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, int64_t, int64_t, int64_t, int64_t,
    int64_t, const aclIntArray *, int64_t, char *, char *, int64_t, int64_t,
    int64_t, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
aclnnStatus aclnnGenericBlockSparseAttentionGradMetadata(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

#ifdef __cplusplus
}
#endif

#endif
