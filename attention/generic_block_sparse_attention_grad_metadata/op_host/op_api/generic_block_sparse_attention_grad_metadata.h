#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *GenericBlockSparseAttentionGradMetadata(const aclTensor *, const aclTensor *, const aclTensor *,
                                                         const aclTensor *, const aclTensor *, const aclTensor *,
                                                         int64_t, int64_t, int64_t, int64_t, int64_t, const aclTensor *,
                                                         int64_t, char *, char *, int64_t, int64_t, int64_t, int64_t,
                                                         const aclTensor *, int64_t, int64_t, const char *,
                                                         aclOpExecutor *);
}

#endif
