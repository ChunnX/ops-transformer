/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ALL_TO_ALLV_GROUPED_MATMUL_V2_
#define OP_API_INC_ALL_TO_ALLV_GROUPED_MATMUL_V2_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Computes all-to-all-v followed by grouped matmul using an initialized CANN SHMEM/UDMA context.
 *
 * gmmX is a symmetric BF16 [M, K] allocation. gmmWeight is BF16 [E, K, N]. sendCounts is laid out as
 * [destination rank, destination local expert], recvCounts as [source rank, local expert], and both contain
 * epWorldSize * E non-negative values. gmmY is BF16 [sum(recvCounts), N].
 *
 * shmemContext is the device context pointer returned by the caller's initialized SHMEM runtime. The descriptor is
 * borrowed for executor construction; ownership remains with the caller. The returned workspace must be allocated
 * from the same symmetric SHMEM heap as gmmX.
 */
ACLNN_API aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, void *shmemContext, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, aclTensor *gmmY, uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnAlltoAllvGroupedMatMulV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_TO_ALLV_GROUPED_MATMUL_V2_
