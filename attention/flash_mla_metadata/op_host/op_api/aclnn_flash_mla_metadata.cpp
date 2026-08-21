/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_flash_mla_metadata.cpp
 * \brief
 */

#include "aclnn_flash_mla_metadata.h"
#include "l0_flash_mla_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#include "../flash_mla_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cacheSeqlensOptional, const aclTensor *sequsedQOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ,
    int64_t maxSeqlenKv, int64_t maskMode, const char *layoutQ, const char *layoutKv, const char *layoutOut,
    const aclTensor *metaData, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashMlaMetadata,
                   DFX_IN(cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional, numHeadsQ, numHeadsKv, headDim,
                          batchSize, maxSeqlenQ, maxSeqlenKv, maskMode, layoutQ, layoutKv, layoutOut),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = FlashMlaMetadataCheck::ParamsCheck(cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional,
                                                  numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv,
                                                  maskMode, layoutQ, layoutKv, layoutOut, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    // m0070 接口映射到 AICPU 4 输入槽位（l0 保持 flash_attn 布局不变）：
    //   slot0=cuSeqlensQ, slot1=cuSeqlensKv(nullptr), slot2=sequsedQ, slot3=sequsedKv←cacheSeqlens
    //   （AICPU kernel 从 sequsedKv 端口读每 batch KV cache 长度，cache_seqlens 即该语义）
    // win_left/win_right 为 l0 旧 attr 名，恒 -1（kernel 不消费）。
    const aclTensor *empty = nullptr;
    auto output = l0op::FlashMlaMetadata(cuSeqlensQOptional, empty, sequsedQOptional, cacheSeqlensOptional,
                                         batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, maskMode,
                                         -1, -1, layoutQ, layoutKv, layoutOut, socVersion, aicCoreNum,
                                         aivCoreNum, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashMlaMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashMlaMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
