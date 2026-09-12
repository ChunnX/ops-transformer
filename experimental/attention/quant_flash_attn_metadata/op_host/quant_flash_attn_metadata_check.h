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
 * \file quant_flash_attn_metadata_check.h
 * \brief 骨架阶段：ParamsCheck 先不做任何约束校验，后续实现阶段再收紧
 */

#include <unordered_set>
#include <string>
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

class QuantFlashAttnMetadataCheck {
public:
    static inline aclnnStatus ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
                                          const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                                          int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                          int64_t numHeadsKv, int64_t headDim, int64_t headDimV, int64_t quantMode,
                                          int64_t maskMode, int64_t winLeft, int64_t winRight, const char *layoutQ,
                                          const char *layoutQDescale, const char *layoutKv, const char *layoutOut,
                                          bool isGradEnabled, const aclTensor *metadata)
    {
        (void)cuSeqlensQOptional;
        (void)cuSeqlensKvOptional;
        (void)sequsedQOptional;
        (void)sequsedKvOptional;
        (void)batchSize;
        (void)maxSeqlenQ;
        (void)maxSeqlenKv;
        (void)numHeadsQ;
        (void)numHeadsKv;
        (void)headDim;
        (void)headDimV;
        (void)quantMode;
        (void)maskMode;
        (void)winLeft;
        (void)winRight;
        (void)layoutQ;
        (void)layoutQDescale;
        (void)layoutKv;
        (void)layoutOut;
        (void)isGradEnabled;
        (void)metadata;
        return ACLNN_SUCCESS;
    }
};

#ifdef __cplusplus
}
#endif