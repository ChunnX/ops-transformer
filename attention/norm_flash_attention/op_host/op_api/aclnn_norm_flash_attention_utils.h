/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_NORM_FLASH_ATTENTION_UTILS_H
#define OP_HOST_OP_API_ACLNN_NORM_FLASH_ATTENTION_UTILS_H

#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "norm_flash_attention.h"

namespace norm_flash_attention {

using namespace op;

struct NormFlashAttentionParamsBase {
    const aclTensor *query = nullptr;
    const aclTensor *key = nullptr;
    const aclTensor *value = nullptr;

    const aclTensor *query_contiguous = nullptr;
    const aclTensor *key_contiguous = nullptr;
    const aclTensor *value_contiguous = nullptr;

    const aclTensor *attentionOut = nullptr;
};

class NormFlashAttentionParamsBuilder {
public:
    static NormFlashAttentionParamsBuilder Create(const aclTensor *query, const aclTensor *key,
                                                  const aclTensor *value, const aclTensor *attentionOut)
    {
        NormFlashAttentionParamsBuilder b;
        b.p_.query = query;
        b.p_.key = key;
        b.p_.value = value;
        b.p_.attentionOut = attentionOut;
        return b;
    }

    NormFlashAttentionParamsBase Build() const
    {
        return p_;
    }

private:
    NormFlashAttentionParamsBase p_;
};

class NormFlashAttentionHandler {
public:
    virtual ~NormFlashAttentionHandler() = default;

protected:
    virtual bool CheckNotNull(void)
    {
        OP_CHECK_NULL(params_.query, return false);
        OP_CHECK_NULL(params_.key, return false);
        OP_CHECK_NULL(params_.value, return false);
        OP_CHECK_NULL(params_.attentionOut, return false);
        return true;
    }

    virtual bool CheckEmptyTensor(void)
    {
        if (params_.query->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query tensor is empty");
            return false;
        }
        if (params_.key->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "key tensor is empty");
            return false;
        }
        if (params_.value->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "value tensor is empty");
            return false;
        }
        return true;
    }

    virtual bool CheckInputOutDims() = 0;
    virtual bool CheckInputOutShape() = 0;
    virtual bool CheckDtypeValid() = 0;
    virtual bool CheckFormat() = 0;

    virtual aclnnStatus CheckParams()
    {
        CHECK_RET(CheckNotNull(), ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(CheckEmptyTensor(), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckInputOutDims(), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckInputOutShape(), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDtypeValid(), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckFormat(), ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }

    virtual aclnnStatus ConvertDataContiguous()
    {
        params_.query_contiguous = l0op::Contiguous(params_.query, l0Executor_);
        CHECK_RET(params_.query_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        params_.key_contiguous = l0op::Contiguous(params_.key, l0Executor_);
        CHECK_RET(params_.key_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        params_.value_contiguous = l0op::Contiguous(params_.value, l0Executor_);
        CHECK_RET(params_.value_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        return ACLNN_SUCCESS;
    }

public:
    void Initialize(const char *interfaceName, NormFlashAttentionParamsBase &params,
                  uint64_t *workspaceSize, aclOpExecutor **executor)
    {
        interfaceName_ = interfaceName;
        params_ = params;
        workspaceSize_ = workspaceSize;
        executor_ = executor;
    }

    aclnnStatus Process()
    {
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        l0Executor_ = uniqueExecutor.get();

        auto ret = CheckParams();
        CHECK_RET(ret == ACLNN_SUCCESS, ret);

        if (params_.attentionOut->IsEmpty()) {
            *workspaceSize_ = 0ULL;
            uniqueExecutor.ReleaseTo(executor_);
            return ACLNN_SUCCESS;
        }

        ret = ConvertDataContiguous();
        CHECK_RET(ret == ACLNN_SUCCESS, ret);

        // TODO: 调用实际的算子实现
        // auto outParams = l0op::NormFlashAttention(
        //     params_.query_contiguous, params_.key_contiguous, params_.value_contiguous,
        //     l0Executor_);
        // CHECK_RET(outParams != std::tuple(nullptr), ACLNN_ERR_INNER_NULLPTR);

        // auto out1 = std::get<0>(outParams);
        // auto retView = l0op::ViewCopy(out1, params_.attentionOut, l0Executor_);
        // CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize_ = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor_);
        return ACLNN_SUCCESS;
    }

protected:
    std::string interfaceName_;
    NormFlashAttentionParamsBase params_;
    uint64_t *workspaceSize_;
    aclOpExecutor **executor_;
    aclOpExecutor *l0Executor_;
};

} // namespace norm_flash_attention

#endif // OP_HOST_OP_API_ACLNN_NORM_FLASH_ATTENTION_UTILS_H
