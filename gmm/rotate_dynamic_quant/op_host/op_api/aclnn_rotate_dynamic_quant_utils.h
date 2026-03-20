/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_ROTATE_DYNAMIC_QUANT_UTILS_H
#define OP_HOST_OP_API_ACLNN_ROTATE_DYNAMIC_QUANT_UTILS_H

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
#include "rotate_dynamic_quant.h"

namespace rotate_dynamic_quant {

using namespace op;

/**
 * @brief RotateDynamicQuant算子参数结构体
 */
struct RotateDynamicQuantParamsBase {
    // ========== 原始输入 tensor（用户传入，不可修改） ==========
    const aclTensor *x = nullptr;
    const aclTensor *rot = nullptr;

    // ========== 连续格式 tensor（内部使用，指向转换后的 tensor） ==========
    const aclTensor *x_contiguous = nullptr;
    const aclTensor *rot_contiguous = nullptr;

    // ========== 输出 tensor（用户传入，用于拷贝结果） ==========
    const aclTensor *yOut = nullptr;
    const aclTensor *scaleOut = nullptr;

    // ========== 属性参数 ==========
    int32_t dst_type = 0;  // 0=INT8, 1=INT32
};

/**
 * @brief Builder 类，用于构建 RotateDynamicQuantParamsBase 参数结构体
 */
class RotateDynamicQuantParamsBuilder {
public:
    static RotateDynamicQuantParamsBuilder Create(const aclTensor *x, const aclTensor *rot,
                                                   const aclTensor *yOut, const aclTensor *scaleOut)
    {
        RotateDynamicQuantParamsBuilder b;
        b.p_.x = x;
        b.p_.rot = rot;
        b.p_.yOut = yOut;
        b.p_.scaleOut = scaleOut;
        return b;
    }

    RotateDynamicQuantParamsBuilder &SetDstType(int32_t dst_type)
    {
        p_.dst_type = dst_type;
        return *this;
    }

    RotateDynamicQuantParamsBase Build() const
    {
        return p_;
    }

private:
    RotateDynamicQuantParamsBase p_;
};

/**
 * @brief Handler 基类，定义多 SOC 架构的通用接口
 */
class RotateDynamicQuantHandler {
public:
    virtual ~RotateDynamicQuantHandler() = default;

protected:
    virtual bool CheckNotNull(void)
    {
        OP_CHECK_NULL(params_.x, return false);
        OP_CHECK_NULL(params_.rot, return false);
        OP_CHECK_NULL(params_.yOut, return false);
        OP_CHECK_NULL(params_.scaleOut, return false);
        return true;
    }

    virtual bool CheckEmptyTensor(void)
    {
        if (params_.x->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor is empty");
            return false;
        }
        if (params_.rot->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rot tensor is empty");
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
        // 将输入tensor转换为连续格式
        params_.x_contiguous = l0op::Contiguous(params_.x, l0Executor_);
        CHECK_RET(params_.x_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        params_.rot_contiguous = l0op::Contiguous(params_.rot, l0Executor_);
        CHECK_RET(params_.rot_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        return ACLNN_SUCCESS;
    }

public:
    void Initialize(const char *interfaceName, RotateDynamicQuantParamsBase &params,
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

        // 空 Tensor 场景处理
        if (params_.yOut->IsEmpty()) {
            *workspaceSize_ = 0ULL;
            uniqueExecutor.ReleaseTo(executor_);
            return ACLNN_SUCCESS;
        }

        ret = ConvertDataContiguous();
        CHECK_RET(ret == ACLNN_SUCCESS, ret);

        // 执行算子
        auto outParams = l0op::RotateDynamicQuant(
            params_.x_contiguous, params_.rot_contiguous, params_.dst_type,
            l0Executor_);
        CHECK_RET(outParams != std::tuple(nullptr), ACLNN_ERR_INNER_NULLPTR);

        // 拷贝输出到用户输出tensor
        auto yOutResult = std::get<0>(outParams);
        auto scaleOutResult = std::get<1>(outParams);

        auto retViewY = l0op::ViewCopy(yOutResult, params_.yOut, l0Executor_);
        CHECK_RET(retViewY != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto retViewScale = l0op::ViewCopy(scaleOutResult, params_.scaleOut, l0Executor_);
        CHECK_RET(retViewScale != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize_ = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor_);
        return ACLNN_SUCCESS;
    }

protected:
    std::string interfaceName_;
    RotateDynamicQuantParamsBase params_;
    uint64_t *workspaceSize_;
    aclOpExecutor **executor_;
    aclOpExecutor *l0Executor_;
};

} // namespace rotate_dynamic_quant

#endif // OP_HOST_OP_API_ACLNN_ROTATE_DYNAMIC_QUANT_UTILS_H
