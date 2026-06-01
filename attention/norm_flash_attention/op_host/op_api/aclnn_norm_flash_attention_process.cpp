/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include "aclnn_norm_flash_attention_utils.h"
#include "aclnn_norm_flash_attention_base_handler.h"
#include "aclnn_norm_flash_attention.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "platform/soc_spec.h"
#include "norm_flash_attention.h"

using namespace op;
using namespace norm_flash_attention;

class NormFlashAttentionHandlerFactory {
private:
    std::unordered_map<NpuArch, std::unique_ptr<NormFlashAttentionHandler>> handlers_;

public:
    void registerHandler(NpuArch npuArch, std::unique_ptr<NormFlashAttentionHandler> handler)
    {
        handlers_[npuArch] = std::move(handler);
    }

    NormFlashAttentionHandler *getHandler(NpuArch npuArch)
    {
        auto it = handlers_.find(npuArch);
        return it != handlers_.end() ? it->second.get() : nullptr;
    }
};

static aclnnStatus aclnnNormFlashAttentionGetWorkspaceSizeCommon(const char* interfaceName,
    NormFlashAttentionParamsBase &params, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    NormFlashAttentionHandlerFactory factory;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();

    factory.registerHandler(NpuArch::DAV_2201,
        std::make_unique<norm_flash_attention_base::NormFlashAttentionBaseHandler>());

    if (auto *handler = factory.getHandler(npuArch)) {
        handler->Initialize(interfaceName, params, workspaceSize, executor);
        return handler->Process();
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s: the soc version is not support", interfaceName);
    }

    return ACLNN_ERR_PARAM_INVALID;
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNormFlashAttention 的第一段接口，根据具体的计算流程，计算 workspace 大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] query: 输入 tensor，shape 为 [B, S, H, D]，数据类型支持：DT_FLOAT16, DT_FLOAT, DT_BF16
 * @param [in] key: 输入 tensor，shape 为 [B, S, H, D]，数据类型支持：DT_FLOAT16, DT_FLOAT, DT_BF16
 * @param [in] value: 输入 tensor，shape 为 [B, S, H, D]，数据类型支持：DT_FLOAT16, DT_FLOAT, DT_BF16
 * @param [out] attentionOut: 输出 tensor，shape 为 [B, S, H, D]，数据类型：DT_FLOAT16, DT_FLOAT, DT_BF16
 * @param [out] workspaceSize: 返回需要在 npu device 侧申请的 workspace 大小
 * @param [out] executor: 返回 op 执行器，包含了算子计算流程
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnNormFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnNormFlashAttention,
                    DFX_IN(query, key, value),
                    DFX_OUT(attentionOut));

    NormFlashAttentionParamsBase params =
        NormFlashAttentionParamsBuilder::Create(query, key, value, attentionOut)
        .Build();

    return aclnnNormFlashAttentionGetWorkspaceSizeCommon(__FUNCTION__, params, workspaceSize, executor);
}

/**
 * @brief aclnnNormFlashAttention 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 *
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址
 * @param [in] workspaceSize: 在 npu device 侧申请的 workspace 大小，由第一段接口获取
 * @param [in] executor: op 执行器，包含了算子计算流程
 * @param [in] stream: acl stream 流
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnNormFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNormFlashAttention);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in NormFlashAttention launch aicore");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
