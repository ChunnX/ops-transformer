/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include "aclnn_rotate_dynamic_quant_utils.h"
#include "aclnn_rotate_dynamic_quant_base_handler.h"
#include "aclnn_rotate_dynamic_quant.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "rotate_dynamic_quant.h"

using namespace op;
using namespace rotate_dynamic_quant;

/**
 * @brief HandlerFactory 类，用于根据 SOC 版本创建对应的 Handler
 */
class RotateDynamicQuantHandlerFactory {
private:
    std::unordered_map<NpuArch, std::unique_ptr<RotateDynamicQuantHandler>> handlers_;

public:
    void registerHandler(NpuArch npuArch, std::unique_ptr<RotateDynamicQuantHandler> handler)
    {
        handlers_[npuArch] = std::move(handler);
    }

    RotateDynamicQuantHandler *getHandler(NpuArch npuArch)
    {
        auto it = handlers_.find(npuArch);
        return it != handlers_.end() ? it->second.get() : nullptr;
    }
};

/**
 * @brief 公共处理函数，根据 SOC 版本路由到对应的 Handler
 */
static aclnnStatus aclnnRotateDynamicQuantGetWorkspaceSizeCommon(const char* interfaceName,
    RotateDynamicQuantParamsBase &params, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    RotateDynamicQuantHandlerFactory factory;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();

    // 注册 DAV_2201 版本的 Handler
    factory.registerHandler(NpuArch::DAV_2201,
        std::make_unique<rotate_dynamic_quant_base::RotateDynamicQuantBaseHandler>());

    // 获取当前 SOC 版本对应的 Handler
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
 * @brief aclnnRotateDynamicQuant 的第一段接口，根据具体的计算流程，计算 workspace 大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 输入 tensor，shape 为 [M, N]，数据类型支持：DT_FLOAT16, DT_BFLOAT16
 * @param [in] rot: 旋转矩阵 tensor，shape 为 [N, N]，数据类型与 x 保持一致
 * @param [in] dstType: 输出数据类型属性，0 表示 DT_INT8，1 表示 DT_INT32
 * @param [out] yOut: 输出 tensor，shape 为 [M, N]，数据类型：DT_INT8 或 DT_INT32
 * @param [out] scaleOut: 缩放系数 tensor，shape 为 [M]，数据类型：DT_FLOAT
 * @param [out] workspaceSize: 返回需要在 npu device 侧申请的 workspace 大小
 * @param [out] executor: 返回 op 执行器，包含了算子计算流程
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnRotateDynamicQuantGetWorkspaceSize(
    const aclTensor *x, const aclTensor *rot,
    int64_t dstType,
    aclTensor *yOut, aclTensor *scaleOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnRotateDynamicQuant,
                    DFX_IN(x, rot, dstType),
                    DFX_OUT(yOut, scaleOut));

    // 使用 Builder 模式构建参数结构体
    RotateDynamicQuantParamsBase params =
        RotateDynamicQuantParamsBuilder::Create(x, rot, yOut, scaleOut)
        .SetDstType(dstType)
        .Build();

    // 调用公共接口
    return aclnnRotateDynamicQuantGetWorkspaceSizeCommon(__FUNCTION__, params, workspaceSize, executor);
}

/**
 * @brief aclnnRotateDynamicQuant 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 *
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址
 * @param [in] workspaceSize: 在 npu device 侧申请的 workspace 大小，由第一段接口获取
 * @param [in] executor: op 执行器，包含了算子计算流程
 * @param [in] stream: acl stream 流
 * @return aclnnStatus: 返回状态码
 */
aclnnStatus aclnnRotateDynamicQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRotateDynamicQuant);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in RotateDynamicQuant launch aicore");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
