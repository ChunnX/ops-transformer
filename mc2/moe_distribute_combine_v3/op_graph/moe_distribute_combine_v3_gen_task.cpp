/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file moe_distribute_combine_v3_gen_task.cpp
 * \brief MoeDistributeCombineV3 的 gen_task：插入 kHcom hidden input，
 *        由 GE 运行期调 HCCL 插件（HcomCreateComResourceMC2）填充 HCCL 托管的 context 地址，
 *        kernel 通过 GetHcclContext 读取，不再依赖 GE 图资产 Const。
 */
#include <vector>
#include <set>
#include <string>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"

namespace ops {
// V3 仅 A5 MTE 场景（fusion pass 只转换非 ccu 节点），与 V2 MTE 分支一致挂 kfc_stream
ge::Status MoeDistributeCombineV3CalcParamFunc(gert::ExeResGenerationContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Do V3 general calc param.");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status MoeDistributeCombineV3GenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (tasks.empty()) {
        OPS_LOG_E(nodeName, "Failed to get task for mc2 v3 node.");
        return ge::GRAPH_FAILED;
    }
    // 与 V2 A5 MTE 一致：D&C 仅插入 kHcom hidden input，不拉 aicpu task。
    // group 数固定为 2（group_ep + group_tp，由 fusion pass 保留在 V3 节点 attr 上），
    // 与 V2 的 GROUP_CNT_OF_MOE_DISTRIBUTE 一致。
    constexpr int32_t groupCnt = 2;
    if (Mc2MoeGenTaskOpsUtils::McMoeInsertHiddenInputForAicore(context, groupCnt, tasks) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Insert hidden input for v3 node failed.");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(nodeName, "Insert kHcom hidden input for v3 node successfully.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeCombineV3)
    .CalcOpParam(MoeDistributeCombineV3CalcParamFunc)
    .GenerateTask(MoeDistributeCombineV3GenTaskFunc);
} // namespace ops
