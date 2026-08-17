/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"

#include "chunk_gated_delta_rule_compute_wy.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkGatedDeltaRuleComputeWy);

const std::array<const aclTensor *, 5> ChunkGatedDeltaRuleComputeWy(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    int64_t chunkSize,
    const aclTensor *qKernelOut,
    const aclTensor *kKernelOut,
    const aclTensor *wKernelOut,
    const aclTensor *uKernelOut,
    const aclTensor *gKernelOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleComputeWy, q, k, v, g, beta, chunkSize, qKernelOut, kKernelOut, wKernelOut, uKernelOut,
           gKernelOut);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleComputeWy, OP_INPUT(q, k, v, g, beta), OP_OUTPUT(qKernelOut, kKernelOut, wKernelOut, uKernelOut, gKernelOut),
        OP_ATTR(chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {qKernelOut, kKernelOut, wKernelOut, uKernelOut, gKernelOut};
}

} // namespace l0op
