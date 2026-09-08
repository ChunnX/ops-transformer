/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "fallback/fallback.h"
#include "fallback/fallback_comm.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t SORTED_EXPERTS_INDEXQ = 0;

static graphStatus MoeComputeHostExecuteFunc(OpExecuteContext *host_api_ctx)
{
    OP_LOGD("aclnnFallback", "MoeComputeExpertTokens fallback begin");

    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto sortedExperts = host_api_ctx->GetInputTensor(SORTED_EXPERTS_INDEXQ);
    OP_CHECK_IF(sortedExperts == nullptr, OP_LOGE("aclnnfallback", "sortedExperts is null"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
    const int32_t *num_experts = attrs->GetAttrPointer<int32_t>(0);
    OP_CHECK_IF(*num_experts < 1, OP_LOGE("aclnnfallback", "num_experts can not be smaller than 1"),
                return GRAPH_FAILED);

    auto output = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(output == nullptr, OP_LOGE("aclnnfallback", "output is null"), return GRAPH_FAILED);

    // execute opapi
    auto api_ret = EXEC_OPAPI_CMD(aclnnMoeComputeExpertTokens, sortedExperts, *num_experts, output);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "api_ret faild:%u", api_ret), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MoeComputeExpertTokens).OpExecuteFunc(MoeComputeHostExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
