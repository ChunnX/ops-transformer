/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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

/*!
 * \file fallback_moe_init_routing_quant.cpp
 * \brief
 */
#include "op_graph/op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t xInputIndex = 0;
constexpr size_t rowIdxInputIndex = 1;
constexpr size_t expertIdxInputIndex = 2;

constexpr size_t expandedXOutputIndex = 0;
constexpr size_t expandedRowIdxOutputIndex = 1;
constexpr size_t expandedExpertIdxOutputIndex = 2;

static graphStatus MoeInitRoutingQuantExecuteFunc(OpExecuteContext* host_api_ctx) {
  OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

  auto x_ge = host_api_ctx->GetInputTensor(xInputIndex);
  OP_CHECK_IF(x_ge == nullptr, OP_LOGE("aclnnfallback", "x_ge is null"), return GRAPH_FAILED);
  auto row_idx_ge = host_api_ctx->GetInputTensor(rowIdxInputIndex);
  OP_CHECK_IF(row_idx_ge == nullptr, OP_LOGE("aclnnfallback", "row_idx_ge is null"), return GRAPH_FAILED);
  auto expert_idx_ge = host_api_ctx->GetInputTensor(expertIdxInputIndex);
  OP_CHECK_IF(expert_idx_ge == nullptr, OP_LOGE("aclnnfallback", "expert_idx_ge is null"), return GRAPH_FAILED);

  auto expanded_x_ge = host_api_ctx->GetOutputTensor(expandedXOutputIndex);
  OP_CHECK_IF(expanded_x_ge == nullptr, OP_LOGE("aclnnfallback", "expanded_x_ge is null"), return GRAPH_FAILED);
  auto expanded_row_idx_ge = host_api_ctx->GetOutputTensor(expandedRowIdxOutputIndex);
  OP_CHECK_IF(expanded_row_idx_ge == nullptr, OP_LOGE("aclnnfallback", "expanded_row_idx_ge is null"),
           return GRAPH_FAILED);
  auto expanded_expert_idx_ge = host_api_ctx->GetOutputTensor(expandedExpertIdxOutputIndex);
  OP_CHECK_IF(expanded_expert_idx_ge == nullptr, OP_LOGE("aclnnfallback", "expanded_expert_idx_ge is null"),
           return GRAPH_FAILED);

  auto attrs = host_api_ctx->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
  const int64_t* active_num = attrs->GetAttrPointer<int64_t>(0);
  const float* scale = attrs->GetAttrPointer<float>(1);
  const float* offset = attrs->GetAttrPointer<float>(2);
  // execute opapi
  auto api_ret = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnMoeInitRoutingQuant, x_ge, row_idx_ge, expert_idx_ge,
                                             active_num, scale, offset, expanded_x_ge, expanded_row_idx_ge,
                                             expanded_expert_idx_ge);
  OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "api_ret faild:%u", api_ret), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPL_OP(MoeInitRoutingQuant).OpExecuteFunc(MoeInitRoutingQuantExecuteFunc);

}  // namespace fallback

#ifdef __cplusplus
}
#endif
