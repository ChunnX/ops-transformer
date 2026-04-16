/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "mc2_log.h"

namespace fallback {
using namespace ge;
using namespace gert;

const char *MoeDistributeCombineInfo = "MoeDistributeCombineFallback";

static graphStatus MoeDistributeCombineExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OP_LOGD(MoeDistributeCombineInfo, "start to fallback for moeDistributeCombine");
  OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE(MoeDistributeCombineInfo,"host_api_ctx is null"), return ge::GRAPH_FAILED);
  const auto expand_x = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
  OP_CHECK_IF(expand_x == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expand_x is null"), return ge::GRAPH_FAILED);

  const auto expert_ids = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
  OP_CHECK_IF(expert_ids == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expert_ids is null"), return ge::GRAPH_FAILED);

  const auto expand_idx = host_api_ctx->GetInputTensor(static_cast<size_t>(2));
  OP_CHECK_IF(expand_idx == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expand_idx is null"), return ge::GRAPH_FAILED);

  const auto ep_send_counts = host_api_ctx->GetInputTensor(static_cast<size_t>(3));
  OP_CHECK_IF(ep_send_counts == nullptr, OP_LOGE(MoeDistributeCombineInfo,"ep_send_counts is null"), return ge::GRAPH_FAILED);

  const auto expert_scales = host_api_ctx->GetInputTensor(static_cast<size_t>(4));
  OP_CHECK_IF(expert_scales == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expert_scales is null"), return ge::GRAPH_FAILED);

  const auto tp_send_counts = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(5));
  OP_CHECK_IF(tp_send_counts == nullptr, OP_LOGE(MoeDistributeCombineInfo,"tp_send_counts is null"), return ge::GRAPH_FAILED);

  const auto x_active_mask = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(6));

  const auto activation_scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(7));

  const auto weight_scale = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(8));

  const auto group_list = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(9));

  const auto expand_scales = host_api_ctx->GetOptionalInputTensor(static_cast<size_t>(10));
  
  const auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
  OP_CHECK_IF(y == nullptr, OP_LOGE(MoeDistributeCombineInfo,"y is null"), return ge::GRAPH_FAILED);

  const auto attrs = host_api_ctx->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(MoeDistributeCombineInfo,"attrs is null"), return ge::GRAPH_FAILED);

  const auto *group_ep = attrs->GetStr(static_cast<size_t>(0));
  OP_CHECK_IF(group_ep == nullptr, OP_LOGE(MoeDistributeCombineInfo,"group_ep is null"), return ge::GRAPH_FAILED);

  const auto *ep_word_size = attrs->GetInt(static_cast<size_t>(1));
  OP_CHECK_IF(ep_word_size == nullptr, OP_LOGE(MoeDistributeCombineInfo,"ep_word_size is null"), return ge::GRAPH_FAILED);

  const auto *ep_rank_id = attrs->GetInt(static_cast<size_t>(2));
  OP_CHECK_IF(ep_rank_id == nullptr, OP_LOGE(MoeDistributeCombineInfo,"ep_rank_id is null"), return ge::GRAPH_FAILED);

  const auto *moe_expert_num = attrs->GetInt(static_cast<size_t>(3));
  OP_CHECK_IF(moe_expert_num == nullptr, OP_LOGE(MoeDistributeCombineInfo,"moe_expert_num is null"), return ge::GRAPH_FAILED);

  const auto *group_tp = attrs->GetStr(static_cast<size_t>(4));
  OP_CHECK_IF(group_tp == nullptr, OP_LOGE(MoeDistributeCombineInfo,"group_tp is null"), return ge::GRAPH_FAILED);

  const auto *tp_word_size = attrs->GetInt(static_cast<size_t>(5));
  OP_CHECK_IF(tp_word_size == nullptr, OP_LOGE(MoeDistributeCombineInfo,"tp_word_size is null"), return ge::GRAPH_FAILED);

  const auto *tp_rank_id = attrs->GetInt(static_cast<size_t>(6));
  OP_CHECK_IF(tp_rank_id == nullptr, OP_LOGE(MoeDistributeCombineInfo,"tp_rank_id is null"), return ge::GRAPH_FAILED);

  const auto *expert_shard_type = attrs->GetInt(static_cast<size_t>(7));
  OP_CHECK_IF(expert_shard_type == nullptr, OP_LOGE(MoeDistributeCombineInfo,"expert_shard_type is null"), return ge::GRAPH_FAILED);

  const auto *shared_expert_num = attrs->GetInt(static_cast<size_t>(8));
  OP_CHECK_IF(shared_expert_num == nullptr, OP_LOGE(MoeDistributeCombineInfo,"shared_expert_num is null"),
    return ge::GRAPH_FAILED);

  const auto *shared_expert_rank_num = attrs->GetInt(static_cast<size_t>(9));
  OP_CHECK_IF(shared_expert_rank_num == nullptr, OP_LOGE(MoeDistributeCombineInfo,"shared_expert_rank_num is null"),
    return ge::GRAPH_FAILED);

  const int64_t *global_bs_ptr = attrs->GetInt(static_cast<size_t>(10));
  OP_CHECK_IF(global_bs_ptr == nullptr, OP_LOGE(MoeDistributeCombineInfo,"global_bs_ptr is null"), return ge::GRAPH_FAILED);

  const int64_t *out_dtype_ptr = attrs->GetInt(static_cast<size_t>(11));
  OP_CHECK_IF(out_dtype_ptr == nullptr, OP_LOGE(MoeDistributeCombineInfo,"out_dtype is null"), return ge::GRAPH_FAILED);

  const int64_t *comm_quant_mode_ptr = attrs->GetInt(static_cast<size_t>(12));
  OP_CHECK_IF(comm_quant_mode_ptr == nullptr, OP_LOGE(MoeDistributeCombineInfo,"comm_quant_mode is null"), return ge::GRAPH_FAILED);

  const int64_t *group_list_type_ptr = attrs->GetInt(static_cast<size_t>(13));
  OP_CHECK_IF(group_list_type_ptr == nullptr, OP_LOGE(MoeDistributeCombineInfo,"group_list_type is null"), return ge::GRAPH_FAILED);

  const auto api_ret = EXEC_OPAPI_CMD(aclnnMoeDistributeCombine, expand_x, expert_ids, expand_idx, ep_send_counts,
    expert_scales, tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales, 
    group_ep, *ep_word_size, *ep_rank_id, *moe_expert_num, group_tp, *tp_word_size, *tp_rank_id,
    *expert_shard_type, *shared_expert_num, *shared_expert_rank_num, *global_bs_ptr, *out_dtype_ptr, 
    *comm_quant_mode_ptr, *group_list_type_ptr, y);
  OP_CHECK_IF(api_ret != ge::GRAPH_SUCCESS, OP_LOGE(MoeDistributeCombineInfo,"aclnn api error code %u", api_ret), return api_ret);
  return GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeCombine).OpExecuteFunc(MoeDistributeCombineExecuteFunc);

}  // namespace fallback