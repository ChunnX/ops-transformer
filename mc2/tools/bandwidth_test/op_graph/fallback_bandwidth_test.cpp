/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "mc2_common_log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

const char *BandwidthTestInfo = "BandwidthTestFallback";

static graphStatus BandwidthTestExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_LOGD(BandwidthTestInfo, "start to fallback for BandwidthTest");
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE(BandwidthTestInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);

    const auto x = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
    OP_CHECK_IF(x == nullptr, OP_LOGE(BandwidthTestInfo, "x is null"), return ge::GRAPH_FAILED);

    const auto dst_rank_id = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
    OP_CHECK_IF(dst_rank_id == nullptr, OP_LOGE(BandwidthTestInfo, "dst_rank_id is null"), return ge::GRAPH_FAILED);

    const auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
    OP_CHECK_IF(y == nullptr, OP_LOGE(BandwidthTestInfo, "y is null"), return ge::GRAPH_FAILED);

    const auto receive_cnt = host_api_ctx->GetOutputTensor(static_cast<size_t>(1));
    OP_CHECK_IF(receive_cnt == nullptr, OP_LOGE(BandwidthTestInfo, "receive_cnt is null"), return ge::GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(BandwidthTestInfo, "attrs is null"), return ge::GRAPH_FAILED);

    const auto *group = attrs->GetStr(static_cast<size_t>(0));
    OP_CHECK_IF(group == nullptr, OP_LOGE(BandwidthTestInfo, "group is null"), return ge::GRAPH_FAILED);

    const auto *world_size = attrs->GetInt(static_cast<size_t>(1));
    OP_CHECK_IF(world_size == nullptr, OP_LOGE(BandwidthTestInfo, "world_size is null"), return ge::GRAPH_FAILED);

    const auto *max_bs = attrs->GetInt(static_cast<size_t>(2));
    OP_CHECK_IF(max_bs == nullptr, OP_LOGE(BandwidthTestInfo, "max_bs is null"), return ge::GRAPH_FAILED);

    const auto *mode = attrs->GetInt(static_cast<size_t>(3));
    OP_CHECK_IF(mode == nullptr, OP_LOGE(BandwidthTestInfo, "mode is null"), return ge::GRAPH_FAILED);

    const auto *ccl_buffer_size = attrs->GetInt(static_cast<size_t>(4));
    OP_CHECK_IF(ccl_buffer_size == nullptr, OP_LOGE(BandwidthTestInfo, "ccl_buffer_size is null"),
                                                                        return ge::GRAPH_FAILED);

    const auto *comm_alg = attrs->GetStr(static_cast<size_t>(5));
    OP_CHECK_IF(comm_alg == nullptr, OP_LOGE(BandwidthTestInfo, "comm_alg is null"), return ge::GRAPH_FAILED);

    const auto *aiv_num = attrs->GetInt(static_cast<size_t>(6));
    OP_CHECK_IF(aiv_num == nullptr, OP_LOGE(BandwidthTestInfo, "aiv_num is null"), return ge::GRAPH_FAILED);

    const auto api_ret = EXEC_OPAPI_CMD(aclnnBandwidthTest, x, dst_rank_id, group, *world_size,
    *max_bs, *mode, *ccl_buffer_size, *comm_alg, *aiv_num, y, receive_cnt);
    OP_CHECK_IF(api_ret != ge::GRAPH_SUCCESS, OP_LOGE(BandwidthTestInfo, "aclnn api error code %u", api_ret),
            return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(BandwidthTest).OpExecuteFunc(BandwidthTestExecuteFunc);

}  // namespace fallback

#ifdef __cplusplus
}
#endif