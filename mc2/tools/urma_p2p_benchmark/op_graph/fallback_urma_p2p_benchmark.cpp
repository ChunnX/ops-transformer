/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

const char *UrmaP2pBenchmarkInfo = "UrmaP2pBenchmarkFallback";

static graphStatus UrmaP2pBenchmarkExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_LOGD(UrmaP2pBenchmarkInfo, "start to fallback for UrmaP2pBenchmark");
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "host_api_ctx is null"), return ge::GRAPH_FAILED);

    // 输入: context(index 0), x(index 1), dst_rank_id(index 2)
    const auto context = host_api_ctx->GetInputTensor(static_cast<size_t>(0));
    OP_CHECK_IF(context == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "context is null"), return ge::GRAPH_FAILED);

    const auto x = host_api_ctx->GetInputTensor(static_cast<size_t>(1));
    OP_CHECK_IF(x == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "x is null"), return ge::GRAPH_FAILED);

    const auto dst_rank_id = host_api_ctx->GetInputTensor(static_cast<size_t>(2));
    OP_CHECK_IF(dst_rank_id == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "dst_rank_id is null"), return ge::GRAPH_FAILED);

    // 输出: y(index 0), receive_cnt(index 1), performance_info(index 2)
    const auto y = host_api_ctx->GetOutputTensor(static_cast<size_t>(0));
    OP_CHECK_IF(y == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "y is null"), return ge::GRAPH_FAILED);

    const auto receive_cnt = host_api_ctx->GetOutputTensor(static_cast<size_t>(1));
    OP_CHECK_IF(receive_cnt == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "receive_cnt is null"), return ge::GRAPH_FAILED);

    const auto performance_info = host_api_ctx->GetOutputTensor(static_cast<size_t>(2));
    OP_CHECK_IF(performance_info == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "performance_info is null"), return ge::GRAPH_FAILED);

    const auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "attrs is null"), return ge::GRAPH_FAILED);

    // 属性索引: 0=group, 1=world_size, 2=max_bs, 3=mode, 4=ccl_buffer_size, 5=comm_alg, 6=dst_rank_id, 7=aiv_num
    const auto *group = attrs->GetStr(static_cast<size_t>(0));
    OP_CHECK_IF(group == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "group is null"), return ge::GRAPH_FAILED);

    const auto *world_size = attrs->GetInt(static_cast<size_t>(1));
    OP_CHECK_IF(world_size == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "world_size is null"), return ge::GRAPH_FAILED);

    const auto *max_bs = attrs->GetInt(static_cast<size_t>(2));
    OP_CHECK_IF(max_bs == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "max_bs is null"), return ge::GRAPH_FAILED);

    const auto *mode = attrs->GetInt(static_cast<size_t>(3));
    OP_CHECK_IF(mode == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "mode is null"), return ge::GRAPH_FAILED);

    const auto *ccl_buffer_size = attrs->GetInt(static_cast<size_t>(4));
    OP_CHECK_IF(ccl_buffer_size == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "ccl_buffer_size is null"),
                                                                        return ge::GRAPH_FAILED);

    const auto *comm_alg = attrs->GetStr(static_cast<size_t>(5));
    OP_CHECK_IF(comm_alg == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "comm_alg is null"), return ge::GRAPH_FAILED);

    const auto *dst_rank_id_attr = attrs->GetInt(static_cast<size_t>(6));
    OP_CHECK_IF(dst_rank_id_attr == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "dst_rank_id is null"),
                                                                        return ge::GRAPH_FAILED);

    const auto *aiv_num = attrs->GetInt(static_cast<size_t>(7));
    OP_CHECK_IF(aiv_num == nullptr, OP_LOGE(UrmaP2pBenchmarkInfo, "aiv_num is null"), return ge::GRAPH_FAILED);

    // 调用aclnnUrmaP2pBenchmark接口（双模式: SQE/Hcomm）
    const auto api_ret = EXEC_OPAPI_CMD(aclnnUrmaP2pBenchmark, x, group, *world_size,
    *max_bs, *mode, *ccl_buffer_size, comm_alg, *dst_rank_id_attr, *aiv_num, y, receive_cnt, performance_info);
    OP_CHECK_IF(api_ret != ge::GRAPH_SUCCESS, OP_LOGE(UrmaP2pBenchmarkInfo, "aclnn api error code %u", api_ret),
            return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(UrmaP2pBenchmark).OpExecuteFunc(UrmaP2pBenchmarkExecuteFunc);

}  // namespace fallback

#ifdef __cplusplus
}
#endif
