/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
**/

/*!
 * \file quant_reduce_scatter_mc2_context.h
 * \brief quant_reduce_scatter mc2 context management for HCCL communication
 */
#ifndef QUANT_REDUCE_SCATTER_MC2_CONTEXT_H
#define QUANT_REDUCE_SCATTER_MC2_CONTEXT_H

#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM HCCL_CHANNEL_SUPPORT_VERSION
#endif

#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "hccl/hccl_rank_graph.h"
#include "aclnn/aclnn_base.h"
#include "log/log.h"
#include "opdev/common_types.h"
#include "common/op_api/mc2_context.h"
#include "common/op_kernel/mc2_quant_reduce_scatter_context.h"

namespace Mc2Aclnn {

class Mc2QrsContext {
public:
    static aclnnStatus GetMc2ContextTensorForQrs(const char *group, const char *opName, uint64_t &hcclBuffSize,
                                                 aclTensor *&mc2Context, int64_t &worldSize);

private:
    /* for quant_reduce_scatter */
    static aclnnStatus GetHcclCommResourceForQrs(const HcclComm &hcclHandle, const CommEngine &engine,
                                                 const CommProtocol &protocol, Mc2QuantReduceScatterContext *mc2ContextStruct);
    static aclnnStatus CreatMc2ContextForQrs(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                             const CommEngine &engine, const CommProtocol &protocol,
                                             Mc2QuantReduceScatterContext *mc2ContextStruct, void *&ctx,
                                             uint64_t &hcclBuffSize);
    static aclnnStatus CreatMc2ContextTensorForQrs(void *ctx, aclTensor *&mc2Context);

    void *hcclLibHandle_ = nullptr;
    uint64_t hcclBuffSize_ = 0;
};

} // namespace Mc2Aclnn

#endif

#endif // QUANT_REDUCE_SCATTER_MC2_CONTEXT_H
