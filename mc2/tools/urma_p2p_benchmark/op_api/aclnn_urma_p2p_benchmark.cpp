/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_urma_p2p_benchmark.h"
#include <algorithm>
#include <cstring>
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnnInner_urma_p2p_benchmark.h"

#ifdef BUILD_OPEN_PROJECT
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
// 如果没有hcomm_version.h，定义HCOMM_VERSION_NUM为0（不支持Mc2Context）
#define HCOMM_VERSION_NUM 0
#endif
// HCCL_CHANNEL_SUPPORT_VERSION由mc2_context.h内部定义，不在此处重复定义
#if HCOMM_VERSION_NUM >= 89999700
#include "common/op_api/mc2_context.h"
#include "p2p_hcomm_context.h"
#endif
#endif

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 校验输入指针非空
static bool CheckNotNull(const aclTensor* x, char* group,
                         aclTensor* y, aclTensor* receiveCnt, aclTensor* performanceInfo)
{
    OP_LOGD("aclnn_urma_p2p_benchmark CheckNotNull start");
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(receiveCnt, return false);
    OP_CHECK_NULL(performanceInfo, return false);
    OP_LOGD("aclnn_urma_p2p_benchmark CheckNotNull success");

    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group name is Empty");
        return false;
    }
    return true;
}

// 校验参数范围合法性
static aclnnStatus CheckParams(const aclTensor* x, char* group,
                               int64_t worldSize, int64_t maxBs, int64_t mode,
                               int64_t cclBufferSize, int64_t dstRankIdAttr,
                               aclTensor* y, aclTensor* receiveCnt, aclTensor* performanceInfo)
{
    OP_LOGD("aclnn_urma_p2p_benchmark CheckParams start");
    CHECK_RET(CheckNotNull(x, group, y, receiveCnt, performanceInfo), ACLNN_ERR_PARAM_NULLPTR);

    if (worldSize <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "worldSize must be positive, got %ld", worldSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (maxBs <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maxBs must be positive, got %ld", maxBs);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (cclBufferSize <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cclBufferSize must be positive, got %ld", cclBufferSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // mode范围: 0=overhead+latency, 1=bandwidth
    if (mode < 0 || mode > 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode must be in [0, 1], got %ld", mode);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // dstRankIdAttr范围: -1表示全部rank, 0~worldSize-1表示指定rank
    if (dstRankIdAttr < -1 || dstRankIdAttr >= worldSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dstRankIdAttr must be in [-1, worldSize-1], got %ld, worldSize=%ld",
                 dstRankIdAttr, worldSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group name exceeds max length %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    OP_LOGD("aclnn_urma_p2p_benchmark CheckParams success");
    return ACLNN_SUCCESS;
}

// 第一段接口：计算workspace大小
// 双模式设计:
//   - SQE级(commAlg=="sqe"): Mc2Context用默认UB_MEM协议创建，但kernel使用HcclCombinOpParam（框架自动管理）
//   - Hcomm级(commAlg=="hcomm"): Mc2Context用COMM_PROTOCOL_UBC_CTP协议创建，kernel使用Mc2MoeContext
aclnnStatus aclnnUrmaP2pBenchmarkGetWorkspaceSize(
    const aclTensor* x,
    char* group, int64_t worldSize, int64_t maxBs, int64_t mode,
    int64_t cclBufferSize, char* commAlg, int64_t dstRankIdAttr, int64_t aivNum,
    aclTensor* y, aclTensor* receiveCnt, aclTensor* performanceInfo,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnnUrmaP2pBenchmarkGetWorkspaceSize start");
    auto ret_param = CheckParams(x, group, worldSize, maxBs, mode, cclBufferSize, dstRankIdAttr, y, receiveCnt, performanceInfo);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus aclnnRet = ACLNN_ERR_INNER;

#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    aclTensor *mc2Context = nullptr;
    uint64_t hcclBuffSize = 0;
    // 关键: SQE和Hcomm必须使用不同的opName(上下文标签)
    // 否则SQE先运行后缓存8KB的Mc2Context(不含hcommHandle_)，
    // Hcomm复用时hcommHandle_读垃圾值导致WriteNbi/Drain死锁
    const char *opName = strcmp(commAlg, "hcomm") == 0 ? "urma_p2p_benchmark_hcomm" : "urma_p2p_benchmark";
    if (strcmp(commAlg, "hcomm") == 0) {
        // Hcomm模式: 使用本地5参数版P2pHcommContext(含hcommHandle_)
        // 创建COMM_PROTOCOL_UBC_CTP协议的通信上下文
        aclnnRet = P2pHcommAclnn::P2pHcommContextHelper::GetP2pHcommContextTensor(
            group, opName, hcclBuffSize, mc2Context, CommProtocol::COMM_PROTOCOL_UBC_CTP);
    } else {
        // SQE模式: 使用common的4参数版Mc2Context(不含hcommHandle_)
        aclnnRet = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(
            group, opName, hcclBuffSize, mc2Context);
    }
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    aclnnRet = aclnnInnerUrmaP2pBenchmarkGetWorkspaceSize(
        mc2Context, x, group, worldSize, maxBs, mode, cclBufferSize, commAlg, dstRankIdAttr, aivNum,
        y, receiveCnt, performanceInfo, workspaceSize, executor);
#endif

    OP_LOGD("aclnnUrmaP2pBenchmarkGetWorkspaceSize end");
    return aclnnRet;
}

// 第二段接口：执行计算
aclnnStatus aclnnUrmaP2pBenchmark(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream)
{
    OP_LOGD("aclnnUrmaP2pBenchmark start");

    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }

    aclnnStatus ret = aclnnInnerUrmaP2pBenchmark(workspace, workspaceSize, executor, stream);
    OP_LOGD("aclnnUrmaP2pBenchmark end");
    return ret;
}

#ifdef __cplusplus
}
#endif
