/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except
 * in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of
 * the License.
 */

/*!
 * \file aclnn_quant_matmul_all_reduce.cpp
 * \brief Quantized MatmulAllReduce API implementation
 */
#include "aclnn_quant_matmul_all_reduce.h"

#include "aclnnInner_matmul_all_reduce.h"
#include "matmul_all_reduce_util.h"
#include "common/utils/hccl_util.h"
#include "mc2_comm_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// External profiling functions
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// GetWorkspaceSize for QuantMatmulAllReduce operation
aclnnStatus aclnnQuantMatmulAllReduceGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const aclTensor* dequantScale,
    const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, const aclTensor* output,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    uint64_t startTime = NnopbaseMsprofSysTime();

    // Step 1: Parameter validation
    auto retParam =
        QuantMatmulAllReduceCheckParams(x1, x2, bias, dequantScale, nullptr, x3, reduceOp, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    // Step 2: Convert dequantScale data type from INT64 to UINT64
    auto dequantTensor = const_cast<aclTensor*>(dequantScale);
    if (dequantTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmulAllReduce, dequant is nullptr.");
        return ACLNN_ERR_INNER;
    }
    if (dequantTensor->GetDataType() == op::DataType::DT_INT64) {
        dequantTensor->SetDataType(op::DataType::DT_UINT64);
    }

    // Step 3: Call inner implementation
    aclnnStatus ret = InnerQuantMatmulAllReduceGetWorkspaceSize(
        x1, x2, bias, x3, dequantTensor, nullptr, group, reduceOp, commTurn, output, workspaceSize, executor);
    OP_LOGD("QuantMatmulAllReduce, end ret %d", ret);

    // Step 4: Report profiling info
    static NnopbaseDfxId quantDfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(startTime, quantDfxId);
    return ret;
}

// Execute QuantMatmulAllReduce operation
aclnnStatus aclnnQuantMatmulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    uint64_t startTime = NnopbaseMsprofSysTime();

    // Configure HCCL server type based on NPU architecture and communication mode
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
            if (commMode == Mc2Comm::COMM_MODE_AICPU) {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        }
    }

    // Execute the inner matmul all reduce operation
    aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, workspaceSize, executor, stream);
    OP_API_CHECK(ret != ACLNN_SUCCESS, {
        OP_LOGE(ACLNN_ERR_INNER, "QuantMatmulAllReduceLaunchTask fail, ret: %d.", ret);
        return ACLNN_ERR_INNER;
    });

    // Report profiling info for execution phase
    static NnopbaseDfxId quantExecDfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(startTime, quantExecDfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif