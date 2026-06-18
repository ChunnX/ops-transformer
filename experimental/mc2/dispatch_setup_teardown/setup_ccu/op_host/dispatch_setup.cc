/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <memory>
#include <hccl/hccl_res_expt.h>

#include "log.h"
#include "common.h"
#include "dispatch_setup.h"
#include "utils.h"
#include "exec_op.h"

using namespace std;
using namespace ops_dispatch_setup;

static HcclResult InitAlgResourceCtx(HcclComm comm, OpParam &param,
                                      std::unique_ptr<AlgResourceCtxSerializable> &resCtxHost)
{
    void *ctx = nullptr;
    uint64_t size = 0;

    if (HcclEngineCtxGet(comm, param.tag, param.engine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("[HcclDispatchSetup] Engine context already exists, reuse it");
        param.ctxSize = size;
        char *resCtxSequence = static_cast<char *>(ctx);
        std::vector<char> ctxData(resCtxSequence, resCtxSequence + param.ctxSize);
        resCtxHost->DeSerialize(ctxData);
    } else {
        HCCL_INFO("[HcclDispatchSetup] Creating engine context");
        HcclResult ret = AllocAlgResource(comm, param, *resCtxHost);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("failed to alloc alg resource.");
            return ret;
        }
        std::vector<char> seq = resCtxHost->Serialize();
        uint64_t ctxSize = seq.size();

        void *newCtx = nullptr;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, param.engine, ctxSize, &newCtx));
        memcpy_s(newCtx, ctxSize, seq.data(), ctxSize);
        param.ctxSize = ctxSize;
        HCCL_INFO("Execute GetAlgResCCU success.");
    }

    return HCCL_SUCCESS;
}

HcclResult HcclDispatchSetup(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, 
                                HcclComm comm, aclrtStream stream)
{
    HCCL_INFO("Start to execute HcclDispatchSetup");

    // 1.校验参数是否为空
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);

    // 2.获取算子参数信息
    OpParam param;
    
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    uint32_t perDataSize = SIZE_TABLE[dataType];
    uint64_t inputSize = sendCount * perDataSize;
    uint64_t outputSize = inputSize * param.rankSize;
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", "dispatch_setup");
    if (ret <= 0) {
        HCCL_ERROR("[HcclDispatchSetup] Failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(GetDeviceType(&param.devType));
    if (param.devType != DEVICE_TYPE_A5 && param.devType != DEVICE_TYPE_A6) {  // a6 同卡握手同样适用
        HCCL_ERROR("[HcclDispatchSetup] Not Support Device Type [%u]", param.devType);
        return HCCL_E_INTERNAL;
    }

    param.stream = stream;
    CHK_RET(HcclGetCommName(comm, param.commName));
    HCCL_INFO("[HcclDispatchSetup] commName: %s", param.commName);

    param.opMode = OpMode::OPBASE;
    param.engine = CommEngine::COMM_ENGINE_CCU;

    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.count = sendCount;
    param.dataType = dataType;
    // dispatch_setup: 本地拷贝路径不依赖 opType/algType, 保持默认值

    // 3. 创建资源
    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    CHK_RET(InitAlgResourceCtx(comm, param, resCtxHost));

    // 4.下发 CCU 任务
    CHK_RET(ExecOp(param, *resCtxHost));

    HCCL_INFO("HcclDispatchSetup executed successfully");
    return HCCL_SUCCESS;
}