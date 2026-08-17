/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file host_launcher.h
 * \brief Host 侧编排：PrepareAllGatherMatmul
 *
 * 负责 CCU 通信资源编排（channel/ins/variable/event 分配 + kernelArg/taskArgs/mappedAddrs 填充）。
 * 不包含 <<<>>> kernel launch——launch 留在 .asc 主文件中。
 *
 * 被 ccu_all_gather_matmul_kernel.asc 的 AllGatherMatmulCustom() 调用。
 */

#pragma once

// 编排结果：填充后交给 .asc 中的 launch 函数
struct AllGatherMatmulContext {
    HCommCcuSchd schd;
    CcuInsHandle insHandle{0};
    std::shared_ptr<CcuKernelArgAllGatherMesh1DMem2Mem> kernelArg;
    CcuMappedAddrs mappedAddrs;
    uint64_t taskArgs[HCOMM_HOST_TASK_ARG_NUM];
    uint64_t inputSize;
    uint32_t rankId;
    uint32_t rankSize;
};

// ============================================================================
// CCU 通信资源编排：channel/ins/variable/event 分配 + kernelArg/taskArgs/mappedAddrs 填充
// 不做 kernel launch（<<<>>> 留在 .asc）
// ============================================================================
inline HcclResult PrepareAllGatherMatmul(void *sendBuf, void *recvBuf, uint64_t sendCount,
                                          HcclDataType dataType, HcclComm comm,
                                          AllGatherMatmulContext &ctx)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);

    if (dataType != HCCL_DATA_TYPE_FP16 && dataType != HCCL_DATA_TYPE_FP32) {
        return HCCL_E_PARA;
    }
    if (sendCount == 0) {
        return HCCL_SUCCESS;
    }

    HCCLCHECK(HcclGetRankId(comm, &ctx.rankId));
    HCCLCHECK(HcclGetRankSize(comm, &ctx.rankSize));

    uint32_t perDataSize = SIZE_TABLE[dataType];
    ctx.inputSize = sendCount * perDataSize;

    std::vector<ChannelHandle> kernelChannels;
    HCCLCHECK(AcquireCcuChannels(comm, ctx.rankId, ctx.rankSize, kernelChannels));

    CcuInsHandle insHandle{0};
    HCCLCHECK(CreateCcuInsWithEnabledDies(insHandle));
    HcclResult assignRet = HcclCommAssignCcuIns(comm, insHandle);
    if (assignRet != HCCL_SUCCESS) {
        (void)HcommCcuInsDestroy(insHandle);
        return assignRet;
    }
    ctx.insHandle = insHandle;

    CcuVariableHandle varHandle{0};
    HCCLCHECK(HcommCcuVariableAlloc(insHandle, DIE_ID, HCOMM_HOST_TASK_ARG_NUM, &varHandle));
    for (uint32_t i = 0; i < HCOMM_HOST_TASK_ARG_NUM; i++) {
        HCCLCHECK(HcommCcuVariableGetAddr(varHandle, i, &ctx.mappedAddrs.startVa[i]));
    }

    CcuEventHandle eventHandle{0};
    HCCLCHECK(HcommCcuEventAlloc(insHandle, DIE_ID, CKE_NUM, &eventHandle));
    for (uint32_t i = 0; i < CKE_NUM; i++) {
        HCCLCHECK(HcommCcuEventGetAddr(eventHandle, i, &ctx.mappedAddrs.eventVa[i]));
    }

    ctx.kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>();
    ctx.kernelArg->rankSize = ctx.rankSize;
    ctx.kernelArg->rankId = ctx.rankId;
    ctx.kernelArg->channelCount = static_cast<uint32_t>(kernelChannels.size());
    ctx.kernelArg->varHandle = varHandle;
    ctx.kernelArg->varNum = HCOMM_HOST_TASK_ARG_NUM;
    ctx.kernelArg->eventHandle = eventHandle;
    ctx.kernelArg->eventNum = CKE_NUM;
    for (uint32_t i = 0; i < kernelChannels.size(); ++i) {
        ctx.kernelArg->channels[i] = kernelChannels[i];
    }

    uint64_t token = 0;
    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(sendBuf), ctx.inputSize, &token));

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;
    auto goSize = CalGoSize(ctx.inputSize, config);

    ctx.taskArgs[0] = reinterpret_cast<uint64_t>(sendBuf);
    ctx.taskArgs[1] = reinterpret_cast<uint64_t>(recvBuf);
    ctx.taskArgs[2] = token;
    ctx.taskArgs[3] = 0;
    ctx.taskArgs[4] = ctx.inputSize * ctx.rankId;
    ctx.taskArgs[5] = ctx.inputSize;
    ctx.taskArgs[6] = goSize[0];
    ctx.taskArgs[7] = goSize[1];
    ctx.taskArgs[8] = goSize[2];
    ctx.taskArgs[9] = goSize[3];

    ctx.schd = {1, 0, 0x01};

    const uint64_t recvCount = sendCount * ctx.rankSize;
    if (recvCount > sendCount * CCU_MAX_RANK_SIZE) {
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}
