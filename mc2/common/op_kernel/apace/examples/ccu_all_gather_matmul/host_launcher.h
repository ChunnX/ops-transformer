/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file host_launcher.h
 * \brief Host 侧编排：PrepareAllGatherMatmul（第三阶段：commTurn round CCU launch + AIC 直轮询）
 *
 * commTurn 通用设计（新方案：1 次 CCU launch，kernel 内循环 commTurn 次）：
 *   - 1 组 Variable（commTurn × HCOMM_HOST_TASK_ARG_NUM 个，1 个 varHandle 管理）
 *   - 1 组 Event（commTurn + 1 个，1 个 eventHandle 管理：eventVa[0]=wake, eventVa[d+1]=done）
 *   - 1 组 kernelArg（varNum = commTurn × HCOMM_HOST_TASK_ARG_NUM，eventNum = commTurn + 1）
 *   - inputOffsets/outputOffsets 拷贝到 device GM 供 AIC 读取
 *
 * 不包含 <<<>>> kernel launch——launch 留在 .asc 主文件中。
 */

#pragma once

// 编排结果：填充后交给 .asc 中的 launch 函数
struct AllGatherMatmulContext {
    HCommCcuSchd schd;
    CcuInsHandle insHandle{0};
    // 单组 kernelArg（管理 commTurn × HCOMM_HOST_TASK_ARG_NUM 个 Variable + commTurn+1 个 Event）
    std::shared_ptr<CcuKernelArgAllGatherMesh1DMem2Mem> kernelArg;
    // mappedAddrs：commTurn 组 startVa[d][i] + commTurn+1 个 eventVa
    CcuMappedAddrs mappedAddrs;
    // commTurn 组 inputOffset/outputOffset（host 侧数组，需拷贝到 device）
    uint64_t inputOffsets[MAX_COMMTURN];
    uint64_t outputOffsets[MAX_COMMTURN];
    // device 侧 inputOffset/outputOffset 数组地址（aclrtMalloc 返回的 device 地址，launch 时转 __gm__）
    uint64_t d_inputOffsetsAddr{0};
    uint64_t d_outputOffsetsAddr{0};
    uint64_t token;
    uint64_t sliceSize;      // 每 round 每 rank 通信分片大小 = tileM*K*perDataSize
    uint64_t inputSize;      // 每 rank sendBuf 全量大小 = M*K*perDataSize
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t commTurn;       // 实际 round 数
    // goSize（所有 round 相同，基于 sliceSize 计算）
    uint64_t goSize[4];
};

// ============================================================================
// CCU 通信资源编排（新方案：1 组 Variable(commTurn*10) + 1 组 Event(commTurn+1)）
// 不做 kernel launch（<<<>>> 留在 .asc）
// ============================================================================
inline HcclResult PrepareAllGatherMatmul(void *sendBuf, void *recvBuf, uint64_t sendCount,
                                          HcclDataType dataType, HcclComm comm,
                                          uint32_t tileM, uint32_t K,
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
    ctx.inputSize = sendCount * perDataSize;  // M*K*perDataSize（每 rank sendBuf 全量）
    // 每 round 每 rank 通信分片大小 = tileM*K*perDataSize
    ctx.sliceSize = static_cast<uint64_t>(tileM) * static_cast<uint64_t>(K) * perDataSize;
    // commTurn = M / tileM（每 rank 的 M 分成多少 round）
    uint32_t M = static_cast<uint32_t>(sendCount / K);
    ctx.commTurn = M / tileM;
    if (ctx.commTurn > MAX_COMMTURN) {
        return HCCL_E_PARA;
    }

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

    // ---- 分配 1 组 Variable（commTurn × HCOMM_HOST_TASK_ARG_NUM 个，1 个 varHandle）----
    // 第 d round 的第 i 个 Variable 即 varHandle 管理的第 (d*HCOMM_HOST_TASK_ARG_NUM + i) 个资源
    uint32_t totalVarNum = ctx.commTurn * HCOMM_HOST_TASK_ARG_NUM;
    CcuVariableHandle varHandle{0};
    HCCLCHECK(HcommCcuVariableAlloc(insHandle, DIE_ID, totalVarNum, &varHandle));
    for (uint32_t d = 0; d < ctx.commTurn; d++) {
        for (uint32_t i = 0; i < HCOMM_HOST_TASK_ARG_NUM; i++) {
            HCCLCHECK(HcommCcuVariableGetAddr(varHandle, d * HCOMM_HOST_TASK_ARG_NUM + i,
                                              &ctx.mappedAddrs.startVa[d][i]));
        }
    }

    // ---- 分配 1 组 Event（commTurn + 1 个，1 个 eventHandle）----
    // eventVa[0] = wake（AIC→CCU，仅 round 0），eventVa[d+1] = round d done（CCU→AIC）
    uint32_t totalEventNum = ctx.commTurn + 1;
    CcuEventHandle eventHandle{0};
    HCCLCHECK(HcommCcuEventAlloc(insHandle, DIE_ID, totalEventNum, &eventHandle));
    for (uint32_t i = 0; i < totalEventNum; i++) {
        HCCLCHECK(HcommCcuEventGetAddr(eventHandle, i, &ctx.mappedAddrs.eventVa[i]));
    }

    // ---- 构造单组 kernelArg（channel 共用，varNum/eventNum 为总量）----
    ctx.kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>();
    ctx.kernelArg->rankSize = ctx.rankSize;
    ctx.kernelArg->rankId = ctx.rankId;
    ctx.kernelArg->channelCount = static_cast<uint32_t>(kernelChannels.size());
    ctx.kernelArg->varHandle = varHandle;
    ctx.kernelArg->varNum = totalVarNum;
    ctx.kernelArg->eventHandle = eventHandle;
    ctx.kernelArg->eventNum = totalEventNum;
    ctx.kernelArg->commTurn = ctx.commTurn;
    for (uint32_t i = 0; i < kernelChannels.size(); ++i) {
        ctx.kernelArg->channels[i] = kernelChannels[i];
    }

    // ---- token（sendBuf 全量，所有 round 共用）----
    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(sendBuf), ctx.inputSize, &ctx.token));

    // ---- goSize：基于每 round 的 sliceSize 计算（所有 round sliceSize 相同）----
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;
    auto goSize = CalGoSize(ctx.sliceSize, config);
    ctx.goSize[0] = goSize[0];
    ctx.goSize[1] = goSize[1];
    ctx.goSize[2] = goSize[2];
    ctx.goSize[3] = goSize[3];

    // ---- 构造 commTurn 组 inputOffset/outputOffset ----
    // recvBuf 布局：[rank0 tile0 | rank1 tile0 | ... | rankR tile0 | rank0 tile1 | rank1 tile1 | ...]
    // round r 的 outputOffset = r * rankSize * sliceSize + rankId * sliceSize
    // round r 的 inputOffset = r * sliceSize（sendBuf 按 tile 分段）
    for (uint32_t r = 0; r < ctx.commTurn; r++) {
        ctx.inputOffsets[r] = static_cast<uint64_t>(r) * ctx.sliceSize;
        ctx.outputOffsets[r] = static_cast<uint64_t>(r) * ctx.rankSize * ctx.sliceSize
                             + static_cast<uint64_t>(ctx.rankId) * ctx.sliceSize;
    }

    // ---- 拷贝 inputOffsets/outputOffsets 到 device GM（供 AIC kernel 读取）----
    // 注意：本函数返回 HcclResult，不能用 ACLCHECK（返回 aclError），需手动检查
    void *dInputOffsets = nullptr;
    aclError aclRet = aclrtMalloc(&dInputOffsets, ctx.commTurn * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_ONLY);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    void *dOutputOffsets = nullptr;
    aclRet = aclrtMalloc(&dOutputOffsets, ctx.commTurn * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_ONLY);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtMemcpy(dInputOffsets, ctx.commTurn * sizeof(uint64_t),
                         ctx.inputOffsets, ctx.commTurn * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtMemcpy(dOutputOffsets, ctx.commTurn * sizeof(uint64_t),
                         ctx.outputOffsets, ctx.commTurn * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    ctx.d_inputOffsetsAddr = reinterpret_cast<uint64_t>(dInputOffsets);
    ctx.d_outputOffsetsAddr = reinterpret_cast<uint64_t>(dOutputOffsets);

    ctx.schd = {1, 0, 0x01};

    // 校验 recvBuf 容量：commTurn × rankSize × sliceSize 不超过 recvBuf 分配大小
    const uint64_t totalRecvSize = static_cast<uint64_t>(ctx.rankSize) * ctx.sliceSize * ctx.commTurn;
    const uint64_t recvBufSize = sendCount * ctx.rankSize * perDataSize;
    if (totalRecvSize > recvBufSize) {
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}
