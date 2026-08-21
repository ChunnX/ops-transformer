/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file host_launcher.h
 * \brief Host 侧编排：PrepareAllGatherMatmul（第四阶段：MXFP8，两阶段通信）
 *
 * 第四阶段 commTurn+1 组 Variable + commTurn+2 个 Event：
 *   - 组 0 = scale round（全量 scale AllGather）
 *   - 组 1..commTurn = data round（每轮 tileM 行 FP8 data AllGather）
 *   - eventVa[0] = wake, eventVa[1] = scale done, eventVa[d+2] = data round d done
 */

#pragma once

// 编排结果：填充后交给 .asc 中的 launch 函数
struct AllGatherMatmulContext {
    HCommCcuSchd schd;
    CcuInsHandle insHandle{0};
    // 单组 kernelArg（管理 (commTurn+1) × HCOMM_HOST_TASK_ARG_NUM 个 Variable + commTurn+2 个 Event）
    std::shared_ptr<CcuKernelArgAllGatherMesh1DMem2Mem> kernelArg;
    // mappedAddrs：(commTurn+1) 组 startVa[d][i] + commTurn+2 个 eventVa
    CcuMappedAddrs mappedAddrs;
    // commTurn 组 data inputOffset/outputOffset（host 侧数组，需拷贝到 device）
    uint64_t inputOffsets[MAX_COMMTURN];
    uint64_t outputOffsets[MAX_COMMTURN];
    // 1 组 scale inputOffset/outputOffset（scale 只 1 轮）
    uint64_t scaleInputOffsets[1];
    uint64_t scaleOutputOffsets[1];
    // device 侧地址
    uint64_t d_inputOffsetsAddr{0};
    uint64_t d_outputOffsetsAddr{0};
    uint64_t d_scaleInputOffsetsAddr{0};
    uint64_t d_scaleOutputOffsetsAddr{0};
    // token
    uint64_t token;          // data token（sendBuf）
    uint64_t scaleToken;     // scale token（sendScaleBuf）
    // 通信参数
    uint64_t dataSliceSize;  // 每 round 每 rank data 通信字节 = tileM*K*1(FP8)
    uint64_t scaleSliceSize; // 全量 scale 通信字节 = M*scaleBytesPerMRow
    uint64_t inputSize;      // 每 rank sendBuf 全量 = M*K*1
    uint64_t scaleInputSize; // 每 rank sendScaleBuf 全量 = M*scaleBytesPerMRow
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t commTurn;       // data round 数（不含 scale round）
    // goSize（data 和 scale 各一组）
    uint64_t goSize[4];
    uint64_t scaleGoSize[4];
};

// ============================================================================
// CCU 通信资源编排（第四阶段：MXFP8，两阶段通信）
// 不做 kernel launch（<<<>>> 留在 .asc）
// ============================================================================
inline HcclResult PrepareAllGatherMatmul(void *sendBuf, void *recvBuf,
                                           void *sendScaleBuf, void *recvScaleBuf,
                                           uint64_t sendCount, uint64_t scaleCount,
                                           HcclDataType dataType, HcclComm comm,
                                           uint32_t tileM, uint32_t K,
                                           AllGatherMatmulContext &ctx)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(sendScaleBuf);
    CHK_PTR_NULL(recvScaleBuf);

    // 第四阶段：dataType 必须是 FP8（HCCL_DATA_TYPE_INT8 复用，1 byte per element）
    if (dataType != HCCL_DATA_TYPE_INT8) {
        return HCCL_E_PARA;
    }
    if (sendCount == 0) {
        return HCCL_SUCCESS;
    }

    HCCLCHECK(HcclGetRankId(comm, &ctx.rankId));
    HCCLCHECK(HcclGetRankSize(comm, &ctx.rankSize));

    constexpr uint64_t MXFP_DIVISOR = 64;
    constexpr uint64_t MXFP_MULTI = 2;
    uint32_t perDataSize = SIZE_TABLE[dataType];  // FP8 = 1 byte (HCCL_DATA_TYPE_INT8)
    ctx.inputSize = sendCount * perDataSize;  // M*K*1（每 rank sendBuf 全量）

    // MXFP8 scale 参数
    uint32_t M = static_cast<uint32_t>(sendCount / K);
    uint64_t scaleKLen = (K + MXFP_DIVISOR - 1) / MXFP_DIVISOR * MXFP_MULTI;
    uint64_t scaleBytesPerMRow = scaleKLen * 1;  // E8M0 = 1 byte
    ctx.scaleInputSize = scaleCount;  // M * scaleBytesPerMRow

    // data 通信参数：每 round 每 rank = tileM*K*1
    ctx.dataSliceSize = static_cast<uint64_t>(tileM) * static_cast<uint64_t>(K) * perDataSize;
    // scale 通信参数：全量 = M * scaleBytesPerMRow
    ctx.scaleSliceSize = static_cast<uint64_t>(M) * scaleBytesPerMRow;

    // commTurn = M / tileM（data 分多少 round）
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

    // ---- 分配 Variable：(commTurn+1) × HCOMM_HOST_TASK_ARG_NUM 个 ----
    // 组 0 = scale, 组 1..commTurn = data
    uint32_t totalVarNum = (ctx.commTurn + 1) * HCOMM_HOST_TASK_ARG_NUM;
    CcuVariableHandle varHandle{0};
    HCCLCHECK(HcommCcuVariableAlloc(insHandle, DIE_ID, totalVarNum, &varHandle));
    for (uint32_t d = 0; d < ctx.commTurn + 1; d++) {
        for (uint32_t i = 0; i < HCOMM_HOST_TASK_ARG_NUM; i++) {
            HCCLCHECK(HcommCcuVariableGetAddr(varHandle, d * HCOMM_HOST_TASK_ARG_NUM + i,
                                               &ctx.mappedAddrs.startVa[d][i]));
        }
    }

    // ---- 分配 Event：commTurn + 2 个 ----
    // eventVa[0]=wake, eventVa[1]=scale done, eventVa[d+2]=data round d done
    uint32_t totalEventNum = ctx.commTurn + 2;
    CcuEventHandle eventHandle{0};
    HCCLCHECK(HcommCcuEventAlloc(insHandle, DIE_ID, totalEventNum, &eventHandle));
    for (uint32_t i = 0; i < totalEventNum; i++) {
        HCCLCHECK(HcommCcuEventGetAddr(eventHandle, i, &ctx.mappedAddrs.eventVa[i]));
    }

    // ---- 构造 kernelArg ----
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

    // ---- token（data 和 scale 分别获取）----
    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(sendBuf), ctx.inputSize, &ctx.token));
    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(sendScaleBuf), ctx.scaleInputSize, &ctx.scaleToken));

    // ---- goSize：data 和 scale 各一组 ----
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;

    auto dataGoSize = CalGoSize(ctx.dataSliceSize, config);
    ctx.goSize[0] = dataGoSize[0];
    ctx.goSize[1] = dataGoSize[1];
    ctx.goSize[2] = dataGoSize[2];
    ctx.goSize[3] = dataGoSize[3];

    auto scaleGoSizeResult = CalGoSize(ctx.scaleSliceSize, config);
    ctx.scaleGoSize[0] = scaleGoSizeResult[0];
    ctx.scaleGoSize[1] = scaleGoSizeResult[1];
    ctx.scaleGoSize[2] = scaleGoSizeResult[2];
    ctx.scaleGoSize[3] = scaleGoSizeResult[3];

    // ---- data inputOffset/outputOffset（commTurn 组）----
    for (uint32_t r = 0; r < ctx.commTurn; r++) {
        ctx.inputOffsets[r] = static_cast<uint64_t>(r) * ctx.dataSliceSize;
        ctx.outputOffsets[r] = static_cast<uint64_t>(r) * ctx.rankSize * ctx.dataSliceSize
                             + static_cast<uint64_t>(ctx.rankId) * ctx.dataSliceSize;
    }

    // ---- scale inputOffset/outputOffset（1 组，全量）----
    ctx.scaleInputOffsets[0] = 0;
    ctx.scaleOutputOffsets[0] = static_cast<uint64_t>(ctx.rankId) * ctx.scaleSliceSize;

    // ---- 拷贝 offsets 到 device GM ----
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

    // scale offsets（1 个元素）
    void *dScaleInputOffsets = nullptr;
    aclRet = aclrtMalloc(&dScaleInputOffsets, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_ONLY);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    void *dScaleOutputOffsets = nullptr;
    aclRet = aclrtMalloc(&dScaleOutputOffsets, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_ONLY);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtMemcpy(dScaleInputOffsets, sizeof(uint64_t),
                         ctx.scaleInputOffsets, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtMemcpy(dScaleOutputOffsets, sizeof(uint64_t),
                         ctx.scaleOutputOffsets, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    ctx.d_scaleInputOffsetsAddr = reinterpret_cast<uint64_t>(dScaleInputOffsets);
    ctx.d_scaleOutputOffsetsAddr = reinterpret_cast<uint64_t>(dScaleOutputOffsets);

    ctx.schd = {1, 0, 0x01};

    // 校验 recvBuf 容量
    const uint64_t totalRecvSize = static_cast<uint64_t>(ctx.rankSize) * ctx.dataSliceSize * ctx.commTurn;
    const uint64_t recvBufSize = sendCount * ctx.rankSize * perDataSize;
    if (totalRecvSize > recvBufSize) {
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}
