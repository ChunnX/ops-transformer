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
// PrepareAllGatherMatmul — Host 侧 CCU 通信资源编排（不做 kernel launch）
//
// 职责：为 AllGather + MXFP8 Matmul 通算并行准备所有 CCU 通信资源，
//       填充 AllGatherMatmulContext，供 .asc 中的 <<<>>> launch 使用。
//
// 编排步骤：
//   1. 校验参数（dataType 必须是 INT8/FP8，sendCount 非 0）
//   2. 获取 rankId/rankSize，计算 MXFP8 scale 参数（scaleKLen, scaleBytesPerMRow）
//   3. 计算 data/scale 通信参数（sliceSize, inputSize, commTurn）
//   4. 获取 CCU channels（每对 rank 一条点到点通道）
//   5. 创建 CCU 实例（跨 DIE 资源池）
//   6. 分配 Variable：(commTurn+1)×10 个（scale round 1 组 + data round commTurn 组）
//   7. 分配 Event：commTurn+2 个（1 wake + 1 scale done + commTurn data done）
//   8. 构造 kernelArg（封装 channels/vars/events/commTurn）
//   9. 获取 token（data 和 scale 分别获取内存访问令牌）
//  10. 计算 goSize（data 和 scale 各一组，CCU 本地拷贝 loop 参数）
//  11. 计算 offsets（data 的 commTurn 组 + scale 的 1 组，拷贝到 device GM）
//
// 资源布局总结：
//   Variable: startVa[0..commTurn][0..9] — 组 0=scale, 组 1..commTurn=data
//   Event:    eventVa[0]=wake, [1]=scale done, [2..commTurn+1]=data round done
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

    // ---- 校验参数 ----
    // 第四阶段：dataType 必须是 FP8（HCCL_DATA_TYPE_INT8 复用，1 byte per element）
    if (dataType != HCCL_DATA_TYPE_INT8) {
        return HCCL_E_PARA;
    }
    if (sendCount == 0) {
        return HCCL_SUCCESS;
    }

    // ---- 获取 rank 信息 ----
    HCCLCHECK(HcclGetRankId(comm, &ctx.rankId));
    HCCLCHECK(HcclGetRankSize(comm, &ctx.rankSize));

    // ---- 计算 MXFP8 通信参数 ----
    // MXFP8 格式：每 64 个 FP8 元素共享 1 个 E8M0 scale，但 scale 按 MXFP_MULTI=2 分组存储
    constexpr uint64_t MXFP_DIVISOR = 64;
    constexpr uint64_t MXFP_MULTI = 2;
    uint32_t perDataSize = SIZE_TABLE[dataType];  // FP8 = 1 byte (HCCL_DATA_TYPE_INT8)
    ctx.inputSize = sendCount * perDataSize;  // M*K*1（每 rank sendBuf 全量）

    // M = sendCount / K（矩阵行数），scaleKLen = ceil(K/64)*2（每行 scale 元素数）
    uint32_t M = static_cast<uint32_t>(sendCount / K);
    uint64_t scaleKLen = (K + MXFP_DIVISOR - 1) / MXFP_DIVISOR * MXFP_MULTI;
    uint64_t scaleBytesPerMRow = scaleKLen * 1;  // E8M0 = 1 byte
    ctx.scaleInputSize = scaleCount;  // M * scaleBytesPerMRow

    // data 通信参数：每 round 每 rank = tileM 行 × K 列 × 1 byte
    ctx.dataSliceSize = static_cast<uint64_t>(tileM) * static_cast<uint64_t>(K) * perDataSize;
    // scale 通信参数：全量 M 行 × scaleBytesPerMRow（scale 只 1 轮，不分 round）
    ctx.scaleSliceSize = static_cast<uint64_t>(M) * scaleBytesPerMRow;

    // commTurn = M / tileM（data 分多少 round 通信+计算）
    // 例如 M=512, tileM=256 → commTurn=2
    ctx.commTurn = M / tileM;
    if (ctx.commTurn > MAX_COMMTURN) {
        return HCCL_E_PARA;
    }

    // ---- 分配 CCU channels ----
    // 为每个远端 rank 获取一条点到点 CCU 通信通道
    std::vector<ChannelHandle> kernelChannels;
    HCCLCHECK(AcquireCcuChannels(comm, ctx.rankId, ctx.rankSize, kernelChannels));

    // ---- 创建 CCU 实例 ----
    // 跨 2 个 DIE 创建 CCU 资源池（Variable/Event/Buffer/Loop/Thread）
    CcuInsHandle insHandle{0};
    HCCLCHECK(CreateCcuInsWithEnabledDies(insHandle));
    HcclResult assignRet = HcclCommAssignCcuIns(comm, insHandle);
    if (assignRet != HCCL_SUCCESS) {
        (void)HcommCcuInsDestroy(insHandle);
        return assignRet;
    }
    ctx.insHandle = insHandle;

    // ---- 分配 Variable：(commTurn+1) × HCOMM_HOST_TASK_ARG_NUM 个 ----
    // 组 0 = scale round（全量 scale AllGather 参数）
    // 组 1..commTurn = data round（每轮 tileM 行 FP8 data AllGather 参数）
    // 每组 10 个 Variable：[0]=input, [1]=output, [2]=token, [3]=inputOffset,
    //   [4]=outputOffset, [5]=sliceSize, [6..9]=goSize[0..3]
    // HcommCcuVariableGetAddr 获取每个 Variable 的 GM 后背地址，存入 mappedAddrs.startVa[d][i]
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
    // eventVa[0]   = wake   （AIC block0 → CCU，唯一唤醒信号）
    // eventVa[1]   = scale AllGather done（CCU → AIC，所有核轮询）
    // eventVa[d+2] = data round d done（CCU → AIC），d = 0..commTurn-1
    // HcommCcuEventGetAddr 获取每个 Event 的 GM 后背地址，存入 mappedAddrs.eventVa[i]
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

    // ---- 获取 token（data 和 scale 分别获取）----
    // token 是 CCU 内存访问令牌，用于安全校验——CCU 核访问 GM 地址时需携带对应 token
    // data 和 scale 的 sendBuf 在不同地址，需要分别获取 token
    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(sendBuf), ctx.inputSize, &ctx.token));
    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(sendScaleBuf), ctx.scaleInputSize, &ctx.scaleToken));

    // ---- 计算 goSize（data 和 scale 各一组）----
    // goSize 是 CCU 本地拷贝的 loop 参数（offset, loopIterNum, parallelParam, residual）
    // 用于将 sliceSize 分解为完整 loop 块 + 尾部扩展
    // LoopGroupConfig: msInterleave=8, loopCount=8, memSlice=8*4096=32768 → loopSize=262144
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;

    // data goSize（每 round 的 sliceSize = tileM*K*1）
    auto dataGoSize = CalGoSize(ctx.dataSliceSize, config);
    ctx.goSize[0] = dataGoSize[0];
    ctx.goSize[1] = dataGoSize[1];
    ctx.goSize[2] = dataGoSize[2];
    ctx.goSize[3] = dataGoSize[3];

    // scale goSize（全量 scaleSliceSize = M*scaleBytesPerMRow）
    auto scaleGoSizeResult = CalGoSize(ctx.scaleSliceSize, config);
    ctx.scaleGoSize[0] = scaleGoSizeResult[0];
    ctx.scaleGoSize[1] = scaleGoSizeResult[1];
    ctx.scaleGoSize[2] = scaleGoSizeResult[2];
    ctx.scaleGoSize[3] = scaleGoSizeResult[3];

    // ---- 计算 data inputOffset/outputOffset（commTurn 组）----
    //
    // ─── AllGather 前的数据排布（sendBuf，每 rank 独立）───
    //
    //   sendBuf = M 行 × K 列 × 1 byte(FP8)，按 round 切分：
    //
    //   round 0: rows [0, tileM)         → inputOffset[0] = 0
    //   round 1: rows [tileM, 2*tileM)   → inputOffset[1] = dataSliceSize
    //   round r: rows [r*tileM, (r+1)*tileM) → inputOffset[r] = r * dataSliceSize
    //
    //   dataSliceSize = tileM * K * 1 byte（每 round 每 rank 的通信量）
    //
    // ─── AllGather 后的数据排布（recvBuf，所有 rank 汇聚）───
    //
    //   recvBuf = rankSize * M 行 × K 列 × 1 byte(FP8)
    //   各 rank 的数据按 rankId 顺序拼接，每个 round 汇聚一组：
    //
    //   round 0 汇聚区: [rank0 的 round0 tileM 行][rank1 的 round0 tileM 行]...
    //     → outputOffset[0] = 0 * rankSize * dataSliceSize + rankId * dataSliceSize
    //
    //   round 1 汇聚区: [rank0 的 round1 tileM 行][rank1 的 round1 tileM 行]...
    //     → outputOffset[1] = 1 * rankSize * dataSliceSize + rankId * dataSliceSize
    //
    //   round r 汇聚区: 偏移 = r * rankSize * dataSliceSize + rankId * dataSliceSize
    //
    //   即 recvBuf 中 round r 的数据起始 = r * (rankSize * tileM * K) + rankId * (tileM * K)
    //   每 round AllGather 后的总行数 = rankSize * tileM（所有 rank 的 tileM 行拼接）
    //
    // ─── 具体示例（2 rank, M=512, tileM=256, K=256, commTurn=2）───
    //
    //   sendBuf (rank 0):
    //     round 0: rows 0-255  (inputOffset=0)
    //     round 1: rows 256-511 (inputOffset=256*256=65536)
    //
    //   recvBuf (rank 0):
    //     round 0: [rank0 rows 0-255][rank1 rows 0-255]  (outputOffset=0)
    //     round 1: [rank0 rows 256-511][rank1 rows 256-511] (outputOffset=2*65536=131072)
    //
    //   matmul round 0: A = recvBuf[0 : 512*256]      (512 行 = 2*256)
    //   matmul round 1: A = recvBuf[131072 : 131072+512*256]
    //
    for (uint32_t r = 0; r < ctx.commTurn; r++) {
        ctx.inputOffsets[r] = static_cast<uint64_t>(r) * ctx.dataSliceSize;
        ctx.outputOffsets[r] = static_cast<uint64_t>(r) * ctx.rankSize * ctx.dataSliceSize
                             + static_cast<uint64_t>(ctx.rankId) * ctx.dataSliceSize;
    }

    // ---- 计算 scale inputOffset/outputOffset（1 组，全量）----
    //
    // ─── AllGather 前的 scale 排布（sendScaleBuf，每 rank 独立）───
    //
    //   sendScaleBuf = M 行 × scaleKLen 列 × 1 byte(E8M0)
    //   scaleKLen = ceil(K/64) * 2，例如 K=256 → scaleKLen=8
    //   scale 只 1 轮全量通信，不分 round：
    //     inputOffset = 0（从头开始）
    //
    // ─── AllGather 后的 scale 排布（recvScaleBuf，所有 rank 汇聚）───
    //
    //   recvScaleBuf = rankSize * M 行 × scaleKLen 列 × 1 byte(E8M0)
    //   各 rank 的 scale 按 rankId 顺序拼接：
    //     [rank0 的 M 行 scale][rank1 的 M 行 scale]...
    //     outputOffset = rankId * scaleSliceSize
    //     scaleSliceSize = M * scaleBytesPerMRow（全量 scale 字节数）
    //
    // ─── matmul 读取时的 scale 偏移（在 AicDirectPollProcess 中计算）───
    //
    //   round d 的 scaleA 起始 = recvScaleBuf + d * rankSize * tileM * scaleBytesPerMRow
    //   即每 round 取 rankSize * tileM 行的 scale（与 data 的 tile 行数对齐）
    //
    ctx.scaleInputOffsets[0] = 0;
    ctx.scaleOutputOffsets[0] = static_cast<uint64_t>(ctx.rankId) * ctx.scaleSliceSize;

    // ---- 拷贝 offsets 到 device GM ----
    // AIC block 0 在 device 侧预写参数桥时需要读取这些 offset 值，
    // 所以必须先拷贝到 device GM 供 kernel 访问
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
