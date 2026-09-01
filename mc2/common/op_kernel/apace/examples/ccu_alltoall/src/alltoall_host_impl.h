/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file alltoall_host_impl.h
 * \brief Host 侧编排：PrepareAllToAll / ReleaseAllToAllContext（阶段1：FP16 单流，纯通信）
 *
 * 由 ccu_all_gather_matmul 的 host_launcher.h 改造而来：
 *   - Variable：commTurn × 11 个（每 round 一组，无 scale 轮——AG 是 (commTurn+1)×10）
 *   - Event：commTurn + 1 个（[0]=wake，[r+1]=round r done——AG 是 commTurn+2 个）
 *   - offsets 语义按 AllToAll 换块语义重算（见下方注释块）
 *   - dtype 校验 INT8 → FP16
 *
 * ⚠️ 依赖前置符号（由 .asc 在 include 本文件之前定义；调整 .asc 前文顺序时须保证仍在场）：
 *   宏：     HCCLCHECK / CHK_PTR_NULL
 *   常量：   SIZE_TABLE / MAX_COMMTURN / HCOMM_HOST_TASK_ARG_NUM / CCU_MEMSLICES_PER_LOOP /
 *            CCU_LOCAL_COPY_LOOP_COUNT / CCU_MS_INTERLEAVE / CCU_MS_SIZE
 *   类型：   CcuMappedAddrs / A2aGoSize / CcuKernelArgAllToAllMesh1D / LoopGroupConfig /
 *            AllToAllContext 的字段类型（CcuInsHandle 等 hccl 类型）
 *   函数：   AcquireCcuChannels / CreateCcuInsWithEnabledDies / CalGoSize / MakeLocalCopyConfig
 *
 * 资源布局总结（槽位语义的单一事实源见 .asc 的 BridgeSlot 枚举）：
 *   Variable: startVa[0..commTurn-1][0..10] — 每 round 一组 11 槽
 *   Event:    eventVa[0]=wake, eventVa[r+1]=round r done
 */

#pragma once

// 编排结果：填充后交给 .asc 中的 launch 函数
struct AllToAllContext {
    // asc-comm 正式版调度描述符（4 字段：numBlocks/reserved/phyDieMask/binaryCacheTag，
    // 旧草稿 HCommCcuSchd 3 字段版已废弃；类型声明见 ccu_host_launch.h）
    asccomm_ccu_schd schd;
    CcuInsHandle insHandle{0};
    // 单组 kernelArg（管理 commTurn × HCOMM_HOST_TASK_ARG_NUM(11) 个 Variable + commTurn+1 个 Event）
    std::shared_ptr<CcuKernelArgAllToAllMesh1D> kernelArg;
    // mappedAddrs：commTurn 组 startVa[r][i] + commTurn+1 个 eventVa
    CcuMappedAddrs mappedAddrs;
    // commTurn 组 per-round 偏移（host 侧数组，需拷贝到 device 供 AIC 桥 kernel 读取）
    uint64_t srcOffsets[MAX_COMMTURN];
    uint64_t dstOffsets[MAX_COMMTURN];
    uint64_t sliceSizes[MAX_COMMTURN];
    // device 侧地址
    uint64_t d_srcOffsetsAddr{0};
    uint64_t d_dstOffsetsAddr{0};
    uint64_t d_sliceSizesAddr{0};
    // token（sendBuf 内存访问令牌）
    uint64_t token{0};
    // 通信参数
    uint64_t inputSize;     // 每 rank sendBuf 全量字节 = sendCount * sizeof(fp16)
    uint64_t blockSize;     // 每 rank 单块字节 = inputSize / rankSize（均匀 AllToAll 的块大小）
    uint64_t srcStride;     // 块间距字节 = blockSize（紧邻切分，strideCount == dataCount 场景）
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t commTurn;      // 通信轮数（blockSize 再按 commTurn 均分）
    // CCU DIE 信息（AcquireCcuChannels 查询所得，替代旧版硬编码 DIE_ID=1：
    // dieId 用于 Variable/Event 分配，dieMask 用于 schd.phyDieMask）
    uint32_t dieId{0};
    uint64_t dieMask{0};
    // goSize（CCU 本地拷贝 loop 参数，CalGoSize(subBytes) 结果；各 round subBytes 相同故全轮共用一组）
    A2aGoSize goSize;
};

// ============================================================================
// ─── AllToAll offsets 语义（与 AllGather 的本质差异）───
//
//   sendBuf（每 rank，FP16，均分 R 块，块大小 blockSize 字节）：
//     block j = 发往 rank j 的数据：sendBuf[j*blockSize, (j+1)*blockSize)
//     block myRank = 留给本地的数据（GroupCopy 本地搬运）
//
//   recvBuf（每 rank，与 sendBuf 等量——AllToAll 无膨胀）：
//     block r = 来自 rank r 的数据：recvBuf[r*blockSize, (r+1)*blockSize)
//     即远端 rank j 把它的 sendBuf 第 myRank 块写到本 rank 的 recvBuf 第 myRank 块
//     （发送方 rankId 决定接收方落点：dstOffset 含 myRank*blockSize 项，host 预乘烧入）
//
//   commTurn 轮按块内均分（sub = blockSize / commTurn）：
//     round r: srcOffset[r] = r * sub
//               dstOffset[r] = myRank * blockSize + r * sub   ← myRank 项 host 预乘烧入，
//                                                                CCU 侧无标量×Variable 运算
//               sliceSize[r] = sub,  srcStride = blockSize
//     语义：每轮各 rank 交换"自己第 myRank 块内的第 r 子段"
//     commTurn=1 时退化为 srcOffset=0、dstOffset=myRank*blockSize、sliceSize=blockSize（最简形态）
//
//   精度校验期望（sample_main 侧）：
//     recvBuf 第 r 块 == rank r 的 sendBuf 第 myRank 块
// ============================================================================

// ----------------------------------------------------------------------------
// ValidateAllToAllParams — 前置检查（check 阶段，不触碰任何 CCU 资源）
// 职责：指针/参数合法性校验 + rank 信息查询 + 切分尺寸计算（填 ctx 的纯数值字段）。
// 返回：HCCL_SUCCESS   参数合法，调用方可继续 PrepareAllToAll
//       HCCL_E_PARA    参数非法（指针空/dtype 错/整除性不满足/commTurn 越界）
//       特殊：sendCount==0 时返回 HCCL_SUCCESS 且 skip=true（无数据可传，调用方
//             不应 Prepare 也不应 launch——skip 语义显式返回，不藏在 ctx 状态里）
// ----------------------------------------------------------------------------
inline HcclResult ValidateAllToAllParams(void *sendBuf, void *recvBuf, HcclComm comm,
                                          uint64_t sendCount, HcclDataType dataType,
                                          uint32_t commTurn, AllToAllContext &ctx, bool &skip)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    if (sendCount == 0) {
        skip = true;  // 无数据可传：合法但跳过（显式 skip，取代原"早退 SUCCESS + kernelArg null"的隐式契约）
        return HCCL_SUCCESS;
    }

    // rank 信息查询（校验阶段完成，Prepare 不再查询）
    HCCLCHECK(HcclGetRankId(comm, &ctx.rankId));
    HCCLCHECK(HcclGetRankSize(comm, &ctx.rankSize));

    // rankSize 范围校验：
    //   ==0 除零风险；>CCU_MAX_RANK_SIZE(16) 越界 device 侧 channels[16] 数组，
    //   且 Event 的 uint16 位掩码 (1<<rankSize)-1 依赖 ≤16（此前仅靠 sample 侧守卫间接保证）
    if (ctx.rankSize == 0 || ctx.rankSize > CCU_MAX_RANK_SIZE) {
        return HCCL_E_PARA;
    }
    const uint32_t rankSize = ctx.rankSize;
    // dtype 校验：阶段1 固定 FP16（SIZE_TABLE[HCCL_DATA_TYPE_FP16] = 2 字节）
    if (dataType != HCCL_DATA_TYPE_FP16) {
        return HCCL_E_PARA;
    }
    const uint64_t perDataSize = SIZE_TABLE[dataType];
    ctx.inputSize = sendCount * perDataSize;  // 每 rank sendBuf 全量字节

    // 均匀 AllToAll：sendBuf 均分 R 块，块大小必须整除（字节粒度校验）
    if (ctx.inputSize % rankSize != 0) {
        return HCCL_E_PARA;
    }
    ctx.blockSize = ctx.inputSize / rankSize;
    ctx.srcStride = ctx.blockSize;  // 紧邻切分：stride == blockSize

    if (commTurn == 0 || commTurn > MAX_COMMTURN) {
        return HCCL_E_PARA;
    }
    ctx.commTurn = commTurn;
    // 块内按 commTurn 均分：子段字节数必须整除且保持 FP16 元素对齐（偶数字节）
    if (ctx.blockSize % ctx.commTurn != 0) {
        return HCCL_E_PARA;
    }
    const uint64_t subBytes = ctx.blockSize / ctx.commTurn;
    if (subBytes % perDataSize != 0) {  // 子段必须是完整 FP16 元素
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

// ----------------------------------------------------------------------------
// AcquireChannelsAndIns — 获取 CCU channels（每远端 rank 一条点到点通道）+ 创建并绑定 CCU 实例
// 【对齐 asc-comm ccu_direct 官方实现：dieId 由 AcquireCcuChannels 查询端点属性获得，
//   dieMask = 1<<dieId 供 schd 使用；dieId 越界（>= CCU_DIE_NUM）报参数错】
// ----------------------------------------------------------------------------
inline HcclResult AcquireChannelsAndIns(HcclComm comm, uint32_t rankId, uint32_t rankSize,
                                         std::vector<ChannelHandle> &kernelChannels, AllToAllContext &ctx)
{
    uint32_t dieId = 0;
    HCCLCHECK(AcquireCcuChannels(comm, rankId, rankSize, kernelChannels, dieId));
    if (dieId >= CCU_DIE_NUM) {
        return HCCL_E_PARA;
    }
    ctx.dieId = dieId;
    ctx.dieMask = 1ULL << dieId;

    // 跨 2 个 DIE 创建 CCU 资源池（Variable/Event/Buffer/Loop/Thread）
    CcuInsHandle insHandle{0};
    HCCLCHECK(CreateCcuInsWithEnabledDies(insHandle));
    HcclResult assignRet = HcclCommAssignCcuIns(comm, insHandle);
    if (assignRet != HCCL_SUCCESS) {
        (void)HcommCcuInsDestroy(insHandle);
        return assignRet;
    }
    ctx.insHandle = insHandle;
    return HCCL_SUCCESS;
}

// ----------------------------------------------------------------------------
// AllocBridgeResources — 共享内存参数区（"参数桥"，见 .asc BridgeSlot 术语注）Variable/Event 分配 + kernelArg 构造
// Variable: commTurn × 11 个（配额校验：commTurn=16 时 176 < CCU_DEFAULT_RES_FRACTION_VARIABLE=400）
// Event:    commTurn + 1 个（1 wake + commTurn 个 round done）
// ----------------------------------------------------------------------------
inline HcclResult AllocBridgeResources(AllToAllContext &ctx, const std::vector<ChannelHandle> &kernelChannels)
{
    // DIE_ID 由端点属性动态查询（ctx.dieId，见 AcquireChannelsAndIns），替代旧版硬编码
    const uint8_t ccuDieId = static_cast<uint8_t>(ctx.dieId);
    uint32_t totalVarNum = ctx.commTurn * HCOMM_HOST_TASK_ARG_NUM;
    CcuVariableHandle varHandle{0};
    HCCLCHECK(HcommCcuVariableAlloc(ctx.insHandle, ccuDieId, totalVarNum, &varHandle));
    for (uint32_t r = 0; r < ctx.commTurn; r++) {
        for (uint32_t i = 0; i < HCOMM_HOST_TASK_ARG_NUM; i++) {
            HCCLCHECK(HcommCcuVariableGetAddr(varHandle, r * HCOMM_HOST_TASK_ARG_NUM + i,
                                               &ctx.mappedAddrs.startVa[r][i]));
        }
    }

    uint32_t totalEventNum = ctx.commTurn + 1;
    CcuEventHandle eventHandle{0};
    HCCLCHECK(HcommCcuEventAlloc(ctx.insHandle, ccuDieId, totalEventNum, &eventHandle));
    for (uint32_t i = 0; i < totalEventNum; i++) {
        HCCLCHECK(HcommCcuEventGetAddr(eventHandle, i, &ctx.mappedAddrs.eventVa[i]));
    }

    ctx.kernelArg = std::make_shared<CcuKernelArgAllToAllMesh1D>();
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
    return HCCL_SUCCESS;
}

// ----------------------------------------------------------------------------
// ComputeAndCopyOffsets — goSize 计算 + per-round offsets 计算 + 拷贝到 device GM
// （AIC block 0 预写参数桥时从 device GM 数组读取这些值，必须先拷过去）
// ----------------------------------------------------------------------------
inline HcclResult ComputeAndCopyOffsets(AllToAllContext &ctx)
{
    // goSize：CCU 本地拷贝 loop 参数（各 round 的 sliceSize 相同 = subBytes，全轮共用一组）
    const uint64_t subBytes = ctx.blockSize / ctx.commTurn;
    ctx.goSize = CalGoSize(subBytes, MakeLocalCopyConfig());

    // srcOffset[r] = r * sub                      （sendBuf 第 myRank 块内的第 r 子段）
    // dstOffset[r] = myRank * blockSize + r * sub （recvBuf 第 myRank 块内的第 r 子段）
    // sliceSize[r] = sub
    for (uint32_t r = 0; r < ctx.commTurn; r++) {
        ctx.srcOffsets[r] = static_cast<uint64_t>(r) * subBytes;
        ctx.dstOffsets[r] = static_cast<uint64_t>(ctx.rankId) * ctx.blockSize
                          + static_cast<uint64_t>(r) * subBytes;
        ctx.sliceSizes[r] = subBytes;
    }

    // 注意：本函数返回 HcclResult，aclxx 失败须转 HCCL_E_INTERNAL（ACLCHECK 宏会 return
    // aclError，类型不兼容——AG 原版即用此局部变量模式）
    const uint64_t bytes = ctx.commTurn * sizeof(uint64_t);
    void *dSrcOffsets = nullptr;
    void *dDstOffsets = nullptr;
    void *dSliceSizes = nullptr;
    aclError aclRet = aclrtMalloc(&dSrcOffsets, bytes, ACL_MEM_MALLOC_HUGE_ONLY);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtMalloc(&dDstOffsets, bytes, ACL_MEM_MALLOC_HUGE_ONLY);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtMalloc(&dSliceSizes, bytes, ACL_MEM_MALLOC_HUGE_ONLY);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }

    aclRet = aclrtMemcpy(dSrcOffsets, bytes, ctx.srcOffsets, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtMemcpy(dDstOffsets, bytes, ctx.dstOffsets, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtMemcpy(dSliceSizes, bytes, ctx.sliceSizes, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }

    ctx.d_srcOffsetsAddr = reinterpret_cast<uint64_t>(dSrcOffsets);
    ctx.d_dstOffsetsAddr = reinterpret_cast<uint64_t>(dDstOffsets);
    ctx.d_sliceSizesAddr = reinterpret_cast<uint64_t>(dSliceSizes);
    return HCCL_SUCCESS;
}

// ----------------------------------------------------------------------------
// ReleaseAllToAllContext — 释放 CCU 资源（device GM 上的 offset 数组 + CCU 实例）
// 与 PrepareAllToAll 的分配配对（谁分配谁释放，成对出现在本文件）
// ----------------------------------------------------------------------------
inline HcclResult ReleaseAllToAllContext(AllToAllContext &ctx)
{
    aclError aclRet = ACL_SUCCESS;
    if (ctx.d_srcOffsetsAddr) {
        aclRet = aclrtFree(reinterpret_cast<void *>(ctx.d_srcOffsetsAddr));
        if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
        ctx.d_srcOffsetsAddr = 0;
    }
    if (ctx.d_dstOffsetsAddr) {
        aclRet = aclrtFree(reinterpret_cast<void *>(ctx.d_dstOffsetsAddr));
        if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
        ctx.d_dstOffsetsAddr = 0;
    }
    if (ctx.d_sliceSizesAddr) {
        aclRet = aclrtFree(reinterpret_cast<void *>(ctx.d_sliceSizesAddr));
        if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
        ctx.d_sliceSizesAddr = 0;
    }
    if (ctx.insHandle) {
        HCCLCHECK(HcommCcuInsDestroy(ctx.insHandle));
        ctx.insHandle = 0;
    }
    return HCCL_SUCCESS;
}

// ============================================================================
// PrepareAllToAll — Host 侧 CCU 通信资源准备（prepare 阶段，不做 kernel launch）
// 前置条件：ValidateAllToAllParams 已通过（skip=false）——本函数假定参数合法，不再校验。
// 步骤：
//   1. AcquireChannelsAndIns   — CCU channels + CCU 实例创建绑定
//   2. AllocBridgeResources    — 共享内存参数区 Variable/Event 分配 + kernelArg 构造
//   3. token 获取（CCU 内存访问令牌：CCU 核访问 GM 地址时需携带）
//   4. ComputeAndCopyOffsets   — goSize + offsets + device 拷贝
// 释放侧对偶：ReleaseAllToAllContext（由调用方在双流 sync 后调用）
// ============================================================================
inline HcclResult PrepareAllToAll(void *sendBuf,
                                  HcclComm comm, uint32_t commTurn, AllToAllContext &ctx)
{
    // sendBuf 仅用于 token 获取；参数校验与切分尺寸已在 ValidateAllToAllParams 完成

    // ---- 1. CCU channels + CCU 实例 ----
    std::vector<ChannelHandle> kernelChannels;
    HCCLCHECK(AcquireChannelsAndIns(comm, ctx.rankId, ctx.rankSize, kernelChannels, ctx));

    // ---- 2. 共享内存参数区 Variable/Event + kernelArg ----
    HCCLCHECK(AllocBridgeResources(ctx, kernelChannels));

    // ---- 3. token ----
    // 对齐 AG 已验证行为：仅对 sendBuf 获取（远端访问 recvBuf 沿用该令牌机制）
    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(sendBuf), ctx.inputSize, &ctx.token));

    // ---- 4. goSize + offsets + device 拷贝 ----
    HCCLCHECK(ComputeAndCopyOffsets(ctx));

    // asccomm_ccu_schd：CCU kernel 调度描述符 {mission数=1, 保留, dieMask, binaryCacheTag=0}
    // （dieMask 动态 = 1<<dieId，旧版硬编码 0x01；binaryCacheTag=0 对齐官方示例默认值）
    ctx.schd = {1, 0, ctx.dieMask, 0};

    return HCCL_SUCCESS;
}
