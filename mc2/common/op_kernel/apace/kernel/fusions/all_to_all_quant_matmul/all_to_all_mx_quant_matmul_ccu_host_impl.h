/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_to_all_mx_quant_matmul_ccu_host_impl.h
 * \brief AllToAll MX Quant Matmul（CCU 版）host 侧编排
 *
 * 编排函数：ValidateAllToAllMatmulParams（check，不触碰 CCU 资源）→
 * PrepareAllToAllMatmul（prepare，经 core/ccu_comm 分配 CCU 资源并填充 tiling 资源字段）→
 * AllToAllMatmulCustom（见 all_to_all_ccu_kernel_impl.asc，双流 launch）→
 * ReleaseAllToAllMatmulContext（释放，调用方在迭代循环外统一调用一次）。
 * tiling 填充分工：main 构造 comm.tileM + matmul 全部字段 → Validate 填 comm 形状/轮次
 * → Prepare 填 comm 资源字段 → launch 时整体经 <<<>>> 传 kernel。
 *
 * 位置说明：本文件是融合专属 host 编排（形状校验/切分计算/双流 launch 编排），
 * 与 device 侧文件同置于 kernel/fusions/ 目录；通信资源管理与同步原语在
 * core/ccu_comm/（ccu_comm.h + all_to_all/ccu_all_to_all_comm.h），由锚点 .asc 先行引入。
 */

#pragma once

// 编排结果：填充后交给 .asc 中的 AllToAllMatmulCustom
struct AllToAllMatmulContext {
    // asc-comm 调度描述符（numBlocks/reserved/phyDieMask/binaryCacheTag）
    asccomm_ccu_schd schd;
    CcuInsHandle insHandle{0};
    // 单组 kernelArg（管理 (dataCommTurn+1) × HCOMM_HOST_TASK_ARG_NUM(11) 个 Variable
    // + dataCommTurn+2 个 Event；kernelArg->commTurn = dataCommTurn + 1）
    std::shared_ptr<CcuKernelArgAllToAllMesh1D> kernelArg;
    // 统一 tiling：matmul 由 Custom 入参拷入；comm 由 Validate（形状）+ Prepare（资源）填充
    ccuAllToAllMatmulTilingData tiling{};
    // data 轮 per-round 偏移（host 侧中转：拷贝到 device 后仅保留 comm.d_* 地址进 kernel）
    uint64_t srcOffsets[MAX_COMMTURN];
    uint64_t dstOffsets[MAX_COMMTURN];
    uint64_t sliceSizes[MAX_COMMTURN];
    // workspace 布局与 token 获取用的全量尺寸
    uint64_t inputSize{0};      // x1 全量字节 = M*K（workspace data 区大小）
    uint64_t scaleInputSize{0}; // x1Scale 全量字节 = M*scaleKLen（workspace scale 区大小）
    // CCU DIE 信息（AcquireCcuChannels 查询所得：dieId 用于资源分配，dieMask 用于 schd）
    uint32_t dieId{0};
    uint64_t dieMask{0};
};

// 前置检查（check 阶段，不触碰任何 CCU 资源）：指针/参数合法性校验 + rank 信息查询
// + 全部切分尺寸计算（填 ctx.tiling.comm 的形状/轮次字段）。
// 返回 HCCL_SUCCESS / HCCL_E_PARA；特殊：sendCount==0 返回 HCCL_SUCCESS 且 skip=true
inline HcclResult ValidateAllToAllMatmulParams(void *x1, void *workspace, void *x1Scale,
                                         HcclComm comm,
                                         uint64_t sendCount, uint64_t scaleCount,
                                         uint64_t workspaceSize,
                                         HcclDataType dataType,
                                         AllToAllMatmulContext &ctx, bool &skip)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(x1);
    CHK_PTR_NULL(workspace);
    CHK_PTR_NULL(x1Scale);
    if (sendCount == 0) {
        skip = true;
        return HCCL_SUCCESS;
    }

    HCCLCHECK(HcclGetRankId(comm, &ctx.tiling.comm.rankId));
    HCCLCHECK(HcclGetRankSize(comm, &ctx.tiling.comm.rankSize));

    // rankSize == 0 除零风险；> CCU_MAX_RANK_SIZE(16) 越界 device 侧 channels 数组
    const uint32_t rankSize = ctx.tiling.comm.rankSize;
    if (rankSize == 0 || rankSize > CCU_MAX_RANK_SIZE) {
        return HCCL_E_PARA;
    }

    // dtype 固定 FP8 E4M3FN（1 字节，按 HCCL_DATA_TYPE_INT8 通道传）
    if (dataType != HCCL_DATA_TYPE_INT8) {
        return HCCL_E_PARA;
    }
    const uint64_t perDataSize = SIZE_TABLE[dataType];

    // MXFP8：K 须为 64 倍数（scale 组对齐），scaleKLen = K/64*2
    constexpr uint64_t MXFP_DIVISOR = 64;
    constexpr uint64_t MXFP_MULTI = 2;
    const QuantMatmulTilingData &mmTile = ctx.tiling.matmul;
    const uint64_t K = mmTile.k;
    if (K == 0 || K % MXFP_DIVISOR != 0) {
        return HCCL_E_PARA;
    }
    const uint64_t scaleKLen = K / MXFP_DIVISOR * MXFP_MULTI;

    if (sendCount % K != 0) {
        return HCCL_E_PARA;
    }
    const uint64_t M = sendCount / K;
    if (M % rankSize != 0) {
        return HCCL_E_PARA;  // token 轴均分 R 块，须整除
    }
    const uint64_t blockRows = M / rankSize;

    // matmul.m 必须与通信切分一致（kernel 的 originalM/problemShape.M 均取自它）
    if (mmTile.m != blockRows) {
        return HCCL_E_PARA;
    }
    if (mmTile.n == 0 || mmTile.baseM == 0 || mmTile.baseN == 0 ||
        mmTile.baseK == 0 || mmTile.baseK % MXFP_DIVISOR != 0 ||
        mmTile.usedCoreNum == 0) {
        return HCCL_E_PARA;
    }

    // dataCommTurn = blockRows/tileM（tileM 即 WaitTile 的通信轮映射粒度）
    const uint64_t tileM = ctx.tiling.comm.tileM;
    if (tileM == 0 || blockRows % tileM != 0) {
        return HCCL_E_PARA;
    }
    if (blockRows / tileM > MAX_COMMTURN) {
        return HCCL_E_PARA;
    }
    ctx.tiling.comm.dataCommTurn = static_cast<uint32_t>(blockRows / tileM);

    ctx.inputSize = sendCount * perDataSize;
    if (ctx.inputSize % rankSize != 0) {
        return HCCL_E_PARA;
    }
    ctx.tiling.comm.blockSize = ctx.inputSize / rankSize;
    ctx.tiling.comm.srcStride = ctx.tiling.comm.blockSize;
    ctx.tiling.comm.subBytes = ctx.tiling.comm.blockSize / ctx.tiling.comm.dataCommTurn;
    if (ctx.tiling.comm.subBytes % perDataSize != 0) {
        return HCCL_E_PARA;
    }

    // scale 全量 1 轮
    ctx.scaleInputSize = static_cast<uint64_t>(M) * scaleKLen;
    if (scaleCount != ctx.scaleInputSize) {
        return HCCL_E_PARA;
    }
    if (ctx.scaleInputSize % rankSize != 0) {
        return HCCL_E_PARA;
    }
    ctx.tiling.comm.scaleSliceSize = ctx.scaleInputSize / rankSize;
    ctx.tiling.comm.scaleSrcStride = ctx.tiling.comm.scaleSliceSize;
    ctx.tiling.comm.scaleDstOffset = static_cast<uint64_t>(ctx.tiling.comm.rankId)
                                   * ctx.tiling.comm.scaleSliceSize;

    // workspace [recvData | recvScale]
    ctx.tiling.comm.workspaceScaleOffset = ctx.inputSize;
    if (workspaceSize < ctx.inputSize + ctx.scaleInputSize) {
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

// 获取 CCU channels（每远端 rank 一条点到点通道）+ 创建 CCU 实例。
// 实例所有权自管（不调用 HcclCommAssignCcuIns）：由 ReleaseAllToAllMatmulContext 经
// core 的 ReleaseCcuContext 与创建严格配对
inline HcclResult AcquireCommChannelsAndInstance(HcclComm comm, uint32_t rankId, uint32_t rankSize,
                                         std::vector<ChannelHandle> &kernelChannels, AllToAllMatmulContext &ctx)
{
    uint32_t dieId = 0;
    HCCLCHECK(AcquireCcuChannels(comm, rankId, rankSize, kernelChannels, dieId));
    if (dieId >= CCU_DIE_NUM) {
        return HCCL_E_PARA;
    }
    ctx.dieId = dieId;
    ctx.dieMask = 1ULL << dieId;

    CcuInsHandle insHandle{0};
    HCCLCHECK(CreateCcuInsWithEnabledDies(insHandle));
    ctx.insHandle = insHandle;
    return HCCL_SUCCESS;
}

// goSize 计算 + per-round offsets 计算 + 拷贝到 device GM
// （AIC 融合 kernel block 0 预写参数区时经 comm.d_*OffsetsAddr 读 device 偏移表）
inline HcclResult ComputeCommOffsets(AllToAllMatmulContext &ctx)
{
    const CcuCommTilingData &comm = ctx.tiling.comm;
    ctx.tiling.comm.goSize = CalGoSize(comm.subBytes, MakeLocalCopyConfig());
    ctx.tiling.comm.scaleGoSize = CalGoSize(comm.scaleSliceSize, MakeLocalCopyConfig());

    for (uint32_t t = 0; t < comm.dataCommTurn; t++) {
        ctx.srcOffsets[t] = static_cast<uint64_t>(t) * comm.subBytes;
        ctx.dstOffsets[t] = static_cast<uint64_t>(comm.rankId) * comm.blockSize
                          + static_cast<uint64_t>(t) * comm.subBytes;
        ctx.sliceSizes[t] = comm.subBytes;
    }

    const uint64_t bytes = comm.dataCommTurn * sizeof(uint64_t);
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

    ctx.tiling.comm.d_srcOffsetsAddr = reinterpret_cast<uint64_t>(dSrcOffsets);
    ctx.tiling.comm.d_dstOffsetsAddr = reinterpret_cast<uint64_t>(dDstOffsets);
    ctx.tiling.comm.d_sliceSizesAddr = reinterpret_cast<uint64_t>(dSliceSizes);
    return HCCL_SUCCESS;
}

// 释放 CCU 资源（转发 core/ccu_comm 的 ReleaseCcuContext），与 Prepare 首帧分配配对。
// 调用时机：迭代循环外统一一次（须在双流 sync 之后——实例销毁后 MMIO 地址失效）
inline HcclResult ReleaseAllToAllMatmulContext(AllToAllMatmulContext &ctx)
{
    return ReleaseCcuContext(ctx.tiling.comm, ctx.insHandle);
}

// Host 侧 CCU 通信资源准备（prepare 阶段，不做 kernel launch）。
// 前置条件：ValidateAllToAllMatmulParams 已通过。步骤：
//   0. 迭代间复用守卫——insHandle 已就绪且资源规模与当前 shape 核对一致 → 直接返回
//      （shape 变化须先 Release 再以新 ctx 重建；buffer 地址须保持稳定）
//   1. AcquireCommChannelsAndInstance — CCU channels + CCU 实例
//   2. AllocCommRegion（core）— Variable/Event 分配 + kernelArg 构造（mappedAddrs 烧入 tiling）
//   3. token 获取（x1 与 x1Scale 两个 buffer 分别获取）
//   4. ComputeCommOffsets — goSize + offsets + device 拷贝
inline HcclResult PrepareAllToAllMatmul(void *x1, void *x1Scale,
                                  HcclComm comm, AllToAllMatmulContext &ctx)
{
    if (ctx.insHandle != 0) {
        const uint32_t turn = ctx.tiling.comm.dataCommTurn;
        if (ctx.kernelArg->commTurn != turn + 1 ||
            ctx.kernelArg->varNum != (turn + 1) * HCOMM_HOST_TASK_ARG_NUM ||
            ctx.kernelArg->eventNum != turn + 2) {
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    std::vector<ChannelHandle> kernelChannels;
    HCCLCHECK(AcquireCommChannelsAndInstance(comm, ctx.tiling.comm.rankId, ctx.tiling.comm.rankSize,
                                    kernelChannels, ctx));

    HCCLCHECK(AllocCommRegion(ctx.insHandle, ctx.dieId, ctx.tiling.comm, kernelChannels, ctx.kernelArg));

    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(x1), ctx.inputSize,
                                  &ctx.tiling.comm.token));
    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(x1Scale), ctx.scaleInputSize,
                                  &ctx.tiling.comm.scaleToken));

    HCCLCHECK(ComputeCommOffsets(ctx));

    // binaryCacheTag 必须按 rank 区分：缓存是进程级单例，单进程多 rank 线程共享，
    // 同 tag 会让 rank1-3 命中 rank0 的 handle（绑定 rank0 设备）→ launch 报错 →
    // rank0 跨 rank 屏障等不齐 → 挂死。tag = kernel 名哈希 + rankId
    ctx.schd = {1, 0, ctx.dieMask,
                HcommCcuGetLaunchHashTag("CcuAllToAllMesh1DKernel") + ctx.tiling.comm.rankId};

    return HCCL_SUCCESS;
}
