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
 * \file ccu_all_to_all_comm.h
 * \brief CCU AllToAll 通信服务——通信 tiling 契约 + AIC 侧同步原语 + host 侧资源装配
 *
 * 事件编号契约（AIC↔CCU 同步五步，消费方按此对接）：
 *   ev[0] = wake（AIC commit → CCU process，循环前唯一一次）
 *   ev[1] = scale 轮 done、ev[t+2] = data 轮 t done（CCU notify → AIC wait，
 *   等待节奏由消费方 kernel 自定——如经 CommPolicy 逐 tile 等待）
 * 槽位契约：每轮 COMM_SLOT_NUM(11) 槽；组 0 = scale 轮全量，组 1..dataCommTurn = data 轮
 *
 * 消费方接入（host）：AcquireCcuChannels + CreateCcuInsWithEnabledDies（ccu_comm.h）→
 * AllocCommRegion → 填 CcuCommTilingData 形状字段 → CcuAllToAllMesh1DKernel<<<>>>
 */

#pragma once

#include "../ccu_comm.h"

// CCU AllToAll 通信侧 tiling（通信标量散参全部收纳于此；形状/轮次由消费方 host 填，
// 资源字段由 AllocCommRegion/消费方 host 填）
#pragma pack(push, 8)
struct alignas(8) CcuCommTilingData {
    // ---- 形状/轮次（消费方 host 填；kernel 侧只读）----
    uint32_t rankId{0};
    uint32_t rankSize{0};
    uint32_t dataCommTurn{0};         // data 轮数（CCU 总轮数 = dataCommTurn+1，组 0 = scale 轮）
    uint32_t tileM{0};                // 每 data 轮传输行数
    uint64_t blockSize{0};            // data 每块字节 = blockRows*K
    uint64_t srcStride{0};            // 块间距字节 = blockSize（紧邻切分）
    uint64_t subBytes{0};             // 每 data 轮子段字节 = tileM*K
    uint64_t scaleSliceSize{0};       // scale 每块字节（全量 1 轮）
    uint64_t scaleSrcStride{0};       // = scaleSliceSize
    uint64_t scaleDstOffset{0};       // = rankId*scaleSliceSize（host 预乘烧入）
    uint64_t workspaceScaleOffset{0}; // workspace 内 recvScale 区偏移
    // ---- CCU 资源（host 填；AIC 写通信参数区时消费）----
    uint64_t token{0};                // x1（data）的 CCU 内存访问令牌
    uint64_t scaleToken{0};           // x1Scale 的 CCU 内存访问令牌
    A2aGoSize goSize{};               // data 本地拷贝 loop 参数
    A2aGoSize scaleGoSize{};          // scale 本地拷贝 loop 参数
    CcuMappedAddrs mappedAddrs{};     // 通信参数区 XN/CKE 寄存器槽位 MMIO 地址映射
    uint64_t d_srcOffsetsAddr{0};     // per-round data src 偏移表（device GM）
    uint64_t d_dstOffsetsAddr{0};     // per-round data dst 偏移表
    uint64_t d_sliceSizesAddr{0};     // per-round data sliceSize 表
};
#pragma pack(pop)

// 等待 CCU 某轮通信完成：绕 DCache 直读 CKE 寄存器槽位（MMIO uncached 语义）
__aicore__ inline void WaitCcuCommDone(uint64_t eventVa)
{
    __gm__ uint64_t *doneAddr = reinterpret_cast<__gm__ uint64_t *>(eventVa);
    while (AscendC::ReadGmByPassDCache(doneAddr) != 1) {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        AscendC::Nop<800>();
    }
}

// 按轮号 + 槽位枚举写通信参数区（XN 寄存器 MMIO 直写，绕 DCache）
__aicore__ inline void WriteCommMappedAddr(const CcuMappedAddrs &addrs, uint32_t r,
                                           CommSlot slot, uint64_t value)
{
    __gm__ uint64_t *addr = reinterpret_cast<__gm__ uint64_t *>(addrs.startVa[r][slot]);
    AscendC::WriteGmByPassDCache<uint64_t>(addr, value);
}

// AIC 侧同步第 1 步【prepare】：预写全部轮的通信参数（block 0 调用；
// 组 0 = scale 轮全量 A2A，组 1..dataCommTurn = data 轮 t = 组 t+1）
__aicore__ inline void PrepareAllToAllComm(const CcuCommTilingData &comm,
                                           __gm__ void *x1, __gm__ void *x1Scale,
                                           __gm__ void *recvBuf, __gm__ void *recvScaleBuf)
{
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_INPUT, reinterpret_cast<uint64_t>(x1Scale));
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_OUTPUT, reinterpret_cast<uint64_t>(recvScaleBuf));
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_TOKEN, comm.scaleToken);
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_SRC_OFFSET, 0);
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_DST_OFFSET, comm.scaleDstOffset);
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_SLICE_SIZE, comm.scaleSliceSize);
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_SRC_STRIDE, comm.scaleSrcStride);
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_GO_ADDR_OFFSET, comm.scaleGoSize.addrOffset);
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_GO_LOOP_ITER, comm.scaleGoSize.loopIterNum);
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_GO_LOOP_EXTEND, comm.scaleGoSize.loopExtendNum);
    WriteCommMappedAddr(comm.mappedAddrs, 0, SLOT_GO_TAIL_SIZE, comm.scaleGoSize.tailSize);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

    __gm__ const uint64_t *srcOffsets =
        reinterpret_cast<__gm__ const uint64_t *>(comm.d_srcOffsetsAddr);
    __gm__ const uint64_t *dstOffsets =
        reinterpret_cast<__gm__ const uint64_t *>(comm.d_dstOffsetsAddr);
    __gm__ const uint64_t *sliceSizes =
        reinterpret_cast<__gm__ const uint64_t *>(comm.d_sliceSizesAddr);
    for (uint32_t t = 0; t < comm.dataCommTurn; t++) {
        const uint32_t r = t + 1;
        const uint64_t srcOff = srcOffsets[t];
        const uint64_t dstOff = dstOffsets[t];
        const uint64_t sliceSize = sliceSizes[t];

        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_INPUT, reinterpret_cast<uint64_t>(x1));
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_OUTPUT, reinterpret_cast<uint64_t>(recvBuf));
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_TOKEN, comm.token);
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_SRC_OFFSET, srcOff);
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_DST_OFFSET, dstOff);
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_SLICE_SIZE, sliceSize);
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_SRC_STRIDE, comm.srcStride);
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_GO_ADDR_OFFSET, comm.goSize.addrOffset);
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_GO_LOOP_ITER, comm.goSize.loopIterNum);
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_GO_LOOP_EXTEND, comm.goSize.loopExtendNum);
        WriteCommMappedAddr(comm.mappedAddrs, r, SLOT_GO_TAIL_SIZE, comm.goSize.tailSize);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    }
}

// AIC 侧同步第 2 步【commit】：写 wake event，通知 CCU 启动通信（block 0 调用）
__aicore__ inline void CommitCcuComm(const CcuMappedAddrs &mappedAddrs)
{
    __gm__ uint64_t *wakeAddr = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[0]);
    AscendC::WriteGmByPassDCache<uint64_t>(wakeAddr, 1);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
}

// 通信参数区 Variable(XN)/Event(CKE) 寄存器预约 + kernelArg 构造（host）。
// 预约的是 CCU 片上硬件寄存器槽位（申请期完成 MMIO 地址映射并烧入 comm.mappedAddrs），
// 非 GM 内存；释放随 HcommCcuInsDestroy 统一解映射还池
inline HcclResult AllocCommRegion(CcuInsHandle insHandle, uint32_t dieId, CcuCommTilingData &comm,
                                  const std::vector<ChannelHandle> &kernelChannels,
                                  std::shared_ptr<CcuKernelArgAllToAllMesh1D> &kernelArg)
{
    const uint8_t ccuDieId = static_cast<uint8_t>(dieId);
    const uint32_t totalVarNum = (comm.dataCommTurn + 1) * HCOMM_HOST_TASK_ARG_NUM;
    CcuVariableHandle varHandle{0};
    HCCLCHECK(HcommCcuVariableAlloc(insHandle, ccuDieId, totalVarNum, &varHandle));
    for (uint32_t r = 0; r < comm.dataCommTurn + 1; r++) {
        for (uint32_t i = 0; i < HCOMM_HOST_TASK_ARG_NUM; i++) {
            HCCLCHECK(HcommCcuVariableGetAddr(varHandle, r * HCOMM_HOST_TASK_ARG_NUM + i,
                                               &comm.mappedAddrs.startVa[r][i]));
        }
    }

    const uint32_t totalEventNum = comm.dataCommTurn + 2;
    CcuEventHandle eventHandle{0};
    HCCLCHECK(HcommCcuEventAlloc(insHandle, ccuDieId, totalEventNum, &eventHandle));
    for (uint32_t i = 0; i < totalEventNum; i++) {
        HCCLCHECK(HcommCcuEventGetAddr(eventHandle, i, &comm.mappedAddrs.eventVa[i]));
    }

    kernelArg = std::make_shared<CcuKernelArgAllToAllMesh1D>();
    kernelArg->rankSize = comm.rankSize;
    kernelArg->rankId = comm.rankId;
    kernelArg->channelCount = static_cast<uint32_t>(kernelChannels.size());
    kernelArg->varHandle = varHandle;
    kernelArg->varNum = totalVarNum;
    kernelArg->eventHandle = eventHandle;
    kernelArg->eventNum = totalEventNum;
    // CCU kernel 循环次数 = dataCommTurn + 1（round 0 = scale A2A，round 1..c = data 轮）
    kernelArg->commTurn = comm.dataCommTurn + 1;
    for (uint32_t i = 0; i < kernelChannels.size(); ++i) {
        kernelArg->channels[i] = kernelChannels[i];
    }
    return HCCL_SUCCESS;
}

// 释放通信资源（device GM 上的 offset 表 + CCU 实例），与 AllocCommRegion 配对。
// 调用时机：迭代循环外统一一次（须在双流 sync 之后——实例销毁后 MMIO 地址失效）
inline HcclResult ReleaseCcuContext(CcuCommTilingData &comm, CcuInsHandle &insHandle)
{
    aclError aclRet = ACL_SUCCESS;
    if (comm.d_srcOffsetsAddr) {
        aclRet = aclrtFree(reinterpret_cast<void *>(comm.d_srcOffsetsAddr));
        if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
        comm.d_srcOffsetsAddr = 0;
    }
    if (comm.d_dstOffsetsAddr) {
        aclRet = aclrtFree(reinterpret_cast<void *>(comm.d_dstOffsetsAddr));
        if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
        comm.d_dstOffsetsAddr = 0;
    }
    if (comm.d_sliceSizesAddr) {
        aclRet = aclrtFree(reinterpret_cast<void *>(comm.d_sliceSizesAddr));
        if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
        comm.d_sliceSizesAddr = 0;
    }
    if (insHandle) {
        HCCLCHECK(HcommCcuInsDestroy(insHandle));
        insHandle = 0;
    }
    return HCCL_SUCCESS;
}
