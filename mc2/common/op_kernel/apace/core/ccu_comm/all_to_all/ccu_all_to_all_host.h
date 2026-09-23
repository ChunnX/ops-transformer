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
 * \file ccu_all_to_all_host.h
 * \brief CCU AllToAll——host 侧资源装配
 *
 * 消费方接入（host）：AcquireCcuChannels + CreateCcuInsWithEnabledDies（ccu_comm.h）→
 * AllocAllToAllRegion → 填 CcuCommTilingData 形状字段（_tiling.h）→
 * CcuAllToAllMesh1DKernel<<<>>>（ccu_all_to_all_kernel.asc）
 */

#pragma once

#include "../ccu_comm.h"
#include "ccu_all_to_all_tiling.h"

// ---- host 侧资源装配 ----

// CKE 事件槽位预约 + kernelArg 构造（host）。预约的是 CCU 片上硬件寄存器槽位
// （申请期完成 MMIO 地址映射并烧入 comm.mappedAddrs），非 GM 内存；释放随
// HcommCcuInsDestroy 统一解映射还池。通信参数不走寄存器槽位——随 kernelArg->stage
// 按值直达 CCU kernel（消费方 host 每次调用刷新）
inline HcclResult AllocAllToAllRegion(CcuInsHandle insHandle, uint32_t dieId, CcuCommTilingData &comm,
                                      const std::vector<ChannelHandle> &kernelChannels,
                                      std::shared_ptr<CcuKernelArgAllToAllMesh1D> &kernelArg)
{
    const uint8_t ccuDieId = static_cast<uint8_t>(dieId);

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
    kernelArg->eventHandle = eventHandle;
    kernelArg->eventNum = totalEventNum;
    // CCU kernel 循环次数 = dataCommTurn + 1（round 0 = scale A2A，round 1..c = data 轮）
    kernelArg->commTurn = comm.dataCommTurn + 1;
    // kernelArg->stage 由消费方 host 每次调用经 RefreshCcuCommStage 刷新（通信参数随参直达）
    for (uint32_t i = 0; i < kernelChannels.size(); ++i) {
        kernelArg->channels[i] = kernelChannels[i];
    }
    return HCCL_SUCCESS;
}

// 释放 CCU 资源（CCU 实例销毁），与 AllocAllToAllRegion 配对。
// 调用时机：迭代循环外统一一次（须在双流 sync 之后——实例销毁后 MMIO 地址失效）
inline HcclResult ReleaseCcuContext(CcuCommTilingData &comm, CcuInsHandle &insHandle)
{
    if (insHandle) {
        HCCLCHECK(HcommCcuInsDestroy(insHandle));
        insHandle = 0;
    }
    return HCCL_SUCCESS;
}
