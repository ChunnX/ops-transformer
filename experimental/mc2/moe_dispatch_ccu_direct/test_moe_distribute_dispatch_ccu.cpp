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
 * \file test_moe_distribute_dispatch_ccu.cpp
 * \brief A6 CCU MoeDistributeDispatchCcuSetup / Teardown host 驱动（v3）。
 *
 * v3 关键变更：
 *   - 新增 routingTableDev：host 端预算 expertId/rank 映射，喂给 setup（绕过 device
 *     端 Var/const 除法不支持的约束）。
 *   - expertIds 改为多样化路由数据（覆盖每个 rank），让 0.5RTT 的 wishCntXn/totalCntXn
 *     真的有跨卡流量可对比。原 memset 0 路径会让 totalCntXn 永远凑不齐。
 *   - 第二次 Launch 后显式 memset expertTokenNumsDev 为 0（device 端 v3 不再写）。
 *   - commCmdInfoBytes 64KB 充足，前 epWorldSize 槽位是正式槽，后 epWorldSize 槽
 *     位是 setup 第 7 阶段 scratch（互不冲突）。
 *
 * A5 上跑不动 A6 0.5RTT；本驱动面向 A6 真机 / 仿真。
 */

#include <acl/acl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/ccu_kernel.h"
#include "hcomm/ccu/hccl_ccu_res.h"
#include "moe_distribute_dispatch_ccu_setup.h"
#include "moe_distribute_dispatch_ccu_teardown.h"

static const char *kRootInfoPath = "/tmp/ccu_a6_rootinfo.bin";
static const char *kRootInfoTmp  = "/tmp/ccu_a6_rootinfo.bin.tmp";

static bool ExchangeRootInfo(int32_t rankId, HcclRootInfo *rootInfo)
{
    if (rankId == 0) {
        if (HcclGetRootInfo(rootInfo) != HCCL_SUCCESS) { return false; }
        FILE *fp = fopen(kRootInfoTmp, "wb");
        if (fp == nullptr) { return false; }
        size_t wn = fwrite(rootInfo->internal, 1, HCCL_ROOT_INFO_BYTES, fp);
        (void)fclose(fp);
        if (wn != HCCL_ROOT_INFO_BYTES) { return false; }
        (void)rename(kRootInfoTmp, kRootInfoPath);
        return true;
    }
    for (int i = 0; i < 600; ++i) {
        if (access(kRootInfoPath, R_OK) == 0) {
            FILE *fp = fopen(kRootInfoPath, "rb");
            if (fp != nullptr) {
                size_t rn = fread(rootInfo->internal, 1, HCCL_ROOT_INFO_BYTES, fp);
                (void)fclose(fp);
                if (rn == HCCL_ROOT_INFO_BYTES) { return true; }
            }
        }
        usleep(100 * 1000);
    }
    return false;
}

#define ACL_CHECK(expr) do { aclError _r = (expr); if (_r != ACL_SUCCESS) { printf("[ERROR] " #expr " ret=%d\n", (int)_r); return -1; } } while (0)
#define HCCL_CHECK(expr) do { HcclResult _r = (expr); if (_r != HCCL_SUCCESS) { printf("[ERROR] " #expr " ret=%d\n", (int)_r); return -1; } } while (0)

int main(int argc, char *argv[])
{
    int32_t rankId = 0, rankSize = 2;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--rank_id") == 0)        rankId   = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--rank_size") == 0) rankSize = atoi(argv[i + 1]);
    }
    const int32_t deviceId = rankId;

    const uint32_t bs           = 8;
    const uint32_t hiddenSize   = 7168;
    const uint32_t topK         = 8;
    const uint32_t moeExpertNum = 32;
    const uint64_t tokenMsgLen  = static_cast<uint64_t>(hiddenSize) * sizeof(uint16_t);
    const uint32_t moeExpertPerRankNum = moeExpertNum / rankSize;

    printf("[rank %d] MoeDistributeDispatchCcu host driver v3 start (device %d, rankSize %d, "
           "bs=%u, topK=%u, moeExpertNum=%u, moeExpertPerRankNum=%u)\n",
           rankId, deviceId, rankSize, bs, topK, moeExpertNum, moeExpertPerRankNum);

    // ---- ACL + 通信域 ----
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    HcclRootInfo rootInfo;
    if (!ExchangeRootInfo(rankId, &rootInfo)) { printf("[rank %d][ERROR] rootInfo 交换失败\n", rankId); return -1; }
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 200;
    HcclComm hcclComm = nullptr;
    HCCL_CHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &commConfig, &hcclComm));

    // ---- HcclThreadAcquire + HcclChannelAcquire（CCU 引擎，0.5RTT tag）----
    ThreadHandle threadHandle = 0;
    HCCL_CHECK(HcclThreadAcquire(hcclComm, CommEngine::COMM_ENGINE_CCU, 1, 0, &threadHandle));

    const uint32_t channelNum = static_cast<uint32_t>(rankSize) - 1;
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    HcclChannelDescInit(channelDescs.data(), channelNum);
    for (uint32_t k = 0; k < channelNum; ++k) {
        uint32_t remoteRank = (k < static_cast<uint32_t>(rankId)) ? k : k + 1;
        channelDescs[k].remoteRank = remoteRank;
        channelDescs[k].channelProtocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        // channelDescs[k].channelGroupTag[0] = MOE_DISPATCH_CCU_TAG;   // 等 ChannelDesc 增量字段就位（见 OPEN_QUESTIONS P0-3）
        channelDescs[k].memHandleNum = 0;
    }
    std::vector<ChannelHandle> channels(channelNum);
    HCCL_CHECK(HcclChannelAcquire(hcclComm, CommEngine::COMM_ENGINE_CCU,
                                  channelDescs.data(), channelNum, channels.data()));

    // ---- 注册 setup / teardown 两个 CcuKernel ----
    std::vector<uint32_t> subCommRanks(rankSize);
    for (int32_t r = 0; r < rankSize; ++r) subCommRanks[r] = r;

    auto setupArg = std::make_shared<hcomm::CcuKernelArgMoeDistributeDispatchCcuSetup>(
        rankSize, rankId, moeExpertNum, bs, topK, hiddenSize, subCommRanks);
    auto teardownArg = std::make_shared<hcomm::CcuKernelArgMoeDistributeDispatchCcuTeardown>(
        rankSize, rankId, moeExpertNum, bs, topK, hiddenSize, subCommRanks);

    hcomm::KernelCreator setupCreator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<hcomm::CcuKernelMoeDistributeDispatchCcuSetup>(arg);
    };
    hcomm::KernelCreator teardownCreator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<hcomm::CcuKernelMoeDistributeDispatchCcuTeardown>(arg);
    };

    CcuKernelHandle setupKernel = 0, teardownKernel = 0;
    HCCL_CHECK(HcclCcuKernelRegister(hcclComm, &setupKernel,
                                      static_cast<void *>(&setupCreator),
                                      static_cast<void *>(setupArg.get())));
    HCCL_CHECK(HcclCcuKernelRegister(hcclComm, &teardownKernel,
                                      static_cast<void *>(&teardownCreator),
                                      static_cast<void *>(teardownArg.get())));
    HCCL_CHECK(HcclCcuKernelRegisterFinish(hcclComm));

    // ---- GM 缓冲分配 ----
    auto devAlloc = [](size_t bytes) -> void * {
        void *p = nullptr;
        aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (p != nullptr) aclrtMemset(p, bytes, 0, bytes);
        return p;
    };
    const size_t xBytes               = static_cast<size_t>(bs) * hiddenSize * sizeof(uint16_t);
    const size_t expertIdsBytes       = static_cast<size_t>(bs) * topK * sizeof(uint32_t);
    const size_t routingTableBytes    = static_cast<size_t>(bs) * topK * sizeof(uint32_t);
    const size_t yBytes               = static_cast<size_t>(rankSize) * bs * tokenMsgLen;
    const size_t expandIdxBytes       = static_cast<size_t>(bs) * topK * sizeof(uint32_t);
    const size_t commCmdInfoBytes     = 64 * 1024;  // 前 epWorldSize 是正式槽，后 epWorldSize 是 scratch
    const size_t expandXBytes         = static_cast<size_t>(rankSize) * bs * tokenMsgLen;
    const size_t assistInfoBytes      = static_cast<size_t>(rankSize) * 128 * sizeof(uint32_t);
    const size_t expertTokenNumsBytes = static_cast<size_t>(moeExpertPerRankNum) * sizeof(uint64_t);

    void *xDev               = devAlloc(xBytes);
    void *expertIdsDev       = devAlloc(expertIdsBytes);
    void *routingTableDev    = devAlloc(routingTableBytes);
    void *yDev               = devAlloc(yBytes);
    void *expandIdxDev       = devAlloc(expandIdxBytes);
    void *commCmdInfoDev     = devAlloc(commCmdInfoBytes);
    void *expandXDev         = devAlloc(expandXBytes);
    void *assistInfoDev      = devAlloc(assistInfoBytes);
    void *expertTokenNumsDev = devAlloc(expertTokenNumsBytes);

    // ---- 准备测试数据：x BF16 / expertIds 多样化路由 / routingTable 预算 ----
    // x：每个字节填 0x01（BF16 视图下是 ~1.5e-39 量级，仅为非零数据流），保证不为全 0
    std::vector<uint8_t> hostX(xBytes, 0x01);
    ACL_CHECK(aclrtMemcpy(xDev, xBytes, hostX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // expertIds：用线性路由覆盖所有 rank，保证每个 rank 都收到 token、
    //   wishCntXn/totalCntXn 真正有跨卡比对可触发。
    // routingTable[idx] = expertIds[idx] / moeExpertPerRankNum
    std::vector<uint32_t> hostExpertIds(bs * topK);
    std::vector<uint32_t> hostRoutingTable(bs * topK);
    for (uint32_t i = 0; i < bs; ++i) {
        for (uint32_t j = 0; j < topK; ++j) {
            uint32_t idx = i * topK + j;
            hostExpertIds[idx]    = idx % moeExpertNum;
            hostRoutingTable[idx] = hostExpertIds[idx] / moeExpertPerRankNum;
        }
    }
    ACL_CHECK(aclrtMemcpy(expertIdsDev,    expertIdsBytes,    hostExpertIds.data(),
                          expertIdsBytes,    ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(routingTableDev, routingTableBytes, hostRoutingTable.data(),
                          routingTableBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    if (rankId == 0) {
        // 打印路由分布做 sanity check
        std::vector<uint32_t> rankHit(rankSize, 0);
        for (uint32_t idx = 0; idx < bs * topK; ++idx) ++rankHit[hostRoutingTable[idx]];
        printf("[rank 0] routing histogram:");
        for (int r = 0; r < rankSize; ++r) printf(" rank%d=%u", r, rankHit[r]);
        printf("\n");
    }

    // ---- TaskArg + Launch ----
    auto setupTask = std::make_unique<hcomm::CcuTaskArgMoeDistributeDispatchCcuSetup>(
        reinterpret_cast<uint64_t>(xDev),
        reinterpret_cast<uint64_t>(expertIdsDev),
        reinterpret_cast<uint64_t>(routingTableDev),
        reinterpret_cast<uint64_t>(yDev),
        reinterpret_cast<uint64_t>(expandIdxDev),
        reinterpret_cast<uint64_t>(commCmdInfoDev));

    const uint32_t expectedPeerCount = static_cast<uint32_t>(rankSize - 1);
    auto teardownTask = std::make_unique<hcomm::CcuTaskArgMoeDistributeDispatchCcuTeardown>(
        reinterpret_cast<uint64_t>(xDev),
        reinterpret_cast<uint64_t>(yDev),
        reinterpret_cast<uint64_t>(expertIdsDev),
        reinterpret_cast<uint64_t>(commCmdInfoDev),
        reinterpret_cast<uint64_t>(expandXDev),
        reinterpret_cast<uint64_t>(assistInfoDev),
        reinterpret_cast<uint64_t>(expertTokenNumsDev),
        expectedPeerCount);

    HCCL_CHECK(HcclCcuKernelLaunch(hcclComm, threadHandle, setupKernel,
                                    static_cast<void *>(setupTask.get())));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    HCCL_CHECK(HcclCcuKernelLaunch(hcclComm, threadHandle, teardownKernel,
                                    static_cast<void *>(teardownTask.get())));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // expertTokenNums device 端 v3 不写；host 端显式归零，保证 IO 字段是合法值
    ACL_CHECK(aclrtMemset(expertTokenNumsDev, expertTokenNumsBytes, 0, expertTokenNumsBytes));

    printf("[rank %d] dispatch_setup + dispatch_teardown launched and synced\n", rankId);

    // ---- 释放 ----
    aclrtFree(xDev); aclrtFree(expertIdsDev); aclrtFree(routingTableDev);
    aclrtFree(yDev); aclrtFree(expandIdxDev); aclrtFree(commCmdInfoDev);
    aclrtFree(expandXDev); aclrtFree(assistInfoDev); aclrtFree(expertTokenNumsDev);
    HCCL_CHECK(HcclCommDestroy(hcclComm));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    printf("[rank %d] done\n", rankId);
    return 0;
}
