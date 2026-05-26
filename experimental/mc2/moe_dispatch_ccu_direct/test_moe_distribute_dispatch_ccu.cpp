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
 * \brief A6 CCU MoeDistributeDispatchCcuSetup / Teardown host 驱动（v4）。
 *
 * v4 端到端 IO 契约对齐库上 op_def：
 *   - KernelArg 新增 groupEp（REQUIRED attr，库上 HCCL 通信域名）；commAlg 改 string；
 *     sharedExpertNum / commType 默认值对齐 op_def。
 *   - SetupTaskArg 字段顺序：x, expert_ids, scales, x_active_mask, y, expand_idx, comm_cmd_info,
 *     + routing_table（v3 内部 scratch，库上无）。
 *   - TeardownTaskArg 字段顺序：x, y, expert_ids, comm_cmd_info, expand_x, dynamic_scales,
 *     assist_info_for_combine, expert_token_nums；删 expectedPeerCount（device 端从
 *     KernelArg.epWorldSize 推导）。
 *   - dynamic_scales 从 OPTIONAL 升 REQUIRED：host alloc buffer，sync 后归零。
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

    // ---- op_def 对齐参数 ----
    const std::string groupEp    = "epgroup_v4";            // op_def REQUIRED
    const uint32_t bs            = 8;
    const uint32_t hiddenSize    = 7168;
    const uint32_t topK          = 8;
    const uint32_t moeExpertNum  = 32;
    // 简化版固定：sharedExpertNum=0 (op_def 默认 1，但 950 平台当前仅支持 0)，
    //              sharedExpertRankNum=0，quantMode=0
    const uint32_t sharedExpertNum     = 0;
    const uint32_t sharedExpertRankNum = 0;
    const uint32_t quantMode           = 0;
    // v9: commType 改 2 (=URMA)。op_def 默认 0，但 docs §约束明确 950 当前仅支持 2。
    // v8 之前误传 0 跟 op_def 默认值对齐——这是 op_def 注册期 fallback，跟实际平台支持范围不同。
    const uint32_t commType            = 2;
    const std::string commAlg          = "";                 // 对齐 op_def 默认值（预留字段）
    const uint32_t expertTokenNumsType = 1;                  // 对齐 op_def 默认值

    const uint64_t tokenMsgLen         = static_cast<uint64_t>(hiddenSize) * sizeof(uint16_t);
    const uint32_t moeExpertPerRankNum = moeExpertNum / (rankSize - sharedExpertRankNum);

    printf("[rank %d] MoeDistributeDispatchCcu host driver v4 start "
           "(device=%d, rankSize=%d, groupEp=%s, bs=%u, topK=%u, moeExpertNum=%u, "
           "moeExpertPerRankNum=%u)\n",
           rankId, deviceId, rankSize, groupEp.c_str(), bs, topK, moeExpertNum, moeExpertPerRankNum);

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

    // ---- HcclThreadAcquire + HcclChannelAcquire ----
    ThreadHandle threadHandle = 0;
    HCCL_CHECK(HcclThreadAcquire(hcclComm, CommEngine::COMM_ENGINE_CCU, 1, 0, &threadHandle));

    const uint32_t channelNum = static_cast<uint32_t>(rankSize) - 1;
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    HcclChannelDescInit(channelDescs.data(), channelNum);
    for (uint32_t k = 0; k < channelNum; ++k) {
        uint32_t remoteRank = (k < static_cast<uint32_t>(rankId)) ? k : k + 1;
        channelDescs[k].remoteRank = remoteRank;
        channelDescs[k].channelProtocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        // channelDescs[k].channelGroupTag[0] = MOE_DISPATCH_CCU_TAG;  // 等 ChannelDesc 增量字段就位（OPEN_QUESTIONS P0-3）
        channelDescs[k].memHandleNum = 0;
    }
    std::vector<ChannelHandle> channels(channelNum);
    HCCL_CHECK(HcclChannelAcquire(hcclComm, CommEngine::COMM_ENGINE_CCU,
                                  channelDescs.data(), channelNum, channels.data()));

    // ---- 注册 setup / teardown 两个 CcuKernel ----
    auto setupArg = std::make_shared<hcomm::CcuKernelArgMoeDistributeDispatchCcuSetup>(
        groupEp, rankSize, rankId, moeExpertNum, bs, topK, hiddenSize,
        0 /*expertShardType*/, sharedExpertNum, sharedExpertRankNum, quantMode,
        0 /*globalBs*/, commType, commAlg);
    // v10 ★ teardown 不再是 CcuKernel——改为 AIV ascendc kernel 直调路径，
    //   不需要 CcuKernelArg / Creator / Register；改用 tiling buffer + `<<<>>>` 直 launch。

    // setupArg 仍是 CcuKernelArg（CCU 路径 setup）
    setupArg->channels = channels;   // 基类公开字段，HcclCcuKernelRegister 内部读取

    hcomm::KernelCreator setupCreator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<hcomm::CcuKernelMoeDistributeDispatchCcuSetup>(arg);
    };

    // v10：只 register setup；teardown 不走 hcomm 注册
    CcuKernelHandle setupKernel = 0;
    HCCL_CHECK(HcclCcuKernelRegister(hcclComm, &setupKernel,
                                      static_cast<void *>(&setupCreator),
                                      static_cast<void *>(setupArg.get())));
    HCCL_CHECK(HcclCcuKernelRegisterFinish(hcclComm));

    // ---- GM 缓冲分配 ----
    auto devAlloc = [](size_t bytes) -> void * {
        void *p = nullptr;
        aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (p != nullptr) aclrtMemset(p, bytes, 0, bytes);
        return p;
    };
    // v6：严格按库上 aclnnMoeDistributeDispatch{Setup,Teardown}.md docs 字段公式 alloc。
    // 公式来源：MoeDistributeDispatchSetup 官方 aclnn 接口文档 §约束说明 5。
    //   tokenMsgSize（非量化）  = Align256(H) * sizeof(YOutType)
    //   A（teardown 接收维度）  = BS * epWorldSize * min(localExpertNum, K)
    //   localExpertNum         = moeExpertNum / (epWorldSize - sharedExpertRankNum)
    auto align256 = [](uint32_t n) -> uint32_t { return ((n + 255U) / 256U) * 256U; };
    const uint32_t localExpertNum         = moeExpertPerRankNum;
    const size_t   tokenMsgSizeBytes      = static_cast<size_t>(align256(hiddenSize)) * sizeof(uint16_t);
    const uint64_t tokenWithMetaLenBytes  = tokenMsgSizeBytes + 32;     // v15 ★ +32B metadata 三元组
    const size_t   A                      = static_cast<size_t>(bs) * rankSize
                                            * static_cast<size_t>(localExpertNum < topK ? localExpertNum : topK);

    // op_def Input
    const size_t xBytes               = static_cast<size_t>(bs) * hiddenSize * sizeof(uint16_t);                   // (Bs, H)
    const size_t expertIdsBytes       = static_cast<size_t>(bs) * topK * sizeof(uint32_t);                         // (Bs, K) INT32
    const size_t scalesBytes          = static_cast<size_t>(sharedExpertNum + moeExpertNum)
                                        * hiddenSize * sizeof(float);                                              // OPT (sharedExpertNum+moeExpertNum, H) FLOAT
    const size_t xActiveMaskBytes     = static_cast<size_t>(bs) * sizeof(uint8_t);                                 // OPT (Bs,) BOOL

    // op_def Output（setup）
    // v13 ★ op_def yAddr 字段语义恢复库上（src 端发送缓冲），caller 按 op_def 公式 alloc 占位，
    //   device 端不读不写。对端接收 window 由独立的 commWindowAddr 字段承载（工程内部 scratch）。
    // v15 ★ commWindow / expandX 每槽含 32B metadata 三元组（tokenWithMetaLen 替代 tokenMsgLen）
    //        xWithMetaBytes: host 预拼装 (bs*K, tokenWithMetaLen) 给 dispatch_setup 段 7 阶段 B 用
    const size_t opDefYBytes          = static_cast<size_t>(bs) * (topK + sharedExpertNum) * tokenMsgSizeBytes;
    const size_t commWindowBytes      = static_cast<size_t>(rankSize) * bs * topK * tokenWithMetaLenBytes;  // v15 ★
    const size_t xWithMetaBytes       = static_cast<size_t>(bs) * topK * tokenWithMetaLenBytes;             // v15 ★
    const size_t expandIdxBytes       = static_cast<size_t>(bs) * topK * sizeof(uint32_t);
    const size_t commCmdInfoBytes     = (static_cast<size_t>(bs) * (topK + sharedExpertNum)
                                         + static_cast<size_t>(rankSize) * localExpertNum)
                                        * 16 * sizeof(int32_t);

    // op_def Output（teardown）
    // v15 ★ expandX 也是紧密 (rankSize*bs*topK, tokenWithMetaLen) worst-case alloc
    const size_t expandXBytes         = static_cast<size_t>(rankSize) * bs * topK * tokenWithMetaLenBytes;  // v15 ★
    const size_t dynamicScalesBytes   = A * sizeof(float);                                                         // (A,) FLOAT
    const size_t assistInfoBytes      = A * 128 * sizeof(int32_t);                                                 // (A*128,) INT32
    const size_t expertTokenNumsBytes = static_cast<size_t>(localExpertNum) * sizeof(uint64_t);                    // (localExpertNum,) INT64
    // 内部 scratch（库上无）
    const size_t routingTableBytes    = static_cast<size_t>(bs) * topK * sizeof(uint32_t);

    void *xDev               = devAlloc(xBytes);
    void *expertIdsDev       = devAlloc(expertIdsBytes);
    void *scalesDev          = devAlloc(scalesBytes);
    void *xActiveMaskDev     = devAlloc(xActiveMaskBytes);
    void *yDev               = devAlloc(opDefYBytes);            // v13 ★ op_def yAddr 占位
    void *commWindowDev      = devAlloc(commWindowBytes);        // v13/v15 接收 window（v15 含 metadata）
    void *xWithMetaDev       = devAlloc(xWithMetaBytes);         // v15 ★ host 预拼装 x+metadata
    void *expandIdxDev       = devAlloc(expandIdxBytes);
    void *commCmdInfoDev     = devAlloc(commCmdInfoBytes);

    // v11 ★ commCmdInfo 前 epWorldSize 槽位初始化为 sentinel -1（0xFFFFFFFF）
    //   teardown AIV 轮询此区域判断 src rank 是否已到位（== -1 表示未到位）。
    //   后段（scratch 区）保持 0 默认值不变。
    const size_t cmdInfoSentinelBytes = static_cast<size_t>(rankSize) * sizeof(int32_t);
    ACL_CHECK(aclrtMemset(commCmdInfoDev, cmdInfoSentinelBytes, 0xFF, cmdInfoSentinelBytes));
    void *expandXDev         = devAlloc(expandXBytes);
    void *dynamicScalesDev   = devAlloc(dynamicScalesBytes);
    void *assistInfoDev      = devAlloc(assistInfoBytes);
    void *expertTokenNumsDev = devAlloc(expertTokenNumsBytes);
    void *routingTableDev    = devAlloc(routingTableBytes);

    // ---- 测试数据 ----
    std::vector<uint8_t> hostX(xBytes, 0x01);
    ACL_CHECK(aclrtMemcpy(xDev, xBytes, hostX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));

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

    // v16 ★ 构造 xWithMetaDev：metadata 含 dstByteOff（host 预算，combine_setup 直接用作 GM 偏移）
    const size_t bytesPerToken = static_cast<size_t>(hiddenSize) * sizeof(uint16_t);
    std::vector<uint8_t> hostXWithMeta(xWithMetaBytes, 0);
    for (uint32_t i = 0; i < bs; ++i) {
        for (uint32_t j = 0; j < topK; ++j) {
            uint32_t idx = i * topK + j;
            uint8_t* rowBase = hostXWithMeta.data() + idx * tokenWithMetaLenBytes;
            std::memcpy(rowBase, hostX.data() + i * bytesPerToken, bytesPerToken);
            int32_t* meta = reinterpret_cast<int32_t*>(rowBase + tokenMsgSizeBytes);
            meta[0] = rankId;
            meta[1] = static_cast<int32_t>(i);
            meta[2] = static_cast<int32_t>(j);
            meta[3] = static_cast<int32_t>(idx * tokenWithMetaLenBytes);    // v16 ★ dstByteOff
        }
    }
    ACL_CHECK(aclrtMemcpy(xWithMetaDev, xWithMetaBytes, hostXWithMeta.data(),
                          xWithMetaBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // v15 ★ expandX padding 预初始化为 sentinel（int32 视角 = -1）
    ACL_CHECK(aclrtMemset(expandXDev, expandXBytes, 0xFF, expandXBytes));

    if (rankId == 0) {
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
        reinterpret_cast<uint64_t>(scalesDev),
        reinterpret_cast<uint64_t>(xActiveMaskDev),
        reinterpret_cast<uint64_t>(yDev),               // v13 ★ op_def yAddr 占位
        reinterpret_cast<uint64_t>(expandIdxDev),
        reinterpret_cast<uint64_t>(commCmdInfoDev),
        reinterpret_cast<uint64_t>(commWindowDev),      // v13 ★ 接收 window
        reinterpret_cast<uint64_t>(routingTableDev),
        reinterpret_cast<uint64_t>(xWithMetaDev));      // v15 ★ host 预拼装 x+metadata

    // v10 ★ teardown tiling buffer 准备（host 填充 + memcpy 到 device）
    MoeDistributeDispatchCcuTeardownTilingData hostTiling = {};
    hostTiling.epWorldSize         = rankSize;
    hostTiling.epRankId            = rankId;
    hostTiling.moeExpertNum        = moeExpertNum;
    hostTiling.moeExpertPerRankNum = moeExpertPerRankNum;
    hostTiling.bs                  = bs;
    hostTiling.topK                = topK;
    hostTiling.hiddenSize          = hiddenSize;
    hostTiling.expertTokenNumsType = expertTokenNumsType;
    hostTiling.tokenMsgLen         = tokenMsgSizeBytes;
    hostTiling.tokenWithMetaLen    = tokenWithMetaLenBytes;                                     // v15 ★
    hostTiling.segBytes            = static_cast<uint64_t>(bs) * topK * tokenWithMetaLenBytes;  // v15 ★
    void *tilingDev = devAlloc(sizeof(hostTiling));
    ACL_CHECK(aclrtMemcpy(tilingDev, sizeof(hostTiling),
                          &hostTiling, sizeof(hostTiling), ACL_MEMCPY_HOST_TO_DEVICE));

    // ==================== Launch ====================
    // (1) setup: CCU 路径（hcomm）。
    //     v10 setup CCU 段 9 内部已做 WaitAllPeersArriveNotify + ClearAllPeersArriveNotify，
    //     return 时所有对端 0.5RTT 数据已就位（不再异步）。
    HCCL_CHECK(HcclCcuKernelLaunch(hcclComm, threadHandle, setupKernel,
                                    static_cast<void *>(setupTask.get())));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // (2) teardown: AIV ascendc kernel via `<<<>>>` 直调（"AIV 直驱"）。
    //     stream 顺序保证此时 setup 已完成、y/commCmdInfo 内容稳定。
    //     注意 `<<<blockDim, l2_ctrl, stream>>>` 语法需 ascendc 工具链支持；
    //     A5/纯 host gcc 编译环境下该行不能直接编译——本工程目标 A6 ascendc 仿真/真机。
    constexpr uint32_t kTeardownBlockDim = 1;   // 简化 demo（生产可拆按 epWorldSize 并行）
    moe_distribute_dispatch_ccu_teardown<<<kTeardownBlockDim, nullptr, stream>>>(
        xDev, yDev, expertIdsDev, commCmdInfoDev,
        expandXDev, dynamicScalesDev,
        assistInfoDev, expertTokenNumsDev,
        commWindowDev,                  // v13 ★ device 端实际读的接收 window
        /*workspace=*/nullptr, tilingDev);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // ---- v10 简化版输出归零（device 端不写的 IO 字段）----
    ACL_CHECK(aclrtMemset(dynamicScalesDev,   dynamicScalesBytes,   0, dynamicScalesBytes));
    ACL_CHECK(aclrtMemset(expertTokenNumsDev, expertTokenNumsBytes, 0, expertTokenNumsBytes));

    printf("[rank %d] dispatch_setup + dispatch_teardown launched and synced\n", rankId);

    // ---- 释放 ----
    aclrtFree(xDev); aclrtFree(expertIdsDev); aclrtFree(scalesDev); aclrtFree(xActiveMaskDev);
    aclrtFree(yDev); aclrtFree(commWindowDev); aclrtFree(xWithMetaDev);   // v15 ★
    aclrtFree(expandIdxDev); aclrtFree(commCmdInfoDev);
    aclrtFree(expandXDev); aclrtFree(dynamicScalesDev);
    aclrtFree(assistInfoDev); aclrtFree(expertTokenNumsDev);
    aclrtFree(routingTableDev);
    aclrtFree(tilingDev);   // v10 ★
    HCCL_CHECK(HcclCommDestroy(hcclComm));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    printf("[rank %d] done\n", rankId);
    return 0;
}
