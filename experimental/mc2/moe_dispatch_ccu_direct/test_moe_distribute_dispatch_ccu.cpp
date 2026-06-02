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
 * \brief A6 CCU MoeDistributeDispatchCcuSetup 独立验证用例（v21：去 teardown，纯 0.5RTT 通信验证）。
 *
 * 目标：验证 A6 CCU 0.5RTT 通信接口功能正确，端到端闭环——只跑 setup（CCU 引擎），
 *   不要 AIV teardown、不要 <<<>>>，因此整个工程纯 host g++ 即可编译（与 hcomm-test 同形态）。
 *
 * 覆盖的 0.5RTT 芯片功能（dispatch_setup.cpp 内）：
 *   clearallpeersarriveNotify / NotifyRecord+NotifyWait(PreSync 换对端地址+token) /
 *   RecordExpectCount(wishCnt 字节对账) / WriteNbWithArriveNotify(跨卡 0.5RTT 写) /
 *   WaitallpeersarriveNotify(收齐同步) / LocalCopyNb(自发自本地搬运)。
 *
 * 调用范式（严格遵守 hcomm-test/ccu_kernel_test + oam-tools/hccl_test）：
 *   MPI 多进程(mpirun) → comm(HcclGetRootInfo + MPI_Bcast) → channel(RankGraphGetLinks + UBC_CTP + notifyNum=2)
 *   → thread(HcclThreadAcquireWithStream) → register dispatch_setup CCU → Launch → host 校验 commWindow。
 *
 * 闭环判据（host 直接读本卡 commWindow）：
 *   本卡 commWindow 分 R 段（dispatch_setup 段 4：mySlotBase 用 srcRank 段偏移），第 s 段 = src 卡 s 发来的 token。
 *   每卡 token data 全填 byte=(rankId+1)、metadata.srcRank=rankId（int64）。
 *   故本卡 commWindow[s 段] 前 cnt 槽：data 全 == (s+1) 且 metadata.srcRank == s
 *     （cnt = routing 里 dst==本卡 的 token 数，各卡 routing 相同）。
 *   全命中 → 0.5RTT 跨卡写 + arrive notify + 收齐同步 全部正确。
 *
 * 编译/运行：A5 g++ 可编译骨架，但链接/运行需 A6（0.5RTT 上层 API 由 A6 hcomm 提供）。
 */

#include <acl/acl.h>
#include <mpi.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "hccl/hccl_types.h"        // HcclRootInfo / HCCL_ROOT_INFO_BYTES
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"   // RankGraphGetLinks + CommLink/EndpointDesc
#include "hcomm/ccu/ccu_kernel.h"
#include "hcomm/ccu/hccl_ccu_res.h"
#include "moe_distribute_dispatch_ccu_setup.h"

// MPI 广播 rootinfo：root rank 取 rootinfo，MPI_Bcast 给所有 rank（对齐 oam init_hcclComm）
static bool ExchangeRootInfoMpi(int rankId, int rootRank, HcclRootInfo *rootInfo)
{
    if (rankId == rootRank) {
        if (HcclGetRootInfo(rootInfo) != HCCL_SUCCESS) { return false; }
    }
    MPI_Bcast(rootInfo->internal, HCCL_ROOT_INFO_BYTES, MPI_CHAR, rootRank, MPI_COMM_WORLD);
    return true;
}

// A6 范式 channel 获取：本卡 srcRank → 对端 dstRank，从 RankGraph link 填 endpoint（UBC_CTP + notifyNum=2）
static HcclResult HcclGetOneChannel(HcclComm comm, CommEngine engine,
    uint32_t srcRank, uint32_t dstRank, ChannelHandle *channel)
{
    uint32_t netLayer = 0, listSize = 0;
    CommLink *linkList = nullptr;
    HcclResult r = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize);
    if (r != HCCL_SUCCESS) { return r; }

    HcclChannelDesc desc;
    HcclChannelDescInit(&desc, 1);
    const CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    bool found = false;
    for (uint32_t idx = 0; idx < listSize; ++idx) {
        CommLink link = linkList[idx];
        if (link.linkAttr.linkProtocol == protocol) {
            desc.remoteRank             = dstRank;
            desc.notifyNum              = 2;
            desc.channelProtocol        = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc      = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol= link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr= link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc     = link.dstEndpointDesc.loc;
            found = true;
            break;
        }
    }
    if (!found) {
        printf("[ERROR] no UBC_CTP link between rank %u and %u\n", srcRank, dstRank);
        return HCCL_E_NOT_FOUND;
    }
    return HcclChannelAcquire(comm, engine, &desc, 1, channel);
}

#define ACL_CHECK(expr) do { aclError _r = (expr); if (_r != ACL_SUCCESS) { printf("[ERROR] " #expr " ret=%d\n", (int)_r); return -1; } } while (0)
#define HCCL_CHECK(expr) do { HcclResult _r = (expr); if (_r != HCCL_SUCCESS) { printf("[ERROR] " #expr " ret=%d\n", (int)_r); return -1; } } while (0)

int main(int argc, char *argv[])
{
    // ---- MPI 多进程：rank/rankSize 从 MPI 取（mpirun -n N 拉起），deviceId = rank ----
    MPI_Init(&argc, &argv);
    int rankId = 0, rankSize = 2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
    const int rootRank = 0;
    const int32_t deviceId = rankId;   // 2p demo：rank==device（多卡可改 rankId % npus）

    // ---- 验证参数（小尺寸便于校验 + 快速；BF16 非量化）----
    const uint32_t bs            = 8;
    const uint32_t hiddenSize    = 256;
    const uint32_t topK          = 2;
    const uint32_t moeExpertNum  = 8;       // 须能被 rankSize 整除（moeExpertPerRankNum = moeExpertNum/R）
    const uint32_t sharedExpertRankNum = 0;
    const uint32_t quantMode           = 0;  // 0=BF16
    const uint32_t moeExpertPerRankNum = moeExpertNum / (rankSize - sharedExpertRankNum);

    printf("[rank %d] dispatch_setup CCU 0.5RTT 验证 start (device=%d, R=%d, bs=%u, H=%u, topK=%u, "
           "moeExpertNum=%u, perRank=%u)\n",
           rankId, deviceId, rankSize, bs, hiddenSize, topK, moeExpertNum, moeExpertPerRankNum);

    // ---- ACL + Stream ----
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtSetStreamFailureMode(stream, 1));   // 遇错即停（对齐 oam device_init，A6 调试友好）

    // ---- 通信域（rootinfo MPI_Bcast）----
    HcclRootInfo rootInfo;
    if (!ExchangeRootInfoMpi(rankId, rootRank, &rootInfo)) { printf("[rank %d][ERROR] rootInfo 交换失败\n", rankId); return -1; }
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 200;
    HcclComm hcclComm = nullptr;
    HCCL_CHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &commConfig, &hcclComm));

    // ---- thread 绑 stream（A6 范式）----
    ThreadHandle threadHandle = 0;
    HCCL_CHECK(HcclThreadAcquireWithStream(hcclComm, CommEngine::COMM_ENGINE_CCU, stream, 1, &threadHandle));

    // ---- channel：逐对端 RankGraphGetLinks 填 endpoint（UBC_CTP + notifyNum=2）----
    //   channel 序与 kernel channelIdx=(r<self)?r:r-1 一致：channels[k] → dstRank=(k<rankId)?k:k+1。
    const uint32_t channelNum = static_cast<uint32_t>(rankSize) - 1;
    std::vector<ChannelHandle> channels(channelNum);
    for (uint32_t k = 0; k < channelNum; ++k) {
        uint32_t dstRank = (k < static_cast<uint32_t>(rankId)) ? k : k + 1;
        HCCL_CHECK(HcclGetOneChannel(hcclComm, CommEngine::COMM_ENGINE_CCU,
                                     static_cast<uint32_t>(rankId), dstRank, &channels[k]));
    }

    // ---- 注册 dispatch_setup CCU kernel ----
    auto setupArg = std::make_shared<hcomm::CcuKernelArgMoeDistributeDispatchCcuSetup>(
        rankSize, rankId, moeExpertNum, bs, topK, hiddenSize, quantMode);
    setupArg->channels = channels;
    hcomm::KernelCreator setupCreator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<hcomm::CcuKernelMoeDistributeDispatchCcuSetup>(arg);
    };
    CcuKernelHandle setupKernel = 0;
    HCCL_CHECK(HcclCcuKernelRegister(hcclComm, &setupKernel,
                                      static_cast<void *>(&setupCreator),
                                      static_cast<void *>(setupArg.get())));
    HCCL_CHECK(HcclCcuKernelRegisterFinish(hcclComm));

    // ---- 尺寸推导（与 dispatch_setup.cpp 公式严格一致）----
    auto align256 = [](uint32_t n) -> uint32_t { return ((n + 255U) / 256U) * 256U; };
    const size_t   tokenMsgSizeBytes     = static_cast<size_t>(align256(hiddenSize)) * sizeof(uint16_t);  // BF16: Align256(H)*2
    const uint64_t tokenWithMetaLenBytes = tokenMsgSizeBytes + 32;     // +32B metadata（4×int64）
    const size_t   bytesPerToken         = static_cast<size_t>(hiddenSize) * sizeof(uint16_t);            // 真实 BF16 数据段

    // ---- buffer 字节数（只留 dispatch_setup 实际读写的）----
    const size_t expertIdsBytes   = static_cast<size_t>(bs) * topK * sizeof(int64_t);   // TaskArg 占位（device 不读）
    const size_t commWindowBytes  = static_cast<size_t>(rankSize) * bs * topK * tokenWithMetaLenBytes;
    const size_t xWithMetaBytes   = static_cast<size_t>(bs) * topK * tokenWithMetaLenBytes;
    const size_t routingTableBytes= static_cast<size_t>(bs) * topK * sizeof(int64_t);   // host 预算路由 LUT（int64：LoadVariable 8B）
    const size_t commCmdInfoBytes = static_cast<size_t>(rankSize) * 64;                 // 仅 ready 区用（远大于 R*4，无害）
    const size_t oneBufBytes      = static_cast<size_t>(rankSize) * sizeof(int32_t);    // ready 写源（R 个 int32=1）

    auto devAlloc = [](size_t bytes) -> void * {
        void *p = nullptr;
        aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (p != nullptr) aclrtMemset(p, bytes, 0, bytes);
        return p;
    };
    void *expertIdsDev    = devAlloc(expertIdsBytes);
    void *commWindowDev   = devAlloc(commWindowBytes);     // 接收 window（含 metadata），host 校验源
    void *xWithMetaDev    = devAlloc(xWithMetaBytes);      // host 预拼装 x+metadata（数据源）
    void *routingTableDev = devAlloc(routingTableBytes);
    void *commCmdInfoDev  = devAlloc(commCmdInfoBytes);
    void *oneBufDev       = devAlloc(oneBufBytes);

    // commWindow 清 0：未收到的槽保持 0，与 token pattern(=s+1≥1) 可区分，便于定位漏收。
    ACL_CHECK(aclrtMemset(commWindowDev, commWindowBytes, 0, commWindowBytes));
    // oneBuf：R 个 int32=1（必须 memcpy，不能 memset 1——memset 按字节填会得 0x01010101）。
    {
        std::vector<int32_t> hostOne(rankSize, 1);
        ACL_CHECK(aclrtMemcpy(oneBufDev, oneBufBytes, hostOne.data(), oneBufBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // ---- 路由表（各卡相同的确定性路由）+ expertIds 占位 ----
    std::vector<int64_t> hostExpertIds(bs * topK);
    std::vector<int64_t> hostRoutingTable(bs * topK);
    for (uint32_t i = 0; i < bs; ++i) {
        for (uint32_t j = 0; j < topK; ++j) {
            uint32_t idx = i * topK + j;
            hostExpertIds[idx]    = idx % moeExpertNum;
            hostRoutingTable[idx] = hostExpertIds[idx] / moeExpertPerRankNum;   // dst rank ∈ [0, R)
        }
    }
    ACL_CHECK(aclrtMemcpy(expertIdsDev,    expertIdsBytes,    hostExpertIds.data(),    expertIdsBytes,    ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(routingTableDev, routingTableBytes, hostRoutingTable.data(), routingTableBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- 数据源 xWithMeta：data 段全填 (rankId+1) 标识发送方；metadata 4×int64 ----
    std::vector<uint8_t> hostXWithMeta(xWithMetaBytes, 0);
    for (uint32_t i = 0; i < bs; ++i) {
        for (uint32_t j = 0; j < topK; ++j) {
            uint32_t idx = i * topK + j;
            uint8_t* rowBase = hostXWithMeta.data() + idx * tokenWithMetaLenBytes;
            std::memset(rowBase, static_cast<uint8_t>(rankId + 1), bytesPerToken);   // data 段标识 srcRank
            int64_t* meta = reinterpret_cast<int64_t*>(rowBase + tokenMsgSizeBytes);
            meta[0] = static_cast<int64_t>(rankId);                       // srcRank
            meta[1] = static_cast<int64_t>(i);                            // srcSlot
            meta[2] = static_cast<int64_t>(j);                            // dstK
            meta[3] = static_cast<int64_t>(idx * tokenMsgSizeBytes);      // dstByteOff（combine 干净槽偏移，本用例不消费）
        }
    }
    ACL_CHECK(aclrtMemcpy(xWithMetaDev, xWithMetaBytes, hostXWithMeta.data(), xWithMetaBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    if (rankId == 0) {
        std::vector<uint32_t> rankHit(rankSize, 0);
        for (uint32_t idx = 0; idx < bs * topK; ++idx) ++rankHit[hostRoutingTable[idx]];
        printf("[rank 0] routing histogram:");
        for (int r = 0; r < rankSize; ++r) printf(" rank%d=%u", r, rankHit[r]);
        printf("\n");
    }

    // ---- token（host GetTokenInfo(addr,size)：device 访存鉴权凭证）----
    const uint64_t xWithMetaToken  = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(xWithMetaDev),  static_cast<uint64_t>(xWithMetaBytes));
    const uint64_t commWindowToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(commWindowDev), static_cast<uint64_t>(commWindowBytes));
    const uint64_t commCmdInfoToken= hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(commCmdInfoDev),static_cast<uint64_t>(commCmdInfoBytes));
    const uint64_t oneBufToken     = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(oneBufDev),     static_cast<uint64_t>(oneBufBytes));

    // ---- TaskArg + Launch dispatch_setup（CCU 0.5RTT 通信）----
    //   TaskArg 字段序：expertIds, commCmdInfo, commWindow, routingTable, xWithMeta,
    //                   xWithMetaToken, commWindowToken, commCmdInfoToken, oneBuf, oneBufToken。
    auto setupTask = std::make_unique<hcomm::CcuTaskArgMoeDistributeDispatchCcuSetup>(
        reinterpret_cast<uint64_t>(expertIdsDev),
        reinterpret_cast<uint64_t>(commCmdInfoDev),
        reinterpret_cast<uint64_t>(commWindowDev),
        reinterpret_cast<uint64_t>(routingTableDev),
        reinterpret_cast<uint64_t>(xWithMetaDev),
        xWithMetaToken, commWindowToken, commCmdInfoToken,
        reinterpret_cast<uint64_t>(oneBufDev), oneBufToken);

    HCCL_CHECK(HcclCcuKernelLaunch(hcclComm, threadHandle, setupKernel, static_cast<void *>(setupTask.get())));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[rank %d] dispatch_setup launched + synced\n", rankId);

    // ==================== 校验：host 读本卡 commWindow ====================
    //   本卡应从每个 src 卡 s 收到 cnt 个 token（cnt = routing 里 dst==本卡 的数；各卡 routing 相同）。
    uint32_t cntPerSrc = 0;
    for (uint32_t idx = 0; idx < bs * topK; ++idx)
        if (static_cast<uint32_t>(hostRoutingTable[idx]) == static_cast<uint32_t>(rankId)) ++cntPerSrc;

    std::vector<uint8_t> hostWin(commWindowBytes);
    ACL_CHECK(aclrtMemcpy(hostWin.data(), commWindowBytes, commWindowDev, commWindowBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = true;
    uint32_t failS = 0, failM = 0;
    const size_t segBytes = static_cast<size_t>(bs) * topK * tokenWithMetaLenBytes;   // 每 src 段字节
    for (uint32_t s = 0; s < static_cast<uint32_t>(rankSize) && pass; ++s) {
        for (uint32_t m = 0; m < cntPerSrc && pass; ++m) {
            const uint8_t* slot = hostWin.data() + s * segBytes + static_cast<size_t>(m) * tokenWithMetaLenBytes;
            const uint8_t want = static_cast<uint8_t>(s + 1);
            for (size_t off = 0; off < bytesPerToken; ++off) {
                if (slot[off] != want) { pass = false; failS = s; failM = m; break; }
            }
            if (!pass) break;
            int64_t srcRank = *reinterpret_cast<const int64_t*>(slot + tokenMsgSizeBytes);
            if (srcRank != static_cast<int64_t>(s)) { pass = false; failS = s; failM = m; break; }
        }
    }
    if (pass) {
        printf("[rank %d] PASS: dispatch_setup 0.5RTT 通信验证通过——本卡 commWindow 各 src 段(每段 %u token)"
               " data 与 metadata.srcRank 全部正确\n", rankId, cntPerSrc);
    } else {
        printf("[rank %d] FAIL: commWindow 校验失败 src=%u m=%u（期望 data==%u, srcRank==%u）\n",
               rankId, failS, failM, failS + 1, failS);
    }

    // ---- 释放 ----
    aclrtFree(expertIdsDev); aclrtFree(commWindowDev); aclrtFree(xWithMetaDev);
    aclrtFree(routingTableDev); aclrtFree(commCmdInfoDev); aclrtFree(oneBufDev);
    HCCL_CHECK(HcclCommDestroy(hcclComm));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    printf("[rank %d] done\n", rankId);
    MPI_Finalize();
    return pass ? 0 : 1;
}
