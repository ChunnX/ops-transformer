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
 * \file test_moe_distribute_combine_ccu.cpp
 * \brief A6 CCU MoeDistributeCombineCcuSetup 独立验证用例（v21：去 teardown，纯 0.5RTT 通信验证）。
 *
 * 与 dispatch 用例完全解耦：combine_setup 的输入由 host 直接 mock（不再依赖 dispatch_teardown 的产物），
 *   只跑 setup（CCU 引擎），无 AIV teardown、无 <<<>>>，整个工程纯 host g++ 即可编译。
 *
 * 覆盖的 0.5RTT 芯片功能（combine_setup.cpp 内，与 dispatch 同一套 API、反向数据 pattern）：
 *   clearallpeersarriveNotify / NotifyRecord+NotifyWait(PreSync) / RecordExpectCount(wishCnt) /
 *   WriteNbWithArriveNotify(跨卡 0.5RTT 写) / WaitallpeersarriveNotify(收齐) / LocalCopyNb(自反自)。
 *
 * combine_setup 语义（反向于 dispatch）：本卡作为 src，把 expandX 的干净 token 按 owner 分段发回 owner 卡的
 *   commWindow[dstByteOff]。按 owner 分段消费：段 r 跑 cnt[r] 次（negCnt[r]=-cnt[r] 倒计数，CCU 无减法故由
 *   host 预算负值），每次读 assistInfo[t]=dstByteOff，self 段走 LocalCopyNb、cross 段走 WriteNbWithArriveNotify。
 *
 * host mock（让 R 卡对称、可校验）：每卡给每个 owner r 发 cnt=bs 个 token，A=R*bs。
 *   - expandX[owner r 段][m]   : data 全填 byte=(rankId+1) 标识 src=本卡
 *   - assistInfo[r*bs + m]     : dstByteOff = (epRankId*bs + m)*tokenMsgLen
 *                                （本卡发给 owner r 的第 m 个 → 落在 owner 卡 commWindow 的"本卡专属段"第 m 槽）
 *   - commCmdInfo.sendBytes[r] : cnt*tokenMsgLen ；commCmdInfo.negCnt[r] : (uint64)(0-cnt)
 *   - commWindow               : R*bs*tokenMsgLen 干净槽，memset 0
 *
 * 闭环判据（host 读本卡 commWindow，本卡作为 owner 收齐各 src）：
 *   commWindow[(s*bs + m)*tokenMsgLen] 的 data 全 == (s+1)，for s∈[0,R), m∈[0,bs)。
 *   全命中 → 反向 0.5RTT 跨卡写 + arrive notify + 收齐同步 全部正确。
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
#include "moe_distribute_combine_ccu_setup.h"

// MPI 广播 rootinfo（对齐 oam init_hcclComm）
static bool ExchangeRootInfoMpi(int rankId, int rootRank, HcclRootInfo *rootInfo)
{
    if (rankId == rootRank) {
        if (HcclGetRootInfo(rootInfo) != HCCL_SUCCESS) { return false; }
    }
    MPI_Bcast(rootInfo->internal, HCCL_ROOT_INFO_BYTES, MPI_CHAR, rootRank, MPI_COMM_WORLD);
    return true;
}

// A6 范式 channel 获取（UBC_CTP + notifyNum=2）
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
    // ---- MPI 多进程 ----
    MPI_Init(&argc, &argv);
    int rankId = 0, rankSize = 2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
    const int rootRank = 0;
    const int32_t deviceId = rankId;

    // ---- 验证参数（BF16 非量化）----
    const uint32_t bs            = 8;
    const uint32_t hiddenSize    = 256;
    const uint32_t topK          = 2;
    const uint32_t moeExpertNum  = 8;
    const uint32_t commQuantMode = 0;        // 0=BF16
    const uint32_t cnt           = bs;       // 本卡发给每个 owner 的 token 数（mock：每段固定 bs 个）
    const uint32_t A             = static_cast<uint32_t>(rankSize) * cnt;   // 本卡 expandX 总 token 数

    printf("[rank %d] combine_setup CCU 0.5RTT 验证 start (device=%d, R=%d, bs=%u, H=%u, cnt/owner=%u, A=%u)\n",
           rankId, deviceId, rankSize, bs, hiddenSize, cnt, A);

    // ---- ACL + Stream ----
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ACL_CHECK(aclrtSetStreamFailureMode(stream, 1));

    // ---- 通信域 ----
    HcclRootInfo rootInfo;
    if (!ExchangeRootInfoMpi(rankId, rootRank, &rootInfo)) { printf("[rank %d][ERROR] rootInfo 交换失败\n", rankId); return -1; }
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 200;
    HcclComm hcclComm = nullptr;
    HCCL_CHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &commConfig, &hcclComm));

    // ---- thread + channel ----
    ThreadHandle threadHandle = 0;
    HCCL_CHECK(HcclThreadAcquireWithStream(hcclComm, CommEngine::COMM_ENGINE_CCU, stream, 1, &threadHandle));

    const uint32_t channelNum = static_cast<uint32_t>(rankSize) - 1;
    std::vector<ChannelHandle> channels(channelNum);
    for (uint32_t k = 0; k < channelNum; ++k) {
        uint32_t dstRank = (k < static_cast<uint32_t>(rankId)) ? k : k + 1;
        HCCL_CHECK(HcclGetOneChannel(hcclComm, CommEngine::COMM_ENGINE_CCU,
                                     static_cast<uint32_t>(rankId), dstRank, &channels[k]));
    }

    // ---- 注册 combine_setup CCU kernel ----
    auto setupArg = std::make_shared<hcomm::CcuKernelArgMoeDistributeCombineCcuSetup>(
        rankSize, rankId, moeExpertNum, bs, topK, hiddenSize, commQuantMode);
    setupArg->channels = channels;
    hcomm::KernelCreator setupCreator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<hcomm::CcuKernelMoeDistributeCombineCcuSetup>(arg);
    };
    CcuKernelHandle setupKernel = 0;
    HCCL_CHECK(HcclCcuKernelRegister(hcclComm, &setupKernel,
                                      static_cast<void *>(&setupCreator),
                                      static_cast<void *>(setupArg.get())));
    HCCL_CHECK(HcclCcuKernelRegisterFinish(hcclComm));

    // ---- 尺寸推导（与 combine_setup.cpp 公式严格一致；BF16 干净槽 tokenMsgLen=Align256(H)*2）----
    auto align256 = [](uint32_t n) -> uint32_t { return ((n + 255U) / 256U) * 256U; };
    const size_t tokenMsgLen   = static_cast<size_t>(align256(hiddenSize)) * sizeof(uint16_t);
    const size_t bytesPerToken = static_cast<size_t>(hiddenSize) * sizeof(uint16_t);

    // ---- buffer 字节数 ----
    const size_t expandXBytes    = static_cast<size_t>(A) * tokenMsgLen;                  // 本卡反向数据源（干净，无 metadata）
    const size_t assistInfoBytes = static_cast<size_t>(A) * sizeof(int64_t);             // 每 token dstByteOff（int64：LoadVariable 8B）
    const size_t commWindowBytes = static_cast<size_t>(rankSize) * cnt * tokenMsgLen;    // 本卡作为 owner：R 个 src × cnt 槽
    const size_t commCmdInfoBytes= static_cast<size_t>(rankSize) * 64;                   // ready + sendBytes + negCnt（远大于所需，无害）
    const size_t oneBufBytes     = static_cast<size_t>(rankSize) * sizeof(int32_t);

    auto devAlloc = [](size_t bytes) -> void * {
        void *p = nullptr;
        aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (p != nullptr) aclrtMemset(p, bytes, 0, bytes);
        return p;
    };
    void *expandXDev     = devAlloc(expandXBytes);
    void *assistInfoDev  = devAlloc(assistInfoBytes);
    void *commWindowDev  = devAlloc(commWindowBytes);     // 接收 window（干净），host 校验源
    void *combCmdInfoDev = devAlloc(commCmdInfoBytes);
    void *oneBufDev      = devAlloc(oneBufBytes);

    ACL_CHECK(aclrtMemset(commWindowDev, commWindowBytes, 0, commWindowBytes));   // 未收到的槽保持 0
    {
        std::vector<int32_t> hostOne(rankSize, 1);
        ACL_CHECK(aclrtMemcpy(oneBufDev, oneBufBytes, hostOne.data(), oneBufBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // ---- mock expandX：data 段全填 (rankId+1) 标识 src=本卡 ----
    std::vector<uint8_t> hostExpandX(expandXBytes, 0);
    for (uint32_t t = 0; t < A; ++t) {
        std::memset(hostExpandX.data() + static_cast<size_t>(t) * tokenMsgLen,
                    static_cast<uint8_t>(rankId + 1), bytesPerToken);
    }
    ACL_CHECK(aclrtMemcpy(expandXDev, expandXBytes, hostExpandX.data(), expandXBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- mock assistInfo：按 owner 分段，每段 cnt 个；本卡发给 owner r 的第 m 个 → dstByteOff=(epRankId*bs+m)*tokenMsgLen ----
    //   owner 卡 commWindow 布局 = [src0 的 cnt 槽][src1 的 cnt 槽]…；本卡(src=rankId)落在第 rankId 段第 m 槽。
    std::vector<int64_t> hostAssist(A);
    for (uint32_t r = 0; r < static_cast<uint32_t>(rankSize); ++r) {
        for (uint32_t m = 0; m < cnt; ++m) {
            uint32_t t = r * cnt + m;                                       // 全局紧密 idx（与 expandX 同序）
            hostAssist[t] = static_cast<int64_t>((static_cast<uint32_t>(rankId) * cnt + m) * tokenMsgLen);
        }
    }
    ACL_CHECK(aclrtMemcpy(assistInfoDev, assistInfoBytes, hostAssist.data(), assistInfoBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- mock commCmdInfo：sendBytes[r]=cnt*tokenMsgLen，negCnt[r]=(uint64)(0-cnt) ----
    //   布局（与 combine_setup.cpp 段 5 严格一致）：ready[0..R) int32 │ 8 对齐 │ sendBytes[0..R) int64 │ negCnt[0..R) int64
    const uint64_t cinfoByteOff = ((static_cast<uint64_t>(rankSize) * sizeof(int32_t) + 7U) / 8U) * 8U;
    {
        std::vector<uint8_t> hostCmd(commCmdInfoBytes, 0);
        int64_t* sendBytes = reinterpret_cast<int64_t*>(hostCmd.data() + cinfoByteOff);
        int64_t* negCnt    = sendBytes + rankSize;
        for (uint32_t r = 0; r < static_cast<uint32_t>(rankSize); ++r) {
            sendBytes[r] = static_cast<int64_t>(static_cast<uint64_t>(cnt) * tokenMsgLen);
            negCnt[r]    = static_cast<int64_t>(static_cast<uint64_t>(0) - static_cast<uint64_t>(cnt));  // = -cnt（倒计数初值）
        }
        ACL_CHECK(aclrtMemcpy(combCmdInfoDev, commCmdInfoBytes, hostCmd.data(), commCmdInfoBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // ---- token ----
    const uint64_t expandXToken    = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(expandXDev),     static_cast<uint64_t>(expandXBytes));
    const uint64_t commWindowToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(commWindowDev),  static_cast<uint64_t>(commWindowBytes));
    const uint64_t combCmdInfoToken= hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(combCmdInfoDev), static_cast<uint64_t>(commCmdInfoBytes));
    const uint64_t oneBufToken     = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(oneBufDev),      static_cast<uint64_t>(oneBufBytes));

    // ---- TaskArg + Launch combine_setup ----
    //   TaskArg 字段序：expandX, assistInfo, commCmdInfo, commWindow, expandXToken, commWindowToken, commCmdInfoToken, oneBuf, oneBufToken。
    auto setupTask = std::make_unique<hcomm::CcuTaskArgMoeDistributeCombineCcuSetup>(
        reinterpret_cast<uint64_t>(expandXDev),
        reinterpret_cast<uint64_t>(assistInfoDev),
        reinterpret_cast<uint64_t>(combCmdInfoDev),
        reinterpret_cast<uint64_t>(commWindowDev),
        expandXToken, commWindowToken, combCmdInfoToken,
        reinterpret_cast<uint64_t>(oneBufDev), oneBufToken);

    HCCL_CHECK(HcclCcuKernelLaunch(hcclComm, threadHandle, setupKernel, static_cast<void *>(setupTask.get())));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[rank %d] combine_setup launched + synced\n", rankId);

    // ==================== 校验：host 读本卡 commWindow（本卡作为 owner）====================
    //   各 src 卡 s 发给本卡的第 m 个 token 落在 commWindow[(s*bs + m)*tokenMsgLen]，data 全 == (s+1)。
    std::vector<uint8_t> hostWin(commWindowBytes);
    ACL_CHECK(aclrtMemcpy(hostWin.data(), commWindowBytes, commWindowDev, commWindowBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = true;
    uint32_t failS = 0, failM = 0;
    for (uint32_t s = 0; s < static_cast<uint32_t>(rankSize) && pass; ++s) {
        for (uint32_t m = 0; m < cnt && pass; ++m) {
            const uint8_t* slot = hostWin.data() + (static_cast<size_t>(s) * cnt + m) * tokenMsgLen;
            const uint8_t want = static_cast<uint8_t>(s + 1);
            for (size_t off = 0; off < bytesPerToken; ++off) {
                if (slot[off] != want) { pass = false; failS = s; failM = m; break; }
            }
        }
    }
    if (pass) {
        printf("[rank %d] PASS: combine_setup 0.5RTT 通信验证通过——本卡 commWindow 各 src 段(每段 %u token)"
               " data 全部正确\n", rankId, cnt);
    } else {
        printf("[rank %d] FAIL: commWindow 校验失败 src=%u m=%u（期望 data==%u）\n",
               rankId, failS, failM, failS + 1);
    }

    // ---- 释放 ----
    aclrtFree(expandXDev); aclrtFree(assistInfoDev); aclrtFree(commWindowDev);
    aclrtFree(combCmdInfoDev); aclrtFree(oneBufDev);
    HCCL_CHECK(HcclCommDestroy(hcclComm));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    printf("[rank %d] done\n", rankId);
    MPI_Finalize();
    return pass ? 0 : 1;
}
