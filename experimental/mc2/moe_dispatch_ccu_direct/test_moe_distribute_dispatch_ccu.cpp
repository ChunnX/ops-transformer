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

    // ---- 参数 ----
    const std::string groupEp    = "epgroup_v4";            // 仅 host 日志用
    const uint32_t bs            = 8;
    const uint32_t hiddenSize    = 7168;
    const uint32_t topK          = 8;
    const uint32_t moeExpertNum  = 32;
    const uint32_t sharedExpertRankNum = 0;                  // moeExpertPerRankNum 分母
    const uint32_t quantMode           = 0;                  // 0=BF16, 1=MXFP8

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
        rankSize, rankId, moeExpertNum, bs, topK, hiddenSize, quantMode);
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
    auto align32  = [](uint32_t n) -> uint32_t { return ((n + 31U)  / 32U)  * 32U;  };

    // v17 Phase 2 ★ MXFP8 模式 H 必须是 32 的倍数（block 对齐）
    //   dispatch single driver 不跑 combine_teardown，无 64 倍数约束（那个是 BF16 视图整除约束）
    //   但 MXFP8 block size = 32 → H 至少要 32 的倍数才能正确分块
    if (quantMode == 1 && (hiddenSize % 32U != 0)) {
        printf("[rank %d][ERROR] MXFP8 模式要求 hiddenSize 是 32 的倍数（block 对齐），"
               "当前 H=%u\n", rankId, hiddenSize);
        return -1;
    }

    const uint32_t localExpertNum         = moeExpertPerRankNum;
    // v17 ★ tokenMsgLen 按 quantMode 切（与 dispatch_setup.cpp constructor 公式严格一致）
    //   quantMode=0 (BF16): Align256(H) * 2
    //   quantMode=1 (MXFP8): Align32(H + H/32) —— FP8 data + E8M0 scale
    const size_t   tokenMsgSizeBytes      = (quantMode == 0)
        ? static_cast<size_t>(align256(hiddenSize)) * sizeof(uint16_t)
        : static_cast<size_t>(align32(hiddenSize + hiddenSize / 32U));
    const uint64_t tokenWithMetaLenBytes  = tokenMsgSizeBytes + 32;     // v15 ★ +32B metadata
    const size_t   A                      = static_cast<size_t>(bs) * rankSize
                                            * static_cast<size_t>(localExpertNum < topK ? localExpertNum : topK);

    // ---- buffer 字节数（只留 device 实际读写的）----
    const size_t xBytes               = static_cast<size_t>(bs) * hiddenSize * sizeof(uint16_t);  // hostX 用（构造 xWithMeta）
    const size_t expertIdsBytes       = static_cast<size_t>(bs) * topK * sizeof(int64_t);         // ★ int64：CCU LoadVariable 读 8 字节
    const size_t commWindowBytes      = static_cast<size_t>(rankSize) * bs * topK * tokenWithMetaLenBytes;
    const size_t xWithMetaBytes       = static_cast<size_t>(bs) * topK * tokenWithMetaLenBytes;
    const size_t expandIdxBytes       = static_cast<size_t>(bs) * topK * sizeof(int64_t);         // op 输出占位（v20.6 CCU 不再写，host 不读，置零即可）
    const size_t commCmdInfoBytes     = (static_cast<size_t>(bs) * topK
                                         + static_cast<size_t>(rankSize) * localExpertNum)
                                        * 16 * sizeof(int32_t);   // 远大于 v20.6 布局所需（ready+sendBytes+negCnt），无害
    const size_t expandXBytes         = A * tokenMsgSizeBytes;       // v20.6 ★ 干净（无 metadata），A 个 tokenMsgLen
    const size_t assistInfoBytes      = A * 1 * sizeof(int64_t);     // v20.7 ★ 每 token 单字段 dstByteOff（owner 由 combine 分段隐含）
    const size_t routingTableBytes    = static_cast<size_t>(bs) * topK * sizeof(int64_t);   // 内部 scratch ★ int64：CCU LoadVariable 读 8 字节

    void *expertIdsDev       = devAlloc(expertIdsBytes);
    void *commWindowDev      = devAlloc(commWindowBytes);        // 接收 window（含 metadata）
    void *xWithMetaDev       = devAlloc(xWithMetaBytes);         // host 预拼装 x+metadata
    void *expandIdxDev       = devAlloc(expandIdxBytes);
    void *commCmdInfoDev     = devAlloc(commCmdInfoBytes);

    // 收口 ★ commWindow 空 slot 初始化为 0xFF sentinel（metadata.srcRank = -1）。
    //   dispatch_teardown 收口后【自数】有效 token（按 metadata.srcRank==r 判定），空 slot 必须用
    //   sentinel 与"rank 0 真实发来的 token (srcRank=0)"区分；devAlloc 默认 memset 0 不够，显式覆 0xFF。
    ACL_CHECK(aclrtMemset(commWindowDev, commWindowBytes, 0xFF, commWindowBytes));

    // 收口 ★ commCmdInfo 简化为只剩 ready 区，从偏移 0 起（删了 tokenCount/scratch 区）。
    //   teardown 轮询 ready 区 [0,R) 全 1 → 数据齐。alloc 大小保持不变（无害，远大于 R）。
    //   devAlloc 已整块 memset 0，ready 区初始 0（未就绪）；这里显式重申 ready 区清零。
    const size_t readyRegionByteOff = 0;        // 收口 ★ ready 区从偏移 0 起（与 setup/teardown 一致）
    const size_t readyRegionBytes   = static_cast<size_t>(rankSize) * sizeof(int32_t);
    ACL_CHECK(aclrtMemset(static_cast<uint8_t*>(commCmdInfoDev) + readyRegionByteOff,
                          readyRegionBytes, 0, readyRegionBytes));

    // 收口 ★ oneBuf：ready 写源（host 填 R 个 int32=1）。setup 段 10 用 1 次 LocalCopyNb 整块拷到
    //   commCmdInfo ready 区。⚠️ 必须用 aclrtMemcpy 从 host vector(R,1) 拷——不能用 aclrtMemset 填 1，
    //   memset 按字节填 → int32 会变成 0x01010101 而非 1。
    const size_t oneBufBytes = static_cast<size_t>(rankSize) * sizeof(int32_t);
    void *oneBufDev          = devAlloc(oneBufBytes);
    {
        std::vector<int32_t> hostOne(rankSize, 1);
        ACL_CHECK(aclrtMemcpy(oneBufDev, oneBufBytes, hostOne.data(), oneBufBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void *expandXDev         = devAlloc(expandXBytes);
    void *assistInfoDev      = devAlloc(assistInfoBytes);
    void *routingTableDev    = devAlloc(routingTableBytes);

    // ---- 测试数据 ----
    std::vector<uint8_t> hostX(xBytes, 0x01);   // 仅用于构造 xWithMeta（dispatch_setup 读 xWithMeta）

    std::vector<int64_t> hostExpertIds(bs * topK);      // ★ int64：device 端 LoadVariable 读 8 字节
    std::vector<int64_t> hostRoutingTable(bs * topK);   // ★ int64：同上
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

    // v17 ★ 构造 xWithMetaDev：按 quantMode 切 data 段格式
    //   quantMode=0 (BF16): rowBase[0..hiddenSize*2) = hostX[i 行] BF16 数据
    //   quantMode=1 (MXFP8): rowBase[0..H) = FP8 data + rowBase[H..H+H/32) = E8M0 scale
    //   metadata 4 个 INT32 写到 rowBase[tokenMsgSize..tokenMsgSize+16)（同 v16）
    const size_t bytesPerToken = static_cast<size_t>(hiddenSize) * sizeof(uint16_t);
    std::vector<uint8_t> hostXWithMeta(xWithMetaBytes, 0);
    for (uint32_t i = 0; i < bs; ++i) {
        for (uint32_t j = 0; j < topK; ++j) {
            uint32_t idx = i * topK + j;
            uint8_t* rowBase = hostXWithMeta.data() + idx * tokenWithMetaLenBytes;

            if (quantMode == 0) {
                std::memcpy(rowBase, hostX.data() + i * bytesPerToken, bytesPerToken);
            } else {
                // v17 ★ MXFP8 mock 量化（dispatch single 路径，与 combine driver 同款）
                const uint8_t* srcBF16 = hostX.data() + i * bytesPerToken;
                uint8_t* dstFP8   = rowBase;
                uint8_t* dstScale = rowBase + hiddenSize;
                for (uint32_t e = 0; e < hiddenSize; ++e) {
                    dstFP8[e] = srcBF16[e * 2 + 1];                // mock: BF16 高字节作 FP8
                }
                for (uint32_t b = 0; b < hiddenSize / 32; ++b) {
                    dstScale[b] = 0x80;                            // mock E8M0 = 2^0
                }
            }

            // ★ metadata 按 int64 排布（每字段 8 字节）：CCU LoadVariable/XN 槽固定 8 字节，
            //   字段必须 8 字节隔开，否则相邻字段被读进高 32 位污染比较（CCU_RESOURCE_XN_PER_SIZE=8）。
            int64_t* meta = reinterpret_cast<int64_t*>(rowBase + tokenMsgSizeBytes);
            meta[0] = static_cast<int64_t>(rankId);                          // srcRank @字段0
            meta[1] = static_cast<int64_t>(i);                               // srcSlot @字段1
            meta[2] = static_cast<int64_t>(j);                               // dstK    @字段2
            meta[3] = static_cast<int64_t>(idx * tokenMsgSizeBytes);         // v20.6 ★ dstByteOff @字段3=(i*K+j)*tokenMsgLen（combine 干净槽偏移）
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

    // ---- v19 ★ 计算 4 个本地 buffer 的 token（hcomm GetTokenInfo(addr, size)）----
    //   token 范式（亲核实 hccl mesh1d + ccu_alg_template_base.cc GetToken）：每块本地 buffer
    //   按 (虚拟地址, 字节数) 调 GetTokenInfo 算 token，传给 device 作访存鉴权凭证。
    //   device 端 LocalAddr/RemoteAddr 的 .token 必须设（见 setup.cpp）。
    const uint64_t xWithMetaToken  = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(xWithMetaDev),  static_cast<uint64_t>(xWithMetaBytes));
    const uint64_t commWindowToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(commWindowDev), static_cast<uint64_t>(commWindowBytes));
    const uint64_t commCmdInfoToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(commCmdInfoDev), static_cast<uint64_t>(commCmdInfoBytes));
    // v20.6 ★ expandIdx 不再由 CCU 写（StoreVariableAt 依赖铲除），故不再算其 token。
    // 收口 ★ oneBuf 独立 token（token 跟随 buffer，单独 GetTokenInfo）
    const uint64_t oneBufToken = hcomm::CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(oneBufDev), static_cast<uint64_t>(oneBufBytes));

    // ---- TaskArg + Launch ----
    auto setupTask = std::make_unique<hcomm::CcuTaskArgMoeDistributeDispatchCcuSetup>(
        reinterpret_cast<uint64_t>(expertIdsDev),
        reinterpret_cast<uint64_t>(commCmdInfoDev),
        reinterpret_cast<uint64_t>(commWindowDev),
        reinterpret_cast<uint64_t>(routingTableDev),
        reinterpret_cast<uint64_t>(xWithMetaDev),
        xWithMetaToken, commWindowToken, commCmdInfoToken,
        reinterpret_cast<uint64_t>(oneBufDev), oneBufToken);   // 收口 ★ ready 写源（末尾追加）

    // v10 ★ teardown tiling buffer 准备（host 填充 + memcpy 到 device）
    MoeDistributeDispatchCcuTeardownTilingData hostTiling = {};
    hostTiling.epWorldSize      = rankSize;
    hostTiling.bs               = bs;
    hostTiling.topK             = topK;
    hostTiling.tokenMsgLen      = tokenMsgSizeBytes;
    hostTiling.tokenWithMetaLen = tokenWithMetaLenBytes;
    void *tilingDev = devAlloc(sizeof(hostTiling));
    ACL_CHECK(aclrtMemcpy(tilingDev, sizeof(hostTiling),
                          &hostTiling, sizeof(hostTiling), ACL_MEMCPY_HOST_TO_DEVICE));

    // ==================== Launch ====================
    // (1) setup: CCU 路径（hcomm）。
    //     v10 setup CCU 段 9 内部已做 WaitallpeersarriveNotify + clearallpeersarriveNotify，
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
        commCmdInfoDev, commWindowDev, expandXDev, assistInfoDev, tilingDev);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    printf("[rank %d] dispatch_setup + dispatch_teardown launched and synced\n", rankId);

    // ---- 释放 ----
    aclrtFree(expertIdsDev); aclrtFree(commWindowDev); aclrtFree(xWithMetaDev);
    aclrtFree(expandIdxDev); aclrtFree(commCmdInfoDev);
    aclrtFree(oneBufDev);                                       // 收口 ★
    aclrtFree(expandXDev); aclrtFree(assistInfoDev);
    aclrtFree(routingTableDev);
    aclrtFree(tilingDev);
    HCCL_CHECK(HcclCommDestroy(hcclComm));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    printf("[rank %d] done\n", rankId);
    return 0;
}
