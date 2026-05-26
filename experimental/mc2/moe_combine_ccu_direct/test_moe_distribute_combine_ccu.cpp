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
 * \brief v12 dispatch + combine 完整链路 demo + 数据闭环校验（双流并行架构）。
 *
 * 流程：
 *   1) host 准备 x（每 token 不同 byte pattern 便于校验）+ expertIds + routingTable + expert_scales
 *   2) Launch dispatch_setup CCU (异步发数据)
 *   3) Launch dispatch_teardown AIV (轮询等齐 + DataCopy y → expand_x)
 *   4) (跳过 expert，identity 模式：expand_x 不动)
 *   5) Launch combine_setup CCU (反向发 expand_x)
 *   6) Launch combine_teardown AIV (轮询等齐 + last-write-wins → xOut)
 *   7) 校验 xOut == x（端到端数据闭环）
 *
 * identity expert + last-write-wins 模式下，xOut 应严格等于原始 x：
 *   - 本卡 token t_i 在 dispatch 中被发到 K 个 rank 的 K 个槽位
 *   - 每个槽位的 expert 处理结果（identity = 原值 t_i）反向回到本卡 yCombine[r 段, slot 槽]
 *   - combine_teardown 按 (i, j) 顺序 last-write-wins 写 xOut[i]，最后一次 j=K-1 写的就是 t_i
 *   - 闭环成立
 *
 * 本驱动在 A6 ascendc 仿真/真机下编译运行；A5 host gcc 下 `<<<>>>` 行不可直接编译。
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
#include "moe_distribute_combine_ccu_setup.h"
#include "moe_distribute_combine_ccu_teardown.h"

static const char *kRootInfoPath = "/tmp/ccu_a6_combine_rootinfo.bin";
static const char *kRootInfoTmp  = "/tmp/ccu_a6_combine_rootinfo.bin.tmp";

static bool ExchangeRootInfo(int32_t rankId, HcclRootInfo *rootInfo) {
    if (rankId == 0) {
        if (HcclGetRootInfo(rootInfo) != HCCL_SUCCESS) return false;
        FILE *fp = fopen(kRootInfoTmp, "wb");
        if (!fp) return false;
        size_t wn = fwrite(rootInfo->internal, 1, HCCL_ROOT_INFO_BYTES, fp);
        fclose(fp);
        if (wn != HCCL_ROOT_INFO_BYTES) return false;
        rename(kRootInfoTmp, kRootInfoPath);
        return true;
    }
    for (int i = 0; i < 600; ++i) {
        if (access(kRootInfoPath, R_OK) == 0) {
            FILE *fp = fopen(kRootInfoPath, "rb");
            if (fp) {
                size_t rn = fread(rootInfo->internal, 1, HCCL_ROOT_INFO_BYTES, fp);
                fclose(fp);
                if (rn == HCCL_ROOT_INFO_BYTES) return true;
            }
        }
        usleep(100 * 1000);
    }
    return false;
}

#define ACL_CHECK(expr) do { aclError _r = (expr); if (_r != ACL_SUCCESS) { printf("[ERROR] " #expr " ret=%d\n", (int)_r); return -1; } } while (0)
#define HCCL_CHECK(expr) do { HcclResult _r = (expr); if (_r != HCCL_SUCCESS) { printf("[ERROR] " #expr " ret=%d\n", (int)_r); return -1; } } while (0)

// v13 ★ 加 GM_ADDR commWindow（与 teardown.h / 实际 .cpp 11 参签名一致）
extern "C" __global__ __aicore__ void moe_distribute_dispatch_ccu_teardown(
    GM_ADDR x, GM_ADDR y, GM_ADDR expertIds, GM_ADDR commCmdInfo,
    GM_ADDR expandX, GM_ADDR dynamicScales,
    GM_ADDR assistInfoForCombine, GM_ADDR expertTokenNums,
    GM_ADDR commWindow,
    GM_ADDR workspace, GM_ADDR tilingGM);

// v16 ★ 12 参（删 yCombine 和 routingTable，加 commWindow 复用 dispatch 同款）
extern "C" __global__ __aicore__ void moe_distribute_combine_ccu_teardown(
    GM_ADDR expandX, GM_ADDR quantExpandX, GM_ADDR expertIds, GM_ADDR expandIdx,
    GM_ADDR expertScales, GM_ADDR commCmdInfo, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
    GM_ADDR xOut,
    GM_ADDR commWindow,
    GM_ADDR workspace, GM_ADDR tilingGM);

int main(int argc, char *argv[]) {
    int32_t rankId = 0, rankSize = 2;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--rank_id") == 0)        rankId   = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--rank_size") == 0) rankSize = atoi(argv[i + 1]);
    }
    const int32_t deviceId = rankId;

    // ---- demo 参数（v12 简化，便于校验）----
    const std::string groupEp = "epgroup_v12";
    const uint32_t bs           = 8;
    const uint32_t hiddenSize   = 256;
    const uint32_t topK         = 2;
    const uint32_t moeExpertNum = 8;       // → moeExpertPerRankNum = 4
    // 简化（与 950 当前支持范围一致 + comm_type=2 是 URMA）
    const uint32_t sharedExpertNum     = 0;
    const uint32_t sharedExpertRankNum = 0;
    const uint32_t quantMode           = 0;      // dispatch
    const uint32_t commQuantMode       = 0;      // combine
    const uint32_t commType            = 2;
    const std::string commAlg          = "";
    const uint32_t expertTokenNumsType = 1;

    const uint32_t localExpertNum     = moeExpertNum / (rankSize - sharedExpertRankNum);
    auto align256 = [](uint32_t n) -> uint32_t { return ((n + 255U) / 256U) * 256U; };
    const uint64_t tokenMsgSizeBytes = static_cast<uint64_t>(align256(hiddenSize)) * sizeof(uint16_t);
    const uint64_t tokenWithMetaLenBytes = tokenMsgSizeBytes + 32;   // v15 ★ +32B metadata 三元组
    const size_t A = static_cast<size_t>(bs) * rankSize * (localExpertNum < topK ? localExpertNum : topK);

    printf("[rank %d] v12 dispatch+combine demo: bs=%u, topK=%u, H=%u, moeExpertNum=%u, "
           "localExpertNum=%u, A=%zu, tokenMsgLen=%lu\n",
           rankId, bs, topK, hiddenSize, moeExpertNum, localExpertNum, A,
           (unsigned long)tokenMsgSizeBytes);

    // ---- ACL + Stream + Comm ----
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
        channelDescs[k].memHandleNum = 0;
    }
    std::vector<ChannelHandle> channels(channelNum);
    HCCL_CHECK(HcclChannelAcquire(hcclComm, CommEngine::COMM_ENGINE_CCU,
                                  channelDescs.data(), channelNum, channels.data()));

    // ==================== 注册 2 个 CCU kernel（setup 只）====================
    // teardown 是 AIV ascendc kernel，通过 `<<<>>>` 直调，不 register
    auto dispatchSetupArg = std::make_shared<hcomm::CcuKernelArgMoeDistributeDispatchCcuSetup>(
        groupEp, rankSize, rankId, moeExpertNum, bs, topK, hiddenSize,
        0, sharedExpertNum, sharedExpertRankNum, quantMode, 0, commType, commAlg);
    dispatchSetupArg->channels = channels;

    auto combineSetupArg = std::make_shared<hcomm::CcuKernelArgMoeDistributeCombineCcuSetup>(
        groupEp, rankSize, rankId, moeExpertNum, bs, topK, hiddenSize,
        0, sharedExpertNum, sharedExpertRankNum, 0, commQuantMode, commType, commAlg);
    combineSetupArg->channels = channels;

    hcomm::KernelCreator dispatchSetupCreator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<hcomm::CcuKernelMoeDistributeDispatchCcuSetup>(arg);
    };
    hcomm::KernelCreator combineSetupCreator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<hcomm::CcuKernelMoeDistributeCombineCcuSetup>(arg);
    };

    CcuKernelHandle dispatchSetupKernel = 0, combineSetupKernel = 0;
    HCCL_CHECK(HcclCcuKernelRegister(hcclComm, &dispatchSetupKernel,
                                      static_cast<void *>(&dispatchSetupCreator),
                                      static_cast<void *>(dispatchSetupArg.get())));
    HCCL_CHECK(HcclCcuKernelRegister(hcclComm, &combineSetupKernel,
                                      static_cast<void *>(&combineSetupCreator),
                                      static_cast<void *>(combineSetupArg.get())));
    HCCL_CHECK(HcclCcuKernelRegisterFinish(hcclComm));

    // ==================== Alloc 所有 buffer ====================
    auto devAlloc = [](size_t bytes) -> void * {
        void *p = nullptr;
        aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (p) aclrtMemset(p, bytes, 0, bytes);
        return p;
    };

    // dispatch op_def Input/Output
    const size_t xBytes               = static_cast<size_t>(bs) * hiddenSize * sizeof(uint16_t);
    const size_t expertIdsBytes       = static_cast<size_t>(bs) * topK * sizeof(uint32_t);
    const size_t scalesBytes          = static_cast<size_t>(sharedExpertNum + moeExpertNum) * hiddenSize * sizeof(float);
    const size_t xActiveMaskBytes     = static_cast<size_t>(bs) * sizeof(uint8_t);
    // v15 ★ buffer 大小公式全面切换到 tokenWithMetaLenBytes（每槽 = data + 32B metadata）
    //   commWindow / expandX / yCombine 三大 buffer 都按 rankSize * bs * topK 槽 × tokenWithMetaLen 算
    //   xWithMeta: host 预拼装 (bs*K, tokenWithMetaLen)，dispatch_setup 段 7 阶段 B 一次 RDMA 把 data+meta 整体送
    //   assistInfoForCombine: 紧密区 sum_r(commCmdInfo[r]) * 3 个 INT32（worst-case rankSize*bs*topK*3*4 字节）
    const size_t opDefYBytes          = static_cast<size_t>(bs) * (topK + sharedExpertNum) * tokenMsgSizeBytes;
    const size_t commWindowBytes      = static_cast<size_t>(rankSize) * bs * topK * tokenWithMetaLenBytes; // v15 ★
    const size_t xWithMetaBytes       = static_cast<size_t>(bs) * topK * tokenWithMetaLenBytes;            // v15 ★
    const size_t yBytes               = opDefYBytes;
    const size_t expandIdxBytes       = static_cast<size_t>(bs) * topK * sizeof(uint32_t);
    const size_t dispCmdInfoBytes     = (static_cast<size_t>(bs) * (topK + sharedExpertNum)
                                         + static_cast<size_t>(rankSize) * localExpertNum) * 16 * sizeof(int32_t);
    const size_t expandXBytes         = static_cast<size_t>(rankSize) * bs * topK * tokenWithMetaLenBytes; // v15 ★
    const size_t dynamicScalesBytes   = A * sizeof(float);
    // v15 ★ assistInfoForCombine 新语义：每个紧密 token 对应 3 个 INT32 三元组
    //   worst-case alloc = rankSize * bs * topK * 3 * 4 字节（保留 op_def shape 上界 A*128*4 兼容）
    const size_t assistInfoBytes      = A * 128 * sizeof(int32_t);
    const size_t expertTokenNumsBytes = static_cast<size_t>(localExpertNum) * sizeof(uint64_t);
    // combine op_def
    const size_t quantExpandXBytes    = expandXBytes;       // 与 expandX 同公式（v15 含 metadata）
    const size_t combCmdInfoBytes     = dispCmdInfoBytes;
    const size_t expertScalesBytes    = static_cast<size_t>(bs) * topK * sizeof(float);
    const size_t xOutBytes            = xBytes;
    const size_t routingTableBytes    = static_cast<size_t>(bs) * topK * sizeof(uint32_t);
    // v16 ★ 删除 yCombineBytes —— combine 阶段复用 dispatch 同一块 commWindowDev

    void *xDev               = devAlloc(xBytes);
    void *expertIdsDev       = devAlloc(expertIdsBytes);
    void *scalesDev          = devAlloc(scalesBytes);
    void *xActiveMaskDev     = devAlloc(xActiveMaskBytes);
    void *yDev               = devAlloc(opDefYBytes);   // v13：op_def yAddr 占位 alloc，device 端不动
    void *commWindowDev      = devAlloc(commWindowBytes); // v13 ★ 对端接收 window（v15 含 metadata）
    void *xWithMetaDev       = devAlloc(xWithMetaBytes);  // v15 ★ host 预拼装 (bs*K, tokenWithMetaLen)
    void *expandIdxDev       = devAlloc(expandIdxBytes);
    void *dispCmdInfoDev     = devAlloc(dispCmdInfoBytes);
    void *expandXDev         = devAlloc(expandXBytes);
    void *dynamicScalesDev   = devAlloc(dynamicScalesBytes);
    void *assistInfoDev      = devAlloc(assistInfoBytes);
    void *expertTokenNumsDev = devAlloc(expertTokenNumsBytes);
    void *quantExpandXDev    = devAlloc(quantExpandXBytes);   // host 0，combine 占位
    void *combCmdInfoDev     = devAlloc(combCmdInfoBytes);
    void *expertScalesDev    = devAlloc(expertScalesBytes);
    void *xOutDev            = devAlloc(xOutBytes);
    void *routingTableDev    = devAlloc(routingTableBytes);
    // v16 ★ 不再 alloc yCombineDev —— combine 阶段复用 commWindowDev

    // ==================== Demo 数据准备 ====================
    // x: 每个 token i 全部填 byte = (i + 1)，便于校验闭环
    //   t0 → 全 0x01, t1 → 全 0x02, ..., t7 → 全 0x08
    std::vector<uint8_t> hostX(xBytes);
    const size_t bytesPerToken = static_cast<size_t>(hiddenSize) * sizeof(uint16_t);
    for (uint32_t i = 0; i < bs; ++i) {
        std::memset(hostX.data() + i * bytesPerToken, static_cast<uint8_t>(i + 1), bytesPerToken);
    }
    ACL_CHECK(aclrtMemcpy(xDev, xBytes, hostX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // v16 ★ 构造 xWithMetaDev：每行 idx=i*K+j = x[i] data + 32B metadata
    //   metadata layout（4 个 INT32 + 16B padding = 32B）：
    //     [0..4):   srcRank = rankId
    //     [4..8):   srcSlot = i
    //     [8..12):  dstK = j
    //     [12..16): dstByteOff = (i*K + j) * tokenWithMetaLen  ← v16 ★ host 预算
    //     [16..32): pad (16 字节)
    //
    //   dstByteOff 设计原理：combine_setup 反向把 token 写到对端 commWindow[slot=srcSlot*K+dstK]
    //   = commWindow[i*K + j] = commWindow[idx]，所以 dstByteOff = idx * tokenWithMetaLen
    //   device 端 CCU 严格清单不支持 Var*Var/Var*Imm，host 预算后 device 直接 LoadVariable 用作 GM 偏移
    std::vector<uint8_t> hostXWithMeta(xWithMetaBytes, 0);
    for (uint32_t i = 0; i < bs; ++i) {
        for (uint32_t j = 0; j < topK; ++j) {
            uint32_t idx = i * topK + j;
            uint8_t* rowBase = hostXWithMeta.data() + idx * tokenWithMetaLenBytes;
            std::memcpy(rowBase, hostX.data() + i * bytesPerToken, bytesPerToken);
            int32_t* meta = reinterpret_cast<int32_t*>(rowBase + tokenMsgSizeBytes);
            meta[0] = rankId;                                                            // srcRank
            meta[1] = static_cast<int32_t>(i);                                            // srcSlot
            meta[2] = static_cast<int32_t>(j);                                            // dstK
            meta[3] = static_cast<int32_t>(idx * tokenWithMetaLenBytes);                  // v16 ★ dstByteOff
        }
    }
    ACL_CHECK(aclrtMemcpy(xWithMetaDev, xWithMetaBytes, hostXWithMeta.data(),
                          xWithMetaBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // v15 ★ 把 expandX padding 区预初始化为 0xFF（int32 视角 = -1 sentinel）
    //   dispatch_teardown 紧密拷完后，后面区域保留 sentinel，
    //   combine_setup 段 5/7 host unroll bs*K 次时，padding token metadata.srcRank=-1
    //   不命中任何 r ∈ [0, rankSize) 的 CCU_IF，自动跳过 ✓
    ACL_CHECK(aclrtMemset(expandXDev, expandXBytes, 0xFF, expandXBytes));

    // expertIds: 线性路由覆盖所有 rank
    std::vector<uint32_t> hostExpertIds(bs * topK);
    std::vector<uint32_t> hostRoutingTable(bs * topK);
    const uint32_t moeExpertPerRankNum = moeExpertNum / (rankSize - sharedExpertRankNum);
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

    // expert_scales: 全 1.0（last-write-wins demo 模式不使用，但 op_def 要求）
    std::vector<float> hostExpertScales(bs * topK, 1.0f);
    ACL_CHECK(aclrtMemcpy(expertScalesDev, expertScalesBytes, hostExpertScales.data(),
                          expertScalesBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // ==================== Stage 1: dispatch_setup CCU ====================
    // 初始化 dispatch commCmdInfo sentinel: 前 epWorldSize 槽 = -1 (0xFFFFFFFF)
    const size_t cmdSentinelBytes = static_cast<size_t>(rankSize) * sizeof(int32_t);
    ACL_CHECK(aclrtMemset(dispCmdInfoDev, cmdSentinelBytes, 0xFF, cmdSentinelBytes));

    auto dispatchSetupTask = std::make_unique<hcomm::CcuTaskArgMoeDistributeDispatchCcuSetup>(
        reinterpret_cast<uint64_t>(xDev),
        reinterpret_cast<uint64_t>(expertIdsDev),
        reinterpret_cast<uint64_t>(scalesDev),
        reinterpret_cast<uint64_t>(xActiveMaskDev),
        reinterpret_cast<uint64_t>(yDev),                // v13: op_def yAddr 占位
        reinterpret_cast<uint64_t>(expandIdxDev),
        reinterpret_cast<uint64_t>(dispCmdInfoDev),
        reinterpret_cast<uint64_t>(commWindowDev),       // v13 ★ commWindow
        reinterpret_cast<uint64_t>(routingTableDev),
        reinterpret_cast<uint64_t>(xWithMetaDev));       // v15 ★ host 预拼装 x+metadata

    HCCL_CHECK(HcclCcuKernelLaunch(hcclComm, threadHandle, dispatchSetupKernel,
                                    static_cast<void *>(dispatchSetupTask.get())));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[rank %d] stage 1 dispatch_setup done\n", rankId);

    // ==================== Stage 2: dispatch_teardown AIV ====================
    MoeDistributeDispatchCcuTeardownTilingData dispTearTiling = {};
    dispTearTiling.epWorldSize         = rankSize;
    dispTearTiling.epRankId            = rankId;
    dispTearTiling.moeExpertNum        = moeExpertNum;
    dispTearTiling.moeExpertPerRankNum = moeExpertPerRankNum;
    dispTearTiling.bs                  = bs;
    dispTearTiling.topK                = topK;
    dispTearTiling.hiddenSize          = hiddenSize;
    dispTearTiling.expertTokenNumsType = expertTokenNumsType;
    dispTearTiling.tokenMsgLen         = tokenMsgSizeBytes;
    dispTearTiling.tokenWithMetaLen    = tokenWithMetaLenBytes;                                     // v15 ★
    dispTearTiling.segBytes            = static_cast<uint64_t>(bs) * topK * tokenWithMetaLenBytes;  // v15 ★
    void *dispTilingDev = devAlloc(sizeof(dispTearTiling));
    ACL_CHECK(aclrtMemcpy(dispTilingDev, sizeof(dispTearTiling),
                          &dispTearTiling, sizeof(dispTearTiling), ACL_MEMCPY_HOST_TO_DEVICE));

    moe_distribute_dispatch_ccu_teardown<<<1, nullptr, stream>>>(
        xDev, yDev, expertIdsDev, dispCmdInfoDev,
        expandXDev, dynamicScalesDev, assistInfoDev, expertTokenNumsDev,
        commWindowDev,       // v13 ★ 新增 commWindow 参数（device 端实际读这个，op_def yAddr 占位）
        nullptr, dispTilingDev);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    // device 端不写的字段 host 归零
    ACL_CHECK(aclrtMemset(dynamicScalesDev,   dynamicScalesBytes,   0, dynamicScalesBytes));
    ACL_CHECK(aclrtMemset(expertTokenNumsDev, expertTokenNumsBytes, 0, expertTokenNumsBytes));
    printf("[rank %d] stage 2 dispatch_teardown done\n", rankId);

    // ==================== Stage 3: 跳过 expert 处理（identity 模式）====================
    // 生产场景这里跑 GroupedMatmul 等 expert 计算；本 demo 不动 expandX 用 identity 验证闭环。

    // ==================== Stage 4: combine_setup CCU ====================
    // 初始化 combine commCmdInfo sentinel
    ACL_CHECK(aclrtMemset(combCmdInfoDev, cmdSentinelBytes, 0xFF, cmdSentinelBytes));
    // v16 ★ combine 阶段复用 commWindowDev，dispatch 残留数据需清零（防止 combine_teardown 误读 dispatch 数据）
    ACL_CHECK(aclrtMemset(commWindowDev, commWindowBytes, 0, commWindowBytes));

    auto combineSetupTask = std::make_unique<hcomm::CcuTaskArgMoeDistributeCombineCcuSetup>(
        reinterpret_cast<uint64_t>(expandXDev),           // op_def Input 1
        reinterpret_cast<uint64_t>(expertIdsDev),          // op_def Input 2
        reinterpret_cast<uint64_t>(assistInfoDev),         // op_def Input 3
        reinterpret_cast<uint64_t>(quantExpandXDev),       // op_def Output 1 (host 0)
        reinterpret_cast<uint64_t>(combCmdInfoDev),        // op_def Output 2
        reinterpret_cast<uint64_t>(commWindowDev),         // v16 ★ 复用 dispatch 同款 commWindow
        reinterpret_cast<uint64_t>(routingTableDev));      // v16 device 端不读，保留 IO 契约

    HCCL_CHECK(HcclCcuKernelLaunch(hcclComm, threadHandle, combineSetupKernel,
                                    static_cast<void *>(combineSetupTask.get())));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[rank %d] stage 4 combine_setup done\n", rankId);

    // ==================== Stage 5: combine_teardown AIV ====================
    MoeDistributeCombineCcuTeardownTilingData combTearTiling = {};
    combTearTiling.epWorldSize         = rankSize;
    combTearTiling.epRankId            = rankId;
    combTearTiling.moeExpertNum        = moeExpertNum;
    combTearTiling.moeExpertPerRankNum = moeExpertPerRankNum;
    combTearTiling.bs                  = bs;
    combTearTiling.topK                = topK;
    combTearTiling.hiddenSize          = hiddenSize;
    combTearTiling.commQuantMode       = commQuantMode;
    combTearTiling.tokenMsgLen         = tokenMsgSizeBytes;
    combTearTiling.tokenWithMetaLen    = tokenWithMetaLenBytes;                                     // v15 ★
    combTearTiling.segBytes            = static_cast<uint64_t>(bs) * topK * tokenWithMetaLenBytes;  // v15 ★
    void *combTilingDev = devAlloc(sizeof(combTearTiling));
    ACL_CHECK(aclrtMemcpy(combTilingDev, sizeof(combTearTiling),
                          &combTearTiling, sizeof(combTearTiling), ACL_MEMCPY_HOST_TO_DEVICE));

    moe_distribute_combine_ccu_teardown<<<1, nullptr, stream>>>(
        expandXDev, quantExpandXDev, expertIdsDev, expandIdxDev,
        expertScalesDev, combCmdInfoDev, nullptr /*xActiveMask*/, nullptr /*sharedExpertX*/,
        xOutDev,
        commWindowDev,             // v16 ★ 复用 dispatch 同款 commWindow（替代 yCombine + routingTable）
        nullptr /*workspace*/, combTilingDev);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[rank %d] stage 5 combine_teardown done\n", rankId);

    // ==================== Stage 6: 校验数据闭环 ====================
    std::vector<uint8_t> hostXOut(xOutBytes);
    ACL_CHECK(aclrtMemcpy(hostXOut.data(), xOutBytes, xOutDev, xOutBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = true;
    uint32_t firstMismatchToken = 0;
    for (uint32_t i = 0; i < bs; ++i) {
        const uint8_t expectedByte = static_cast<uint8_t>(i + 1);
        for (size_t off = 0; off < bytesPerToken; ++off) {
            if (hostXOut[i * bytesPerToken + off] != expectedByte) {
                if (pass) firstMismatchToken = i;
                pass = false;
                break;
            }
        }
        if (!pass) break;
    }

    if (pass) {
        printf("[rank %d] ★ PASS: 数据闭环验证通过——xOut == x（identity expert + last-write-wins）\n", rankId);
    } else {
        printf("[rank %d] ✗ FAIL: 闭环失败，首个不匹配 token = %u\n", rankId, firstMismatchToken);
    }

    // ==================== 释放 ====================
    aclrtFree(xDev); aclrtFree(expertIdsDev); aclrtFree(scalesDev); aclrtFree(xActiveMaskDev);
    aclrtFree(yDev); aclrtFree(commWindowDev); aclrtFree(xWithMetaDev);   // v15 ★
    aclrtFree(expandIdxDev); aclrtFree(dispCmdInfoDev);
    aclrtFree(expandXDev); aclrtFree(dynamicScalesDev); aclrtFree(assistInfoDev); aclrtFree(expertTokenNumsDev);
    aclrtFree(quantExpandXDev); aclrtFree(combCmdInfoDev); aclrtFree(expertScalesDev); aclrtFree(xOutDev);
    aclrtFree(routingTableDev);   // v16 ★ 已删 yCombineDev
    aclrtFree(dispTilingDev); aclrtFree(combTilingDev);

    HCCL_CHECK(HcclCommDestroy(hcclComm));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    printf("[rank %d] done\n", rankId);
    return pass ? 0 : 1;
}
