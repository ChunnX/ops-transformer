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

// 彻底精简签名（与 teardown.h / .cpp 严格一致）
extern "C" __global__ __aicore__ void moe_distribute_dispatch_ccu_teardown(
    GM_ADDR commCmdInfo, GM_ADDR commWindow, GM_ADDR expandX,
    GM_ADDR assistInfoForCombine, GM_ADDR tilingGM);

extern "C" __global__ __aicore__ void moe_distribute_combine_ccu_teardown(
    GM_ADDR commCmdInfo, GM_ADDR commWindow, GM_ADDR xOut, GM_ADDR tilingGM);

int main(int argc, char *argv[]) {
    int32_t rankId = 0, rankSize = 2;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--rank_id") == 0)        rankId   = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--rank_size") == 0) rankSize = atoi(argv[i + 1]);
    }
    const int32_t deviceId = rankId;

    // ---- demo 参数（v12 简化，便于校验）----
    const uint32_t bs           = 8;
    const uint32_t hiddenSize   = 256;
    const uint32_t topK         = 2;
    const uint32_t moeExpertNum = 8;       // → moeExpertPerRankNum = 4
    const uint32_t sharedExpertRankNum = 0;
    // quantMode：0 = 非量化 BF16，1 = MXFP8（host 端预量化）
    //   commQuantMode 必须 = quantMode（dispatch 发什么格式 combine 收同款，两端 tokenMsgLen 公式须一致）
    const uint32_t quantMode           = 0;      // dispatch quantMode (0=BF16, 1=MXFP8)
    const uint32_t commQuantMode       = quantMode;

    const uint32_t localExpertNum     = moeExpertNum / (rankSize - sharedExpertRankNum);
    auto align256 = [](uint32_t n) -> uint32_t { return ((n + 255U) / 256U) * 256U; };
    auto align32  = [](uint32_t n) -> uint32_t { return ((n + 31U)  / 32U)  * 32U;  };

    // v17 Phase 2 ★ MXFP8 模式 H 约束 runtime check：H 必须是 64 的倍数
    //   原因：combine_teardown 按 BF16 视图整除 copyElemsPerToken = (H + H/32) / 2，
    //         要求 (H + H/32) = H*33/32 是 2 的倍数 ↔ H 是 64 的倍数
    //   生产 H ∈ {256, 4096, 7168, 8192} 都满足；违反则字节级闭环最后 1 字节 mismatch
    if (quantMode == 1 && (hiddenSize % 64U != 0)) {
        printf("[rank %d][ERROR] MXFP8 模式要求 hiddenSize 是 64 的倍数（当前 H=%u 不满足，"
               "原因见 combine_teardown.cpp 注释）\n", rankId, hiddenSize);
        return -1;
    }
    // v17 ★ tokenMsgLen 按 quantMode 切（与 dispatch_setup.cpp constructor 公式严格一致）
    const uint64_t tokenMsgSizeBytes = (quantMode == 0)
        ? static_cast<uint64_t>(align256(hiddenSize)) * sizeof(uint16_t)              // BF16: Align256(H)*2
        : static_cast<uint64_t>(align32(hiddenSize + hiddenSize / 32U));             // MXFP8: Align32(H + H/32)
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
        rankSize, rankId, moeExpertNum, bs, topK, hiddenSize, quantMode);
    dispatchSetupArg->channels = channels;

    auto combineSetupArg = std::make_shared<hcomm::CcuKernelArgMoeDistributeCombineCcuSetup>(
        rankSize, rankId, moeExpertNum, bs, topK, hiddenSize, commQuantMode);
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

    // ---- buffer 字节数（只留 device 实际读写的）----
    //   commWindow / expandX 每槽 = tokenWithMetaLen（data + 32B metadata）
    //   assistInfo：紧密区每 token 3 个 INT32；保留 A*128*4 兼容（远大于实际写量）
    const size_t xBytes               = static_cast<size_t>(bs) * hiddenSize * sizeof(uint16_t);   // hostX：构造 xWithMeta + 校验
    const size_t expertIdsBytes       = static_cast<size_t>(bs) * topK * sizeof(uint32_t);         // dispatch_setup 读
    const size_t commWindowBytes      = static_cast<size_t>(rankSize) * bs * topK * tokenWithMetaLenBytes;
    const size_t xWithMetaBytes       = static_cast<size_t>(bs) * topK * tokenWithMetaLenBytes;
    const size_t expandIdxBytes       = static_cast<size_t>(bs) * topK * sizeof(uint32_t);         // dispatch_setup 写
    const size_t dispCmdInfoBytes     = (static_cast<size_t>(bs) * topK
                                         + static_cast<size_t>(rankSize) * localExpertNum) * 16 * sizeof(int32_t);
    const size_t expandXBytes         = A * tokenWithMetaLenBytes;   // op_def 标准 A 公式
    const size_t assistInfoBytes      = A * 128 * sizeof(int32_t);   // dispatch_teardown 写 metadata 三元组
    const size_t combCmdInfoBytes     = dispCmdInfoBytes;
    // xOut 字节数按 quantMode 切：BF16=bs*2H；MXFP8=bs*(H+H/32)
    const size_t xOutBytes            = (quantMode == 0)
        ? xBytes
        : static_cast<size_t>(bs) * (hiddenSize + hiddenSize / 32U);
    const size_t routingTableBytes    = static_cast<size_t>(bs) * topK * sizeof(uint32_t);         // dispatch_setup 读

    void *expertIdsDev       = devAlloc(expertIdsBytes);
    void *commWindowDev      = devAlloc(commWindowBytes); // 对端接收 window（含 metadata），dispatch/combine 复用
    void *xWithMetaDev       = devAlloc(xWithMetaBytes);  // host 预拼装 (bs*K, tokenWithMetaLen)
    void *expandIdxDev       = devAlloc(expandIdxBytes);
    void *dispCmdInfoDev     = devAlloc(dispCmdInfoBytes);
    void *expandXDev         = devAlloc(expandXBytes);
    void *assistInfoDev      = devAlloc(assistInfoBytes);
    void *combCmdInfoDev     = devAlloc(combCmdInfoBytes);
    void *xOutDev            = devAlloc(xOutBytes);
    void *routingTableDev    = devAlloc(routingTableBytes);

    // ==================== Demo 数据准备 ====================
    // x: 每个 token i 全部填 byte = (i + 1)，便于校验闭环
    //   t0 → 全 0x01, t1 → 全 0x02, ..., t7 → 全 0x08
    std::vector<uint8_t> hostX(xBytes);
    const size_t bytesPerToken = static_cast<size_t>(hiddenSize) * sizeof(uint16_t);
    for (uint32_t i = 0; i < bs; ++i) {
        std::memset(hostX.data() + i * bytesPerToken, static_cast<uint8_t>(i + 1), bytesPerToken);
    }
    // hostX 仅用于构造 xWithMeta + Stage 6 校验（dispatch_setup 读 xWithMeta，无需单独 xDev）

    // v17 ★ 构造 xWithMetaDev：按 quantMode 切 data 段格式
    //   layout（每行 tokenWithMetaLen 字节）：
    //     [0..tokenMsgSize):    token data
    //       - quantMode=0 (BF16): hiddenSize*2 字节 BF16 数据 + padding to Align256(H)*2
    //       - quantMode=1 (MXFP8): hiddenSize 字节 FP8 data + hiddenSize/32 字节 E8M0 scale + padding
    //     [tokenMsgSize..tokenMsgSize+16): metadata 4 个 INT32
    //       [0..4):   srcRank = rankId
    //       [4..8):   srcSlot = i
    //       [8..12):  dstK = j
    //       [12..16): dstByteOff = idx * tokenWithMetaLen
    //     [tokenMsgSize+16..tokenWithMetaLen): pad (16 字节)
    std::vector<uint8_t> hostXWithMeta(xWithMetaBytes, 0);
    for (uint32_t i = 0; i < bs; ++i) {
        for (uint32_t j = 0; j < topK; ++j) {
            uint32_t idx = i * topK + j;
            uint8_t* rowBase = hostXWithMeta.data() + idx * tokenWithMetaLenBytes;

            if (quantMode == 0) {
                // 非量化：直接 memcpy BF16 数据
                std::memcpy(rowBase, hostX.data() + i * bytesPerToken, bytesPerToken);
            } else {
                // v17 ★ MXFP8 mock 量化：host 端把 BF16 cast 成 FP8 + 算 per-32-block E8M0 scale
                //   [0..H):    FP8 data (H bytes)：mock 取 BF16 高字节为 FP8（仅 demo，非 IEEE FP8）
                //   [H..H+H/32): E8M0 scale (H/32 bytes)：mock 全填 0x80 (表 2^0 unit scale)
                //   注：SE 信息——"dispatch_setup_ccu 拿到的输入是已经量化好的结果"
                //   生产版本应在 host 端用真实 MXFP8 量化算法（E4M3 + E8M0 per-32-block）
                const uint8_t* srcBF16 = hostX.data() + i * bytesPerToken;
                uint8_t* dstFP8   = rowBase;                       // FP8 data 段
                uint8_t* dstScale = rowBase + hiddenSize;          // E8M0 scale 段
                for (uint32_t e = 0; e < hiddenSize; ++e) {
                    dstFP8[e] = srcBF16[e * 2 + 1];                // mock: BF16 高字节作 FP8
                }
                for (uint32_t b = 0; b < hiddenSize / 32; ++b) {
                    dstScale[b] = 0x80;                            // mock E8M0 = 2^0 (unit scale)
                }
            }

            int32_t* meta = reinterpret_cast<int32_t*>(rowBase + tokenMsgSizeBytes);
            meta[0] = rankId;                                                            // srcRank
            meta[1] = static_cast<int32_t>(i);                                            // srcSlot
            meta[2] = static_cast<int32_t>(j);                                            // dstK
            meta[3] = static_cast<int32_t>(idx * tokenWithMetaLenBytes);                  // v16 dstByteOff
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

    // expert_scales 已删：combine_teardown 简化版 last-write-wins 不做加权

    // ==================== Stage 1: dispatch_setup CCU ====================
    // 初始化 dispatch commCmdInfo sentinel: 前 epWorldSize 槽 = -1 (0xFFFFFFFF)
    const size_t cmdSentinelBytes = static_cast<size_t>(rankSize) * sizeof(int32_t);
    ACL_CHECK(aclrtMemset(dispCmdInfoDev, cmdSentinelBytes, 0xFF, cmdSentinelBytes));

    auto dispatchSetupTask = std::make_unique<hcomm::CcuTaskArgMoeDistributeDispatchCcuSetup>(
        reinterpret_cast<uint64_t>(expertIdsDev),
        reinterpret_cast<uint64_t>(expandIdxDev),
        reinterpret_cast<uint64_t>(dispCmdInfoDev),
        reinterpret_cast<uint64_t>(commWindowDev),
        reinterpret_cast<uint64_t>(routingTableDev),
        reinterpret_cast<uint64_t>(xWithMetaDev));

    HCCL_CHECK(HcclCcuKernelLaunch(hcclComm, threadHandle, dispatchSetupKernel,
                                    static_cast<void *>(dispatchSetupTask.get())));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[rank %d] stage 1 dispatch_setup done\n", rankId);

    // ==================== Stage 2: dispatch_teardown AIV ====================
    MoeDistributeDispatchCcuTeardownTilingData dispTearTiling = {};
    dispTearTiling.epWorldSize      = rankSize;
    dispTearTiling.bs               = bs;
    dispTearTiling.topK             = topK;
    dispTearTiling.tokenMsgLen      = tokenMsgSizeBytes;
    dispTearTiling.tokenWithMetaLen = tokenWithMetaLenBytes;
    void *dispTilingDev = devAlloc(sizeof(dispTearTiling));
    ACL_CHECK(aclrtMemcpy(dispTilingDev, sizeof(dispTearTiling),
                          &dispTearTiling, sizeof(dispTearTiling), ACL_MEMCPY_HOST_TO_DEVICE));

    moe_distribute_dispatch_ccu_teardown<<<1, nullptr, stream>>>(
        dispCmdInfoDev, commWindowDev, expandXDev, assistInfoDev, dispTilingDev);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[rank %d] stage 2 dispatch_teardown done\n", rankId);

    // ==================== Stage 3: 跳过 expert 处理（identity 模式）====================
    // 生产场景这里跑 GroupedMatmul 等 expert 计算；本 demo 不动 expandX 用 identity 验证闭环。

    // ==================== Stage 4: combine_setup CCU ====================
    // 初始化 combine commCmdInfo sentinel
    ACL_CHECK(aclrtMemset(combCmdInfoDev, cmdSentinelBytes, 0xFF, cmdSentinelBytes));
    // v16 ★ combine 阶段复用 commWindowDev，dispatch 残留数据需清零（防止 combine_teardown 误读 dispatch 数据）
    ACL_CHECK(aclrtMemset(commWindowDev, commWindowBytes, 0, commWindowBytes));

    auto combineSetupTask = std::make_unique<hcomm::CcuTaskArgMoeDistributeCombineCcuSetup>(
        reinterpret_cast<uint64_t>(expandXDev),
        reinterpret_cast<uint64_t>(combCmdInfoDev),
        reinterpret_cast<uint64_t>(commWindowDev));

    HCCL_CHECK(HcclCcuKernelLaunch(hcclComm, threadHandle, combineSetupKernel,
                                    static_cast<void *>(combineSetupTask.get())));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[rank %d] stage 4 combine_setup done\n", rankId);

    // ==================== Stage 5: combine_teardown AIV ====================
    MoeDistributeCombineCcuTeardownTilingData combTearTiling = {};
    combTearTiling.epWorldSize      = rankSize;
    combTearTiling.bs               = bs;
    combTearTiling.topK             = topK;
    combTearTiling.hiddenSize       = hiddenSize;
    combTearTiling.commQuantMode    = commQuantMode;     // 切 BF16/MXFP8 拷贝路径
    combTearTiling.tokenWithMetaLen = tokenWithMetaLenBytes;
    void *combTilingDev = devAlloc(sizeof(combTearTiling));
    ACL_CHECK(aclrtMemcpy(combTilingDev, sizeof(combTearTiling),
                          &combTearTiling, sizeof(combTearTiling), ACL_MEMCPY_HOST_TO_DEVICE));

    moe_distribute_combine_ccu_teardown<<<1, nullptr, stream>>>(
        combCmdInfoDev, commWindowDev, xOutDev, combTilingDev);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    printf("[rank %d] stage 5 combine_teardown done\n", rankId);

    // ==================== Stage 6: 校验数据闭环 ====================
    // v17 Phase 2 ★ MXFP8 路径也支持 byte-level 校验（字节透传方案）：
    //   BF16:  xOut[i] = bs 个 hostX[i]（每 token 全 byte=(i+1) BF16 数据，2*H 字节）
    //   MXFP8: xOut[i] = hostXWithMeta[i*K + 0][0..H+H/32) （FP8 data + E8M0 scale）
    //          identity 模式 + last-write-wins：本卡 (i, j=0) 副本经路由→回本卡 commWindow[i*K]→xOut[i]
    std::vector<uint8_t> hostXOut(xOutBytes);
    ACL_CHECK(aclrtMemcpy(hostXOut.data(), xOutBytes, xOutDev, xOutBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = true;
    if (quantMode == 0) {
        // BF16: 校验 hostXOut[i] 每字节都等于 (i+1)（hostX 的填充模式）
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
            printf("[rank %d] ★ PASS (BF16): 数据闭环验证通过——xOut == x (identity + last-write-wins)\n", rankId);
        } else {
            printf("[rank %d] ✗ FAIL (BF16): 闭环失败，首个不匹配 token = %u\n", rankId, firstMismatchToken);
        }
    } else {
        // v17 Phase 2 ★ MXFP8: byte-level 校验 xOut[i] 字节流 == hostXWithMeta[i*K+0] 的 FP8+scale 段
        //   xOutBytesPerToken = H + H/32
        //   hostXWithMeta[i*K+0][0..H+H/32) = host 端 mock 量化后的 FP8 data + E8M0 scale
        //   identity 模式：dispatch→teardown→combine_setup→commWindow[i*K]→combine_teardown→xOut[i]
        //   全程 byte-透传，xOut[i] 应字节级等于原始 mock 量化内容
        const size_t xOutBytesPerToken = hiddenSize + hiddenSize / 32U;
        uint32_t firstMismatchToken = 0;
        for (uint32_t i = 0; i < bs; ++i) {
            const uint8_t* expected = hostXWithMeta.data() + (i * topK + 0) * tokenWithMetaLenBytes;
            const uint8_t* actual   = hostXOut.data() + i * xOutBytesPerToken;
            if (std::memcmp(actual, expected, xOutBytesPerToken) != 0) {
                if (pass) firstMismatchToken = i;
                pass = false;
                break;
            }
        }
        if (pass) {
            printf("[rank %d] ★ PASS (MXFP8): 数据闭环验证通过——xOut == hostXWithMeta[i*K+0].FP8+scale 段\n",
                   rankId);
        } else {
            printf("[rank %d] ✗ FAIL (MXFP8): 闭环失败，首个不匹配 token = %u\n", rankId, firstMismatchToken);
        }
    }

    // ==================== 释放 ====================
    aclrtFree(expertIdsDev); aclrtFree(commWindowDev); aclrtFree(xWithMetaDev);
    aclrtFree(expandIdxDev); aclrtFree(dispCmdInfoDev);
    aclrtFree(expandXDev); aclrtFree(assistInfoDev);
    aclrtFree(combCmdInfoDev); aclrtFree(xOutDev);
    aclrtFree(routingTableDev);
    aclrtFree(dispTilingDev); aclrtFree(combTilingDev);

    HCCL_CHECK(HcclCommDestroy(hcclComm));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    printf("[rank %d] done\n", rankId);
    return pass ? 0 : 1;
}
