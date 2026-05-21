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
 * \file test_moe_dispatch_ccu_direct.cpp
 * \brief CCU dispatch setup/teardown 两算子的 <<<>>> 直调端到端冒烟测试 host 侧。
 *
 * 调用编排对齐 SE 草图的双流模型：
 *   - stream1（主流）：承载前序算子 + teardown(AICore 引擎)
 *   - stream2（CCU 流）：承载 setup(CCU 引擎)，与 stream1 并行以做通算 overlap
 *   - stream2 经 event 等待 stream1 到达 dispatch 触发点后才启动（草图 stream2.wait(stream1)）
 * 支持多进程：每进程经 --rank_id 独占一张卡，--rank_size 指定总进程数。
 *
 * 控制面接入：
 *   [已实现] 建通信域 + 拿本地通信缓冲区
 *       走 RootInfo 路径（HcclGetRootInfo + HcclCommInitRootInfoConfig），免 rank_table；
 *       一进程一卡、多进程组成通信域。
 *   [暂缓] 点对点通道 + 远端通信缓冲区（HcclChannelAcquire 等）
 *       经实机验证：当前环境（Ascend950PR）对照搬 hcomm 文档样例的 HcclChannelAcquire
 *       返回 NOT_SUPPORT（AIV / CCU 引擎、UB_MEM 协议均试过）。待拿到 SE 实际可用的
 *       channel 调用代码、或真实 CCU kernel 落地阶段再接入。
 * 控制面只做 setup，不做数据面通信；kernel 为空壳，端到端跑完即视为通过（精度不涉及）。
 */

#include <acl/acl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "hccl/hccl_comm.h"
#include "moe_dispatch_ccu_direct.h"

// HcclGetHcclBuffer 声明在 hccl_res.h，但该头内含依赖 memset_s 的 inline 函数，
// 直接 include 易触发编译依赖问题；按本仓库既有写法改用 extern "C" 前置声明。
extern "C" HcclResult HcclGetHcclBuffer(HcclComm comm, void **buffer, uint64_t *size);

// rootInfo 进程间交换文件（多进程共享，run.sh 每次运行前清理）
static const char *kRootInfoPath = "/tmp/ccu_direct_rootinfo.bin";
static const char *kRootInfoTmp = "/tmp/ccu_direct_rootinfo.bin.tmp";

/*
 * 在多进程间交换 HcclRootInfo —— 免 rank_table 组通信域的标准做法。
 * rank 0 调 HcclGetRootInfo 生成 rootInfo，写临时文件后 rename 原子发布；
 * 其余 rank 轮询等待该文件就绪再读取。rootInfo 为定长 HCCL_ROOT_INFO_BYTES。
 */
static bool ExchangeRootInfo(int32_t rankId, HcclRootInfo *rootInfo)
{
    if (rankId == 0) {
        if (HcclGetRootInfo(rootInfo) != HCCL_SUCCESS) {
            printf("[rank 0][ERROR] HcclGetRootInfo failed\n");
            return false;
        }
        FILE *fp = fopen(kRootInfoTmp, "wb");
        if (fp == nullptr) {
            printf("[rank 0][ERROR] 打开 rootInfo 临时文件失败\n");
            return false;
        }
        size_t wn = fwrite(rootInfo->internal, 1, HCCL_ROOT_INFO_BYTES, fp);
        (void)fclose(fp);
        if (wn != HCCL_ROOT_INFO_BYTES) {
            printf("[rank 0][ERROR] 写 rootInfo 失败\n");
            return false;
        }
        (void)rename(kRootInfoTmp, kRootInfoPath);   // rename 原子，避免其他 rank 读到半截
        return true;
    }
    // 其余 rank：轮询等待 rootInfo 就绪，最多等 60 秒
    for (int i = 0; i < 600; ++i) {
        if (access(kRootInfoPath, R_OK) == 0) {
            FILE *fp = fopen(kRootInfoPath, "rb");
            if (fp != nullptr) {
                size_t rn = fread(rootInfo->internal, 1, HCCL_ROOT_INFO_BYTES, fp);
                (void)fclose(fp);
                if (rn == HCCL_ROOT_INFO_BYTES) {
                    return true;
                }
            }
        }
        usleep(100 * 1000);   // 0.1 秒
    }
    printf("[rank %d][ERROR] 等待 rootInfo 超时\n", rankId);
    return false;
}

int main(int argc, char *argv[])
{
    // ---- 解析 --rank_id / --rank_size：每进程独占一张卡，rank_size 为通信域规模 ----
    int32_t rankId = 0;
    int32_t rankSize = 2;   // 默认 2p
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--rank_id") == 0) {
            rankId = static_cast<int32_t>(atoi(argv[i + 1]));
        } else if (strcmp(argv[i], "--rank_size") == 0) {
            rankSize = static_cast<int32_t>(atoi(argv[i + 1]));
        }
    }
    const int32_t deviceId = rankId;

#define ACL_CHECK(expr)                                                          \
    do {                                                                         \
        aclError _ret = (expr);                                                  \
        if (_ret != ACL_SUCCESS) {                                               \
            printf("[rank %d][ERROR] %s:%d  %s  failed, ret=%d\n",               \
                   rankId, __FILE__, __LINE__, #expr, static_cast<int>(_ret));   \
            return -1;                                                           \
        }                                                                        \
    } while (0)

    // ---- 形状参数（与注册算子单测一致：bs=8, h=7168, k=8）----
    const uint32_t bs = 8;
    const uint32_t h = 7168;
    const uint32_t k = 8;
    const uint32_t blockDim = 1;   // 空壳 kernel，单核拉起即可

    printf("[rank %d] ==== CCU dispatch <<<>>> 直调端到端冒烟测试（device %d, rank_size %d）====\n",
           rankId, deviceId, rankSize);

    // ---- ACL 初始化 ----
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));

    // ============================================================
    // [1/6] 控制面：建通信域 + 拿本地通信缓冲区
    //   一进程一卡、免 rank_table，走 RootInfo 路径：
    //     HcclGetRootInfo（rank 0 生成）-> 进程间交换 -> HcclCommInitRootInfoConfig
    //   建好通信域后用 HcclGetHcclBuffer 拿本端 HCCL 通信缓冲区。
    //   点对点通道 + 远端缓冲区（HcclChannelAcquire）暂缓，见文件头说明。
    // ============================================================
    HcclRootInfo rootInfo;
    if (!ExchangeRootInfo(rankId, &rootInfo)) {
        printf("[rank %d][ERROR] rootInfo 交换失败\n", rankId);
        return -1;
    }
    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize = 200;   // HCCL 通信缓冲区大小（MB），可按需自定义
    HcclComm hcclComm = nullptr;
    HcclResult hcclRet = HcclCommInitRootInfoConfig(
        static_cast<uint32_t>(rankSize), &rootInfo,
        static_cast<uint32_t>(rankId), &commConfig, &hcclComm);
    if (hcclRet != HCCL_SUCCESS || hcclComm == nullptr) {
        printf("[rank %d][ERROR] HcclCommInitRootInfoConfig failed, ret=%d\n",
               rankId, static_cast<int>(hcclRet));
        return -1;
    }
    uint32_t commRankId = 0;
    uint32_t commRankSize = 0;
    (void)HcclGetRankId(hcclComm, &commRankId);
    (void)HcclGetRankSize(hcclComm, &commRankSize);
    void *localCommBuf = nullptr;
    uint64_t localCommBufSize = 0;
    hcclRet = HcclGetHcclBuffer(hcclComm, &localCommBuf, &localCommBufSize);
    if (hcclRet != HCCL_SUCCESS) {
        printf("[rank %d][ERROR] HcclGetHcclBuffer failed, ret=%d\n",
               rankId, static_cast<int>(hcclRet));
        return -1;
    }
    printf("[rank %d] [1/6] 控制面：通信域 rank %u/%u，本地 HCCL buffer=%p (%.1f MB)\n",
           rankId, commRankId, commRankSize, localCommBuf,
           static_cast<double>(localCommBufSize) / 1024.0 / 1024.0);

    // ---- 双流编排（对齐 SE 草图）----
    // stream1 = 主流：承载前序算子与 teardown(AICore 引擎)
    // stream2 = CCU 流：承载 setup(CCU 引擎)，与 stream1 并行以做通算 overlap
    // depEvent = 跨流依赖句柄，实现草图的 stream2.wait(stream1)
    aclrtStream stream1 = nullptr;
    aclrtStream stream2 = nullptr;
    aclrtEvent depEvent = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream1));
    ACL_CHECK(aclrtCreateStream(&stream2));
    ACL_CHECK(aclrtCreateEvent(&depEvent));

    // ---- GM 缓冲分配（空壳 kernel 不解引用，尺寸取宽松安全值）----
    std::vector<void *> blocks;
    auto devAlloc = [&blocks, rankId](size_t bytes) -> uint8_t * {
        void *p = nullptr;
        if (aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            printf("[rank %d][ERROR] aclrtMalloc %zu bytes failed\n", rankId, bytes);
            return nullptr;
        }
        blocks.push_back(p);
        return static_cast<uint8_t *>(p);
    };

    const size_t tokenBytes = static_cast<size_t>(bs) * h * 2;          // x: [bs, h] bf16
    const size_t expandBytes = static_cast<size_t>(bs) * k * h * 2;     // 展开后: [bs*k, h] bf16
    const size_t idxBytes = static_cast<size_t>(bs) * k * 4;            // 索引类: [bs*k] int32
    const size_t scalesBytes = static_cast<size_t>(bs) * 4;
    const size_t maskBytes = 256;
    const size_t commBytes = 64 * 1024;                                 // commCmdInfo（comm_context 载体）
    const size_t expertNumBytes = 256;                                  // expertTokenNums
    const size_t workspaceBytes = 4 * 1024 * 1024;                      // workspace
    const size_t tilingBytes = 16 * 1024;                               // tiling GM

    // setup 入参 / 出参
    uint8_t *x = devAlloc(tokenBytes);
    uint8_t *expertIds = devAlloc(idxBytes);
    uint8_t *scales = devAlloc(scalesBytes);
    uint8_t *xActiveMask = devAlloc(maskBytes);
    uint8_t *yOut = devAlloc(expandBytes);
    uint8_t *expandIdxOut = devAlloc(idxBytes);
    uint8_t *commCmdInfo = devAlloc(commBytes);
    uint8_t *workspace = devAlloc(workspaceBytes);
    uint8_t *tilingSetup = devAlloc(tilingBytes);
    // teardown 出参
    uint8_t *expandXOut = devAlloc(expandBytes);
    uint8_t *dynamicScalesOut = devAlloc(idxBytes);
    uint8_t *assistInfoOut = devAlloc(idxBytes);
    uint8_t *expertTokenNumsOut = devAlloc(expertNumBytes);
    uint8_t *tilingTeardown = devAlloc(tilingBytes);

    bool allocOk = x && expertIds && scales && xActiveMask && yOut && expandIdxOut &&
                   commCmdInfo && workspace && tilingSetup && expandXOut &&
                   dynamicScalesOut && assistInfoOut && expertTokenNumsOut && tilingTeardown;
    if (!allocOk) {
        printf("[rank %d][ERROR] GM 缓冲分配失败\n", rankId);
        return -1;
    }
    printf("[rank %d] [2/6] GM 缓冲分配完成（%zu 块）\n", rankId, blocks.size());

    // ---- 草图: stream1 主流执行到 dispatch 触发点 ----
    ACL_CHECK(aclrtRecordEvent(depEvent, stream1));
    // ---- 草图: stream2.wait(stream1) —— CCU 流等主流到点后才启动 setup ----
    ACL_CHECK(aclrtStreamWaitEvent(stream2, depEvent));

    // ---- 草图: setup_ccu_kernel<<<stream2>>> —— setup 发往 CCU 流 ----
    moe_distribute_dispatch_ccu_setup_do(
        blockDim, stream2,
        x, expertIds, scales, xActiveMask,
        yOut, expandIdxOut, commCmdInfo, workspace, tilingSetup);
    printf("[rank %d] [3/6] setup     <<<>>> 直调 → stream2（CCU 流）\n", rankId);

    // ---- 草图: teardown_aicore_kernel<<<stream1>>> —— teardown 发往主流 ----
    moe_distribute_dispatch_ccu_teardown_do(
        blockDim, stream1,
        x, yOut, expertIds, commCmdInfo,
        expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
        workspace, tilingTeardown);
    printf("[rank %d] [4/6] teardown  <<<>>> 直调 → stream1（主流）\n", rankId);

    // ---- 两条流各自同步，确保 setup/teardown 都执行完再释放 ----
    ACL_CHECK(aclrtSynchronizeStream(stream2));
    ACL_CHECK(aclrtSynchronizeStream(stream1));
    printf("[rank %d] [5/6] stream2 + stream1 同步完成\n", rankId);

    // ---- 资源释放 ----
    for (void *p : blocks) {
        aclrtFree(p);
    }
    ACL_CHECK(aclrtDestroyEvent(depEvent));
    ACL_CHECK(aclrtDestroyStream(stream2));
    ACL_CHECK(aclrtDestroyStream(stream1));
    hcclRet = HcclCommDestroy(hcclComm);   // 销毁通信域
    if (hcclRet != HCCL_SUCCESS) {
        printf("[rank %d][ERROR] HcclCommDestroy failed, ret=%d\n", rankId, static_cast<int>(hcclRet));
        return -1;
    }
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    printf("[rank %d] [6/6] 资源释放完成（含通信域销毁）\n", rankId);
    printf("[rank %d] ==== 端到端 <<<>>> 双流直调 + 控制面（通信域）跑通（PASS）====\n", rankId);
    return 0;
}
