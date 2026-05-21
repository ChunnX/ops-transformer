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
 * 支持多进程 2p：每进程经 --rank_id 独占一张卡，进程内再分双流。
 *
 * 控制面（HcclComm / channel / HcclBuffer / remote_va_table）待真实 CCU kernel
 * 落地后接入，本轮不实现 —— 当前 kernel 为空壳、不消费 comm_context。
 * kernel 为空壳，端到端跑完即视为通过（精度不涉及）。
 */

#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "moe_dispatch_ccu_direct.h"

int main(int argc, char *argv[])
{
    // ---- 解析 --rank_id：每个进程独占一张卡（2p 即起 rank 0 / rank 1 两进程）----
    int32_t rankId = 0;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--rank_id") == 0) {
            rankId = static_cast<int32_t>(atoi(argv[i + 1]));
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

    printf("[rank %d] ==== CCU dispatch <<<>>> 直调端到端冒烟测试（device %d）====\n",
           rankId, deviceId);

    // ---- ACL 初始化 ----
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));

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
    printf("[rank %d] [1/5] GM 缓冲分配完成（%zu 块）\n", rankId, blocks.size());


    
    // ============================================================
    // 【控制面 —— 待真实 CCU kernel 落地后接入，本轮（方案 A）不实现】
    //   真实流程参考 SE 草图 + hcomm/include/hccl：
    //     HcclCommInitAll(...)          单机多卡建通信域（免 rank_table）
    //     HcclGetHcclBuffer(...)        拿本地 HCCL buffer
    //     HcclChannelAcquire(...)       建 p2p channel
    //     HcclGetRemoteIpcHcclBuf(...)  取各 rank 远端 buffer VA → remote_va_table
    //     aclrtMemcpy 把 remote_va_table 搬上卡，作为 setup kernel 入参
    //   当前 kernel 为空壳、不消费 comm_context，故此处暂不接入。
    // ============================================================

    // ---- 草图: stream1 主流执行到 dispatch 触发点 ----
    // 真实场景此处是 dispatch 之前的主流算子；冒烟工程用 record event 标记该触发点
    ACL_CHECK(aclrtRecordEvent(depEvent, stream1));

    // ---- 草图: stream2.wait(stream1) —— CCU 流等主流到点后才启动 setup ----
    ACL_CHECK(aclrtStreamWaitEvent(stream2, depEvent));

    // ---- 草图: setup_ccu_kernel<<<stream2>>> —— setup 发往 CCU 流 ----
    moe_distribute_dispatch_ccu_setup_do(
        blockDim, stream2,
        x, expertIds, scales, xActiveMask,
        yOut, expandIdxOut, commCmdInfo, workspace, tilingSetup);
    printf("[rank %d] [2/5] setup     <<<>>> 直调 → stream2（CCU 流）\n", rankId);

    // ---- 草图: teardown_aicore_kernel<<<stream1>>> —— teardown 发往主流 ----
    // teardown 经 commCmdInfo 消费 setup 产出（comm_context 载体）；空壳阶段二者并行无害，
    // 真实 kernel 阶段二者依赖由 device 侧 buffer 信号协调（草图“buffer 0 写 1”）。
    moe_distribute_dispatch_ccu_teardown_do(
        blockDim, stream1,
        x, yOut, expertIds, commCmdInfo,
        expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
        workspace, tilingTeardown);
    printf("[rank %d] [3/5] teardown  <<<>>> 直调 → stream1（主流）\n", rankId);

    // ---- 两条流各自同步，确保 setup/teardown 都执行完再释放 ----
    ACL_CHECK(aclrtSynchronizeStream(stream2));
    ACL_CHECK(aclrtSynchronizeStream(stream1));
    printf("[rank %d] [4/5] stream2 + stream1 同步完成\n", rankId);

    // ---- 资源释放 ----
    for (void *p : blocks) {
        aclrtFree(p);
    }
    ACL_CHECK(aclrtDestroyEvent(depEvent));
    ACL_CHECK(aclrtDestroyStream(stream2));
    ACL_CHECK(aclrtDestroyStream(stream1));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    printf("[rank %d] [5/5] 资源释放完成\n", rankId);
    printf("[rank %d] ==== 端到端 <<<>>> 双流直调跑通（PASS）====\n", rankId);
    return 0;
}
