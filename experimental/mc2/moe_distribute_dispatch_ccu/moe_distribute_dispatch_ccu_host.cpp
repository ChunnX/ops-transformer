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
 * \file moe_distribute_dispatch_ccu_host.cpp
 * \brief dispatch-ccu host 直调编排入口（实现设计文档 §7）。
 *
 * 链路：HcclThreadAcquire(CCU) → 反查 stream2 → HcclChannelAcquire(fullmesh, UBC_CTP)
 *      → 逐 channel 取 peer buffer 构造 remote_va_table（议题 5「前同步」）
 *      → 注册 CCU setup kernel → 构造 teardown tiling → 双流 wait
 *      → HcclCcuKernelLaunch(setup, CCU) → DispatchTeardownLaunch(teardown, AICore)
 *
 * 假设与置信度（§14）：
 *   [✅ 已核验] 所有 Hccl* / HcclCcu* API 签名 —— hcomm/include/hccl/hccl_res.h、
 *              hcomm/include/hcomm_res_defs.h、hcomm/pkg_inc/hcomm/ccu/hccl_ccu_res.h
 *   [⚠ 待上机] V10：算子 host 层直接裸调 HcclCcuKernelRegister —— 仓库内仅框架
 *              (hccl op_common.cc:1355) 调过，无算子直调先例，须在 A6 仿真器验证
 *   [⚠ 待上机] V3 ：CCU channel 用 COMM_PROTOCOL_UBC_CTP，A6 仿真器支持性待实测
 *   [⚠ 待上机] 双流编排顺序与架构师碎片一致，未在 A6 仿真器跑通
 *
 * 本文件为 host 编译 C++（不含 <<<>>>，teardown 启动经 DispatchTeardownLaunch 包装）。
 */
#include "moe_distribute_dispatch_ccu_host.h"

#include <vector>
#include <memory>
#include <cstdio>

#include "hccl/hccl_res.h"     // 控制面：HcclThreadAcquire / HcclChannelAcquire / HcclChannelGetHcclBuffer ...
#include "hcomm_res_defs.h"    // COMM_ENGINE_CCU / COMM_PROTOCOL_UBC_CTP / ChannelHandle / ThreadHandle
#include "hccl_ccu_res.h"      // HcclCcuKernelRegister / RegisterFinish / Launch / CcuKernelHandle
#include "ccu_kernel.h"        // hcomm::CcuKernel / KernelCreator
#include "dispatch_setup_ccu.h"
#include "moe_distribute_dispatch_ccu.h"  // DispatchTeardownLaunch（device 侧 <<<>>> 包装）

// host 侧返回值校验：非 0 即打印并透传返回码
#define DISPATCH_CHK(expr, msg)                                                       \
    do {                                                                              \
        auto _r = (expr);                                                             \
        if (_r != 0) {                                                                \
            printf("[ERROR][MoeDistributeDispatchCcu] %s, ret=%d\n", (msg), (int)_r);  \
            return static_cast<HcclResult>(_r);                                       \
        }                                                                             \
    } while (0)

namespace {
// 按设计文档 §10 计算 Per-Rank HBM GM Buffer 三段布局偏移，连同规模一起写进 teardown tiling。
void FillDispatchTiling(DispatchCcuTilingData& t, const DispatchCcuParam& p)
{
    t.worldSize = p.worldSize;
    t.myRank    = p.myRank;
    t.tokenNum  = p.tokenNum;
    t.hidden    = p.hidden;
    t.topk      = p.topk;
    t.expertNum = p.expertNum;
    // DATA 区每卡 token 槽位上限：P0 保守取「本卡 token 全量路由到单卡」的最坏情况
    t.maxTokenPerRank = p.tokenNum * p.topk;
    // 三段顺序：READY_FLAGS → METADATA → DATA
    t.readyFlagsOffset = 0U;
    t.metadataOffset   = t.readyFlagsOffset +
                         static_cast<uint64_t>(p.worldSize) * DISPATCH_CCU_READY_FLAG_BYTES;
    t.dataOffset       = t.metadataOffset +
                         static_cast<uint64_t>(p.worldSize) * DISPATCH_CCU_META_PER_RANK;
}
}  // namespace

HcclResult MoeDistributeDispatchCcu(HcclComm comm, void* x, void* expertIds, void* expandXOut,
                                    const DispatchCcuParam& param, aclrtStream stream1)
{
    const uint32_t worldSize = param.worldSize;

    // ── 1. 申请 CCU 通信线程 ──
    // notifyNumPerThread 预分配 Notify 事件位：fullmesh 下本卡对每个 peer 需收 / 发各一个，取 worldSize*2 留余量。
    ThreadHandle ccuThread = 0;
    DISPATCH_CHK(HcclThreadAcquire(comm, COMM_ENGINE_CCU, /*threadNum=*/1U,
                                   /*notifyNumPerThread=*/worldSize * 2U, &ccuThread),
                 "HcclThreadAcquire(CCU) failed");

    // ── 2. 反查 CCU thread 绑定的 stream2（双流编排用）──
    // 选「HcclThreadAcquire + ResGetInfo」而非 HcclThreadAcquireWithStream：后者头注释标注"当前适用
    // CPU_TS 场景"，对 CCU 引擎可用性未验证（设计文档 V9）。这里让 hcomm 自建流，再反查回来。
    aclrtStream stream2 = nullptr;
    DISPATCH_CHK(HcclThreadResGetInfo(comm, ccuThread, THREAD_RES_TYPE_STREAM,
                                      sizeof(aclrtStream), reinterpret_cast<void**>(&stream2)),
                 "HcclThreadResGetInfo(stream) failed");

    // ── 3. fullmesh：对每个 peer 建一条 CCU channel ──
    // channelProtocol 必须 COMM_PROTOCOL_UBC_CTP —— CCU mem2mem 算子实证用此协议（设计文档 §2 F13）。
    std::vector<HcclChannelDesc> descs(worldSize);
    DISPATCH_CHK(HcclChannelDescInit(descs.data(), worldSize), "HcclChannelDescInit failed");
    for (uint32_t i = 0; i < worldSize; ++i) {
        descs[i].remoteRank      = i;
        descs[i].channelProtocol = COMM_PROTOCOL_UBC_CTP;
        descs[i].notifyNum       = 2U;  // 每 channel 收 / 发各一个 Notify
    }
    std::vector<ChannelHandle> channels(worldSize, 0);
    DISPATCH_CHK(HcclChannelAcquire(comm, COMM_ENGINE_CCU, descs.data(), worldSize, channels.data()),
                 "HcclChannelAcquire(CCU) failed");

    // ── 4. 前同步：kernel 启动前构造 remote_va_table（议题 5）──
    // 逐 channel 取 peer 的 HCCL Buffer 首地址 → 组成 rank→addr 表 → 拷到 device。
    // setup kernel 运行时按目的 rank 查此表，得到对端 HBM 写入基址。
    std::vector<uint64_t> remoteVa(worldSize, 0U);
    for (uint32_t i = 0; i < worldSize; ++i) {
        void* buf = nullptr;
        uint64_t sz = 0U;
        DISPATCH_CHK(HcclChannelGetHcclBuffer(comm, channels[i], &buf, &sz),
                     "HcclChannelGetHcclBuffer failed");
        remoteVa[i] = reinterpret_cast<uint64_t>(buf);
    }
    void* remoteVaTableDev = nullptr;
    const uint64_t vaTableBytes = static_cast<uint64_t>(worldSize) * sizeof(uint64_t);
    DISPATCH_CHK(aclrtMalloc(&remoteVaTableDev, vaTableBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc remote_va_table failed");
    DISPATCH_CHK(aclrtMemcpy(remoteVaTableDev, vaTableBytes, remoteVa.data(), vaTableBytes,
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy remote_va_table failed");

    // ── 5. 注册 CCU setup kernel ──
    // CcuKernelArg = 静态配置（channels / rank），注册期传入；creator 是返回 CcuKernel 子类实例的工厂 lambda。
    // 注意生命周期：creator / kernelArg 须存活到 Register→RegisterFinish→Launch 全部完成。本函数内三步顺序
    // 执行、均在 return 之前，故栈对象可行；若框架在 Launch 之后仍引用，须提升为堆 / 成员对象（V10 验证项）。
    auto kernelArg = std::make_shared<DispatchSetupKernelArg>(worldSize, param.myRank, channels, param);
    hcomm::KernelCreator creator =
        [](const hcomm::CcuKernelArg& a) -> std::unique_ptr<hcomm::CcuKernel> {
            return std::make_unique<DispatchSetupCcu>(a);
        };
    CcuKernelHandle kernelHandle = 0;
    DISPATCH_CHK(HcclCcuKernelRegister(comm, &kernelHandle,
                                       static_cast<void*>(&creator),
                                       static_cast<void*>(kernelArg.get())),
                 "HcclCcuKernelRegister failed (设计文档 V10：算子裸调待验证)");
    DISPATCH_CHK(HcclCcuKernelRegisterFinish(comm), "HcclCcuKernelRegisterFinish failed");

    // ── 6. 本卡 HCCL Buffer（teardown 在此读 peer 推来的数据 / 等收讫 flag）──
    // 用 Cleared 版本：READY_FLAGS 区天然为 0，省一次 host 端清零。
    void* localBuffer = nullptr;
    uint64_t localSize = 0U;
    DISPATCH_CHK(HcclGetHcclBufferCleared(comm, &localBuffer, &localSize),
                 "HcclGetHcclBufferCleared failed");

    // ── 7. 构造 teardown tiling 并拷到 device ──
    DispatchCcuTilingData tiling{};
    FillDispatchTiling(tiling, param);
    void* tilingDev = nullptr;
    DISPATCH_CHK(aclrtMalloc(&tilingDev, sizeof(DispatchCcuTilingData), ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc tiling failed");
    DISPATCH_CHK(aclrtMemcpy(tilingDev, sizeof(DispatchCcuTilingData), &tiling,
                             sizeof(DispatchCcuTilingData), ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy tiling failed");

    // ── 8. 双流编排：stream2(CCU) 等 stream1(AICore) 到同步点 ──
    // 架构师碎片 "stream2.wait(stream1)"：上游 gate/topk 在 stream1 算完，CCU setup 才能开始发数。
    aclrtEvent ev = nullptr;
    DISPATCH_CHK(aclrtCreateEvent(&ev), "aclrtCreateEvent failed");
    DISPATCH_CHK(aclrtRecordEvent(ev, stream1), "aclrtRecordEvent failed");
    DISPATCH_CHK(aclrtStreamWaitEvent(stream2, ev), "aclrtStreamWaitEvent failed");

    // ── 9. 启动 setup（CCU 引擎，经 ccuThread）──
    // CcuTaskArg = 动态地址（本次运行真实 GM 地址），启动期传入。
    auto taskArg = std::make_unique<DispatchSetupTaskArg>(
        reinterpret_cast<uint64_t>(x), reinterpret_cast<uint64_t>(expertIds),
        reinterpret_cast<uint64_t>(remoteVaTableDev), param.tokenNum, param.hidden);
    DISPATCH_CHK(HcclCcuKernelLaunch(comm, ccuThread, kernelHandle, static_cast<void*>(taskArg.get())),
                 "HcclCcuKernelLaunch(setup) failed");

    // ── 10. 启动 teardown（AICore，<<<>>>，stream1）──
    // <<<>>> 语法只能由 ascendc 编译器处理、不能出现在本 host 文件，故经 DispatchTeardownLaunch 包装
    // （实现见 moe_distribute_dispatch_ccu.cpp）。
    DispatchTeardownLaunch(param.blockDim, stream1,
                           reinterpret_cast<uint8_t*>(localBuffer),
                           reinterpret_cast<uint8_t*>(expandXOut),
                           reinterpret_cast<uint8_t*>(tilingDev));

    // 资源（ccuThread / channels / kernelHandle / remoteVaTableDev / tilingDev / ev）生命周期：
    // P0 暂裸调不回收；P1 用 RAII 收口（设计文档 §7 注、§14 P1 计划）。
    (void)localSize;
    return HCCL_SUCCESS;
}
